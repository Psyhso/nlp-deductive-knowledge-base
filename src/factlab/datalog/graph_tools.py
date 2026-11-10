"""Вспомогательные обходы бинарных отношений; Python 3.10+, stdlib.

Граф строится только по положительным исходным фактам. Выбор транзитивного
отношения — ответственность вызывающего кода: сам обход этого не доказывает.
Отрицательные факты не означают отсутствие пути и не участвуют в обходе.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
from types import MappingProxyType
from typing import Iterable, Mapping

from factlab.datalog.fact_models import GroundAtom, GroundLiteral, ValidationError


class GraphCycleError(ValidationError):
    """Порядок предпосылок невозможен из-за цикла в выбранном отношении."""


@dataclass(frozen=True, slots=True)
class BFSResult:
    source_predicate: str
    target_predicate: str
    facts: frozenset[GroundLiteral]
    new_facts: frozenset[GroundLiteral]
    cycle_vertices: frozenset[str]
    # Один предок для каждой достижимой вершины, а не полный путь каждой пары.
    _predecessors: Mapping[str, Mapping[str, str | None]] = field(repr=False)
    _cycle_parents: Mapping[str, str] = field(repr=False)

    def path(self, start: str, end: str) -> tuple[str, ...] | None:
        """Кратчайший непустой путь по исходным рёбрам, либо None.

        Для start == end возвращается цикл с повторением start на конце.
        Путь нулевой длины не считается фактом отношения.
        Время восстановления — O(длина пути).
        """
        predecessors = self._predecessors.get(start)
        if predecessors is None:
            return None
        if start == end:
            current = self._cycle_parents.get(start)
            if current is None:
                return None
            reverse_path = [current]
            while current != start:
                current = predecessors[current]
                assert current is not None
                reverse_path.append(current)
            return tuple(reversed(reverse_path)) + (start,)
        if end not in predecessors:
            return None
        reverse_path = [end]
        current = end
        while current != start:
            current = predecessors[current]
            assert current is not None
            reverse_path.append(current)
        return tuple(reversed(reverse_path))


def _check_predicate(predicate: str) -> None:
    # Единые с моделями требования к идентификаторам.
    GroundAtom(predicate, ("node", "node"))


def _graph(
    initial: Iterable[GroundLiteral], predicate: str
) -> tuple[frozenset[GroundLiteral], dict[str, tuple[str, ...]]]:
    _check_predicate(predicate)
    original: set[GroundLiteral] = set()
    adjacency: dict[str, set[str]] = {}
    for literal in initial:
        if not isinstance(literal, GroundLiteral):
            raise ValidationError("initial: ожидаются объекты GroundLiteral.")
        original.add(literal)
        if literal.atom.predicate != predicate:
            continue
        if len(literal.atom.args) != 2:
            raise ValidationError(
                f"Для обхода {predicate!r} нужны бинарные факты; "
                f"получено {len(literal.atom.args)} аргументов."
            )
        if literal.negated:
            continue
        source, target = literal.atom.args
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())
    # Сортируем списки смежности один раз. Это задаёт стабильный выбор путей.
    return frozenset(original), {
        node: tuple(sorted(neighbors)) for node, neighbors in adjacency.items()
    }


def transitive_closure_bfs(
    initial: Iterable[GroundLiteral],
    source_predicate: str,
    target_predicate: str | None = None,
) -> BFSResult:
    """Найти все пары, связанные непустым путём исходного отношения.

    Если target_predicate задан отдельно, например parent -> ancestor,
    исходные рёбра parent тоже дают ancestor-пары, но существующие ancestor
    не используются как рёбра. new_facts — разность результата и initial.

    Циклы допустимы для общего отношения. cycle_vertices содержит ровно
    вершины, принадлежащие циклам, включая петли. Для строгих порядков
    вызывающая сторона может отдельно считать их ошибкой.

    После однократной сортировки графа обходы занимают O(n(n + m));
    хранение результата и предков — O(n² + m). Полные пути не сохраняются.
    """
    if target_predicate is None:
        target_predicate = source_predicate
    _check_predicate(target_predicate)
    original, adjacency = _graph(initial, source_predicate)
    all_predecessors: dict[str, Mapping[str, str | None]] = {}
    cycle_parents: dict[str, str] = {}
    facts: set[GroundLiteral] = set()

    for start in sorted(adjacency):
        predecessors: dict[str, str | None] = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor == start and start not in cycle_parents:
                    # BFS первым находит кратчайший непустой возврат.
                    cycle_parents[start] = current
                if neighbor not in predecessors:
                    predecessors[neighbor] = current
                    queue.append(neighbor)
        for end in predecessors:
            if end != start or start in cycle_parents:
                facts.add(GroundLiteral(GroundAtom(target_predicate, (start, end))))
        all_predecessors[start] = MappingProxyType(predecessors)

    closure = frozenset(facts)
    return BFSResult(
        source_predicate=source_predicate,
        target_predicate=target_predicate,
        facts=closure,
        new_facts=closure - original,
        cycle_vertices=frozenset(cycle_parents),
        _predecessors=MappingProxyType(all_predecessors),
        _cycle_parents=MappingProxyType(cycle_parents),
    )


def prerequisite_order(
    initial: Iterable[GroundLiteral],
    predicate: str = "requires",
    nodes: Iterable[str] = (),
) -> tuple[str, ...]:
    """План с предпосылками раньше зависимых объектов.

    requires(C, P) означает «C требует P», поэтому топологическая сортировка
    выполняется по обратным рёбрам P -> C. Дополнительные nodes позволяют
    включить изолированные объекты. При цикле возникает GraphCycleError.
    Сложность O(n + m + n log n) с детерминированной очередью вершин.
    """
    _, adjacency = _graph(initial, predicate)
    dependents: dict[str, set[str]] = {node: set() for node in adjacency}
    for node in nodes:
        GroundAtom(predicate, (node, node))
        dependents.setdefault(node, set())
    indegree = {node: 0 for node in dependents}
    for dependent, prerequisites in adjacency.items():
        for prerequisite in prerequisites:
            dependents[prerequisite].add(dependent)
            indegree[dependent] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        ordered.append(node)
        for dependent in dependents[node]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(ordered) != len(indegree):
        raise GraphCycleError(
            f"В отношении {predicate!r} обнаружен цикл: "
            "невозможно поставить каждую предпосылку раньше зависимого объекта."
        )
    return tuple(ordered)
