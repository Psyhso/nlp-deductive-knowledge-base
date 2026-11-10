"""Полунаивный вывод безопасных правил над конечными подписанными фактами.

Используется только явное отрицание: отсутствие P(a) не означает ¬P(a).
Ни переменных в результатах, ни создания новых сущностей, ни NAF здесь нет.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Sequence

from factlab.datalog.fact_models import (
    ExtractionResult, ExtractionSchema, GroundAtom, GroundLiteral, ValidationError,
)


_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*", flags=re.ASCII)


@dataclass(frozen=True, slots=True)
class Var:
    """Структурная переменная правила; строка в аргументе всегда константа."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _VARIABLE.fullmatch(self.name):
            raise ValidationError("Var.name: ожидается имя вида [A-Za-z_][A-Za-z0-9_]*.")


@dataclass(frozen=True, slots=True)
class LiteralPattern:
    predicate: str
    args: tuple[Var | str, ...]
    negated: bool = False

    def __post_init__(self) -> None:
        # GroundAtom задаёт тот же синтаксис имён, что и загрузчик фактов.
        GroundAtom(self.predicate, ("placeholder",))
        if not isinstance(self.args, tuple) or not self.args:
            raise ValidationError("LiteralPattern.args: требуется непустой tuple.")
        if type(self.negated) is not bool:
            raise ValidationError("LiteralPattern.negated: требуется bool.")
        for argument in self.args:
            if isinstance(argument, Var):
                continue
            if not isinstance(argument, str):
                raise ValidationError("Аргумент правила должен быть Var или id сущности.")
            GroundAtom(self.predicate, (argument,))


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    body: tuple[LiteralPattern, ...]
    head: LiteralPattern

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValidationError("Rule.id: требуется непустая строка.")
        if not isinstance(self.body, tuple) or not self.body:
            raise ValidationError("Rule.body: требуется непустой tuple условий.")
        if any(not isinstance(item, LiteralPattern) for item in self.body):
            raise ValidationError("Rule.body: ожидаются LiteralPattern.")
        if not isinstance(self.head, LiteralPattern):
            raise ValidationError("Rule.head: ожидается LiteralPattern.")


@dataclass(frozen=True, slots=True)
class Derivation:
    rule_id: str
    parents: tuple[GroundLiteral, ...]
    bindings: tuple[tuple[str, str], ...]
    round: int


@dataclass(frozen=True, slots=True)
class InferenceStats:
    """Счётчики работы, включая заключительный непродуктивный раунд.

    candidate_checks — кандидаты, переданные сопоставлению одного условия;
    completed_matches — полные согласованные сопоставления тела правила;
    delta_variants — запущенные варианты с непустым отношением в якоре Δ.
    Это не время работы: построение индексов в счётчики не входит.
    """

    candidate_checks: int = 0
    completed_matches: int = 0
    delta_variants: int = 0


@dataclass(frozen=True, slots=True)
class InferenceResult:
    initial: frozenset[GroundLiteral]
    facts: frozenset[GroundLiteral]
    proofs: dict[GroundLiteral, Derivation]
    rounds: int
    stats: InferenceStats
    conflicts: tuple[GroundAtom, ...]

    @property
    def new_facts(self) -> frozenset[GroundLiteral]:
        return self.facts - self.initial


def literal_key(literal: GroundLiteral) -> tuple[str, tuple[str, ...], bool]:
    """Стабильный порядок литералов для вывода и выбора первого доказательства."""
    return literal.atom.predicate, literal.atom.args, literal.negated


def validate_rules(
    extraction: ExtractionResult,
    rules: Sequence[Rule],
    schema: ExtractionSchema,
) -> None:
    """Проверить типы, арность, константы и безопасность всего набора правил.

    ExtractionResult может быть создан вручную, поэтому совместимость входных
    фактов со схемой проверяется повторно. Цитаты здесь не интерпретируются.
    """
    if not isinstance(extraction, ExtractionResult):
        raise ValidationError("extraction: ожидается ExtractionResult.")
    if not isinstance(schema, ExtractionSchema):
        raise ValidationError("schema: ожидается ExtractionSchema.")
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise ValidationError("rules: ожидается последовательность Rule.")
    entities = {entity.id: entity for entity in extraction.entities}
    predicates = {spec.name: spec for spec in schema.predicates}
    for entity in entities.values():
        if entity.type not in schema.entity_types:
            raise ValidationError(f"Сущность {entity.id!r}: неизвестный тип {entity.type!r}.")

    def check_pattern(pattern: LiteralPattern, variable_types: dict[str, str], path: str) -> None:
        spec = predicates.get(pattern.predicate)
        if spec is None:
            raise ValidationError(f"{path}: неизвестный предикат {pattern.predicate!r}.")
        if len(pattern.args) != spec.arity:
            raise ValidationError(f"{path}: ожидается {spec.arity} аргументов.")
        if pattern.negated and not schema.allow_explicit_negation:
            raise ValidationError(f"{path}: явное отрицание отключено в схеме.")
        for argument, expected_type in zip(pattern.args, spec.arg_types):
            if isinstance(argument, Var):
                previous = variable_types.setdefault(argument.name, expected_type)
                if previous != expected_type:
                    raise ValidationError(
                        f"{path}: переменная {argument.name!r} имеет несовместимые типы "
                        f"{previous!r} и {expected_type!r}."
                    )
            else:
                entity = entities.get(argument)
                if entity is None:
                    raise ValidationError(f"{path}: неизвестная константа {argument!r}.")
                if entity.type != expected_type:
                    raise ValidationError(
                        f"{path}: константа {argument!r} должна иметь тип {expected_type!r}."
                    )

    for fact in extraction.facts:
        literal = fact.literal
        check_pattern(
            LiteralPattern(literal.atom.predicate, literal.atom.args, literal.negated),
            {}, "Входной факт",
        )
    rule_ids: set[str] = set()
    for rule in rules:
        if not isinstance(rule, Rule):
            raise ValidationError("rules: каждый элемент должен быть Rule.")
        if rule.id in rule_ids:
            raise ValidationError(f"Повторяющийся id правила {rule.id!r}.")
        rule_ids.add(rule.id)
        variable_types: dict[str, str] = {}
        for index, pattern in enumerate(rule.body):
            check_pattern(pattern, variable_types, f"Правило {rule.id}, body[{index}]")
        body_variables = set(variable_types)
        head_variables = {arg.name for arg in rule.head.args if isinstance(arg, Var)}
        if not head_variables <= body_variables:
            raise ValidationError(
                f"Правило {rule.id}: небезопасные переменные заключения "
                f"{', '.join(sorted(head_variables - body_variables))}."
            )
        check_pattern(rule.head, variable_types, f"Правило {rule.id}, head")


class _Index:
    """Индексы по имени/знаку и по каждому связанному аргументу."""

    def __init__(self, facts: frozenset[GroundLiteral]) -> None:
        self.relations: dict[tuple[str, bool], list[GroundLiteral]] = defaultdict(list)
        self.arguments: dict[tuple[str, bool, int, str], list[GroundLiteral]] = defaultdict(list)
        for literal in sorted(facts, key=literal_key):
            atom = literal.atom
            self.relations[atom.predicate, literal.negated].append(literal)
            for position, argument in enumerate(atom.args):
                self.arguments[atom.predicate, literal.negated, position, argument].append(literal)

    def has_relation(self, pattern: LiteralPattern) -> bool:
        return bool(self.relations.get((pattern.predicate, pattern.negated)))

    def candidates(
        self, pattern: LiteralPattern, bindings: dict[str, str],
    ) -> Sequence[GroundLiteral]:
        candidates = self.relations.get((pattern.predicate, pattern.negated), ())
        for position, argument in enumerate(pattern.args):
            value = bindings.get(argument.name) if isinstance(argument, Var) else argument
            if value is None:
                continue
            posting = self.arguments.get((pattern.predicate, pattern.negated, position, value), ())
            if len(posting) < len(candidates):
                candidates = posting
            if not candidates:
                break
        return candidates


@dataclass(slots=True)
class _Counters:
    candidate_checks: int = 0
    completed_matches: int = 0
    delta_variants: int = 0


def _match(pattern: LiteralPattern, literal: GroundLiteral, bindings: dict[str, str]) -> dict[str, str] | None:
    result = bindings.copy()
    for argument, value in zip(pattern.args, literal.atom.args):
        if isinstance(argument, Var):
            previous = result.setdefault(argument.name, value)
            if previous != value:
                return None
        elif argument != value:
            return None
    return result


def _joins(
    body: tuple[LiteralPattern, ...], sources: tuple[_Index, ...], counters: _Counters,
) -> Iterator[tuple[dict[str, str], tuple[GroundLiteral, ...]]]:
    # Явный стек: длина тела не расходует стек рекурсивных вызовов Python.
    stack = [(0, {}, (), iter(sources[0].candidates(body[0], {})))]
    while stack:
        position, bindings, parents, candidates = stack[-1]
        try:
            literal = next(candidates)
        except StopIteration:
            stack.pop()
            continue
        counters.candidate_checks += 1
        updated = _match(body[position], literal, bindings)
        if updated is None:
            continue
        updated_parents = parents + (literal,)
        next_position = position + 1
        if next_position == len(body):
            counters.completed_matches += 1
            yield updated, updated_parents
        else:
            stack.append((
                next_position, updated, updated_parents,
                iter(sources[next_position].candidates(body[next_position], updated)),
            ))


def _instantiate(pattern: LiteralPattern, bindings: dict[str, str]) -> GroundLiteral:
    return GroundLiteral(
        GroundAtom(pattern.predicate, tuple(
            bindings[arg.name] if isinstance(arg, Var) else arg for arg in pattern.args
        )),
        pattern.negated,
    )


def semi_naive(
    extraction: ExtractionResult,
    rules: Sequence[Rule],
    schema: ExtractionSchema,
) -> InferenceResult:
    """Вычислить замыкание, сохранив по одному доказательству новых литералов.

    Для якоря i берём old=K−Δ на позициях <i, Δ на i и K на >i.
    Поэтому каждое сопоставление, содержащее новые факты, рассматривается
    ровно в варианте с его первым новым фактом. Раунд читает только снимок K.
    rounds считает продуктивные слои; исходным фактам соответствует слой 0.

    Отрицательные атомы учитываются инкрементально. Поиск конфликтов проходит
    только по ним: O(k) при среднем O(1) для hash lookup. Стабильная сортировка
    отчёта из c конфликтов отдельно требует O(c log c).
    """
    validate_rules(extraction, rules, schema)
    initial = frozenset(fact.literal for fact in extraction.facts)
    known = initial
    delta = initial
    negative_atoms = {literal.atom for literal in initial if literal.negated}
    proofs: dict[GroundLiteral, Derivation] = {}
    rounds = 0
    counters = _Counters()
    while delta:
        old_index = _Index(known - delta)
        delta_index = _Index(delta)
        known_index = _Index(known)
        pending: dict[GroundLiteral, Derivation] = {}
        for rule in rules:
            for anchor, pattern in enumerate(rule.body):
                if not delta_index.has_relation(pattern):
                    continue
                counters.delta_variants += 1
                sources = tuple(
                    old_index if index < anchor else delta_index if index == anchor else known_index
                    for index in range(len(rule.body))
                )
                for bindings, parents in _joins(rule.body, sources, counters):
                    conclusion = _instantiate(rule.head, bindings)
                    if conclusion not in known and conclusion not in pending:
                        pending[conclusion] = Derivation(
                            rule.id, parents, tuple(sorted(bindings.items())), rounds + 1,
                        )
        if not pending:
            break
        delta = frozenset(pending)
        negative_atoms.update(literal.atom for literal in delta if literal.negated)
        known = known | delta
        proofs.update(pending)
        rounds += 1
    conflicts = tuple(sorted(
        (atom for atom in negative_atoms if GroundLiteral(atom) in known),
        key=lambda atom: (atom.predicate, atom.args),
    ))
    return InferenceResult(
        initial, known, proofs, rounds,
        InferenceStats(counters.candidate_checks, counters.completed_matches, counters.delta_variants),
        conflicts,
    )
