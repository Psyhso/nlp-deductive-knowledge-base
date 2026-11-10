"""JSON -> проверенные факты -> прямой вывод -> JSON с объяснениями.

Python 3.10+, только стандартная библиотека. Вызовов GPT здесь нет.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterator


@dataclass(frozen=True, order=True)
class Fact:
    predicate: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Atom:
    predicate: str
    args: tuple[str | Var, ...]


@dataclass(frozen=True)
class Rule:
    id: str
    body: tuple[Atom, ...]
    head: Atom


@dataclass(frozen=True)
class Proof:
    rule_id: str
    parents: tuple[Fact, ...]
    substitution: tuple[tuple[str, str], ...]
    round: int


# Имена типов аргументов. При смене предметной области меняется схема.
SCHEMA = {"requires": ("course", "course")}
C, Q, P = Var("C"), Var("Q"), Var("P")
RULES = (
    Rule(
        id="R1",
        body=(Atom("requires", (C, Q)), Atom("requires", (Q, P))),
        head=Atom("requires", (C, P)),
    ),
)


def exact_keys(value: object, keys: set[str], where: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{where}: ожидаются ровно поля {sorted(keys)}")


def text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: ожидается непустая строка")
    return value


def load_extraction(
    data: object, original: str, schema: dict[str, tuple[str, ...]]
) -> tuple[dict[str, dict], dict[Fact, list[str]], list[dict]]:
    """Проверяет структуру и цитаты; истинность утверждений не проверяет."""
    exact_keys(data, {"entities", "facts", "unparsed"}, "JSON")
    for field in ("entities", "facts", "unparsed"):
        if not isinstance(data[field], list):
            raise ValueError(f"{field}: ожидается массив")

    entities: dict[str, dict] = {}
    allowed_types = {t for args in schema.values() for t in args}
    for e in data["entities"]:
        exact_keys(e, {"id", "label", "type"}, "entity")
        eid = text(e["id"], "entity.id")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", eid) or eid in entities:
            raise ValueError(f"Недопустимый или повторный entity.id: {eid}")
        text(e["label"], "entity.label")
        if not isinstance(e["type"], str) or e["type"] not in allowed_types:
            raise ValueError(f"Неизвестный тип сущности: {e['type']}")
        entities[eid] = dict(e)

    sources: dict[Fact, list[str]] = {}
    for f in data["facts"]:
        exact_keys(f, {"predicate", "args", "evidence"}, "fact")
        predicate = text(f["predicate"], "fact.predicate")
        if predicate not in schema:
            raise ValueError(f"Неизвестный предикат: {predicate}")
        args = f["args"]
        if not isinstance(args, list) or len(args) != len(schema[predicate]):
            raise ValueError(f"Неверная арность предиката {predicate}")
        for arg, required_type in zip(args, schema[predicate]):
            if not isinstance(arg, str) or arg not in entities:
                raise ValueError(f"Неизвестная сущность: {arg}")
            if entities[arg]["type"] != required_type:
                raise ValueError(f"Неверный тип аргумента {predicate}: {arg}")
        evidence = f["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("fact.evidence: нужен непустой массив цитат")
        fact = Fact(predicate, tuple(args))
        quotes = sources.setdefault(fact, [])
        for quote in evidence:
            text(quote, "evidence")
            if quote not in original:
                raise ValueError(f"Цитаты нет в исходном тексте: {quote}")
            if quote not in quotes:
                quotes.append(quote)

    for item in data["unparsed"]:
        exact_keys(item, {"text", "reason"}, "unparsed")
        if text(item["text"], "unparsed.text") not in original:
            raise ValueError("Фрагмента unparsed нет в исходном тексте")
        text(item["reason"], "unparsed.reason")
    return entities, sources, data["unparsed"]


def matches(
    body: tuple[Atom, ...],
    facts: tuple[Fact, ...],
    env: dict[str, str] | None = None,
    parents: tuple[Fact, ...] = (),
) -> Iterator[tuple[dict[str, str], tuple[Fact, ...]]]:
    """Согласованное сопоставление ВСЕХ атомов тела с известными фактами."""
    env = {} if env is None else env
    if not body:
        yield env, parents
        return
    pattern, *tail = body
    for fact in facts:
        if pattern.predicate != fact.predicate or len(pattern.args) != len(fact.args):
            continue
        candidate = dict(env)
        for term, value in zip(pattern.args, fact.args):
            if isinstance(term, Var):
                if term.name in candidate and candidate[term.name] != value:
                    break
                candidate[term.name] = value
            elif term != value:
                break
        else:
            yield from matches(tuple(tail), facts, candidate, parents + (fact,))


def validate_rules(
    rules: tuple[Rule, ...], schema: dict[str, tuple[str, ...]]
) -> None:
    seen: set[str] = set()
    for rule in rules:
        if not rule.id or rule.id in seen or not rule.body:
            raise ValueError("Нужны уникальные id правил и непустые тела")
        seen.add(rule.id)
        bound = {t.name for a in rule.body for t in a.args if isinstance(t, Var)}
        var_types: dict[str, str] = {}
        for atom in (*rule.body, rule.head):
            if atom.predicate not in schema or len(atom.args) != len(schema[atom.predicate]):
                raise ValueError(f"Неверный предикат/арность в правиле {rule.id}")
            for term, kind in zip(atom.args, schema[atom.predicate]):
                if isinstance(term, Var):
                    if term.name in var_types and var_types[term.name] != kind:
                        raise ValueError(f"Несовместимые типы переменной {term.name}")
                    var_types[term.name] = kind
                elif not isinstance(term, str) or not term:
                    raise ValueError("Допустимы только Var и строковые константы")
        if any(isinstance(t, Var) and t.name not in bound for t in rule.head.args):
            raise ValueError(f"Небезопасное правило: {rule.id}")


def infer(
    initial: set[Fact],
    rules: tuple[Rule, ...],
    schema: dict[str, tuple[str, ...]],
) -> tuple[set[Fact], dict[Fact, Proof]]:
    """Наивный прямой вывод с синхронными раундами и первым доказательством."""
    validate_rules(rules, schema)
    for fact in initial:
        if fact.predicate not in schema or len(fact.args) != len(schema[fact.predicate]):
            raise ValueError("Неверный начальный факт")
        if any(not isinstance(a, str) or not a for a in fact.args):
            raise ValueError("В начальных фактах допустимы только константы")
    known = set(initial)
    proofs: dict[Fact, Proof] = {}
    round_number = 0
    while True:
        round_number += 1
        snapshot = tuple(sorted(known))
        pending: dict[Fact, Proof] = {}
        for rule in rules:
            for env, parents in matches(rule.body, snapshot):
                args = tuple(env[t.name] if isinstance(t, Var) else t for t in rule.head.args)
                conclusion = Fact(rule.head.predicate, args)
                if conclusion not in known and conclusion not in pending:
                    pending[conclusion] = Proof(
                        rule.id, parents, tuple(sorted(env.items())), round_number
                    )
        if not pending:
            return known, proofs
        known.update(pending)
        proofs.update(pending)


def serialize(
    entities: dict[str, dict],
    sources: dict[Fact, list[str]],
    known: set[Fact],
    proofs: dict[Fact, Proof],
    unparsed: list[dict],
) -> dict:
    ids = {f: f"F{i}" for i, f in enumerate(sorted(known), start=1)}
    rows = []
    for fact in sorted(known):
        labels = [entities.get(arg, {}).get("label", arg) for arg in fact.args]
        row = {
            "id": ids[fact],
            "predicate": fact.predicate,
            "args": list(fact.args),
            "display": f"{fact.predicate}({', '.join(labels)})",
        }
        if fact in sources:
            row.update(origin="extracted", evidence=sources[fact])
        else:
            proof = proofs[fact]
            row.update(
                origin="derived",
                rule=proof.rule_id,
                parents=[ids[p] for p in proof.parents],
                substitution=dict(proof.substitution),
                round=proof.round,
            )
        rows.append(row)
    return {
        "entities": list(entities.values()),
        "facts": rows,
        "new_fact_ids": [ids[f] for f in sorted(proofs)],
        # Это проверка предметной области, а не ошибка логического алгоритма.
        "cycle_entities": [
            f.args[0]
            for f in sorted(known)
            if f.predicate == "requires" and len(f.args) == 2 and f.args[0] == f.args[1]
        ],
        "unparsed": unparsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="Ответ GPT без Markdown")
    parser.add_argument("--text", type=Path, required=True, help="Исходный текст")
    parser.add_argument("--out", type=Path, required=True, help="Файл результата")
    args = parser.parse_args()
    data = json.loads(args.input_json.read_text(encoding="utf-8-sig"))
    original = args.text.read_text(encoding="utf-8-sig")
    entities, sources, unparsed = load_extraction(data, original, SCHEMA)
    known, proofs = infer(set(sources), RULES, SCHEMA)
    result = serialize(entities, sources, known, proofs, unparsed)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Input facts: {len(sources)}; new facts: {len(proofs)}; total: {len(known)}")


if __name__ == "__main__":
    main()
