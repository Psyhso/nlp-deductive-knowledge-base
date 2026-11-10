"""Запуск полунаивного вывода и необязательных проверок BFS.

По умолчанию использует отдельный учебный пример, не изменяя исходный JSON.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from factlab.datalog.fact_models import ExtractionResult, ExtractionSchema, GroundLiteral, ValidationError
from factlab.datalog.fact_models import load_extraction, load_schema
from factlab.datalog.graph_tools import GraphCycleError, prerequisite_order, transitive_closure_bfs
from factlab.datalog.inference import InferenceResult, Rule, Var, literal_key, semi_naive
from factlab.datalog.rules_io import load_rules


ROOT = Path(__file__).resolve().parents[1] / "data" / "profiles" / "basic"


def is_pure_transitive_program(predicate: str, rules: tuple[Rule, ...]) -> bool:
    """BFS по исходным рёбрам проверяет только чистое транзитивное замыкание.

    Если другие правила тоже создают рёбра этого отношения, такой BFS уже не
    является полным независимым эталоном для результата общего движка.
    """
    definitions = [r for r in rules if r.head.predicate == predicate]
    if len(definitions) != 1:
        return False
    rule = definitions[0]
    if rule.head.negated or len(rule.head.args) != 2 or len(rule.body) != 2:
        return False
    if any(p.negated or p.predicate != predicate or len(p.args) != 2 for p in rule.body):
        return False
    if any(not isinstance(v, Var) for p in (*rule.body, rule.head) for v in p.args):
        return False
    start, end = rule.head.args
    if start == end:
        return False
    for first, second in (rule.body, tuple(reversed(rule.body))):
        middle = first.args[1]
        if (
            middle not in (start, end)
            and first.args == (start, middle)
            and second.args == (middle, end)
        ):
            return True
    return False


def _rule_data(rule: Rule) -> dict:
    def pattern_data(pattern) -> dict:
        return {
            "predicate": pattern.predicate,
            "args": [{"var": t.name} if isinstance(t, Var) else t for t in pattern.args],
            "negated": pattern.negated,
        }
    return {
        "id": rule.id,
        "body": [pattern_data(p) for p in rule.body],
        "head": pattern_data(rule.head),
    }


def build_report(
    extraction: ExtractionResult,
    result: InferenceResult,
    rules: tuple[Rule, ...],
    schema: ExtractionSchema,
    bfs_predicates: tuple[str, ...] = (),
    strict_orders: tuple[str, ...] = (),
    acyclic_predicates: tuple[str, ...] = (),
) -> dict:
    """Сериализовать вывод, конфликты и выбранные графовые проверки."""
    labels = {e.id: e.label for e in extraction.entities}
    ids = {literal: f"F{i}" for i, literal in enumerate(sorted(result.facts, key=literal_key), 1)}
    sources = {f.literal: f.evidence for f in extraction.facts}
    facts = []
    for literal in sorted(result.facts, key=literal_key):
        atom = literal.atom
        names = [labels[arg] for arg in atom.args]
        row = {
            "id": ids[literal],
            "predicate": atom.predicate,
            "args": list(atom.args),
            "negated": literal.negated,
            "display": f"{'¬' if literal.negated else ''}{atom.predicate}({', '.join(names)})",
        }
        if literal in result.initial:
            row.update(origin="extracted", evidence=list(sources[literal]))
        else:
            proof = result.proofs[literal]
            row.update(
                origin="derived",
                proof={
                    "rule": proof.rule_id,
                    "parents": [ids[parent] for parent in proof.parents],
                    "bindings": dict(proof.bindings),
                    "round": proof.round,
                },
            )
        facts.append(row)
    conflicts = [
        {
            "predicate": atom.predicate,
            "args": list(atom.args),
            "positive_fact": ids[GroundLiteral(atom)],
            "negative_fact": ids[GroundLiteral(atom, True)],
            "message": "Утверждение и его явное отрицание; требуется проверка пользователем.",
        }
        for atom in result.conflicts
    ]

    graph_checks = []
    strict_set = set(strict_orders)
    acyclic_set = strict_set | set(acyclic_predicates)
    predicates = {spec.name: spec for spec in schema.predicates}
    for predicate in sorted(set(bfs_predicates) | acyclic_set):
        spec = predicates.get(predicate)
        if spec is None or spec.arity != 2:
            raise ValidationError(f"BFS требует известный бинарный предикат: {predicate!r}.")
        if predicate in strict_set and spec.arg_types[0] != spec.arg_types[1]:
            raise ValidationError("Для строгого порядка оба аргумента должны иметь один тип.")
        graph = transitive_closure_bfs(result.initial, predicate)
        cycles = sorted(graph.cycle_vertices)
        item = {
            "predicate": predicate,
            "strict_order": predicate in strict_set,
            "acyclic_required": predicate in acyclic_set,
            "reachable_pairs": len(graph.facts),
            "cycle_vertices": cycles,
            "example_cycle": list(graph.path(cycles[0], cycles[0])) if cycles else None,
            "comparison": "skipped",
            "reason": "Сравнение требует единственного правила транзитивности для этого отношения.",
        }
        pure_transitivity = is_pure_transitive_program(predicate, rules)
        if pure_transitivity:
            actual = frozenset(
                literal for literal in result.facts
                if literal.atom.predicate == predicate and not literal.negated
            )
            item["comparison"] = "passed" if actual == graph.facts else "failed"
            del item["reason"]
            item["missing_in_engine"] = [
                list(f.atom.args) for f in sorted(graph.facts - actual, key=literal_key)
            ]
            item["extra_in_engine"] = [
                list(f.atom.args) for f in sorted(actual - graph.facts, key=literal_key)
            ]
            examples = sorted(
                (f for f in result.new_facts if f in graph.facts), key=literal_key
            )
            if examples:
                example = examples[0]
                path = graph.path(*example.atom.args)
                item["example_shortest_explanation"] = {
                    "fact_id": ids[example],
                    "path": list(path),
                    "labels": [labels[node] for node in path],
                }
        if predicate in acyclic_set:
            # Для чистого замыкания исходный и замкнутый графы имеют одинаковые
            # циклы. В общем наборе правил порядок проверяем по ПОЛНОМУ выводу.
            full_graph = graph if pure_transitivity else transitive_closure_bfs(result.facts, predicate)
            full_cycles = sorted(full_graph.cycle_vertices)
            item["strict_cycle_vertices"] = full_cycles
            item["acyclic_cycle_vertices"] = full_cycles
            item["strict_example_cycle"] = (
                list(full_graph.path(full_cycles[0], full_cycles[0])) if full_cycles else None
            )
            if predicate in strict_set:
                all_nodes = tuple(
                    e.id for e in extraction.entities if e.type == spec.arg_types[0]
                )
                try:
                    order_source = result.initial if pure_transitivity else result.facts
                    order = prerequisite_order(order_source, predicate, all_nodes)
                    item["prerequisite_order"] = list(order)
                    item["prerequisite_order_labels"] = [labels[node] for node in order]
                except GraphCycleError:
                    item["prerequisite_order"] = None
                    item["order_error"] = "Цикл в выбранном строгом порядке; нужен разбор пользователем."
        graph_checks.append(item)

    verification_failed = any(g["comparison"] == "failed" for g in graph_checks)
    needs_review = bool(conflicts) or any(g.get("strict_cycle_vertices") for g in graph_checks)
    status = "verification_failed" if verification_failed else "needs_review" if needs_review else "ok"
    return {
        "schema_version": "1.0",
        "status": status,
        "needs_user_review": needs_review,
        "entities": [asdict(e) for e in extraction.entities],
        "rules": [_rule_data(rule) for rule in rules],
        "facts": facts,
        "new_fact_ids": [ids[f] for f in sorted(result.new_facts, key=literal_key)],
        "statistics": {
            "initial_facts": len(result.initial),
            "new_facts": len(result.new_facts),
            "total_facts": len(result.facts),
            "productive_rounds": result.rounds,
            **asdict(result.stats),
        },
        "conflicts": conflicts,
        "graph_checks": graph_checks,
        "unparsed": [asdict(u) for u in extraction.unparsed],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "extraction.json")
    parser.add_argument("--text", type=Path, default=ROOT / "example.txt")
    parser.add_argument("--schema", type=Path, default=ROOT / "schema.json")
    parser.add_argument("--rules", type=Path, default=ROOT / "rules.json")
    parser.add_argument("--out", type=Path, default=Path("inference-result.json"))
    parser.add_argument("--bfs", action="append", default=[], metavar="PREDICATE")
    parser.add_argument("--strict-order", action="append", default=[], metavar="PREDICATE")
    parser.add_argument("--acyclic", action="append", default=[], metavar="PREDICATE")
    args = parser.parse_args()
    inputs = (args.input, args.text, args.schema, args.rules)
    if any(args.out.resolve() == path.resolve() for path in inputs):
        parser.error("--out должен отличаться от исходных файлов.")
    try:
        schema = load_schema(args.schema)
        extraction = load_extraction(
            args.input, args.text.read_text(encoding="utf-8-sig"), schema
        )
        rules = load_rules(args.rules)
        result = semi_naive(extraction, rules, schema)
        report = build_report(
            extraction, result, rules, schema, tuple(args.bfs), tuple(args.strict_order),
            tuple(args.acyclic),
        )
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (ValidationError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    print(
        f"Initial: {len(result.initial)}; new: {len(result.new_facts)}; "
        f"rounds: {result.rounds}; status: {report['status']}"
    )
    for row in report["facts"]:
        if row["origin"] == "derived":
            print(f"  {row['display']}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
