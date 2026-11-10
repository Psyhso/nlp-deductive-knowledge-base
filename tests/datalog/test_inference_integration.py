"""Сверка независимых методов и проверка границы JSON правил/отчёта."""

from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

from factlab.datalog.fact_models import (
    Entity, ExtractedFact, ExtractionResult, ExtractionSchema, GroundAtom,
    GroundLiteral, PredicateSpec, ValidationError, load_extraction, load_schema,
)
from factlab.datalog.graph_tools import transitive_closure_bfs
from factlab.datalog.inference import LiteralPattern, Rule, Var, semi_naive
from factlab.datalog.rules_io import load_rules, loads_rules
from factlab.datalog.run_inference import build_report, is_pure_transitive_program


ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "datalog"


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema(ROOT / "schema.json")
        self.rules = load_rules(ROOT / "rules.json")
        source = (ROOT / "text.txt").read_text(encoding="utf-8")
        self.extraction = load_extraction(ROOT / "extraction.json", source, self.schema)

    def test_demo_new_facts_bfs_provenance_and_order(self):
        result = semi_naive(self.extraction, self.rules, self.schema)
        self.assertEqual(
            {f.atom.args for f in result.new_facts},
            {("e1", "e3"), ("e1", "e4"), ("e2", "e4")},
        )
        self.assertEqual(result.rounds, 2)
        report = build_report(
            self.extraction, result, self.rules, self.schema, ("requires",), ("requires",)
        )
        self.assertEqual(report["status"], "ok")
        graph = report["graph_checks"][0]
        self.assertEqual(graph["comparison"], "passed")
        self.assertEqual(graph["example_shortest_explanation"]["path"], ["e1", "e2", "e3"])
        order = graph["prerequisite_order"]
        self.assertLess(order.index("e4"), order.index("e3"))
        self.assertLess(order.index("e3"), order.index("e2"))
        self.assertLess(order.index("e2"), order.index("e1"))
        byid = {f["id"]: f for f in report["facts"]}
        for fid in report["new_fact_ids"]:
            row = byid[fid]
            for parent_id in row["proof"]["parents"]:
                parent = byid[parent_id]
                parent_round = parent.get("proof", {}).get("round", 0)
                self.assertLess(parent_round, row["proof"]["round"])
        json.dumps(report, ensure_ascii=False, allow_nan=False)

    def test_inferred_conflict_requests_review_with_both_facts(self):
        extra = ExtractedFact(
            GroundLiteral(GroundAtom("requires", ("e4", "e6"))), ("test input",)
        )
        extraction = replace(self.extraction, facts=self.extraction.facts + (extra,))
        result = semi_naive(extraction, self.rules, self.schema)
        report = build_report(extraction, result, self.rules, self.schema)
        self.assertEqual(report["status"], "needs_review")
        self.assertTrue(report["needs_user_review"])
        self.assertIn(GroundAtom("requires", ("e1", "e6")), result.conflicts)
        byid = {f["id"]: f for f in report["facts"]}
        conflict = report["conflicts"][0]
        self.assertFalse(byid[conflict["positive_fact"]]["negated"])
        self.assertTrue(byid[conflict["negative_fact"]]["negated"])

    def test_cycles_warn_only_when_order_is_declared_strict(self):
        extra = ExtractedFact(
            GroundLiteral(GroundAtom("requires", ("e2", "e1"))), ("test input",)
        )
        extraction = replace(self.extraction, facts=self.extraction.facts + (extra,))
        result = semi_naive(extraction, self.rules, self.schema)
        normal = build_report(extraction, result, self.rules, self.schema, ("requires",))
        self.assertEqual(normal["status"], "ok")
        strict = build_report(
            extraction, result, self.rules, self.schema, strict_orders=("requires",)
        )
        self.assertEqual(strict["status"], "needs_review")
        check = strict["graph_checks"][0]
        self.assertEqual(check["strict_cycle_vertices"], ["e1", "e2"])
        self.assertEqual(check["strict_example_cycle"], ["e1", "e2", "e1"])
        self.assertIsNone(check["prerequisite_order"])

    def test_bfs_comparison_is_skipped_for_other_edge_producing_rules(self):
        x, y = Var("X"), Var("Y")
        identity = Rule(
            "identity", (LiteralPattern("requires", (x, y)),),
            LiteralPattern("requires", (x, y)),
        )
        rules = self.rules + (identity,)
        result = semi_naive(self.extraction, rules, self.schema)
        report = build_report(self.extraction, result, rules, self.schema, ("requires",))
        self.assertFalse(is_pure_transitive_program("requires", rules))
        self.assertEqual(report["graph_checks"][0]["comparison"], "skipped")

    def test_reversed_conjunction_is_still_pure_transitivity(self):
        rule = self.rules[0]
        self.assertTrue(is_pure_transitive_program(
            "requires", (replace(rule, body=tuple(reversed(rule.body))),)
        ))

    def test_independent_bfs_matches_engine_on_seeded_graphs(self):
        rng = random.Random(1247)
        for _ in range(40):
            n = rng.randrange(1, 8)
            entities = tuple(Entity(f"n{i}", f"N{i}", "course", (f"N{i}",)) for i in range(n))
            inputs = tuple(
                ExtractedFact(GroundLiteral(GroundAtom("requires", (a.id, b.id))), ("edge",))
                for a in entities for b in entities if rng.random() < .24
            )
            extraction = ExtractionResult(entities, inputs, ())
            result = semi_naive(extraction, self.rules, self.schema)
            bfs = transitive_closure_bfs(result.initial, "requires")
            self.assertEqual(result.facts, bfs.facts)

    def test_parent_to_ancestor_agrees_with_bfs(self):
        schema = ExtractionSchema(
            ("person",),
            (
                PredicateSpec("parent", ("person", "person"), "Первый — родитель второго."),
                PredicateSpec("ancestor", ("person", "person"), "Первый — предок второго."),
            ),
            False,
        )
        entities = tuple(Entity(e, e, "person", (e,)) for e in ("a", "b", "c"))
        inputs = tuple(
            ExtractedFact(GroundLiteral(GroundAtom("parent", pair)), ("edge",))
            for pair in (("a", "b"), ("b", "c"))
        )
        extraction = ExtractionResult(entities, inputs, ())
        x, y, z = Var("X"), Var("Y"), Var("Z")
        rules = (
            Rule("seed", (LiteralPattern("parent", (x, y)),), LiteralPattern("ancestor", (x, y))),
            Rule(
                "step", (LiteralPattern("parent", (x, y)), LiteralPattern("ancestor", (y, z))),
                LiteralPattern("ancestor", (x, z)),
            ),
        )
        result = semi_naive(extraction, rules, schema)
        bfs = transitive_closure_bfs(result.initial, "parent", "ancestor")
        self.assertEqual(
            frozenset(f for f in result.facts if f.atom.predicate == "ancestor"), bfs.facts
        )
        self.assertEqual(bfs.path("a", "c"), ("a", "b", "c"))
        self.assertNotIn(GroundLiteral(GroundAtom("parent", ("a", "c"))), result.facts)

    def test_json_rule_term_and_boolean_are_structural(self):
        data = json.loads((ROOT / "rules.json").read_text(encoding="utf-8"))
        data["rules"][0]["head"]["negated"] = "false"
        with self.assertRaises(ValidationError):
            loads_rules(json.dumps(data))
        data["rules"][0]["head"]["negated"] = False
        data["rules"][0]["head"]["args"][0] = {"function": "new_object"}
        with self.assertRaises(ValidationError):
            loads_rules(json.dumps(data))

    def test_json_rules_reject_unknown_fields_duplicate_keys_and_constants(self):
        for payload in (
            '{"schema_version":"1.0","schema_version":"1.0","rules":[]}',
            '{"schema_version":"1.0","rules":[],"code":"print(1)"}',
            '{"schema_version":"1.0","rules":NaN}',
            '{"schema_version":"1.0","rules":[]',
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                loads_rules(payload)

    def test_cli_refuses_to_overwrite_input(self):
        input_path = ROOT / "extraction.json"
        before = input_path.read_bytes()
        process = subprocess.run(
            [sys.executable, "-B", "-m", "factlab.datalog.run_inference",
             "--input", str(input_path), "--out", str(input_path)],
            capture_output=True,
        )
        self.assertEqual(process.returncode, 2)
        self.assertEqual(input_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
