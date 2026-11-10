"""Проверка свойств вывода и границы JSON, не качества GPT."""

import copy
import json
from pathlib import Path
import unittest

from factlab.legacy.pipeline import Atom, C, Fact, RULES, Rule, SCHEMA, Var, infer, load_extraction, serialize


ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "legacy"


class InferenceTests(unittest.TestCase):
    def test_two_new_dependencies_and_proofs(self):
        initial = {
            Fact("requires", ("ai", "ml")),
            Fact("requires", ("ml", "prog")),
            Fact("requires", ("ai", "logic")),
            Fact("requires", ("logic", "math")),
        }
        known, proofs = infer(initial, RULES, SCHEMA)
        self.assertEqual(
            known - initial,
            {Fact("requires", ("ai", "prog")), Fact("requires", ("ai", "math"))},
        )
        for conclusion, proof in proofs.items():
            self.assertEqual(proof.round, 1)
            self.assertTrue(set(proof.parents) <= initial)
            env = dict(proof.substitution)
            rule = RULES[0]
            expected_head = Fact(rule.head.predicate, tuple(env[t.name] for t in rule.head.args))
            self.assertEqual(expected_head, conclusion)
            for pattern, parent in zip(rule.body, proof.parents):
                self.assertEqual(tuple(env[t.name] for t in pattern.args), parent.args)

    def test_multiround_and_disconnected_components(self):
        initial = {
            Fact("requires", ("a", "b")),
            Fact("requires", ("b", "c")),
            Fact("requires", ("c", "d")),
            Fact("requires", ("x", "y")),
        }
        known, proofs = infer(initial, RULES, SCHEMA)
        self.assertIn(Fact("requires", ("a", "d")), known)
        self.assertEqual(proofs[Fact("requires", ("a", "d"))].round, 2)
        self.assertNotIn(Fact("requires", ("a", "y")), known)
        repeated, repeated_proofs = infer(known, RULES, SCHEMA)
        self.assertEqual(repeated, known)
        self.assertEqual(repeated_proofs, {})

    def test_cycle_terminates(self):
        initial = {Fact("requires", ("a", "b")), Fact("requires", ("b", "a"))}
        known, _ = infer(initial, RULES, SCHEMA)
        self.assertEqual(len(known), 4)
        self.assertIn(Fact("requires", ("a", "a")), known)

    def test_unary_predicates_constants_and_consistent_variables(self):
        schema = {"student": ("person",), "person": ("person",), "same": ("person", "person")}
        x = Var("X")
        rules = (
            Rule("person_rule", (Atom("student", (x,)),), Atom("person", (x,))),
            Rule("same_rule", (Atom("same", (x, x)),), Atom("student", (x,))),
            Rule("constant_rule", (Atom("person", ("anna",)),), Atom("student", ("anna",))),
        )
        initial = {Fact("same", ("anna", "anna")), Fact("same", ("ivan", "petr"))}
        known, _ = infer(initial, rules, schema)
        self.assertIn(Fact("person", ("anna",)), known)
        self.assertNotIn(Fact("student", ("ivan",)), known)

    def test_unsafe_rule_rejected(self):
        rules = (Rule("bad", (Atom("requires", (C, "a")),), Atom("requires", (C, Var("NEW")))),)
        with self.assertRaisesRegex(ValueError, "Небезопасное"):
            infer(set(), rules, SCHEMA)

    def test_empty_input(self):
        self.assertEqual(infer(set(), RULES, SCHEMA), (set(), {}))


class JsonBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "extraction.json").read_text(encoding="utf-8"))
        self.original = (ROOT / "text.txt").read_text(encoding="utf-8")

    def test_fixture_roundtrip_and_duplicate_evidence(self):
        self.data["facts"].append(copy.deepcopy(self.data["facts"][0]))
        entities, sources, unparsed = load_extraction(self.data, self.original, SCHEMA)
        self.assertEqual(len(sources), 4)
        self.assertTrue(all(len(quotes) == 1 for quotes in sources.values()))
        known, proofs = infer(set(sources), RULES, SCHEMA)
        result = serialize(entities, sources, known, proofs, unparsed)
        self.assertEqual(len(result["new_fact_ids"]), 2)
        self.assertEqual(len(result["unparsed"]), 1)
        json.dumps(result, ensure_ascii=False)

    def test_invalid_inputs_rejected(self):
        variants = []
        for field, value in (
            ("predicate", "invented"),
            ("args", ["c1"]),
            ("args", ["c1", "missing"]),
            ("evidence", ["Выдуманная цитата"]),
            ("evidence", []),
        ):
            data = copy.deepcopy(self.data)
            data["facts"][0][field] = value
            variants.append(data)
        data = copy.deepcopy(self.data)
        data["facts"][0]["negative"] = True
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["entities"][0]["type"] = "person"
        variants.append(data)
        for i, data in enumerate(variants):
            with self.subTest(case=i), self.assertRaises(ValueError):
                load_extraction(data, self.original, SCHEMA)


if __name__ == "__main__":
    unittest.main()
