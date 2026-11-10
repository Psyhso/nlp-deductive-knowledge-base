"""Offline-проверка двух схем: вывод, границы правил и объяснения без LLM."""

from dataclasses import replace
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2] / "src" / "factlab" / "data" / "profiles"

from factlab.datalog.fact_models import ExtractedFact, GroundAtom, GroundLiteral, load_extraction, load_schema
from factlab.datalog.graph_tools import transitive_closure_bfs
from factlab.datalog.inference import semi_naive
from factlab.datalog.rules_io import load_rules
from factlab.datalog.run_inference import build_report


def literal(predicate, *args, negated=False):
    return GroundLiteral(GroundAtom(predicate, tuple(args)), negated)


def profile(name):
    directory = ROOT / name
    schema = load_schema(directory / "schema.json")
    rules = load_rules(directory / "rules.json")
    text = (directory / "example.txt").read_text(encoding="utf-8")
    extraction = load_extraction(directory / "extraction.json", text, schema)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return schema, rules, extraction, metadata


class ProfileTests(unittest.TestCase):
    def test_both_fixtures_load_match_counts_and_have_no_conflicts(self):
        for name in ("university", "family"):
            with self.subTest(profile=name):
                schema, rules, extraction, metadata = profile(name)
                result = semi_naive(extraction, rules, schema)
                self.assertEqual(len(result.initial), metadata["expected"]["initial_facts"])
                self.assertEqual(len(result.new_facts), metadata["expected"]["new_facts"])
                self.assertFalse(result.conflicts)
                self.assertEqual(set(metadata["rule_descriptions"]), {r.id for r in rules})
                self.assertEqual({p.rule_id for p in result.proofs.values()}, {r.id for r in rules})
                known_ids = {e.id for e in extraction.entities}
                self.assertTrue(all(set(f.atom.args) <= known_ids for f in result.facts))

    def test_university_projection_classification_and_organization_hierarchy(self):
        schema, rules, extraction, _ = profile("university")
        result = semi_naive(extraction, rules, schema)
        for target in (
            literal("teaches", "anna", "logic"),
            literal("teacher", "anna"),
            literal("part_of", "department", "university"),
            literal("teaches_at", "anna", "logic", "faculty"),
            literal("teaches_at", "anna", "logic", "university"),
        ):
            with self.subTest(target=target):
                self.assertIn(target, result.new_facts)
        self.assertGreater(result.proofs[literal("teacher", "anna")].round, 1)
        self.assertGreater(result.proofs[literal("teaches_at", "anna", "logic", "university")].round, 1)

    def test_university_dependency_composition_across_three_rounds(self):
        schema, rules, extraction, _ = profile("university")
        result = semi_naive(extraction, rules, schema)
        target = literal("needs_course", "boris", "sets")
        self.assertIn(literal("prerequisite_for", "sets", "discrete"), result.initial)
        self.assertNotIn(literal("requires", "discrete", "sets"), result.initial)
        self.assertEqual(result.proofs[literal("requires", "discrete", "sets")].rule_id,
                         "requires_from_prerequisite")
        self.assertIn(literal("requires", "ai", "sets"), result.new_facts)
        self.assertIn(literal("prerequisite_for", "sets", "ai"), result.new_facts)
        self.assertIn(target, result.new_facts)
        self.assertEqual(result.rounds, 3)
        self.assertEqual(result.proofs[target].round, 3)
        self.assertEqual(result.proofs[target].rule_id, "study_prerequisite")

    def test_inverse_input_spelling_preserves_complete_closure(self):
        variants = (
            ("university", "prerequisite_for", ("sets", "discrete"), "requires", ("discrete", "sets")),
            ("family", "child", ("vera", "boris"), "parent", ("boris", "vera")),
        )
        for name, inverse, inverse_args, direct, direct_args in variants:
            with self.subTest(profile=name):
                schema, rules, extraction, _ = profile(name)
                original = semi_naive(extraction, rules, schema)
                revised = replace(extraction, facts=tuple(
                    ExtractedFact(literal(direct, *direct_args), fact.evidence)
                    if fact.literal == literal(inverse, *inverse_args) else fact
                    for fact in extraction.facts
                ))
                alternative = semi_naive(revised, rules, schema)
                self.assertEqual(original.facts, alternative.facts)

    def test_collaboration_is_symmetric_but_not_transitive_or_reflexive(self):
        schema, rules, extraction, _ = profile("university")
        result = semi_naive(extraction, rules, schema)
        collaboration = {f.atom.args for f in result.facts if f.atom.predicate == "collaborates"}
        self.assertEqual(collaboration, {
            ("anna", "boris"), ("boris", "anna"), ("boris", "vera"), ("vera", "boris")
        })
        self.assertNotIn(literal("collaborates", "anna", "vera"), result.facts)
        self.assertNotIn(literal("requires", "logic", "ai"), result.facts)

    def test_shared_teaching_organization_does_not_imply_collaboration(self):
        schema, rules, extraction, _ = profile("university")
        extra = ExtractedFact(literal("teaches_at", "vera", "logic", "department"), ("offline input",))
        result = semi_naive(replace(extraction, facts=extraction.facts + (extra,)), rules, schema)
        self.assertNotIn(literal("collaborates", "anna", "vera"), result.facts)
        self.assertNotIn(literal("may_teach", "vera", "logic"), result.facts)
        self.assertIn(literal("teacher", "vera"), result.facts)

    def test_explicit_prohibition_derives_negative_permission_only(self):
        schema, rules, extraction, _ = profile("university")
        result = semi_naive(extraction, rules, schema)
        self.assertIn(literal("may_teach", "vera", "logic", negated=True), result.new_facts)
        self.assertNotIn(literal("may_teach", "boris", "logic", negated=True), result.facts)
        self.assertNotIn(literal("requires", "logic", "philosophy", negated=True), result.facts)
        self.assertNotIn(literal("requires", "ai", "philosophy"), result.facts)
        self.assertEqual({f for f in result.new_facts if f.negated}, {
            literal("may_teach", "vera", "logic", negated=True)
        })

    def test_permission_and_prohibition_are_reported_as_conflict_without_explosion(self):
        schema, rules, extraction, _ = profile("university")
        extra = ExtractedFact(literal("may_teach", "vera", "logic"), ("offline input",))
        revised = replace(extraction, facts=extraction.facts + (extra,))
        result = semi_naive(revised, rules, schema)
        self.assertEqual(result.conflicts, (GroundAtom("may_teach", ("vera", "logic")),))
        self.assertIn(literal("may_teach", "vera", "logic"), result.facts)
        self.assertIn(literal("may_teach", "vera", "logic", negated=True), result.facts)
        self.assertNotIn(literal("teacher", "vera"), result.facts)
        report = build_report(revised, result, rules, schema)
        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(len(report["conflicts"]), 1)

    def test_university_bfs_skips_mixed_sources_but_checks_pure_part_of(self):
        schema, rules, extraction, metadata = profile("university")
        result = semi_naive(extraction, rules, schema)
        checks = metadata["graph_checks"]
        report = build_report(extraction, result, rules, schema,
                              tuple(checks["bfs_predicates"]), tuple(checks["strict_orders"]),
                              acyclic_predicates=tuple(checks.get("acyclic_predicates", [])))
        self.assertEqual(report["status"], "ok")
        self.assertEqual({g["predicate"]: g["comparison"] for g in report["graph_checks"]},
                         {"requires": "skipped", "part_of": "passed"})

    def test_family_ancestors_join_and_parent_direction(self):
        schema, rules, extraction, _ = profile("family")
        result = semi_naive(extraction, rules, schema)
        self.assertIn(literal("child", "vera", "boris"), result.initial)
        self.assertNotIn(literal("parent", "boris", "vera"), result.initial)
        self.assertEqual(result.proofs[literal("parent", "boris", "vera")].rule_id,
                         "parent_from_child")
        self.assertIn(literal("child", "boris", "anna"), result.new_facts)
        self.assertIn(literal("ancestor", "anna", "gleb"), result.new_facts)
        self.assertEqual(result.proofs[literal("ancestor", "anna", "gleb")].round, 3)
        self.assertEqual({f.atom.args for f in result.facts if f.atom.predicate == "grandparent"},
                         {("anna", "vera"), ("boris", "gleb")})
        self.assertNotIn(literal("parent", "anna", "vera"), result.facts)
        self.assertNotIn(literal("parent", "boris", "anna"), result.facts)
        self.assertNotIn(literal("grandparent", "anna", "gleb"), result.facts)
        self.assertFalse(any(f.negated for f in result.new_facts))

    def test_family_spouse_symmetry_does_not_follow_from_parenthood(self):
        schema, rules, extraction, _ = profile("family")
        extra = ExtractedFact(literal("parent", "darya", "boris"), ("offline input",))
        # Удаляем явно указанный брак: общий ребёнок сам по себе его не доказывает.
        revised = replace(extraction, facts=tuple(
            f for f in extraction.facts if f.literal.atom.predicate != "spouse"
        ) + (extra,))
        result = semi_naive(revised, rules, schema)
        self.assertFalse(any(f.atom.predicate == "spouse" for f in result.facts))
        ordinary = semi_naive(extraction, rules, schema)
        self.assertEqual({f.atom.args for f in ordinary.facts if f.atom.predicate == "spouse"},
                         {("anna", "darya"), ("darya", "anna")})

    def test_family_recursive_ancestor_matches_bfs_without_closing_parent(self):
        schema, rules, extraction, metadata = profile("family")
        result = semi_naive(extraction, rules, schema)
        # Независимо нормализуем явные child в рёбра parent до обхода.
        # Из самого результата движка рёбра для эталона не берём.
        normalized_input = result.initial | frozenset(
            literal("parent", f.atom.args[1], f.atom.args[0])
            for f in result.initial if f.atom.predicate == "child" and not f.negated
        )
        bfs = transitive_closure_bfs(normalized_input, "parent", "ancestor")
        self.assertEqual({f for f in result.facts if f.atom.predicate == "ancestor"}, set(bfs.facts))
        report = build_report(extraction, result, rules, schema,
                              strict_orders=tuple(metadata["graph_checks"]["strict_orders"]),
                              acyclic_predicates=tuple(metadata["graph_checks"].get("acyclic_predicates", [])))
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["graph_checks"][0]["comparison"], "skipped")
        self.assertFalse(report["graph_checks"][0]["strict_cycle_vertices"])
        self.assertTrue(report["graph_checks"][0]["acyclic_required"])
        self.assertFalse(report["graph_checks"][0]["strict_order"])


if __name__ == "__main__":
    unittest.main()
