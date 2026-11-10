"""Семантические проверки движка, включая независимый наивный эталон."""

from __future__ import annotations

import random
import unittest

from factlab.datalog.fact_models import (
    Entity, ExtractedFact, ExtractionResult, ExtractionSchema, GroundAtom,
    GroundLiteral, PredicateSpec, ValidationError,
)
from factlab.datalog.inference import LiteralPattern, Rule, Var, semi_naive


X, Y, Z = Var("X"), Var("Y"), Var("Z")


def lit(predicate, *args, negated=False):
    return GroundLiteral(GroundAtom(predicate, args), negated)


def pat(predicate, *args, negated=False):
    return LiteralPattern(predicate, args, negated)


def schema(negation=False):
    return ExtractionSchema(
        ("node",), tuple(
            PredicateSpec(name, ("node",) * arity, "Тестовый предикат")
            for name, arity in (("edge", 2), ("reachable", 2), ("p", 1),
                                ("q", 1), ("r", 1), ("tri", 3), ("pair", 2))
        ), negation,
    )


def extraction(*facts, names=("a", "b", "c", "d")):
    return ExtractionResult(
        tuple(Entity(name, name, "node", (name,)) for name in names),
        tuple(ExtractedFact(fact, ("fixture",)) for fact in facts), (),
    )


def transitive(predicate="edge"):
    return Rule("transitive", (pat(predicate, X, Y), pat(predicate, Y, Z)), pat(predicate, X, Z))


def naive_oracle(initial, rules):
    """Маленький независимый полный перебор; только тест, не production API."""
    known = set(initial)
    while True:
        pending = set()
        snapshot = tuple(known)
        for rule in rules:
            states = [dict()]
            for condition in rule.body:
                next_states = []
                for state in states:
                    for fact in snapshot:
                        if (condition.predicate != fact.atom.predicate
                                or condition.negated != fact.negated):
                            continue
                        updated = dict(state)
                        matched = True
                        for term, value in zip(condition.args, fact.atom.args):
                            if isinstance(term, Var):
                                if term.name in updated and updated[term.name] != value:
                                    matched = False
                                    break
                                updated[term.name] = value
                            elif term != value:
                                matched = False
                                break
                        if matched:
                            next_states.append(updated)
                states = next_states
            for state in states:
                pending.add(lit(rule.head.predicate, *(
                    state[term.name] if isinstance(term, Var) else term
                    for term in rule.head.args
                ), negated=rule.head.negated))
        pending -= known
        if not pending:
            return frozenset(known)
        known |= pending


class SemiNaiveTests(unittest.TestCase):
    def test_unary_bootstrap_and_productive_rounds(self):
        rules = (
            Rule("pq", (pat("p", X),), pat("q", X)),
            Rule("qr", (pat("q", X),), pat("r", X)),
        )
        result = semi_naive(extraction(lit("p", "a")), rules, schema())
        self.assertEqual(result.new_facts, {lit("q", "a"), lit("r", "a")})
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.proofs[lit("r", "a")].round, 2)

    def test_nonlinear_transitivity_long_chain(self):
        names = tuple(f"n{i}" for i in range(9))
        data = extraction(*(lit("edge", names[i], names[i + 1]) for i in range(8)), names=names)
        result = semi_naive(data, (transitive(),), schema())
        expected = {lit("edge", names[i], names[j]) for i in range(9) for j in range(i + 1, 9)}
        self.assertEqual(result.facts, expected)
        self.assertEqual(result.rounds, 3)
        self.assertEqual(len(result.new_facts), 28)

    def test_delta_partition_avoids_duplicate_all_delta_joins(self):
        data = extraction(lit("edge", "a", "b"), lit("edge", "b", "c"), lit("edge", "c", "d"))
        result = semi_naive(data, (transitive(),), schema())
        self.assertEqual(result.stats.completed_matches, 4)
        self.assertEqual(len(result.new_facts), 3)

    def test_cycles_terminate_and_include_positive_length_self_paths(self):
        data = extraction(lit("edge", "a", "b"), lit("edge", "b", "a"))
        result = semi_naive(data, (transitive(),), schema())
        self.assertEqual(result.new_facts, {lit("edge", "a", "a"), lit("edge", "b", "b")})
        self.assertEqual(len(result.facts), 4)

    def test_mixed_predicates_and_ternary_join(self):
        data = extraction(lit("tri", "a", "b", "c"), lit("p", "b"), lit("p", "d"))
        rule = Rule("ternary", (pat("tri", X, Y, Z), pat("p", Y)), pat("pair", X, Z))
        self.assertEqual(semi_naive(data, (rule,), schema()).new_facts, {lit("pair", "a", "c")})

    def test_ternary_head_and_disconnected_variables(self):
        data = extraction(lit("p", "a"), lit("p", "b"), lit("q", "c"))
        rule = Rule("cartesian", (pat("p", X), pat("q", Y)), pat("tri", X, Y, "d"))
        result = semi_naive(data, (rule,), schema())
        self.assertEqual(result.new_facts, {lit("tri", "a", "c", "d"), lit("tri", "b", "c", "d")})

    def test_repeated_variable_requires_equality_within_atom(self):
        data = extraction(lit("edge", "a", "a"), lit("edge", "a", "b"))
        rule = Rule("diagonal", (pat("edge", X, X),), pat("p", X))
        self.assertEqual(semi_naive(data, (rule,), schema()).new_facts, {lit("p", "a")})

    def test_repeated_variable_must_agree_across_body(self):
        data = extraction(lit("edge", "a", "b"), lit("p", "c"))
        rule = Rule("join", (pat("edge", X, Y), pat("p", Y)), pat("q", X))
        self.assertFalse(semi_naive(data, (rule,), schema()).new_facts)

    def test_ground_body_and_head_use_existing_constants(self):
        data = extraction(lit("p", "a"))
        rule = Rule("ground", (pat("p", "a"),), pat("q", "b"))
        self.assertEqual(semi_naive(data, (rule,), schema()).new_facts, {lit("q", "b")})

    def test_only_explicit_negative_body_matches(self):
        data = extraction(lit("p", "a", negated=True), lit("p", "b"))
        rule = Rule("negative", (pat("p", X, negated=True),), pat("q", X))
        result = semi_naive(data, (rule,), schema(True))
        self.assertEqual(result.new_facts, {lit("q", "a")})
        self.assertNotIn(lit("q", "c"), result.facts)

    def test_inferred_contradiction_reported_without_explosion(self):
        data = extraction(lit("p", "a"), lit("q", "a"))
        rules = (
            Rule("negate", (pat("q", X),), pat("p", X, negated=True)),
            Rule("follow_negative", (pat("p", X, negated=True),), pat("r", X)),
        )
        result = semi_naive(data, rules, schema(True))
        self.assertEqual(result.conflicts, (GroundAtom("p", ("a",)),))
        self.assertIn(lit("p", "a"), result.facts)
        self.assertIn(lit("p", "a", negated=True), result.facts)
        self.assertIn(lit("r", "a"), result.facts)
        self.assertNotIn(lit("r", "b"), result.facts)

    def test_proof_parents_are_from_earlier_rounds(self):
        data = extraction(lit("edge", "a", "b"), lit("edge", "b", "c"), lit("edge", "c", "d"))
        result = semi_naive(data, (transitive(),), schema())
        self.assertEqual(set(result.proofs), set(result.new_facts))
        for literal, proof in result.proofs.items():
            self.assertEqual(proof.rule_id, "transitive")
            self.assertEqual(len(proof.parents), 2)
            for parent in proof.parents:
                self.assertTrue(parent in result.initial or result.proofs[parent].round < proof.round)
            substitution = dict(proof.bindings)
            self.assertEqual(literal.atom.args, (substitution["X"], substitution["Z"]))

    def test_inputs_unchanged_and_proofs_deterministic(self):
        facts = (lit("edge", "a", "b"), lit("edge", "b", "c"), lit("edge", "c", "d"))
        data = extraction(*facts)
        before = tuple(data.facts)
        rules = (transitive(),)
        left = semi_naive(data, rules, schema())
        right = semi_naive(extraction(*reversed(facts)), rules, schema())
        self.assertEqual(data.facts, before)
        self.assertEqual(left, right)
        self.assertIsInstance(left.facts, frozenset)

    def test_empty_input_and_empty_rules(self):
        empty = semi_naive(extraction(), (transitive(),), schema())
        self.assertEqual(empty.rounds, 0)
        self.assertFalse(empty.facts)
        original = extraction(lit("p", "a"))
        result = semi_naive(original, (), schema())
        self.assertEqual(result.facts, result.initial)
        self.assertEqual(result.rounds, 0)

    def test_no_implicit_transitivity_or_symmetry(self):
        data = extraction(lit("edge", "a", "b"), lit("edge", "b", "c"))
        result = semi_naive(data, (), schema())
        self.assertNotIn(lit("edge", "a", "c"), result.facts)
        self.assertNotIn(lit("edge", "b", "a"), result.facts)

    def test_recursive_copy_then_transitive_closure_different_predicate(self):
        data = extraction(lit("edge", "a", "b"), lit("edge", "b", "c"))
        rules = (
            Rule("seed", (pat("edge", X, Y),), pat("reachable", X, Y)),
            Rule("extend", (pat("reachable", X, Y), pat("edge", Y, Z)), pat("reachable", X, Z)),
        )
        result = semi_naive(data, rules, schema())
        self.assertEqual(result.facts, naive_oracle(result.initial, rules))
        self.assertNotIn(lit("edge", "a", "c"), result.facts)
        self.assertIn(lit("reachable", "a", "c"), result.facts)

    def test_random_graphs_agree_with_independent_naive_oracle(self):
        randomizer = random.Random(4517)
        names = ("a", "b", "c", "d")
        rules = (
            transitive(),
            Rule("loop", (pat("edge", X, X),), pat("p", X)),
            Rule("pair", (pat("p", X), pat("p", Y)), pat("pair", X, Y)),
        )
        for trial in range(35):
            with self.subTest(trial=trial):
                data = extraction(*(lit("edge", a, b) for a in names for b in names if randomizer.random() < 0.2))
                result = semi_naive(data, rules, schema())
                self.assertEqual(result.facts, naive_oracle(result.initial, rules))


class RuleValidationTests(unittest.TestCase):
    def assert_invalid(self, rules, *, data=None, used_schema=None):
        with self.assertRaises(ValidationError):
            semi_naive(data or extraction(lit("p", "a")), rules, used_schema or schema())

    def test_unsafe_head_variable(self):
        self.assert_invalid((Rule("unsafe", (pat("p", X),), pat("q", Y)),))

    def test_unknown_predicate_and_wrong_arity(self):
        for pattern in (pat("missing", X), pat("edge", X)):
            with self.subTest(pattern=pattern):
                self.assert_invalid((Rule("invalid", (pattern,), pat("p", X)),))

    def test_fresh_constant_in_head_or_body_rejected(self):
        self.assert_invalid((Rule("fresh_head", (pat("p", X),), pat("q", "fresh")),))
        self.assert_invalid((Rule("fresh_body", (pat("p", "fresh"),), pat("q", "a")),))

    def test_negation_must_be_enabled_for_input_body_and_head(self):
        self.assert_invalid((Rule("negative_head", (pat("p", X),), pat("q", X, negated=True)),))
        self.assert_invalid((Rule("negative_body", (pat("p", X, negated=True),), pat("q", X)),))
        self.assert_invalid((), data=extraction(lit("p", "a", negated=True)))

    def test_rule_id_unique_and_nonempty_body(self):
        rule = Rule("same", (pat("p", X),), pat("q", X))
        self.assert_invalid((rule, rule))
        with self.assertRaises(ValidationError):
            Rule("empty", (), pat("p", "a"))

    def test_function_terms_and_string_variables_rejected(self):
        for argument in (123, ("f", "a"), "?X", "f(a)"):
            with self.subTest(argument=argument), self.assertRaises(ValidationError):
                pat("p", argument)
        # Строка "x" является константой, а не неявной переменной.
        self.assert_invalid((Rule("no_magic", (pat("p", "x"),), pat("q", "a")),))

    def test_typed_variables_and_constants(self):
        used_schema = ExtractionSchema(("person", "course"), (
            PredicateSpec("person", ("person",), "Человек"),
            PredicateSpec("course", ("course",), "Курс"),
        ), False)
        data = ExtractionResult((
            Entity("a", "А", "person", ("А",)),
            Entity("b", "Б", "course", ("Б",)),
        ), (ExtractedFact(lit("person", "a"), ("А",)),), ())
        self.assert_invalid((Rule("bad_variable", (pat("person", X),), pat("course", X)),),
                            data=data, used_schema=used_schema)
        self.assert_invalid((Rule("bad_constant", (pat("person", X),), pat("course", "a")),),
                            data=data, used_schema=used_schema)

    def test_manually_created_input_is_revalidated_against_schema(self):
        self.assert_invalid((), data=extraction(lit("missing", "a")))
        self.assert_invalid((), data=extraction(lit("edge", "a")))
        unknown_type = ExtractionResult((Entity("a", "А", "unknown", ("А",)),), (), ())
        self.assert_invalid((), data=unknown_type)


if __name__ == "__main__":
    unittest.main()
