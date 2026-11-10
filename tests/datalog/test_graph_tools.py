"""Проверка BFS и топологического плана; запуск: python -m unittest -v."""

import unittest

from factlab.datalog.fact_models import GroundAtom, GroundLiteral, ValidationError
from factlab.datalog.graph_tools import (
    GraphCycleError, prerequisite_order, transitive_closure_bfs,
)


def fact(*args: str, predicate: str = "requires", negated: bool = False) -> GroundLiteral:
    return GroundLiteral(GroundAtom(predicate, args), negated)


class TransitiveClosureTests(unittest.TestCase):
    def test_chain_and_new_facts(self):
        initial = (fact("a", "b"), fact("b", "c"), fact("c", "d"))
        result = transitive_closure_bfs(initial, "requires")
        self.assertEqual(len(result.facts), 6)
        self.assertEqual(result.new_facts, frozenset({
            fact("a", "c"), fact("a", "d"), fact("b", "d"),
        }))
        self.assertEqual(result.path("a", "d"), ("a", "b", "c", "d"))
        self.assertFalse(result.cycle_vertices)

    def test_branching_uses_shortest_path_and_sorted_tie(self):
        initial = [fact("a", "c"), fact("c", "d"), fact("a", "b"),
                   fact("b", "d"), fact("d", "e"), fact("a", "e")]
        result = transitive_closure_bfs(initial, "requires")
        self.assertEqual(result.path("a", "d"), ("a", "b", "d"))
        self.assertEqual(result.path("a", "e"), ("a", "e"))
        self.assertEqual(
            transitive_closure_bfs(reversed(initial), "requires").path("a", "d"),
            result.path("a", "d"),
        )

    def test_disconnected_graph(self):
        result = transitive_closure_bfs([fact("a", "b"), fact("x", "y")], "requires")
        self.assertEqual(len(result.facts), 2)
        self.assertIsNone(result.path("a", "y"))
        self.assertIsNone(result.path("missing", "missing"))
        self.assertIsNone(result.path("b", "a"))

    def test_nonreflexive_unless_nonempty_cycle(self):
        result = transitive_closure_bfs([fact("a", "b")], "requires")
        self.assertNotIn(fact("a", "a"), result.facts)
        self.assertNotIn(fact("b", "b"), result.facts)
        self.assertIsNone(result.path("a", "a"))
        self.assertIsNone(result.path("b", "b"))

    def test_self_loop(self):
        result = transitive_closure_bfs([fact("a", "a")], "requires")
        self.assertEqual(result.facts, frozenset({fact("a", "a")}))
        self.assertEqual(result.cycle_vertices, frozenset({"a"}))
        self.assertEqual(result.path("a", "a"), ("a", "a"))
        self.assertFalse(result.new_facts)

    def test_two_cycle_and_downstream_vertex(self):
        result = transitive_closure_bfs(
            [fact("a", "b"), fact("b", "a"), fact("b", "c")], "requires"
        )
        self.assertEqual(result.cycle_vertices, frozenset({"a", "b"}))
        self.assertEqual(result.path("a", "a"), ("a", "b", "a"))
        self.assertEqual(result.path("b", "b"), ("b", "a", "b"))
        self.assertIsNone(result.path("c", "c"))

    def test_three_cycle(self):
        result = transitive_closure_bfs(
            [fact("a", "b"), fact("b", "c"), fact("c", "a")], "requires"
        )
        self.assertEqual(len(result.facts), 9)
        self.assertEqual(result.cycle_vertices, frozenset({"a", "b", "c"}))
        self.assertEqual(result.path("a", "a"), ("a", "b", "c", "a"))

    def test_shortest_cycle_is_selected(self):
        result = transitive_closure_bfs(
            [fact("a", "b"), fact("b", "c"), fact("c", "a"), fact("b", "a")],
            "requires",
        )
        self.assertEqual(result.path("a", "a"), ("a", "b", "a"))
        with_loop = transitive_closure_bfs(result.facts | {fact("a", "a")}, "requires")
        self.assertEqual(with_loop.path("a", "a"), ("a", "a"))

    def test_negative_literals_do_not_create_edges(self):
        result = transitive_closure_bfs(
            [fact("a", "b"), fact("b", "c", negated=True), fact("a", "a", negated=True)],
            "requires",
        )
        self.assertEqual(result.facts, frozenset({fact("a", "b")}))
        self.assertFalse(result.cycle_vertices)
        self.assertIsNone(result.path("a", "c"))

    def test_positive_path_can_conflict_with_explicit_negative(self):
        result = transitive_closure_bfs(
            [fact("a", "b"), fact("b", "c"), fact("a", "c", negated=True)],
            "requires",
        )
        self.assertIn(fact("a", "c"), result.new_facts)

    def test_closure_in_other_predicate(self):
        initial = [fact("a", "b", predicate="parent"),
                   fact("b", "c", predicate="parent"),
                   fact("a", "b", predicate="ancestor"),
                   fact("c", "d", predicate="ancestor")]
        result = transitive_closure_bfs(initial, "parent", "ancestor")
        self.assertEqual(result.source_predicate, "parent")
        self.assertEqual(result.target_predicate, "ancestor")
        self.assertEqual(result.facts, frozenset({
            fact("a", "b", predicate="ancestor"),
            fact("a", "c", predicate="ancestor"),
            fact("b", "c", predicate="ancestor"),
        }))
        self.assertEqual(len(result.new_facts), 2)
        self.assertNotIn(fact("a", "d", predicate="ancestor"), result.facts)
        self.assertEqual(result.path("a", "c"), ("a", "b", "c"))

    def test_wrong_arity_is_rejected(self):
        for literal in (fact("a"), fact("a", "b", "c"), fact("a", negated=True)):
            with self.subTest(literal=literal), self.assertRaisesRegex(ValidationError, "бинарные"):
                transitive_closure_bfs([literal], "requires")

    def test_other_predicates_are_ignored_even_nonbinary(self):
        result = transitive_closure_bfs([fact("a", predicate="course")], "requires")
        self.assertFalse(result.facts)

    def test_generator_input_and_duplicate_edges(self):
        result = transitive_closure_bfs(
            (value for value in [fact("a", "b"), fact("a", "b"), fact("b", "c")]),
            "requires",
        )
        self.assertEqual(len(result.facts), 3)

    def test_invalid_input_and_predicate(self):
        with self.assertRaises(ValidationError):
            transitive_closure_bfs([GroundAtom("requires", ("a", "b"))], "requires")
        with self.assertRaises(ValidationError):
            transitive_closure_bfs([], "NotAName")
        with self.assertRaises(ValidationError):
            transitive_closure_bfs([], "requires", "")

    def test_paths_are_not_materialized(self):
        length = 35
        result = transitive_closure_bfs(
            (fact(f"n{i}", f"n{i+1}") for i in range(length)), "requires"
        )
        # На цепи хранится один предок на достижимую пару, а не длинные пути.
        self.assertEqual(
            sum(len(parents) for parents in result._predecessors.values()),
            (length + 1) * (length + 2) // 2,
        )
        self.assertEqual(len(result.path("n0", f"n{length}")), length + 1)
        with self.assertRaises(TypeError):
            result._predecessors["n0"]["n1"] = "n2"


class PrerequisiteOrderTests(unittest.TestCase):
    def test_prerequisite_first(self):
        self.assertEqual(
            prerequisite_order([fact("ai", "logic"), fact("logic", "discrete")]),
            ("discrete", "logic", "ai"),
        )

    def test_isolated_nodes_are_included_and_sorted(self):
        self.assertEqual(prerequisite_order([], nodes=["z", "a", "m", "a"]), ("a", "m", "z"))

    def test_sparse_graph_and_isolated_nodes(self):
        ordered = prerequisite_order([fact("c", "b")], nodes=["a", "d"])
        self.assertEqual(ordered, ("a", "b", "c", "d"))

    def test_negative_edges_ignored(self):
        self.assertEqual(prerequisite_order([fact("b", "a", negated=True)]), ())

    def test_cycle_rejected(self):
        with self.assertRaises(GraphCycleError):
            prerequisite_order([fact("a", "b"), fact("b", "a")])
        with self.assertRaises(GraphCycleError):
            prerequisite_order([fact("a", "a")])

    def test_wrong_arity_and_node_rejected(self):
        with self.assertRaises(ValidationError):
            prerequisite_order([fact("a")])
        with self.assertRaises(ValidationError):
            prerequisite_order([], nodes=["Invalid"])


if __name__ == "__main__":
    unittest.main()
