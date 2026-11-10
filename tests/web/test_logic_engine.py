"""Семантика общего режима: открытые области, кванторы, вывод и границы."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import unittest
from unittest.mock import patch

import z3

from factlab.web.logic_engine import (
    LogicValidationError, analyze_bundle, format_formula, validate_bundle,
)


def ent(name):
    return {"entity": name}


def var(name):
    return {"var": name}


def atom(name, *args):
    return {"op": "atom", "predicate": name, "args": list(args)}


def neg(formula):
    return {"op": "not", "arg": formula}


def both(*args):
    return {"op": "and", "args": list(args)}


def either(*args):
    return {"op": "or", "args": list(args)}


def implies(left, right):
    return {"op": "implies", "left": left, "right": right}


def eq(left, right):
    return {"op": "eq", "left": left, "right": right}


def quant(op, body, *names, sort="person"):
    return {"op": op, "vars": [{"id": name, "sort": sort} for name in names], "body": body}


def bundle(predicates=None, entities=None):
    predicates = {"p": []} if predicates is None else predicates
    entities = {} if entities is None else entities
    source = "Исходное утверждение. " + " ".join(entities.values())
    data = {
        "version": "1.0", "sorts": [{"id": "person", "label": "Человек"}],
        "predicates": [{"id": name, "label": name, "arg_sorts": sorts} for name, sorts in predicates.items()],
        "entities": [{"id": name, "label": label, "sort": "person", "mentions": [label]}
                     for name, label in entities.items()],
        "premises": [], "proposals": [], "queries": [], "unparsed": [],
    }
    return data, source


def premise(data, name, formula, evidence="Исходное утверждение."):
    data["premises"].append({"id": name, "formula": formula, "evidence": [evidence], "explanation": name})


def proposal(data, name, formula, origin="definition"):
    data["proposals"].append({"id": name, "formula": formula, "origin": origin, "explanation": name})


def query(data, name, formula):
    data["queries"].append({"id": name, "formula": formula, "text": name})


def statuses(report):
    return {item["id"]: item["status"] for item in report["queries"]}


def facts(report):
    return {(item["predicate"], tuple(item["args"]), item["negated"]) for item in report["facts"]}


def legend():
    source = (
        "Любой марксист — диалектик, но не всякий диалектик — марксист. "
        "Любой марксист — материалист, но не всякий материалист — марксист. "
        "Гегель был диалектик, но не материалист. "
        "Фейербах был материалист, но не диалектик. "
        "Итак, если бы Гегель и Фейербах могли объединиться в один кружок, "
        "то Маркс уже не понадобился бы."
    )
    data, _ = bundle({"marxist": ["person"], "dialectician": ["person"],
                      "materialist": ["person"], "circle": [], "marx_unneeded": []},
                     {"hegel": "Гегель", "feuerbach": "Фейербах"})
    x = var("x")
    m, d, t = (atom(name, x) for name in ("marxist", "dialectician", "materialist"))
    premise(data, "all_m_d", quant("forall", implies(m, d), "x"), "Любой марксист — диалектик")
    premise(data, "not_all_d_m", quant("exists", both(d, neg(m)), "x"), "не всякий диалектик — марксист")
    premise(data, "all_m_t", quant("forall", implies(m, t), "x"), "Любой марксист — материалист")
    premise(data, "not_all_t_m", quant("exists", both(t, neg(m)), "x"), "не всякий материалист — марксист")
    premise(data, "hegel_d", atom("dialectician", ent("hegel")), "Гегель был диалектик")
    premise(data, "hegel_not_t", neg(atom("materialist", ent("hegel"))), "но не материалист")
    premise(data, "feuerbach_t", atom("materialist", ent("feuerbach")), "Фейербах был материалист")
    premise(data, "feuerbach_not_d", neg(atom("dialectician", ent("feuerbach"))), "но не диалектик")
    query(data, "true_consequence", neg(atom("marxist", ent("hegel"))))
    query(data, "false_consequence", atom("marxist", ent("hegel")))
    query(data, "circle_conclusion", implies(atom("circle"), atom("marx_unneeded")))
    data["unparsed"] = [{"text": "Маркс уже не понадобился бы", "reason": "Не задан критерий нужности человека."}]
    return data, source


def siblings():
    data, source = bundle({"brother": ["person", "person"], "sister": ["person", "person"],
                           "sibling": ["person", "person"], "male": ["person"], "female": ["person"]},
                          {"gleb": "Глеб", "anya": "Аня", "petya": "Петя"})
    premise(data, "gleb_brother_anya", atom("brother", ent("gleb"), ent("anya")))
    premise(data, "anya_sister_petya", atom("sister", ent("anya"), ent("petya")))
    x, y = var("x"), var("y")
    proposal(data, "brother_definition", quant("forall", implies(atom("brother", x, y), both(atom("sibling", x, y), atom("male", x))), "x", "y"))
    proposal(data, "sister_definition", quant("forall", implies(atom("sister", x, y), both(atom("sibling", x, y), atom("female", x))), "x", "y"))
    proposal(data, "sibling_symmetry", quant("forall", implies(atom("sibling", x, y), atom("sibling", y, x)), "x", "y"))
    proposal(data, "male_sibling", quant("forall", implies(both(atom("male", x), atom("sibling", x, y)), atom("brother", x, y)), "x", "y"))
    proposal(data, "female_sibling", quant("forall", implies(both(atom("female", x), atom("sibling", x, y)), atom("sister", x, y)), "x", "y"))
    proposal(data, "no_self_sibling", quant("forall", neg(atom("sibling", x, x)), "x"))
    query(data, "anya_sister_gleb", atom("sister", ent("anya"), ent("gleb")))
    query(data, "gleb_brother_petya", atom("brother", ent("gleb"), ent("petya")))
    return data, source


class LogicEngineTests(unittest.TestCase):
    def test_discovers_positive_and_negative_chain_consequences_without_queries(self):
        data, source = bundle({"robot": ["person"], "mechanism": ["person"],
                               "device": ["person"]}, {"ray": "Рэй", "nora": "Нора"})
        x = var("x")
        premise(data, "robots_are_mechanisms", quant("forall", implies(atom("robot", x), atom("mechanism", x)), "x"))
        premise(data, "mechanisms_are_devices", quant("forall", implies(atom("mechanism", x), atom("device", x)), "x"))
        premise(data, "ray_is_robot", atom("robot", ent("ray")))
        premise(data, "nora_is_device", atom("device", ent("nora")))
        premise(data, "nora_is_not_mechanism", neg(atom("mechanism", ent("nora"))))

        report = analyze_bundle(validate_bundle(data, source))

        self.assertEqual(data["queries"], [])
        self.assertEqual(report["queries"], [])
        self.assertEqual(report["consistency"], "sat")
        extracted = {(item["predicate"], tuple(item["args"]), item["negated"])
                     for item in report["facts"] if item["origin"] == "extracted"}
        derived = {(item["predicate"], tuple(item["args"]), item["negated"])
                   for item in report["facts"] if item["origin"] == "derived"}
        self.assertEqual(extracted, {("robot", ("ray",), False),
                                     ("device", ("nora",), False),
                                     ("mechanism", ("nora",), True)})
        self.assertEqual(derived, {("mechanism", ("ray",), False),
                                   ("device", ("ray",), False),
                                   ("robot", ("nora",), True)})
        self.assertTrue(all(item["support"] for item in report["facts"]))
        self.assertEqual(report["statistics"]["named_atoms_checked"], 6)
        self.assertFalse(report["statistics"]["truncated"])

    def test_discovers_disjunctive_syllogism_without_queries(self):
        data, source = bundle({"studies": ["person", "course"]}, {"anna": "Анна"})
        data["sorts"].append({"id": "course", "label": "Курс"})
        for name, label in (("logic", "Логика"), ("ai", "ИИ")):
            data["entities"].append({"id": name, "label": label, "sort": "course", "mentions": [label]})
            source += " " + label
        premise(data, "one_of_two_courses", either(atom("studies", ent("anna"), ent("logic")),
                                                   atom("studies", ent("anna"), ent("ai"))))
        premise(data, "not_logic", neg(atom("studies", ent("anna"), ent("logic"))))

        report = analyze_bundle(validate_bundle(data, source))

        self.assertEqual(data["queries"], [])
        self.assertEqual(report["queries"], [])
        self.assertEqual(report["consistency"], "sat")
        self.assertEqual(facts(report), {("studies", ("anna", "logic"), True),
                                        ("studies", ("anna", "ai"), False)})
        origins = {tuple(item["args"]): item["origin"] for item in report["facts"]}
        self.assertEqual(origins, {("anna", "logic"): "extracted", ("anna", "ai"): "derived"})
        derived = next(item for item in report["facts"] if item["origin"] == "derived")
        self.assertEqual(set(derived["support"]), {"one_of_two_courses", "not_logic"})
        self.assertFalse(report["statistics"]["truncated"])

    def test_legend_has_two_negative_consequences_and_no_rhetorical_inference(self):
        data, source = legend()
        normalized = validate_bundle(json.dumps(data, ensure_ascii=False), source)
        report = analyze_bundle(normalized)
        self.assertEqual(report["consistency"], "sat")
        self.assertEqual(statuses(report), {"true_consequence": "entailed", "false_consequence": "refuted", "circle_conclusion": "undetermined"})
        self.assertEqual(len(report["facts"]), 6)
        derived = [item for item in report["facts"] if item["origin"] == "derived"]
        self.assertEqual({tuple(item["args"]) for item in derived}, {("hegel",), ("feuerbach",)})
        self.assertTrue(all(item["predicate"] == "marxist" and item["negated"] for item in derived))
        self.assertEqual(sum(item["origin"] == "extracted" for item in report["facts"]), 4)
        for item in derived:
            self.assertTrue(item["support"])
            self.assertIn("¬marxist", item["display"])
        self.assertFalse(report["statistics"]["truncated"])

    def test_existential_can_require_an_unnamed_witness(self):
        data, source = bundle({"p": ["person"]}, {"a": "Анна"})
        premise(data, "someone", quant("exists", atom("p", var("x")), "x"))
        premise(data, "not_anna", neg(atom("p", ent("a"))))
        query(data, "unnamed_exists", quant("exists", neg(eq(var("x"), ent("a"))), "x"))
        report = analyze_bundle(validate_bundle(data, source))
        self.assertEqual(report["consistency"], "sat")
        self.assertEqual(statuses(report)["unnamed_exists"], "entailed")
        self.assertEqual(facts(report), {("p", ("a",), True)})
        self.assertTrue(report["limits"]["open_domain"])

    def test_existence_does_not_force_property_of_the_only_named_object(self):
        data, source = bundle({"p": ["person"]}, {"a": "Анна"})
        premise(data, "someone", quant("exists", atom("p", var("x")), "x"))
        query(data, "anna_p", atom("p", ent("a")))
        report = analyze_bundle(validate_bundle(data, source))
        self.assertEqual(statuses(report)["anna_p"], "undetermined")
        self.assertEqual(report["facts"], [])

    def test_names_have_no_implicit_unique_name_assumption(self):
        data, source = bundle({}, {"a": "Анна", "b": "Аня"})
        query(data, "same_person", eq(ent("a"), ent("b")))
        report = analyze_bundle(validate_bundle(data, source))
        self.assertEqual(statuses(report)["same_person"], "undetermined")
        self.assertFalse(report["limits"]["unique_names"])
        premise(data, "same", eq(ent("a"), ent("b")))
        report = analyze_bundle(validate_bundle(data, source))
        self.assertEqual(statuses(report)["same_person"], "entailed")

    def test_explicit_inequality_is_supported(self):
        data, source = bundle({}, {"a": "Анна", "b": "Аня"})
        premise(data, "different", neg(eq(ent("a"), ent("b"))))
        query(data, "same", eq(ent("a"), ent("b")))
        self.assertEqual(statuses(analyze_bundle(validate_bundle(data, source)))["same"], "refuted")

    def test_disjunctive_syllogism_on_propositional_letters(self):
        data, source = bundle({"p": [], "q": []})
        premise(data, "choice", either(atom("p"), atom("q")))
        premise(data, "not_p", neg(atom("p")))
        query(data, "q_holds", atom("q"))
        report = analyze_bundle(validate_bundle(data, source))
        self.assertEqual(statuses(report)["q_holds"], "entailed")
        self.assertEqual(facts(report), {("p", (), True), ("q", (), False)})
        q_fact = next(item for item in report["facts"] if item["predicate"] == "q")
        self.assertEqual(set(q_fact["support"]), {"choice", "not_p"})

    def test_disjunction_alone_does_not_select_a_disjunct(self):
        data, source = bundle({"p": [], "q": []})
        premise(data, "choice", either(atom("p"), atom("q")))
        query(data, "q_holds", atom("q"))
        report = analyze_bundle(validate_bundle(data, source))
        self.assertEqual(statuses(report)["q_holds"], "undetermined")
        self.assertEqual(report["facts"], [])

    def test_biconditional_and_universal_instantiation(self):
        data, source = bundle({"p": ["person"], "q": ["person"]}, {"a": "Анна"})
        premise(data, "equivalent", quant("forall", {"op": "iff", "left": atom("p", var("x")), "right": atom("q", var("x"))}, "x"))
        premise(data, "anna_p", atom("p", ent("a")))
        query(data, "anna_q", atom("q", ent("a")))
        report = analyze_bundle(validate_bundle(data, source))
        self.assertEqual(statuses(report)["anna_q"], "entailed")
        self.assertIn(("q", ("a",), False), facts(report))

    def test_false_and_true_constants_and_empty_connectives(self):
        data, source = bundle({})
        query(data, "truth", {"op": "true"})
        query(data, "falsehood", {"op": "false"})
        query(data, "empty_and", both())
        query(data, "empty_or", either())
        result = statuses(analyze_bundle(validate_bundle(data, source)))
        self.assertEqual(result, {"truth": "entailed", "falsehood": "refuted", "empty_and": "entailed", "empty_or": "refuted"})

    def test_definitions_are_inactive_until_selected(self):
        data, source = siblings()
        validated = validate_bundle(data, source)
        original = analyze_bundle(validated)
        self.assertEqual(statuses(original)["anya_sister_gleb"], "undetermined")
        self.assertTrue(all(not item["accepted"] for item in original["proposals"]))
        ids = [item["id"] for item in data["proposals"]]
        expanded = analyze_bundle(validated, ids)
        self.assertEqual(expanded["consistency"], "sat")
        self.assertEqual(statuses(expanded)["anya_sister_gleb"], "entailed")
        self.assertEqual(statuses(expanded)["gleb_brother_petya"], "undetermined")
        self.assertTrue(all(item["accepted"] for item in expanded["proposals"]))

    def test_quantified_premise_and_its_negation_need_only_consistency_check(self):
        formula = quant("forall", quant("exists", atom("r", var("x"), var("y")), "y"), "x")
        for negative in (False, True):
            with self.subTest(negative=negative):
                data, source = bundle({"r": ["person", "person"]})
                asserted = neg(formula) if negative else formula
                premise(data, "given", asserted)
                query(data, "positive", formula)
                query(data, "negative", neg(formula))
                query(data, "opposite_of_assertion", neg(asserted))
                report = analyze_bundle(validate_bundle(data, source), max_named_atoms=0)
                self.assertEqual(report["consistency"], "sat")
                self.assertEqual(statuses(report), {
                    "positive": "refuted" if negative else "entailed",
                    "negative": "entailed" if negative else "refuted",
                    "opposite_of_assertion": "refuted",
                })
                self.assertTrue(all(item["support"] == ["given"] for item in report["queries"]))
                self.assertEqual(report["statistics"]["solver_checks"], 1)

    def test_only_accepted_quantified_proposals_enter_direct_query_cache(self):
        data, source = bundle({"p": ["person"]})
        formula = quant("forall", atom("p", var("x")), "x")
        proposal(data, "definition", formula)
        query(data, "positive", formula)
        query(data, "negative", neg(formula))
        normalized = validate_bundle(data, source)
        inactive = analyze_bundle(normalized, max_named_atoms=0)
        self.assertEqual(statuses(inactive), {"positive": "undetermined", "negative": "undetermined"})
        self.assertTrue(all(item["support"] == [] for item in inactive["queries"]))
        active = analyze_bundle(normalized, ["definition"], max_named_atoms=0)
        self.assertEqual(statuses(active), {"positive": "entailed", "negative": "refuted"})
        self.assertTrue(all(item["support"] == ["definition"] for item in active["queries"]))
        self.assertEqual(active["statistics"]["solver_checks"], 1)

    def test_sibling_transitivity_is_not_assumed(self):
        data, source = siblings()
        premise(data, "not_related", neg(atom("sibling", ent("gleb"), ent("petya"))))
        report = analyze_bundle(validate_bundle(data, source), [item["id"] for item in data["proposals"]])
        self.assertEqual(report["consistency"], "sat")
        self.assertEqual(statuses(report)["gleb_brother_petya"], "refuted")
        self.assertEqual(statuses(report)["anya_sister_gleb"], "entailed")

    def test_inconsistency_is_reported_instead_of_explosion(self):
        data, source = bundle({"p": [], "q": []})
        premise(data, "positive", atom("p"))
        premise(data, "negative", neg(atom("p")))
        query(data, "anything", atom("q"))
        report = analyze_bundle(validate_bundle(data, source))
        self.assertEqual(report["consistency"], "unsat")
        self.assertEqual(set(report["consistency_support"]), {"positive", "negative"})
        self.assertEqual(statuses(report)["anything"], "inconsistent")
        self.assertEqual(report["facts"], [])

    def test_accepted_assumption_can_explain_inconsistency(self):
        data, source = bundle()
        premise(data, "positive", atom("p"))
        proposal(data, "negative_assumption", neg(atom("p")), "assumption")
        normalized = validate_bundle(data, source)
        self.assertEqual(analyze_bundle(normalized)["consistency"], "sat")
        report = analyze_bundle(normalized, ["negative_assumption"])
        self.assertEqual(report["consistency"], "unsat")
        self.assertEqual(set(report["consistency_support"]), {"positive", "negative_assumption"})

    def test_selected_proposal_ids_are_checked(self):
        data, source = bundle()
        proposal(data, "definition", atom("p"))
        normalized = validate_bundle(data, source)
        for selection in (["missing"], ["definition", "definition"], "definition", [True]):
            with self.subTest(selection=selection), self.assertRaises(LogicValidationError):
                analyze_bundle(normalized, selection)

    def test_named_atom_budget_is_explicit_and_lazy(self):
        data, source = bundle({"relation": ["person", "person"]}, {f"p{i}": f"Имя{i}" for i in range(20)})
        report = analyze_bundle(validate_bundle(data, source), max_named_atoms=2)
        self.assertEqual(report["statistics"]["named_atoms_total"], 400)
        self.assertEqual(report["statistics"]["named_atoms_checked"], 2)
        self.assertTrue(report["statistics"]["truncated"])
        self.assertEqual(report["statistics"]["solver_checks"], 5)

    def test_global_deadline_returns_unknown_without_fake_falsehood(self):
        data, source = bundle()
        query(data, "p_holds", atom("p"))
        normalized = validate_bundle(data, source)
        with patch("factlab.web.logic_engine.monotonic", side_effect=[0.0] + [10.0] * 20):
            report = analyze_bundle(normalized, timeout_ms=1)
        self.assertEqual(report["consistency"], "unknown")
        self.assertEqual(statuses(report)["p_holds"], "unknown")
        self.assertEqual(report["facts"], [])
        self.assertEqual(report["statistics"]["solver_checks"], 0)
        self.assertTrue(report["statistics"]["deadline_reached"])

    def test_solver_unknown_during_query_is_not_undetermined_or_refuted(self):
        data, source = bundle()
        query(data, "p_holds", atom("p"))
        with patch.object(z3.Solver, "check", side_effect=[z3.sat, z3.unknown, z3.unknown]), patch.object(z3.Solver, "reason_unknown", return_value="timeout"):
            report = analyze_bundle(validate_bundle(data, source), max_named_atoms=0)
        self.assertEqual(report["consistency"], "sat")
        self.assertEqual(statuses(report)["p_holds"], "unknown")
        self.assertEqual(report["queries"][0]["reason_unknown"], "timeout")
        self.assertEqual(report["facts"], [])

    def test_two_analyses_do_not_share_symbols_or_constraints(self):
        data, source = bundle()
        premise(data, "p_true", atom("p"))
        first = analyze_bundle(validate_bundle(data, source))
        data["premises"][0]["formula"] = neg(atom("p"))
        second = analyze_bundle(validate_bundle(data, source))
        self.assertEqual(facts(first), {("p", (), False)})
        self.assertEqual(facts(second), {("p", (), True)})

    def test_threaded_analyses_use_private_z3_contexts(self):
        def run(negative):
            data, source = bundle()
            premise(data, "given", neg(atom("p")) if negative else atom("p"))
            return facts(analyze_bundle(validate_bundle(data, source)))

        with ThreadPoolExecutor(max_workers=2) as executor:
            positive = executor.submit(run, False)
            negative = executor.submit(run, True)
            self.assertEqual(positive.result(), {("p", (), False)})
            self.assertEqual(negative.result(), {("p", (), True)})

    def test_multisort_relation_and_equality(self):
        data, source = bundle({"p": ["person", "organization"]}, {"anna": "Анна"})
        data["sorts"].append({"id": "organization", "label": "Организация"})
        data["entities"].append({"id": "msu", "label": "МГУ", "sort": "organization", "mentions": ["МГУ"]})
        source += " МГУ"
        premise(data, "works", atom("p", ent("anna"), ent("msu")))
        query(data, "works_query", atom("p", ent("anna"), ent("msu")))
        report = analyze_bundle(validate_bundle(data, source))
        self.assertEqual(statuses(report)["works_query"], "entailed")
        query(data, "bad_equality", eq(ent("anna"), ent("msu")))
        with self.assertRaisesRegex(LogicValidationError, "разных сортов"):
            validate_bundle(data, source)

    def test_formatting_uses_labels_and_preserves_logical_structure(self):
        data, source = bundle({"p": ["person"]}, {"anna": "Анна"})
        data["predicates"][0]["label"] = "преподаватель"
        formula = quant("forall", implies(atom("p", var("x")), neg(eq(var("x"), ent("anna")))), "x")
        normalized = validate_bundle(data, source)
        rendered = format_formula(formula, normalized)
        self.assertIn("∀ x:Человек", rendered)
        self.assertIn("преподаватель(x)", rendered)
        self.assertIn("¬(x = Анна)", rendered)


class LogicValidationTests(unittest.TestCase):
    def setUp(self):
        self.data, self.source = bundle({"p": ["person"]}, {"a": "Анна"})
        premise(self.data, "pa", atom("p", ent("a")))

    def test_duplicate_json_fields_are_rejected(self):
        raw = json.dumps(self.data, ensure_ascii=False).replace('"version": "1.0"', '"version": "1.0", "version": "1.0"')
        with self.assertRaisesRegex(LogicValidationError, "Повторяющийся ключ"):
            validate_bundle(raw, self.source)

    def test_missing_extra_and_duplicate_ids_are_rejected(self):
        variations = []
        extra = deepcopy(self.data)
        extra["code"] = "print('never executed')"
        variations.append(extra)
        missing = deepcopy(self.data)
        missing.pop("unparsed")
        variations.append(missing)
        duplicate = deepcopy(self.data)
        duplicate["entities"].append(deepcopy(duplicate["entities"][0]))
        variations.append(duplicate)
        mixed_id = deepcopy(self.data)
        proposal(mixed_id, "pa", atom("p", ent("a")))
        variations.append(mixed_id)
        for data in variations:
            with self.subTest(data=data), self.assertRaises(LogicValidationError):
                validate_bundle(data, self.source)

    def test_unbound_variables_unknown_entities_and_wrong_arity_are_rejected(self):
        invalid = [atom("p", var("x")), atom("p", ent("missing")), atom("p"), atom("unknown", ent("a"))]
        for formula in invalid:
            self.data["premises"][0]["formula"] = formula
            with self.subTest(formula=formula), self.assertRaises(LogicValidationError):
                validate_bundle(self.data, self.source)

    def test_wrong_sort_and_term_shape_are_rejected(self):
        self.data["sorts"].append({"id": "other", "label": "Другое"})
        self.data["entities"][0]["sort"] = "other"
        with self.assertRaises(LogicValidationError):
            validate_bundle(self.data, self.source)
        self.data["entities"][0]["sort"] = "person"
        self.data["premises"][0]["formula"]["args"] = [{"entity": "a", "var": "x"}]
        with self.assertRaises(LogicValidationError):
            validate_bundle(self.data, self.source)

    def test_fabricated_evidence_mentions_and_unparsed_quotes_are_rejected(self):
        evidence = deepcopy(self.data)
        evidence["premises"][0]["evidence"] = ["Такого утверждения не было."]
        mentions = deepcopy(self.data)
        mentions["entities"][0]["mentions"] = ["Борис"]
        unparsed = deepcopy(self.data)
        unparsed["unparsed"] = [{"text": "несуществующий фрагмент", "reason": "unknown"}]
        for data in (evidence, mentions, unparsed):
            with self.subTest(data=data), self.assertRaisesRegex(LogicValidationError, "цитата отсутствует"):
                validate_bundle(data, self.source)

    def test_quote_formatting_is_tolerated_without_rewriting_user_data(self):
        source = (
            "Анна прочитала\r\n\t«Айболит»\u00a0—\u00a0и выступила.\n"
            "Неизвестно,\tчитала ли Ольга."
        )
        self.data["entities"][0]["mentions"] = ["анна"]
        self.data["premises"][0]["evidence"] = ['АННА ПРОЧИТАЛА "Аи\u0306болит" – И ВЫСТУПИЛА.']
        self.data["unparsed"] = [{"text": "неизвестно, читала ли ольга.", "reason": "unknown"}]
        self.assertEqual(validate_bundle(self.data, source), self.data)

    def test_quote_normalization_preserves_words_negation_and_numbers(self):
        source = "Анна не прочитала книгу за 0,5 часа."
        invalid_quotes = [
            "Анна прочитала книгу за 0,5 часа.",
            "Анна не прочитала книгу за 05 часа.",
            "Анна не прочитала книгу за 0.5 часа.",
            "Анна книгу не прочитала за 0,5 часа.",
            "Анна непрочитала книгу за 0,5 часа.",
            "Анна не прочитала роман за 0,5 часа.",
        ]
        for quote in invalid_quotes:
            self.data["premises"][0]["evidence"] = [quote]
            with self.subTest(quote=quote), self.assertRaisesRegex(LogicValidationError, "цитата отсутствует"):
                validate_bundle(self.data, source)

    def test_invalid_operations_nonstandard_constants_and_depth_are_rejected(self):
        self.data["premises"][0]["formula"] = {"op": "execute", "code": "exit()"}
        with self.assertRaises(LogicValidationError):
            validate_bundle(self.data, self.source)
        raw = json.dumps(self.data).replace('"1.0"', "NaN", 1)
        with self.assertRaises(LogicValidationError):
            validate_bundle(raw, self.source)
        formula = atom("p", ent("a"))
        for _ in range(40):
            formula = neg(formula)
        self.data["premises"][0]["formula"] = formula
        with self.assertRaisesRegex(LogicValidationError, "глубины"):
            validate_bundle(self.data, self.source)

    def test_nested_quantification_keeps_bindings_closed(self):
        good = quant("forall", quant("exists", eq(var("x"), var("y")), "y"), "x")
        self.data["premises"][0]["formula"] = good
        normalized = validate_bundle(self.data, self.source)
        self.assertEqual(analyze_bundle(normalized)["consistency"], "sat")
        self.data["premises"][0]["formula"] = quant("forall", atom("p", var("x")), "y")
        with self.assertRaisesRegex(LogicValidationError, "свободная"):
            validate_bundle(self.data, self.source)

    def test_analyze_validates_ast_even_for_direct_dict_callers(self):
        self.data["premises"][0]["formula"] = atom("p", var("x"))
        with self.assertRaises(LogicValidationError):
            analyze_bundle(self.data)

    def test_non_json_values_invalid_origins_and_oversized_dict_are_rejected(self):
        invalid_origin = deepcopy(self.data)
        proposal(invalid_origin, "bad_origin", atom("p", ent("a")))
        invalid_origin["proposals"][0]["origin"] = {"unexpected": "object"}
        non_json = deepcopy(self.data)
        non_json["unparsed"] = object()
        oversized = deepcopy(self.data)
        oversized["unparsed"] = [{"text": "x" * 200_001, "reason": "oversized"}]
        for data in (invalid_origin, non_json, oversized):
            with self.subTest(data_type=type(data)), self.assertRaises(LogicValidationError):
                validate_bundle(data, self.source)

    def test_budget_arguments_are_strict(self):
        for value in (0, 8_001, True, 1.5):
            with self.subTest(timeout=value), self.assertRaises(LogicValidationError):
                analyze_bundle(self.data, timeout_ms=value)
        for value in (-1, 121, True, 1.5):
            with self.subTest(atoms=value), self.assertRaises(LogicValidationError):
                analyze_bundle(self.data, max_named_atoms=value)


if __name__ == "__main__":
    unittest.main()
