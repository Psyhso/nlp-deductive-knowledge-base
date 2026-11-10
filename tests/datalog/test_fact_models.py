"""Смысловые границы контракта: python -m unittest -v test_fact_models.py."""

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from factlab.datalog.fact_models import (
    ConflictError,
    Entity,
    ExtractionResult,
    GroundAtom,
    GroundLiteral,
    ValidationError,
    dump_extraction,
    dumps_extraction,
    load_extraction,
    load_schema,
    loads_extraction,
    loads_schema,
)


SOURCE = (
    "Петров преподаёт Логику. Для ИИ требуется Логика. "
    "ИИ не требует Философии. Для ИИ необходим курс Логика. "
    "Возможно, Петров преподаёт ИИ. Все преподаватели — люди."
)
SCHEMA = {
    "schema_version": "1.0",
    "entity_types": ["person", "course"],
    "predicates": [
        {"name": "teaches", "arg_types": ["person", "course"],
         "description": "Первый аргумент преподаёт курс, указанный вторым аргументом."},
        {"name": "requires", "arg_types": ["course", "course"],
         "description": "Первый курс требует предварительного изучения второго."},
        {"name": "course", "arg_types": ["course"],
         "description": "Объект явно назван учебным курсом."},
        {"name": "assignment", "arg_types": ["person", "course", "course"],
         "description": "Первый объект получил задание из второго курса для третьего курса."},
    ],
    "allow_explicit_negation": True,
}
EXTRACTION = {
    "schema_version": "1.0",
    "entities": [
        {"id": "petrov", "label": "Петров", "type": "person", "mentions": ["Петров"]},
        {"id": "logic", "label": "Логика", "type": "course", "mentions": ["Логику", "Логика"]},
        {"id": "ai", "label": "ИИ", "type": "course", "mentions": ["ИИ"]},
        {"id": "philosophy", "label": "Философия", "type": "course", "mentions": ["Философии"]},
    ],
    "facts": [
        {"predicate": "teaches", "args": ["petrov", "logic"], "negated": False,
         "evidence": ["Петров преподаёт Логику."]},
        {"predicate": "requires", "args": ["ai", "logic"], "negated": False,
         "evidence": ["Для ИИ требуется Логика."]},
        {"predicate": "requires", "args": ["ai", "philosophy"], "negated": True,
         "evidence": ["ИИ не требует Философии."]},
    ],
    "unparsed": [
        {"text": "Возможно, Петров преподаёт ИИ.", "reason": "Модальность."},
        {"text": "Все преподаватели — люди.", "reason": "Общее правило, не ground-факт."},
    ],
}


class FactModelsTests(unittest.TestCase):
    def setUp(self):
        self.schema = loads_schema(json.dumps(SCHEMA, ensure_ascii=False))
        self.data = copy.deepcopy(EXTRACTION)

    def parse(self, data=None, schema=None):
        return loads_extraction(
            json.dumps(self.data if data is None else data, ensure_ascii=False),
            SOURCE, self.schema if schema is None else schema,
        )

    def test_roundtrip_keeps_sign_and_quotes(self):
        result = self.parse()
        self.assertEqual(loads_extraction(dumps_extraction(result), SOURCE, self.schema), result)
        self.assertEqual(result.positive_atoms, frozenset({
            GroundAtom("teaches", ("petrov", "logic")),
            GroundAtom("requires", ("ai", "logic")),
        }))
        self.assertEqual(result.negative_atoms,
                         frozenset({GroundAtom("requires", ("ai", "philosophy"))}))
        self.assertIn("Философия", dumps_extraction(result))
        self.assertEqual(len(result.unparsed), 2)

    def test_file_roundtrip_and_schema_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            schema_path = Path(directory) / "схема.json"
            schema_path.write_text(json.dumps(SCHEMA, ensure_ascii=False), encoding="utf-8-sig")
            schema = load_schema(schema_path)
            output_path = Path(directory) / "результат.json"
            result = self.parse()
            dump_extraction(result, output_path)
            self.assertEqual(load_extraction(output_path, SOURCE, schema), result)

    def test_no_fact_is_created_from_entity_metadata(self):
        self.data["facts"] = []
        result = self.parse()
        self.assertEqual(result.facts, ())
        self.assertEqual(result.positive_atoms, frozenset())

    def test_missing_atom_stays_unknown(self):
        result = self.parse()
        absent = GroundAtom("teaches", ("petrov", "ai"))
        self.assertNotIn(absent, result.positive_atoms)
        self.assertNotIn(absent, result.negative_atoms)

    def test_explicit_negative_requires_permission(self):
        schema_json = copy.deepcopy(SCHEMA)
        schema_json["allow_explicit_negation"] = False
        schema = loads_schema(json.dumps(schema_json))
        with self.assertRaisesRegex(ValidationError, "отрицание отключено"):
            self.parse(schema=schema)
        self.data["facts"] = self.data["facts"][:2]
        self.assertEqual(len(self.parse(schema=schema).facts), 2)

    def test_negated_is_strict_boolean(self):
        for value in (0, 1, "false", None):
            with self.subTest(value=value):
                self.data["facts"][0]["negated"] = value
                with self.assertRaises(ValidationError):
                    self.parse()
        with self.assertRaises(ValidationError):
            GroundLiteral(GroundAtom("course", ("ai",)), 1)

    def test_duplicate_facts_merge_quotes_without_reordering(self):
        duplicate = copy.deepcopy(self.data["facts"][1])
        duplicate["evidence"] = ["Для ИИ необходим курс Логика.", "Для ИИ требуется Логика."]
        self.data["facts"].append(duplicate)
        result = self.parse()
        self.assertEqual(len(result.facts), 3)
        self.assertEqual(result.facts[1].evidence, (
            "Для ИИ требуется Логика.", "Для ИИ необходим курс Логика.",
        ))
        self.assertEqual(result.facts[0].literal.atom.predicate, "teaches")

    def test_positive_negative_conflict_is_not_silently_overwritten(self):
        opposite = copy.deepcopy(self.data["facts"][1])
        opposite["negated"] = True
        self.data["facts"].append(opposite)
        with self.assertRaisesRegex(ConflictError, "requires\\(ai, logic\\)"):
            self.parse()

    def test_duplicate_entity_id_is_rejected(self):
        self.data["entities"].append(copy.deepcopy(self.data["entities"][0]))
        with self.assertRaisesRegex(ValidationError, "повторяющийся id"):
            self.parse()

    def test_unknown_objects_and_variables_are_rejected(self):
        for argument in ("unknown_person", "X", "?x", "f(petrov)", {"id": "petrov"}):
            with self.subTest(argument=argument):
                self.data["facts"][0]["args"][0] = argument
                with self.assertRaises(ValidationError):
                    self.parse()

    def test_argument_types_and_positions_are_checked(self):
        self.data["facts"][0]["args"] = ["logic", "petrov"]
        with self.assertRaisesRegex(ValidationError, "ожидается тип"):
            self.parse()

    def test_arity_and_fixed_vocabulary_are_checked(self):
        self.data["facts"][0]["args"] = ["petrov"]
        with self.assertRaisesRegex(ValidationError, "ожидается 2 аргументов"):
            self.parse()
        self.data = copy.deepcopy(EXTRACTION)
        self.data["facts"][0]["predicate"] = "invents"
        with self.assertRaisesRegex(ValidationError, "неизвестный предикат"):
            self.parse()

    def test_unary_and_ternary_are_not_hardcoded_as_binary(self):
        self.data["facts"] = [
            {"predicate": "course", "args": ["logic"], "negated": False,
             "evidence": ["необходим курс Логика"]},
            {"predicate": "assignment", "args": ["petrov", "logic", "ai"],
             "negated": False, "evidence": [SOURCE]},
        ]
        result = self.parse()
        self.assertIn(GroundAtom("course", ("logic",)), result.positive_atoms)
        self.assertIn(GroundAtom("assignment", ("petrov", "logic", "ai")),
                      result.positive_atoms)
        # Структурный валидатор не проверяет, следует ли assignment по смыслу из цитаты.

    def test_quotes_must_occur_in_original_source(self):
        for section, key, invalid in (
            ("entities", "mentions", ["Несуществующее имя"]),
            ("facts", "evidence", ["Петров преподает Логику."]),
            ("unparsed", "text", "Другая фраза"),
        ):
            with self.subTest(section=section):
                self.data = copy.deepcopy(EXTRACTION)
                self.data[section][0][key] = invalid
                with self.assertRaisesRegex(ValidationError, "цитата отсутствует"):
                    self.parse()

    def test_missing_and_unknown_fields_include_rule_injection(self):
        for location in ("root", "entities", "facts", "unparsed"):
            for mode in ("missing", "extra"):
                with self.subTest(location=location, mode=mode):
                    self.data = copy.deepcopy(EXTRACTION)
                    target = self.data if location == "root" else self.data[location][0]
                    if mode == "missing":
                        del target[next(iter(target))]
                    else:
                        target["rules"] = [{"head": "invented"}]
                    with self.assertRaises(ValidationError):
                        self.parse()

    def test_empty_collections_allowed_but_mandatory_quotes_nonempty(self):
        result = self.parse({"schema_version": "1.0", "entities": [], "facts": [], "unparsed": []})
        self.assertEqual(result, ExtractionResult((), (), ()))
        for section, field in (("entities", "mentions"), ("facts", "evidence")):
            with self.subTest(section=section):
                self.data = copy.deepcopy(EXTRACTION)
                self.data[section][0][field] = []
                with self.assertRaises(ValidationError):
                    self.parse()

    def test_empty_and_bad_schema_values_are_rejected(self):
        mutations = [
            ("entity_types", ["course", "course"]),
            ("predicates", [{"name": "nullary", "arg_types": [], "description": "Описание."}]),
            ("predicates", [{"name": "unknown", "arg_types": ["missing_type"],
                             "description": "Описание."}]),
            ("predicates", [{"name": "unknown", "arg_types": ["course"], "description": ""}]),
            ("predicates", [SCHEMA["predicates"][0], SCHEMA["predicates"][0]]),
            ("allow_explicit_negation", 1),
            ("schema_version", 1.0),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                schema = copy.deepcopy(SCHEMA)
                schema[key] = value
                with self.assertRaises(ValidationError):
                    loads_schema(json.dumps(schema))

    def test_all_json_layers_reject_duplicate_keys(self):
        cases = (
            '{"schema_version":"1.0","schema_version":"1.0"}',
            '{"entities":[{"id":"a","id":"b"}]}',
            '{"facts":[{"negated":false,"negated":true}]}',
        )
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValidationError, "повторяющийся ключ"):
                    loads_extraction(raw, SOURCE, self.schema)
        with self.assertRaisesRegex(ValidationError, "повторяющийся ключ"):
            loads_schema('{"predicates":[{"name":"a","name":"b"}]}')

    def test_malformed_json_and_nonstandard_constants_are_rejected(self):
        for raw in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}',
                    "[]", "null", "```json\n{}\n```", '{"x":}', "{} trailing"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValidationError):
                    loads_extraction(raw, SOURCE, self.schema)

    def test_models_are_immutable_and_atoms_hashable(self):
        result = self.parse()
        with self.assertRaises(FrozenInstanceError):
            result.entities[0].label = "Изменено"
        with self.assertRaises(ValidationError):
            Entity("petrov", "Петров", "person", ["Петров"])
        self.assertIsInstance(result.facts, tuple)
        self.assertEqual(len({result.facts[0].literal.atom, result.facts[0].literal.atom}), 1)


if __name__ == "__main__":
    unittest.main()
