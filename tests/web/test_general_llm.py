"""Offline tests for general FOL formalization and hand-checked fixtures."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

import httpx

from factlab.web.general_llm import TEXT_PLACEHOLDER, bundle_json_schema, formalize_text
from factlab.web.llm_client import LLMError, LLMSettings, request_json
from factlab.web.logic_engine import analyze_bundle, validate_bundle


APP_DIR = Path(__file__).resolve().parents[2] / "src" / "factlab"
FIXTURES = APP_DIR / "data" / "general_examples"
PROMPT = (APP_DIR / "prompts" / "formalization.txt").read_text(encoding="utf-8")


def fixture(name: str) -> tuple[str, dict]:
    directory = FIXTURES / name
    return (
        (directory / "text.txt").read_text(encoding="utf-8"),
        json.loads((directory / "bundle.json").read_text(encoding="utf-8")),
    )


def completion(content: str, *, finish_reason="stop", usage=None):
    result = {
        "id": "chatcmpl-general-test", "object": "chat.completion", "created": 1,
        "model": "test-general",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


class GeneralLLMTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = LLMSettings("test-general-secret", "https://provider.invalid/v1", "test-general")
        self.requests = []

    async def mock_call(self, *, settings=None, text=None, template=PROMPT, payload=None, status=200, model_override=None):
        source, bundle = fixture("disjunction")

        def handler(request):
            self.requests.append(request)
            return httpx.Response(status, json=completion(json.dumps(bundle, ensure_ascii=False)) if payload is None else payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await formalize_text(
                source if text is None else text,
                template, settings or self.settings, model_override,
                http_client=client,
            )
            self.assertFalse(client.is_closed)
            return result

    async def test_one_call_contains_only_current_text_and_general_contract(self):
        result = await self.mock_call()
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].url.path, "/v1/chat/completions")
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(len(body["messages"]), 2)
        prompt = body["messages"][1]["content"]
        self.assertIn(fixture("disjunction")[0], prompt)
        self.assertNotIn(TEXT_PLACEHOLDER, prompt)
        self.assertNotIn("ЗАМЕНИ ЭТУ СТРОКУ КОРРЕКТНОЙ СХЕМОЙ JSON", prompt)
        self.assertNotIn("temperature", body)
        self.assertEqual(json.loads(result.raw_json)["version"], "1.0")
        self.assertIsNone(result.usage)

    async def test_own_recursive_json_schema_without_silent_downgrade(self):
        await self.mock_call(settings=replace(self.settings, response_format="json_schema"))
        body = json.loads(self.requests[0].content)
        response_format = body["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(schema, bundle_json_schema())
        self.assertEqual(set(schema["properties"]), {"version", "sorts", "predicates", "entities", "premises", "proposals", "queries", "unparsed"})
        self.assertNotIn("schema_version", schema["properties"])
        self.assertNotIn("facts", schema["properties"])
        self.assertEqual(schema["properties"]["premises"]["items"]["properties"]["formula"], {"$ref": "#/$defs/formula"})

    async def test_text_format_remains_explicitly_unconstrained_at_api_level(self):
        await self.mock_call(settings=replace(self.settings, response_format="text"))
        self.assertNotIn("response_format", json.loads(self.requests[0].content))

    async def test_format_rejection_does_not_trigger_paid_fallback(self):
        with self.assertRaises(LLMError) as raised:
            await self.mock_call(
                settings=replace(self.settings, response_format="json_schema"),
                status=400,
                payload={"error": {"message": "provider unsupported recursive schema test-general-secret"}},
            )
        self.assertEqual(raised.exception.code, "request_rejected")
        self.assertNotIn("test-general-secret", str(raised.exception))
        self.assertEqual(len(self.requests), 1)

    async def test_missing_or_duplicate_placeholder_is_rejected_before_request(self):
        for template in ("template without placeholder", TEXT_PLACEHOLDER + TEXT_PLACEHOLDER):
            with self.assertRaises(LLMError) as raised:
                await self.mock_call(template=template)
            self.assertEqual(raised.exception.code, "invalid_prompt")
        self.assertEqual(self.requests, [])

    async def test_marker_inside_input_is_data_not_second_substitution(self):
        await self.mock_call(text=f"source contains {TEXT_PLACEHOLDER}")
        prompt = json.loads(self.requests[0].content)["messages"][1]["content"]
        self.assertIn(f"<source_text>\nsource contains {TEXT_PLACEHOLDER}\n</source_text>", prompt)

    async def test_reasoning_options_and_usage_are_preserved(self):
        usage = {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700, "completion_tokens_details": {"reasoning_tokens": 150}, "prompt_tokens_details": {"cached_tokens": 100}}
        result = await self.mock_call(
            settings=replace(self.settings, reasoning_effort="low", token_limit_param="max_completion_tokens"),
            payload=completion("{}", usage=usage),
            model_override="overridden-model",
        )
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["max_completion_tokens"], 6000)
        self.assertEqual(result.model, "overridden-model")
        self.assertEqual(result.usage["total_tokens"], 700)
        self.assertEqual(result.usage["completion_tokens_details"]["reasoning_tokens"], 150)

    async def test_no_model_retry_to_repair_non_json(self):
        result = await self.mock_call(payload=completion("not-json"))
        self.assertEqual(result.raw_json, "not-json")
        self.assertEqual(len(self.requests), 1)

    async def test_truncation_keeps_usage_and_never_retries(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 6000, "total_tokens": 6100}
        with self.assertRaises(LLMError) as raised:
            await self.mock_call(payload=completion("{", finish_reason="length", usage=usage))
        self.assertEqual(raised.exception.code, "truncated_response")
        self.assertEqual(raised.exception.usage["total_tokens"], 6100)
        self.assertEqual(len(self.requests), 1)

    async def test_shared_api_requires_schema_when_strict(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: self.fail("No request expected"))) as client:
            with self.assertRaises(LLMError) as raised:
                await request_json([], replace(self.settings, response_format="json_schema"), http_client=client)
        self.assertEqual(raised.exception.code, "invalid_schema")

    def test_strict_schema_has_only_closed_required_objects_and_all_fol_ops(self):
        schema = bundle_json_schema()

        def verify(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertFalse(node["additionalProperties"])
                    self.assertEqual(set(node["properties"]), set(node["required"]))
                for value in node.values():
                    verify(value)
            elif isinstance(node, list):
                for value in node:
                    verify(value)

        verify(schema)
        ops = {op for variant in schema["$defs"]["formula"]["anyOf"] for op in variant["properties"]["op"]["enum"]}
        self.assertEqual(ops, {"atom", "not", "and", "or", "implies", "iff", "forall", "exists", "eq", "true", "false"})
        self.assertEqual(len(schema["$defs"]["term"]["anyOf"]), 2)


class GeneralFixtureTests(unittest.TestCase):
    def test_all_fixtures_pass_actual_server_validator(self):
        for name in ("discovery", "legend", "siblings", "disjunction"):
            with self.subTest(name=name):
                source, bundle = fixture(name)
                self.assertEqual(validate_bundle(bundle, source), bundle)

    def test_legend_keeps_assignment_and_counterfactual_outside_premises(self):
        source, bundle = fixture("legend")
        self.assertEqual(len(bundle["premises"]), 8)
        self.assertEqual(len(bundle["entities"]), 2)
        self.assertEqual(len(bundle["predicates"]), 3)
        self.assertEqual(bundle["proposals"], [])
        self.assertEqual({item["formula"]["op"] for item in bundle["premises"]}, {"forall", "exists", "atom", "not"})
        self.assertTrue(bundle["unparsed"][0]["text"].startswith("Ниже приведена легенда. Запишите"))
        self.assertIn("Маркс уже не понадобился бы", bundle["unparsed"][1]["text"])
        self.assertNotIn("Маркс", {item["label"] for item in bundle["entities"]})
        report = analyze_bundle(validate_bundle(bundle, source), max_named_atoms=0)
        self.assertEqual(report["consistency"], "sat")
        self.assertEqual([item["status"] for item in report["queries"]], ["entailed", "refuted", "entailed", "entailed"])

    def test_sibling_definitions_are_separate_and_do_not_imply_transitivity_or_sex_from_name(self):
        source, bundle = fixture("siblings")
        validated = validate_bundle(bundle, source)
        self.assertEqual(len(bundle["premises"]), 2)
        self.assertEqual(len(bundle["proposals"]), 7)
        self.assertEqual({item["origin"] for item in bundle["proposals"]}, {"definition"})
        no_definitions = analyze_bundle(validated, max_named_atoms=0)
        self.assertEqual([item["status"] for item in no_definitions["queries"]], ["undetermined"] * 3)
        definitions = analyze_bundle(validated, [item["id"] for item in bundle["proposals"]], max_named_atoms=0)
        self.assertEqual(definitions["consistency"], "sat")
        self.assertFalse(definitions["limits"]["unique_names"])
        self.assertEqual([item["status"] for item in definitions["queries"]], ["entailed", "undetermined", "undetermined"])

    def test_disjunction_is_not_split_into_ground_premises(self):
        source, bundle = fixture("disjunction")
        self.assertEqual([item["formula"]["op"] for item in bundle["premises"]], ["or", "not"])
        report = analyze_bundle(validate_bundle(bundle, source))
        self.assertEqual(report["consistency"], "sat")
        self.assertEqual([item["status"] for item in report["queries"]], ["entailed", "refuted"])
        derived = [item for item in report["facts"] if item["origin"] == "derived"]
        self.assertTrue(any(item["predicate"] == "studies" and item["args"] == ["anna", "ai"] and not item["negated"] for item in derived))


if __name__ == "__main__":
    unittest.main()
