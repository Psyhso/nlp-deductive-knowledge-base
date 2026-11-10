"""Проверки всего HTTP-пайплайна без платных запросов к модели."""

import asyncio
import copy
import json
import unittest

import httpx
from fastapi.testclient import TestClient

from factlab.web.app import MAX_TEXT_CHARS, create_app, load_context
from factlab.web.llm_client import LLMError, LLMResponse, LLMSettings


SETTINGS = LLMSettings("test-key-must-not-leak", "https://provider.invalid/v1", "test-model")


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.context = load_context("basic")
        self.calls = []

        async def fake_extractor(**kwargs):
            self.calls.append(kwargs)
            return LLMResponse(
                self.context.demo_json, "test-model",
                {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
            )

        self.fake = fake_extractor
        self.client = TestClient(create_app(SETTINGS, self.fake, self.context))

    def test_bootstrap_and_static_do_not_expose_key(self):
        response = self.client.get("/api/bootstrap")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(SETTINGS.api_key, response.text)
        self.assertTrue(response.json()["configuration"]["ready"])
        self.assertTrue(response.json()["configuration"]["has_api_key"])
        self.assertIn("schema_json", response.json()["defaults"])
        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn("text/html", index.headers["content-type"])
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.calls, [])

    def test_demo_never_calls_llm(self):
        response = self.client.post("/api/demo")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["demo"])
        self.assertIsNone(data["usage"])
        self.assertEqual(data["source_text"], self.context.demo_text)
        self.assertEqual(data["report"]["statistics"]["new_facts"], 3)
        self.assertEqual(self.calls, [])

    def test_analyze_validates_and_infers_with_one_call(self):
        response = self.client.post(
            "/api/analyze", json={"text": self.context.demo_text, "model": "selected-model"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["model_override"], "selected-model")
        self.assertEqual(data["usage"]["total_tokens"], 600)
        self.assertFalse(data["demo"])
        self.assertEqual(data["report"]["statistics"]["new_facts"], 3)
        self.assertEqual(data["report"]["graph_checks"][0]["comparison"], "passed")
        self.assertTrue(any(f["negated"] for f in data["extraction"]["facts"]))

    def test_extract_stops_before_inference(self):
        response = self.client.post("/api/extract", json={"text": self.context.demo_text})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["report"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(response.json()["extraction"]["facts"]), 5)

    def test_reinfer_edited_json_without_llm(self):
        edited = json.loads(self.context.demo_json)
        edited["facts"] = [f for f in edited["facts"] if f["args"] != ["e3", "e4"]]
        response = self.client.post(
            "/api/infer",
            json={"text": self.context.demo_text, "extraction_json": json.dumps(edited)},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["report"]["statistics"]["new_facts"], 1)
        self.assertEqual(self.calls, [])

    def test_infer_rejects_invented_quotes(self):
        edited = json.loads(self.context.demo_json)
        edited["facts"][0]["evidence"] = ["Предложение, которого нет."]
        response = self.client.post(
            "/api/infer",
            json={"text": self.context.demo_text, "extraction_json": json.dumps(edited)},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_extraction")

    def test_invalid_model_json_keeps_output_and_usage_without_retry(self):
        calls = []

        async def broken(**kwargs):
            calls.append(kwargs)
            return LLMResponse("not json", "test-model", {"total_tokens": 123})

        client = TestClient(create_app(SETTINGS, broken, self.context))
        response = client.post("/api/analyze", json={"text": self.context.demo_text})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["raw_output"], "not json")
        self.assertEqual(response.json()["usage"]["total_tokens"], 123)
        self.assertEqual(len(calls), 1)

    def test_full_markdown_fence_is_accepted_but_prose_is_not(self):
        async def fenced(**kwargs):
            return LLMResponse(chr(96) * 3 + "json\n" + self.context.demo_json + "\n" + chr(96) * 3, "m", None)

        client = TestClient(create_app(SETTINGS, fenced, self.context))
        response = client.post("/api/analyze", json={"text": self.context.demo_text})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["raw_json"].startswith("{"))

    def test_model_error_is_structured(self):
        async def failed(**kwargs):
            raise LLMError("timeout", "Модель не ответила вовремя.")

        client = TestClient(create_app(SETTINGS, failed, self.context))
        response = client.post("/api/analyze", json={"text": self.context.demo_text})
        self.assertEqual(response.status_code, 504)
        self.assertNotIn(SETTINGS.api_key, response.text)

    def test_missing_key_can_still_run_demo(self):
        client = TestClient(create_app(
            LLMSettings("", "https://provider.invalid/v1", ""), context=self.context
        ))
        response = client.post("/api/analyze", json={"text": self.context.demo_text})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "not_configured")
        self.assertEqual(client.post("/api/demo").status_code, 200)

    def test_initial_contradiction_is_reported_before_inference(self):
        data = json.loads(self.context.demo_json)
        opposite = copy.deepcopy(data["facts"][0])
        opposite["negated"] = True
        data["facts"].append(opposite)
        response = self.client.post(
            "/api/infer",
            json={"text": self.context.demo_text, "extraction_json": json.dumps(data)},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "conflicting_facts")

    def test_new_contradiction_is_reported_with_both_proofs(self):
        data = json.loads(self.context.demo_json)
        quote = "Для изучения Теории множеств требуется Философия."
        data["facts"].append({
            "predicate": "requires", "args": ["e4", "e6"], "negated": False, "evidence": [quote],
        })
        response = self.client.post(
            "/api/infer",
            json={
                "text": self.context.demo_text + "\n" + quote,
                "extraction_json": json.dumps(data, ensure_ascii=False),
            },
        )
        self.assertEqual(response.status_code, 200)
        report = response.json()["report"]
        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(len(report["conflicts"]), 1)
        self.assertTrue(report["conflicts"][0]["positive_fact"])
        self.assertTrue(report["conflicts"][0]["negative_fact"])

    def test_cycle_is_visible_as_strict_order_warning(self):
        data = json.loads(self.context.demo_json)
        quote = "Для изучения Логики требуется ИИ."
        data["facts"].append({
            "predicate": "requires", "args": ["e2", "e1"], "negated": False, "evidence": [quote],
        })
        response = self.client.post(
            "/api/infer",
            json={
                "text": self.context.demo_text + "\n" + quote,
                "extraction_json": json.dumps(data, ensure_ascii=False),
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["report"]["status"], "needs_review")
        self.assertEqual(
            response.json()["report"]["graph_checks"][0]["strict_cycle_vertices"], ["e1", "e2"]
        )

    def test_bad_requests_do_not_reach_llm(self):
        for data in (
            {"text": ""},
            {"text": " \n "},
            {"text": "a" * (MAX_TEXT_CHARS + 1)},
            {"text": self.context.demo_text, "base_url": "https://other.invalid"},
        ):
            with self.subTest(data=str(data)[:80]):
                self.assertEqual(self.client.post("/api/analyze", json=data).status_code, 422)
        self.assertEqual(self.calls, [])


class ConcurrentRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_paid_operation_is_rejected_while_busy(self):
        context = load_context("basic")
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def slow(**kwargs):
            calls.append(kwargs)
            entered.set()
            await release.wait()
            return LLMResponse(context.demo_json, "m", None)

        app = create_app(SETTINGS, slow, context)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            first = asyncio.create_task(client.post("/api/analyze", json={"text": context.demo_text}))
            await entered.wait()
            second = await client.post("/api/analyze", json={"text": context.demo_text})
            self.assertEqual(second.status_code, 429)
            self.assertEqual(second.json()["error"]["code"], "busy")
            release.set()
            self.assertEqual((await first).status_code, 200)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
