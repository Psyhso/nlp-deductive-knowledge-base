"""Network-free client tests using the real SDK and httpx.MockTransport."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

import httpx

from factlab.web.llm_client import LLMError, LLMSettings, SCHEMA_PLACEHOLDER, TEXT_PLACEHOLDER, extract_json


SCHEMA = json.dumps(
    {
        "schema_version": "1.0",
        "entity_types": ["course", "person"],
        "predicates": [{"name": "requires", "arg_types": ["course", "course"], "description": "requires"}],
        "allow_explicit_negation": True,
    },
    ensure_ascii=False,
)
TEMPLATE = f"Формат JSON:\n<schema_json>{SCHEMA_PLACEHOLDER}</schema_json>\n<text>{TEXT_PLACEHOLDER}</text>"


def answer(
    content: str | None = '{"schema_version":"1.0","entities":[],"facts":[],"unparsed":[]}',
    *,
    finish_reason: str = "stop",
    refusal: str | None = None,
    usage: dict | None = None,
) -> dict:
    result = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": refusal,
                    "reasoning_content": "PRIVATE REASONING MUST NOT BE RETURNED",
                },
            }
        ],
    }
    if usage is not None:
        result["usage"] = usage
    return result


class LLMClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = LLMSettings("test-secret-never-print", "https://provider.invalid/v1", "test-model")
        self.requests: list[httpx.Request] = []

    async def call(self, *, payload: dict | None = None, status: int = 200, settings=None, **kwargs):
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(status, json=answer() if payload is None else payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await extract_json(
                "Для ИИ нужна Логика.", SCHEMA, TEMPLATE, settings or self.settings,
                http_client=client, **kwargs,
            )
            self.assertFalse(client.is_closed, "Injected HTTP client must remain owned by its caller")
            return result

    def assert_safe(self, error: LLMError) -> None:
        self.assertNotIn(self.settings.api_key, str(error))
        self.assertNotIn("PROVIDER PRIVATE BODY", str(error))
        self.assertNotIn("provider.invalid", str(error))

    async def test_wire_request_one_call_and_final_content_only(self):
        result = await self.call()
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.url.path, "/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], f"Bearer {self.settings.api_key}")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_tokens"], 6000)
        self.assertNotIn("max_completion_tokens", body)
        self.assertNotIn("temperature", body)
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(len(body["messages"]), 2)
        self.assertIn(SCHEMA, body["messages"][1]["content"])
        self.assertIn("Для ИИ нужна Логика.", body["messages"][1]["content"])
        self.assertNotIn(SCHEMA_PLACEHOLDER, body["messages"][1]["content"])
        self.assertNotIn(TEXT_PLACEHOLDER, body["messages"][1]["content"])
        self.assertEqual(json.loads(result.raw_json)["schema_version"], "1.0")
        self.assertNotIn("PRIVATE REASONING", result.raw_json)
        self.assertIsNone(result.usage)

    async def test_usage_keeps_reasoning_and_cache_without_double_counting(self):
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 350,
            "total_tokens": 1350,
            "prompt_tokens_details": {"cached_tokens": 200},
            "completion_tokens_details": {"reasoning_tokens": 250},
            "private_provider_extension": "PRIVATE",
        }
        result = await self.call(payload=answer(usage=usage))
        self.assertEqual(result.usage["total_tokens"], 1350)
        self.assertEqual(result.usage["completion_tokens"], 350)
        self.assertEqual(result.usage["completion_tokens_details"]["reasoning_tokens"], 250)
        self.assertEqual(result.usage["prompt_tokens_details"]["cached_tokens"], 200)
        self.assertNotIn("private_provider_extension", result.usage)

    async def test_reasoning_and_completion_token_limit(self):
        settings = replace(self.settings, reasoning_effort="low", token_limit_param="max_completion_tokens", max_output_tokens=1234)
        await self.call(settings=settings)
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["max_completion_tokens"], 1234)
        self.assertNotIn("max_tokens", body)

    async def test_text_mode_sends_no_response_format(self):
        await self.call(settings=replace(self.settings, response_format="text"))
        self.assertNotIn("response_format", json.loads(self.requests[0].content))

    async def test_json_schema_is_closed_and_has_enums(self):
        await self.call(settings=replace(self.settings, response_format="json_schema"))
        response_format = json.loads(self.requests[0].content)["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]

        def verify_closed(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertFalse(node["additionalProperties"])
                    self.assertEqual(set(node["properties"]), set(node["required"]))
                for value in node.values():
                    verify_closed(value)
            elif isinstance(node, list):
                for value in node:
                    verify_closed(value)

        verify_closed(schema)
        self.assertEqual(set(schema["properties"]), {"schema_version", "entities", "facts", "unparsed"})
        entity = schema["properties"]["entities"]["items"]["properties"]
        fact = schema["properties"]["facts"]["items"]["properties"]
        self.assertEqual(entity["type"]["enum"], ["course", "person"])
        self.assertEqual(set(entity), {"id", "label", "type", "mentions"})
        self.assertEqual(fact["predicate"]["enum"], ["requires"])
        self.assertEqual(set(fact), {"predicate", "args", "negated", "evidence"})
        self.assertEqual(fact["args"], {"type": "array", "items": {"type": "string"}})

    async def test_model_override_works_when_default_model_is_empty(self):
        result = await self.call(settings=replace(self.settings, model=""), model_override="provider/new-model")
        self.assertEqual(result.model, "provider/new-model")
        self.assertEqual(json.loads(self.requests[0].content)["model"], "provider/new-model")

    async def test_missing_prompt_marker_prevents_request(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: self.fail("No request expected"))) as client:
            with self.assertRaises(LLMError) as raised:
                await extract_json("text", SCHEMA, "bad template", self.settings, http_client=client)
        self.assertEqual(raised.exception.code, "invalid_prompt")

    async def test_prompt_replacement_does_not_reinterpret_inserted_markers(self):
        captures = []

        def handler(request):
            captures.append(json.loads(request.content))
            return httpx.Response(200, json=answer())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await extract_json(TEXT_PLACEHOLDER + SCHEMA_PLACEHOLDER, SCHEMA, TEMPLATE, self.settings, http_client=client)
        prompt = captures[0]["messages"][1]["content"]
        self.assertTrue(prompt.endswith(f"<text>{TEXT_PLACEHOLDER}{SCHEMA_PLACEHOLDER}</text>"))

    async def test_no_parsing_or_hidden_second_call_for_invalid_json(self):
        result = await self.call(payload=answer("not JSON"))
        self.assertEqual(result.raw_json, "not JSON")
        self.assertEqual(len(self.requests), 1)

    async def test_provider_http_errors_are_sanitized_and_never_retried(self):
        codes = {401: "authentication_error", 403: "permission_error", 429: "rate_limit", 500: "provider_error", 404: "request_rejected"}
        for status, code in codes.items():
            with self.subTest(status=status):
                before = len(self.requests)
                with self.assertRaises(LLMError) as raised:
                    await self.call(payload={"error": {"message": f"PROVIDER PRIVATE BODY {self.settings.api_key}", "type": "server_error"}}, status=status)
                self.assertEqual(raised.exception.code, code)
                self.assert_safe(raised.exception)
                self.assertEqual(len(self.requests) - before, 1)

    async def test_timeout_is_bounded_and_not_retried(self):
        calls = []

        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout(f"PROVIDER PRIVATE BODY {self.settings.api_key}", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(LLMError) as raised:
                await extract_json("text", SCHEMA, TEMPLATE, self.settings, http_client=client)
        self.assertEqual(raised.exception.code, "timeout")
        self.assert_safe(raised.exception)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].extensions["timeout"]["read"], 90.0)

    async def test_connection_error_is_sanitized(self):
        def handler(request):
            raise httpx.ConnectError(f"PROVIDER PRIVATE BODY {self.settings.api_key}", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(LLMError) as raised:
                await extract_json("text", SCHEMA, TEMPLATE, self.settings, http_client=client)
        self.assertEqual(raised.exception.code, "connection_error")
        self.assert_safe(raised.exception)

    async def test_truncated_response_is_not_presented_as_valid_json(self):
        with self.assertRaises(LLMError) as raised:
            await self.call(payload=answer('{"entities": [', finish_reason="length", usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}))
        self.assertEqual(raised.exception.code, "truncated_response")
        self.assertEqual(raised.exception.usage["total_tokens"], 30)
        self.assertEqual(len(self.requests), 1)

    async def test_refusal_and_content_filter(self):
        for payload in (answer(None, refusal="PROVIDER PRIVATE BODY"), answer(None, finish_reason="content_filter")):
            with self.assertRaises(LLMError) as raised:
                await self.call(payload=payload)
            self.assertEqual(raised.exception.code, "refusal")
            self.assert_safe(raised.exception)

    async def test_empty_final_content_rejects_reasoning_only_response(self):
        for content in (None, "", "   "):
            with self.assertRaises(LLMError) as raised:
                await self.call(payload=answer(content))
            self.assertEqual(raised.exception.code, "empty_response")

    async def test_empty_choices(self):
        payload = answer()
        payload["choices"] = []
        with self.assertRaises(LLMError) as raised:
            await self.call(payload=payload)
        self.assertEqual(raised.exception.code, "empty_response")

    async def test_damaged_success_body_is_sanitized(self):
        responses = (
            httpx.Response(200, json={"error": {"message": f"PROVIDER PRIVATE BODY {self.settings.api_key}"}}),
            httpx.Response(200, json=None),
            httpx.Response(200, text=f"PROVIDER PRIVATE BODY {self.settings.api_key}"),
            httpx.Response(200, content=b"invalid-json", headers={"content-type": "application/json"}),
        )
        for response in responses:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _, response=response: response)) as client:
                with self.assertRaises(LLMError) as raised:
                    await extract_json("text", SCHEMA, TEMPLATE, self.settings, http_client=client)
            self.assertEqual(raised.exception.code, "provider_error")
            self.assert_safe(raised.exception)

    async def test_invalid_settings_fail_without_requests(self):
        cases = (
            {"api_key": ""}, {"model": ""}, {"response_format": "xml"},
            {"token_limit_param": "other"}, {"max_output_tokens": 0},
            {"max_output_tokens": True}, {"timeout_seconds": 0},
            {"timeout_seconds": float("inf")}, {"base_url": "not-a-url"},
            {"base_url": "https://secret:token@provider.invalid/v1"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(LLMError):
                    await self.call(settings=replace(self.settings, **changes))
        self.assertEqual(self.requests, [])

    async def test_invalid_output_schema_prevents_request(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: self.fail("No request expected"))) as client:
            with self.assertRaises(LLMError) as raised:
                await extract_json("text", "not-json", TEMPLATE, replace(self.settings, response_format="json_schema"), http_client=client)
        self.assertEqual(raised.exception.code, "invalid_schema")

    def test_settings_are_immutable_and_do_not_print_api_key(self):
        self.assertTrue(self.settings.ready)
        self.assertFalse(replace(self.settings, api_key="").ready)
        self.assertFalse(replace(self.settings, model="").ready)
        self.assertNotIn(self.settings.api_key, repr(self.settings))
        with self.assertRaises(FrozenInstanceError):
            self.settings.model = "changed"


if __name__ == "__main__":
    unittest.main()
