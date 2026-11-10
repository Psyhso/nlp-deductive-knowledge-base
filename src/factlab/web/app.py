"""Локальный FastAPI-интерфейс: LLM -> проверка фактов -> логический вывод."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Awaitable, Callable
from urllib.parse import urlsplit

from dotenv import dotenv_values
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from factlab.datalog.fact_models import (
    ConflictError, ExtractionResult, ExtractionSchema, ValidationError,
    dumps_extraction, load_schema, loads_extraction,
)
from factlab.datalog.inference import Rule, semi_naive
from factlab.datalog.rules_io import load_rules
from factlab.datalog.run_inference import build_report
from factlab.web.llm_client import LLMError, LLMResponse, LLMSettings, extract_json


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data" / "profiles"
MAX_TEXT_CHARS = 30_000
MAX_JSON_CHARS = 200_000
MAX_ENTITIES = 100
MAX_FACTS = 600
DEFAULT_PROFILE = "university"
PROFILE_ROOTS = {
    "university": DATA_ROOT / "university",
    "basic": DATA_ROOT / "basic",
    "family": DATA_ROOT / "family",
}


def profile_metadata(profile_id: str) -> dict:
    """Профиль выбирается из реестра, а не из переданного пользователем пути."""
    if profile_id not in PROFILE_ROOTS:
        raise ValueError("Неизвестный профиль знаний.")
    if profile_id == "basic":
        return {
            "id": "basic", "title": "Курсы — базовый пример",
            "description": "Одна транзитивность: косвенные зависимости между курсами.",
            "graph_checks": {"bfs_predicates": ["requires"], "strict_orders": ["requires"]},
            "rule_descriptions": {
                "requires_transitive": {
                    "title": "Косвенная предпосылка", "kind": "Транзитивность",
                    "description": "Если курсу нужен второй курс, а тому третий, первый зависит и от третьего.",
                },
            },
        }
    return json.loads((PROFILE_ROOTS[profile_id] / "metadata.json").read_text(encoding="utf-8"))


def profile_info(metadata: dict) -> dict:
    return {key: metadata[key] for key in ("id", "title", "description", "rule_descriptions")}


@dataclass(frozen=True)
class CoreContext:
    schema: ExtractionSchema
    schema_json: str
    rules: tuple[Rule, ...]
    rules_json: str
    prompt_template: str
    demo_text: str
    demo_json: str
    profile: dict
    bfs_predicates: tuple[str, ...]
    strict_orders: tuple[str, ...]
    acyclic_predicates: tuple[str, ...]


def load_context(profile_id: str = DEFAULT_PROFILE) -> CoreContext:
    metadata = profile_metadata(profile_id)
    root = PROFILE_ROOTS[profile_id]
    return CoreContext(
        schema=load_schema(root / "schema.json"),
        schema_json=(root / "schema.json").read_text(encoding="utf-8"),
        rules=load_rules(root / "rules.json"),
        rules_json=(root / "rules.json").read_text(encoding="utf-8"),
        prompt_template=(ROOT.parent / "prompts" / "extraction.txt").read_text(encoding="utf-8"),
        demo_text=(root / "example.txt").read_text(encoding="utf-8"),
        demo_json=(root / "extraction.json").read_text(encoding="utf-8"),
        profile=profile_info(metadata),
        bfs_predicates=tuple(metadata["graph_checks"]["bfs_predicates"]),
        strict_orders=tuple(metadata["graph_checks"]["strict_orders"]),
        acyclic_predicates=tuple(metadata["graph_checks"].get("acyclic_predicates", ())),
    )


def load_settings(env_file: Path | None = None) -> LLMSettings:
    """Среда запуска имеет приоритет над локальным .env. Секрет не печатается."""
    if env_file is None:
        env_file = Path(os.environ.get("FACTLAB_ENV_FILE", ".env")).resolve()
    values = {**dotenv_values(env_file), **os.environ}

    def value(name: str, default: str = "", fallback: str = "") -> str:
        selected = values.get(name)
        if selected is None and fallback:
            selected = values.get(fallback)
        return str(selected if selected is not None else default).strip()

    try:
        limit = int(value("LLM_MAX_OUTPUT_TOKENS", "6000"))
        timeout = float(value("LLM_TIMEOUT_SECONDS", "90"))
    except ValueError as error:
        raise ValueError("В .env неверно заданы лимит токенов или время ожидания.") from error
    if not 1 <= limit <= 32_000 or not 1 <= timeout <= 110:
        raise ValueError("Лимит токенов: 1–32000; ожидание ответа: 1–110 секунд.")
    endpoint = value("LLM_BASE_URL", "https://api.openai.com/v1", "OPENAI_BASE_URL")
    try:
        parsed = urlsplit(endpoint)
        valid = (
            parsed.scheme in {"https", "http"} and parsed.hostname
            and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment
        )
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("LLM_BASE_URL должен быть HTTP(S)-адресом без ключа, пароля и query.")
    response_format = value("LLM_RESPONSE_FORMAT", "json_object")
    token_param = value("LLM_TOKEN_LIMIT_PARAM", "max_tokens")
    if response_format not in {"json_object", "json_schema", "text"}:
        raise ValueError("LLM_RESPONSE_FORMAT: json_object, json_schema или text.")
    if token_param not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("LLM_TOKEN_LIMIT_PARAM: max_tokens или max_completion_tokens.")
    return LLMSettings(
        api_key=value("LLM_API_KEY", fallback="OPENAI_API_KEY"),
        base_url=endpoint.rstrip("/"),
        model=value("LLM_MODEL", fallback="OPENAI_MODEL"),
        max_output_tokens=limit,
        timeout_seconds=timeout,
        response_format=response_format,
        token_limit_param=token_param,
        reasoning_effort=value("LLM_REASONING_EFFORT") or None,
    )


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    model: str | None = Field(default=None, max_length=180)
    prompt_template: str | None = Field(default=None, min_length=1, max_length=40_000)
    profile: str | None = Field(default=None, min_length=1, max_length=50)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Введите текст для извлечения фактов.")
        return value

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str | None) -> str | None:
        return (value.strip() or None) if value is not None else None


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    extraction_json: str = Field(min_length=1, max_length=MAX_JSON_CHARS)
    profile: str | None = Field(default=None, min_length=1, max_length=50)


class DemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    profile: str | None = Field(default=None, min_length=1, max_length=50)


class AppProblem(Exception):
    def __init__(
        self, code: str, message: str, status: int = 422,
        raw_output: str | None = None, usage: dict | None = None,
    ):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.raw_output, self.usage = raw_output, usage


def normalize_json(raw: str) -> str:
    """Принимается только целый JSON или одна полная Markdown-ограда."""
    stripped = raw.strip()
    fenced = re.fullmatch(r"\x60{3}(?:json)?\s*\n(.*?)\n\x60{3}", stripped, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else stripped


def validate_extraction(raw: str, source: str, context: CoreContext) -> ExtractionResult:
    if len(raw) > MAX_JSON_CHARS:
        raise AppProblem("output_too_large", "Ответ слишком велик для этого учебного прототипа.")
    try:
        extraction = loads_extraction(normalize_json(raw), source, context.schema)
    except ConflictError as error:
        raise AppProblem(
            "conflicting_facts", str(error), 409, raw_output=raw,
        ) from error
    except ValidationError as error:
        raise AppProblem(
            "invalid_extraction", str(error), raw_output=raw,
        ) from error
    if len(extraction.entities) > MAX_ENTITIES or len(extraction.facts) > MAX_FACTS:
        raise AppProblem(
            "too_many_facts",
            f"Для одного запуска допускается до {MAX_ENTITIES} сущностей и {MAX_FACTS} исходных фактов.",
            raw_output=raw,
        )
    return extraction


def infer_report(extraction: ExtractionResult, context: CoreContext) -> dict:
    result = semi_naive(extraction, context.rules, context.schema)
    return build_report(
        extraction, result, context.rules, context.schema,
        bfs_predicates=context.bfs_predicates, strict_orders=context.strict_orders,
        acyclic_predicates=context.acyclic_predicates,
    )


def result_payload(
    extraction: ExtractionResult, source: str, report: dict | None,
    usage: dict | None = None, model: str = "", demo: bool = False,
    profile: dict | None = None,
) -> dict:
    normalized = dumps_extraction(extraction)
    return {
        "source_text": source,
        "extraction": json.loads(normalized),
        "raw_json": normalized,
        "report": report,
        "usage": usage,
        "model": model,
        "demo": demo,
        "profile": profile,
    }


def create_app(
    settings: LLMSettings | None = None,
    extractor: Callable[..., Awaitable[LLMResponse]] | None = None,
    context: CoreContext | None = None,
    formalizer: Callable[..., Awaitable[LLMResponse]] | None = None,
) -> FastAPI:
    application = FastAPI(title="Лаборатория фактов", version="2.0.0")
    application.state.settings = settings if settings is not None else load_settings()
    application.state.extractor = extractor or extract_json
    application.state.context = context or load_context()
    application.state.contexts = {application.state.context.profile["id"]: application.state.context}
    application.state.gate = asyncio.Semaphore(1)

    def context_for(profile_id: str | None) -> CoreContext:
        selected = profile_id if profile_id is not None else application.state.context.profile["id"]
        if selected not in PROFILE_ROOTS:
            raise AppProblem("unknown_profile", "Выберите существующий профиль знаний.", 404)
        if selected not in application.state.contexts:
            application.state.contexts[selected] = load_context(selected)
        return application.state.contexts[selected]

    @application.exception_handler(AppProblem)
    async def problem_handler(request: Request, error: AppProblem):
        item = {"code": error.code, "message": error.message}
        if error.raw_output is not None:
            item["raw_output"] = error.raw_output
        return JSONResponse(
            {"error": item, "usage": error.usage}, status_code=error.status
        )

    @application.exception_handler(LLMError)
    async def llm_error_handler(request: Request, error: LLMError):
        status = 503 if error.code == "not_configured" else 502
        if "timeout" in error.code:
            status = 504
        elif "rate" in error.code:
            status = 429
        item = {"code": error.code, "message": error.message}
        if error.raw_output is not None:
            item["raw_output"] = error.raw_output[:MAX_JSON_CHARS]
        return JSONResponse({"error": item, "usage": error.usage}, status_code=status)

    @application.exception_handler(RequestValidationError)
    async def request_error_handler(request: Request, error: RequestValidationError):
        return JSONResponse(
            {"error": {"code": "invalid_request", "message": "Проверьте поля запроса и допустимую длину текста."}},
            status_code=422,
        )

    @application.exception_handler(ValidationError)
    async def core_error_handler(request: Request, error: ValidationError):
        return JSONResponse(
            {"error": {"code": "invalid_rules_or_data", "message": str(error)}},
            status_code=422,
        )

    @application.middleware("http")
    async def local_headers(request: Request, call_next):
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > 1_000_000:
            return JSONResponse(
                {"error": {"code": "request_too_large", "message": "Запрос превышает 1 МБ."}},
                status_code=413,
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/") or request.url.path in {"/", "/general", "/profiles"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/")
    async def index():
        return FileResponse(ROOT / "static" / "general.html")

    @application.get("/general")
    async def general_page():
        return FileResponse(ROOT / "static" / "general.html")

    @application.get("/profiles")
    async def profiles_page():
        return FileResponse(ROOT / "static" / "index.html")

    @application.get("/api/bootstrap")
    async def bootstrap(profile: str | None = None):
        config = application.state.settings
        core = context_for(profile)
        return {
            "profiles": [
                {key: metadata[key] for key in ("id", "title", "description")}
                for metadata in (profile_metadata(name) for name in PROFILE_ROOTS)
            ],
            "profile": core.profile,
            "configuration": {
                "ready": config.ready, "model": config.model,
                "has_api_key": bool(config.api_key.strip()),
                "base_url": config.base_url, "response_format": config.response_format,
            },
            "defaults": {
                "text": core.demo_text, "schema_json": core.schema_json,
                "rules_json": core.rules_json, "prompt_template": core.prompt_template,
            },
            "limits": {"max_text_chars": MAX_TEXT_CHARS},
        }

    async def guarded_extract(payload: ExtractRequest, with_inference: bool):
        core = context_for(payload.profile)
        try:
            await asyncio.wait_for(application.state.gate.acquire(), timeout=.1)
        except TimeoutError:
            raise AppProblem("busy", "Предыдущий запрос ещё выполняется. Дождитесь результата.", 429)
        try:
            llm = await application.state.extractor(
                text=payload.text, schema_json=core.schema_json,
                prompt_template=payload.prompt_template or core.prompt_template,
                settings=application.state.settings, model_override=payload.model,
            )
            try:
                extraction = validate_extraction(llm.raw_json, payload.text, core)
            except AppProblem as error:
                error.usage = llm.usage
                if error.status == 422:
                    error.status = 502
                raise
            report = await run_in_threadpool(infer_report, extraction, core) if with_inference else None
            return result_payload(extraction, payload.text, report, llm.usage, llm.model, profile=core.profile)
        finally:
            application.state.gate.release()

    @application.post("/api/extract")
    async def extract(payload: ExtractRequest):
        return await guarded_extract(payload, False)

    @application.post("/api/analyze")
    async def analyze(payload: ExtractRequest):
        return await guarded_extract(payload, True)

    @application.post("/api/infer")
    async def infer(payload: InferRequest):
        core = context_for(payload.profile)
        extraction = validate_extraction(payload.extraction_json, payload.text, core)
        report = await run_in_threadpool(infer_report, extraction, core)
        return result_payload(extraction, payload.text, report, profile=core.profile)

    @application.post("/api/demo")
    async def demo(payload: DemoRequest | None = None):
        core = context_for(payload.profile if payload else None)
        extraction = validate_extraction(core.demo_json, core.demo_text, core)
        report = await run_in_threadpool(infer_report, extraction, core)
        return result_payload(extraction, core.demo_text, report, model="Демонстрация", demo=True, profile=core.profile)

    from factlab.web.general_api import register_general_routes
    register_general_routes(application, ROOT, formalizer)
    application.mount("/static", StaticFiles(directory=ROOT / "static", check_dir=False), name="static")
    return application


app = create_app()
