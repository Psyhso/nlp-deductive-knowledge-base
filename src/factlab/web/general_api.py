"""Свободная формализация текста и отдельная локальная проверка следствий."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from typing import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from factlab.web.general_llm import formalize_text
from factlab.web.llm_client import LLMResponse, LLMSettings
from factlab.web.logic_engine import analyze_bundle, format_formula, validate_bundle


MAX_TEXT_CHARS = 12_000
MAX_BUNDLE_CHARS = 200_000
EXAMPLES = {
    'discovery': 'Автоматический поиск новых фактов',
    'legend': 'Легенда: Гегель и Фейербах',
    'siblings': 'Брат и сестра',
    'disjunction': 'Или и отрицание',
}


class GeneralRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator('text')
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('Введите исходный текст.')
        return value


class FormalizeRequest(GeneralRequest):
    model: str | None = Field(default=None, max_length=180)
    prompt_template: str | None = Field(default=None, min_length=1, max_length=40_000)


class AnalyzeRequest(GeneralRequest):
    bundle_json: str = Field(min_length=1, max_length=MAX_BUNDLE_CHARS)
    accepted_proposal_ids: list[str] = Field(default_factory=list, max_length=30)


class ExampleRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    example_id: str = Field(default='discovery', min_length=1, max_length=40)


class GeneralProblem(Exception):
    def __init__(self, code: str, message: str, status: int = 422,
                 raw_output: str | None = None, usage: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.raw_output, self.usage = raw_output, usage


def normalize_json(raw: str) -> str:
    raw = raw.strip()
    match = re.fullmatch(r'\x60{3}(?:json)?\s*\n(.*?)\n\x60{3}', raw, re.DOTALL)
    return match.group(1).strip() if match else raw


def validate_output(raw: str, source: str) -> dict:
    if len(raw) > MAX_BUNDLE_CHARS:
        raise GeneralProblem('bundle_too_large', 'Формализация превышает допустимый размер.')
    try:
        return validate_bundle(normalize_json(raw), source)
    except ValueError as error:
        raise GeneralProblem('invalid_formalization', str(error), raw_output=raw) from error


def preview_bundle(bundle: dict) -> dict:
    preview = {name: bundle[name] for name in ('sorts', 'predicates', 'entities', 'unparsed')}
    for collection in ('premises', 'proposals', 'queries'):
        preview[collection] = [
            {**{key: value for key, value in row.items() if key != 'formula'},
             'formula_text': format_formula(row['formula'], bundle)}
            for row in bundle[collection]
        ]
    return preview


def response_payload(bundle: dict, source: str, *, report: dict | None = None,
                     usage: dict | None = None, model: str = '', demo: bool = False) -> dict:
    return {
        'source_text': source, 'bundle': bundle,
        'raw_json': json.dumps(bundle, ensure_ascii=False, indent=2),
        'preview': preview_bundle(bundle), 'report': report,
        'usage': usage, 'model': model, 'demo': demo,
    }


def register_general_routes(
    application: FastAPI,
    root: Path,
    formalizer: Callable[..., Awaitable[LLMResponse]] | None = None,
) -> None:
    examples_root = root.parent / 'data' / 'general_examples'
    prompt_template = (root.parent / 'prompts' / 'formalization.txt').read_text(encoding='utf-8')
    application.state.formalizer = formalizer or formalize_text
    application.state.logic_gate = asyncio.Semaphore(1)

    def load_example(example_id: str) -> tuple[str, str]:
        if example_id not in EXAMPLES:
            raise GeneralProblem('unknown_example', 'Выберите существующий пример.', 404)
        folder = examples_root / example_id
        return ((folder / 'text.txt').read_text(encoding='utf-8'),
                (folder / 'bundle.json').read_text(encoding='utf-8'))

    async def analyze(bundle: dict, accepted: list[str]) -> dict:
        try:
            await asyncio.wait_for(application.state.logic_gate.acquire(), timeout=.1)
        except TimeoutError:
            raise GeneralProblem('logic_busy', 'Предыдущая логическая проверка ещё выполняется.', 429)
        try:
            try:
                return await run_in_threadpool(analyze_bundle, bundle, accepted)
            except ValueError as error:
                raise GeneralProblem('invalid_logic_input', str(error)) from error
        finally:
            application.state.logic_gate.release()

    @application.exception_handler(GeneralProblem)
    async def problem_handler(request: Request, error: GeneralProblem):
        item = {'code': error.code, 'message': error.message}
        if error.raw_output is not None:
            item['raw_output'] = error.raw_output
        return JSONResponse({'error': item, 'usage': error.usage}, status_code=error.status)

    @application.get('/api/general/bootstrap')
    async def bootstrap():
        settings: LLMSettings = application.state.settings
        examples = [
            {'id': example_id, 'title': title, 'text': load_example(example_id)[0]}
            for example_id, title in EXAMPLES.items()
        ]
        return {
            'configuration': {
                'ready': settings.ready, 'has_api_key': bool(settings.api_key.strip()),
                'model': settings.model, 'base_url': settings.base_url,
            },
            'defaults': {'text': examples[0]['text'], 'prompt_template': prompt_template},
            'examples': examples, 'limits': {'max_text_chars': MAX_TEXT_CHARS},
        }

    @application.post('/api/general/extract')
    async def extract(payload: FormalizeRequest):
        try:
            await asyncio.wait_for(application.state.gate.acquire(), timeout=.1)
        except TimeoutError:
            raise GeneralProblem('busy', 'Предыдущий запрос к модели ещё выполняется.', 429)
        try:
            result = await application.state.formalizer(
                text=payload.text, prompt_template=payload.prompt_template or prompt_template,
                settings=application.state.settings, model_override=payload.model,
            )
            try:
                bundle = validate_output(result.raw_json, payload.text)
            except GeneralProblem as error:
                error.status = 502
                error.usage = result.usage
                raise
            return response_payload(bundle, payload.text, usage=result.usage, model=result.model)
        finally:
            application.state.gate.release()

    @application.post('/api/general/analyze')
    async def analyze_request(payload: AnalyzeRequest):
        bundle = validate_output(payload.bundle_json, payload.text)
        report = await analyze(bundle, payload.accepted_proposal_ids)
        return response_payload(bundle, payload.text, report=report)

    @application.post('/api/general/demo')
    async def demo(payload: ExampleRequest | None = None):
        source, raw = load_example(payload.example_id if payload else 'discovery')
        bundle = validate_output(raw, source)
        report = await analyze(bundle, [])
        return response_payload(bundle, source, report=report, model='Демонстрация', demo=True)
