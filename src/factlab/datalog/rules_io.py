"""Строгий формат JSON для заранее заданных правил, без исполнения строк кода."""

from __future__ import annotations

import json
from pathlib import Path

from factlab.datalog.fact_models import ValidationError
from factlab.datalog.inference import LiteralPattern, Rule, Var


def _object(value: object, keys: set[str], path: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValidationError(f"{path}: ожидаются ровно поля {sorted(keys)}.")
    return value


def _array(value: object, path: str) -> list:
    if not isinstance(value, list):
        raise ValidationError(f"{path}: ожидается массив JSON.")
    return value


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Повторяющийся ключ JSON: {key!r}.")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValidationError(f"Нестандартная константа JSON: {value!r}.")


def _pattern(value: object, path: str) -> LiteralPattern:
    data = _object(value, {"predicate", "args", "negated"}, path)
    args = []
    for i, term in enumerate(_array(data["args"], f"{path}.args")):
        if isinstance(term, dict):
            variable = _object(term, {"var"}, f"{path}.args[{i}]")
            args.append(Var(variable["var"]))
        elif isinstance(term, str):
            args.append(term)
        else:
            raise ValidationError(f"{path}.args[{i}]: нужен id или объект {{var: имя}}.")
    return LiteralPattern(data["predicate"], tuple(args), data["negated"])


def loads_rules(json_text: str) -> tuple[Rule, ...]:
    """Читает структуру; соответствие схеме и реестру проверяет движок."""
    if not isinstance(json_text, str):
        raise ValidationError("Для загрузки правил нужна JSON-строка.")
    try:
        data = json.loads(json_text, object_pairs_hook=_pairs, parse_constant=_constant)
    except json.JSONDecodeError as error:
        raise ValidationError(
            f"Невалидный JSON правил: строка {error.lineno}, столбец {error.colno}."
        ) from error
    data = _object(data, {"schema_version", "rules"}, "rules")
    if data["schema_version"] != "1.0":
        raise ValidationError("Поддерживается только schema_version '1.0'.")
    result = []
    for i, raw in enumerate(_array(data["rules"], "rules.rules")):
        path = f"rules.rules[{i}]"
        rule = _object(raw, {"id", "body", "head"}, path)
        result.append(
            Rule(
                id=rule["id"],
                body=tuple(
                    _pattern(p, f"{path}.body[{j}]")
                    for j, p in enumerate(_array(rule["body"], f"{path}.body"))
                ),
                head=_pattern(rule["head"], f"{path}.head"),
            )
        )
    return tuple(result)


def load_rules(path: str | Path) -> tuple[Rule, ...]:
    return loads_rules(Path(path).read_text(encoding="utf-8-sig"))
