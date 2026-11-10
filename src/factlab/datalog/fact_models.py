"""Модели извлечённых ground-литералов и строгая работа с JSON.

Python 3.10+, только стандартная библиотека. Вывода новых фактов здесь нет.
Проверка цитат подтверждает их наличие в тексте, но не истинность извлечения.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*", flags=re.ASCII)


class ValidationError(ValueError):
    """JSON или объект не соответствует контракту извлечения."""


class ConflictError(ValidationError):
    """Один атом одновременно утверждается и явно отрицается."""


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path}: требуется непустая строка.")
    return value


def _identifier(value: Any, path: str) -> str:
    _string(value, path)
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(
            f"{path}: требуется идентификатор вида [a-z][a-z0-9_]*; "
            "переменные, выражения и вложенные термы недопустимы."
        )
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ValidationError(f"{path}: требуется JSON boolean / Python bool.")
    return value


def _version(value: Any) -> None:
    if value != SCHEMA_VERSION or not isinstance(value, str):
        raise ValidationError(f"schema_version: поддерживается только {SCHEMA_VERSION!r}.")


def _tuple(value: Any, path: str) -> tuple:
    if not isinstance(value, tuple):
        raise ValidationError(f"{path}: в Python-модели требуется tuple.")
    return value


def _strings_tuple(value: Any, path: str, *, nonempty: bool = True) -> tuple[str, ...]:
    _tuple(value, path)
    if nonempty and not value:
        raise ValidationError(f"{path}: требуется хотя бы одна строка.")
    for index, item in enumerate(value):
        _string(item, f"{path}[{index}]")
    return value


def _deduplicate(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    label: str
    type: str
    mentions: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.id, "Entity.id")
        _string(self.label, "Entity.label")
        _string(self.type, "Entity.type")
        _strings_tuple(self.mentions, "Entity.mentions")
        object.__setattr__(self, "mentions", _deduplicate(self.mentions))


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    name: str
    arg_types: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        _identifier(self.name, "PredicateSpec.name")
        _strings_tuple(self.arg_types, "PredicateSpec.arg_types")
        _string(self.description, "PredicateSpec.description")

    @property
    def arity(self) -> int:
        return len(self.arg_types)


@dataclass(frozen=True, slots=True)
class ExtractionSchema:
    entity_types: tuple[str, ...]
    predicates: tuple[PredicateSpec, ...]
    allow_explicit_negation: bool
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        _strings_tuple(self.entity_types, "ExtractionSchema.entity_types", nonempty=False)
        _tuple(self.predicates, "ExtractionSchema.predicates")
        _boolean(self.allow_explicit_negation, "ExtractionSchema.allow_explicit_negation")
        if len(set(self.entity_types)) != len(self.entity_types):
            raise ValidationError("ExtractionSchema.entity_types: повторяющиеся типы.")
        names: set[str] = set()
        for spec in self.predicates:
            if not isinstance(spec, PredicateSpec):
                raise ValidationError("ExtractionSchema.predicates: ожидается PredicateSpec.")
            if spec.name in names:
                raise ValidationError(f"ExtractionSchema.predicates: повтор предиката {spec.name!r}.")
            names.add(spec.name)
            for arg_type in spec.arg_types:
                if arg_type not in self.entity_types:
                    raise ValidationError(
                        f"Предикат {spec.name!r}: неизвестный тип аргумента {arg_type!r}."
                    )


@dataclass(frozen=True, slots=True)
class GroundAtom:
    predicate: str
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.predicate, "GroundAtom.predicate")
        _tuple(self.args, "GroundAtom.args")
        if not self.args:
            raise ValidationError("GroundAtom.args: требуется хотя бы один аргумент.")
        for index, argument in enumerate(self.args):
            _identifier(argument, f"GroundAtom.args[{index}]")


@dataclass(frozen=True, slots=True)
class GroundLiteral:
    atom: GroundAtom
    negated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.atom, GroundAtom):
            raise ValidationError("GroundLiteral.atom: ожидается GroundAtom.")
        _boolean(self.negated, "GroundLiteral.negated")


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    literal: GroundLiteral
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.literal, GroundLiteral):
            raise ValidationError("ExtractedFact.literal: ожидается GroundLiteral.")
        _strings_tuple(self.evidence, "ExtractedFact.evidence")
        object.__setattr__(self, "evidence", _deduplicate(self.evidence))


@dataclass(frozen=True, slots=True)
class UnparsedFragment:
    text: str
    reason: str

    def __post_init__(self) -> None:
        _string(self.text, "UnparsedFragment.text")
        _string(self.reason, "UnparsedFragment.reason")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    entities: tuple[Entity, ...]
    facts: tuple[ExtractedFact, ...]
    unparsed: tuple[UnparsedFragment, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        _tuple(self.entities, "ExtractionResult.entities")
        _tuple(self.facts, "ExtractionResult.facts")
        _tuple(self.unparsed, "ExtractionResult.unparsed")
        entity_ids: set[str] = set()
        for entity in self.entities:
            if not isinstance(entity, Entity):
                raise ValidationError("ExtractionResult.entities: ожидается Entity.")
            if entity.id in entity_ids:
                raise ValidationError(f"Повторяющийся id сущности {entity.id!r}.")
            entity_ids.add(entity.id)
        # Словарь сохраняет порядок первого появления факта.
        merged: dict[GroundLiteral, ExtractedFact] = {}
        for fact in self.facts:
            if not isinstance(fact, ExtractedFact):
                raise ValidationError("ExtractionResult.facts: ожидается ExtractedFact.")
            for argument in fact.literal.atom.args:
                if argument not in entity_ids:
                    raise ValidationError(f"Неизвестный id сущности {argument!r} в факте.")
            opposite = GroundLiteral(fact.literal.atom, not fact.literal.negated)
            if opposite in merged:
                atom = fact.literal.atom
                label = f"{atom.predicate}({', '.join(atom.args)})"
                raise ConflictError(f"Противоречие: одновременно получены {label} и ¬{label}.")
            previous = merged.get(fact.literal)
            if previous is None:
                merged[fact.literal] = fact
            else:
                merged[fact.literal] = ExtractedFact(
                    fact.literal, previous.evidence + fact.evidence
                )
        for fragment in self.unparsed:
            if not isinstance(fragment, UnparsedFragment):
                raise ValidationError("ExtractionResult.unparsed: ожидается UnparsedFragment.")
        object.__setattr__(self, "facts", tuple(merged.values()))

    @property
    def positive_atoms(self) -> frozenset[GroundAtom]:
        return frozenset(f.literal.atom for f in self.facts if not f.literal.negated)

    @property
    def negative_atoms(self) -> frozenset[GroundAtom]:
        return frozenset(f.literal.atom for f in self.facts if f.literal.negated)


def _object(value: Any, fields: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: требуется JSON object.")
    missing = fields - value.keys()
    extra = value.keys() - fields
    if missing:
        raise ValidationError(f"{path}: отсутствуют поля {', '.join(sorted(missing))}.")
    if extra:
        raise ValidationError(f"{path}: неизвестные поля {', '.join(sorted(extra))}.")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{path}: требуется JSON array.")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"JSON: повторяющийся ключ {key!r}.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValidationError(f"JSON: нестандартная константа {value!r} недопустима.")


def _parse_json(json_text: str) -> Any:
    if not isinstance(json_text, str):
        raise ValidationError("Для чтения JSON требуется строка.")
    try:
        return json.loads(
            json_text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
    except json.JSONDecodeError as error:
        raise ValidationError(
            f"Невалидный JSON: {error.msg}; строка {error.lineno}, столбец {error.colno}."
        ) from error


def loads_schema(json_text: str) -> ExtractionSchema:
    """Прочитать и проверить фиксированный словарь предикатов из JSON-строки."""
    data = _object(
        _parse_json(json_text),
        {"schema_version", "entity_types", "predicates", "allow_explicit_negation"},
        "schema",
    )
    specs: list[PredicateSpec] = []
    for index, raw in enumerate(_array(data["predicates"], "schema.predicates")):
        path = f"schema.predicates[{index}]"
        item = _object(raw, {"name", "arg_types", "description"}, path)
        specs.append(
            PredicateSpec(
                item["name"], tuple(_array(item["arg_types"], f"{path}.arg_types")),
                item["description"],
            )
        )
    return ExtractionSchema(
        entity_types=tuple(_array(data["entity_types"], "schema.entity_types")),
        predicates=tuple(specs),
        allow_explicit_negation=data["allow_explicit_negation"],
        schema_version=data["schema_version"],
    )


def load_schema(path: str | Path) -> ExtractionSchema:
    """Прочитать схему из UTF-8 файла. BOM допустим только на уровне файла."""
    return loads_schema(Path(path).read_text(encoding="utf-8-sig"))


def _source_substrings(values: tuple[str, ...], source_text: str, path: str) -> None:
    for index, value in enumerate(values):
        if value not in source_text:
            raise ValidationError(f"{path}[{index}]: цитата отсутствует в исходном тексте: {value!r}.")


def loads_extraction(
    json_text: str, source_text: str, schema: ExtractionSchema
) -> ExtractionResult:
    """Проверить JSON, ссылки, типы, цитаты и получить неизменяемые объекты.

    Отрицание здесь только явное. Отсутствующий атом не считается отрицательным.
    Проверка не устанавливает, следует ли факт по смыслу из приложенной цитаты.
    """
    if not isinstance(source_text, str):
        raise ValidationError("source_text: требуется строка.")
    if not isinstance(schema, ExtractionSchema):
        raise ValidationError("schema: ожидается ExtractionSchema.")
    data = _object(
        _parse_json(json_text), {"schema_version", "entities", "facts", "unparsed"}, "extraction"
    )
    _version(data["schema_version"])
    entities: list[Entity] = []
    registry: dict[str, Entity] = {}
    for index, raw in enumerate(_array(data["entities"], "entities")):
        path = f"entities[{index}]"
        item = _object(raw, {"id", "label", "type", "mentions"}, path)
        entity = Entity(
            id=item["id"], label=item["label"], type=item["type"],
            mentions=tuple(_array(item["mentions"], f"{path}.mentions")),
        )
        if entity.id in registry:
            raise ValidationError(f"{path}.id: повторяющийся id сущности {entity.id!r}.")
        if entity.type not in schema.entity_types:
            raise ValidationError(f"{path}.type: неизвестный тип {entity.type!r}.")
        _source_substrings(entity.mentions, source_text, f"{path}.mentions")
        registry[entity.id] = entity
        entities.append(entity)

    predicates = {spec.name: spec for spec in schema.predicates}
    facts: list[ExtractedFact] = []
    for index, raw in enumerate(_array(data["facts"], "facts")):
        path = f"facts[{index}]"
        item = _object(raw, {"predicate", "args", "negated", "evidence"}, path)
        atom = GroundAtom(
            predicate=item["predicate"], args=tuple(_array(item["args"], f"{path}.args"))
        )
        literal = GroundLiteral(atom, item["negated"])
        fact = ExtractedFact(literal, tuple(_array(item["evidence"], f"{path}.evidence")))
        spec = predicates.get(atom.predicate)
        if spec is None:
            raise ValidationError(f"{path}.predicate: неизвестный предикат {atom.predicate!r}.")
        if len(atom.args) != spec.arity:
            raise ValidationError(
                f"{path}.args: для {atom.predicate!r} ожидается {spec.arity} аргументов, "
                f"получено {len(atom.args)}."
            )
        for position, (argument, expected_type) in enumerate(zip(atom.args, spec.arg_types)):
            entity = registry.get(argument)
            if entity is None:
                raise ValidationError(f"{path}.args[{position}]: неизвестный id {argument!r}.")
            if entity.type != expected_type:
                raise ValidationError(
                    f"{path}.args[{position}]: для {argument!r} ожидается тип {expected_type!r}, "
                    f"получен {entity.type!r}."
                )
        if literal.negated and not schema.allow_explicit_negation:
            raise ValidationError(f"{path}.negated: явное отрицание отключено в схеме.")
        _source_substrings(fact.evidence, source_text, f"{path}.evidence")
        facts.append(fact)

    unparsed: list[UnparsedFragment] = []
    for index, raw in enumerate(_array(data["unparsed"], "unparsed")):
        path = f"unparsed[{index}]"
        item = _object(raw, {"text", "reason"}, path)
        fragment = UnparsedFragment(item["text"], item["reason"])
        _source_substrings((fragment.text,), source_text, f"{path}.text")
        unparsed.append(fragment)
    return ExtractionResult(
        entities=tuple(entities), facts=tuple(facts), unparsed=tuple(unparsed),
        schema_version=data["schema_version"],
    )


def load_extraction(
    path: str | Path, source_text: str, schema: ExtractionSchema
) -> ExtractionResult:
    """Прочитать извлечение из UTF-8 файла и выполнить полную проверку контракта."""
    return loads_extraction(Path(path).read_text(encoding="utf-8-sig"), source_text, schema)


def dumps_extraction(result: ExtractionResult) -> str:
    """Сериализовать модели в исходный плоский JSON-контракт (без вывода)."""
    if not isinstance(result, ExtractionResult):
        raise ValidationError("result: ожидается ExtractionResult.")
    data = {
        "schema_version": result.schema_version,
        "entities": [
            {"id": e.id, "label": e.label, "type": e.type, "mentions": list(e.mentions)}
            for e in result.entities
        ],
        "facts": [
            {
                "predicate": f.literal.atom.predicate,
                "args": list(f.literal.atom.args),
                "negated": f.literal.negated,
                "evidence": list(f.evidence),
            }
            for f in result.facts
        ],
        "unparsed": [{"text": u.text, "reason": u.reason} for u in result.unparsed],
    }
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def dump_extraction(result: ExtractionResult, path: str | Path) -> None:
    """Сохранить модели в UTF-8 JSON; каталог должен существовать."""
    Path(path).write_text(dumps_extraction(result), encoding="utf-8")
