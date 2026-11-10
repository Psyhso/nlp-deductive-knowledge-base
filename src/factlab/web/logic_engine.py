"""Проверяемый JSON -> многосортная логика первого порядка -> Z3.

Области открыты: перечисленные имена не исчерпывают мир и без явного
неравенства могут обозначать один объект. Модель Z3 никогда не превращается
в извлечённые факты. Показываются только доказанные литералы об именах.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import json
import re
from time import monotonic
from typing import Any
import unicodedata

import z3


class LogicValidationError(ValueError):
    """Некорректный пакет формализации; никакой код из JSON не исполняется."""


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}", re.ASCII)
_ROOT_KEYS = {
    "version", "sorts", "predicates", "entities", "premises", "proposals",
    "queries", "unparsed",
}
MAX_JSON_CHARS = 200_000
MAX_FORMULA_DEPTH = 32
MAX_AST_NODES = 3_000
MAX_ARITY = 4
MAX_TIMEOUT_MS = 8_000
MAX_NAMED_ATOMS = 120
_COUNTS = {
    "sorts": 12, "predicates": 40, "entities": 60, "premises": 60,
    "proposals": 30, "queries": 20, "unparsed": 60,
}
_QUOTE_TYPOGRAPHY = str.maketrans({
    **dict.fromkeys('«»„“”‟', '"'),
    **dict.fromkeys("‘’‚‛", "'"),
    **dict.fromkeys("‐‑‒–—", "-"),
})


def _normalize_quote(text: str) -> str:
    """Сравнивать цитаты без различий регистра, пробелов и типографики.

    Слова, их порядок, отрицания, числа и прочая пунктуация сохраняются:
    это поиск подстроки, а не нечёткий поиск или проверка пересказа.
    """
    text = unicodedata.normalize("NFC", text).translate(_QUOTE_TYPOGRAPHY).casefold()
    return " ".join(text.split())


def _object(value: Any, keys: set[str], path: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise LogicValidationError(f"{path}: ожидаются ровно поля {sorted(keys)}.")
    return value


def _array(value: Any, path: str, limit: int, minimum: int = 0) -> list:
    if not isinstance(value, list) or not minimum <= len(value) <= limit:
        raise LogicValidationError(f"{path}: требуется массив длины {minimum}–{limit}.")
    return value


def _string(value: Any, path: str, limit: int = 2_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise LogicValidationError(f"{path}: требуется непустая строка до {limit} символов.")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise LogicValidationError(f"{path}: нужен id вида [A-Za-z_][A-Za-z0-9_] до 80 символов.")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise LogicValidationError(f"Повторяющийся ключ JSON: {key!r}.")
        result[key] = value
    return result


def _nonstandard_constant(value: str) -> None:
    raise LogicValidationError(f"Нестандартная константа JSON: {value}.")


class _Validator:
    def __init__(self, source: str | None, citation_warnings: list[dict] | None = None) -> None:
        self.normalized_source = _normalize_quote(source) if source is not None else None
        self.citation_warnings = citation_warnings
        self.sorts: dict[str, dict] = {}
        self.predicates: dict[str, dict] = {}
        self.entities: dict[str, dict] = {}
        self.nodes = 0

    def quote(self, value: Any, path: str) -> str:
        quote = _string(value, path, 30_000)
        if self.normalized_source is not None and _normalize_quote(quote) not in self.normalized_source:
            message = (
                f"{path}: цитата отсутствует в исходном тексте "
                "даже без учёта регистра, пробелов и типографики."
            )
            if self.citation_warnings is None:
                raise LogicValidationError(message)
            self.citation_warnings.append({
                "code": "citation_mismatch", "path": path, "quote": quote,
                "message": "Цитата не найдена в исходном тексте. Модель могла сократить или пересказать фразу.",
            })
        return quote

    def quotes(self, value: Any, path: str) -> list[str]:
        result = []
        for index, item in enumerate(_array(value, path, 20, 1)):
            quote = self.quote(item, f"{path}[{index}]")
            if quote in result:
                raise LogicValidationError(f"{path}: повторяющаяся цитата.")
            result.append(quote)
        return result

    def term(self, value: Any, variables: dict[str, str], path: str) -> tuple[dict, str]:
        if not isinstance(value, dict) or len(value) != 1:
            raise LogicValidationError(f"{path}: нужен терм {{var: id}} или {{entity: id}}.")
        if "var" in value:
            name = _identifier(value["var"], path + ".var")
            if name not in variables:
                raise LogicValidationError(f"{path}: свободная переменная {name!r}.")
            return {"var": name}, variables[name]
        if "entity" in value:
            name = _identifier(value["entity"], path + ".entity")
            if name not in self.entities:
                raise LogicValidationError(f"{path}: неизвестная сущность {name!r}.")
            return {"entity": name}, self.entities[name]["sort"]
        raise LogicValidationError(f"{path}: разрешены только var и entity.")

    def formula(
        self, value: Any, variables: dict[str, str], path: str, depth: int = 0,
    ) -> dict:
        self.nodes += 1
        if self.nodes > MAX_AST_NODES or depth > MAX_FORMULA_DEPTH:
            raise LogicValidationError("Превышен лимит числа узлов или глубины формул.")
        if not isinstance(value, dict) or not isinstance(value.get("op"), str):
            raise LogicValidationError(f"{path}: требуется объект формулы с op.")
        op = value["op"]
        if op == "atom":
            _object(value, {"op", "predicate", "args"}, path)
            name = _identifier(value["predicate"], path + ".predicate")
            if name not in self.predicates:
                raise LogicValidationError(f"{path}: неизвестный предикат {name!r}.")
            expected = self.predicates[name]["arg_sorts"]
            args = _array(value["args"], path + ".args", MAX_ARITY)
            if len(args) != len(expected):
                raise LogicValidationError(f"{path}: у {name} ожидается {len(expected)} аргументов.")
            terms = []
            for index, (argument, sort) in enumerate(zip(args, expected)):
                term, actual = self.term(argument, variables, f"{path}.args[{index}]")
                if actual != sort:
                    raise LogicValidationError(f"{path}.args[{index}]: ожидается сорт {sort!r}.")
                terms.append(term)
            return {"op": op, "predicate": name, "args": terms}
        if op in {"true", "false"}:
            _object(value, {"op"}, path)
            return {"op": op}
        if op == "not":
            _object(value, {"op", "arg"}, path)
            return {"op": op, "arg": self.formula(value["arg"], variables, path + ".arg", depth + 1)}
        if op in {"and", "or"}:
            _object(value, {"op", "args"}, path)
            return {
                "op": op,
                "args": [
                    self.formula(item, variables, f"{path}.args[{index}]", depth + 1)
                    for index, item in enumerate(_array(value["args"], path + ".args", 40))
                ],
            }
        if op in {"implies", "iff"}:
            _object(value, {"op", "left", "right"}, path)
            return {
                "op": op,
                "left": self.formula(value["left"], variables, path + ".left", depth + 1),
                "right": self.formula(value["right"], variables, path + ".right", depth + 1),
            }
        if op == "eq":
            _object(value, {"op", "left", "right"}, path)
            left, left_sort = self.term(value["left"], variables, path + ".left")
            right, right_sort = self.term(value["right"], variables, path + ".right")
            if left_sort != right_sort:
                raise LogicValidationError(f"{path}: сравниваются объекты разных сортов.")
            return {"op": op, "left": left, "right": right}
        if op in {"forall", "exists"}:
            _object(value, {"op", "vars", "body"}, path)
            scope = dict(variables)
            bindings = []
            for index, item in enumerate(_array(value["vars"], path + ".vars", 8, 1)):
                item_path = f"{path}.vars[{index}]"
                _object(item, {"id", "sort"}, item_path)
                name = _identifier(item["id"], item_path + ".id")
                sort = _identifier(item["sort"], item_path + ".sort")
                if sort not in self.sorts:
                    raise LogicValidationError(f"{item_path}: неизвестный сорт {sort!r}.")
                if name in scope:
                    raise LogicValidationError(f"{item_path}: переменная {name!r} уже связана в этой области.")
                scope[name] = sort
                bindings.append({"id": name, "sort": sort})
            return {
                "op": op, "vars": bindings,
                "body": self.formula(value["body"], scope, path + ".body", depth + 1),
            }
        raise LogicValidationError(f"{path}: неизвестная операция {op!r}.")

    def bundle(self, raw: str | dict) -> dict:
        if isinstance(raw, str):
            if len(raw) > MAX_JSON_CHARS:
                raise LogicValidationError("Пакет JSON слишком большой.")
            try:
                data = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonstandard_constant)
            except (json.JSONDecodeError, RecursionError) as error:
                raise LogicValidationError("Невалидный JSON формализации.") from error
        else:
            data = raw
            # Вызовы Python не обходят тот же лимит, который есть у JSON-строк.
            try:
                serialized = json.dumps(data, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError, RecursionError) as error:
                raise LogicValidationError("Пакет должен состоять только из допустимых значений JSON.") from error
            if len(serialized) > MAX_JSON_CHARS:
                raise LogicValidationError("Пакет JSON слишком большой.")
        _object(data, _ROOT_KEYS, "bundle")
        if data["version"] != "1.0":
            raise LogicValidationError("Поддерживается только version '1.0'.")
        arrays = {key: _array(data[key], key, limit) for key, limit in _COUNTS.items()}
        result: dict[str, Any] = {"version": "1.0"}

        for kind, registry, keys in (
            ("sorts", self.sorts, {"id", "label"}),
            ("predicates", self.predicates, {"id", "label", "arg_sorts"}),
            ("entities", self.entities, {"id", "label", "sort", "mentions"}),
        ):
            normalized = []
            for index, item in enumerate(arrays[kind]):
                path = f"{kind}[{index}]"
                _object(item, keys, path)
                name = _identifier(item["id"], path + ".id")
                if name in registry:
                    raise LogicValidationError(f"{path}: повторяющийся id {name!r}.")
                current = {"id": name, "label": _string(item["label"], path + ".label", 300)}
                if kind == "predicates":
                    sorts = []
                    for sort in _array(item["arg_sorts"], path + ".arg_sorts", MAX_ARITY):
                        sort = _identifier(sort, path + ".arg_sorts")
                        if sort not in self.sorts:
                            raise LogicValidationError(f"{path}: неизвестный сорт {sort!r}.")
                        sorts.append(sort)
                    current["arg_sorts"] = sorts
                elif kind == "entities":
                    sort = _identifier(item["sort"], path + ".sort")
                    if sort not in self.sorts:
                        raise LogicValidationError(f"{path}: неизвестный сорт {sort!r}.")
                    current["sort"] = sort
                    current["mentions"] = self.quotes(item["mentions"], path + ".mentions")
                registry[name] = current
                normalized.append(current)
            result[kind] = normalized

        # Предпосылки и предложения имеют общий namespace: их id используются в support.
        assertion_ids: set[str] = set()
        query_ids: set[str] = set()
        for kind, keys in (
            ("premises", {"id", "formula", "evidence", "explanation"}),
            ("proposals", {"id", "formula", "origin", "explanation"}),
            ("queries", {"id", "formula", "text"}),
        ):
            normalized = []
            for index, item in enumerate(arrays[kind]):
                path = f"{kind}[{index}]"
                _object(item, keys, path)
                name = _identifier(item["id"], path + ".id")
                seen = query_ids if kind == "queries" else assertion_ids
                if name in seen:
                    raise LogicValidationError(f"{path}: повторяющийся id {name!r}.")
                seen.add(name)
                current = {
                    "id": name,
                    "formula": self.formula(item["formula"], {}, path + ".formula"),
                }
                if kind == "premises":
                    current["evidence"] = self.quotes(item["evidence"], path + ".evidence")
                if kind == "proposals":
                    if not isinstance(item["origin"], str) or item["origin"] not in {"definition", "assumption"}:
                        raise LogicValidationError(f"{path}.origin: definition или assumption.")
                    current["origin"] = item["origin"]
                field = "text" if kind == "queries" else "explanation"
                current[field] = _string(item[field], path + "." + field)
                normalized.append(current)
            result[kind] = normalized

        result["unparsed"] = []
        for index, item in enumerate(arrays["unparsed"]):
            path = f"unparsed[{index}]"
            _object(item, {"text", "reason"}, path)
            text = self.quote(item["text"], path + ".text")
            result["unparsed"].append({"text": text, "reason": _string(item["reason"], path + ".reason")})
        return result


def validate_bundle(
    raw: str | dict, source: str, *, citation_warnings: list[dict] | None = None,
) -> dict:
    """Проверить структуру, замкнутость, сорта и цитаты с допуском форматирования.

    Если передан список citation_warnings, несовпадения цитат записываются
    в него вместо исключения. Структура и формулы проверяются строго всегда.
    """
    if not isinstance(source, str):
        raise LogicValidationError("source: требуется исходный текст.")
    return _Validator(source, citation_warnings).bundle(raw)


class _Formatter:
    def __init__(self, bundle: dict) -> None:
        self.predicates = {item["id"]: item["label"] for item in bundle["predicates"]}
        self.entities = {item["id"]: item["label"] for item in bundle["entities"]}
        self.sorts = {item["id"]: item["label"] for item in bundle["sorts"]}

    def term(self, term: dict) -> str:
        return term["var"] if "var" in term else self.entities[term["entity"]]

    def formula(self, formula: dict) -> str:
        op = formula["op"]
        if op == "atom":
            name = self.predicates[formula["predicate"]]
            return name + ("(" + ", ".join(self.term(arg) for arg in formula["args"]) + ")" if formula["args"] else "")
        if op in {"true", "false"}:
            return "⊤" if op == "true" else "⊥"
        if op == "not":
            child = formula["arg"]
            rendered = self.formula(child)
            return "¬" + rendered if child["op"] in {"atom", "eq", "true", "false", "not"} else "¬(" + rendered + ")"
        if op in {"and", "or"}:
            if not formula["args"]:
                return "⊤" if op == "and" else "⊥"
            separator = " ∧ " if op == "and" else " ∨ "
            return "(" + separator.join(self.formula(arg) for arg in formula["args"]) + ")"
        if op in {"implies", "iff"}:
            separator = " → " if op == "implies" else " ↔ "
            return "(" + self.formula(formula["left"]) + separator + self.formula(formula["right"]) + ")"
        if op == "eq":
            return "(" + self.term(formula["left"]) + " = " + self.term(formula["right"]) + ")"
        quantifier = "∀" if op == "forall" else "∃"
        variables = ", ".join(item["id"] + ":" + self.sorts[item["sort"]] for item in formula["vars"])
        return quantifier + " " + variables + " " + self.formula(formula["body"])


def format_formula(formula: dict, bundle: dict) -> str:
    """Читаемое представление без исполнения строк и потери скобок."""
    validator = _Validator(None)
    normalized = validator.bundle(bundle)
    validator.nodes = 0
    formula = validator.formula(formula, {}, "formula")
    return _Formatter(normalized).formula(formula)


class _Compiler:
    def __init__(self, bundle: dict, context: z3.Context) -> None:
        self.context = context
        self.sorts = {item["id"]: z3.DeclareSort("sort_" + item["id"], context) for item in bundle["sorts"]}
        self.entities = {
            item["id"]: z3.Const("entity_" + item["id"], self.sorts[item["sort"]])
            for item in bundle["entities"]
        }
        self.predicates = {}
        for item in bundle["predicates"]:
            name = "predicate_" + item["id"]
            self.predicates[item["id"]] = (
                z3.Function(name, *(self.sorts[name] for name in item["arg_sorts"]), z3.BoolSort(context))
                if item["arg_sorts"] else z3.Bool(name, context)
            )
        self.variable_count = 0

    def term(self, value: dict, variables: dict) -> Any:
        return variables[value["var"]] if "var" in value else self.entities[value["entity"]]

    def formula(self, value: dict, variables: dict | None = None) -> z3.BoolRef:
        variables = {} if variables is None else variables
        op = value["op"]
        if op == "atom":
            predicate = self.predicates[value["predicate"]]
            return predicate(*(self.term(arg, variables) for arg in value["args"])) if value["args"] else predicate
        if op in {"true", "false"}:
            return z3.BoolVal(op == "true", self.context)
        if op == "not":
            return z3.Not(self.formula(value["arg"], variables))
        if op in {"and", "or"}:
            args = [self.formula(arg, variables) for arg in value["args"]]
            if not args:
                return z3.BoolVal(op == "and", self.context)
            return z3.And(*args) if op == "and" else z3.Or(*args)
        if op == "implies":
            return z3.Implies(self.formula(value["left"], variables), self.formula(value["right"], variables))
        if op == "iff":
            return self.formula(value["left"], variables) == self.formula(value["right"], variables)
        if op == "eq":
            return self.term(value["left"], variables) == self.term(value["right"], variables)
        scope = dict(variables)
        bound = []
        for item in value["vars"]:
            self.variable_count += 1
            variable = z3.Const(f"bound_{self.variable_count}_{item['id']}", self.sorts[item["sort"]])
            scope[item["id"]] = variable
            bound.append(variable)
        body = self.formula(value["body"], scope)
        return z3.ForAll(bound, body) if op == "forall" else z3.Exists(bound, body)


def _key(formula: dict) -> str:
    return json.dumps(formula, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ground_literals(formula: dict):
    """Только непосредственно утверждённые литералы и члены конъюнкции."""
    if formula["op"] == "and":
        for child in formula["args"]:
            yield from _ground_literals(child)
    elif formula["op"] == "atom" and all("entity" in term for term in formula["args"]):
        yield formula, False
    elif formula["op"] == "not" and formula["arg"]["op"] == "atom":
        atom = formula["arg"]
        if all("entity" in term for term in atom["args"]):
            yield atom, True


def analyze_bundle(
    bundle: dict, accepted_proposal_ids: list[str] | None = None, *,
    timeout_ms: int = MAX_TIMEOUT_MS, max_named_atoms: int = MAX_NAMED_ATOMS,
) -> dict:
    """Проверить согласованность, запросы и оба знака именованных атомов.

    Каждая проверка следования — UNSAT(посылки ∧ ¬запрос). Unknown не
    превращается в ложность. При несовместных посылках следствия не выдаются.
    Лимит времени общий, включая последовательность запросов и атомов.
    """
    started = monotonic()
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= MAX_TIMEOUT_MS:
        raise LogicValidationError(f"timeout_ms: целое число 1–{MAX_TIMEOUT_MS}.")
    if type(max_named_atoms) is not int or not 0 <= max_named_atoms <= MAX_NAMED_ATOMS:
        raise LogicValidationError(f"max_named_atoms: целое число 0–{MAX_NAMED_ATOMS}.")
    deadline = started + timeout_ms / 1_000
    bundle = _Validator(None).bundle(bundle)
    accepted = [] if accepted_proposal_ids is None else accepted_proposal_ids
    if not isinstance(accepted, list) or any(not isinstance(name, str) for name in accepted):
        raise LogicValidationError("accepted_proposal_ids: требуется массив id предложений.")
    known_proposals = {item["id"] for item in bundle["proposals"]}
    if len(set(accepted)) != len(accepted) or any(name not in known_proposals for name in accepted):
        raise LogicValidationError("accepted_proposal_ids: неизвестное или повторяющееся предложение.")
    accepted_set = set(accepted)
    formatter = _Formatter(bundle)
    entity_ids: dict[str, list[str]] = defaultdict(list)
    for entity in bundle["entities"]:
        entity_ids[entity["sort"]].append(entity["id"])
    total_atoms = 0
    for predicate in bundle["predicates"]:
        count = 1
        for sort in predicate["arg_sorts"]:
            count *= len(entity_ids[sort])
        total_atoms += count

    report = {
        "consistency": "unknown", "consistency_support": [], "reason_unknown": None,
        "premises": [
            {"id": item["id"], "formula_text": formatter.formula(item["formula"]),
             "evidence": item["evidence"], "explanation": item["explanation"]}
            for item in bundle["premises"]
        ],
        "proposals": [
            {"id": item["id"], "formula_text": formatter.formula(item["formula"]),
             "origin": item["origin"], "explanation": item["explanation"],
             "accepted": item["id"] in accepted_set}
            for item in bundle["proposals"]
        ],
        "queries": [], "facts": [], "unparsed": bundle["unparsed"],
        "statistics": {
            "premises": len(bundle["premises"]), "accepted_proposals": len(accepted),
            "queries": len(bundle["queries"]), "named_atoms_total": total_atoms,
            "named_atoms_checked": 0, "named_atoms_unknown": 0,
            "solver_checks": 0, "elapsed_ms": 0, "truncated": False,
            "deadline_reached": False,
        },
        "limits": {"timeout_ms": timeout_ms, "max_named_atoms": max_named_atoms,
                   "open_domain": True, "unique_names": False},
        "explanation": (
            "Классическая логика с открытыми непустыми областями: именованные объекты "
            "не исчерпывают мир, разные имена не означают неравенство. Отсутствие "
            "доказательства не является отрицанием. Вывод ограничен временем и числом "
            "проверяемых именованных атомов; модельные объекты не выдаются за факты."
        ),
    }
    stats = report["statistics"]
    context = z3.Context()
    compiler = _Compiler(bundle, context)
    solver = z3.Solver(ctx=context)
    assertions = bundle["premises"] + [item for item in bundle["proposals"] if item["id"] in accepted_set]
    trackers = {}
    assertion_order = {item["id"]: index for index, item in enumerate(assertions)}
    for index, item in enumerate(assertions):
        tracker = z3.Bool(f"assertion_tracker_{index}", context)
        trackers[str(tracker)] = item["id"]
        solver.assert_and_track(compiler.formula(item["formula"]), tracker)

    def core() -> list[str]:
        ids = {trackers[str(item)] for item in solver.unsat_core() if str(item) in trackers}
        return sorted(ids, key=assertion_order.__getitem__)

    def check(extra: z3.BoolRef | None = None) -> tuple[str, list[str], str | None]:
        remaining = int((deadline - monotonic()) * 1_000)
        if remaining <= 0:
            stats["deadline_reached"] = True
            return "unknown", [], "Достигнут общий лимит времени анализа."
        solver.set(timeout=remaining)
        if extra is not None:
            solver.push()
            solver.add(extra)
        try:
            stats["solver_checks"] += 1
            result = solver.check()
            if result == z3.unsat:
                return "unsat", core(), None
            if result == z3.sat:
                return "sat", [], None
            return "unknown", [], solver.reason_unknown()
        finally:
            if extra is not None:
                solver.pop()

    consistency, support, reason = check()
    report["consistency"] = consistency
    report["consistency_support"] = support
    report["reason_unknown"] = reason
    if consistency != "sat":
        for item in bundle["queries"]:
            query = {
                "id": item["id"], "text": item["text"],
                "formula_text": formatter.formula(item["formula"]),
                "status": "inconsistent" if consistency == "unsat" else "unknown",
                "support": support if consistency == "unsat" else [],
            }
            if reason:
                query["reason_unknown"] = reason
            report["queries"].append(query)
        stats["truncated"] = bool(total_atoms or consistency == "unknown")
        stats["elapsed_ms"] = round((monotonic() - started) * 1_000)
        return report

    # Поддержка непосредственных литералов известна уже после SAT-проверки.
    direct: dict[str, tuple[dict, bool, list[str]]] = {}
    cache: dict[str, tuple[str, list[str], str | None]] = {}
    for item in bundle["premises"]:
        for atom, negated in _ground_literals(item["formula"]):
            key = _key(atom)
            if key not in direct:
                direct[key] = (atom, negated, [])
            if item["id"] not in direct[key][2]:
                direct[key][2].append(item["id"])
            cache[key] = ("refuted" if negated else "entailed", direct[key][2], None)

    # После SAT каждая активная посылка уже доказана сама собой. Это верно
    # и для сложных формул с кванторами: повторная проверка их отрицаний
    # может потратить весь лимит на поиск модели без новой информации.
    for item in assertions:
        formula = item["formula"]
        cache.setdefault(_key(formula), ("entailed", [item["id"]], None))
        cache.setdefault(_key({"op": "not", "arg": formula}), ("refuted", [item["id"]], None))
        if formula["op"] == "not":
            cache.setdefault(_key(formula["arg"]), ("refuted", [item["id"]], None))

    def classify(formula: dict) -> tuple[str, list[str], str | None]:
        key = _key(formula)
        if key in cache:
            return cache[key]
        expression = compiler.formula(formula)
        negative, neg_support, neg_reason = check(z3.Not(expression))
        if negative == "unsat":
            answer = ("entailed", neg_support, None)
        else:
            positive, pos_support, pos_reason = check(expression)
            if positive == "unsat":
                answer = ("refuted", pos_support, None)
            elif negative == positive == "sat":
                answer = ("undetermined", [], None)
            else:
                answer = ("unknown", [], neg_reason or pos_reason or "Решатель не установил результат.")
        cache[key] = answer
        return answer

    checked: set[str] = set()

    def append_fact(atom: dict, negated: bool, fact_support: list[str], origin: str) -> None:
        literal = {"op": "not", "arg": atom} if negated else atom
        report["facts"].append({
            "predicate": atom["predicate"], "args": [term["entity"] for term in atom["args"]],
            "negated": negated, "display": formatter.formula(literal),
            "origin": origin, "support": list(dict.fromkeys(fact_support)),
        })

    for key, (atom, negated, fact_support) in direct.items():
        if len(checked) >= max_named_atoms:
            break
        checked.add(key)
        append_fact(atom, negated, fact_support, "extracted")

    for item in bundle["queries"]:
        status, query_support, unknown_reason = classify(item["formula"])
        query = {"id": item["id"], "text": item["text"],
                 "formula_text": formatter.formula(item["formula"]),
                 "status": status, "support": query_support}
        if unknown_reason:
            query["reason_unknown"] = unknown_reason
        report["queries"].append(query)

    for predicate in bundle["predicates"]:
        if len(checked) >= max_named_atoms or monotonic() >= deadline:
            break
        domains = [entity_ids[sort] for sort in predicate["arg_sorts"]]
        for args in product(*domains):
            if len(checked) >= max_named_atoms or monotonic() >= deadline:
                break
            atom = {"op": "atom", "predicate": predicate["id"],
                    "args": [{"entity": name} for name in args]}
            key = _key(atom)
            if key in checked:
                continue
            checked.add(key)
            status, fact_support, _ = classify(atom)
            if status == "entailed":
                append_fact(atom, False, fact_support, "derived")
            elif status == "refuted":
                append_fact(atom, True, fact_support, "derived")
            elif status == "unknown":
                stats["named_atoms_unknown"] += 1
    stats["named_atoms_checked"] = len(checked)
    stats["deadline_reached"] = stats["deadline_reached"] or monotonic() >= deadline
    stats["truncated"] = len(checked) < total_atoms or stats["named_atoms_unknown"] > 0
    stats["elapsed_ms"] = round((monotonic() - started) * 1_000)
    return report
