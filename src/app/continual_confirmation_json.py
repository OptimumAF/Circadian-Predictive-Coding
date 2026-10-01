"""Strict finite JSON and canonical fingerprint schema helpers.

Inputs are decoded JSON only. Outputs are checked values or useful failures;
no model, dataset, training, score, tensor reconstruction or IO belongs here.
"""

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Any

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianNetworkSnapshot,
    ReplayRetentionBudget,
    ReplaySnapshot,
)
from src.core.controlled_parent_selection import ParentSelectionDecision, ParentSelectionSettings
from src.core.neuron_adaptation import NeuronLineageSnapshot
from src.core.replay_retention import ReplayRetentionPolicy


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_DATACLASSES = {
    f"{kind.__module__}.{kind.__qualname__}": {field.name for field in fields(kind)}
    for kind in (
        CircadianConfig,
        CircadianNetworkSnapshot,
        ReplayRetentionBudget,
        ReplaySnapshot,
        ParentSelectionDecision,
        ParentSelectionSettings,
        NeuronLineageSnapshot,
        ReplayRetentionPolicy,
    )
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"confirmation JSON {message}")


def object_fields(value: Any, names: set[str], context: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == names, f"{context} fields differ")
    return value


def integer(value: Any, context: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{context} integer differs")
    return value


def hash_value(value: Any, context: str) -> str:
    require(type(value) is str and _HASH.fullmatch(value) is not None, f"{context} hash differs")
    return value


def same_json(actual: Any, expected: Any, context: str) -> None:
    require(type(actual) is type(expected), f"{context} value type differs")
    if type(expected) is dict:
        object_fields(actual, set(expected), context)
        for key, value in expected.items():
            same_json(actual[key], value, f"{context}/{key}")
    elif type(expected) is list:
        require(len(actual) == len(expected), f"{context} length differs")
        for left, right in zip(actual, expected, strict=True):
            same_json(left, right, context)
    else:
        require(actual == expected, f"{context} value differs")


def finite_json(value: Any, depth: int = 0) -> None:
    require(depth <= 40, "nesting limit exceeded")
    if type(value) is dict:
        for key, item in value.items():
            require(type(key) is str, "object key is not a string")
            require(
                key
                not in {
                    "development",
                    "a_after_a",
                    "a_after_b",
                    "b_after_b",
                    "final_mean_task_accuracy",
                    "signed_forgetting_a",
                    "retention_ratio_a",
                },
                "scored field in unscored facts",
            )
            finite_json(item, depth + 1)
    elif type(value) is list:
        for item in value:
            finite_json(item, depth + 1)
    elif type(value) is float:
        require(isfinite(value), "nonfinite number")
    else:
        require(value is None or type(value) in (str, int, bool), "unsupported JSON value")


def canonical_mapping(value: Any) -> dict[str, Any]:
    rows = object_fields(value, {"dict"}, "canonical mapping")["dict"]
    require(type(rows) is list, "canonical mapping rows differ")
    result: dict[str, Any] = {}
    for row in rows:
        require(
            type(row) is list and len(row) == 2 and type(row[0]) is str,
            "canonical mapping entry differs",
        )
        require(row[0] not in result, "duplicate canonical mapping key")
        result[row[0]] = row[1]
    require(list(result) == sorted(result), "canonical mapping order differs")
    return result


def canonical_dataclass(value: Any, class_name: str) -> dict[str, Any]:
    row = object_fields(value, {"dataclass", "fields"}, "canonical dataclass")
    require(
        row["dataclass"] == class_name and class_name in _DATACLASSES,
        "canonical dataclass type differs",
    )
    return object_fields(row["fields"], _DATACLASSES[class_name], "canonical dataclass members")


def canonical_array(value: Any, shape: tuple[int, ...], dtype: str = "<f8") -> None:
    row = object_fields(value, {"array_dtype", "shape", "bytes_sha256"}, "canonical array")
    same_json(row["shape"], list(shape), "canonical array shape")
    require(row["array_dtype"] == dtype, "canonical array dtype differs")
    hash_value(row["bytes_sha256"], "canonical array bytes")


def canonical_value(value: Any) -> None:
    if value is None or type(value) in (str, bool, int, float):
        finite_json(value)
        return
    require(type(value) is dict, "canonical node differs")
    if "array_dtype" in value:
        row = object_fields(value, {"array_dtype", "shape", "bytes_sha256"}, "canonical array")
        require(
            row["array_dtype"] in {"<f8", "<i8", "<i4", "|b1"} and type(row["shape"]) is list,
            "canonical array dtype/shape differs",
        )
        for dimension in row["shape"]:
            integer(dimension, "canonical dimension")
        hash_value(row["bytes_sha256"], "canonical array bytes")
    elif "dict" in value:
        for item in canonical_mapping(value).values():
            canonical_value(item)
    elif "dataclass" in value:
        class_name = value["dataclass"]
        require(
            type(class_name) is str and class_name in _DATACLASSES, "unknown canonical dataclass"
        )
        for item in canonical_dataclass(value, class_name).values():
            canonical_value(item)
    elif "generator" in value:
        object_fields(value, {"generator"}, "canonical generator")
        rng_state(value)
    elif "numpy_scalar" in value:
        row = object_fields(value, {"numpy_scalar", "value"}, "canonical scalar")
        require(
            row["numpy_scalar"] in {"<f8", "<i8", "<i4", "|b1"}, "canonical scalar dtype differs"
        )
        expected_type = (
            float if row["numpy_scalar"] == "<f8" else bool if row["numpy_scalar"] == "|b1" else int
        )
        require(type(row["value"]) is expected_type, "canonical scalar value type differs")
        finite_json(row["value"])
    else:
        tags = set(value) & {"list", "tuple", "set", "frozenset", "deque"}
        require(len(tags) == 1, "unknown canonical container")
        tag = next(iter(tags))
        object_fields(value, {tag, "maxlen"} if tag == "deque" else {tag}, "canonical container")
        require(type(value[tag]) is list, "canonical container rows differ")
        if tag == "deque" and value["maxlen"] is not None:
            integer(value["maxlen"], "canonical deque maxlen")
        if tag in {"set", "frozenset"}:
            require(
                all(type(item) is str for item in value[tag])
                and value[tag] == sorted(set(value[tag])),
                "canonical set order/uniqueness differs",
            )
        for item in value[tag]:
            canonical_value(item)


def rng_state(value: Any) -> dict[str, Any]:
    mapping = canonical_mapping(object_fields(value, {"generator"}, "generator")["generator"])
    object_fields(mapping, {"bit_generator", "state", "has_uint32", "uinteger"}, "PCG64 state")
    nested = canonical_mapping(mapping["state"])
    object_fields(nested, {"state", "inc"}, "PCG64 body")
    require(mapping["bit_generator"] == "PCG64", "RNG kind differs")
    for item in nested.values():
        require(integer(item, "PCG64 value") < 2**128, "PCG64 value overflow")
    require(
        type(mapping["has_uint32"]) is int and mapping["has_uint32"] in (0, 1),
        "PCG64 cached flag differs",
    )
    require(
        integer(mapping["uinteger"], "PCG64 cached value") < 2**32, "PCG64 cached value overflow"
    )
    return {**mapping, "state": nested}


def state_digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
