"""Whole-wire admission and bounded data-only DAG materialization.

Array encoding/materialization have explicit hooks after complete size/schema
admission. Fixed data constructors and PCG64 state assignment grant no native
model, callback, live-owner, filesystem or restore authority.
"""

import base64
from collections import deque
from dataclasses import fields
from math import prod
import re
from typing import Any

import numpy as np

from src.adapters.coupled_composite_graph import RECORD_TYPES, scalar
from src.adapters.coupled_composite_capture import _CoupledGraphPreflight
from src.adapters.numpy_checkpoint_frames import exact_object
from src.core.coupled_energy import CoupledParameters
from src.core.inbox_codec_policy import NUMERIC_DTYPES
from src.core.managed_composite_state import CompositeCaptureLimits, SourceRecord

_DTYPES = {name: np.dtype(name) for name in (*NUMERIC_DTYPES, "|b1")}
_BASE64 = re.compile(r"[A-Za-z0-9+/]*={0,2}\Z")
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_SEQUENCES = frozenset({"tuple", "list", "set", "frozenset", "deque"})


def encode_array(value: np.ndarray, node: dict[str, Any]) -> None:
    """Sole raw-array encoding hook; caller has admitted exact complete wire size."""
    node["data"] = base64.b64encode(value.tobytes(order=node["order"])).decode("ascii")


def decode_array(node: dict[str, Any]) -> np.ndarray:
    """Sole array allocation hook; caller has admitted every node and frame."""
    raw = base64.b64decode(node["data"], validate=True)
    if base64.b64encode(raw).decode("ascii") != node["data"]:
        raise ValueError("noncanonical composite array bytes")
    result = (
        np.frombuffer(raw, dtype=_DTYPES[node["dtype"]])
        .reshape(node["shape"], order=node["order"])
        .copy(order=node["order"])
    )
    if not np.isfinite(result).all():
        raise ValueError("nonfinite decoded composite array")
    result.flags.writeable = node["writeable"]
    return result


def _array_size(node: dict[str, Any], limits: CompositeCaptureLimits, placeholders: bool) -> int:
    exact_object(node, {"tag", "dtype", "shape", "order", "writeable", "data"})
    dtype, shape = node["dtype"], node["shape"]
    if type(dtype) is not str or dtype not in _DTYPES:
        raise ValueError("unsupported exact composite array dtype")
    if (
        type(shape) is not list
        or len(shape) > 2
        or any(type(size) is not int or not 0 <= size <= limits.max_dimension for size in shape)
    ):
        raise ValueError("composite array shape exceeds rank/dimension bound")
    if (
        type(node["order"]) is not str
        or node["order"] not in ("C", "F")
        or type(node["writeable"]) is not bool
    ):
        raise ValueError("invalid exact composite array storage flags")
    if node["order"] == "F" and (len(shape) < 2 or any(size <= 1 for size in shape)):
        raise ValueError("ambiguous contiguous composite array requires C order")
    size = prod(shape) * _DTYPES[dtype].itemsize
    text = node["data"]
    if placeholders:
        if type(text) is not str or text != "":
            raise ValueError("encoder array placeholders must have empty data")
        return size
    if (
        type(text) is not str
        or len(text) != 4 * ((size + 2) // 3)
        or _BASE64.fullmatch(text) is None
    ):
        raise ValueError("composite base64 length/alphabet differs from array bytes")
    padding = (3 - size % 3) % 3
    actual = 2 if text.endswith("==") else 1 if text.endswith("=") else 0
    if padding != actual or (
        padding and _ALPHABET.find(text[-padding - 1]) & (15 if padding == 2 else 3)
    ):
        raise ValueError("noncanonical composite base64 padding")
    return size


def preflight_wire(
    root: Any,
    nodes: Any,
    limits: CompositeCaptureLimits,
    max_layers: int,
    *,
    placeholders: bool = False,
) -> None:
    """Validate the complete bounded DAG before invoking decode_array anywhere."""
    if type(nodes) is not list or not nodes or len(nodes) > limits.max_nodes:
        raise ValueError("composite wire node inventory exceeds bound")
    edges: list[list[int]] = [[] for _ in nodes]
    scalar_leaves: set[int] = set()
    tokens = 0

    def token(value: Any, owner: int | None = None) -> int | None:
        nonlocal tokens
        tokens += 1
        if tokens > limits.max_nodes or type(value) is not dict:
            raise ValueError("composite wire token/edge inventory exceeds bound")
        if value.keys() == {"value"}:
            scalar(value["value"], limits)
            if owner is not None:
                scalar_leaves.add(owner)
            return None
        exact_object(value, {"ref"})
        index = value["ref"]
        if type(index) is not int or not 0 <= index < len(nodes):
            raise ValueError("composite reference requires an in-range exact node ID")
        if owner is not None:
            edges[owner].append(index)
        return index

    root_index = token(root)
    if root_index is None:
        raise ValueError("complete composite root must reference a record")
    root_node = nodes[root_index]
    if (
        type(root_node) is not dict
        or root_node.get("tag") != "record"
        or root_node.get("type") != "src.core.managed_composite_state.ManagedCompositeState"
    ):
        raise ValueError("complete composite root requires exact ManagedCompositeState")
    array_bytes = 0
    # Keep the fixed tag inventory in one loop so every node is admitted before
    # resolving any edge or allocating an array. Materialization stays separate.
    for index, node in enumerate(nodes):
        if type(node) is not dict or type(node.get("tag")) is not str:
            raise ValueError("composite node requires an exact fixed tag")
        tag = node["tag"]
        if tag == "array":
            array_bytes += _array_size(node, limits, placeholders)
            if array_bytes > limits.max_array_bytes:
                raise ValueError("aggregate unique composite arrays exceed byte bound")
        elif tag == "record":
            exact_object(node, {"tag", "type", "fields"})
            name, pairs = node["type"], node["fields"]
            if type(name) is not str or name not in RECORD_TYPES:
                raise ValueError("unknown composite data record type")
            expected = tuple(field.name for field in fields(RECORD_TYPES[name]))
            if type(pairs) is not list or len(pairs) != len(expected):
                raise ValueError("incomplete composite record fields")
            for pair, field in zip(pairs, expected):
                if (
                    type(pair) is not list
                    or len(pair) != 2
                    or type(pair[0]) is not str
                    or pair[0] != field
                ):
                    raise ValueError("composite record fields differ from ordered schema")
                token(pair[1], index)
            if RECORD_TYPES[name] is CoupledParameters:
                ref = pairs[0][1]
                if type(ref) is not dict or ref.keys() != {"ref"}:
                    raise ValueError("coupled hidden weights require a tuple node")
                weights = nodes[ref["ref"]]
                if (
                    type(weights) is not dict
                    or weights.get("tag") != "tuple"
                    or type(weights.get("items")) is not list
                    or not 1 <= len(weights["items"]) <= max_layers
                ):
                    raise ValueError("coupled native depth exceeds codec max_layers")
            if RECORD_TYPES[name] is SourceRecord:
                _source_schema(pairs, nodes)
        elif tag in _SEQUENCES or tag == "dict":
            exact_object(node, {"tag", "items", "maxlen"} if tag == "deque" else {"tag", "items"})
            items = node["items"]
            if type(items) is not list or len(items) > limits.max_nodes:
                raise ValueError("composite collection exceeds bound")
            if tag == "deque" and (
                node["maxlen"] is not None
                and (
                    type(node["maxlen"]) is not int
                    or not 0 <= node["maxlen"] < 2**63
                    or len(items) > node["maxlen"]
                )
            ):
                raise ValueError("invalid composite deque maxlen")
            for item in items:
                if tag == "dict":
                    if type(item) is not list or len(item) != 2:
                        raise ValueError("composite dictionary requires token pairs")
                    token(item[0], index)
                    token(item[1], index)
                else:
                    token(item, index)
        elif tag == "pcg64":
            exact_object(node, {"tag", "state"})
            token(node["state"], index)
        else:
            raise ValueError("unknown composite node tag")
    active: set[int] = set()
    heights: dict[int, int] = {}

    def height(index: int) -> int:
        if index in active:
            raise ValueError("cyclic composite wire graph")
        if index in heights:
            return heights[index]
        active.add(index)
        if len(active) - 1 > limits.max_depth:
            raise ValueError("composite wire graph exceeds depth bound")
        result = max(
            (1 + height(child) for child in edges[index]),
            default=1 if index in scalar_leaves else 0,
        )
        if result > limits.max_depth:
            raise ValueError("composite wire graph exceeds depth bound")
        active.remove(index)
        heights[index] = result
        return result

    height(root_index)
    if len(heights) != len(nodes):
        raise ValueError("unreachable composite wire nodes")


def _source_schema(pairs: list[Any], nodes: list[Any]) -> None:
    """Admit dynamic SourceRecord field names before any array allocation."""
    kind_token, fields_token = pairs[0][1], pairs[1][1]
    if type(kind_token) is not dict or kind_token.keys() != {"value"}:
        raise ValueError("projected source kind requires an exact scalar")
    kind = kind_token["value"]
    if type(kind) is not str or kind not in _CoupledGraphPreflight._source_schemas:
        raise ValueError("unknown projected composite source kind")

    def items(value: Any) -> list[Any]:
        if (
            type(value) is not dict
            or value.keys() != {"ref"}
            or type(value["ref"]) is not int
            or not 0 <= value["ref"] < len(nodes)
        ):
            raise ValueError("projected source fields require exact tuple references")
        node = nodes[value["ref"]]
        if (
            type(node) is not dict
            or node.keys() != {"tag", "items"}
            or node["tag"] != "tuple"
            or type(node["items"]) is not list
        ):
            raise ValueError("projected source fields require tuple nodes")
        return node["items"]

    names = []
    for entry in items(fields_token):
        pair = items(entry)
        if (
            len(pair) != 2
            or type(pair[0]) is not dict
            or pair[0].keys() != {"value"}
            or type(pair[0]["value"]) is not str
        ):
            raise ValueError("projected source field requires a named pair")
        names.append(pair[0]["value"])
    if tuple(names) != tuple(sorted(_CoupledGraphPreflight._source_schemas[kind])):
        raise ValueError("projected source fields differ from complete schema")


def materialize(root: dict[str, Any], nodes: list[dict[str, Any]]) -> Any:
    """Build exactly one object per admitted node ID using fixed constructors."""
    memo: dict[int, Any] = {}

    def token(value: dict[str, Any]) -> Any:
        if "value" in value:
            return value["value"]
        index = value["ref"]
        if index in memo:
            return memo[index]
        node = nodes[index]
        tag = node["tag"]
        if tag == "array":
            result: Any = decode_array(node)
        elif tag == "record":
            result = RECORD_TYPES[node["type"]](
                **{name: token(child) for name, child in node["fields"]}
            )
        elif tag == "dict":
            pairs = [(token(key), token(child)) for key, child in node["items"]]
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ValueError("duplicate composite dictionary keys")
        elif tag == "pcg64":
            result = np.random.Generator(np.random.PCG64(0))
            result.bit_generator.state = token(node["state"])
        else:
            items = [token(child) for child in node["items"]]
            constructors = {"tuple": tuple, "list": list, "set": set, "frozenset": frozenset}
            result = (
                deque(items, maxlen=node["maxlen"]) if tag == "deque" else constructors[tag](items)
            )
            if tag in ("set", "frozenset") and len(result) != len(items):
                raise ValueError("duplicate composite set members")
        memo[index] = result
        return result

    return token(root)
