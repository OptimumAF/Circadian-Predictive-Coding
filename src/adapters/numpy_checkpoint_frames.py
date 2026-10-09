"""Bounded explicit native NumPy frames; no generic graph loading or IO.

Supports finite f8, i8, i4 and boolean values in contiguous C/F storage.
Schemas determine all shapes/dtypes. Callers preflight aggregate payload bounds.
"""

from base64 import b64decode, b64encode
import json
from math import prod
from typing import Any

import numpy as np

DTYPES: dict[str, np.dtype[Any]] = {
    "<f8": np.dtype(np.float64),
    "<i8": np.dtype(np.int64),
    "<i4": np.dtype(np.int32),
    "|b1": np.dtype(np.bool_),
}


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def exact_object(value, keys):
    if type(value) is not dict or value.keys() != keys:
        raise ValueError("checkpoint fields differ from exact supported schema")
    return value


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate checkpoint JSON key")
        result[key] = value
    return result


def frame_size(shape, dtype) -> int:
    return prod(shape) * DTYPES[dtype].itemsize


def empty_frame(shape, dtype):
    # C and F tags have equal wire length, so either gives the exact size bound.
    return dict(dtype=dtype, shape=list(shape), order="C", data="")


def native_order(value) -> str:
    # Singleton axes/vectors have both flags; C is their unique canonical tag.
    if value.flags.c_contiguous:
        return "C"
    if value.flags.f_contiguous:
        return "F"
    raise ValueError("unsupported noncontiguous native array storage")


def validate_native_array(value, shape, dtype) -> None:
    if (
        type(value) is not np.ndarray
        or value.dtype != DTYPES[dtype]
        or value.shape != shape
        or not np.all(np.isfinite(value))
    ):
        raise ValueError("native array dtype/shape/finite values differ")
    native_order(value)


def validate_frame(value, shape, dtype) -> None:
    exact_object(value, {"dtype", "shape", "order", "data"})
    if (
        type(value["shape"]) is not list
        or any(type(v) is not int for v in value["shape"])
        or value["shape"] != list(shape)
        or value["dtype"] != dtype
        or type(value["data"]) is not str
        or len(value["data"]) != 4 * ((frame_size(shape, dtype) + 2) // 3)
    ):
        raise ValueError("frame dtype/shape/byte length differs")
    order = value["order"]
    if type(order) is not str or order not in ("C", "F"):
        raise ValueError("unsupported exact frame storage order")
    if order == "F" and (len(shape) == 1 or any(dimension == 1 for dimension in shape)):
        raise ValueError("ambiguous contiguous storage requires canonical C tag")


def encode_frame(value, shape, dtype):
    validate_native_array(value, shape, dtype)
    order = native_order(value)
    return dict(
        dtype=dtype,
        shape=list(shape),
        order=order,
        data=b64encode(value.astype(dtype, copy=False).tobytes(order=order)).decode("ascii"),
    )


def decode_frame(value, shape, dtype):
    validate_frame(value, shape, dtype)
    raw = b64decode(value["data"], validate=True)
    if len(raw) != frame_size(shape, dtype) or b64encode(raw).decode("ascii") != value["data"]:
        raise ValueError("noncanonical or wrong array bytes")
    if dtype == "|b1" and any(v not in (0, 1) for v in raw):
        raise ValueError("boolean bytes must be canonical zero/one")
    order = value["order"]
    result = (
        np.frombuffer(raw, dtype=dtype)
        .reshape(shape, order=order)
        .astype(DTYPES[dtype], order=order, copy=True)
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite native array values")
    return result
