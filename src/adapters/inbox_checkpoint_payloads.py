"""Bounded exact portable real numeric inbox frames after metadata validation.

No native update, payload authority, IO, learner policy or live restoration.
"""

from base64 import b64decode, b64encode
from math import isfinite, prod
from struct import iter_unpack
import numpy as np
from src.adapters.numpy_checkpoint_frames import canonical_json, exact_object, native_order

FORMATS = {
    "i1": "b",
    "u1": "B",
    "i2": "h",
    "u2": "H",
    "i4": "i",
    "u4": "I",
    "i8": "q",
    "u8": "Q",
    "f2": "e",
    "f4": "f",
    "f8": "d",
}


def payload_shape(value, width, policy, limits, encoded) -> tuple[int, int]:
    if encoded:
        exact_object(value, {"dtype", "shape", "order", "data"})
        shape = value["shape"]
    else:
        if type(value) is not np.ndarray:
            raise ValueError("unsupported native inbox payload type")
        shape = value.shape
    if (
        type(shape) not in (list, tuple)
        or len(shape) != 2
        or any(type(n) is not int or not 0 < n <= limits.max_dimension for n in shape)
        or shape[0] > policy.max_batch_rows
        or shape[1] != width
    ):
        raise ValueError("inbox frame differs from independently expected shape policy")
    return shape[0], shape[1]


def dtype_name(value, allowed, encoded):
    name = value["dtype"] if encoded else value.dtype.str
    if type(name) is not str or name not in allowed:
        raise ValueError("inbox dtype differs from independently expected real numeric policy")
    return name


def byte_size(shape, dtype):
    return prod(shape) * int(dtype[2:])


def validate_frame(value, shape, dtype):
    if (
        type(value["shape"]) is not list
        or value["shape"] != list(shape)
        or type(value["data"]) is not str
        or len(value["data"]) != 4 * ((byte_size(shape, dtype) + 2) // 3)
    ):
        raise ValueError("inbox frame shape/byte length differs")
    if (
        type(value["order"]) is not str
        or value["order"] not in ("C", "F")
        or (value["order"] == "F" and 1 in shape)
    ):
        raise ValueError("inbox frame requires canonical C/F storage tag")


def validate_native_payloads(specs):
    arrays = []
    for row, field, shape, dtype in specs:
        value = row[field]
        native_order(value)
        if not np.all(np.isfinite(value)):
            raise ValueError("native inbox array must be finite")
        if field == "targets" and np.any((value < 0) | (value > 1)):
            raise ValueError("inbox binary/soft labels must lie in zero/one range")
        arrays.append(value)
    if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :]):
        raise ValueError("unsupported shared inbox payload ownership edge")


def specifications(data, policy, limits, encoded):
    specs = []
    rows_by_key: dict[tuple[str, str], int] = {}
    for name, field, width, allowed in (
        ("experiences", "features", policy.input_dim, policy.feature_dtypes),
        ("labels", "targets", 1, policy.target_dtypes),
    ):
        for row in data[name]:
            value = row[field]
            shape = payload_shape(value, width, policy, limits, encoded)
            dtype = dtype_name(value, allowed, encoded)
            if encoded:
                validate_frame(value, shape, dtype)
            key = row["episode_id"], row["sample_id"]
            if key in rows_by_key and rows_by_key[key] != shape[0]:
                raise ValueError("paired inbox payload batch sizes differ")
            rows_by_key[key] = shape[0]
            specs.append((row, field, shape, dtype))
    if sum(byte_size(shape, dtype) for _, _, shape, dtype in specs) > limits.max_array_bytes:
        raise ValueError("aggregate inbox arrays exceed original bound")
    if not encoded:
        validate_native_payloads(specs)
    return specs


def size_preflight(body, specs, limits):
    saved = [row[field] for row, field, _, _ in specs]
    try:
        for row, field, shape, dtype in specs:
            row[field] = dict(dtype=dtype, shape=list(shape), order="C", data="")
        size = len(canonical_json(body)) + sum(
            4 * ((byte_size(shape, dtype) + 2) // 3) for _, _, shape, dtype in specs
        )
        if size > limits.max_encoded_bytes:
            raise ValueError("complete inbox wire exceeds original bound")
    finally:
        for (row, field, _, _), value in zip(specs, saved):
            row[field] = value


def encode_frame(value, shape, dtype):
    order = native_order(value)
    return dict(
        dtype=dtype,
        shape=list(shape),
        order=order,
        data=b64encode(value.tobytes(order=order)).decode("ascii"),
    )


def encode_payloads(specs):
    for row, field, shape, dtype in specs:
        row[field] = encode_frame(row[field], shape, dtype)


def decode_payloads(specs):
    payloads = []
    # Validate all raw finite/range data before materializing any NumPy array.
    for row, field, shape, dtype in specs:
        frame = row[field]
        raw = b64decode(frame["data"], validate=True)
        if len(raw) != byte_size(shape, dtype) or b64encode(raw).decode("ascii") != frame["data"]:
            raise ValueError("noncanonical or mismatched inbox bytes")
        fmt = ("<" if dtype[0] == "|" else dtype[0]) + FORMATS[dtype[1:]]
        if any(
            not isfinite(value) or (field == "targets" and not 0 <= value <= 1)
            for (value,) in iter_unpack(fmt, raw)
        ):
            raise ValueError("inbox arrays must be finite with valid soft labels")
        payloads.append(raw)
    for (row, field, shape, dtype), raw in zip(specs, payloads):
        order = row[field]["order"]
        row[field] = np.frombuffer(raw, dtype=dtype).reshape(shape, order=order).copy(order=order)
