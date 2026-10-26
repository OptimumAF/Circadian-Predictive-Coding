"""Bounded scalar BP record readback using the existing full native snapshot.

Consumes JSON tensor observations for small CPU BP heads, returns owned validated
TorchFeatureSnapshot. No IO, pickle, runtime/event/consent/budget authority or
training; the experiment binds source identity and owns any later updates.
"""

from dataclasses import fields
from typing import Any

import numpy as np

from src.adapters.torch_feature_learner import (
    TorchFeaturePolicy,
    TorchFeatureSnapshot,
    _parameter_shapes,
)
from src.shared.torch_runtime import require_torch


def _tensor_record(record: Any, shape: tuple[int, ...]):
    if (
        type(record) is not dict
        or record.keys() != {"dtype", "shape", "values"}
        or record["dtype"] != "torch.float32"
        or type(record["shape"]) is not list
        or record["shape"] != list(shape)
        or any(type(n) is not int for n in record["shape"])
    ):
        raise ValueError("BP tensor record requires exact float32 scalar geometry")
    rows = record["values"]
    if (
        type(rows) is not list
        or len(rows) != shape[0]
        or any(type(row) is not list or len(row) != shape[1] for row in rows)
        or any(type(v) not in (int, float) for row in rows for v in row)
    ):
        raise ValueError("BP tensor record values differ from bounded shape")
    values = np.array(rows, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        native = values.astype(np.float32)
    if not np.isfinite(native).all() or not np.array_equal(values, native.astype(np.float64)):
        raise ValueError("BP tensor record must preserve exact finite float32 values")
    return require_torch().from_numpy(native)


def bp_snapshot_from_record(record: Any) -> TorchFeatureSnapshot:
    """Read a small complete BP state; never recreate its application authority."""
    if type(record) is not dict or record.keys() != {f.name for f in fields(TorchFeatureSnapshot)}:
        raise ValueError("complete BP snapshot record fields required")
    if type(record["policy"]) is not dict:
        raise ValueError("complete BP policy record required")
    policy = TorchFeaturePolicy(**record["policy"])
    if policy.method != "bp" or policy.feature_dim > 16:
        raise ValueError("record readback supports small BP feature heads only")
    shapes = _parameter_shapes(policy)
    if type(record["head_state"]) is not dict or not set(shapes) <= record["head_state"].keys():
        raise ValueError("complete BP head record required")
    head = dict(record["head_state"])
    for name, shape in shapes.items():
        head[name] = _tensor_record(head[name], shape)
    gradients = record["gradients"]
    if type(gradients) is not dict or gradients.keys() != shapes.keys():
        raise ValueError("complete BP gradient record required")
    gradients = {
        name: None if value is None else _tensor_record(value, shapes[name])
        for name, value in gradients.items()
    }
    raw_optimizer = record["optimizer_state"]
    if (
        type(raw_optimizer) is not dict
        or raw_optimizer.keys() != {"state", "param_groups"}
        or type(raw_optimizer["state"]) is not dict
    ):
        raise ValueError("complete BP optimizer record required")
    optimizer = dict(raw_optimizer)
    converted: dict[int, Any] = {}
    for key, value in optimizer["state"].items():
        if (
            type(key) is not str
            or key not in {"0", "1", "2", "3"}
            or type(value) is not dict
            or value.keys() != {"momentum_buffer"}
        ):
            raise ValueError("BP optimizer requires canonical bounded JSON parameter IDs")
        converted[int(key)] = {
            "momentum_buffer": _tensor_record(
                value["momentum_buffer"], tuple(shapes.values())[int(key)]
            )
        }
    optimizer["state"] = converted
    return TorchFeatureSnapshot(
        **{
            **record,
            "policy": policy,
            "head_state": head,
            "gradients": gradients,
            "optimizer_state": optimizer,
        }
    )
