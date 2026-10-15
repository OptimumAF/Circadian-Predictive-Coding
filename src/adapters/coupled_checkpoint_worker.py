"""Bounded checkpoint staging/restore worker for supervised interruption fixtures.

Stage pauses only after accepted file-store fsync/close and never publishes its
replacement. Read restores full native state and predicts a fixed input without
training. Inputs are explicit local paths/bindings/caps, not owner authority.
No subprocess launching, generic object loading or native update is provided.
"""

from dataclasses import asdict
import json
from math import isfinite
import os
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.adapters.numpy_checkpoint_frames import exact_object, unique_pairs
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.infra.checkpoint_files import BoundedCheckpointFileStore

_LINE_CAP = 65536
_OUTPUT_CAP = 262144
_REQUEST_FIELDS = {"op", "path", "binding", "limits", "expected_sha256"}


def _nonfinite(value):
    raise ValueError(f"nonfinite worker JSON: {value}")


def _finite_float(value):
    number = float(value)
    if not isfinite(number):
        raise ValueError("nonfinite worker JSON number")
    return number


def _request() -> dict[str, Any]:
    raw = sys.stdin.buffer.readline(_LINE_CAP + 1)
    if not raw or len(raw) > _LINE_CAP or not raw.endswith(b"\n"):
        raise ValueError("worker requires one bounded complete JSON line")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=unique_pairs,
        parse_constant=_nonfinite,
        parse_float=_finite_float,
    )
    if type(value) is not dict:
        raise ValueError("worker request must be an exact JSON object")
    return value


def _emit(value: dict[str, Any]) -> None:
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(raw) > _LINE_CAP or len(raw) > _OUTPUT_CAP:
        raise ValueError("worker response exceeds its output bounds")
    count = sys.stdout.buffer.write(raw)
    if type(count) is not int or count != len(raw):
        raise OSError("worker response write was incomplete")
    sys.stdout.buffer.flush()


def _controls(request):
    op = request.get("op")
    if type(op) is not str or op not in ("stage", "read"):
        raise ValueError("worker operation must be stage or read")
    exact_object(request, _REQUEST_FIELDS | ({"payload_utf8"} if op == "stage" else set()))
    if type(request["path"]) is not str or not Path(request["path"]).is_absolute():
        raise ValueError("worker target path must be an absolute native string")
    binding = exact_object(request["binding"], {"source_sha256", "policy_sha256"})
    limits = exact_object(
        request["limits"], {"max_encoded_bytes", "max_array_bytes", "max_layers", "max_dimension"}
    )
    binding, limits = CodecBinding(**binding), CodecLimits(**limits)
    require_codec_digest(request["expected_sha256"])
    # This bounded fixture worker intentionally has local 65536-byte encoded
    # AND aggregate-array ceilings, even when the general codec permits more.
    # Request metadata cannot relax the worker's independent transport ceiling.
    if limits.max_encoded_bytes > _LINE_CAP or limits.max_array_bytes > _LINE_CAP:
        raise ValueError("worker codec limits exceed the bounded request allowance")
    if op == "stage" and type(request["payload_utf8"]) is not str:
        raise ValueError("stage payload must be an exact UTF-8 string")
    return op, Path(request["path"]), binding, limits


def _stop() -> None:
    request = _request()
    exact_object(request, {"op"})
    if type(request["op"]) is not str or request["op"] != "stop":
        raise ValueError("worker expected exact stop operation")


class _StageStopped(Exception):
    pass


def _stage(request, path, binding, limits):
    codec = CoupledCheckpointCodec()
    raw = request["payload_utf8"].encode("utf-8")
    snapshot = codec.decode(
        raw, binding=binding, expected_sha256=request["expected_sha256"], limits=limits
    )

    def pause(temporary, destination):
        if Path(destination) != path:
            raise ValueError("worker staging destination changed")
        _emit(
            dict(event="staged", temporary=Path(temporary).name, digest=request["expected_sha256"])
        )
        _stop()
        raise _StageStopped()

    with patch("src.infra.checkpoint_files.os.replace", side_effect=pause):
        try:
            BoundedCheckpointFileStore(path, codec).save(snapshot, binding=binding, limits=limits)
        except _StageStopped:
            return
    raise RuntimeError("stage mode must never publish a replacement")


def _model(snapshot: CoupledSnapshot) -> CoupledPredictiveCodingNetwork:
    parameters = snapshot.parameters
    input_dim = parameters.hidden_weights[0].shape[0]
    if input_dim != 2:
        raise ValueError("worker fixed prediction fixture requires input_dim 2")
    widths = tuple(weight.shape[1] for weight in parameters.hidden_weights)
    if snapshot.wake_state is None:
        model = CoupledPredictiveCodingNetwork(2, widths, seed=0, precisions=snapshot.precisions)
    else:
        wake = snapshot.wake_state
        policy = CoupledCircadianConfig(
            wake.chemical_decay,
            wake.chemical_buildup_rate,
            wake.plasticity_sensitivity,
            wake.min_plasticity,
        )
        model = CoupledCircadianPredictiveCodingNetwork(
            2, widths, seed=0, precisions=snapshot.precisions, config=policy
        )
    model.restore_state(snapshot)
    return model


def _plain_snapshot(snapshot: CoupledSnapshot):
    return json.loads(
        json.dumps(asdict(snapshot), allow_nan=False, default=lambda value: value.tolist())
    )


def main() -> int:
    _emit(dict(event="hello", pid=os.getpid(), version=list(sys.version_info[:3])))
    request = _request()
    op, path, binding, limits = _controls(request)
    if op == "stage":
        _stage(request, path, binding, limits)
    else:
        snapshot = BoundedCheckpointFileStore(path, CoupledCheckpointCodec()).load(
            binding=binding, expected_sha256=request["expected_sha256"], limits=limits
        )
        model = _model(snapshot)
        features = np.array([[0.2, -0.3], [-0.4, 0.6]], dtype=np.float64)
        _emit(
            dict(
                event="restored",
                snapshot=_plain_snapshot(model.snapshot_state()),
                predictions=model.predict_proba(features).tolist(),
            )
        )
        _stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
