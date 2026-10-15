"""One bounded native update for a surviving coordinator's admitted request.

This is separate from the unchanged read/stage checkpoint worker. It never
publishes a file or decides completion authority; the parent validates B.
The opt-in interruption fixture pauses after computing/encoding B but discloses
only control-flow telemetry, never checkpoint bytes or a completion claim.
"""

import argparse
from hashlib import sha256
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.adapters.coupled_checkpoint_worker import _emit, _model, _request, _stop
from src.adapters.numpy_checkpoint_frames import exact_object
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest


def main(*, pause_before_result: bool = False) -> int:
    if type(pause_before_result) is not bool:
        raise ValueError("pause_before_result must be an exact boolean")
    _emit(dict(event="hello", pid=os.getpid(), version=list(sys.version_info[:3])))
    request = _request()
    exact_object(request, {"op", "binding", "limits", "expected_sha256", "payload_utf8"})
    if type(request["op"]) is not str or request["op"] != "update":
        raise ValueError("native worker requires exact update operation")
    binding = CodecBinding(**exact_object(request["binding"], {"source_sha256", "policy_sha256"}))
    limits = CodecLimits(
        **exact_object(
            request["limits"],
            {"max_encoded_bytes", "max_array_bytes", "max_layers", "max_dimension"},
        )
    )
    if limits.max_encoded_bytes > 65536 or limits.max_array_bytes > 65536:
        raise ValueError("native worker codec caps exceed65536")
    require_codec_digest(request["expected_sha256"])
    if type(request["payload_utf8"]) is not str:
        raise ValueError("native worker requires exact UTF-8 checkpoint text")
    codec = CoupledCheckpointCodec()
    snapshot = codec.decode(
        request["payload_utf8"].encode("utf-8"),
        binding=binding,
        expected_sha256=request["expected_sha256"],
        limits=limits,
    )
    if tuple(weight.shape for weight in snapshot.parameters.hidden_weights) != (
        (2, 3),
        (3, 2),
    ) or tuple(snapshot.precisions) != (1.0, 1.0):
        raise ValueError("native worker requires frozen D2/H(3,2), precisions(1,1)")
    model = _model(snapshot)
    features = np.array([[0.2, -0.3], [-0.4, 0.6]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    result = model.train_epoch(features.copy(), targets.copy(), 0.02, 2, 0.05)
    payload = codec.encode(model.snapshot_state(), binding=binding, limits=limits)
    if pause_before_result:
        _emit(dict(event="computed", training_calls=1))
        _stop()
        return 0
    _emit(
        dict(
            event="updated",
            payload_utf8=payload.decode("utf-8"),
            digest=sha256(payload).hexdigest(),
            energy=result.energy,
            predictions=model.predict_proba(features).tolist(),
            training_calls=1,
        )
    )
    _stop()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause-before-result", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(main(pause_before_result=arguments.pause_before_result))
