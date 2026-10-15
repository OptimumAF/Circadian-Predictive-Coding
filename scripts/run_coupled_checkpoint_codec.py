"""Show complete coupled snapshot bytes, refusal and native continuation."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot

FAMILIES = ("pc", "neutral_cpc", "active_cpc")
LIMITS = CodecLimits(65536, 8192, 4, 16)
BINDING = CodecBinding(
    sha256(b"coupled codec engineering fixture source label v1").hexdigest(),
    sha256(b"coupled codec engineering fixed external policy label v1").hexdigest(),
)


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="coupled_snapshot_codec_engineering_v1",
        scope="engineering_roundtrip_not_scientific_comparison",
        families=list(FAMILIES),
        geometry=dict(input_dim=2, hidden_dims=[3, 2], precisions=[1.0, 1.0]),
        seed=23,
        external_training_settings=dict(
            learning_rate=0.02, inference_steps=2, inference_learning_rate=0.05
        ),
        warm_updates_per_family=2,
        continuation_updates_per_branch=1,
        native_training_calls=12,
        native_energy_evaluations=48,
        limits=asdict(LIMITS),
        binding=asdict(BINDING),
        binding_scope="Illustrative independently known fixture labels, not source sealing or authority",
        limitations="Native model bytes only. Adapter rates, runtime cursor, lifecycle, permissions, budgets, disk durability and process ownership remain external. No crash-recovery, serving or learning-benefit claim.",
    )


def _model(family: str) -> CoupledPredictiveCodingNetwork:
    if family == "pc":
        return CoupledPredictiveCodingNetwork(2, (3, 2), seed=23, precisions=(1.0, 1.0))
    config = (
        CoupledCircadianConfig.matched_pc_control()
        if family == "neutral_cpc"
        else CoupledCircadianConfig()
    )
    return CoupledCircadianPredictiveCodingNetwork(
        2, (3, 2), seed=23, precisions=(1.0, 1.0), config=config
    )


def _same(left: CoupledSnapshot, right: CoupledSnapshot) -> None:
    np.testing.assert_equal(asdict(left), asdict(right))


def _train(model: CoupledPredictiveCodingNetwork, x: np.ndarray, y: np.ndarray) -> float:
    return model.train_epoch(x.copy(), y.copy(), 0.02, 2, 0.05).energy


def _refused(codec: CoupledCheckpointCodec, raw: bytes, digest: str, binding: CodecBinding) -> str:
    try:
        codec.decode(raw, binding=binding, expected_sha256=digest, limits=LIMITS)
    except ValueError as error:
        return str(error)
    raise AssertionError("invalid bytes or binding were accepted")


def _family_example(family: str, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    model = _model(family)
    warm_energies = [_train(model, x, y) for _ in range(2)]
    source = model.snapshot_state()
    codec = CoupledCheckpointCodec()
    raw = codec.encode(source, binding=BINDING, limits=LIMITS)
    digest = sha256(raw).hexdigest()
    decoded = codec.decode(raw, binding=BINDING, expected_sha256=digest, limits=LIMITS)
    _same(source, decoded)
    if codec.encode(decoded, binding=BINDING, limits=LIMITS) != raw:
        raise AssertionError("round-trip bytes changed")
    restored = _model(family)
    restored.restore_state(decoded)
    _same(source, restored.snapshot_state())
    predictions = restored.predict_proba(x.copy())
    np.testing.assert_array_equal(predictions, model.predict_proba(x.copy()))
    refusals = dict(
        corrupt_digest=_refused(codec, raw[:-1] + b"!", digest, BINDING),
        wrong_policy_binding=_refused(codec, raw, digest, replace(BINDING, policy_sha256="f" * 64)),
    )
    _same(source, model.snapshot_state())
    _same(source, restored.snapshot_state())
    source_record, decoded_record = asdict(source), asdict(decoded)
    # Decoded storage is caller-owned; editing it after native restore cannot
    # mutate the immutable bytes, independent source or restored model.
    decoded.parameters.output_bias[:] = 999
    _same(source, model.snapshot_state())
    _same(source, restored.snapshot_state())
    if sha256(raw).hexdigest() != digest:
        raise AssertionError("immutable encoded bytes changed")
    before = restored.snapshot_state()
    energy = _train(model, x, y)
    restored_energy = _train(restored, x, y)
    _same(model.snapshot_state(), restored.snapshot_state())
    if energy != restored_energy:
        raise AssertionError("decoded continuation energy changed")
    if np.array_equal(
        before.parameters.output_weight, restored.snapshot_state().parameters.output_weight
    ):
        raise AssertionError("continuation did not change output weights")
    return dict(
        family=family,
        warm_energies=warm_energies,
        source_snapshot=source_record,
        decoded_snapshot=decoded_record,
        encoded_utf8=raw.decode("utf-8"),
        encoded_bytes=len(raw),
        encoded_sha256=digest,
        predictions_before_continuation=predictions,
        refusals=refusals,
        decoded_storage_detached=True,
        continued_energy=energy,
        continued_source=asdict(model.snapshot_state()),
        continued_restored=asdict(restored.snapshot_state()),
    )


def run_example() -> dict[str, Any]:
    x = np.array([[0.2, -0.3], [-0.4, 0.6]], dtype=np.float64)
    y = np.array([[1.0], [0.0]], dtype=np.float64)
    rows = [_family_example(family, x, y) for family in FAMILIES]
    for stage in ("source_snapshot", "continued_source"):
        np.testing.assert_equal(rows[0][stage]["parameters"], rows[1][stage]["parameters"])
    return dict(**describe_example(), training=dict(role="train", features=x, targets=y), rows=rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported example value: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Preview without model work")
    args = parser.parse_args()
    encoded = json.dumps(
        describe_example() if args.describe else run_example(),
        default=_json_default,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    if len((encoded + "\n").replace("\n", os.linesep).encode("utf-8")) > 2 * 1024**2:
        raise RuntimeError("coupled codec example exceeds2MiB")
    print(encoded)


if __name__ == "__main__":
    main()
