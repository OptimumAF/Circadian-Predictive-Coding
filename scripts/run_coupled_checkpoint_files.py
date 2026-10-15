"""Demonstrate bounded native checkpoint files and precommit failure recovery."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from scripts.run_coupled_checkpoint_codec import (
    BINDING,
    FAMILIES,
    LIMITS,
    _json_default,
    _model,
    _same,
    _train,
)
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.coupled_predictive_coding import CoupledSnapshot
from src.infra.checkpoint_files import BoundedCheckpointFileStore


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="coupled_checkpoint_files_engineering_v1",
        families=list(FAMILIES),
        seed=23,
        geometry=dict(input_dim=2, hidden_dims=[3, 2], precisions=[1.0, 1.0]),
        external_training_settings=dict(
            learning_rate=0.02, inference_steps=2, inference_learning_rate=0.05
        ),
        native_training_calls=12,
        native_energy_evaluations=48,
        directory_policy="Actual run requires a new dedicated --directory; description performs no IO or model work",
        operations=[
            "saveA",
            "loadA",
            "continue",
            "failed_fsync_saveB",
            "recoverA",
            "saveB",
            "loadB",
        ],
        binding=asdict(BINDING),
        limits=asdict(LIMITS),
        binding_scope="Illustrative fixture labels; caller-owned consistency inputs, not provenance or runtime owner authority",
        limitations="Single cooperating writer and trusted parent. Model bytes only; no power-loss, hostile filesystem, runtime cursor/budget/lifecycle/permission/owner or full crash-recovery guarantee.",
    )


def _payload(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(LIMITS.max_encoded_bytes + 1)
    if len(raw) > LIMITS.max_encoded_bytes:
        raise AssertionError("example file exceeded its byte allowance")
    return raw


def _load(store: BoundedCheckpointFileStore[CoupledSnapshot], digest: str) -> CoupledSnapshot:
    return store.load(binding=BINDING, expected_sha256=digest, limits=LIMITS)


def _refusals(
    store: BoundedCheckpointFileStore[CoupledSnapshot], old_digest: str, digest: str, size: int
) -> dict[str, str]:
    outcomes = {}
    for name, expected, binding, limits in (
        ("stale_digest", old_digest, BINDING, LIMITS),
        ("wrong_policy_binding", digest, replace(BINDING, policy_sha256="f" * 64), LIMITS),
        ("encoded_cap", digest, BINDING, replace(LIMITS, max_encoded_bytes=size - 1)),
    ):
        try:
            store.load(binding=binding, expected_sha256=expected, limits=limits)
        except ValueError as error:
            outcomes[name] = str(error)
        else:
            raise AssertionError(f"file control {name} was accepted")
    return outcomes


def _family_example(directory: Path, family: str, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    model = _model(family)
    warm_energies = [_train(model, x, y) for _ in range(2)]
    original = model.snapshot_state()
    store = BoundedCheckpointFileStore(directory / f"{family}.json", CoupledCheckpointCodec())
    digest_a = store.save(original, binding=BINDING, limits=LIMITS)
    raw_a = _payload(store.path)
    loaded_a = _load(store, digest_a)
    _same(original, loaded_a)
    restored = _model(family)
    restored.restore_state(loaded_a)
    np.testing.assert_array_equal(model.predict_proba(x.copy()), restored.predict_proba(x.copy()))
    expected_energy = _train(model, x, y)
    actual_energy = _train(restored, x, y)
    _same(model.snapshot_state(), restored.snapshot_state())
    if expected_energy != actual_energy:
        raise AssertionError("file round-trip changed continued native energy")
    candidate_b = model.snapshot_state()
    with patch(
        "src.infra.checkpoint_files.os.fsync",
        side_effect=OSError("injected precommit file fsync failure"),
    ):
        try:
            store.save(candidate_b, binding=BINDING, limits=LIMITS)
        except OSError as error:
            failure = str(error)
        else:
            raise AssertionError("injected precommit file failure was ignored")
    if _payload(store.path) != raw_a:
        raise AssertionError("failed replacement changed the previous checkpoint")
    recovered_a = _load(store, digest_a)
    _same(recovered_a, original)
    _same(model.snapshot_state(), candidate_b)
    _same(restored.snapshot_state(), candidate_b)
    if list(directory.glob(".*.tmp")):
        raise AssertionError("precommit failure left an owned temporary file")
    digest_b = store.save(candidate_b, binding=BINDING, limits=LIMITS)
    raw_b = _payload(store.path)
    reloaded_b = _load(store, digest_b)
    _same(candidate_b, reloaded_b)
    refusals = _refusals(store, digest_a, digest_b, len(raw_b))
    if raw_a == raw_b or digest_a == digest_b or _payload(store.path) != raw_b:
        raise AssertionError("replacement or later refusal changed the wrong file state")
    _same(model.snapshot_state(), candidate_b)
    _same(restored.snapshot_state(), candidate_b)
    return dict(
        family=family,
        path=str(store.path),
        warm_energies=warm_energies,
        checkpoint_a=dict(
            digest=digest_a, payload_utf8=raw_a.decode("utf-8"), snapshot=asdict(original)
        ),
        loaded_a=asdict(loaded_a),
        precommit_failure=failure,
        previous_file_preserved=True,
        recovered_a=asdict(recovered_a),
        checkpoint_b=dict(
            digest=digest_b, payload_utf8=raw_b.decode("utf-8"), snapshot=asdict(candidate_b)
        ),
        reloaded_b=asdict(reloaded_b),
        continued_original=asdict(model.snapshot_state()),
        continued_restored=asdict(restored.snapshot_state()),
        continued_energy=actual_energy,
        refusals=refusals,
    )


def run_example(directory: Path) -> dict[str, Any]:
    # Explicit fresh directory prevents a repeat example from replacing unrelated
    # user files. Existing regular-file replacement is exercised inside this run.
    directory = directory.absolute()
    directory.mkdir(parents=True, exist_ok=False)
    x = np.array([[0.2, -0.3], [-0.4, 0.6]], dtype=np.float64)
    y = np.array([[1.0], [0.0]], dtype=np.float64)
    rows = [_family_example(directory, family, x, y) for family in FAMILIES]
    for phase in ("checkpoint_a", "checkpoint_b"):
        np.testing.assert_equal(
            rows[0][phase]["snapshot"]["parameters"], rows[1][phase]["snapshot"]["parameters"]
        )
    return dict(
        **describe_example(),
        directory=str(directory),
        training=dict(role="train", features=x, targets=y),
        rows=rows,
        remaining_temporary_files=[str(path) for path in directory.glob(".*.tmp")],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Preview without IO or model work")
    parser.add_argument("--directory", type=Path, help="New dedicated directory for actual files")
    args = parser.parse_args()
    if not args.describe and args.directory is None:
        parser.error("--directory is required unless --describe is selected")
    report = describe_example() if args.describe else run_example(args.directory)
    encoded = json.dumps(report, default=_json_default, allow_nan=False, sort_keys=True, indent=2)
    if len((encoded + "\n").replace("\n", os.linesep).encode("utf-8")) > 2 * 1024**2:
        raise RuntimeError("checkpoint file example exceeds2MiB")
    print(encoded)


if __name__ == "__main__":
    main()
