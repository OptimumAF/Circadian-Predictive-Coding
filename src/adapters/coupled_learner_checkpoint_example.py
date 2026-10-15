"""Fixed full-policy checkpoint continuation for PC and neutral/active CPC.

Inputs are a fresh private directory and fixed train-only engineering data.
Outputs compare exact full learner bytes/diagnostics/predictions before and after
restore. No native objective change, tuning, managed authority or learning claim.
"""

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_interruption import _model
from src.adapters.coupled_learner_checkpoint import (
    CoupledLearnerCheckpointCodec,
    capture_coupled_learner,
    restore_coupled_learner,
)
from src.adapters.coupled_learners import CoupledLearner
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_inference import CoupledInferencePolicy
from src.infra.checkpoint_files import BoundedCheckpointFileStore

BINDING = CodecBinding(
    sha256(b"coupled learner checkpoint engineering source v1").hexdigest(),
    sha256(b"coupled learner checkpoint engineering policy v1").hexdigest(),
)
LIMITS = CodecLimits(65536, 8192, 4, 16)
FAMILIES = ("pc", "neutral_cpc", "active_cpc")
MODES = ("fixed", "guarded")


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="coupled_learner_checkpoint_engineering_v1",
        families=list(FAMILIES),
        inference_modes=list(MODES),
        seed=23,
        geometry=dict(input_dim=2, hidden_dims=[3, 2], precisions=[1.0, 1.0]),
        training=dict(
            features=[[0.2, -0.3], [-0.4, 0.6]],
            targets=[[1.0], [0.0]],
            role="train",
            learning_rate=0.025,
            inference_steps=3,
            inference_learning_rate=0.07,
            guarded_policy=dict(
                min_steps=1, gradient_tolerance=1e6, relative_energy_tolerance=None
            ),
        ),
        native_training_calls=24,
        native_energy_evaluation_cap=120,
        logical_constructor_cap=48,
        prediction_cap=32,
        process_launches=0,
        binding=asdict(BINDING),
        limits=asdict(LIMITS),
        limitations="Fixed train-only engineering continuation. No managed runtime/owner restoration, recovery authority, performance comparison or scientific benefit claim.",
    )


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _bytes(codec, learner):
    return codec.encode(capture_coupled_learner(learner), binding=BINDING, limits=LIMITS)


def _case(directory, family, mode):
    x = np.array([[0.2, -0.3], [-0.4, 0.6]], dtype=np.float64)
    y = np.array([[1.0], [0.0]], dtype=np.float64)
    policy = CoupledInferencePolicy(1, 1e6, None) if mode == "guarded" else None
    original = CoupledLearner(
        _model(family),
        learning_rate=0.025,
        inference_steps=3,
        inference_learning_rate=0.07,
        inference_policy=policy,
    )
    for _ in range(2):
        original.train_batch(x.copy(), y.copy())
    codec = CoupledLearnerCheckpointCodec()
    checkpoint = capture_coupled_learner(original)
    if family != "pc":
        wake = checkpoint.native_state.wake_state
        _require(
            wake is not None and np.any(wake.chemical_state != 0).item(),
            "CPC checkpoint must retain genuine chemistry",
        )
    raw_a = codec.encode(checkpoint, binding=BINDING, limits=LIMITS)
    prediction_a = original.predict(x.copy()).tolist()
    path = directory / f"{family}-{mode}.json"
    _require(not path.exists(), "learner checkpoint example refuses existing output files")
    store = BoundedCheckpointFileStore(path, codec)
    digest_a = store.save(checkpoint, binding=BINDING, limits=LIMITS)
    _require(digest_a == sha256(raw_a).hexdigest(), "saved full learner digest differs")
    detached = store.load(binding=BINDING, expected_sha256=digest_a, limits=LIMITS)
    loaded_a = codec.encode(detached, binding=BINDING, limits=LIMITS)
    _require(loaded_a == raw_a, "bounded learner file roundtrip differs")
    restored = restore_coupled_learner(
        loaded_a, binding=BINDING, expected_sha256=digest_a, limits=LIMITS
    )
    restored_a = _bytes(codec, restored)
    _require(restored_a == raw_a, "restored learner policy/native state differs from A")
    _require(
        restored.predict(x.copy()).tolist() == prediction_a,
        "restored A prediction differs from original",
    )
    original_result = original.train_batch(x.copy(), y.copy())
    restored_result = restored.train_batch(x.copy(), y.copy())
    raw_b, resumed_b = _bytes(codec, original), _bytes(codec, restored)
    prediction_b = original.predict(x.copy()).tolist()
    _require(
        resumed_b == raw_b
        and restored_result == original_result
        and restored.predict(x.copy()).tolist() == prediction_b,
        "continued learner full bytes/diagnostic/prediction differ",
    )
    _require(
        codec.encode(checkpoint, binding=BINDING, limits=LIMITS) == raw_a,
        "continued learning mutated the captured checkpoint",
    )
    for expected_wakes, raw in ((2, raw_a), (3, raw_b)):
        wake = json.loads(raw)["native"]["wake_state"]
        if family == "pc":
            _require(wake is None, "PC acquired wake state")
        else:
            _require(wake["wake_updates"] == expected_wakes, "CPC lost complete wake history")
    row = dict(
        family=family,
        mode=mode,
        policy=asdict(checkpoint.policy),
        checkpoint_a_utf8=raw_a.decode(),
        restored_a_utf8=restored_a.decode(),
        uninterrupted_b_utf8=raw_b.decode(),
        resumed_b_utf8=resumed_b.decode(),
        checkpoint_sha256=digest_a,
        original_diagnostic=asdict(original_result),
        restored_diagnostic=asdict(restored_result),
        prediction_a=prediction_a,
        prediction_b=prediction_b,
        checkpoint_unchanged=True,
        complete_continuation_exact=True,
    )
    _require(
        len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 32768,
        "learner checkpoint summary exceeds fixed case cap",
    )
    return row


def run_example(directory: Path) -> dict[str, Any]:
    directory = Path(directory).resolve()
    _require(directory.is_dir(), "learner checkpoint example needs an existing private directory")
    expected = [directory / f"{family}-{mode}.json" for mode in MODES for family in FAMILIES]
    _require(not any(path.exists() for path in expected), "learner example refuses existing files")
    rows = [_case(directory, family, mode) for mode in MODES for family in FAMILIES]
    fields = ("hidden_weights", "hidden_biases", "output_weight", "output_bias")
    for offset in (0, 3):
        for stage in ("checkpoint_a_utf8", "uninterrupted_b_utf8"):
            pc = json.loads(rows[offset][stage])["native"]["arrays"]
            neutral = json.loads(rows[offset + 1][stage])["native"]["arrays"]
            _require(
                all(pc[field] == neutral[field] for field in fields),
                "PC-neutral parameter frames differ under matched checkpoint policy",
            )
    return dict(
        protocol=describe_example(),
        cases=rows,
        pc_neutral_parameters_exact=True,
        full_policy_continuation_verified=True,
        managed_recovery_verified=False,
    )
