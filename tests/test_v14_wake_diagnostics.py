"""P5.2b1 records core update diagnostics without changing v14 work."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import isfinite

import pytest

from scripts.run_continual_trigger_replay_outcomes import serialize_outcome_comparison
from scripts.run_continual_trigger_replay_training import serialize_training_study
from src.app.continual_trigger_replay_outcomes import score_trigger_replay_training_study
from src.app.continual_trigger_replay_runner import _parameter_digest
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.app.continual_trigger_replay_training_study import run_trigger_replay_training_study
from src.app.wake_diagnostic import validate_wake_diagnostic_trials
from src.core.backprop_mlp import NUMPY_BACKPROP_LOSS_ID
from src.core.predictive_coding import NUMPY_PC_ENERGY_ID
from src.core.circadian_predictive_coding import NUMPY_CIRCADIAN_ENERGY_ID


def test_should_capture_one_real_metric_per_successful_v14_wake_update() -> None:
    study = run_trigger_replay_training_study(
        fixed_trigger_replay_manifest(), capture_wake_diagnostics=True
    )
    validate_wake_diagnostic_trials(study.trials)
    assert len(study.trials) == 6
    assert sum(len(trial.wake_diagnostics) for trial in study.trials) == 432
    first = study.trials[0].wake_diagnostics[:3]
    assert [
        (row.seed, row.arm, row.phase, row.epoch, row.global_epoch, row.method) for row in first
    ] == [
        (47, "periodic", "a", 1, 1, "backprop"),
        (47, "periodic", "a", 1, 1, "predictive_coding"),
        (47, "periodic", "a", 1, 1, "circadian_predictive_coding"),
    ]
    assert [row.metric_name for row in first] == ["loss", "energy", "energy"]
    assert [row.metric_definition for row in first] == [
        NUMPY_BACKPROP_LOSS_ID,
        NUMPY_PC_ENERGY_ID,
        NUMPY_CIRCADIAN_ENERGY_ID,
    ]
    assert [row.measurement_stage for row in first] == [
        "pre_parameter_update",
        "post_inference_pre_parameter_update",
        "post_inference_pre_parameter_update",
    ]
    assert all(
        isfinite(row.metric_value) for trial in study.trials for row in trial.wake_diagnostics
    )
    assert all(
        access.role != "final_test"
        for trial in study.trials
        for access in trial.pending.audit.accesses
    )


def test_should_leave_v14_training_and_scored_bytes_unchanged_when_capture_enabled() -> None:
    manifest = fixed_trigger_replay_manifest()
    baseline = run_trigger_replay_training_study(manifest)
    measured = run_trigger_replay_training_study(manifest, capture_wake_diagnostics=True)
    for old, new in zip(baseline.trials, measured.trials, strict=True):
        assert old.wake_work == new.wake_work
        assert old.pending.audit.accesses == new.pending.audit.accesses
        assert not old.wake_diagnostics
        assert len(new.wake_diagnostics) == 3 * len(new.opportunities)
        for name in ("backprop_model", "predictive_model", "circadian_model"):
            assert _parameter_digest(getattr(old.pending.state, name)) == _parameter_digest(
                getattr(new.pending.state, name)
            )
    assert serialize_training_study(baseline) == serialize_training_study(measured)
    assert sha256(serialize_training_study(measured).encode()).hexdigest() == (
        "174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324"
    )
    scored = serialize_outcome_comparison(score_trigger_replay_training_study(measured)).encode()
    assert sha256(scored).hexdigest() == (
        "ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f"
    )


def test_should_reject_nonfinite_or_misplaced_wake_diagnostic() -> None:
    study = run_trigger_replay_training_study(
        fixed_trigger_replay_manifest(), capture_wake_diagnostics=True
    )
    first_trial = study.trials[0]
    first = first_trial.wake_diagnostics[0]
    bad_value = replace(first, metric_value=float("nan"))
    with pytest.raises(ValueError, match="diagnostic"):
        validate_wake_diagnostic_trials(
            (replace(first_trial, wake_diagnostics=(bad_value, *first_trial.wake_diagnostics[1:])),)
        )
    bad_role = replace(first, train_role_hash="0" * 64)
    with pytest.raises(ValueError, match="diagnostic"):
        validate_wake_diagnostic_trials(
            (replace(first_trial, wake_diagnostics=(bad_role, *first_trial.wake_diagnostics[1:])),)
        )
