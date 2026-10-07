"""Format-10 v14 prefix gates keep every final role sealed."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.app.continual_trigger_replay_runner import run_trigger_replay_training
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.app.v14_trial_checkpoint import (
    V14TrialPrefixCheckpoint,
    build_v14_trial_checkpoint,
    study_from_v14_trial_checkpoint,
    validate_v14_trial_checkpoint,
)


@pytest.fixture(scope="module")
def first_checkpoint() -> V14TrialPrefixCheckpoint:
    manifest = fixed_trigger_replay_manifest()
    first = run_trigger_replay_training(manifest, seed=47, arm="periodic")
    return build_v14_trial_checkpoint(manifest, "a" * 64, False, (first,))


def test_should_keep_next_cell_and_final_roles_sealed(
    first_checkpoint: V14TrialPrefixCheckpoint,
) -> None:
    manifest = fixed_trigger_replay_manifest()
    assert first_checkpoint.next_trial_index == 1
    assert first_checkpoint.next_cell == (47, "adaptive")
    assert first_checkpoint.trials[0].pending.phase_a._source is None
    assert first_checkpoint.trials[0].pending.phase_b._source is None
    assert all(
        access.role != "final_test" for access in first_checkpoint.trials[0].pending.audit.accesses
    )
    with pytest.raises(ValueError, match="all unscored trials"):
        study_from_v14_trial_checkpoint(first_checkpoint, manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("format_version", 9),
        ("source_workspace_sha256", "b" * 64),
        ("config_sha256", "b" * 64),
        ("protocol_sha256", "b" * 64),
        ("capture_wake_diagnostics", True),
        ("next_trial_index", 2),
        ("next_cell", (53, "periodic")),
    ],
)
def test_should_reject_changed_prefix_identity(
    first_checkpoint: V14TrialPrefixCheckpoint, field: str, value: object
) -> None:
    manifest = fixed_trigger_replay_manifest()
    changed = replace(first_checkpoint, **{field: value})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="v14 checkpoint"):
        validate_v14_trial_checkpoint(changed, manifest, "a" * 64, False)


def test_should_reject_forged_role_and_work_in_stored_trial(
    first_checkpoint: V14TrialPrefixCheckpoint,
) -> None:
    manifest = fixed_trigger_replay_manifest()
    trial = first_checkpoint.trials[0]
    event = trial.pending.audit.accesses[0]
    trial.pending.audit.accesses[0] = replace(event, role="final_test")
    with pytest.raises(ValueError, match="final seal"):
        validate_v14_trial_checkpoint(first_checkpoint, manifest, "a" * 64, False)
