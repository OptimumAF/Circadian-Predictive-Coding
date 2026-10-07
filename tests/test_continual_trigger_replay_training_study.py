"""All fixed v14 guarded trials preflight before any final-role release."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

from scripts.run_continual_trigger_replay_training import build_payload, main
from src.app.continual_trigger_replay_runner import run_trigger_replay_training
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.app.continual_trigger_replay_training_study import (
    _validate_matched_seed,
    _validate_trial,
    run_trigger_replay_training_study,
)


def test_should_preflight_all_six_unscored_trials() -> None:
    manifest = fixed_trigger_replay_manifest()
    study = run_trigger_replay_training_study(manifest)
    assert [(trial.seed, trial.arm) for trial in study.trials] == [
        (47, "periodic"),
        (47, "adaptive"),
        (47, "no_sleep"),
        (53, "periodic"),
        (53, "adaptive"),
        (53, "no_sleep"),
    ]
    assert all(
        not trial.pending.phase_a.final_released and not trial.pending.phase_b.final_released
        for trial in study.trials
    )
    assert [
        sum(item.event.outcome == "accepted" for item in trial.opportunities)
        for trial in study.trials
    ] == [6, 0, 0, 6, 0, 0]
    assert [
        sum(
            work.optimizer_updates
            for item in trial.opportunities
            for work in item.applied_by_method
        )
        for trial in study.trials
    ] == [36, 0, 0, 36, 0, 0]
    assert [trial.pending.state.total_prunes for trial in study.trials] == [2, 0, 0, 3, 0, 0]


@pytest.mark.parametrize(
    "change", ["selection", "work", "capacity", "initial", "event", "guard", "structure_id"]
)
def test_should_reject_forged_train_only_facts(change: str) -> None:
    manifest = fixed_trigger_replay_manifest()
    trial = run_trigger_replay_training(manifest, seed=47, arm="periodic")
    opportunities = list(trial.opportunities)
    if change == "selection":
        opportunities[0] = replace(
            opportunities[0], selected_ids=("f" * 64, *opportunities[0].selected_ids[1:])
        )
    elif change == "work":
        target = opportunities[3]
        work = replace(target.applied_by_method[0], optimizer_updates=1)
        opportunities[3] = replace(target, applied_by_method=(work, *target.applied_by_method[1:]))
    elif change == "capacity":
        opportunities[3] = replace(opportunities[3], cumulative_prunes=7)
    elif change == "initial":
        trial = replace(
            trial,
            initial_parameter_digests=(
                ("backprop", "f" * 64),
                *trial.initial_parameter_digests[1:],
            ),
        )
    elif change == "event":
        target = opportunities[0]
        event = replace(target.event, trigger_reason="adaptive")
        opportunities[0] = replace(target, event=event)
        events = list(trial.pending.sleep_events)
        events[0] = event
        trial.pending.sleep_events = tuple(events)
    elif change == "guard":
        target = opportunities[3]
        assert target.event.guard is not None
        event = replace(target.event, guard=replace(target.event.guard, role_hash="f" * 64))
        opportunities[3] = replace(target, event=event)
        events = list(trial.pending.sleep_events)
        events[3] = event
        trial.pending.sleep_events = tuple(events)
    else:
        target = opportunities[11]
        changes = replace(
            target.event.changes,
            proposed_prune_ids=(3,),
            proposed_removed_prune_ids=(3,),
            applied_removed_prune_ids=(3,),
        )
        event = replace(target.event, changes=changes)
        opportunities[11] = replace(target, event=event)
        events = list(trial.pending.sleep_events)
        events[11] = event
        trial.pending.sleep_events = tuple(events)
    trial = replace(trial, opportunities=tuple(opportunities))
    with pytest.raises(ValueError, match="v14 train-only|v14 periodic|v14 structural"):
        _validate_trial(manifest, trial)


def test_should_reject_mismatched_arm_source_or_initial_state() -> None:
    manifest = fixed_trigger_replay_manifest()
    trials = tuple(run_trigger_replay_training(manifest, seed=47, arm=arm) for arm in manifest.arms)
    changed = replace(
        trials[1],
        initial_parameter_digests=(
            ("backprop", "f" * 64),
            *trials[1].initial_parameter_digests[1:],
        ),
    )
    with pytest.raises(ValueError, match="unmatched source or initial work"):
        _validate_matched_seed((trials[0], changed, trials[2]))


def test_should_repeat_exact_train_only_payload_without_durations_or_final_scores() -> None:
    first = build_payload()
    assert first == build_payload()
    parsed = json.loads(first)
    assert len(parsed["rows"]) == 6
    assert all(len(row["opportunities"]) == 24 for row in parsed["rows"])
    assert all(
        "durations" not in item["event"] and "final_test" not in item and "accuracy" not in row
        for row in parsed["rows"]
        for item in row["opportunities"]
    )


def test_should_hash_actual_file_bytes_and_refuse_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = tmp_path / "train-only.json"
    monkeypatch.setattr(sys, "argv", ["training", "--result", str(result)])
    main()
    reported = json.loads(capsys.readouterr().out)
    assert result.read_bytes() == build_payload().encode("utf-8")
    assert reported["sha256"] == sha256(result.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        main()
