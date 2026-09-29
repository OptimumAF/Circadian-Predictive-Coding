"""Fixed two-policy replay side-effect comparison keeps matched controls."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from scripts.run_continual_matched_replay_outcomes import _fixed_manifest
from src.app import continual_replay_side_effect_ablation as ablation
from src.app.continual_matched_replay_runner import (
    MATCHED_REPLAY_TRAINING_PROTOCOL,
    SIDE_EFFECT_TRAINING_PROTOCOL,
    run_matched_replay_training,
)
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.replay_retention import ReplayRetentionPolicy


def test_should_keep_historical_identity_and_score_all_eight_matched_trials() -> None:
    manifest = _fixed_manifest()
    historical = run_matched_replay_training(
        ablation.outcomes._schedule_manifest(manifest, manifest.policies[0]), seed=17
    )
    assert historical.protocol_id == MATCHED_REPLAY_TRAINING_PROTOCOL
    assert historical.pending.state.circadian_model.get_replay_side_effect_policy() == "historical"

    result = ablation.run_replay_side_effect_ablation(
        ablation.ReplaySideEffectAblationManifest(manifest)
    )
    assert result.protocol_id == ablation.SIDE_EFFECT_ABLATION_PROTOCOL
    assert [item.side_effect_policy for item in result.outcomes] == [
        "historical",
        "wake_only_adaptive_v1",
    ]
    for side in result.outcomes:
        assert [retention.policy.name for retention in side.retention] == [
            "recent_fifo",
            "seeded_reservoir",
        ]
        for retention in side.retention:
            assert [seed.arrived.seed for seed in retention.seeds] == [17, 19]
            for seed in retention.seeds:
                assert len(seed.boundaries) == 4
                assert all(boundary.sleep_outcome == "accepted" for boundary in seed.boundaries)
                assert all(work.optimizer_updates == 8 for work in seed.applied_work)
                assert all(
                    event.role != "final_test"
                    or event.action in {"source_release", "label_release"}
                    for event in seed.arrived.role_accesses
                )
    for old, new in zip(result.outcomes[0].retention, result.outcomes[1].retention, strict=True):
        for old_seed, new_seed in zip(old.seeds, new.seeds, strict=True):
            assert old_seed.boundaries == new_seed.boundaries
            assert old_seed.arrived.role_ids == new_seed.arrived.role_ids
            assert old_seed.arrived.role_hashes == new_seed.arrived.role_hashes
            assert old_seed.arrived.metrics.backprop == new_seed.arrived.metrics.backprop
            assert (
                old_seed.arrived.metrics.predictive_coding
                == new_seed.arrived.metrics.predictive_coding
            )


def test_should_release_final_sources_only_after_all_trials_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained = 0
    releases = 0
    original_train = ablation.run_matched_replay_training
    original_release = ablation.release_final_test

    def record_train(*args: Any, **kwargs: Any) -> Any:
        nonlocal trained
        result = original_train(*args, **kwargs)
        trained += 1
        return result

    def record_release(*args: Any, **kwargs: Any) -> Any:
        nonlocal releases
        assert trained == 8
        releases += 1
        return original_release(*args, **kwargs)

    monkeypatch.setattr(ablation, "run_matched_replay_training", record_train)
    monkeypatch.setattr(ablation, "release_final_test", record_release)
    ablation.run_replay_side_effect_ablation(
        ablation.ReplaySideEffectAblationManifest(_fixed_manifest())
    )
    assert trained == 8
    assert releases == 16


def test_should_reject_unmatched_replay_work_before_final_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_train = ablation.run_matched_replay_training

    def forge(*args: Any, **kwargs: Any) -> Any:
        trial = original_train(*args, **kwargs)
        if kwargs.get("side_effect_policy") != "wake_only_adaptive_v1":
            return trial
        boundary = trial.boundaries[0]
        changed = replace(boundary, selected_ids=("0" * 64, *boundary.selected_ids[1:]))
        return replace(trial, boundaries=(changed, *trial.boundaries[1:]))

    monkeypatch.setattr(ablation, "run_matched_replay_training", forge)
    monkeypatch.setattr(
        ablation,
        "release_final_test",
        lambda *_: (_ for _ in ()).throw(AssertionError("final source opened")),
    )
    with pytest.raises(ValueError, match="boundary differs"):
        ablation.run_replay_side_effect_ablation(
            ablation.ReplaySideEffectAblationManifest(_fixed_manifest())
        )


def test_should_commit_no_replay_work_after_guard_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = iter([1.0, 0.0] * 4)
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork, "compute_accuracy", lambda *_, **__: next(scores)
    )
    manifest = ablation.outcomes._schedule_manifest(
        _fixed_manifest(), ReplayRetentionPolicy("recent_fifo")
    )
    trial = run_matched_replay_training(
        manifest, seed=17, side_effect_policy="wake_only_adaptive_v1"
    )
    assert trial.protocol_id == SIDE_EFFECT_TRAINING_PROTOCOL
    assert all(boundary.sleep_outcome == "rolled_back" for boundary in trial.boundaries)
    assert all(
        work.optimizer_updates == 0
        for boundary in trial.boundaries
        for work in boundary.applied_by_method
    )
    assert trial.pending.state.circadian_model.get_sleep_clocks().replay_updates == 0


def test_should_restore_opt_in_model_after_sleep_core_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ablation.outcomes._schedule_manifest(
        _fixed_manifest(), ReplayRetentionPolicy("recent_fifo")
    )
    original_sleep = CircadianPredictiveCodingNetwork.sleep_event
    captured: dict[str, Any] = {}

    def fail_after_sleep(self: CircadianPredictiveCodingNetwork, *args: Any, **kwargs: Any) -> Any:
        captured["model"] = self
        captured["before"] = self.snapshot_state()
        original_sleep(self, *args, **kwargs)
        raise RuntimeError("injected after sleep")

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "sleep_event", fail_after_sleep)
    with pytest.raises(RuntimeError, match="injected after sleep"):
        run_matched_replay_training(manifest, seed=17, side_effect_policy="wake_only_adaptive_v1")
    model = captured["model"]
    before = captured["before"]
    assert model._epoch_count == before.state["_epoch_count"]
    assert model._replay_updates == before.state["_replay_updates"]
    assert model._sleep_events == before.state["_sleep_events"]
    assert model.get_replay_exposure().replay_updates == before.state["_replay_exposure_updates"]
    assert model.get_replay_side_effect_policy() == "wake_only_adaptive_v1"
