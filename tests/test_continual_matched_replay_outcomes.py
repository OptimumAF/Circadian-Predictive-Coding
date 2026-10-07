"""All fixed v9 trials must freeze before one matched final release."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import run_continual_matched_replay_outcomes as outcome_script
from scripts.run_continual_matched_replay_schedule_smoke import _manifest
from src.app import continual_arrived_benchmark as arrived
from src.app import continual_matched_replay_outcomes as outcomes
from src.app import continual_shift_benchmark as base
from src.core.replay_retention import ReplayRetentionPolicy
from src.infra import continual_roles


def _outcome_manifest(reverse_order: bool = False) -> outcomes.MatchedReplayOutcomeManifest:
    fifo = ReplayRetentionPolicy("recent_fifo")
    schedule = _manifest(fifo)
    training = schedule.arrived.training
    if reverse_order:
        training = replace(training, model_order=tuple(reversed(training.model_order)))
    return outcomes.MatchedReplayOutcomeManifest(
        arrived=replace(schedule.arrived, training=training),
        seeds=schedule.seeds,
        policies=(fifo, ReplayRetentionPolicy("seeded_reservoir", 53)),
        replay_updates_per_sleep=schedule.replay_updates_per_sleep,
        pc_replay_inference_steps=schedule.pc_replay_inference_steps,
    )


def test_should_reject_bad_outcome_manifest_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _outcome_manifest()
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **_: (_ for _ in ()).throw(AssertionError("source opened")),
    )
    with pytest.raises(ValueError, match="matched outcome manifest"):
        outcomes.run_matched_replay_outcomes(replace(manifest, seeds=(17, 17)))
    with pytest.raises(ValueError, match="matched outcome manifest"):
        outcomes.run_matched_replay_outcomes(
            replace(manifest, policies=(manifest.policies[0], manifest.policies[0]))
        )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_should_freeze_all_trials_before_final_access_and_score_matched_roles(
    reverse_order: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _outcome_manifest(reverse_order)
    original_a = arrived.generate_two_cluster_dataset_with_transform
    original_b = arrived._generate_phase_b_source
    original_train = outcomes.run_matched_replay_training
    finished: set[tuple[str, int]] = set()
    final_reads: list[str] = []

    class SealedSource:
        def __init__(self, source: Any) -> None:
            self.source = source
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            assert len(finished) == 4
            final_reads.append("input")
            return self.source.test_input

        @property
        def test_target(self) -> Any:
            assert len(finished) == 4
            final_reads.append("target")
            return self.source.test_target

    def train(*args: Any, **kwargs: Any) -> Any:
        result = original_train(*args, **kwargs)
        finished.add((args[0].policy.name, result.seed))
        return result

    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **kwargs: SealedSource(original_a(**kwargs)),
    )
    monkeypatch.setattr(
        arrived,
        "_generate_phase_b_source",
        lambda *args, **kwargs: SealedSource(original_b(*args, **kwargs)),
    )
    monkeypatch.setattr(outcomes, "run_matched_replay_training", train)
    result = outcomes.run_matched_replay_outcomes(manifest)

    assert result.protocol_id == outcomes.MATCHED_REPLAY_OUTCOME_PROTOCOL
    assert len(result.policies) == 2
    assert [item.policy for item in result.policies] == list(manifest.policies)
    assert all([seed.arrived.seed for seed in item.seeds] == [17, 19] for item in result.policies)
    assert len(finished) == 4
    assert final_reads == ["input", "target"] * 8
    for first, second in zip(result.policies[0].seeds, result.policies[1].seeds, strict=True):
        assert first.arrived.role_ids == second.arrived.role_ids
        assert first.arrived.role_hashes == second.arrived.role_hashes
        assert first.arrived.metrics.training_order == manifest.arrived.training.model_order
        assert first.arrived.metrics.backprop.balanced_score >= 0.0
        assert first.arrived.metrics.predictive_coding.balanced_score >= 0.0
        assert first.arrived.metrics.circadian_predictive_coding.balanced_score >= 0.0
        for item in (first, second):
            assert len(item.boundaries) == 4
            assert {work.optimizer_updates for work in item.applied_work} == {8}
            assert {work.inference_iterations for work in item.applied_work} == {0, 16, 24}
            assert all(
                event.event == "global_freeze"
                for event in item.arrived.role_accesses
                if event.role == "final_test"
            )


def test_should_reject_forged_work_before_final_release(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _outcome_manifest()
    original_train = outcomes.run_matched_replay_training

    def forged_train(*args: Any, **kwargs: Any) -> Any:
        result = original_train(*args, **kwargs)
        if args[0].policy.name != "recent_fifo" or result.seed != 17:
            return result
        first = result.boundaries[0]
        work = replace(first.applied_by_method[0], optimizer_updates=99)
        return replace(
            result,
            boundaries=(
                replace(first, applied_by_method=(work, *first.applied_by_method[1:])),
                *result.boundaries[1:],
            ),
        )

    monkeypatch.setattr(outcomes, "run_matched_replay_training", forged_train)
    monkeypatch.setattr(
        outcomes,
        "release_final_test",
        lambda *_: (_ for _ in ()).throw(AssertionError("final opened before work preflight")),
    )
    with pytest.raises(ValueError, match="applied replay work"):
        outcomes.run_matched_replay_outcomes(manifest)


@pytest.mark.parametrize("field", ["retained_order_ids", "selected_ids"])
def test_should_reject_changed_schedule_ids_before_final_release(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _outcome_manifest()
    original_train = outcomes.run_matched_replay_training

    def forged_train(*args: Any, **kwargs: Any) -> Any:
        result = original_train(*args, **kwargs)
        if args[0].policy.name != "recent_fifo" or result.seed != 17:
            return result
        first = result.boundaries[0]
        forged = (
            replace(first, selected_ids=("f" * 64,))
            if field == "selected_ids"
            else replace(first, retained_order_ids=("f" * 64,))
        )
        return replace(result, boundaries=(forged, *result.boundaries[1:]))

    monkeypatch.setattr(outcomes, "run_matched_replay_training", forged_train)
    monkeypatch.setattr(
        outcomes,
        "release_final_test",
        lambda *_: (_ for _ in ()).throw(AssertionError("final opened before schedule preflight")),
    )
    with pytest.raises(ValueError, match="retained or selected boundary"):
        outcomes.run_matched_replay_outcomes(manifest)


def test_should_reject_changed_arrived_role_before_final_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _outcome_manifest()
    original_train = outcomes.run_matched_replay_training

    def forged_train(*args: Any, **kwargs: Any) -> Any:
        result = original_train(*args, **kwargs)
        if args[0].policy.name == "recent_fifo" and result.seed == 17:
            result.pending.phase_a = replace(
                result.pending.phase_a,
                split_hashes={**result.pending.phase_a.split_hashes, "train": "f" * 64},
            )
        return result

    monkeypatch.setattr(outcomes, "run_matched_replay_training", forged_train)
    monkeypatch.setattr(
        outcomes,
        "release_final_test",
        lambda *_: (_ for _ in ()).throw(AssertionError("final opened before role preflight")),
    )
    with pytest.raises(ValueError, match="arrived development role"):
        outcomes.run_matched_replay_outcomes(manifest)


def test_should_reject_forged_circadian_exposure_before_final_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _outcome_manifest()
    original_train = outcomes.run_matched_replay_training

    def forged_train(*args: Any, **kwargs: Any) -> Any:
        result = original_train(*args, **kwargs)
        if args[0].policy.name == "recent_fifo" and result.seed == 17:
            result.pending.state.circadian_model._replay_exposed_ids.clear()
        return result

    monkeypatch.setattr(outcomes, "run_matched_replay_training", forged_train)
    monkeypatch.setattr(
        outcomes,
        "release_final_test",
        lambda *_: (_ for _ in ()).throw(AssertionError("final opened before exposure preflight")),
    )
    with pytest.raises(ValueError, match="retained state or work clock"):
        outcomes.run_matched_replay_outcomes(manifest)


def test_should_reject_different_final_roles_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _outcome_manifest()
    original_release = continual_roles.release_final_test
    released = 0

    def changed_release(roles: Any) -> Any:
        nonlocal released
        bound = original_release(roles)
        released += 1
        if released == 5:
            return replace(bound, split_hashes={**bound.split_hashes, "final_test": "f" * 64})
        return bound

    monkeypatch.setattr(outcomes, "release_final_test", changed_release)
    monkeypatch.setattr(
        base,
        "_score_seed_models",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("scored before role match")),
    )
    with pytest.raises(ValueError, match="matched final roles"):
        outcomes.run_matched_replay_outcomes(manifest)


def test_should_isolate_final_labels_from_training_and_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _outcome_manifest()
    control = outcomes.run_matched_replay_outcomes(manifest)
    original_a = arrived.generate_two_cluster_dataset_with_transform

    def changed_final(**kwargs: Any) -> Any:
        source = original_a(**kwargs)
        if kwargs["seed"] == 17:
            return replace(source, test_target=1.0 - source.test_target)
        return source

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", changed_final)
    changed = outcomes.run_matched_replay_outcomes(manifest)
    assert control.manifest_digest == changed.manifest_digest
    for before_policy, after_policy in zip(control.policies, changed.policies, strict=True):
        for before, after in zip(before_policy.seeds, after_policy.seeds, strict=True):
            assert before.boundaries == after.boundaries
            assert before.applied_work == after.applied_work
            for phase in ("a", "b"):
                for role in ("train", "inner_guard", "outer_selection"):
                    key = f"phase_{phase}_{role}"
                    assert before.arrived.role_hashes[key] == after.arrived.role_hashes[key]
            if before.arrived.seed == 17:
                assert (
                    before.arrived.role_hashes["phase_a_final_test"]
                    != after.arrived.role_hashes["phase_a_final_test"]
                )
                assert (
                    before.arrived.metrics.backprop.phase_a_post_accuracy
                    != after.arrived.metrics.backprop.phase_a_post_accuracy
                )
            else:
                assert before.arrived.metrics.backprop == after.arrived.metrics.backprop


def test_should_write_repeatable_all_trial_outcome_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for path in (first, second):
        monkeypatch.setattr(
            sys, "argv", ["run_continual_matched_replay_outcomes", "--result", str(path)]
        )
        outcome_script.main()
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["protocol_id"] == outcomes.MATCHED_REPLAY_OUTCOME_PROTOCOL
    assert len(payload["policies"]) == 2
    assert all(
        [seed["seed"] for seed in policy["seeds"]] == [17, 19] for policy in payload["policies"]
    )
    for policy in payload["policies"]:
        for seed in policy["seeds"]:
            assert all("durations" not in event for event in seed["sleep_events_without_durations"])
            assert set(seed["metrics"]) == {
                "backprop",
                "predictive_coding",
                "circadian_predictive_coding",
                "replay_retention",
            }
            assert {work["optimizer_updates"] for work in seed["applied_work"]} == {8}
