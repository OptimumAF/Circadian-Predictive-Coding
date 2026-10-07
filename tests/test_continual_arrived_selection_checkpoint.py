"""A v7 selection checkpoint must bind every candidate before final release."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_arrived_selection as selection
from src.app import continual_shift_benchmark as base
from src.app.continual_arrived_checkpoint import (
    arrived_event_digest,
    arrived_sleep_history_digest,
)
from src.app.continual_arrived_benchmark import ContinualArrivedRolesConfig
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
)
from src.infra.circadian_checkpoint_files import (
    TrustedLocalArrivedCheckpointStore,
    TrustedLocalArrivedSelectionCheckpointStore,
)
from src.infra import continual_roles
from src.infra.datasets import LabeledData


class SelectionInterruption(Exception):
    """Stop immediately after a durable candidate or freeze transaction."""


class StopAtSelectionCheckpoint:
    def __init__(self, path: Path, boundary: str) -> None:
        self.store = TrustedLocalArrivedSelectionCheckpointStore(path)
        self.boundary = boundary

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        active = checkpoint.active_v6
        matching = {
            "first_wake": checkpoint.candidate_index == 0
            and active is not None
            and active.seed_index == 0
            and active.phase == "a"
            and active.stage == "wake",
            "first_b_wake": checkpoint.candidate_index == 0
            and active is not None
            and active.seed_index == 0
            and active.phase == "b"
            and active.stage == "wake",
            "candidate_boundary": checkpoint.candidate_index == 1
            and checkpoint.stage == "training"
            and active is None,
            "later_candidate_wake": checkpoint.candidate_index == 1
            and active is not None
            and active.seed_index == 0
            and active.phase == "a"
            and active.stage == "wake",
            "later_b_after_sleep": checkpoint.candidate_index == 1
            and active is not None
            and active.seed_index == 0
            and active.phase == "b"
            and active.stage == "after_sleep"
            and active.phase_epoch_completed == 1,
            "frozen": checkpoint.stage == "frozen",
        }
        if matching[self.boundary]:
            raise SelectionInterruption()


def _assert_same_models(actual: Any, expected: Any) -> None:
    left = actual.state if hasattr(actual, "state") else actual.__dict__
    right = expected.state if hasattr(expected, "state") else expected.__dict__
    assert left.keys() == right.keys()
    for name in left:
        if name == "_replay_memory":
            assert len(left[name]) == len(right[name])
            for first, second in zip(left[name], right[name], strict=True):
                np.testing.assert_array_equal(first.input_batch, second.input_batch)
                np.testing.assert_array_equal(first.target_batch, second.target_batch)
                assert first.priority == second.priority
                assert first.positive_fraction == second.positive_fraction
        else:
            assert pickle.dumps(left[name], protocol=5) == pickle.dumps(right[name], protocol=5)


def _assert_same_completed_candidates(actual: Any, expected: Any) -> None:
    assert len(actual) == len(expected)
    for first, second in zip(actual, expected, strict=True):
        assert first.candidate_id == second.candidate_id
        assert first.trials == second.trials
        assert first.trial_digest == second.trial_digest
        for first_seed, second_seed in zip(
            first.unscored_seeds, second.unscored_seeds, strict=True
        ):
            assert first_seed.seed == second_seed.seed
            assert first_seed.role_ids == second_seed.role_ids
            assert first_seed.role_hashes == second_seed.role_hashes
            assert first_seed.role_accesses == second_seed.role_accesses
            assert first_seed.guard_decisions == second_seed.guard_decisions
            assert first_seed.method_task_information == second_seed.method_task_information
            for field in fields(first_seed.state):
                left = getattr(first_seed.state, field.name)
                right = getattr(second_seed.state, field.name)
                if field.name.startswith(("backprop", "predictive", "circadian")):
                    _assert_same_models(left, right)
                else:
                    assert left == right


def _candidates(reverse_order: bool) -> tuple[selection.ArrivedSelectionCandidate, ...]:
    training = ContinualGlobalSealConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=1,
        phase_b_epochs=1,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=1,
        circadian_sleep_interval_phase_b=1,
        circadian_config=CircadianConfig(
            sleep_mode="components",
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            replay_steps=1,
            replay_memory_size=1,
        ),
        replay_max_examples=4,
        replay_max_bytes=96,
    )
    if reverse_order:
        training = replace(training, model_order=tuple(reversed(training.model_order)))
    first = ContinualArrivedRolesConfig(training, 0.2, 0.2)
    second = replace(
        first,
        training=replace(
            training,
            backprop_learning_rate=training.backprop_learning_rate * 0.7,
            pc_learning_rate=training.pc_learning_rate * 0.7,
            circadian_learning_rate=training.circadian_learning_rate * 0.7,
        ),
    )
    return (
        selection.ArrivedSelectionCandidate("default", first),
        selection.ArrivedSelectionCandidate("lower_rate", second),
    )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_selection_exposes_candidate_and_chosen_sleep_history(
    reverse_order: bool, tmp_path: Path
) -> None:
    candidates = _candidates(reverse_order)
    ordinary = selection.run_arrived_outer_selection(candidates, [17, 19])
    store = TrustedLocalArrivedSelectionCheckpointStore(tmp_path / "sleep-history.ckpt")
    checkpointed = selection.run_arrived_outer_selection(
        candidates, [17, 19], checkpoint_store=store
    )
    interrupted = StopAtSelectionCheckpoint(
        tmp_path / "interrupted-history.ckpt", "candidate_boundary"
    )
    with pytest.raises(SelectionInterruption):
        selection.run_arrived_outer_selection(candidates, [17, 19], checkpoint_store=interrupted)
    assert all(
        saved.sleep_events for saved in interrupted.load().completed_candidates[0].unscored_seeds
    )
    resumed = selection.run_arrived_outer_selection(
        candidates, [17, 19], checkpoint_store=interrupted.store, resume_from_checkpoint=True
    )

    for result in (ordinary, checkpointed, resumed):
        histories = {
            (item.candidate_id, item.seed): item.sleep_events
            for item in result.candidate_sleep_histories
        }
        assert set(histories) == {
            (candidate.candidate_id, seed) for candidate in candidates for seed in (17, 19)
        }
        for (candidate_id, seed), events in histories.items():
            assert len(events) == 2, (candidate_id, seed)
            assert [event.completed_epoch for event in events] == [1, 2]
            assert [event.guard.role for event in events if event.guard is not None] == [
                "inner_guard",
                "inner_guard",
            ]
            assert all(event.outcome in {"accepted", "rolled_back"} for event in events)
            assert all(
                event.durations.attempt_seconds >= event.durations.core_seconds for event in events
            )
        choice = next(
            item for item in result.selections if item.method == "circadian_predictive_coding"
        )
        for final in result.final_seed_results:
            assert (
                final.metrics.circadian_predictive_coding.sleep_events
                == histories[choice.candidate_id, final.seed]
            )
        json.dumps([asdict(item) for item in result.candidate_sleep_histories], allow_nan=False)
        assert result.freeze.trial_digest == selection._trial_digest(result.trials)

    def without_durations(result: selection.ArrivedOuterSelectionResult) -> list[dict[str, Any]]:
        rows = [asdict(item) for item in result.candidate_sleep_histories]
        for row in rows:
            for event in row["sleep_events"]:
                event.pop("durations")
        return rows

    assert without_durations(ordinary) == without_durations(checkpointed)
    assert without_durations(ordinary) == without_durations(resumed)
    assert ordinary == checkpointed == resumed


@pytest.mark.parametrize("reverse_order", [False, True])
def test_selection_binds_each_trial_history_to_frozen_checkpoint(
    reverse_order: bool, tmp_path: Path
) -> None:
    candidates = _candidates(reverse_order)
    ordinary = selection.run_arrived_outer_selection(candidates, [17, 19])
    store = TrustedLocalArrivedSelectionCheckpointStore(tmp_path / "history-bound.ckpt")
    checkpointed = selection.run_arrived_outer_selection(
        candidates, [17, 19], checkpoint_store=store
    )

    for result in (ordinary, checkpointed):
        histories = {
            (item.candidate_id, item.seed): item.sleep_events
            for item in result.candidate_sleep_histories
        }
        circadian_trials = [
            item for item in result.trials if item.method == "circadian_predictive_coding"
        ]
        assert len(circadian_trials) == 4
        for trial in result.trials:
            expected = histories[trial.candidate_id, trial.seed]
            if trial.method == "circadian_predictive_coding":
                assert trial.sleep_events == expected
                assert len(trial.sleep_history_digest) == 64
            else:
                assert trial.sleep_events == ()
                assert trial.sleep_history_digest == ""
        old_trial_rows = []
        for trial in result.trials:
            row = asdict(trial)
            row.pop("sleep_events")
            row.pop("sleep_history_digest")
            old_trial_rows.append(row)
        assert result.freeze.trial_digest == selection._hash_json(old_trial_rows)
        assert len(result.freeze.sleep_history_digest) == 64
        json.dumps([asdict(item) for item in result.trials], allow_nan=False)

    frozen = store.load()
    assert frozen.stage == "frozen"
    assert frozen.freeze is not None
    assert frozen.freeze.sleep_history_digest == checkpointed.freeze.sleep_history_digest
    assert all(len(record.sleep_history_digest) == 64 for record in frozen.completed_candidates)
    assert all(
        item.sleep_events
        for record in frozen.completed_candidates
        for item in record.trials
        if item.method == "circadian_predictive_coding"
    )


def _scheduled_candidates(reverse_order: bool) -> tuple[selection.ArrivedSelectionCandidate, ...]:
    return tuple(
        replace(
            item,
            config=replace(
                item.config,
                training=replace(
                    item.config.training,
                    phase_a_epochs=2,
                    phase_b_epochs=2,
                    circadian_sleep_interval_phase_a=2,
                    circadian_sleep_interval_phase_b=2,
                ),
            ),
        )
        for item in _candidates(reverse_order)
    )


def _inject_one_error_rejection_and_acceptance(
    original_sleep: Any, patch: pytest.MonkeyPatch
) -> Any:
    stage = 0

    def fail_core(*_: Any, **__: Any) -> Any:
        raise RuntimeError("transient v7 sleep")

    def attempt(**kwargs: Any) -> tuple[int, int, int]:
        nonlocal stage
        epoch = kwargs["global_epoch"]
        if stage == 0 and epoch == 2:
            stage = 1
            with patch.context() as core_patch:
                core_patch.setattr(CircadianPredictiveCodingNetwork, "sleep_event", fail_core)
                return original_sleep(**kwargs)
        if stage in {1, 2} and epoch == (2 if stage == 1 else 4):
            scores = iter((0.8, 0.2) if stage == 1 else (0.8, 0.8))
            stage += 1
            with patch.context() as guard_patch:
                guard_patch.setattr(
                    CircadianPredictiveCodingNetwork,
                    "compute_accuracy",
                    lambda *_: next(scores),
                )
                return original_sleep(**kwargs)
        return original_sleep(**kwargs)

    return attempt


@pytest.mark.parametrize("reverse_order", [False, True])
def test_selection_reconciles_error_rejection_acceptance_and_skips_across_resume(
    reverse_order: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = _scheduled_candidates(reverse_order)
    original_sleep = base._apply_scheduled_sleep
    original_train = arrived._train_arrived_seed
    trained: dict[tuple[float, int], Any] = {}

    def capture(config: Any, seed: int, sleep_error_retries: int = 0) -> Any:
        row = original_train(config, seed, sleep_error_retries)
        trained[config.training.circadian_learning_rate, seed] = row
        return row

    with monkeypatch.context() as ordinary_patch:
        ordinary_patch.setattr(
            base,
            "_apply_scheduled_sleep",
            _inject_one_error_rejection_and_acceptance(original_sleep, ordinary_patch),
        )
        ordinary_patch.setattr(arrived, "_train_arrived_seed", capture)
        ordinary = selection.run_arrived_outer_selection(
            candidates, [17, 19], sleep_error_retries=1
        )

    store = TrustedLocalArrivedSelectionCheckpointStore(tmp_path / "v7-error.ckpt")
    original_release = selection.release_final_test

    def sealed_release(*args: Any, **kwargs: Any) -> Any:
        assert store.load().stage == "frozen"
        return original_release(*args, **kwargs)

    with monkeypatch.context() as checkpoint_patch:
        checkpoint_patch.setattr(
            base,
            "_apply_scheduled_sleep",
            _inject_one_error_rejection_and_acceptance(original_sleep, checkpoint_patch),
        )
        checkpoint_patch.setattr(selection, "release_final_test", sealed_release)
        with pytest.raises(RuntimeError, match="transient v7 sleep"):
            selection.run_arrived_outer_selection(candidates, [17, 19], checkpoint_store=store)
        active = store.load().active_v6
        assert active is not None and active.stage == "before_sleep"
        assert active.active_sleep_events[-1].outcome == "error"
        resumed = selection.run_arrived_outer_selection(
            candidates, [17, 19], checkpoint_store=store, resume_from_checkpoint=True
        )

    assert ordinary == resumed
    assert ordinary.freeze.trial_digest == resumed.freeze.trial_digest
    for first_history, second_history in zip(
        ordinary.candidate_sleep_histories,
        resumed.candidate_sleep_histories,
        strict=True,
    ):
        assert (first_history.candidate_id, first_history.seed) == (
            second_history.candidate_id,
            second_history.seed,
        )
        assert len(first_history.sleep_events) == len(second_history.sleep_events)
        for first_event, second_event in zip(
            first_history.sleep_events, second_history.sleep_events, strict=True
        ):
            first_facts = asdict(first_event)
            second_facts = asdict(second_event)
            first_facts.pop("durations")
            second_facts.pop("durations")
            assert first_facts == second_facts
    assert ordinary.freeze.sleep_history_digest == selection._sleep_history_digest(ordinary.trials)
    assert resumed.freeze.sleep_history_digest == selection._sleep_history_digest(resumed.trials)
    first = next(
        item
        for item in ordinary.trials
        if item.candidate_id == candidates[0].candidate_id
        and item.seed == 17
        and item.method == "circadian_predictive_coding"
    )
    assert [item.completed_epoch for item in first.sleep_events] == [1, 2, 2, 3, 4]
    assert [item.outcome for item in first.sleep_events] == [
        "skipped",
        "error",
        "rolled_back",
        "skipped",
        "accepted",
    ]
    assert len(first.guard_decisions) == 2
    assert first.sleep_event_count == 1
    assert first.inner_guard_examples_scored == sum(
        item.guard.examples_scored
        for item in first.sleep_events
        if item.outcome != "error" and item.guard is not None
    )
    assert first.sleep_events[1].guard is not None
    assert first.sleep_events[1].guard.examples_scored > 0
    for trial in ordinary.trials:
        if trial.method != "circadian_predictive_coding":
            assert trial.sleep_events == ()
            assert trial.sleep_event_count == trial.inner_guard_examples_scored == 0
            continue
        completed_guarded = [
            item
            for item in trial.sleep_events
            if item.guard is not None and item.outcome != "error"
        ]
        assert len(trial.guard_decisions) == len(completed_guarded)
        assert trial.inner_guard_examples_scored == sum(
            item.guard.examples_scored for item in completed_guarded if item.guard is not None
        )
        assert trial.sleep_event_count == sum(
            item.outcome == "accepted" for item in trial.sleep_events
        )
    json.dumps([asdict(item) for item in ordinary.trials], allow_nan=False)

    for record, candidate in zip(store.load().completed_candidates, candidates, strict=True):
        for saved in record.unscored_seeds:
            expected = trained[candidate.config.training.circadian_learning_rate, saved.seed]
            for field in fields(saved.state):
                left = getattr(saved.state, field.name)
                right = getattr(expected.state, field.name)
                if field.name.startswith(("backprop", "predictive", "circadian")):
                    _assert_same_models(left, right)
                else:
                    assert left == right


@pytest.mark.parametrize("retry_limit", [-1, True, 1.5])
def test_selection_rejects_invalid_ordinary_sleep_retry_limit(retry_limit: Any) -> None:
    with pytest.raises(ValueError, match="retries"):
        selection.run_arrived_outer_selection(
            _candidates(False), [17, 19], sleep_error_retries=retry_limit
        )


def test_selection_checkpoint_requires_explicit_sleep_error_resume(tmp_path: Path) -> None:
    path = tmp_path / "unused.ckpt"
    store = TrustedLocalArrivedSelectionCheckpointStore(path)
    with pytest.raises(ValueError, match="explicit resume"):
        selection.run_arrived_outer_selection(
            _candidates(False), [17, 19], checkpoint_store=store, sleep_error_retries=1
        )
    assert not path.exists()


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "boundary",
    [
        "first_wake",
        "first_b_wake",
        "candidate_boundary",
        "later_candidate_wake",
        "later_b_after_sleep",
        "frozen",
    ],
)
def test_should_resume_selection_from_candidate_and_freeze_transactions(
    reverse_order: bool, boundary: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = _candidates(reverse_order)
    ordinary = selection.run_arrived_outer_selection(candidates, [17, 19])
    control_store = TrustedLocalArrivedSelectionCheckpointStore(tmp_path / "control.ckpt")
    control = selection.run_arrived_outer_selection(
        candidates, [17, 19], checkpoint_store=control_store
    )
    updates: list[str] = []
    original_update = base._train_named_model_epoch

    def count_update(*args: Any, **kwargs: Any) -> None:
        updates.append(args[-1])
        original_update(*args, **kwargs)

    monkeypatch.setattr(base, "_train_named_model_epoch", count_update)
    interrupted = StopAtSelectionCheckpoint(tmp_path / "interrupted.ckpt", boundary)
    with pytest.raises(SelectionInterruption):
        selection.run_arrived_outer_selection(candidates, [17, 19], checkpoint_store=interrupted)
    checkpoint = interrupted.load()
    count_before_resume = len(updates)
    assert checkpoint.manifest_digest == control.freeze.candidate_manifest_digest
    assert (checkpoint.freeze is not None) == (boundary == "frozen")
    if checkpoint.active_v6 is not None:
        assert b"final_test" not in pickle.dumps(checkpoint.active_v6, protocol=5)

    original_release = selection.release_final_test

    def frozen_release(*args: Any, **kwargs: Any) -> Any:
        assert interrupted.load().stage == "frozen"
        return original_release(*args, **kwargs)

    monkeypatch.setattr(selection, "release_final_test", frozen_release)
    resumed = selection.run_arrived_outer_selection(
        candidates,
        [17, 19],
        checkpoint_store=interrupted.store,
        resume_from_checkpoint=True,
    )

    assert resumed == control == ordinary
    assert len(updates) == 24
    if boundary == "frozen":
        assert len(updates) == count_before_resume
    assert interrupted.load().stage == "frozen"
    assert interrupted.load().freeze == ordinary.freeze
    _assert_same_completed_candidates(
        interrupted.load().completed_candidates, control_store.load().completed_candidates
    )


def _stop_at_boundary(
    candidates: tuple[selection.ArrivedSelectionCandidate, ...],
    path: Path,
    boundary: str,
) -> TrustedLocalArrivedSelectionCheckpointStore:
    interrupted = StopAtSelectionCheckpoint(path, boundary)
    with pytest.raises(SelectionInterruption):
        selection.run_arrived_outer_selection(candidates, [17, 19], checkpoint_store=interrupted)
    return interrupted.store


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "tamper",
    [
        "manifest",
        "stored_manifest",
        "settings",
        "seeds",
        "outer_role",
        "trial",
        "event",
        "active_event",
        "active_role",
        "frozen_choice",
        "old_format",
        "trial_sleep",
        "trial_sleep_digest",
        "candidate_sleep_digest",
        "active_sleep",
        "frozen_sleep",
    ],
)
def test_should_reject_tampered_selection_before_update_or_final_release(
    reverse_order: bool, tamper: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = _candidates(reverse_order)
    boundary = (
        "first_wake"
        if tamper in {"active_event", "active_role"}
        else "later_b_after_sleep"
        if tamper == "active_sleep"
        else "frozen"
        if tamper in {"frozen_choice", "frozen_sleep"}
        else "candidate_boundary"
    )
    store = _stop_at_boundary(candidates, tmp_path / "tampered.ckpt", boundary)
    checkpoint = store.load()
    requested = candidates
    requested_seeds = [17, 19]
    if tamper == "manifest":
        requested = tuple(reversed(candidates))
    elif tamper == "old_format":
        store.save(replace(checkpoint, format_version=7))
    elif tamper == "stored_manifest":
        entries = list(checkpoint.candidate_manifest)
        entries[0] = ("changed", entries[0][1])
        store.save(replace(checkpoint, candidate_manifest=tuple(entries)))
    elif tamper == "settings":
        first = candidates[0]
        changed_config = replace(
            first.config,
            training=replace(first.config.training, backprop_learning_rate=0.11),
        )
        requested = (replace(first, config=changed_config), candidates[1])
    elif tamper == "seeds":
        requested_seeds = [17, 23]
    elif tamper == "outer_role":
        original = arrived.split_phase_decision_roles

        def changed_outer(*args: Any, **kwargs: Any) -> Any:
            roles = original(*args, **kwargs)
            if roles.seed != 17 or roles.phase != "a":
                return roles
            outer = LabeledData(roles.outer_selection.input, 1.0 - roles.outer_selection.target)
            hashes = dict(roles.split_hashes)
            hashes["outer_selection"] = continual_roles._hash_role(
                roles.phase,
                roles.seed,
                "outer_selection",
                roles.sample_ids["outer_selection"],
                outer,
            )
            return replace(roles, outer_selection=outer, split_hashes=MappingProxyType(hashes))

        monkeypatch.setattr(arrived, "split_phase_decision_roles", changed_outer)
    elif tamper in {"trial", "event"}:
        record = checkpoint.completed_candidates[0]
        if tamper == "trial":
            first_trial = replace(record.trials[0], balanced_score=0.123)
            trials = (first_trial, *record.trials[1:])
            record = replace(
                record,
                trials=trials,
                trial_digest=selection._trial_digest(trials),
            )
        else:
            seed_record = record.unscored_seeds[0]
            accesses = seed_record.role_accesses[:-1]
            seed_record = replace(
                seed_record,
                role_accesses=accesses,
                event_digest=arrived_event_digest(
                    accesses, seed_record.guard_decisions, seed_record.method_task_information
                ),
            )
            record = replace(record, unscored_seeds=(seed_record, *record.unscored_seeds[1:]))
        store.save(replace(checkpoint, completed_candidates=(record,)))
    elif tamper in {"trial_sleep", "trial_sleep_digest", "candidate_sleep_digest"}:
        record = checkpoint.completed_candidates[0]
        if tamper == "candidate_sleep_digest":
            record = replace(record, sleep_history_digest="0" * 64)
        else:
            index = next(
                index
                for index, item in enumerate(record.trials)
                if item.method == "circadian_predictive_coding"
            )
            trial_rows = list(record.trials)
            trial = trial_rows[index]
            trial_rows[index] = (
                replace(trial, sleep_events=())
                if tamper == "trial_sleep"
                else replace(trial, sleep_history_digest="0" * 64)
            )
            record = replace(record, trials=tuple(trial_rows))
        store.save(replace(checkpoint, completed_candidates=(record,)))
    elif tamper in {"active_event", "active_role", "active_sleep"}:
        assert checkpoint.active_v6 is not None
        active = checkpoint.active_v6
        if tamper == "active_event":
            accesses = active.active_role_accesses[:-1]
            active = replace(
                active,
                active_role_accesses=accesses,
                active_event_digest=arrived_event_digest(
                    accesses, active.active_guard_decisions, active.active_task_information
                ),
            )
        elif tamper == "active_role":
            hashes = list(active.active_role_hashes)
            hashes[0] = (hashes[0][0], "changed")
            active = replace(active, active_role_hashes=tuple(hashes))
        else:
            events = active.active_sleep_events
            assert events
            changed = replace(events[-1], reason="forged_reason")
            tampered_events = (*events[:-1], changed)
            active = replace(
                active,
                active_sleep_events=tampered_events,
                active_sleep_history_digest=arrived_sleep_history_digest(
                    tampered_events, active.active_event_digest
                ),
            )
        store.save(replace(checkpoint, active_v6=active))
    elif tamper == "frozen_sleep":
        assert checkpoint.freeze is not None
        store.save(
            replace(
                checkpoint,
                freeze=replace(checkpoint.freeze, sleep_history_digest="0" * 64),
            )
        )
    else:
        assert checkpoint.freeze is not None
        freeze = checkpoint.freeze
        first_choice = replace(freeze.choices[0], candidate_id="lower_rate")
        choices = (first_choice, *freeze.choices[1:])
        freeze = replace(
            freeze,
            choices=choices,
            freeze_digest=selection._hash_json(
                (
                    freeze.candidate_manifest_digest,
                    freeze.trial_digest,
                    [asdict(item) for item in choices],
                )
            ),
        )
        store.save(replace(checkpoint, freeze=freeze))

    def no_update(*_: Any, **__: Any) -> None:
        raise AssertionError("resumed model updated before v7 preflight")

    def no_final(*_: Any, **__: Any) -> Any:
        raise AssertionError("final source opened before v7 preflight")

    monkeypatch.setattr(base, "_train_named_model_epoch", no_update)
    monkeypatch.setattr(selection, "release_final_test", no_final)
    if tamper in {"manifest", "stored_manifest", "settings", "seeds"}:
        monkeypatch.setattr(
            arrived,
            "generate_two_cluster_dataset_with_transform",
            lambda **_: no_final(),
        )
    with pytest.raises(ValueError, match="incompatible v7|incompatible v6"):
        selection.run_arrived_outer_selection(
            requested,
            requested_seeds,
            checkpoint_store=store,
            resume_from_checkpoint=True,
        )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_should_keep_final_sources_sealed_and_choice_invariant_at_terminal_resume(
    reverse_order: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = _candidates(reverse_order)
    store = TrustedLocalArrivedSelectionCheckpointStore(tmp_path / "terminal.ckpt")
    original_a = arrived.generate_two_cluster_dataset_with_transform
    original_b = arrived._generate_phase_b_source
    changed_labels = False
    final_reads: list[tuple[int, str, str]] = []

    class SealedSource:
        def __init__(self, source: Any, seed: int, phase: str) -> None:
            self.source = source
            self.seed = seed
            self.phase = phase
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            assert store.load().stage == "frozen"
            final_reads.append((self.seed, self.phase, "input"))
            return self.source.test_input

        @property
        def test_target(self) -> Any:
            assert store.load().stage == "frozen"
            final_reads.append((self.seed, self.phase, "label"))
            original = self.source.test_target
            return (
                1.0 - original
                if changed_labels and self.seed == 17 and self.phase == "a"
                else original
            )

    def source_a(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_a(*args, **kwargs), kwargs["seed"], "a")

    def source_b(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_b(*args, **kwargs), args[1] - 101, "b")

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", source_a)
    monkeypatch.setattr(arrived, "_generate_phase_b_source", source_b)
    baseline = selection.run_arrived_outer_selection(candidates, [17, 19], checkpoint_store=store)
    saved_bytes = store.path.read_bytes()
    assert len(final_reads) == 8
    changed_labels = True

    def no_update(*_: Any, **__: Any) -> None:
        raise AssertionError("terminal v7 resume retrained a candidate")

    monkeypatch.setattr(base, "_train_named_model_epoch", no_update)
    changed = selection.run_arrived_outer_selection(
        candidates, [17, 19], checkpoint_store=store, resume_from_checkpoint=True
    )

    assert len(final_reads) == 16
    assert store.path.read_bytes() == saved_bytes
    assert changed.trials == baseline.trials
    assert changed.selections == baseline.selections
    assert changed.freeze == baseline.freeze
    assert changed.final_seed_results[1] == baseline.final_seed_results[1]
    assert (
        changed.final_seed_results[0].role_hashes["phase_a_final_test"]
        != baseline.final_seed_results[0].role_hashes["phase_a_final_test"]
    )


def test_should_keep_v7_checkpoint_distinct_and_unscored(tmp_path: Path) -> None:
    path = tmp_path / "selection.ckpt"
    store = TrustedLocalArrivedSelectionCheckpointStore(path)
    selection.run_arrived_outer_selection(_candidates(False), [17, 19], checkpoint_store=store)
    checkpoint = store.load()

    assert checkpoint.format_version == 8
    assert checkpoint.stage == "frozen"
    assert checkpoint.candidate_manifest == tuple(
        (item.candidate_id, item.config) for item in _candidates(False)
    )
    assert checkpoint.active_v6 is None
    assert len(checkpoint.completed_candidates) == 2
    for candidate in checkpoint.completed_candidates:
        assert len(candidate.unscored_seeds) == 2
        assert len(candidate.trials) == 6
        for seed in candidate.unscored_seeds:
            assert all("final_test" not in role for role, _ in seed.role_ids)
            assert all("final_test" not in role for role, _ in seed.role_hashes)
            assert all(event.role != "final_test" for event in seed.role_accesses)
        for trial in candidate.trials:
            assert all("final_test" not in role for role, _ in trial.development_role_ids)
            assert all(event.role != "final_test" for event in trial.role_accesses)
    with pytest.raises(ValueError, match="header"):
        TrustedLocalArrivedCheckpointStore(path).load()
