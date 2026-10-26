"""Old A readouts compose with fixed F probes without changing training.

The benchmark is the previously accepted fake TRAIN stub; the actual disposable
probe use-case and support diagnostic run against fake native heads. No physics
or neural learning is evidence in this file.
"""

from copy import deepcopy
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from test_minigrid_distinct_probe import FLAGS, METHODS, composition, raw_probe_data
from test_minigrid_learning_pilot import (
    POINTS,
    evaluator as evaluator,
    pilot_environment,
    same_state,
)
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


READOUT_FLAGS: dict[str, Any] = dict(**FLAGS, old_task_readouts=True)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_refuse_nonboolean_readout_modes_before_resources(monkeypatch, value):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**FLAGS, old_task_readouts=value)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"public_history": True},
        {"invariant_history": True},
        {"public_history": True, "invariant_history": True},
        {"public_history": True, "distinct_probe": True},
        {"invariant_history": True, "distinct_probe": True},
    ],
)
def test_should_require_both_history_modes_and_the_distinct_probe_before_resources(
    monkeypatch, options
):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**options, old_task_readouts=True)


@pytest.mark.parametrize("method", METHODS)
def test_should_preserve_selected_head_settings_and_only_add_readout_costs(monkeypatch, method):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    options = dict(
        public_history=True,
        invariant_history=True,
        distinct_probe=True,
        qualification_method=method,
    )
    previous = pilot.describe_protocol(**options)
    monitored = pilot.describe_protocol(**options, old_task_readouts=True)
    for key in (
        "methods",
        "seed",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "hidden_dim",
        "num_classes",
        "feature_dim_by_method",
        "cpc_configs",
        "task_sequence",
        "checkpoint_updates",
    ):
        assert monitored[key] == previous[key]
    for key in (
        "native_train_calls",
        "training_rows",
        "teacher_episodes",
        "final_episodes",
        "learner_live_episodes",
    ):
        assert monitored["planned_work"][key] == previous["planned_work"][key]
    assert monitored["planned_work"]["score_forwards"] == 62
    assert monitored["planned_work"]["score_rows"] == 558


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"public_history": True},
        {"public_history": True, "invariant_history": True},
        FLAGS,
        {
            "public_history": True,
            "invariant_history": True,
            "distinct_probe": True,
            "qualification_method": "neutral_cpc",
        },
    ],
)
def test_should_preserve_existing_descriptors_when_readouts_are_false(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, old_task_readouts=False
    )


def test_should_omit_the_new_helper_keyword_on_the_false_route(monkeypatch, evaluator):
    case = composition(monkeypatch, evaluator)
    calls = []

    def old_probe(source, state, training, evaluation, **options):
        calls.append(options)
        assert "training_readouts" not in options
        raise RuntimeError("stop at original probe call shape")

    monkeypatch.setattr(pilot, "run_fresh_task_probe", old_probe)
    report = pilot.run_pilot(**FLAGS, old_task_readouts=False)
    assert report["status"] == "incomplete" and len(calls) == 1
    assert report["methods"][0]["charged_updates"] == 128
    assert len(case.work["fit"]) == 5 and not case.native.get("trains")
    assert "training_readouts" not in report["work"]


def test_should_read_original_a_on_each_f_fork_with_separate_costs_and_no_new_training(
    monkeypatch, evaluator
):
    case = composition(monkeypatch, evaluator)
    delegate = pilot.run_fresh_task_probe
    calls = []
    expected = pilot._history_packets(evaluator.sources[:2])
    expected_ids = {batch.episode.episode_id for batch in expected if batch.episode is not None}
    assert len(expected_ids) == 2

    def capture(source, state, training, evaluation, **options):
        monitors = options["training_readouts"]
        calls.append(
            dict(state=deepcopy(state), monitors=deepcopy(monitors), budget=options["budget"])
        )
        assert len(case.work["fit"]) == 5
        assert len(training) == 16 and len(evaluation) == 2 and len(monitors) == 2
        assert [batch.context for batch in training] == ["F-blue", "F-red"] * 8
        assert all(
            batch.role == "train" and batch.context == "A" and batch.features.shape == (9, 296)
            for batch in monitors
        )
        for actual, original in zip(monitors, expected, strict=True):
            assert actual.episode == original.episode and actual.row_ids == original.row_ids
            np.testing.assert_array_equal(actual.features, original.features)
            np.testing.assert_array_equal(actual.targets, original.targets)
            assert not np.shares_memory(actual.features, original.features)
        return delegate(source, state, training, evaluation, **options)

    monkeypatch.setattr(pilot, "run_fresh_task_probe", capture)
    report = pilot.run_pilot(**READOUT_FLAGS)
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert [point for point, _, _ in case.work["fit"]] == list(POINTS)
    assert [call["state"].completed_updates for call in calls] == [0, 128]
    assert calls[0]["budget"] is not calls[1]["budget"]
    assert all(call["budget"].updates_completed == 16 for call in calls)
    assert len(case.native["trains"]) == 32 and len(case.native["predicts"]) == 62
    assert all(np.all(targets == 0) for _, _, targets in case.native["trains"])
    for slot in row["probes"]:
        assert slot["charged_updates"] == slot["report"]["completed_updates"] == 16
        assert slot["report"]["completed_training_examples"] == 144
        assert slot["report"]["score_forward_calls_completed"] == 8
        old = slot["report"]["training_readouts"]
        assert old["status"] == "complete" and old["score_forward_calls_completed"] == 8
        assert old["completed_rows"] == 72 and old["reserved_data_bytes"] == 42768
        assert [point["updates"] for point in old["checkpoints"]] == [0, 4, 8, 16]
        assert {key for point in old["checkpoints"] for key in point["scores"]} == expected_ids
        outcome = slot["old_task_readout_outcomes"]
        assert [point["updates"] for point in outcome["curve"]] == [0, 4, 8, 16]
        assert (
            outcome["ce_gain"] == outcome["accuracy_gain"] == outcome["exact_sequence_gain"] == 0.0
        )
        assert slot["projection_accounting"]["score_rows_returned"] == 144
        assert slot["projection_accounting"]["training_readout_score_rows_returned"] == 72
    support = report["probe_data"]["support"]
    assert support["all_canonical_inputs_unseen"] is True
    assert len(support["probe"]["packets"]) == 18 and support["probe"]["rows"] == 162
    assert {packet["role"] for packet in support["probe"]["packets"]} == {
        "probe_train",
        "probe_test",
    }
    assert report["work"]["native_train_calls"] == 160 and report["work"]["training_rows"] == 1440
    assert report["work"]["probe_score_forwards_completed"] == 16
    assert report["work"]["total_score_forwards_completed"] == 62
    assert report["work"]["total_score_rows_completed"] == 558
    for call in case.work["probes"]:
        assert call["source_before"] == call["source_after"]
        assert (
            call["state_before"] == call["state_after"]
            and call["inputs_before"] == call["inputs_after"]
        )
    assert not case.native.get("environments") and not case.native.get("sessions")
    assert report["scientific_advantage_verified"] is False


def test_should_keep_prior_paid_old_fit_and_stop_after_corruption_in_the_second_probe(
    monkeypatch, evaluator
):
    case = composition(monkeypatch, evaluator)

    def corrupt(owner):
        if len(case.native.get("trains", [])) == 17:
            case.work["slots"][0]["report"]["training_readouts"]["checkpoints"][0]["scores"].clear()

    case.native["on_train"] = corrupt
    report = pilot.run_pilot(**READOUT_FLAGS)
    assert report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    first, second = row["probes"]
    assert first["status"] == "complete" and first["charged_updates"] == 16
    assert first["report"]["training_readouts"]["score_forward_calls_completed"] == 8
    assert all(
        len(point["scores"]) == 2 for point in first["report"]["training_readouts"]["checkpoints"]
    )
    assert second["status"] == "failed" and second["charged_updates"] == 1
    assert second["report"]["completed_training_examples"] == 9
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert len(case.native["trains"]) == 17 and report["comparisons"] is None


@pytest.mark.parametrize("field", ["report", "outcomes", "old_task_readout_outcomes"])
def test_should_refuse_equal_replacement_of_a_completed_monitored_recurrent_probe_before_more_work(
    monkeypatch, evaluator, field
):
    # Reuse only the native-port setup; no five-arm benchmark or physics runs.
    pilot_environment(monkeypatch, evaluator)
    clock = lambda: 1.0
    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(pilot.shared, "monotonic", clock)
    monkeypatch.setattr(pilot, "_same_state", same_state)
    source = pilot.shared._learner("recurrent_bc")
    initial = source.snapshot_state()
    original_source = pickle.dumps(initial)
    row = pilot._new_row("recurrent_bc")
    arm = dict(source=source, row=row)
    data = raw_probe_data()
    monitors = evaluator.sources[:2]
    delegate = pilot.run_fresh_task_probe
    budgets = []

    def collect_budget(*args, **options):
        budgets.append(options["budget"])
        return delegate(*args, **options)

    monkeypatch.setattr(pilot, "run_fresh_task_probe", collect_budget)
    first, second = row["probes"]
    completed_guard = pilot._run_probe(
        first,
        arm,
        initial,
        data,
        sampler=Sampler(),
        deadline=100.0,
        check_boundary=lambda: None,
        training_readouts=monitors,
    )
    completed_guard()
    assert first["status"] == "complete" and first["charged_updates"] == 16
    assert first["report"]["training_readouts"]["score_forward_calls_completed"] == 8
    assert first["report"]["training_readouts"]["completed_rows"] == 72
    assert "projection_accounting" not in first
    paid = (len(evaluator.work["trains"]), len(evaluator.work["predicts"]))
    assert paid == (16, 16)

    owner = first[field]
    replacement = deepcopy(owner)
    assert replacement is not owner and pickle.dumps(replacement) == pickle.dumps(owner)
    first[field] = replacement
    later = deepcopy(initial)
    later.completed_updates, later.training_examples = 128, 1152
    with pytest.raises(ValueError, match="preceding paired probe result changed"):
        pilot._run_probe(
            second,
            arm,
            later,
            data,
            sampler=Sampler(),
            deadline=100.0,
            check_boundary=completed_guard,
            training_readouts=monitors,
        )

    assert len(budgets) == 1 and budgets[0].updates_completed == 16
    assert (len(evaluator.work["trains"]), len(evaluator.work["predicts"])) == paid
    assert second["status"] == "failed" and second["report"] is None
    assert second["charged_updates"] is None
    assert pickle.dumps(source.snapshot_state()) == original_source
