"""Fixed A-return routing preserves the paid F assay and its negative results.

Only existing fake native ports and synthetic public observations are used.
Actual stream/probe helpers perform the sequencing; no environment is executed.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from test_minigrid_distinct_probe import METHODS, composition
from test_minigrid_learning_pilot import evaluator as evaluator
from test_minigrid_old_task_readouts import READOUT_FLAGS
from test_torch_feature_rehearsal_pilot import forbidden


RETURN_FLAGS: dict[str, Any] = dict(**READOUT_FLAGS, return_to_a=True)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_validate_the_return_flag_before_resources(monkeypatch, value):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**READOUT_FLAGS, return_to_a=value)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"public_history": True, "invariant_history": True},
        {"public_history": True, "invariant_history": True, "distinct_probe": True},
        {"public_history": True, "invariant_history": True, "old_task_readouts": True},
        {"public_history": True, "distinct_probe": True, "old_task_readouts": True},
    ],
)
def test_should_require_the_original_monitored_distinct_probe_before_resources(
    monkeypatch, options
):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**options, return_to_a=True)


@pytest.mark.parametrize("method", METHODS)
def test_should_add_the_fixed_return_dose_without_changing_the_selected_learner(
    monkeypatch, method
):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    options = dict(
        public_history=True,
        invariant_history=True,
        distinct_probe=True,
        old_task_readouts=True,
        qualification_method=method,
    )
    old = pilot.describe_protocol(**options)
    new = pilot.describe_protocol(**options, return_to_a=True)
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
        assert new[key] == old[key]
    assert new["probe"]["updates"] == old["probe"]["updates"] == 16
    assert new["planned_work"]["native_train_calls"] == 192
    assert new["planned_work"]["training_rows"] == 1728
    assert new["planned_work"]["score_forwards"] == 94
    assert new["planned_work"]["score_rows"] == 846
    for key in ("teacher_episodes", "final_episodes", "learner_live_episodes"):
        assert new["planned_work"][key] == old["planned_work"][key]


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"public_history": True},
        {"public_history": True, "invariant_history": True},
        READOUT_FLAGS,
    ],
)
def test_should_preserve_prior_descriptors_when_return_is_false(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, return_to_a=False
    )


@pytest.mark.parametrize("history", [False, True])
def test_should_derive_two_owned_updates_only_from_original_a_train_sources(evaluator, history):
    monitors = evaluator.sources[:2]
    if history:
        monitors = pilot._history_packets(monitors)
    before = pickle.dumps(monitors)
    templates = pilot._return_training_sources(monitors)
    assert type(templates) is tuple and len(templates) == 2
    for original, returned, cue in zip(monitors, templates, ("blue", "red"), strict=True):
        assert (
            returned.role == original.role == "train"
            and returned.context == original.context == "A"
        )
        assert returned.features.shape == original.features.shape == (9, 296 if history else 148)
        assert returned.episode is not None and original.episode is not None
        assert returned.episode.episode_id == f"minigrid-return-train-{cue}"
        assert returned.episode.episode_id != original.episode.episode_id
        assert returned.episode.transitions == original.episode.transitions
        assert set(returned.row_ids).isdisjoint(original.row_ids)
        np.testing.assert_array_equal(returned.features, original.features)
        np.testing.assert_array_equal(returned.targets, original.targets)
        assert not np.shares_memory(returned.features, original.features)
        assert not np.shares_memory(returned.targets, original.targets)
    templates[0].features.fill(0.125)
    assert pickle.dumps(monitors) == before


def test_should_omit_return_keywords_on_the_previous_monitored_route(monkeypatch, evaluator):
    case = composition(monkeypatch, evaluator)
    calls = []

    def prior_helper(source, state, training, evaluation, **options):
        calls.append(options)
        assert "return_training" not in options and "return_budget" not in options
        assert len(options["training_readouts"]) == 2
        raise RuntimeError("stop at previous monitored helper call")

    monkeypatch.setattr(pilot, "run_fresh_task_probe", prior_helper)
    report = pilot.run_pilot(**READOUT_FLAGS, return_to_a=False)
    assert report["status"] == "incomplete" and len(calls) == 1
    assert report["methods"][0]["charged_updates"] == 128 and not case.native.get("trains")


def test_should_complete_two_same_fork_returns_with_separate_original_budgets_and_full_costs(
    monkeypatch, evaluator
):
    case = composition(monkeypatch, evaluator)
    delegate = pilot.run_fresh_task_probe
    calls = []

    def capture(source, state, training, evaluation, **options):
        monitors, updates = options["training_readouts"], options["return_training"]
        calls.append(
            dict(
                original=options["budget"],
                returned=options["return_budget"],
                state=deepcopy(state),
                before=pickle.dumps((training, evaluation, monitors, updates)),
            )
        )
        assert options["budget"] is not options["return_budget"]
        assert (
            options["budget"].updates_completed == options["return_budget"].updates_completed == 0
        )
        assert options["return_budget"].budget.max_training_updates == 16
        assert [packet.context for packet in training] == ["F-blue", "F-red"] * 8
        assert all(packet.role == "train" and packet.context == "A" for packet in updates)
        value = delegate(source, state, training, evaluation, **options)
        assert pickle.dumps((training, evaluation, monitors, updates)) == calls[-1]["before"]
        return value

    monkeypatch.setattr(pilot, "run_fresh_task_probe", capture)
    report = pilot.run_pilot(**RETURN_FLAGS)
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert [call["state"].completed_updates for call in calls] == [0, 128]
    all_budgets = [budget for call in calls for budget in (call["original"], call["returned"])]
    assert len({id(budget) for budget in all_budgets}) == 4
    assert all(budget.updates_completed == 16 for budget in all_budgets)
    assert len(case.native["trains"]) == 64 and len(case.native["predicts"]) == 94
    for offset in (0, 32):
        assert all(
            np.all(targets == 0) for _, _, targets in case.native["trains"][offset : offset + 16]
        )
        assert all(
            np.all(targets == 1)
            for _, _, targets in case.native["trains"][offset + 16 : offset + 32]
        )
    for slot in row["probes"]:
        assert (
            slot["status"] == "complete"
            and slot["charged_updates"] == slot["return_charged_updates"] == 16
        )
        assert slot["report"]["completed_updates"] == 16
        returned = slot["report"]["return_phase"]
        assert returned["status"] == "complete" and returned["completed_training_examples"] == 144
        assert returned["reserved_data_bytes"] == 42768
        assert (
            returned["score_forward_calls_completed"]
            == returned["training_readouts"]["score_forward_calls_completed"]
            == 8
        )
        assert returned["training_readouts"]["completed_rows"] == 72
        assert returned["checkpoints"][0]["scores"] == slot["report"]["checkpoints"][-1]["scores"]
        assert (
            returned["training_readouts"]["checkpoints"][0]["scores"]
            == slot["report"]["training_readouts"]["checkpoints"][-1]["scores"]
        )
        for context in ("F", "A"):
            outcome = slot["return_fit_outcomes"][context]
            assert [point["updates"] for point in outcome["curve"]] == [0, 4, 8, 16]
            assert (
                outcome["ce_gain"]
                == outcome["accuracy_gain"]
                == outcome["exact_sequence_gain"]
                == 0.0
            )
        projection = slot["projection_accounting"]
        assert projection["training_rows_returned"] == projection["score_rows_returned"] == 144
        assert projection["return_rows_returned"] == 288
        assert projection["array_bytes_returned"] == 1363968
    assert report["work"]["native_train_calls"] == 192 and report["work"]["training_rows"] == 1728
    assert report["work"]["total_score_forwards_completed"] == 94
    assert report["work"]["total_score_rows_completed"] == 846
    return_work = report["work"]["return_phase"]
    assert return_work["native_train_calls"] == 32 and return_work["training_rows"] == 288
    assert return_work["score_forwards_attempted"] == return_work["score_forwards_completed"] == 32
    assert (
        return_work["score_rows_completed"] == 288 and return_work["reserved_data_bytes"] == 85536
    )
    assert len(report["probe_data"]["support"]["probe"]["packets"]) == 18
    assert report["probe_data"]["support"]["all_canonical_inputs_unseen"] is True
    assert not case.native.get("sessions") and not case.native.get("environments")
    assert report["scientific_advantage_verified"] is False


@pytest.mark.parametrize(
    "field", ["return_phase", "return_training_readouts", "return_fit_outcomes"]
)
def test_should_keep_the_prior_paid_return_and_refuse_equal_replacements_during_the_next_probe(
    monkeypatch, evaluator, field
):
    case = composition(monkeypatch, evaluator)

    def replace_prior(owner):
        if len(case.native.get("trains", [])) == 33:
            slot = case.work["slots"][0]
            if field == "return_phase":
                slot["report"][field] = deepcopy(slot["report"][field])
            elif field == "return_training_readouts":
                phase = slot["report"]["return_phase"]
                phase["training_readouts"] = deepcopy(phase["training_readouts"])
            else:
                slot[field] = deepcopy(slot[field])

    case.native["on_train"] = replace_prior
    report = pilot.run_pilot(**RETURN_FLAGS)
    assert report["status"] == "incomplete" and report["reason"]
    first, second = report["methods"][0]["probes"]
    assert (
        first["status"] == "complete"
        and first["charged_updates"] == first["return_charged_updates"] == 16
    )
    assert first["report"]["return_phase"]["completed_updates"] == 16
    assert second["status"] == "failed" and second["charged_updates"] == 1
    assert second["report"]["completed_training_examples"] == 9
    assert second["report"]["return_phase"]["status"] == "not_run"
    assert second["report"]["return_phase"]["completed_updates"] == 0
    assert len(case.native["trains"]) == 33
    assert report["methods"][0]["charged_updates"] == 128
    assert report["comparisons"] is None


def test_should_report_changed_return_budget_binding_as_false_without_losing_the_paid_update(
    monkeypatch, evaluator
):
    case = composition(monkeypatch, evaluator)
    delegate = pilot.run_fresh_task_probe
    returned_budgets = []

    def capture_budget(source, state, training, evaluation, **options):
        returned_budgets.append(options["return_budget"])
        return delegate(source, state, training, evaluation, **options)

    def replace_limits_after_first_return_commit(owner):
        if len(case.native.get("trains", [])) == 17:
            budget = returned_budgets[0]
            # Change only the resource binding; retain the actual native work
            # and its permanent update counter for the failure readback.
            budget.budget = replace(
                budget.budget, max_wall_seconds=budget.budget.max_wall_seconds + 1.0
            )

    monkeypatch.setattr(pilot, "run_fresh_task_probe", capture_budget)
    case.native["on_train"] = replace_limits_after_first_return_commit
    report = pilot.run_pilot(**RETURN_FLAGS)

    assert report["status"] == "incomplete" and report["reason"]
    first, second = report["methods"][0]["probes"]
    assert first["status"] == "failed"
    assert first["return_budget_binding_preserved"] is False
    assert first["charged_updates"] == first["report"]["completed_updates"] == 16
    assert first["report"]["status"] == "complete"
    assert first["return_charged_updates"] == returned_budgets[0].updates_completed == 1
    returned = first["report"]["return_phase"]
    assert returned["status"] == "failed" and returned["completed_updates"] == 1
    assert returned["completed_training_examples"] == 9
    assert [point["updates"] for point in returned["checkpoints"]] == [0]
    assert second["status"] == "not_run" and second["report"] is None
    assert len(returned_budgets) == 1 and len(case.native["trains"]) == 17
    assert report["methods"][0]["charged_updates"] == 128
