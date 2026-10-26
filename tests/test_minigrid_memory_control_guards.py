"""Paid proof and isolation tests using typed fake count-state ports only.

No real count update/prediction, Torch learner, FIFO, or physics is executed by
these cases. Full production snapshot validation and boundary guards remain real.
"""

from copy import deepcopy
import pickle
from typing import Any

import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemorySnapshot,
    InvariantMemoryPolicy,
)
from src.core.action_memory import ActionMemoryPolicy, ActionMemorySnapshot
from test_minigrid_learning_pilot import evaluator as evaluator
from test_minigrid_memory_control import FLAGS, METHOD, memory_environment
from test_torch_feature_rehearsal_pilot import forbidden


def fake_case(monkeypatch, evaluator):
    monkeypatch.setattr(InvariantActionMemoryLearner, "train_batch", forbidden)
    monkeypatch.setattr(InvariantActionMemoryLearner, "predict", forbidden)
    return memory_environment(monkeypatch, evaluator)


def test_should_use_backend_neutral_typed_guards_without_neural_or_count_backend_work(
    monkeypatch, evaluator
):
    case = fake_case(monkeypatch, evaluator)
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "complete", report["reason"]
    assert len(case.work["trains"]) == 64 and len(case.work["predicts"]) == 94
    assert report["methods"][0]["method"] == METHOD
    assert report["work"]["neural_sgd_updates"] == 0
    assert all(
        call["source_before"] == pickle.dumps(call["source"].snapshot_state())
        for call in case.work["probes"]
    )


@pytest.mark.parametrize(
    "field", ["report", "source_table_observation", "return_phase", "return_fit_outcomes"]
)
def test_should_refuse_equal_replacement_of_paid_count_proofs_before_more_work(
    monkeypatch, evaluator, field
):
    case = fake_case(monkeypatch, evaluator)

    def replace_paid(owner):
        if len(case.work["trains"]) == 33:
            slot = case.work["slots"][0]
            container = slot["report"] if field == "return_phase" else slot
            container[field] = deepcopy(container[field])

    case.work["on_train"] = replace_paid
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    first, second = row["probes"]
    assert (
        first["status"] == "complete"
        and first["charged_updates"] == first["return_charged_updates"] == 16
    )
    assert first["report"]["return_phase"]["completed_updates"] == 16
    assert first["source_table_observation"]["entries"] == 0
    assert second["status"] == "failed" and second["charged_updates"] == 1
    assert second["report"]["completed_training_examples"] == 9
    assert second["report"]["return_phase"]["status"] == "not_run"
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert len(case.work["trains"]) == 33


@pytest.mark.parametrize("fault", ["old_a_array", "saved_count_distribution", "source_extra_field"])
def test_should_preserve_the_paid_first_update_and_refuse_changed_original_evidence(
    monkeypatch, evaluator, fault
):
    case = fake_case(monkeypatch, evaluator)

    def corrupt(owner):
        if len(case.work["trains"]) == 1:
            if fault == "old_a_array":
                case.work["probes"][0]["monitors"][0].features[0, 0] = 0.125
            elif fault == "saved_count_distribution":
                state = case.work["arms"][0]["states"][-1][1]
                entry = state.native_state.entries[0]
                # This stays a valid typed table with identical total exposure
                # and clocks; the full preceding-state proof must still refuse.
                object.__setattr__(entry, "counts", (1, 1151, 0, 0, 0, 0, 0))
            else:
                source = case.work["arms"][0]["source"]
                object.__setattr__(source.state.native_state, "extra", None)

    case.work["on_train"] = corrupt
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    first, second = row["probes"]
    assert first["status"] == "failed" and first["charged_updates"] == 1
    assert (
        first["report"]["completed_updates"] == 1
        and first["report"]["completed_training_examples"] == 9
    )
    assert second["status"] == "not_run" and second["report"] is None
    assert row["charged_updates"] == 128 and len(case.work["trains"]) == 1
    if fault == "source_extra_field":
        assert hasattr(case.work["arms"][0]["source"].state.native_state, "extra")


def test_should_refuse_changed_return_budget_binding_and_keep_count_work_paid(
    monkeypatch, evaluator
):
    case = fake_case(monkeypatch, evaluator)

    def corrupt(owner):
        if len(case.work["trains"]) == 17:
            case.work["probes"][0]["return_budget"].clock = lambda: 1.0

    case.work["on_train"] = corrupt
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "incomplete"
    first, second = report["methods"][0]["probes"]
    assert first["charged_updates"] == first["report"]["completed_updates"] == 16
    assert first["return_charged_updates"] == 1
    assert first["return_budget_binding_preserved"] is False
    assert first["report"]["return_phase"]["completed_updates"] == 1
    assert first["report"]["return_phase"]["completed_training_examples"] == 9
    assert second["status"] == "not_run" and len(case.work["trains"]) == 17


def test_should_refuse_forged_table_observation_after_a_paid_readout_and_restore_its_record(
    monkeypatch, evaluator
):
    case = fake_case(monkeypatch, evaluator)
    original = pilot.memory.table_observation
    observations = []

    def observe(state):
        result = original(state)
        observations.append(result)
        return result

    def corrupt(owner):
        if len(case.work["predicts"]) == 1:
            assert observations
            observations[-1]["entries"] = 64

    monkeypatch.setattr(pilot.memory, "table_observation", observe)
    case.work["on_predict"] = corrupt
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    assert row["readouts"][0]["status"] == "failed"
    assert row["readouts"][0]["table_observation"]["entries"] == 0
    assert row["readouts"][0]["work"]["score_forward_calls_completed"] == 1
    assert row["charged_updates"] == 128 and case.work["trains"] == []
    assert len(case.work["predicts"]) == 1


@pytest.mark.parametrize("fault", ["raw", "capacity", "clock_bool", "extra"])
def test_should_refuse_incompatible_count_checkpoints_without_training_or_scoring(
    monkeypatch, evaluator, fault
):
    case = fake_case(monkeypatch, evaluator)
    native = ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)
    state: Any = InvariantMemorySnapshot(
        InvariantMemoryPolicy(native),
        ActionMemorySnapshot(native, (), 0, 0),
    )
    if fault == "raw":
        state = state.native_state
    elif fault == "capacity":
        native = ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=10752)
        state = InvariantMemorySnapshot(
            InvariantMemoryPolicy(native), ActionMemorySnapshot(native, (), 0, 0)
        )
    elif fault == "clock_bool":
        object.__setattr__(state.native_state, "completed_updates", False)
    else:
        object.__setattr__(state, "extra", None)
    with pytest.raises(ValueError):
        pilot._invariant_snapshot(state, 0, METHOD)
    assert case.work["trains"] == case.work["predicts"] == []


def test_should_use_the_real_final_readback_comparator_without_torch(monkeypatch, evaluator):
    case = fake_case(monkeypatch, evaluator)
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "complete", report["reason"]
    assert len(case.work["readbacks"]) == 1
    assert case.work["readbacks"][0] is pilot.shared._same_value
    assert report["methods"][0]["source_unchanged"] is True
    assert "readback_errors" not in report
