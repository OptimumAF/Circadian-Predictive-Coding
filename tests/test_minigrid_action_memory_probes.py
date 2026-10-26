"""Real bounded memory and the existing probe helper on engineered E arrays.

No benchmark rerun, teacher, physics or neural learner produces these fixtures.
The mature table is an explicit clock-valid capacity fixture, not saved science.
"""

import builtins
from dataclasses import asdict
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_action_memory_probes as pilot
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.app.coupled_attribution import AttributionBatch
from src.app import toy_execution_budget as budgets
from src.core.action_memory import ActionMemoryEntry, ActionMemoryPolicy, ActionMemorySnapshot
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


def snapshots() -> tuple[ActionMemorySnapshot, ActionMemorySnapshot]:
    def policy():
        return ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)

    entries = []
    for index in range(54):
        features = (0.75,) + (0.0,) * 294 + (index / 128,)
        counts = [0] * 7
        counts[index % 7] = 22 if index < 18 else 21
        entries.append(ActionMemoryEntry(features, tuple(counts)))
    return ActionMemorySnapshot(policy(), (), 0, 0), ActionMemorySnapshot(
        policy(), tuple(entries), 128, 1152
    )


def frame(cue: str, step: int) -> tuple[float, ...]:
    return ((-0.25 if cue == "blue" else 0.25),) + (0.0,) * 146 + (step / 16,)


def packet(role: str, cue: str, identity: str) -> AttributionBatch:
    episode = DemonstrationEpisode(
        identity,
        "E-" + cue,
        role,
        tuple(
            DemonstrationTransition(
                step,
                frame(cue, step),
                step % 7,
                0.5 if step == 8 else 0.0,
                frame(cue, step + 1),
                step == 8,
                7,
            )
            for step in range(9)
        ),
        7,
    )
    x, y = episode_arrays(episode)
    return AttributionBatch(
        role,
        episode.context,
        tuple(f"{identity}:step:{step}" for step in range(9)),
        x,
        y,
        episode,
        7,
    )


def reserved_data():
    training = tuple(
        packet(
            "probe_train",
            "blue" if index % 2 == 0 else "red",
            f"minigrid-probe-source-probe_train-{'blue' if index % 2 == 0 else 'red'}:copy:{index}",
        )
        for index in range(16)
    )
    evaluation = tuple(
        packet("probe_test", cue, "minigrid-probe-source-probe_test-" + cue)
        for cue in ("blue", "red")
    )
    return training, evaluation


@pytest.fixture
def assay(monkeypatch):
    originals = snapshots()
    work: dict[str, Any] = dict(
        inputs=originals,
        helpers=[],
        collections=[],
        attempts=[],
        trains=[],
        predictions=[],
        children=[],
        budgets=[],
        on_train=None,
        before_helper=None,
    )
    original_train = ActionMemoryLearner.train_batch
    original_predict = ActionMemoryLearner.predict
    original_fork = ActionMemoryLearner.fork
    original_helper = pilot.run_fresh_task_probe
    original_budget = pilot.ToyBudgetSession
    original_import = builtins.__import__
    clock = lambda: 1.0

    def no_torch(name: str, *args: Any, **options: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            pytest.fail("count assay imported Torch")
        return original_import(name, *args, **options)

    def collect(check_boundary, reports):
        check_boundary()
        data = reserved_data()
        work["collections"].append(data)
        reports.extend(
            dict(
                status="complete",
                role=role,
                context="E-" + cue,
                cue=cue,
                environment_steps_attempted=9,
                environment_steps=9,
                failure_work_may_be_unobserved=False,
            )
            for role in ("probe_train", "probe_test")
            for cue in ("blue", "red")
        )
        callback = work.get("on_collect")
        if callback is not None:
            callback(data)
        check_boundary()
        return data

    def budget(*args, **options):
        session = original_budget(*args, **options)
        work["budgets"].append(session)
        return session

    def helper(source, state, training, evaluation, **options):
        entry = dict(
            source=source, state=state, training=training, evaluation=evaluation, options=options
        )
        work["helpers"].append(entry)
        callback = work["before_helper"]
        if callback is not None:
            callback(entry)
        result = original_helper(source, state, training, evaluation, **options)
        entry["report"] = result
        return result

    def fork(self):
        child = original_fork(self)
        work["children"].append(child)
        return child

    def train(self, features, targets):
        before = self.snapshot_state()
        work["attempts"].append((self, before, features.copy(), targets.copy()))
        if work.get("fail_attempt") == len(work["attempts"]):
            raise RuntimeError("engineered probe update failed")
        result = original_train(self, features, targets)
        work["trains"].append(
            (self, before, self.snapshot_state(), features.copy(), targets.copy())
        )
        callback = work["on_train"]
        if callback is not None:
            callback(self)
        return result

    def predict(self, features):
        result = original_predict(self, features)
        work["predictions"].append((self, self.snapshot_state(), features.copy(), result.copy()))
        return result

    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot.reserved, "_collect_probe_sources", collect)
    monkeypatch.setattr(pilot, "ToyBudgetSession", budget)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", helper)
    monkeypatch.setattr(ActionMemoryLearner, "fork", fork)
    monkeypatch.setattr(ActionMemoryLearner, "train_batch", train)
    monkeypatch.setattr(ActionMemoryLearner, "predict", predict)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    return work


def test_should_describe_the_fixed_assay_without_source_resource_or_collection_work(monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "ToyBudgetSession", forbidden)
    monkeypatch.setattr(pilot.reserved, "_collect_probe_sources", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    protocol = pilot.describe_protocol()
    assert protocol == pilot.describe_protocol()
    encoded = str(protocol)
    assert "observation_memory" in encoded and "296" in encoded and "155136" in encoded
    assert "16" in encoded and "128" in encoded


def test_should_complete_cold_probe_then_preserve_the_mature_atomic_capacity_negative(assay):
    before = pickle.dumps(assay["inputs"])
    result = pilot.run_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert len(result["cohorts"]) == len(assay["helpers"]) == 2
    cold, mature = result["cohorts"]
    assert [row["source_updates"] for row in result["cohorts"]] == [0, 128]
    assert [row["native_training_examples_before"] for row in result["cohorts"]] == [0, 1152]
    assert cold["status"] == "complete" and mature["status"] == "failed"
    assert "capacity" in mature["reason"]
    assert [row["charged_updates"] for row in result["cohorts"]] == [16, 1]
    assert all(row["source_unchanged"] is True for row in result["cohorts"])
    assert all(row["budget_binding_preserved"] is True for row in result["cohorts"])
    assert pickle.dumps(assay["inputs"]) == before
    cold_report, mature_report = cold["report"], mature["report"]
    assert (
        cold_report["attempted_updates"],
        cold_report["completed_updates"],
        cold_report["completed_training_examples"],
    ) == (16, 16, 144)
    assert cold_report["score_forward_calls_completed"] == 8
    assert [point["updates"] for point in cold_report["checkpoints"]] == [0, 4, 8, 16]
    assert (
        cold_report["checkpoints"][-1]["scores"]["E-blue"]["accuracy"]
        == cold_report["checkpoints"][-1]["scores"]["E-red"]["accuracy"]
        == 1
    )
    assert cold_report["decision_diagnostics_enabled"] is True
    assert (
        mature_report["attempted_updates"],
        mature_report["completed_updates"],
        mature_report["attempted_training_examples"],
        mature_report["completed_training_examples"],
    ) == (2, 1, 18, 9)
    assert mature_report["score_forward_calls_completed"] == 2
    assert [point["updates"] for point in mature_report["checkpoints"]] == [0]
    assert len(assay["trains"]) == 17 and len(assay["attempts"]) == 18
    assert len(assay["predictions"]) == 10 and len(assay["collections"]) == 1
    assert len(assay["budgets"]) == 2
    assert [budget.updates_completed for budget in assay["budgets"]] == [16, 1]
    assert all(
        budget.budget.max_training_updates == 16 and budget.started_at == 1.0
        for budget in assay["budgets"]
    )
    assert len(assay["children"]) == 2
    end_states = [child.snapshot_state() for child in assay["children"]]
    assert [
        (state.completed_updates, state.training_examples, len(state.entries))
        for state in end_states
    ] == [(16, 144, 18), (129, 1161, 63)]
    for row, state in zip(result["cohorts"], end_states, strict=True):
        assert row["end_snapshot"] == asdict(state)
    assert end_states[1].entries[:54] == assay["inputs"][1].entries
    # All actual updates use reserved TRAIN labels; scoring uses separate arrays.
    for helper in assay["helpers"]:
        assert helper["source"].snapshot_state() == helper["state"]
        assert helper["options"]["decision_diagnostics"] is True
        assert [packet.context for packet in helper["training"]] == ["E-blue", "E-red"] * 8
        assert all(
            packet.role == "probe_train" and packet.features.shape == (9, 296)
            for packet in helper["training"]
        )
        assert all(
            packet.role == "probe_test" and packet.features.shape == (9, 296)
            for packet in helper["evaluation"]
        )
        assert set(row for packet in helper["training"] for row in packet.row_ids).isdisjoint(
            row for packet in helper["evaluation"] for row in packet.row_ids
        )
        for left in helper["training"]:
            for right in helper["evaluation"]:
                assert not np.shares_memory(left.features, right.features) and not np.shares_memory(
                    left.targets, right.targets
                )


def test_should_stop_after_a_first_cohort_failure_and_keep_every_paid_prefix(assay):
    assay["fail_attempt"] = 3
    before = pickle.dumps(assay["inputs"])
    result = pilot.run_pilot(assay["inputs"])
    first, second = result["cohorts"]
    assert result["status"] == "incomplete" and first["status"] == "failed"
    assert second["status"] == "not_run" and second["report"] is None
    assert first["charged_updates"] == 2
    assert first["report"]["attempted_updates"] == 3 and first["report"]["completed_updates"] == 2
    assert (
        first["report"]["attempted_training_examples"] == 27
        and first["report"]["completed_training_examples"] == 18
    )
    assert first["end_snapshot"]["completed_updates"] == 2
    assert len(assay["helpers"]) == len(assay["children"]) == 1
    assert pickle.dumps(assay["inputs"]) == before


@pytest.mark.parametrize(
    "target", ["source_snapshot", "probe_data", "budget_limit", "budget_clock"]
)
def test_should_refuse_original_snapshot_data_or_budget_mutation_after_a_paid_update(assay, target):
    def mutate(owner):
        if len(assay["trains"]) != 1:
            return
        helper = assay["helpers"][0]
        if target == "source_snapshot":
            object.__setattr__(assay["inputs"][1], "completed_updates", 127)
        elif target == "probe_data":
            helper["training"][0].features[0, 0] = 0.875
        elif target == "budget_limit":
            object.__setattr__(helper["options"]["budget"].budget, "max_training_updates", 15)
        else:
            helper["options"]["budget"].last_clock = 2.0

    assay["on_train"] = mutate
    result = pilot.run_pilot(assay["inputs"])
    first, second = result["cohorts"]
    assert result["status"] == "incomplete" and first["status"] == "failed"
    assert second["status"] == "not_run" and len(assay["helpers"]) == 1
    assert first["report"]["completed_updates"] == 1
    assert first["budget_updates_observed"] == 1
    if target in ("budget_limit", "budget_clock"):
        # The native commit and raw counter survive; damaged original bindings
        # cannot certify a new charge lower bound after the last valid boundary.
        assert first["charged_updates"] == 0
        assert first["charge_readback_unknown"] is True
    else:
        assert first["charged_updates"] == 1
    assert first["report"]["completed_training_examples"] == 9
    assert len(assay["trains"]) == 1


def test_should_refuse_mutated_preceding_probe_report_before_the_next_cohort_can_work(assay):
    def mutate_prior(helper):
        if len(assay["helpers"]) == 2:
            assay["helpers"][0]["report"]["completed_updates"] = 0

    assay["before_helper"] = mutate_prior
    result = pilot.run_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert len(assay["trains"]) == 16
    assert len(assay["predictions"]) == 8
    assert result["cohorts"][1]["status"] == "failed"
    assert result["cohorts"][0]["report"]["completed_updates"] == 16


def test_should_validate_the_preceding_native_success_before_more_probe_work(assay):
    def mutate_prior(helper):
        if len(assay["helpers"]) == 2:
            child = assay["helpers"][0]["source"].child
            assert child is not None
            child.restore_state(assay["inputs"][0])

    assay["before_helper"] = mutate_prior
    result = pilot.run_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert len(assay["trains"]) == 16 and len(assay["predictions"]) == 8
    assert result["cohorts"][0]["report"]["completed_updates"] == 16
    assert result["cohorts"][1]["status"] == "failed"


def test_should_refuse_wrong_reserved_roles_before_any_native_probe_work(assay):
    def change_role(data):
        object.__setattr__(data[0][0], "role", "final_test")

    assay["on_collect"] = change_role
    result = pilot.run_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert all(row["status"] == "not_run" and row["report"] is None for row in result["cohorts"])
    assert not assay["helpers"] and not assay["attempts"] and not assay["predictions"]


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "one",
        "reversed",
        "old_geometry",
        "small_cap",
        "wrong_origin",
        "wrong_rows",
        "extra_field",
    ],
)
def test_should_refuse_invalid_snapshot_input_before_resources_or_collection(monkeypatch, fault):
    supplied: Any = snapshots()
    if fault == "list":
        supplied = list(supplied)
    elif fault == "one":
        supplied = supplied[:1]
    elif fault == "reversed":
        supplied = supplied[::-1]
    elif fault in ("old_geometry", "small_cap"):
        policy = (
            ActionMemoryPolicy()
            if fault == "old_geometry"
            else ActionMemoryPolicy(feature_dim=296, num_classes=7)
        )
        supplied = (ActionMemoryLearner(policy).snapshot_state(), supplied[1])
    elif fault == "wrong_origin":
        object.__setattr__(supplied[1], "completed_updates", 127)
    elif fault == "wrong_rows":
        object.__setattr__(supplied[1], "training_examples", 1151)
    else:
        object.__setattr__(supplied[1], "extra", 1)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot.reserved, "_collect_probe_sources", forbidden)
    with pytest.raises(ValueError):
        pilot.run_pilot(supplied)


def test_should_preserve_actual_charges_when_helper_loses_its_report(assay, monkeypatch):
    def lost(source, state, training, evaluation, **options):
        child = source.fork()
        child.restore_state(state)
        child.train_batch(training[0].features.copy(), training[0].targets.copy())
        options["budget"].record_update()
        raise RuntimeError("engineering paid helper report lost")

    monkeypatch.setattr(pilot, "run_fresh_task_probe", lost)
    result = pilot.run_pilot(assay["inputs"])
    first, second = result["cohorts"]
    assert first["status"] == "failed" and first["report"] is None
    assert first["charged_updates"] == 1
    assert first["unreported_native_work_may_be_unobserved"] is True
    assert second["status"] == "not_run"
    assert result["work"]["attempted_storage_updates"] is None
    assert result["work"]["score_calls_completed"] is None
    assert result["work"]["observed_charged_updates_lower_bound"] == 1


def test_should_recheck_budget_after_the_terminal_source_readback(assay, monkeypatch):
    helper = pilot.run_fresh_task_probe

    def wrap(source, *args, **options):
        result = helper(source, *args, **options)
        original = source.snapshot_state
        calls = []

        def snapshot():
            observed = original()
            calls.append(1)
            # The last source callback of the second post-return guard.
            if len(calls) == 4:
                object.__setattr__(options["budget"].budget, "max_training_updates", 15)
            return observed

        monkeypatch.setattr(source, "snapshot_state", snapshot)
        return result

    monkeypatch.setattr(pilot, "run_fresh_task_probe", wrap)
    result = pilot.run_pilot(assay["inputs"])
    first, second = result["cohorts"]
    assert first["status"] == "failed"
    assert first["budget_binding_preserved"] is False
    assert first["report"]["completed_updates"] == first["charged_updates"] == 16
    assert second["status"] == "not_run"


def test_should_mark_interrupted_collection_counts_as_lower_bounds(assay, monkeypatch):
    def collect(check, reports):
        reports.append(dict(status="complete", environment_steps=9))
        raise RuntimeError("engineering next constructor failure before report")

    monkeypatch.setattr(pilot.reserved, "_collect_probe_sources", collect)
    result = pilot.run_pilot(assay["inputs"])
    assert result["status"] == "incomplete"
    assert result["work"]["teacher_steps_returned"] == 9
    assert result["work"]["teacher_counts_are_lower_bounds"] is True
    assert not assay["helpers"] and not assay["attempts"]


def test_should_not_count_a_rejected_counter_jump_as_a_verified_charge(assay):
    def jump(owner):
        if len(assay["trains"]) == 1:
            # The helper will record the real commit after this callback.
            assay["budgets"][0].updates_completed = 1

    assay["on_train"] = jump
    result = pilot.run_pilot(assay["inputs"])
    first, second = result["cohorts"]
    assert first["status"] == "failed" and second["status"] == "not_run"
    assert first["report"]["completed_updates"] == 1
    assert first["budget_updates_observed"] == 2
    assert first["charged_updates"] == 0
    assert first["charge_readback_unknown"] is True
    assert result["work"]["observed_charged_updates_lower_bound"] == 0


def test_should_stop_when_the_final_charge_readback_changes_its_budget(assay, monkeypatch):
    helper = pilot.run_fresh_task_probe

    def wrap(source, *args, **options):
        result = helper(source, *args, **options)
        child = source.child
        original = child.snapshot_state
        calls = []

        def snapshot():
            observed = original()
            calls.append(1)
            if len(calls) == 4:
                object.__setattr__(options["budget"].budget, "max_training_updates", 15)
            return observed

        monkeypatch.setattr(child, "snapshot_state", snapshot)
        return result

    monkeypatch.setattr(pilot, "run_fresh_task_probe", wrap)
    result = pilot.run_pilot(assay["inputs"])
    first, second = result["cohorts"]
    assert first["status"] == "failed" and first["budget_binding_preserved"] is False
    assert first["charge_readback_unknown"] is True
    assert first["report"]["completed_updates"] == first["charged_updates"] == 16
    assert second["status"] == "not_run"
