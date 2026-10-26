"""Bounded count control routing and one budgeted real-count integration.

The A128 checkpoint tables below are explicit unit fixtures, not observed
training or game evidence. Only the single marked integration uses real count
updates: 64 updates/576 rows and 94 predictions/846 rows. Other tests use fake
native ports with genuine typed snapshots and the production state validators.
"""

from copy import deepcopy
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
    InvariantMemorySnapshot,
    copy_invariant_memory_snapshot,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import (
    ActionMemoryEntry,
    ActionMemoryPolicy,
    ActionMemorySnapshot,
    ACTION_MEMORY_COUNT_DIAGNOSTIC,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from test_minigrid_distinct_probe import FProbeEnvironment
from test_minigrid_learning_pilot import POINTS, evaluator as evaluator, pilot_environment
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


METHOD = "public_invariant_observation_memory"
FLAGS: dict[str, Any] = dict(
    public_history=True,
    invariant_history=True,
    acquisition_only=True,
    distinct_probe=True,
    old_task_readouts=True,
    return_to_a=True,
    memory_control=True,
)
NATIVE_POLICY = ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)
POLICY = InvariantMemoryPolicy(NATIVE_POLICY)


def literal_checkpoint(a_templates, point):
    """Construct exact exposure counts for the stated alternating-A unit input."""
    counts: dict[tuple[float, ...], list[int]] = {}
    if point:
        for packet in a_templates:
            for features, target in zip(packet.features, packet.targets[:, 0], strict=True):
                key = canonicalize_public_history(tuple(float(value) for value in features))
                counts.setdefault(key, [0] * 7)[int(target)] += point // 2
    native = ActionMemorySnapshot(
        NATIVE_POLICY,
        tuple(ActionMemoryEntry(key, tuple(values)) for key, values in counts.items()),
        point,
        9 * point,
    )
    return copy_invariant_memory_snapshot(InvariantMemorySnapshot(POLICY, native))


class FakeMemoryLearner:
    """No native count update or prediction: only valid-state guard fixtures."""

    work: dict[str, Any] = {}

    def __init__(self):
        self.state = InvariantMemorySnapshot(POLICY, ActionMemorySnapshot(NATIVE_POLICY, (), 0, 0))

    @property
    def policy(self):
        return self.snapshot_state().policy

    def snapshot_state(self):
        return copy_invariant_memory_snapshot(self.state)

    def restore_state(self, state):
        self.state = copy_invariant_memory_snapshot(state)

    def fork(self):
        child = FakeMemoryLearner()
        child.restore_state(self.snapshot_state())
        self.work["children"].append(child)
        return child

    def train_batch(self, features, targets):
        before = self.snapshot_state()
        entries = list(before.native_state.entries)
        key = canonicalize_public_history(tuple(float(value) for value in features[0]))
        if not entries:
            entries.append(ActionMemoryEntry(key, (1, 0, 0, 0, 0, 0, 0)))
            values = [0] * 7
        else:
            values = list(entries[0].counts)
        values[int(targets[0, 0])] += len(features)
        entries[0] = ActionMemoryEntry(entries[0].features, tuple(values))
        self.state = InvariantMemorySnapshot(
            before.policy,
            ActionMemorySnapshot(
                before.native_state.policy,
                tuple(entries),
                before.completed_updates + 1,
                before.training_examples + len(features),
            ),
        )
        self.work["trains"].append((self, features.copy(), targets.copy()))
        callback = self.work.get("on_train")
        if callback:
            callback(self)
        return TrainingDiagnostic("engineering_typed_memory_port", 0.5)

    def predict(self, features):
        self.work["predicts"].append((self, features.copy()))
        callback = self.work.get("on_predict")
        if callback:
            callback(self)
        return np.tile(np.array([[0.5, 0.5, 0, 0, 0, 0, 0]], dtype=np.float64), (len(features), 1))


def memory_environment(monkeypatch, evaluator, *, native=False):
    actual_readback = shared._readback
    captured = pilot_environment(monkeypatch, evaluator)
    work: dict[str, Any] = dict(
        rows=[],
        arms=[],
        slots=[],
        probes=[],
        fits=[],
        trains=[],
        predicts=[],
        children=[],
        readbacks=[],
    )
    clock = lambda: 1.0
    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(shared, "monotonic", clock)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "_same_state", forbidden)
    monkeypatch.setattr(shared, "_same_state", forbidden)
    monkeypatch.setattr(FakeMemoryLearner, "work", work)
    monkeypatch.setattr(FProbeEnvironment, "work", evaluator.work)
    monkeypatch.setattr(FProbeEnvironment, "collision", False)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", FProbeEnvironment)
    monkeypatch.setattr(
        pilot, "CueKeyDoorTeacher", lambda: SimpleNamespace(choose_action=lambda public: 0)
    )
    factory, row_factory = pilot._history_learner, pilot._new_row
    score, probe_runner, selected_probe = (
        pilot._score_checkpoint,
        pilot.run_fresh_task_probe,
        pilot._run_probe,
    )

    def learner_factory(method):
        assert method == METHOD
        return factory(method) if native else FakeMemoryLearner()

    def new_row(method):
        row = row_factory(method)
        work["rows"].append(row)
        return row

    def accepted_train(row, packets, sampler, deadline, check_boundary, *, learner_factory):
        assert row["method"] == METHOD and len(packets) == 128
        assert [packet.context for packet in packets] == ["A"] * 128
        assert all(packet.features.shape == (9, 296) for packet in packets)
        check_boundary()
        source = learner_factory(METHOD)
        initial = source.snapshot_state()
        states = tuple((point, literal_checkpoint(packets[:2], point)) for point in POINTS)
        budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=128, max_wall_seconds=deadline - clock()), clock
        )
        for _ in range(128):
            budget.before_update()
            budget.record_update()
        binding = pilot._budget_binding(budget)
        frozen = pickle.dumps((initial, states))
        stream = SimpleNamespace(completed_updates=128, _states=states)
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
            checkpoint_updates=list(POINTS),
            wake_calls_attempted=128,
            wake_calls_returned=128,
            stream_result=dict(
                completed_updates=128,
                training_examples=1152,
                score_forward_calls=0,
                evaluation_scope="training_only",
                checkpoints=[],
            ),
        )

        def guard():
            assert budget.updates_completed == 128 and pilot._budget_binding(budget) == binding
            assert pickle.dumps((source.snapshot_state(), states)) == frozen

        arm = dict(row=row, source=source, states=states, stream=stream, budget=budget, guard=guard)
        work["arms"].append(arm)
        captured["streams"].append((stream, source, deepcopy(packets), budget))
        captured["finished"].append(stream)
        return arm

    def readback(row, source, initial, stream, budget, *, state_equal=None):
        assert state_equal is shared._same_value
        # The prefix stream is a fixture; exercise the real source comparator.
        actual_readback(row, source, initial, None, budget, state_equal=state_equal)
        work["readbacks"].append(state_equal)
        assert source.snapshot_state() == initial
        assert source.snapshot_state().completed_updates == 0
        assert stream.completed_updates == budget.updates_completed == 128

    def fit(source, state, **options):
        assert len(captured["finished"]) == 1
        work["fits"].append(state.completed_updates)
        return score(source, state, **options)

    def fresh_probe(source, state, training, evaluation, **options):
        assert work["fits"] == list(POINTS)
        work["probes"].append(
            dict(
                source=source,
                state=state,
                budget=options["budget"],
                return_budget=options["return_budget"],
                training=training,
                evaluation=evaluation,
                monitors=options["training_readouts"],
                source_before=pickle.dumps(source.snapshot_state()),
            )
        )
        return probe_runner(source, state, training, evaluation, **options)

    def slot(slot, *args, **options):
        work["slots"].append(slot)
        return selected_probe(slot, *args, **options)

    if native:
        train_native, predict_native = (
            InvariantActionMemoryLearner.train_batch,
            InvariantActionMemoryLearner.predict,
        )

        def train(self, features, targets):
            diagnostic = train_native(self, features, targets)
            work["trains"].append((self, features.copy(), targets.copy()))
            return diagnostic

        def predict(self, features):
            probabilities = predict_native(self, features)
            work["predicts"].append((self, features.copy()))
            return probabilities

        monkeypatch.setattr(InvariantActionMemoryLearner, "train_batch", train)
        monkeypatch.setattr(InvariantActionMemoryLearner, "predict", predict)
    monkeypatch.setattr(pilot, "_history_learner", learner_factory)
    monkeypatch.setattr(pilot, "_new_row", new_row)
    monkeypatch.setattr(shared, "_train_arm", accepted_train)
    monkeypatch.setattr(shared, "_readback", readback)
    monkeypatch.setattr(pilot, "_score_checkpoint", fit)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", fresh_probe)
    monkeypatch.setattr(pilot, "_run_probe", slot)
    return SimpleNamespace(work=work, captured=captured)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_refuse_nonboolean_memory_mode_before_resources(monkeypatch, value):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    options = dict(FLAGS, memory_control=value)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**options)


@pytest.mark.parametrize(
    "missing",
    [
        "public_history",
        "invariant_history",
        "acquisition_only",
        "distinct_probe",
        "old_task_readouts",
        "return_to_a",
    ],
)
def test_should_require_every_frozen_recipe_component_before_resources(monkeypatch, missing):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    options = dict(FLAGS)
    options[missing] = False
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**options)


@pytest.mark.parametrize(
    "option,value",
    [
        ("qualification_method", "bp"),
        ("qualification_method", "pc"),
        ("pc_qualification", True),
        ("activation_health", True),
        ("gradient_geometry", True),
        ("train_a_rollout", True),
    ],
)
def test_should_refuse_neural_qualifiers_or_other_modes_before_resources(
    monkeypatch, option, value
):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    options = dict(FLAGS)
    options[option] = value
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**options)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"public_history": True},
        {"public_history": True, "invariant_history": True, "qualification_method": "pc"},
        {
            "public_history": True,
            "invariant_history": True,
            "qualification_method": "bp",
            "acquisition_only": True,
            "distinct_probe": True,
            "old_task_readouts": True,
            "return_to_a": True,
        },
    ],
)
def test_should_preserve_prior_neural_descriptors_when_memory_is_false(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, memory_control=False
    )


def test_should_describe_count_capacity_and_work_without_allocating_neural_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    report = pilot.describe_protocol(**FLAGS)
    assert (
        report["protocol_id"]
        == "minigrid_invariant_memory_control_A_only_distinct_probe_old_task_readouts_a_return_v1"
    )
    assert report["methods"] == [METHOD] and report["primary_comparisons"] == []
    assert report["task_sequence"] == ["A"] * 4
    assert report["checkpoint_updates"] == list(POINTS)
    assert not {
        "hidden_dim",
        "trainable_parameters",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "cpc_configs",
    }.intersection(report)
    memory = report["memory_control"]
    assert memory["maximum_entries"] == 64 and memory["maximum_payload_bytes"] == 155136
    assert memory["entry_payload_bytes"] == 2424
    assert memory["native_updates_are_sgd"] is False
    assert memory["update_definition"] == ACTION_MEMORY_COUNT_DIAGNOSTIC
    assert report["planned_work"]["count_update_calls"] == 192
    assert report["planned_work"]["count_training_rows"] == 1728
    assert report["planned_work"]["neural_sgd_updates"] == 0
    assert report["planned_work"]["score_forwards"] == 94
    assert report["planned_work"]["score_rows"] == 846
    assert (
        report["planned_work"]["final_episodes"]
        == report["planned_work"]["learner_live_episodes"]
        == 0
    )


def test_should_run_the_real_bounded_count_probe_and_return_with_literal_a_checkpoint_fixtures(
    monkeypatch, evaluator
):
    case = memory_environment(monkeypatch, evaluator, native=True)
    before = pickle.dumps(evaluator.sources)
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "complete", report["reason"]
    assert report["protocol"]["methods"] == [METHOD] and report["comparisons"] is None
    row = report["methods"][0]
    assert row["method"] == METHOD and row["charged_updates"] == 128
    assert row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert case.work["fits"] == list(POINTS)
    assert len(case.work["trains"]) == 64 and sum(len(x) for _, x, _ in case.work["trains"]) == 576
    assert len(case.work["predicts"]) == 94 and sum(len(x) for _, x in case.work["predicts"]) == 846
    assert [call["state"].completed_updates for call in case.work["probes"]] == [0, 128]
    budgets = [
        budget for call in case.work["probes"] for budget in (call["budget"], call["return_budget"])
    ]
    assert len({id(budget) for budget in budgets}) == 4 and all(
        budget.updates_completed == 16 for budget in budgets
    )
    for index, slot in enumerate(row["probes"]):
        assert slot["charged_updates"] == slot["return_charged_updates"] == 16
        assert (
            slot["report"]["completed_training_examples"]
            == slot["report"]["return_phase"]["completed_training_examples"]
            == 144
        )
        assert all(
            diagnostic["definition"] == ACTION_MEMORY_COUNT_DIAGNOSTIC
            for diagnostic in slot["report"]["diagnostics"]
        )
        assert slot["outcomes"]["curve"][-1]["accuracy"] == 1.0
        assert slot["return_fit_outcomes"]["A"]["curve"][-1]["exact_sequence_accuracy"] == 1.0
        assert slot["old_task_readout_outcomes"]["curve"][-1]["exact_sequence_accuracy"] == float(
            index
        )
        native = case.work["trains"][index * 32 + 31][0].snapshot_state().native_state
        assert len(native.entries) == 19 and len(native.entries) * 2424 <= 155136
        assert sum(sum(entry.counts) for entry in native.entries) == 288 + index * 1152
        assert all(np.all(y == 0) for _, _, y in case.work["trains"][index * 32 : index * 32 + 16])
        assert all(
            np.all(y == 1) for _, _, y in case.work["trains"][index * 32 + 16 : index * 32 + 32]
        )
    assert report["work"]["native_train_calls"] == 192 and report["work"]["training_rows"] == 1728
    assert (
        report["work"]["count_update_calls"] == 192
        and report["work"]["count_training_rows"] == 1728
    )
    assert report["work"]["neural_sgd_updates"] == 0
    assert report["work"]["total_score_forwards_completed"] == 94
    assert row["readouts"][0]["table_observation"]["entries"] == 0
    assert row["readouts"][-1]["table_observation"]["entries"] == 1
    assert row["readouts"][-1]["table_observation"]["logical_payload_bytes"] == 2424
    assert row["readouts"][-1]["table_observation"]["conflicting_keys"] == 0
    assert [slot["source_table_observation"]["entries"] for slot in row["probes"]] == [0, 1]
    assert report["probe_data"]["support"]["all_canonical_inputs_unseen"] is True
    assert pickle.dumps(evaluator.sources) == before
    assert report["scientific_advantage_verified"] is False
