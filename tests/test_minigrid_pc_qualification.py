"""One-method qualification routing with accepted TRAIN stubs and fake ports.

The real fit/probe helpers execute; only native admission and the already tested
128-update TRAIN arm are replaced. No neural update or real physics is scheduled.
"""

from copy import deepcopy
import json
import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeaturePolicy
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_learning_pilot import (
    Learner,
    POINTS,
    evaluator as evaluator,
    pilot_environment,
    same_state,
)
from test_minigrid_learnability import ProbeEnvironment
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


METHOD = "public_invariant_pc"
FLAGS: dict[str, Any] = dict(public_history=True, invariant_history=True, pc_qualification=True)


def environment(monkeypatch, evaluator, *, method=METHOD, acquisition_only=False):
    selected_method = method
    captured = pilot_environment(monkeypatch, evaluator)
    work: dict[str, Any] = dict(rows=[], fit=[], probes=[], slots=[], arms=[], factory_calls=[])
    clock = lambda: 1.0
    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(pilot.shared, "monotonic", clock)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "_same_state", same_state)
    monkeypatch.setattr(ProbeEnvironment, "work", evaluator.work)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", ProbeEnvironment)
    monkeypatch.setattr(
        pilot, "CueKeyDoorTeacher", lambda: SimpleNamespace(choose_action=lambda public: 0)
    )
    monkeypatch.setattr(pilot, "read_native_activations", forbidden)
    score, probe, run_probe, new_row = (
        pilot._score_checkpoint,
        pilot.run_fresh_task_probe,
        pilot._run_probe,
        pilot._new_row,
    )

    def counter(state, point, method):
        assert method == selected_method and state.policy.method == selected_method
        assert type(point) is int and point in POINTS
        assert state.completed_updates == point and state.training_examples == 9 * point

    def snapshot(state, point, method):
        counter(state, point, method)
        return deepcopy(state)

    def factory(method):
        assert method == selected_method
        work["factory_calls"].append(method)
        return Learner(
            InvariantTorchFeaturePolicy(
                TorchFeaturePolicy(
                    selected_method.removeprefix("public_invariant_"), 296, 8, 17, num_classes=7
                )
            )
        )

    def row(method):
        value = new_row(method)
        work["rows"].append(value)
        return value

    def accepted_train(row, packets, sampler, deadline, check_boundary, *, learner_factory):
        assert row["method"] == selected_method
        assert len(packets) == 128 and all(packet.features.shape == (9, 296) for packet in packets)
        expected_contexts = (
            ["A"] * 128
            if acquisition_only
            else (["A"] * 32 + ["B"] * 32 + ["C_train"] * 32 + ["A"] * 32)
        )
        assert [packet.context for packet in packets] == expected_contexts
        check_boundary()
        source = learner_factory(selected_method)
        initial = source.snapshot_state()
        states = []
        for point in POINTS:
            state = source.snapshot_state()
            state.completed_updates, state.training_examples = point, 9 * point
            states.append((point, state))
        budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=128, max_wall_seconds=deadline - clock()), clock
        )
        for _ in range(128):
            budget.before_update()
            budget.record_update()
        stream = SimpleNamespace(completed_updates=128, _states=states)
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
            checkpoint_updates=list(POINTS),
            stream_result=dict(
                completed_updates=128,
                training_examples=1152,
                score_forward_calls=0,
                evaluation_scope="training_only",
                checkpoints=[],
            ),
            wake_calls_attempted=128,
            wake_calls_returned=128,
        )
        frozen_states = pickle.dumps(states)
        frozen_initial = pickle.dumps(initial)

        def guard():
            assert budget.updates_completed == 128
            assert pickle.dumps(states) == frozen_states
            assert pickle.dumps(source.snapshot_state()) == frozen_initial

        arm = dict(row=row, source=source, states=states, stream=stream, budget=budget, guard=guard)
        captured["streams"].append((stream, source, deepcopy(packets), budget))
        captured["finished"].append(stream)
        work["arms"].append(arm)
        guard()
        return arm

    def readback(row, source, initial, stream, budget):
        assert source.state.completed_updates == source.state.training_examples == 0
        assert same_state(initial, source.snapshot_state())
        assert stream.completed_updates == budget.updates_completed == 128

    def fit(source, state, **options):
        assert len(captured["finished"]) == 1
        assert options["method"] == selected_method
        work["fit"].append((options["updates"], source, deepcopy(state)))
        return score(source, state, **options)

    def native_probe(source, state, training, evaluation, **options):
        assert len(captured["finished"]) == 1
        assert len(work["fit"]) == 5
        assert options["budget"].updates_completed == 0
        call = dict(
            source=source,
            state=state,
            budget=options["budget"],
            training=training,
            evaluation=evaluation,
            source_before=pickle.dumps(source.snapshot_state()),
            state_before=pickle.dumps(state),
            inputs_before=pickle.dumps((training, evaluation)),
        )
        work["probes"].append(call)
        try:
            value = probe(source, state, training, evaluation, **options)
            call["report"] = deepcopy(value)
            return value
        finally:
            call.update(
                source_after=pickle.dumps(source.snapshot_state()),
                state_after=pickle.dumps(state),
                inputs_after=pickle.dumps((training, evaluation)),
            )

    def selected_probe(slot, *args, **options):
        work["slots"].append(slot)
        return run_probe(slot, *args, **options)

    def collected(game):
        assert len(captured["finished"]) == 1

    evaluator.work["on_probe_collection"] = collected
    monkeypatch.setattr(pilot, "_history_learner", factory)
    monkeypatch.setattr(pilot, "_invariant_snapshot", snapshot)
    monkeypatch.setattr(pilot.shared, "_invariant_feature_counter", counter)
    monkeypatch.setattr(pilot.shared, "_train_arm", accepted_train)
    monkeypatch.setattr(pilot.shared, "_readback", readback)
    monkeypatch.setattr(pilot, "_new_row", row)
    monkeypatch.setattr(pilot, "_score_checkpoint", fit)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", native_probe)
    monkeypatch.setattr(pilot, "_run_probe", selected_probe)
    return SimpleNamespace(captured=captured, work=work, native=evaluator.work)


def test_should_describe_one_fixed_pc_case_without_creating_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    protocol = pilot.describe_protocol(**FLAGS)
    assert protocol["protocol_id"] == "minigrid_invariant_pc_learnability_qualification_v1"
    assert protocol["methods"] == [METHOD]
    assert protocol["primary_comparisons"] == []
    assert protocol["checkpoint_updates"] == list(POINTS)
    assert protocol["probe"]["source_updates"] == [0, 128]
    assert protocol["probe"]["updates"] == 16 and protocol["probe"]["checkpoint_updates"] == [
        0,
        4,
        8,
        16,
    ]
    expected = dict(
        benchmark_native_train_calls=128,
        benchmark_training_rows=1152,
        probe_native_train_calls=32,
        probe_training_rows=288,
        native_train_calls=160,
        training_rows=1440,
        known_raw_score_forwards=30,
        known_raw_score_rows=270,
        probe_score_forwards=16,
        probe_score_rows=144,
        score_forwards=46,
        score_rows=414,
        teacher_episodes=10,
        teacher_decisions=90,
        source_environment_steps=90,
        maximum_source_environment_steps=160,
        learner_live_episodes=0,
        final_episodes=0,
        replay_rows=0,
        recurrent_unroll_steps=0,
    )
    assert protocol["planned_work"] == expected
    assert protocol["feature_dim_by_method"] == {METHOD: 296}
    assert protocol["seed"] == 17 and protocol["hidden_dim"] == 8 and protocol["num_classes"] == 7
    assert protocol["learning_rate"] == 0.03 and protocol["inference_steps"] == 2
    assert protocol["inference_learning_rate"] == 0.04
    json.dumps(protocol, allow_nan=False)


@pytest.mark.parametrize(
    "options",
    [
        dict(pc_qualification=None),
        dict(pc_qualification=0),
        dict(pc_qualification=1),
        dict(pc_qualification="yes"),
        dict(pc_qualification=True),
        dict(public_history=True, pc_qualification=True),
        dict(invariant_history=True, pc_qualification=True),
        dict(**FLAGS, gradient_geometry=True),
    ],
)
def test_should_refuse_unfrozen_modes_before_resources(monkeypatch, options):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options)
    with pytest.raises(ValueError):
        pilot.run_pilot(**options)


@pytest.mark.parametrize(
    "options", [{}, dict(public_history=True), dict(public_history=True, invariant_history=True)]
)
def test_should_preserve_original_descriptors_when_qualification_is_false(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, pc_qualification=False
    )


def test_should_finish_one_accepted_train_arm_then_execute_original_fit_and_both_isolated_probes(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator)
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "complete", report["reason"]
    assert len(report["methods"]) == 1 and report["comparisons"] is None
    row = report["methods"][0]
    assert row["method"] == METHOD and row["status"] == "complete"
    assert case.work["factory_calls"] == [METHOD]
    assert case.captured["collection_calls"] == len(case.captured["finished"]) == 1
    assert [point for point, _, _ in case.work["fit"]] == list(POINTS)
    assert [call["state"].completed_updates for call in case.work["probes"]] == [0, 128]
    assert (
        row["charged_updates"] == row["wake_calls_returned"] == 128
        and row["native_training_examples"] == 1152
    )
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert len(case.native["trains"]) == 32
    assert sum(len(x) for _, x, _ in case.native["trains"]) == 288
    assert len(case.native["predicts"]) == 46
    assert sum(len(x) for _, x in case.native["predicts"]) == 414
    assert not case.native.get("environments") and not case.native.get("sessions")
    assert len(case.native["probe_environments"]) == 4
    assert all(
        game.closed and len(game.transitions) == 9 for game in case.native["probe_environments"]
    )
    budgets = [call["budget"] for call in case.work["probes"]]
    assert budgets[0] is not budgets[1] and all(
        b.updates_completed == b.budget.max_training_updates == 16 for b in budgets
    )
    assert all(b is not case.work["arms"][0]["budget"] for b in budgets)
    for call in case.work["probes"]:
        assert call["source_before"] == call["source_after"]
        assert (
            call["state_before"] == call["state_after"]
            and call["inputs_before"] == call["inputs_after"]
        )
        assert call["report"]["completed_training_examples"] == 144
        assert call["report"]["score_forward_calls_completed"] == 8
        assert call["report"]["reserved_data_bytes"] == 384912
    assert row["initial_A_acquired"] is False
    assert row["interpretation"] == "inconclusive_initial_A_fit_incomplete"
    assert all(value == 0 for value in row["signed_metrics"]["ce"].values())
    assert report["work"]["native_train_calls"] == 160 and report["work"]["training_rows"] == 1440
    assert report["work"]["teacher_episodes_observed"] == 10
    assert report["work"]["source_environment_steps_returned"] == 90
    assert report["scientific_advantage_verified"] is False


def test_should_reject_corrupted_preceding_probe_after_a_second_paid_update_and_preserve_original_prefix(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator)

    def corrupt(owner):
        if len(case.native.get("trains", [])) == 17:
            first = case.work["slots"][0]
            first["report"]["completed_updates"] = 0

    case.native["on_train"] = corrupt
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "incomplete" and report["reason"]
    assert report["comparisons"] is None and len(report["methods"]) == 1
    row = report["methods"][0]
    first, second = row["probes"]
    assert first["status"] == first["report"]["status"] == "complete"
    assert first["charged_updates"] == first["report"]["completed_updates"] == 16
    assert first["report"]["completed_training_examples"] == 144
    assert second["status"] == "failed"
    assert second["charged_updates"] == second["report"]["completed_updates"] == 1
    assert second["report"]["completed_training_examples"] == 9
    assert len(case.native["trains"]) == 17
    assert len(case.work["fit"]) == 5 and len(case.work["probes"]) == 2
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert case.work["arms"][0]["budget"].updates_completed == 128
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert report["work"]["native_train_calls"] == 145
    assert report["work"]["training_rows"] == 1305
