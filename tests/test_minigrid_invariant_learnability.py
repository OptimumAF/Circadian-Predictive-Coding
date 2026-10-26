"""Encoding-bound TRAIN fit and isolated E probes on synthetic public episodes.

Exactly four native16-update probes / 576 rows are planned for a full invocation.
All other cases are zero-update admission/score checks. No game, teacher or128-
update benchmark is run, and a probe's CE direction is not assumed in advance.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.app import toy_execution_budget as budgets
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_encoding import frame
from test_torch_feature_learner import assert_tree, tensors
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


RAW = ("bp", "pc", "neutral_cpc", "gated_cpc")
METHODS = tuple("public_invariant_" + method for method in RAW)
PLANNED_NATIVE_UPDATES = 64
PLANNED_NATIVE_ROWS = 576


@pytest.fixture(scope="module", autouse=True)
def native_work_ceiling():
    original = TorchFeatureLearner.train_batch
    patch = pytest.MonkeyPatch()
    work = dict(calls=0, rows=0)

    def count(owner, features, targets):
        work["calls"] += 1
        work["rows"] += len(features)
        assert work["calls"] <= PLANNED_NATIVE_UPDATES and work["rows"] <= PLANNED_NATIVE_ROWS
        return original(owner, features, targets)

    patch.setattr(TorchFeatureLearner, "train_batch", count)
    yield work
    patch.undo()


@pytest.fixture
def case(monkeypatch):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    clock = lambda: 1.0
    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot, "read_native_activations", forbidden)
    return torch, clock


def episode(context, cue, role, identity):
    color = 2 if cue == "blue" else 0
    wall = {"A": 5, "B": 2, "C_train": 0, "E-blue": 4, "E-red": 4}[context]
    frames = tuple(
        frame(
            wall_color=wall,
            direction=i % 4,
            cue_color=color,
            key_color=color,
            door_state=2 if i < 5 else 0,
        )
        for i in range(10)
    )
    return DemonstrationEpisode(
        identity,
        context,
        role,
        tuple(
            DemonstrationTransition(
                index,
                frames[index],
                index % 7,
                0.5 if index == 8 else 0.0,
                frames[index + 1],
                index == 8,
                7,
            )
            for index in range(9)
        ),
        7,
    )


def train_sources():
    return pilot._history_packets(
        tuple(
            pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
            for context in ("A", "B", "C_train")
            for cue in ("blue", "red")
        )
    )


def probe_data():
    train = tuple(
        pilot._packet(
            episode(
                f"E-{cue}",
                cue,
                "probe_train",
                f"minigrid-probe-source-probe_train-{cue}:copy:{index}",
            )
        )
        for index in range(16)
        for cue in (("blue", "red")[index % 2],)
    )
    evaluation = tuple(
        pilot._packet(
            episode(f"E-{cue}", cue, "probe_test", f"minigrid-probe-source-probe_test-{cue}")
        )
        for cue in ("blue", "red")
    )
    return pilot._history_packets(train), pilot._history_packets(evaluation)


def bound(method="bp", **options):
    return InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(
            TorchFeaturePolicy(method, 296, 8, 17, num_classes=7, **options)
        )
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_score_train_without_learning_then_charge_a_separate_original_sixteen_update_probe(
    case, monkeypatch, method
):
    torch, clock = case
    global_rng = torch.get_rng_state().clone()
    numpy_rng = pickle.dumps(np.random.get_state())
    source = pilot._history_learner(method)
    assert type(source) is InvariantTorchFeatureLearner
    assert (
        source.policy.method == method
        and source.policy.encoding_id == INVARIANT_HISTORY_ENCODING_ID
    )
    assert (source.policy.feature_dim, source.policy.hidden_dim, source.policy.num_classes) == (
        296,
        8,
        7,
    )
    assert source.policy.native_policy.seed == 17
    initial = source.snapshot_state()
    checkpoint = source.snapshot_state()
    sources, data = train_sources(), probe_data()
    inputs_before = pickle.dumps((sources, data))
    benchmark = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), clock)
    original_binding = (
        benchmark.budget,
        benchmark.clock,
        benchmark.started_at,
        benchmark.last_clock,
    )
    expected_predictions = [source.predict(packet.features.copy()) for packet in sources]
    fit = pilot._score_checkpoint(
        source,
        checkpoint,
        method=method,
        updates=0,
        sources=sources,
        budget=benchmark,
        check_boundary=lambda: None,
    )
    assert fit["status"] == "complete", fit["reason"]
    assert fit["source_unchanged"] is fit["state_unchanged"] is True
    assert fit["role"] == "train" and fit["scope"] == "teacher_forced_train_fit_not_live_success"
    assert fit["work"]["score_forward_calls_completed"] == 6
    assert fit["work"]["score_rows_completed"] == 54
    assert fit["work"]["internal_projection_rows_returned"] == 54
    assert fit["work"]["internal_projection_array_bytes_returned"] == 54 * 2368
    for score, packet, expected in zip(fit["scores"], sources, expected_predictions, strict=True):
        np.testing.assert_array_equal(score["probabilities"], expected)
        actions = np.argmax(expected, axis=1)
        assert score["predicted_actions"] == actions.tolist()
        assert score["target_actions"] == packet.targets[:, 0].astype(int).tolist()
        ce = float(
            -np.log(
                np.clip(expected[np.arange(9), packet.targets[:, 0].astype(int)], 1e-12, 1)
            ).mean()
        )
        assert score["ce"] == pytest.approx(ce)
        assert score["accuracy"] == pytest.approx(float(np.mean(actions == packet.targets[:, 0])))
    assert benchmark.updates_completed == benchmark.replay_examples_completed == 0
    assert_tree(torch, source.snapshot_state(), initial)
    assert_tree(torch, checkpoint, initial)

    captured: dict[str, Any] = dict(children=[], trains=[])
    original_probe, original_fork, original_train = (
        pilot.run_fresh_task_probe,
        InvariantTorchFeatureLearner.fork,
        InvariantTorchFeatureLearner.train_batch,
    )

    def fork(owner):
        child = original_fork(owner)
        captured["children"].append(child)
        return child

    def train(owner, features, targets):
        captured["trains"].append((owner, features.copy(), targets.copy()))
        return original_train(owner, features, targets)

    def run_probe(owner, state, training, evaluation, **options):
        captured.update(budget=options["budget"], state=state, source=owner)
        assert options["decision_diagnostics"] is True
        assert options["checkpoint_updates"] == (0, 4, 8, 16)
        return original_probe(owner, state, training, evaluation, **options)

    monkeypatch.setattr(InvariantTorchFeatureLearner, "fork", fork)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "train_batch", train)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", run_probe)
    row = pilot._new_row(method)
    slot = row["probes"][0]
    completed = pilot._run_probe(
        slot,
        dict(source=source, row=row, budget=benchmark),
        checkpoint,
        data,
        sampler=Sampler(),
        deadline=10.0,
        check_boundary=lambda: None,
    )
    completed()
    report, probe_budget = slot["report"], captured["budget"]
    assert slot["status"] == report["status"] == "complete"
    assert (
        slot["source_unchanged"]
        is slot["state_unchanged"]
        is slot["budget_binding_preserved"]
        is True
    )
    assert slot["native_updates_before"] == slot["native_training_examples_before"] == 0
    assert (
        slot["charged_updates"]
        == probe_budget.updates_completed
        == report["completed_updates"]
        == 16
    )
    assert (
        report["completed_training_examples"] == 144
        and report["score_forward_calls_completed"] == 8
    )
    assert (
        report["reserved_data_bytes"] == 384912 and report["decision_diagnostics_enabled"] is True
    )
    assert slot["projection_accounting"]["training_rows_returned"] == 144
    assert slot["projection_accounting"]["score_rows_returned"] == 72
    assert slot["projection_accounting"]["array_bytes_returned"] == 216 * 2368
    assert [point["updates"] for point in report["checkpoints"]] == [0, 4, 8, 16]
    assert [point["updates"] for point in slot["outcomes"]["curve"]] == [0, 4, 8, 16]
    assert slot["outcomes"]["ce_gain"] == pytest.approx(
        slot["outcomes"]["curve"][0]["ce"] - slot["outcomes"]["curve"][-1]["ce"]
    )
    assert probe_budget is not benchmark and probe_budget.budget.max_training_updates == 16
    assert probe_budget.started_at == 1.0 and probe_budget.budget.max_wall_seconds == 9.0
    assert probe_budget.replay_examples_completed == 0
    assert len(captured["trains"]) == 16 and sum(len(x) for _, x, _ in captured["trains"]) == 144
    assert all(owner is not source and x.shape == (9, 296) for owner, x, _ in captured["trains"])
    assert len(captured["children"]) == 1 and captured["children"][0] is not source
    child_state = captured["children"][0].snapshot_state()
    assert (child_state.completed_updates, child_state.training_examples) == (16, 144)
    assert all(
        a.data_ptr() != b.data_ptr()
        for a in tensors(torch, child_state)
        for b in tensors(torch, initial)
    )
    assert_tree(torch, source.snapshot_state(), initial)
    assert_tree(torch, checkpoint, initial)
    assert pickle.dumps((sources, data)) == inputs_before
    assert benchmark.updates_completed == 0
    assert (
        benchmark.budget,
        benchmark.clock,
        benchmark.started_at,
        benchmark.last_clock,
    ) == original_binding
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert pickle.dumps(np.random.get_state()) == numpy_rng

    original_report = slot["report"]
    prior_ce = original_report["checkpoints"][0]["scores"]["E-blue"]["ce"]
    original_report["checkpoints"][0]["scores"]["E-blue"]["ce"] = prior_ce + 1
    with pytest.raises(ValueError):
        completed()
    original_report["checkpoints"][0]["scores"]["E-blue"]["ce"] = prior_ce
    completed()
    slot["report"] = deepcopy(original_report)
    with pytest.raises(ValueError):
        completed()
    slot["report"] = original_report
    completed()
    prior_gain = slot["outcomes"]["ce_gain"]
    slot["outcomes"]["ce_gain"] = prior_gain + 1
    with pytest.raises(ValueError):
        completed()
    slot["outcomes"]["ce_gain"] = prior_gain
    completed()
    slot["projection_accounting"]["array_bytes_returned"] = 0
    with pytest.raises(ValueError):
        completed()
    slot["projection_accounting"]["array_bytes_returned"] = 216 * 2368
    completed()
    probe_budget.updates_completed = 15
    with pytest.raises(ValueError):
        completed()
    probe_budget.updates_completed = 16
    completed()
    tensor = tensors(torch, checkpoint)[0]
    tensor_before = tensor.clone()
    tensor.add_(1)
    with pytest.raises(ValueError):
        completed()
    tensor.copy_(tensor_before)
    completed()


@pytest.mark.parametrize(
    "fault",
    [
        "raw_state",
        "raw_label",
        "other_method",
        "wrong_seed",
        "wrong_learning_rate",
        "bool_point",
        "undeclared_point",
        "declared_clock",
        "native_clock",
        "wrapper_extra",
        "fit_role",
    ],
)
def test_should_refuse_unbound_or_mismatched_fit_state_before_callbacks_or_learning(
    case, monkeypatch, fault
):
    torch, clock = case
    source = bound()
    before = source.snapshot_state()
    state: Any = source.snapshot_state()
    sources = train_sources()
    method, point = METHODS[0], 0
    if fault == "raw_state":
        state = state.native_state
    elif fault == "raw_label":
        method = "bp"
    elif fault == "other_method":
        method = METHODS[1]
    elif fault == "wrong_seed":
        altered = replace(state.native_state.policy, seed=18)
        object.__setattr__(state.policy, "native_policy", altered)
        object.__setattr__(state.native_state, "policy", altered)
    elif fault == "wrong_learning_rate":
        state = bound(learning_rate=0.1).snapshot_state()
    elif fault == "bool_point":
        point = False
    elif fault == "undeclared_point":
        point = 1
    elif fault == "declared_clock":
        point = 32
    elif fault == "native_clock":
        object.__setattr__(state.native_state, "completed_updates", True)
    elif fault == "wrapper_extra":
        object.__setattr__(state, "extra", 0)
    else:
        packet = sources[0]
        assert packet.episode is not None
        sources = (
            replace(packet, role="probe_test", episode=replace(packet.episode, role="probe_test")),
            *sources[1:],
        )
    monkeypatch.setattr(InvariantTorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "predict", forbidden)
    with pytest.raises(ValueError):
        pilot._score_checkpoint(
            source,
            state,
            method=method,
            updates=point,
            sources=sources,
            budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), clock),
            check_boundary=forbidden,
        )
    assert_tree(torch, source.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault", ["raw_state", "raw_method", "wrong_method", "point", "bool_point", "wrong_policy"]
)
def test_should_refuse_invalid_probe_checkpoint_before_allocating_an_update_budget(
    case, monkeypatch, fault
):
    _, clock = case
    source = bound()
    state: Any = source.snapshot_state()
    row = pilot._new_row(METHODS[0])
    slot = row["probes"][0]
    if fault == "raw_state":
        state = state.native_state
    elif fault == "raw_method":
        row["method"] = "bp"
    elif fault == "wrong_method":
        row["method"] = METHODS[1]
    elif fault == "point":
        slot["checkpoint_updates"] = 32
    elif fault == "bool_point":
        slot["checkpoint_updates"] = False
    else:
        state = bound(learning_rate=0.1).snapshot_state()
    benchmark = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), clock)
    monkeypatch.setattr(pilot, "ToyBudgetSession", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    with pytest.raises(ValueError):
        pilot._run_probe(
            slot,
            dict(source=source, row=row, budget=benchmark),
            state,
            probe_data(),
            sampler=Sampler(),
            deadline=10.0,
            check_boundary=forbidden,
        )
    assert benchmark.updates_completed == 0


def test_should_describe_the_existing_dose_and_explicit_bound_methods_without_resources(
    monkeypatch,
):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    protocol = pilot.describe_protocol(public_history=True, invariant_history=True)
    assert protocol["methods"] == [*METHODS, "recurrent_bc"]
    assert protocol["checkpoint_updates"] == [0, 32, 64, 96, 128]
    assert protocol["probe"]["source_updates"] == [0, 128]
    assert protocol["probe"]["updates"] == 16 and protocol["probe"]["checkpoint_updates"] == [
        0,
        4,
        8,
        16,
    ]
    assert protocol["planned_work"]["native_train_calls"] == 800
    assert protocol["planned_work"]["training_rows"] == 7200
    assert (
        protocol["planned_work"]["final_episodes"]
        == protocol["planned_work"]["learner_live_episodes"]
        == 0
    )
    assert protocol["feature_dim_by_method"] == {
        method: 148 if method == "recurrent_bc" else 296 for method in (*METHODS, "recurrent_bc")
    }


@pytest.mark.parametrize(
    "options",
    [
        dict(invariant_history=None),
        dict(invariant_history=1),
        dict(invariant_history=True),
        dict(public_history=True, invariant_history=True, gradient_geometry=True),
    ],
)
def test_should_refuse_unfrozen_modes_before_native_resources(monkeypatch, options):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options)
    with pytest.raises(ValueError):
        pilot.run_pilot(**options)


def test_should_leave_both_original_descriptors_identical_when_mode_is_false():
    assert pilot.describe_protocol() == pilot.describe_protocol(invariant_history=False)
    assert pilot.describe_protocol(public_history=True) == pilot.describe_protocol(
        public_history=True, invariant_history=False
    )
