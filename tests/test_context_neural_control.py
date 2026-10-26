"""Scripted workflow and literal native-state admission; not learned outcomes."""

from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_context_control as control
from src.adapters import minigrid_context_torch_feature_learner as port
from src.adapters import minigrid_policy_evaluation as evaluation
from src.app.coupled_attribution import _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.infra import minigrid_memory
from src.shared.torch_runtime import require_torch
from test_arrived_cue_catalogue import sources, episode
from test_torch_feature_learner import NAMES


@dataclass(frozen=True)
class Head:
    completed_updates: int
    training_examples: int


@dataclass(frozen=True)
class State:
    policy: Any
    heads: tuple
    failed_work: Any = None

    @property
    def completed_updates(self):
        return sum(h.completed_updates for h in self.heads)

    @property
    def training_examples(self):
        return sum(h.training_examples for h in self.heads)


@dataclass(frozen=True)
class Receipt:
    update_number: int
    observed_at: int
    arrived_at: int
    applied_at: int


@pytest.fixture
def scripted(monkeypatch):
    box: dict[str, Any] = dict(calls=[])

    class Learner:
        def __init__(self, policy):
            self._state = State(policy, (Head(0, 0), Head(0, 0)))

        def snapshot_state(self):
            return deepcopy(self._state)

        def fork(self):
            return deepcopy(self)

        def restore_state(self, value):
            self._state = deepcopy(value)

        def predict(self, x):
            return np.full((len(x), 7), 1 / 7, dtype=np.float64)

        def train_batch(self, x, y):
            # Scripted counters only: no native model or scientific fit.
            index = int(control.project_features(x)[0, 297])
            heads = list(self._state.heads)
            old = heads[index]
            heads[index] = Head(old.completed_updates + 1, old.training_examples + len(x))
            self._state = State(self._state.policy, tuple(heads))

    class Stream:
        def __init__(self, source, training, heldout, **options):
            assert len(training) == 128 and heldout == ()
            assert options["label_delay"] == 1 and options["training_only"] is True
            assert all(b.role == "train" and len(b.features) == 9 for b in training)
            self.source, self.options, self.n, self.applied_updates = source, options, 0, []
            self.finished = False
            box["stream_owner"] = self
            box["training_budget"] = options["budget"]
            self._runtime = SimpleNamespace(
                actor=SimpleNamespace(
                    snapshot_state=lambda: SimpleNamespace(state=source.snapshot_state())
                ),
                candidate_snapshot=lambda: SimpleNamespace(state=self._candidate_state()),
            )
            options["observe_checkpoint"](0, self.candidate_snapshot(), None)

        def candidate_snapshot(self):
            if self.finished:
                raise ValueError("attribution stream is failed or finished")
            return self._candidate_state()

        def _candidate_state(self):
            n = self.n - 1 if box.get("counter") and self.n == 128 else self.n
            return State(
                self.source._state.policy,
                (Head(n // 2, 9 * (n // 2)), Head(n - n // 2, 9 * (n - n // 2))),
            )

        def run_until(self, n):
            box["calls"].append(n)
            self.options["budget"].record_update()
            self.applied_updates.append(Receipt(n, n - 1, n, n))
            self.n = n
            if n in self.options["checkpoint_updates"]:
                self.options["observe_checkpoint"](n, self.candidate_snapshot(), None)
            if n == 128 and box.get("receipt"):
                self.applied_updates.pop()

        def finish(self):
            self.finished = True
            box["stream"] = dict(
                completed_updates=128,
                training_examples=1152,
                evaluation_scope="training_only",
                score_forward_calls=0,
            )
            return box["stream"]

    actual_paid = control.paid_report_guard

    def paid(value):
        if isinstance(value, dict) and isinstance(value.get("scores"), list):
            box["monitor"] = value
        return actual_paid(value)

    def games(source, state, *, budget, check_boundary):
        box["calls"].append("games")
        assert state.completed_updates == 128 and state.training_examples == 1152
        check_boundary()
        return dict(
            status="complete",
            state_unchanged=True,
            episodes=[
                dict(status="complete", success=False, environment=dict(closed=True))
                for _ in range(2)
            ],
        )

    monkeypatch.setattr(port, "ContextTorchFeatureLearner", Learner)
    monkeypatch.setattr(port, "copy_context_torch_feature_snapshot", deepcopy)
    monkeypatch.setattr(control, "AttributionStream", Stream)
    monkeypatch.setattr(control, "paid_report_guard", paid)
    monkeypatch.setattr(minigrid_memory, "_require_minigrid", lambda: None)
    monkeypatch.setattr(evaluation, "evaluate_context_neural_bp_checkpoint", games)
    return box


def test_should_use_existing_fixed_stream_then_complete_negative_game_readout(scripted):
    observed: list[Any] = []
    report = control.run_neural_a_control(sources()[:2], endpoint_observer=observed.append)
    assert report["status"] == "complete", report["reason"]
    assert scripted["calls"] == [*range(1, 129), "games"]
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert [p["updates"] for p in report["checkpoints"]] == [0, 32, 64, 96, 128]
    assert all(p["status"] == "complete" and len(p["scores"]) == 2 for p in report["checkpoints"])
    assert report["native_endpoint"]["completed_updates"] == 128 and len(observed) == 1
    assert report["training_budget"]["updates_completed"] == 128
    assert report["training_budget"]["initial_limits"] == report["training_budget"]["limits"]
    assert (
        report["readout_budget"]["updates_completed"]
        == report["readout_budget"]["replay_examples_completed"]
        == 0
    )


@pytest.mark.parametrize("fault", ["role", "order", "short", "observer"])
def test_should_refuse_input_or_observer_before_models(monkeypatch, fault):
    packets = sources()[:2]
    options = {}
    if fault == "role":
        packets = (replace(packets[0], role="final_test"), packets[1])
    elif fault == "order":
        packets = packets[::-1]
    elif fault == "short":
        packets = (episode(packets[0].episode_id, rows=8), packets[1])
    else:
        options["endpoint_observer"] = True
    monkeypatch.setattr(port, "ContextTorchFeatureLearner", lambda *a: pytest.fail("model reached"))
    with pytest.raises(ValueError):
        control.run_neural_a_control(packets, **options)


@pytest.mark.parametrize("fault", ["receipt", "counter"])
def test_should_refuse_incomplete_train_before_endpoint_or_games(scripted, fault):
    scripted[fault] = True
    observed: list[Any] = []
    report = control.run_neural_a_control(sources()[:2], endpoint_observer=observed.append)
    assert report["status"] == "failed" and report["games"] is None and observed == []
    assert report["training_budget"]["updates_completed"] == 128


@pytest.mark.parametrize("fault", ["stream", "clock", "monitor", "train_replay", "runtime"])
def test_should_keep_paid_prefix_and_refuse_changed_evidence_before_more_native_work(
    scripted, monkeypatch, fault
):
    def games(source, state, *, budget, check_boundary):
        if fault == "stream":
            scripted["stream"]["training_examples"] = 1153
        elif fault == "clock":
            budget.clock = lambda: 0.0
        elif fault == "train_replay":
            scripted["training_budget"].replay_examples_completed = 1
        elif fault == "runtime":
            scripted["stream_owner"]._runtime = deepcopy(scripted["stream_owner"]._runtime)
        else:
            scripted["monitor"]["scores"][0]["actions"][0] = 6
        check_boundary()
        pytest.fail("changed prior evidence admitted")

    monkeypatch.setattr(evaluation, "evaluate_context_neural_bp_checkpoint", games)
    report = control.run_neural_a_control(sources()[:2])
    assert report["status"] == "failed"
    assert report["stream"]["training_examples"] == 1152
    assert report["checkpoints"][-1]["scores"][0]["actions"][0] == 0


def test_should_admit_real_workflow_budget_and_delay_two_h32_updates_without_games(monkeypatch):
    actual_stream = control.AttributionStream
    captured: dict[str, Any] = {}

    def limited_stream(*args, **kwargs):
        stream = actual_stream(*args, **kwargs)
        captured.update(stream=stream, source=args[0], initial=args[0].snapshot_state())
        budget = kwargs["budget"]
        assert budget.budget.max_replay_examples is None
        assert budget.budget.max_hidden_width is None
        assert budget.budget.max_training_updates == 128
        actual_run = stream.run_until

        def run(n):
            actual_run(n)
            if n == 2:
                raise RuntimeError("intentional software stop after two real workflow updates")

        monkeypatch.setattr(stream, "run_until", run)
        return stream

    monkeypatch.setattr(control, "AttributionStream", limited_stream)
    monkeypatch.setattr(minigrid_memory, "_require_minigrid", lambda: None)
    monkeypatch.setattr(
        evaluation,
        "evaluate_context_neural_bp_checkpoint",
        lambda *a, **k: pytest.fail("incomplete endpoint admitted to games"),
    )
    report = control.run_neural_a_control(sources()[:2])
    assert report["status"] == "failed" and "intentional software stop" in report["reason"]
    assert report["games"] is report["native_endpoint"] is None
    assert report["training_budget"]["updates_completed"] == 2
    assert report["training_budget"]["replay_examples_completed"] == 0
    assert len(report["committed_receipts"]) == 2
    assert report["last_native_state"]["completed_updates"] == 2
    assert report["last_native_state"]["training_examples"] == 18
    state = captured["stream"].candidate_snapshot()
    assert [(h.completed_updates, h.training_examples) for h in state.heads] == [(1, 9), (1, 9)]
    assert _same_value(captured["source"].snapshot_state(), captured["initial"])
    assert _same_value(
        captured["stream"]._runtime.actor.snapshot_state().state, captured["initial"]
    )


def test_should_refuse_missing_runtime_before_native_initialization(monkeypatch):
    def missing():
        raise RuntimeError("missing pinned native runtime")

    monkeypatch.setattr(minigrid_memory, "_require_minigrid", missing)
    monkeypatch.setattr(port, "ContextTorchFeatureLearner", lambda *a: pytest.fail("model reached"))
    report = control.run_neural_a_control(sources()[:2])
    assert report["status"] == "failed" and report["stream"] is report["games"] is None


def test_should_finish_real_stream_then_read_original_runtime_without_reopening(
    scripted, monkeypatch
):
    from src.app.coupled_attribution import AttributionStream

    captured: dict[str, Any] = {}

    def real_stream(*args, **kwargs):
        owner = AttributionStream(*args, **kwargs)
        captured["owner"] = owner
        return owner

    monkeypatch.setattr(control, "AttributionStream", real_stream)
    report = control.run_neural_a_control(sources()[:2])
    assert report["status"] == "complete", report["reason"]
    assert report["stream"]["completed_updates"] == 128
    assert len(report["committed_receipts"]) == 128
    assert report["last_native_state"]["training_examples"] == 1152
    assert "state_capture_error" not in report
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    owner = captured["owner"]
    with pytest.raises(ValueError, match="failed or finished"):
        owner.candidate_snapshot()
    assert owner._runtime.candidate_snapshot().state.completed_updates == 128


def literal_endpoint():
    """Valid shaped literal continuation; no optimizer update or task claim."""
    torch = require_torch()
    owner = port.ContextTorchFeatureLearner(
        port.ContextTorchFeaturePolicy(port.TorchFeaturePolicy("bp", 296, 32, 17, num_classes=7))
    )
    before = owner.snapshot_state()
    heads = []
    for head in before.heads:
        optimizer = deepcopy(head.optimizer_state)
        optimizer["state"] = {
            i: dict(momentum_buffer=torch.zeros_like(head.head_state[name]))
            for i, name in enumerate(NAMES)
        }
        gradients = {name: torch.zeros_like(head.head_state[name]) for name in NAMES}
        heads.append(
            replace(
                head,
                optimizer_state=optimizer,
                gradients=gradients,
                completed_updates=64,
                training_examples=576,
            )
        )
    return owner, replace(before, heads=(heads[0], heads[1]))


def session():
    return ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=150,
            max_process_rss_bytes=768 * 1024**2,
        ),
        lambda: 0.0,
    )


def test_should_bind_literal_neural_endpoint_to_original_known_a_loop(monkeypatch):
    owner, endpoint = literal_endpoint()

    def run(source, state, **options):
        assert source is owner and _same_value(state, endpoint)
        assert options["method"] == "public_context_bp" and options["updates"] == 128
        assert options["sources"] == options["layouts"] == ()
        assert [(l.context, c, r) for l, c, r in options["live_plan"]] == [
            ("A", "blue", "train"),
            ("A", "red", "train"),
        ]
        assert options["state_equal"](state, options["memory_copier"](state))
        assert options["require_initial_source"] is True
        options["extra_check"]()
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    assert (
        evaluation.evaluate_context_neural_bp_checkpoint(
            owner, endpoint, budget=session(), check_boundary=lambda: None
        )["status"]
        == "complete"
    )


@pytest.mark.parametrize("fault", ["rows", "policy", "failed"])
def test_should_refuse_nonqualified_native_geometry_before_game_loop(monkeypatch, fault):
    owner, state = literal_endpoint()
    if fault == "rows":
        state = replace(
            state, heads=(replace(state.heads[0], training_examples=577), state.heads[1])
        )
    elif fault == "failed":
        state = replace(state, failed_work=(0, 0, 0, 0, 0, 0, 0, 0, True))
    else:
        policy = replace(state.policy.native_policy, seed=18)
        state = replace(
            state,
            policy=port.ContextTorchFeaturePolicy(policy),
            heads=tuple(replace(h, policy=policy) for h in state.heads),
        )
    monkeypatch.setattr(
        evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("game loop reached")
    )
    with pytest.raises(ValueError):
        evaluation.evaluate_context_neural_bp_checkpoint(
            owner, state, budget=session(), check_boundary=lambda: None
        )


def test_should_refuse_changed_zero_work_native_budget(monkeypatch):
    owner, state = literal_endpoint()
    budget = session()

    def run(*args, **options):
        budget.clock = lambda: 1.0
        options["extra_check"]()
        pytest.fail("changed native clock admitted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(ValueError, match="budget binding"):
        evaluation.evaluate_context_neural_bp_checkpoint(
            owner, state, budget=budget, check_boundary=lambda: None
        )
