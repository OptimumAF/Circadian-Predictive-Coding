"""Qualified H32 native reads retain the original paid and external-port fences.

These engineering fixtures declare valid endpoint clocks without training. The
existing probability port is scripted, and the two-step environment is a fake:
there are no native predictions, native updates, or actual game trajectories.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_probe_rollout as live
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.app.probe_endpoint_observer import observe_endpoint
from src.app.probe_return_phase import budget_binding
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_minigrid_wide_probe_evaluation import case as case
from test_minigrid_wide_probe_evaluation import evaluate
from test_minigrid_wide_probe_games import declared_state, paid_report


def enclosure(current):
    """Use the ordinary app enclosure, including report restoration and latch."""
    limits = ToyExecutionBudget(max_training_updates=16, max_wall_seconds=20)
    first = ToyBudgetSession(limits, lambda: 10.0)
    second = ToyBudgetSession(replace(limits), lambda: 10.0)
    first.updates_completed = 16
    second.updates_completed = 0 if current.phase == "after_probe" else 16
    report = dict(status="complete", checkpoints=[dict(updates=16, scores=[0.25])])
    value = SimpleNamespace(
        first=first,
        second=second,
        paid=report,
        trusted=deepcopy(report),
        actor=["unchanged deployed actor"],
        rng=require_torch().get_rng_state().clone(),
        report=None,
        boundary=None,
    )

    def parent():
        current.guard()
        if value.actor != ["unchanged deployed actor"]:
            raise ValueError("deployed actor changed during qualified read")
        if not require_torch().equal(require_torch().get_rng_state(), value.rng):
            raise ValueError("global RNG changed during qualified read")

    def run(observer=None):
        def read(owner, phase, boundary):
            assert owner is current.source and phase == current.phase
            value.boundary = boundary
            if observer is None:
                value.report = evaluate(current, qualified_boundary=boundary)
            else:
                observer(boundary)

        observe_endpoint(
            read,
            current.source,
            current.phase,
            report,
            budget=first,
            return_budget=second,
            fixed_budgets=(first, budget_binding(first), deepcopy(limits), budget_binding(second)),
            return_limits=deepcopy(second.budget),
            check_boundary=parent,
        )
        return value.report

    value.run = run
    return value


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("origin,phase", [(0, "after_probe"), (128, "after_return")])
def test_should_preserve_all_actions_reports_and_paid_clocks_when_reads_are_qualified(
    case, method, origin, phase
):
    current = case(method, origin, phase)
    original = evaluate(current)
    current.work.clear()
    boundary = enclosure(current)
    qualified = boundary.run()
    assert qualified == original
    assert qualified["work"]["live_prediction_calls_returned"] == 8
    assert qualified["work"]["environment_steps_returned"] == 8
    assert [episode["context"] for episode in qualified["episodes"]] == [
        "A",
        "A",
        "F-blue",
        "F-red",
    ]
    assert all(
        decision["action"] == 0 and decision["probabilities"] == [1 / 7] * 7
        for episode in qualified["episodes"]
        for decision in episode["predictions"]
    )
    assert boundary.paid == boundary.trusted
    assert boundary.first.updates_completed == 16
    assert boundary.second.updates_completed == (0 if phase == "after_probe" else 16)
    assert boundary.first.started_at == boundary.second.started_at == 10.0
    assert current.budget.updates_completed == current.budget.replay_examples_completed == 0
    assert not current.work.get("native_returns")


def mutate_parent(boundary, target):
    if target == "paid_value":
        boundary.paid["checkpoints"][0]["scores"][0] = 0.5
    elif target == "paid_owner":
        boundary.paid["checkpoints"] = deepcopy(boundary.paid["checkpoints"])
    elif target == "actor":
        boundary.actor[0] = "changed deployed actor"
    elif target == "rng":
        state = require_torch().get_rng_state().clone()
        state[0] ^= 1
        require_torch().set_rng_state(state)
    elif target == "original_clock":
        boundary.first.clock = lambda: 10.0
    elif target == "original_counter":
        boundary.first.updates_completed = 15
    else:
        boundary.second.last_clock = 10.25


@pytest.mark.parametrize("port", ["source_snapshot", "scorer_snapshot", "predict", "clock"])
@pytest.mark.parametrize(
    "target",
    [
        "paid_value",
        "paid_owner",
        "actor",
        "rng",
        "original_clock",
        "original_counter",
        "unobserved_clock",
    ],
)
def test_should_refuse_parent_corruption_from_every_qualified_native_port_before_more_work(
    case, monkeypatch, port, target
):
    current = case()
    boundary = enclosure(current)
    fired = []

    def corrupt():
        if fired:
            return
        fired.append(True)
        mutate_parent(boundary, target)

    original_snapshot = InvariantTorchFeatureLearner.snapshot_state

    def snapshot(owner):
        result = original_snapshot(owner)
        selected = (
            owner is current.source if port == "source_snapshot" else owner is not current.source
        )
        if selected and current.work.get("environments"):
            corrupt()
        return result

    if port.endswith("snapshot"):
        monkeypatch.setattr(InvariantTorchFeatureLearner, "snapshot_state", snapshot)
    elif port == "predict":
        current.work["on_predict"] = lambda owner: corrupt()
    else:

        def clock():
            if current.work.get("environments"):
                corrupt()
            return 10.0

        current.budget.clock = clock

    try:
        with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
            boundary.run()
        report = rejected.value.report
        assert fired
        expected = int(port == "predict")
        assert report["work"]["live_prediction_calls_attempted"] == expected
        assert report["work"]["live_prediction_calls_returned"] == expected
        assert report["work"]["environment_steps_returned"] == 0
        assert report["work"]["failure_work_may_be_unobserved"] is True
        assert [episode["status"] for episode in report["episodes"]] == [
            "failed",
            "not_run",
            "not_run",
            "not_run",
        ]
        assert boundary.paid == boundary.trusted
        assert current.work["environments"][0].closed
    finally:
        require_torch().set_rng_state(boundary.rng)


@pytest.mark.parametrize("target", ["source", "scorer", "checkpoint", "gradient", "child_budget"])
def test_should_preserve_full_native_readback_and_child_budget_after_paid_prediction(case, target):
    current = case()
    boundary = enclosure(current)

    def corrupt(owner):
        if target == "child_budget":
            current.budget.clock = lambda: 10.0
        elif target in ("checkpoint", "gradient"):
            native = current.state.native_state
            values = native.gradients if target == "gradient" else native.head_state
            values["bias_hidden"][0, 0] += 0.25
        else:
            selected = current.source if target == "source" else owner
            state = selected.snapshot_state()
            state.native_state.head_state["bias_hidden"][0, 0] += 0.25
            selected.restore_state(state)

    current.work["on_predict"] = corrupt
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        boundary.run()
    report = rejected.value.report
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_returned"] == 0
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert boundary.paid == boundary.trusted
    assert boundary.first.updates_completed == 16 and boundary.second.updates_completed == 0


@pytest.mark.parametrize("target", ["source", "scorer", "checkpoint"])
def test_should_refuse_native_mutation_from_original_clock_after_the_pre_prediction_proof(
    case, monkeypatch, target
):
    current = case()
    boundary = enclosure(current)
    ready, snapshot_seen, fired = [], [], []
    original_encode = evaluation.PublicInitialFrameHistory.encode
    original_snapshot = InvariantTorchFeatureLearner.snapshot_state

    def encode(owner, observation):
        result = original_encode(owner, observation)
        ready.append(True)
        return result

    def snapshot(owner):
        result = original_snapshot(owner)
        if ready and owner is not current.source:
            snapshot_seen.append(True)
        return result

    def clock():
        if snapshot_seen and not fired:
            fired.append(True)
            if target == "checkpoint":
                current.state.native_state.head_state["bias_hidden"][0, 0] += 0.25
            else:
                selected = current.source if target == "source" else current.work["scorers"][0]
                state = original_snapshot(selected)
                state.native_state.head_state["bias_hidden"][0, 0] += 0.25
                selected.restore_state(state)
        return 10.0

    monkeypatch.setattr(evaluation.PublicInitialFrameHistory, "encode", encode)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "snapshot_state", snapshot)
    boundary.second.clock = clock
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        boundary.run()
    assert ready and snapshot_seen and fired
    report = rejected.value.report
    assert report["work"]["live_prediction_calls_attempted"] == 0
    assert report["work"]["environment_steps_returned"] == 0
    assert current.work["environments"][0].closed
    assert boundary.paid == boundary.trusted


def test_should_latch_a_refused_native_read_even_when_delegate_repairs_paid_values(case):
    current = case()
    boundary = enclosure(current)
    errors = []

    def observer(qualify):
        def corrupt():
            boundary.paid["checkpoints"][0]["scores"][0] = 0.5

        try:
            qualify(corrupt)
        except ValueError as error:
            errors.append(error)
        boundary.paid["checkpoints"][0]["scores"][0] = 0.25
        with pytest.raises(ValueError) as second:
            qualify()
        assert second.value is errors[0]

    with pytest.raises(ValueError):
        boundary.run(observer)
    assert len(errors) == 1 and boundary.paid == boundary.trusted
    assert not current.work.get("attempts") and not current.work.get("environments")


@pytest.mark.parametrize("fault", ["omit", "duplicate", "suppress_error"])
def test_should_refuse_a_boundary_that_drops_repeats_or_suppresses_its_native_read(case, fault):
    current = case()
    suppressed = []

    def invalid_boundary(read_ports=None, *, after_clocks=None):
        if read_ports is None or fault == "omit":
            return
        if fault == "duplicate":
            read_ports()
            read_ports()
            return
        current.state.native_state.head_state["bias_hidden"][0, 0] += 0.25
        try:
            read_ports()
        except Exception as error:
            suppressed.append(error)

    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        evaluate(current, qualified_boundary=invalid_boundary)
    assert not current.work.get("attempts") and not current.work.get("environments")
    assert rejected.value.report["work"]["live_prediction_calls_returned"] == 0
    assert rejected.value.report["work"]["failure_work_may_be_unobserved"] is True
    if fault == "suppress_error":
        assert suppressed


def test_should_compose_both_wide_observer_parent_forms_around_the_native_read(monkeypatch):
    calls = []
    reads = []
    monkeypatch.setattr(live, "monotonic", lambda: 1.0)

    def evaluate_endpoint(learner, state, **options):
        calls.append(options)
        options["check_boundary"]()
        options["qualified_boundary"](lambda: reads.append("qualified native read"))
        return paid_report()

    monkeypatch.setattr(live, "evaluate_wide_neural_probe_checkpoint", evaluate_endpoint)
    for qualified in (False, True):
        observer = live.ProbeRollouts(0, wide_method="bp")
        learner = declared_state("bp", 0, 16)

        def boundary(read_ports=None):
            if read_ports is not None:
                read_ports()

        if qualified:
            observer.observe_qualified(learner, "after_probe", boundary)
        else:
            observer.observe(learner, "after_probe", lambda: None)
        assert callable(calls[-1].get("qualified_boundary"))
        observer.check()
        assert observer.records[0]["status"] == "complete"
        assert observer.records[0]["charged_updates"] == 0
    assert reads == ["qualified native read", "qualified native read"]
