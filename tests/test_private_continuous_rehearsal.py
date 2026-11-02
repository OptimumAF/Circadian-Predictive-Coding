"""#431 literal/tiny full-network rehearsal checks; no original data or RNG."""

from collections import defaultdict
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from src.adapters import continuous_probe as probe
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_probe_rehearsal import _ProbeRehearsalBinding
from src.app.continuous_stream_rehearsal import ContinuousReplayPlan
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core import continuous_bc as bc
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)

WORK: defaultdict[str, int] = defaultdict(int)
CAPS = dict(
    updates=80,
    update_rows=320,
    gradients=160,
    gradient_rows=1000,
    forward_rows=2000,
    solves=80,
    snapshots=20000,
    snapshot_bytes=64 * 1024**2,
)


def charge(key, amount=1):
    WORK[key] += amount
    assert WORK[key] <= CAPS[key], "permanent431fixturecap:" + key


@pytest.fixture(autouse=True)
def meter(monkeypatch):
    def forbidden(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("No initialization/RNG/SGD in431fixture")

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
    for module, name, key, row_key in (
        (bc, "_forward", None, "forward_rows"),
        (bc, "continuous_bc_loss_and_gradients", "gradients", "gradient_rows"),
        (bp.NumpyContinuousBCLearner, "_apply_full_gauss_newton_batch", "updates", "update_rows"),
    ):
        original = getattr(module, name)

        def counted(*args, _original=original, _key=key, _rows=row_key, **kwargs):
            if _key is not None:
                charge(_key)
            charge(_rows, len(args[1] if _key == "updates" else args[0]))
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)
    snapshot, solve = bp._snapshot, np.linalg.lstsq

    def copied(*args, **kwargs):
        charge("snapshots")
        value = snapshot(*args, **kwargs)
        charge("snapshot_bytes", sum(x.nbytes for x in value.parameters + value.velocity))
        return value

    def solved(matrix, *args, **kwargs):
        charge("solves")
        assert max(matrix.shape) <= 64
        return solve(matrix, *args, **kwargs)

    monkeypatch.setattr(bp, "_snapshot", copied)
    monkeypatch.setattr(np.linalg, "lstsq", solved)


def spec():
    return ContinuousActionSpec(2, ("x", "y", "z", "normalized_gripper"), (-1.0,) * 4, (1.0,) * 4)


def source():
    owner = object.__new__(bp.NumpyContinuousBCLearner)
    policy = bp.ContinuousBCPolicy(spec(), hidden_dim=2, seed=17, binary_gripper_axis=3)
    parameters = cast(
        bc.Parameters,
        (
            np.array([[0.2, -0.1], [0.1, 0.3]]),
            np.array([0.05, -0.04]),
            np.array([[0.1, 0.2, -0.15, 0.12], [-0.2, 0.15, 0.2, -0.1]]),
            np.array([0.01, -0.02, 0.03, 0.04]),
        ),
    )
    owner._policy = policy
    owner._state = bp.ContinuousBCSnapshot(
        policy,
        parameters,
        cast(bc.Parameters, tuple(np.full_like(x, 0.01) for x in parameters)),
        2,
        4,
    )
    return owner


def episode(role="probe_train", name=None):
    xs = ((0.2, -0.3), (0.6, 0.1))
    ys = (
        ((0.3, -0.2, 0.1, 1.0), (-0.3, 0.2, 0.1, -1.0))
        if role != "train"
        else ((-0.2, 0.1, 0.2, -1.0), (0.2, -0.1, 0.2, 1.0))
    )
    return ContinuousEpisode(
        name or "literal-" + role,
        "old" if role == "train" else "fresh",
        role,
        spec(),
        tuple(
            ContinuousTransition(i, x, y, 0.0, xs[(i + 1) % 2], i == 1, False)
            for i, (x, y) in enumerate(zip(xs, ys, strict=True))
        ),
    )


def plan(updates=1, optimizer="full-gauss-newton"):
    return probe.ContinuousProbePlan(
        episode(), episode("probe_test"), updates=updates, optimizer=optimizer
    )


def replay(examples=4):
    return (
        ContinuousReplayPlan(1, examples),
        (episode("train"),),
        ToyBudgetSession(ToyExecutionBudget(max_replay_examples=examples), lambda: 0.0),
    )


def run(owner, *, updates=1, examples=4, **options):
    rplan, bank, budget = replay(examples)
    result = probe.run_continuous_probe(
        owner,
        plan(updates),
        training_readouts=bank,
        replay_plan=rplan,
        replay_episodes=bank,
        replay_budget=budget,
        **options,
    )
    return result, budget


def test_should_run_joint_rehearsal_through_original_delayed_runtime_and_keep_source_actor_separate():
    owner = source()
    before = owner.snapshot_state()
    endpoints = []

    def observe(readout, phase, guard):
        guard()
        assert readout is not owner
        state = readout.snapshot_state()
        endpoints.append((phase, state.completed_updates, state.training_examples))
        assert probe.same_continuous_state(owner.snapshot_state(), before)

    result, budget = run(owner, updates=2, endpoint_observer=observe)
    assert result["complete"] is True, result.get("error")
    assert endpoints == [("before_probe", 2, 4), ("after_probe", 4, 12)]
    assert result["progress"] == dict(updates_completed=2, replay_examples_completed=0, next_tick=4)
    assert result["rehearsal"]["charged_exposures"] == budget.replay_examples_completed == 4
    assert result["rehearsal"]["native_attempts"] == result["rehearsal"]["native_returns"] == 2
    assert result["rehearsal"]["bank_episode_ids"] == ["literal-train"]
    assert [s["role"] for s in result["rehearsal"]["row_spans"]] == ["probe_train", "train"]
    assert (
        result["work"]["training_rows_attempted"] == result["work"]["training_rows_completed"] == 8
    )
    assert result["work"]["probe_training_rows_completed"] == 4
    assert [r["training_rows"] for r in result["native_updates"]] == [4, 4]
    assert result["metrics"]["fresh_training_gain"] is not None
    assert set(result["checkpoints"][-1]["scores"]) == {"fresh", "fresh_training", "old_0"}
    assert all(r["episode_id"] == "literal-probe_train" for r in result["applied_updates"])
    assert probe.same_continuous_state(owner.snapshot_state(), before)


def test_should_use_the_explicit_joint_arrays_with_the_unchanged_native_fitter():
    owner = source()
    expected = source()
    rplan, bank, budget = replay()
    x, y = (
        np.asarray([[0.2, -0.3], [0.6, 0.1]]),
        np.asarray([[0.3, -0.2, 0.1, 1.0], [-0.3, 0.2, 0.1, -1.0]]),
    )
    binding = _ProbeRehearsalBinding(episode(), episode("probe_test"), rplan, bank, budget, bank)
    result = binding.update(owner, x, y)
    old = np.asarray([[-0.2, 0.1, 0.2, -1.0], [0.2, -0.1, 0.2, 1.0]])
    direct = expected._apply_full_gauss_newton_batch(
        np.concatenate((x, x)), np.concatenate((y, old))
    )
    assert result.work == direct.work
    assert probe.same_continuous_state(owner.snapshot_state(), expected.snapshot_state())
    assert budget.replay_examples_completed == 2


def test_should_preserve_no_replay_full_fit_and_not_train_on_old_readouts():
    owner = source()
    before = owner.snapshot_state()
    result = probe.run_continuous_probe(owner, plan(), training_readouts=(episode("train"),))
    assert result["complete"] is True, result.get("error")
    assert "rehearsal" not in result
    assert result["native_updates"][0]["training_rows"] == 2
    assert set(result["checkpoints"][-1]["scores"]) == {"fresh", "old_0"}
    assert probe.same_continuous_state(owner.snapshot_state(), before)


def test_should_stop_at_original_replay_allowance_without_refund_or_overshoot():
    result, budget = run(source(), updates=2, examples=2)
    assert result["complete"] is False
    assert budget.replay_examples_completed == 2
    assert result["rehearsal"]["charged_exposures"] == 2
    assert len(result["native_updates"]) == 1
    assert result["rehearsal"]["stopped"] is True
    assert "max_replay_examples" in result["error"]


def test_should_preserve_charged_failed_native_attempt_and_refuse_binding_reuse(monkeypatch):
    owner = source()
    rplan, bank, budget = replay()
    binding = _ProbeRehearsalBinding(episode(), episode("probe_test"), rplan, bank, budget, bank)

    def failed(*args, **kwargs):
        raise RuntimeError("injected paid native failure")

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "_apply_full_gauss_newton_batch", failed)
    x = np.asarray(binding.features)
    y = np.asarray(binding.targets)
    with pytest.raises(RuntimeError, match="injected"):
        binding.update(owner, x, y)
    assert binding.report()["charged_exposures"] == budget.replay_examples_completed == 2
    assert binding.report()["unreturned_native_work_unknown"] is True
    with pytest.raises(RuntimeError, match="stopped"):
        binding.update(owner, x, y)
    assert budget.replay_examples_completed == 2


@pytest.mark.parametrize(
    "change",
    [
        "plan",
        "budget",
        "bank",
        "input",
        "duplicate",
        "capacity",
        "reserved-id",
        "unobserved",
        "spent",
    ],
)
def test_should_refuse_unconsented_or_changed_replay_before_native_work(change):
    rplan, bank, budget = replay()
    readouts = bank
    if change == "plan":
        rplan = replace(rplan, every_updates=2)
    elif change == "budget":
        budget = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=5), lambda: 0.0)
    elif change == "bank":
        bank = (episode("probe_test"),)
    elif change == "input":
        bank = []
    elif change == "duplicate":
        bank = bank + bank
    elif change == "capacity":
        rplan = replace(rplan, max_retained_rows=1)
    elif change == "reserved-id":
        bank = (episode("train", name="literal-probe_test"),)
        readouts = bank
    elif change == "unobserved":
        readouts = ()
    elif change == "spent":
        budget.record_replay(1)
    with pytest.raises(ValueError):
        _ProbeRehearsalBinding(episode(), episode("probe_test"), rplan, bank, budget, readouts)


@pytest.mark.parametrize("change", ["inputs", "budget-owner", "budget-limits", "charge", "shape"])
def test_should_refuse_replaced_prior_authority_or_reserved_arrays_without_more_work(change):
    rplan, bank, budget = replay()
    binding = _ProbeRehearsalBinding(episode(), episode("probe_test"), rplan, bank, budget, bank)
    x = np.asarray(binding.features)
    y = np.asarray(binding.targets)
    if change == "inputs":
        binding.targets = tuple(reversed(binding.targets))
    elif change == "budget-owner":
        binding._budget.session = replay()[2]
    elif change == "budget-limits":
        budget.budget = ToyExecutionBudget(max_replay_examples=4)
    elif change == "charge":
        budget.replay_examples_completed = 1
    elif change == "shape":
        x = x[:1]
    with pytest.raises((ValueError, RuntimeError)):
        binding.update(source(), x, y)
    assert binding.attempts == 0


@pytest.mark.parametrize("mode", ["sgd", "nearest", "missing-plan", "missing-budget", "empty-bank"])
def test_should_keep_new_route_explicit_and_refuse_unsupported_modes(mode):
    rplan, bank, budget = replay()
    owner = source()
    p = plan()
    if mode == "sgd":
        p = plan(optimizer="sgd")
    elif mode == "nearest":
        owner = object()
    elif mode == "missing-plan":
        rplan = None
    elif mode == "missing-budget":
        budget = None
    elif mode == "empty-bank":
        bank = ()
    with pytest.raises(ValueError):
        probe.run_continuous_probe(
            owner,
            p,
            training_readouts=(episode("train"),),
            replay_plan=rplan,
            replay_episodes=bank,
            replay_budget=budget,
        )


def test_should_reject_changed_paid_return_before_more_replay_and_restore_capture(monkeypatch):
    original = probe._RecordedProbeBP.train_batch

    def damaged(port, features, targets):
        diagnostic = original(port, features, targets)
        port.reports[-1]["training_rows"] = 999
        return diagnostic

    monkeypatch.setattr(probe._RecordedProbeBP, "train_batch", damaged)
    owner = source()
    before = owner.snapshot_state()
    result, budget = run(owner, updates=2)
    assert result["complete"] is False
    assert budget.replay_examples_completed == 2
    assert len(result["native_updates"]) == 1
    assert result["native_updates"][0]["training_rows"] == 4
    assert probe.same_continuous_state(owner.snapshot_state(), before)


def test_should_reject_equal_replacement_of_budget_after_first_paid_fit():
    rplan, bank, budget = replay()
    changed = False

    def boundary():
        nonlocal changed
        if budget.replay_examples_completed and not changed:
            budget.budget = ToyExecutionBudget(max_replay_examples=4)
            changed = True

    result = probe.run_continuous_probe(
        source(),
        plan(2),
        training_readouts=bank,
        check_boundary=boundary,
        replay_plan=rplan,
        replay_episodes=bank,
        replay_budget=budget,
    )
    assert result["complete"] is False
    assert changed and budget.replay_examples_completed == 2
    assert result["rehearsal"]["native_attempts"] == 1
    assert len(result["native_updates"]) == 1


def test_should_report_uncertain_native_work_after_post_kernel_failure(monkeypatch):
    original = bp.NumpyContinuousBCLearner._apply_full_gauss_newton_batch

    def lost_return(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected return lost after kernel")

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "_apply_full_gauss_newton_batch", lost_return)
    result, budget = run(source(), updates=2)
    assert result["complete"] is False
    assert "return lost" in result["error"]
    assert budget.replay_examples_completed == 2
    assert result["rehearsal"]["native_attempts"] == 1
    assert result["rehearsal"]["native_returns"] == 0
    assert result["rehearsal"]["unreturned_native_work_unknown"] is True
    assert result["native_updates"] == []


@pytest.mark.parametrize("change", ["budget", "rows", "bank"])
def test_should_capture_successful_native_return_before_refusing_post_native_mutation(
    monkeypatch, change
):
    rplan, bank, budget = replay()
    bindings = []
    init = _ProbeRehearsalBinding.__init__

    def bound(binding, *args, **kwargs):
        init(binding, *args, **kwargs)
        bindings.append(binding)

    monkeypatch.setattr(_ProbeRehearsalBinding, "__init__", bound)
    original = bp.NumpyContinuousBCLearner._apply_full_gauss_newton_batch

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "budget":
            budget.budget = ToyExecutionBudget(max_replay_examples=4)
        elif change == "rows":
            bindings[0].training_rows = 999
        else:
            bindings[0].bank = ()
        return result

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "_apply_full_gauss_newton_batch", changed)
    result = probe.run_continuous_probe(
        source(),
        plan(2),
        training_readouts=bank,
        replay_plan=rplan,
        replay_episodes=bank,
        replay_budget=budget,
    )
    assert result["complete"] is False
    assert budget.replay_examples_completed == result["rehearsal"]["charged_exposures"] == 2
    assert result["rehearsal"]["native_returns"] == 1
    assert result["rehearsal"]["bank_episode_ids"] == ["literal-train"]
    assert result["rehearsal"]["joint_training_rows"] == 4
    assert len(result["native_updates"]) == 1
    assert result["native_updates"][0]["training_rows"] == 4
    assert result["native_updates"][0]["work"]["published_updates"] == 1


def test_should_enforce_original_replay_wall_clock_after_capturing_native_result(monkeypatch):
    rplan, bank, _ = replay()
    clock = [0.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=4, max_wall_seconds=0.5), lambda: clock[0]
    )
    original = bp.NumpyContinuousBCLearner._apply_full_gauss_newton_batch

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 1.0
        return result

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "_apply_full_gauss_newton_batch", delayed)
    result = probe.run_continuous_probe(
        source(),
        plan(2),
        training_readouts=bank,
        replay_plan=rplan,
        replay_episodes=bank,
        replay_budget=budget,
    )
    assert result["complete"] is False
    assert "max_wall_seconds" in result["error"]
    assert budget.replay_examples_completed == 2
    assert len(result["native_updates"]) == 1
