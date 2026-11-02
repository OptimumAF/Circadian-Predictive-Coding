"""Bounded #412 literal disposable update/runtime checks; no datasets or RNG."""

from collections import defaultdict
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from src.adapters import continuous_probe as probe
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_probe_fit import _FullProbeBinding
from src.adapters.numpy_continuous_pc import NumpyContinuousPCLearner
from src.adapters.numpy_nearest_continuous_bc import NumpyNearestContinuousBCLearner
from src.core import continuous_bc as bc
from src.core import continuous_full_gauss_newton as gn
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.learner_ports import TrainingUpdateFailed

WORK: defaultdict[str, int] = defaultdict(int)
CAPS = dict(
    sources=48,
    updates=24,
    update_rows=72,
    solves=24,
    gradients=160,
    gradient_rows=480,
    forwards=512,
    forward_rows=1536,
    snapshots=65536,
    snapshot_bytes=256 * 1024**2,
    sgd=4,
    sgd_rows=12,
    forbidden=0,
)


def charge(key, amount=1):
    WORK[key] += amount
    assert WORK[key] <= CAPS[key], "permanent412fixturecap:" + key


def forbidden(*args, **options):
    charge("forbidden")


@pytest.fixture(autouse=True)
def meter(monkeypatch):
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    for module, name, key, row_key in (
        (bc, "_forward", "forwards", "forward_rows"),
        (bc, "continuous_bc_loss_and_gradients", "gradients", "gradient_rows"),
        (bp.NumpyContinuousBCLearner, "_apply_full_gauss_newton_batch", "updates", "update_rows"),
        (bp.NumpyContinuousBCLearner, "train_batch", "sgd", "sgd_rows"),
    ):
        original = getattr(module, name)

        def counted(*args, _original=original, _key=key, _rows=row_key, **options):
            charge(_key)
            charge(_rows, len(args[1] if _key in ("updates", "sgd") else args[0]))
            return _original(*args, **options)

        monkeypatch.setattr(module, name, counted)
    solve, snapshot, fork = np.linalg.lstsq, bp._snapshot, bp.NumpyContinuousBCLearner.fork

    def forked(owner):
        charge("sources")
        return fork(owner)

    def solved(matrix, *args, **options):
        charge("solves")
        assert max(matrix.shape) <= 64
        return solve(matrix, *args, **options)

    def copied(*args, **options):
        charge("snapshots")
        value = snapshot(*args, **options)
        charge("snapshot_bytes", sum(x.nbytes for x in value.parameters + value.velocity))
        return value

    monkeypatch.setattr(np.linalg, "lstsq", solved)
    monkeypatch.setattr(bp, "_snapshot", copied)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forked)


def source():
    charge("sources")
    owner = object.__new__(bp.NumpyContinuousBCLearner)
    spec = ContinuousActionSpec(2, ("x", "y", "z", "normalized_gripper"), (-1.0,) * 4, (1.0,) * 4)
    policy = bp.ContinuousBCPolicy(spec, hidden_dim=2, seed=17, binary_gripper_axis=3)
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
        6,
    )
    return owner


def episode(role="probe_train"):
    spec = ContinuousActionSpec(2, ("x", "y", "z", "normalized_gripper"), (-1.0,) * 4, (1.0,) * 4)
    xs = ((0.2, -0.3), (0.6, 0.1), (-0.4, 0.5))
    ys = ((0.3, -0.2, 0.1, 1.0), (-0.3, 0.2, 0.1, -1.0), (0.0, 0.1, -0.2, 1.0))
    if role == "probe_test":
        ys = ((0.2, -0.1, 0.2, -1.0), (-0.2, 0.1, 0.2, 1.0), (0.1, 0.2, -0.1, -1.0))
    return ContinuousEpisode(
        "literal-" + role,
        "old" if role == "train" else "fresh",
        role,
        spec,
        tuple(
            ContinuousTransition(i, x, y, 0.0, xs[(i + 1) % 3], i == 2, False)
            for i, (x, y) in enumerate(zip(xs, ys, strict=True))
        ),
    )


def plan(**options):
    return probe.ContinuousProbePlan(episode(), episode("probe_test"), **options)


def arrays():
    binding = _FullProbeBinding.from_training(episode())
    return np.asarray(binding.features), np.asarray(binding.targets)


def assert_original(owner, before):
    assert probe.same_continuous_state(owner.snapshot_state(), before)


def test_private_full_fit_runs_delayed_runtime_without_changing_source_or_actor():
    owner = source()
    before = owner.snapshot_state()
    endpoints = []

    def observe(readout, phase, guard):
        guard()
        assert readout is not owner
        state = readout.snapshot_state()
        endpoints.append((phase, state.completed_updates, state.training_examples))
        assert_original(owner, before)
        if phase == "after_probe":
            assert all(not np.any(x) for x in state.velocity)

    report = probe.run_continuous_probe(
        owner,
        plan(updates=2, optimizer="full-gauss-newton"),
        training_readouts=(episode("train"),),
        endpoint_observer=observe,
    )
    assert report["complete"], report.get("error")
    assert endpoints == [("before_probe", 2, 6), ("after_probe", 4, 12)]
    assert report["progress"]["updates_completed"] == 2
    assert report["progress"]["replay_examples_completed"] == 0
    for row in report["native_updates"]:
        assert row["work"]["published_updates"] == 1
        assert row["work"]["solve_returns"] == 1
        assert row["loss_after"] < row["diagnostic"]["value"]
        assert row["momentum_reset"] is True
    work = report["work"]
    assert work["native_update_counts"]["gradient_returns"] == 4
    assert work["native_update_counts"]["solver_work_known"] is False
    assert work["native_update_counts"]["unknown_partial_work"] is False
    assert work["full_network_momentum_resets"] == 2
    assert work["derived_bp_training_dense_multiply_accumulates"] == 0
    assert len(report["applied_updates"]) == 2
    assert all(e["deferred_exact"] and e["success_validated"] for e in report["events"])
    assert not any(isinstance(v, bp.NumpyContinuousBCLearner) for v in report.values())
    assert_original(owner, before)


def test_default_sgd_probe_retains_original_schema_and_source():
    owner = source()
    before = owner.snapshot_state()
    start = WORK["updates"]
    report = probe.run_continuous_probe(owner, plan())
    assert report["complete"], report.get("error")
    assert WORK["updates"] == start
    assert "optimizer" not in report["plan"]
    assert "bp_sgd_cost_applicable" not in report["work"]
    assert report["native_updates"][0]["work"] is None
    assert_original(owner, before)


def test_public_train_entry_uses_same_owned_commit_and_stays_train_only():
    owner = source()
    x, y = arrays()
    result = owner.train_full_gauss_newton_batch(x, y)
    assert result.work["published_updates"] == 1
    assert result.loss_after < result.diagnostic.value
    for role in ("probe_train", "probe_test", "final"):
        with pytest.raises(ValueError):
            policy = replace(owner._policy, training_role=role)
            owner._policy = policy
            owner._state = replace(owner._state, policy=policy)
            owner.train_full_gauss_newton_batch(x, y)


@pytest.mark.parametrize("role", ["train", "probe_test", "final"])
def test_binding_refuses_other_data_roles_before_update(role):
    with pytest.raises(ValueError):
        _FullProbeBinding.from_training(episode(role))


@pytest.mark.parametrize("bad", [None, True, "other"])
def test_plan_rejects_unknown_optimizer_before_work(bad):
    with pytest.raises(ValueError, match="optimizer"):
        plan(optimizer=bad)


@pytest.mark.parametrize("mode", ["diagnostics", "zero_velocity_control"])
def test_full_fit_refuses_sgd_specific_controls(mode):
    with pytest.raises(ValueError, match="excludes"):
        plan(optimizer="full-gauss-newton", **{mode: True})


@pytest.mark.parametrize("kind", [NumpyContinuousPCLearner, NumpyNearestContinuousBCLearner])
def test_non_bp_optimizer_refused_before_snapshot_or_initialization(kind):
    with pytest.raises(ValueError, match="exact BP"):
        probe.run_continuous_probe(object.__new__(kind), plan(optimizer="full-gauss-newton"))


def test_bound_training_arrays_refuse_evaluation_targets_and_changed_inputs():
    owner = source()
    binding = _FullProbeBinding.from_training(episode())
    x, y = arrays()
    _, evaluation_targets = probe._score_view(episode("probe_test"))
    for wrong_x, wrong_y in ((x + 0.01, y), (x, evaluation_targets), (x.tolist(), y)):
        with pytest.raises(ValueError, match="bound probe_train"):
            binding.update(owner, wrong_x, wrong_y)


def test_failed_solver_is_paid_and_private_port_cannot_retry(monkeypatch):
    owner = source()
    before = owner.snapshot_state()
    called = []

    def lost(*args, **options):
        charge("solves")
        called.append(1)
        raise RuntimeError("lost solve return")

    monkeypatch.setattr(np.linalg, "lstsq", lost)
    port = probe._RecordedProbeBP(owner, [], 1, _FullProbeBinding.from_training(episode()))
    x, y = arrays()
    with pytest.raises(TrainingUpdateFailed) as error:
        port.train_batch(x, y)
    work = error.value.work
    assert work["solve_attempts"] == 1 and work["solve_returns"] == 0
    assert work["unknown_partial_work"] is True and work["published_updates"] == 0
    with pytest.raises(RuntimeError, match="permanently failed"):
        port.train_batch(x, y)
    with pytest.raises(RuntimeError, match="permanently failed"):
        port.fork()
    assert len(called) == 1
    assert_original(owner, before)


def test_runtime_retains_failed_native_work_without_applying_a_receipt(monkeypatch):
    owner = source()
    before = owner.snapshot_state()

    def lost(*args, **options):
        charge("solves")
        raise RuntimeError("lost solve return")

    monkeypatch.setattr(np.linalg, "lstsq", lost)
    report = probe.run_continuous_probe(owner, plan(optimizer="full-gauss-newton"))
    assert not report["complete"]
    assert report["failed_native_work"]["solve_attempts"] == 1
    assert report["failed_native_work"]["published_updates"] == 0
    assert report["applied_updates"] == []
    assert report["progress"]["updates_completed"] == 0
    assert_original(owner, before)


def test_corrupted_private_owner_after_paid_proposal_is_refused(monkeypatch):
    owner = source()
    before = owner.snapshot_state()
    apply = bp.NumpyContinuousBCLearner._apply_full_gauss_newton_batch
    propose = gn.propose_full_gauss_newton

    def corrupted(private, *args, **options):
        def changed(*a, **kw):
            result = propose(*a, **kw)
            private._state.parameters[0][0, 0] += 0.01
            return result

        monkeypatch.setattr(gn, "propose_full_gauss_newton", changed)
        return apply(private, *args, **options)

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "_apply_full_gauss_newton_batch", corrupted)
    report = probe.run_continuous_probe(owner, plan(optimizer="full-gauss-newton"))
    assert not report["complete"]
    assert report["failed_native_work"]["accepted_proposals"] == 1
    assert report["failed_native_work"]["published_updates"] == 0
    assert report["applied_updates"] == []
    assert_original(owner, before)


def test_returned_paid_receipt_corruption_is_refused_and_work_restored(monkeypatch):
    owner = source()
    before = owner.snapshot_state()
    update = probe._RecordedProbeBP.train_batch

    def erased(port, *args, **options):
        result = update(port, *args, **options)
        port.reports[-1]["work"]["solve_returns"] = 0
        return result

    monkeypatch.setattr(probe._RecordedProbeBP, "train_batch", erased)
    report = probe.run_continuous_probe(owner, plan(optimizer="full-gauss-newton"))
    assert not report["complete"]
    assert report["native_updates"][0]["work"]["solve_returns"] == 1
    assert len(report["applied_updates"]) == 1
    assert_original(owner, before)


@pytest.mark.parametrize("change", ["clear", "append"])
def test_all_paid_updates_survive_history_erasure_or_injection(monkeypatch, change):
    owner = source()
    before = owner.snapshot_state()
    update = probe._RecordedProbeBP.train_batch

    def changed(port, *args, **options):
        result = update(port, *args, **options)
        if len(port.reports) == 2:
            if change == "clear":
                port.reports.clear()
            else:
                port.reports.append(dict(port.reports[-1]))
        return result

    monkeypatch.setattr(probe._RecordedProbeBP, "train_batch", changed)
    report = probe.run_continuous_probe(owner, plan(updates=2, optimizer="full-gauss-newton"))
    assert not report["complete"]
    assert len(report["native_updates"]) == 2
    assert report["work"]["native_update_counts"]["solve_returns"] == 2
    assert len(report["applied_updates"]) == 2
    assert_original(owner, before)


def test_replaced_runtime_candidate_refuses_and_captures_original_paid_work(monkeypatch):
    owner = source()
    before = owner.snapshot_state()
    snapshot = probe.ActorShadowRuntime.candidate_snapshot

    def replaced(runtime):
        result = snapshot(runtime)
        if result.wake_updates == 1:
            runtime._candidate = object()
        return result

    monkeypatch.setattr(probe.ActorShadowRuntime, "candidate_snapshot", replaced)
    report = probe.run_continuous_probe(owner, plan(optimizer="full-gauss-newton"))
    assert not report["complete"]
    assert len(report["native_updates"]) == 1
    assert report["work"]["native_update_counts"]["solve_returns"] == 1
    assert len(report["applied_updates"]) == 1
    assert_original(owner, before)
