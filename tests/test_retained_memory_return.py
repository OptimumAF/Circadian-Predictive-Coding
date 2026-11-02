"""NEW literal old-TRAIN return boundaries; no data/SDK/neural work/model RNG."""

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.adapters import continuous_probe as probe
from src.adapters import continuous_nearest_probe as near
from src.adapters import continuous_nearest_return as returning
from src.adapters import numpy_retained_nearest_bc as retained
from src.adapters.numpy_nearest_continuous_bc import NumpyNearestContinuousBCLearner
from src.app.toy_execution_budget import ToyExecutionBudget
from test_retained_memory_probe import source, plan, readouts
from test_retained_nearest_memory import _process_peak_bytes

WORK: dict[str, int] = {}
PRIOR: dict[str, int] = {}
CAPS = dict(
    initializers=160,
    updates=1024,
    incoming_rows=4096,
    forks=2048,
    predictions=2048,
    query_rows=4096,
    candidates=131072,
    coordinates=1572864,
    snapshots=65536,
    validators=262144,
    copy_bytes=1024**3,
    hash_bytes=2 * 1024**3,
    residual_rows=4096,
    endpoints=256,
)


def charge(key, n=1):
    WORK[key] = WORK.get(key, 0) + n
    assert WORK[key] + PRIOR.get(key, 0) <= CAPS[key], key


@pytest.fixture(scope="module", autouse=True)
def finite_work():
    PRIOR.update(json.loads(os.environ.get("RETAINED_RETURN_PRIOR_WORK", "{}")))

    def wrap(method, kind):
        def call(owner, *args, **kwargs):
            charge(kind)
            if kind == "updates":
                charge("incoming_rows", len(args[0]))
            if kind == "predictions":
                rows = len(args[0])
                candidates = rows * len(owner._state.prototype_features)
                charge("query_rows", rows)
                charge("candidates", candidates)
                charge("coordinates", candidates * owner._policy.spec.observation_dim)
            return method(owner, *args, **kwargs)

        return call

    validator = retained.retained_snapshot

    def validate(state, policy, rule):
        charge("validators")
        charge("copy_bytes", state.prototype_features.nbytes + state.prototype_actions.nbytes)
        return validator(state, policy, rule)

    ids = retained._ids

    def hashed(features, targets):
        charge("hash_bytes", features.nbytes + targets.nbytes + 24 * len(features))
        return ids(features, targets)

    residual = probe.diagnostics.continuous_bc_residuals

    def residuals(*args, **kwargs):
        charge("residual_rows", len(args[0]))
        return residual(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        for cls in (NumpyNearestContinuousBCLearner, retained.NumpyRetainedNearestBCLearner):
            for method, kind in (
                ("__init__", "initializers"),
                ("train_batch", "updates"),
                ("fork", "forks"),
                ("predict_with_work", "predictions"),
                ("snapshot_state", "snapshots"),
            ):
                if method in cls.__dict__:
                    patch.setattr(cls, method, wrap(cls.__dict__[method], kind))
        patch.setattr(retained, "retained_snapshot", validate)
        patch.setattr(near, "retained_snapshot", validate)
        patch.setattr(retained, "_ids", hashed)
        patch.setattr(probe.diagnostics, "continuous_bc_residuals", residuals)
        patch.setattr(np.random, "default_rng", lambda *a, **k: pytest.fail("model RNG forbidden"))
        yield
    peak = _process_peak_bytes()
    target = os.environ.get("RETAINED_RETURN_RECEIPT")
    if target:
        with Path(target).open("x") as handle:
            json.dump(
                dict(
                    work=WORK,
                    aggregate={k: v + PRIOR.get(k, 0) for k, v in WORK.items()},
                    actual_pytest_peak_bytes=peak,
                ),
                handle,
                indent=2,
            )
    assert peak <= 512 * 1024**2


def score(report, point, name):
    return report["checkpoints"][point]["scores"][name]["residual"]["overall"][
        "normalized_half_mse"
    ]


def test_lost_old_context_can_return_without_losing_new_fit_or_changing_source():
    owner = source()
    pin = retained.pin_retained_native(owner)
    old = readouts()
    phases = []

    def observer(child, phase, guard):
        charge("endpoints")
        guard()
        assert child is not owner
        phases.append(phase)
        guard()

    result = probe.run_retained_nearest_probe(
        owner, plan(), training_readouts=old, return_training=old[0], endpoint_observer=observer
    )
    assert result["complete"], result
    assert phases == ["before_probe", "after_probe", "after_return"]
    original, returned = result["probe"], result["return_phase"]
    assert original["complete"] and returned["complete"]
    assert score(original, 1, "old_0") > 0 and score(returned, 1, "old_0") == 0
    assert score(original, 1, "fresh_training") == score(returned, 1, "fresh_training") == 0
    assert original["checkpoints"][1]["scores"] == returned["checkpoints"][0]["scores"]
    assert (
        original["progress"]
        == returned["progress"]
        == dict(updates_completed=1, replay_examples_completed=0, next_tick=2)
    )
    assert original["applied_updates"][0]["episode_id"] == "new-train"
    assert returned["applied_updates"][0]["episode_id"] == old[0].episode_id
    assert returned["plan"]["training_role"] == "train"
    assert returned["metrics"]["return_training_gain"] > 0
    assert returned["retention"]["candidate_after_return"]["training_examples"] == 5
    assert result["work"]["training_attempts"] == result["work"]["training_rows_attempted"] == 2
    retained.require_retained_native(owner, pin)


def test_none_return_preserves_default_single_stage_shape():
    result = probe.run_retained_nearest_probe(source(), plan(), return_training=None)
    assert result["complete"] and "return_phase" not in result
    assert result["progress"]["updates_completed"] == 1


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "eval", "final"])
def test_return_refuses_other_roles_before_resource_admission(monkeypatch, role):
    owner = source()
    old = readouts()
    monkeypatch.setattr(
        returning, "ToyBudgetSession", lambda *a, **k: pytest.fail("bad role admitted resources")
    )
    with pytest.raises(ValueError):
        probe.run_retained_nearest_probe(
            owner, plan(), training_readouts=old, return_training=replace(old[0], role=role)
        )


@pytest.mark.parametrize("fault", ["identity", "values"])
def test_return_must_equal_one_declared_old_episode(fault):
    old = readouts()
    episode = old[0]
    episode = (
        replace(episode, episode_id="invented-old")
        if fault == "identity"
        else replace(episode, transitions=(replace(episode.transitions[0], action=(0.9,)),))
    )
    with pytest.raises(ValueError, match="exactly one"):
        probe.run_retained_nearest_probe(
            source(), plan(), training_readouts=old, return_training=episode
        )


def test_lost_return_keeps_prior_success_and_unknown_spent_bank(monkeypatch):
    owner = source()
    old = readouts()
    before = owner.snapshot_state()
    append = retained.NumpyRetainedNearestBCLearner.train_batch
    calls = 0

    def lost(child, features, targets):
        nonlocal calls
        calls += 1
        value = append(child, features, targets)
        if calls == 2:
            raise RuntimeError("literal return ACK lost")
        return value

    monkeypatch.setattr(retained.NumpyRetainedNearestBCLearner, "train_batch", lost)
    result = probe.run_retained_nearest_probe(
        owner, plan(), training_readouts=old, return_training=old[0]
    )
    assert not result["complete"] and result["probe"]["complete"]
    returned = result["return_phase"]
    assert not returned["complete"] and returned["work"]["append_attempts_permanently_spent"] == 1
    assert returned["work"]["unreturned_append_possible"]
    assert returned["retention"]["candidate_after_return"] is None
    assert retained.same_retained_state(owner.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["owner", "limits", "clock"])
def test_prospectively_admitted_return_resources_cannot_change_before_use(monkeypatch, fault):
    original = returning.ToyBudgetSession
    captured = []

    def factory(*a, **k):
        value = original(*a, **k)
        captured.append(value)
        return value

    monkeypatch.setattr(returning, "ToyBudgetSession", factory)

    def observer(child, phase, guard):
        charge("endpoints")
        if phase == "before_probe":
            budget = captured[0]
            if fault == "owner":
                budget.budget = ToyExecutionBudget(max_training_updates=1)
            elif fault == "limits":
                object.__setattr__(budget.budget, "max_training_updates", 2)
            else:
                budget.started_at += 1.0

    old = readouts()
    result = probe.run_retained_nearest_probe(
        source(), plan(), training_readouts=old, return_training=old[0], endpoint_observer=observer
    )
    assert not result["complete"] and result["return_phase"] is None
    assert result["probe"]["work"]["training_attempts"] == 0


def test_prior_probe_evidence_is_sealed_through_return(monkeypatch):
    observe = probe._observe_probe_endpoint
    captured = {}

    def observed(callback, child, phase, report, **kwargs):
        if (
            phase == "after_probe"
            and report["definition"] == "continuous_disposable_retained_nearest_probe_v1"
        ):
            captured["prior"] = report
        return observe(callback, child, phase, report, **kwargs)

    monkeypatch.setattr(probe, "_observe_probe_endpoint", observed)

    def observer(child, phase, guard):
        charge("endpoints")
        if phase == "after_return":
            captured["prior"]["checkpoints"][0]["scores"].clear()

    old = readouts()
    result = probe.run_retained_nearest_probe(
        source(), plan(), training_readouts=old, return_training=old[0], endpoint_observer=observer
    )
    assert not result["complete"]
    assert result["probe"]["checkpoints"][0]["scores"]
    assert result["probe"]["native_updates"][0]["training_rows"] == 1


@pytest.mark.parametrize("stage", ["dispatch", "source_observation"])
def test_forward_clock_change_is_refused_before_return_authority_transfer(monkeypatch, stage):
    core = probe._run_single_probe
    captured = {}

    def run(owner, recipe, **options):
        if type(recipe) is probe._RetainedReturnPlan:
            captured["budget"] = options["return_scope"][0]
            if stage == "dispatch":
                captured["budget"].last_clock = captured["budget"].clock()
        return core(owner, recipe, **options)

    snapshot = retained.NumpyRetainedNearestBCLearner.snapshot_state

    def observed(owner):
        value = snapshot(owner)
        if stage == "source_observation" and "budget" in captured and not captured.get("done"):
            captured["done"] = True
            captured["budget"].last_clock = captured["budget"].clock()
        return value

    monkeypatch.setattr(probe, "_run_single_probe", run)
    monkeypatch.setattr(retained.NumpyRetainedNearestBCLearner, "snapshot_state", observed)
    old = readouts()
    result = probe.run_retained_nearest_probe(
        source(), plan(), training_readouts=old, return_training=old[0]
    )
    assert not result["complete"] and result["return_phase"] is None
    assert result["probe"]["work"]["training_attempts"] == 1
