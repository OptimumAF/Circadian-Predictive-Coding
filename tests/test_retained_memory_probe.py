"""NEW tiny retained-source private probes; no recordings, SDK or model RNG."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.adapters import continuous_probe as probe
from src.adapters import continuous_nearest_probe as nearest
from src.adapters import numpy_retained_nearest_bc as retained
from src.adapters.numpy_nearest_continuous_bc import (
    NumpyNearestContinuousBCLearner,
    FixedInputNearestContinuousBCPolicy,
)
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.replay_retention import ReplayRetentionPolicy
from test_retained_nearest_memory import _process_peak_bytes

SPEC = ContinuousActionSpec(2, ("unit",), (-1.0,), (1.0,))
FIFO = ReplayRetentionPolicy("recent_fifo")
WORK: dict[str, int] = {}
PRIOR: dict[str, int] = {}
CAPS = dict(
    initializers=160,
    updates=512,
    incoming_rows=2048,
    forks=1024,
    predictions=1024,
    query_rows=4096,
    candidates=65536,
    coordinates=786432,
    snapshots=32768,
    validators=131072,
    copy_bytes=512 * 1024**2,
    hash_bytes=1024**3,
    diagnostic_rows=4096,
)


def charge(key, amount=1):
    WORK[key] = WORK.get(key, 0) + amount
    assert WORK[key] + PRIOR.get(key, 0) <= CAPS[key], key


@pytest.fixture(scope="module", autouse=True)
def finite_work():
    PRIOR.update(json.loads(os.environ.get("RETAINED_PROBE_PRIOR_WORK", "{}")))

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
        charge("diagnostic_rows", len(args[0]))
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
        patch.setattr(nearest, "retained_snapshot", validate)
        patch.setattr(retained, "_ids", hashed)
        patch.setattr(probe.diagnostics, "continuous_bc_residuals", residuals)
        patch.setattr(np.random, "default_rng", forbidden)
        yield
    peak = _process_peak_bytes()
    target = os.environ.get("RETAINED_PROBE_RECEIPT")
    if target:
        with Path(target).open("x", encoding="utf8") as handle:
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


def forbidden(*args, **kwargs):
    pytest.fail("unexpected model RNG/forbidden work")


def episode(name, x, y, role, context=None):
    return ContinuousEpisode(
        name,
        context or name,
        role,
        SPEC,
        (ContinuousTransition(0, (float(x), 0.0), (float(y),), 0.0, (float(x), 0.0), False, True),),
    )


def source(rule=FIFO):
    owner = retained.NumpyRetainedNearestBCLearner(
        FixedInputNearestContinuousBCPolicy(SPEC, max_prototypes=2), rule
    )
    owner.train_batch(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), np.array([[-1.0], [-0.5], [0.5]])
    )
    return owner


def plan(**kwargs):
    return probe.ContinuousProbePlan(
        episode("new-train", 3, 0.75, "probe_train", "fresh"),
        episode("new-test", 3, 0.75, "probe_test", "fresh"),
        **kwargs,
    )


def readouts():
    return tuple(
        episode("old" + str(i), 1 if i % 2 == 0 else 2, -0.5 if i % 2 == 0 else 0.5, "train")
        for i in range(4)
    )


def test_new_task_learns_and_old_loss_is_reported_without_mutating_source():
    owner = source()
    pin = retained.pin_retained_native(owner)
    phases = []

    def observe(child, phase, guard):
        guard()
        assert child is not owner
        phases.append(phase)
        guard()

    report = probe.run_retained_nearest_probe(
        owner, plan(), training_readouts=readouts(), endpoint_observer=observe
    )
    assert report["complete"], report.get("error")
    retained.require_retained_native(owner, pin)
    assert phases == ["before_probe", "after_probe"]
    assert report["progress"] == dict(updates_completed=1, replay_examples_completed=0, next_tick=2)
    assert report["events"][0]["deferred_exact"] and report["events"][0]["success_validated"]
    assert report["metrics"]["fresh_training_gain"] > 0
    assert (
        report["checkpoints"][1]["scores"]["fresh_training"]["residual"]["overall"][
            "normalized_half_mse"
        ]
        == 0
    )
    assert report["metrics"]["old_retention_loss"]["old_0"] > 0
    assert len(report["metrics"]["old_retention_loss"]) == 4
    assert report["retention"]["source"]["training_examples"] == 3
    assert report["retention"]["candidate_after_return"]["training_examples"] == 4
    assert report["retention"]["candidate_after_return"]["retained_rows"] == 2
    assert report["work"]["nearest_retrieval"]["distance_candidates_completed"] == 24
    assert report["work"]["score_rows_returned_by_role"] == dict(
        probe_test=2, train=8, probe_train=2
    )


def test_reservoir_runs_with_declared_seed_and_bounded_costs():
    report = probe.run_retained_nearest_probe(
        source(ReplayRetentionPolicy("seeded_reservoir", 17)), plan()
    )
    assert report["complete"], report.get("error")
    assert report["retention"]["candidate_after_return"]["retained_rows"] == 2
    assert report["work"]["nearest_retrieval"]["distance_candidates_completed"] == 8


def test_changed_shared_helpers_preserve_old_exact_append_only_route():
    owner = NumpyNearestContinuousBCLearner(FixedInputNearestContinuousBCPolicy(SPEC))
    owner.train_batch(np.array([[1.0, 0.0], [2.0, 0.0]]), np.array([[-0.5], [0.5]]))
    original = owner.snapshot_state()
    report = probe.run_continuous_probe(owner, plan())
    assert report["complete"], report.get("error")
    assert nearest.same_state(owner.snapshot_state(), original)
    assert report["source_state"]["rows"] == 2
    assert report["work"]["candidate_rows_known_after_return"] == 3
    assert "retention" not in report


@pytest.mark.parametrize(
    "options",
    [
        dict(updates=2),
        dict(diagnostics=True),
        dict(zero_velocity_control=True),
        dict(optimizer="full-gauss-newton"),
    ],
)
def test_explicit_route_refuses_neural_or_repeated_update_options(options):
    with pytest.raises(ValueError):
        probe.run_retained_nearest_probe(object(), plan(**options))


@pytest.mark.parametrize("role", ["train", "eval", "final", "probe_test"])
def test_plan_refuses_training_role_conversion(role):
    with pytest.raises(ValueError):
        probe.ContinuousProbePlan(
            episode("train", 3, 0.75, role, "fresh"),
            episode("test", 3, 0.75, "probe_test", "fresh"),
        )


@pytest.mark.parametrize("kind", ["old_route", "subclass", "too_many", "wrong_readout_role"])
def test_new_route_does_not_widen_existing_or_readout_admission(kind):
    owner = source()
    if kind == "old_route":
        with pytest.raises(ValueError, match="explicit"):
            probe.run_continuous_probe(owner, plan())
    elif kind == "subclass":

        class Other(retained.NumpyRetainedNearestBCLearner):
            pass

        owner.__class__ = Other
        with pytest.raises(ValueError, match="exact"):
            probe.run_retained_nearest_probe(owner, plan())
    else:
        rows = (
            readouts() + (episode("fifth", 1, -0.5, "train"),)
            if kind == "too_many"
            else (episode("reserved", 1, -0.5, "probe_test"),)
        )
        with pytest.raises(ValueError):
            probe.run_retained_nearest_probe(owner, plan(), training_readouts=rows)


@pytest.mark.parametrize(
    "fault", ["source_array", "source_equal_rule", "readout_equal_state", "metadata_ids"]
)
def test_endpoint_refuses_changed_native_state_and_equal_replacements(fault):
    owner = source()

    def observe(child, phase, guard):
        if fault == "source_array":
            owner._state.prototype_actions[0, 0] += 0.1
        elif fault == "source_equal_rule":
            owner._retention = ReplayRetentionPolicy("recent_fifo")
        elif fault == "readout_equal_state":
            child._state = child.snapshot_state()
        else:
            object.__setattr__(
                child._state, "prototype_ids", tuple(reversed(child._state.prototype_ids))
            )

    report = probe.run_retained_nearest_probe(owner, plan(), endpoint_observer=observe)
    assert not report["complete"]
    assert report["work"]["training_attempts"] == 0


def test_lost_return_remains_spent_and_does_not_infer_retained_bank(monkeypatch):
    owner = source()
    before = owner.snapshot_state()
    append = retained.NumpyRetainedNearestBCLearner.train_batch

    def lost(child, features, targets):
        append(child, features, targets)
        raise RuntimeError("literal lost append return")

    monkeypatch.setattr(retained.NumpyRetainedNearestBCLearner, "train_batch", lost)
    report = probe.run_retained_nearest_probe(owner, plan())
    assert not report["complete"] and report["work"]["unreturned_append_possible"]
    assert report["work"]["append_attempts_permanently_spent"] == 1
    assert report["work"]["append_slots_remaining"] == 0
    assert retained.same_retained_state(owner.snapshot_state(), before)
    # No successful child return was observed; source is not the resulting bank.
    assert report["retention"]["candidate_after_return"] is None


def test_wrong_retention_update_is_refused_even_with_valid_native_shape(monkeypatch):
    owner = source()
    append = retained.NumpyRetainedNearestBCLearner.train_batch

    def wrong(child, features, targets):
        return append(child, features, -targets)

    monkeypatch.setattr(retained.NumpyRetainedNearestBCLearner, "train_batch", wrong)
    report = probe.run_retained_nearest_probe(owner, plan())
    assert not report["complete"] and "declared content selection" in report["error"]
    assert len(report["native_updates"]) == report["work"]["training_attempts"] == 1


def test_append_cannot_replace_original_retention_with_value_equal_rule(monkeypatch):
    owner = source()
    append = retained.NumpyRetainedNearestBCLearner.train_batch

    def replaced(child, features, targets):
        result = append(child, features, targets)
        child._retention = ReplayRetentionPolicy("recent_fifo")
        return result

    monkeypatch.setattr(retained.NumpyRetainedNearestBCLearner, "train_batch", replaced)
    report = probe.run_retained_nearest_probe(owner, plan())
    assert not report["complete"] and "runtime ownership" in report.get("error", "")
    assert len(report["native_updates"]) == report["work"]["append_attempts_permanently_spent"] == 1


@pytest.mark.parametrize("fault", ["score", "work", "native_return"])
def test_changed_paid_evidence_is_refused_and_prior_return_preserved(monkeypatch, fault):
    owner = source()
    observe = probe._observe_probe_endpoint

    def observed(callback, readout, phase, report, **kwargs):
        def change(child, stage, guard):
            if stage == "after_probe":
                if fault == "score":
                    report["checkpoints"][0]["scores"].clear()
                elif fault == "work":
                    report["work"]["training_attempts"] = 0
                else:
                    report["native_updates"][0]["training_rows"] = 9
            return callback(child, stage, guard)

        return observe(change, readout, phase, report, **kwargs)

    monkeypatch.setattr(probe, "_observe_probe_endpoint", observed)
    report = probe.run_retained_nearest_probe(owner, plan(), endpoint_observer=lambda *args: None)
    assert not report["complete"]
    assert report["checkpoints"][0]["scores"]
    assert report["work"]["training_attempts"] == 1
    assert report["native_updates"][0]["training_rows"] == 1
