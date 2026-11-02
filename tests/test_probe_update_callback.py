"""Scripted existing-runner routing/accounting; no numerical learner work."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.probe_update import ProbeUpdateFailed, ProbeUpdateReceipt
from test_probe_training_readouts import cohorts
from test_probe_return_phase import return_templates


class Owner:
    def __init__(self, control=None):
        self.control = {} if control is None else control
        self.state = np.array([99.0])

    def fork(self):
        child = Owner(self.control)
        self.control["child"] = child
        return child

    def restore_state(self, state):
        self.state = state.copy()

    def snapshot_state(self):
        return self.state.copy()

    def train_batch(self, *args):
        pytest.fail("auxiliary path called native SGD")

    def predict(self, features):
        return np.full((len(features), 3), 1 / 3)


def session():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=16), lambda: 0.0)


def run(fault=None, *, returning=True):
    source = Owner()
    training, evaluation, anchors = cohorts()
    returns = return_templates(anchors)
    original = deepcopy((training, evaluation, anchors, returns))
    fresh, old = session(), session()
    calls = []
    exposed: dict[str, Any] = {}

    def callback(owner, current, retained, *, phase, check_boundary):
        assert owner is source.control["child"] and owner is not source
        assert current.role == ("probe_train" if phase == "fresh" else "train")
        assert all(b.role == "train" for b in retained)
        assert len(retained) == 2
        calls.append((phase, current.role, tuple(b.role for b in retained)))
        if fault == "inputs":
            current.targets[0, 0] = 2.0
        elif fault == "replacement":
            object.__setattr__(retained[0], "features", retained[0].features.copy())
        elif fault == "role":
            object.__setattr__(current, "role", "train")
        elif fault == "counter":
            fresh.record_update()
        elif fault == "clock":
            fresh.clock = lambda: 0.0
        elif fault == "source":
            source.state += 1
        elif fault == "paid" and phase == "return":
            exposed["report"]["auxiliary_updates"]["events"][0]["work"]["closure_attempts"] += 1
        if fault == "swallow":
            target = current.targets[0, 0]
            current.targets[0, 0] = 2.0
            with pytest.raises(ValueError):
                check_boundary()
            current.targets[0, 0] = target
        else:
            check_boundary()
        receipt = ProbeUpdateReceipt(
            "scripted_auxiliary", dict(native_sgd_updates=0, closure_attempts=1), False
        )
        if fault == "early" or fault == "return" and phase == "return":
            raise ProbeUpdateFailed("paid fit failed", receipt)
        if fault == "untyped":
            return dict(applied=True)
        owner.state += 1
        receipt = replace(receipt, applied=True, loss=0.1)
        if fault == "late":
            raise ProbeUpdateFailed("guard failed after application", receipt)
        return receipt

    def boundary():
        if source.state[0] != 99:
            raise ValueError("source changed")

    kwargs = dict(return_training=returns, return_budget=old) if returning else {}
    from src.app import fresh_task_probe as runner

    original_return = runner._run_return_phase

    def returning_phase(*args, **kw):
        exposed["report"] = args[4]
        return original_return(*args, **kw)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "_run_return_phase", returning_phase)
        report = run_fresh_task_probe(
            cast(Any, source),
            np.array([0.0]),
            training,
            evaluation,
            budget=fresh,
            checkpoint_updates=(0, 4, 8, 16),
            timer=lambda: 1.0,
            training_readouts=anchors,
            update_callback=callback,
            check_boundary=boundary,
            **kwargs,
        )
    from src.app.coupled_attribution import _same_value

    assert _same_value((training, evaluation, anchors, returns), original)
    return report, source, fresh, old, calls


def test_existing_fresh_and_return_loop_applies_only_private_callbacks():
    report, source, fresh, old, calls = run()
    assert report["status"] == report["return_phase"]["status"] == "complete"
    assert fresh.updates_completed == old.updates_completed == 16
    assert source.state[0] == 99 and source.control["child"].state[0] == 32
    assert [c[0] for c in calls] == ["fresh"] * 16 + ["return"] * 16
    for row in (report, report["return_phase"]):
        assert row["diagnostics"] == []
        assert row["auxiliary_updates"]["native_sgd_updates"] == 0
        assert len(row["auxiliary_updates"]["events"]) == 16
        assert row["completed_training_examples"] == 144
        assert [p["updates"] for p in row["checkpoints"]] == [0, 4, 8, 16]
    assert report["auxiliary_updates"]["retained_data_array_bytes"] > 0


@pytest.mark.parametrize(
    "fault",
    ["early", "late", "inputs", "replacement", "role", "counter", "clock", "source", "untyped"],
)
def test_failed_callback_preserves_paid_work_and_refuses_more_updates(fault):
    report, source, fresh, old, calls = run(fault)
    assert report["status"] == "failed" and len(calls) == 1
    event = report["auxiliary_updates"]["events"][0]
    assert event["status"] == "failed"
    assert report["completed_updates"] == int(fault == "late")
    assert fresh.updates_completed == int(fault in ("late", "counter"))
    assert old.updates_completed == 0
    assert report["return_phase"]["status"] == "not_run"
    if fault in ("early", "late"):
        assert event["work_known"] and event["work"]["closure_attempts"] == 1
        assert event["applied"] == (fault == "late")
    else:
        assert not event["work_known"]


@pytest.mark.parametrize("fault", ["return", "paid"])
def test_return_refuses_failure_or_changed_preceding_paid_evidence(fault):
    report, source, fresh, old, calls = run(fault)
    assert fresh.updates_completed == 16 and old.updates_completed == 0
    assert len(calls) == 17
    assert report["return_phase"]["status"] == "failed"
    assert len(report["checkpoints"]) == 4
    assert report["auxiliary_updates"]["events"][0]["work"]["closure_attempts"] == 1


@pytest.mark.parametrize(
    "change",
    [
        dict(update_callback=1),
        dict(rehearsal_factory=lambda: None),
        dict(extra_f_exposure=True),
        dict(short_affine_return=True),
        dict(training_readouts=()),
    ],
)
def test_bad_callback_modes_refused_before_any_owner_or_clock(change):
    training, evaluation, anchors = cohorts()
    kwargs = dict(update_callback=lambda: None, training_readouts=anchors)
    kwargs.update(change)
    with pytest.raises(ValueError, match="update_callback"):
        run_fresh_task_probe(
            cast(Any, None),
            None,
            training,
            evaluation,
            budget=session(),
            checkpoint_updates=(0, 4, 8, 16),
            **kwargs,
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(applied=1),
        dict(loss=float("nan")),
        dict(work={"native_sgd_updates": 1}),
        dict(work={"native_sgd_updates": False}),
        dict(work={"native_sgd_updates": 0, "backwards": -1}),
    ],
)
def test_receipt_refuses_fabricated_or_non_scalar_work(change):
    kwargs: dict[str, Any] = dict(definition="test", applied=True, work={"native_sgd_updates": 0})
    kwargs.update(change)
    with pytest.raises(ValueError):
        ProbeUpdateReceipt(**kwargs)


def test_caught_boundary_refusal_cannot_be_repaired_into_acceptance():
    report, source, fresh, old, calls = run("swallow")
    assert report["status"] == "failed" and len(calls) == 1
    assert fresh.updates_completed == report["completed_updates"] == 1
    assert old.updates_completed == 0
    assert len(report["checkpoints"]) == 1
    assert source.control["child"].state[0] == 1
    assert report["auxiliary_updates"]["events"][0]["applied"]
    assert report["auxiliary_updates"]["events"][0]["status"] == "failed"
    assert report["return_phase"]["status"] == "not_run"
