"""Existing delayed-label sequence with an explicitly different linear objective."""

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_linear_catalog_learner as linear
from src.core.clm_head import FrozenEncoderIdentity
from src.core.clm_catalog import CLMCatalogTargets
from src.core.clm_linear_catalog import OBJECTIVE, linear_catalog_work
from src.core.learner_ports import TrainingUpdateFailed


def parts():
    identity = FrozenEncoderIdentity("literal", "fixed", "fixed", "last", "none", 2)
    values = torch.tensor([[(i + 1) / 13, (-1) ** i / 2] for i in range(12)])
    train, choices = study.build_catalog_training_batches(identity, values)
    bindings = tuple(
        (batch, study.policy_for_phase(phase)["allowed_ids"]) for phase, batch in choices.items()
    )
    return identity, values[-4:].clone(), train, choices, bindings


@pytest.fixture(scope="module", params=[False, True])
def complete(request):
    identity, actions, train, choices, _ = parts()
    report: dict[str, Any] = dict(
        control_probabilities={v: [[1 / 3] * 3 for _ in range(4)] for v in ("v1", "v2")}
    )
    rng = torch.get_rng_state().clone()
    study._run_head_stream(
        SimpleNamespace(identity=identity),
        train,
        choices,
        report,
        mode="none",
        diagnostics=request.param,
        before_step=lambda: None,
        save=lambda: None,
        catalog_training=True,
        train_linear_update=True,
        linear_actions=actions,
    )
    assert torch.equal(torch.get_rng_state(), rng)
    return report, request.param


def test_should_complete_existing_delayed_stream_with_distinct_solve_costs(complete):
    report, diagnostics = complete
    assert report["sequence_complete"] and report["training_objective"] == OBJECTIVE
    assert report["stream"]["completed_updates"] == 12 and report["stream"]["training_pairs"] == 36
    assert report["learner_work"] == linear_catalog_work(
        12, 36, 8 if diagnostics else 5, 29 if diagnostics else 20
    )
    assert "optimizer_attempts" not in report["learner_work"]
    receipts = report["stream"]["receipts"]
    assert all(
        r["observed_at"] == 2 * i
        and r["arrived_at"] == r["applied_at"] == 2 * i + 1
        and r["diagnostic"]["definition"] == OBJECTIVE
        for i, r in enumerate(receipts)
    )
    assert report["linear_last_operation"]["linear_update"]["complete"]
    assert [r["candidate_ids"] for r in report["rows"]][4] == ["defer", "inspect", "route"]
    scored = study.score_sequence(report)
    assert len(scored["checkpoints"]) == 4
    if diagnostics:
        assert len(study.score_train_choices(report)) == 3


@pytest.mark.parametrize(
    "fault", ["work", "receipt", "objective", "extension", "control_kind", "control_values"]
)
def test_should_refuse_corrupted_completed_evidence_before_scoring(complete, monkeypatch, fault):
    report = deepcopy(complete[0])
    if fault == "work":
        report["learner_work"]["optimizer_steps"] = 1
    elif fault == "receipt":
        report["stream"]["receipts"][0]["arrived_at"] = 0
    elif fault == "objective":
        report["stream"]["receipts"][0]["diagnostic"]["definition"] = study.CATALOG_OBJECTIVE
    elif fault == "extension":
        report["training_replay_consent"] = True
    elif fault == "control_kind":
        report["control_kind"] = "released_head"
    else:
        report["control_probabilities"]["v1"][0][0] = 0.5
    monkeypatch.setattr(study, "_score_cached", lambda *a: pytest.fail("scored bad evidence"))
    with pytest.raises(ValueError):
        study.score_sequence(report)


def test_should_map_labels_by_tool_identity_and_mask_inactive_extreme_before_softmax():
    identity, actions, train, choices, bindings = parts()
    learner = linear.CLMLinearCatalogLearner(identity, actions, bindings)
    learner.train_batch(*train[0])
    assert learner._control is not None
    assert learner._control.candidate_ids == ("defer", "inspect", "retire", "route")
    # Hand-set readout-only oracle: inactive defer would underflow all active
    # probabilities if softmax preceded masking. No native/acquisition claim.
    model = replace(
        learner._control.model, coefficients=((0.0,) * 4, (0.0,) * 4, (10000.0, 0.0, 0.0, 0.0))
    )
    control = replace(learner._control, model=model)
    learner._control = learner._control_owner = control
    learner._model_owner, learner._control_values = model, asdict(control)
    np.testing.assert_allclose(
        learner.predict_choices(choices["baseline"]), np.full((4, 3), 1 / 3), atol=1e-7
    )


@pytest.mark.parametrize("fault", ["final", "labels", "actions", "bound_choice", "coefficients"])
def test_should_reject_invalid_input_or_changed_owner_before_more_solves(monkeypatch, fault):
    identity, actions, train, choices, bindings = parts()
    learner = linear.CLMLinearCatalogLearner(identity, actions, bindings)
    batch, labels = train[0]
    if fault == "final":
        object.__setattr__(batch, "role", "final")
    elif fault == "labels":
        labels = CLMCatalogTargets(tuple(reversed(labels.candidate_ids)), labels.candidate_indices)
    elif fault == "actions":
        batch.actions.add_(0.1)
    elif fault == "bound_choice":
        choices["baseline"].states.add_(0.1)
        with pytest.raises(ValueError):
            learner.predict_choices(choices["baseline"])
        return
    else:
        learner.train_batch(batch, labels)
        assert learner._control is not None
        learner._control = replace(learner._control)
    monkeypatch.setattr(linear, "fit_clm_linear_control", lambda *a: pytest.fail("invalid fit"))
    monkeypatch.setattr(
        linear, "update_clm_linear_control", lambda *a: pytest.fail("invalid update")
    )
    with pytest.raises(ValueError):
        learner.train_batch(batch, labels)


def test_should_keep_lost_solve_prefix_and_stop_without_retry(monkeypatch):
    identity, actions, train, _, bindings = parts()
    learner = linear.CLMLinearCatalogLearner(identity, actions, bindings)
    learner.train_batch(*train[0])
    assert learner._control is not None
    original = asdict(learner._control)
    real = linear.update_clm_linear_control

    def lost(*args):
        real(*args)
        raise RuntimeError("lost returned solve")

    monkeypatch.setattr(linear, "update_clm_linear_control", lost)
    with pytest.raises(TrainingUpdateFailed) as caught:
        learner.train_batch(*train[1])
    assert caught.value.work["solve_attempts"] == caught.value.work["solve_returns"] == 2
    assert learner.completed_updates == 1 and learner.work["training_rows_attempted"] == 6
    assert learner.work["unreturned_work_unknown"] and asdict(learner._control) == original
    with pytest.raises(ValueError, match="no retry"):
        learner.train_batch(*train[1])


@pytest.mark.parametrize("fault", ["receipt", "control_kind"])
def test_should_refuse_mutated_prefix_after_first_paid_update(fault):
    identity, actions, train, choices, _ = parts()
    report: dict[str, Any] = {}

    def save():
        if report.get("stream", {}).get("completed_updates") == 1:
            if fault == "receipt":
                report["stream"]["receipts"][0]["arrived_at"] = -1
            else:
                report["control_kind"] = "released_head"

    with pytest.raises(ValueError):
        linear.run_linear_catalog_stream(identity, actions, train, choices, report, save=save)
    assert not report["sequence_complete"] and report["learner_work"]["commits"] == 1
