"""Literal gradient observations: same native updates, no extra model passes."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import importlib.util
import json
from math import fsum, sqrt
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.adapters.clm_catalog_learner import CLMCatalogLearner
from src.adapters.clm_catalog_probe import CLMCatalogProbe
from src.adapters.clm_head_learner import CLMHeadLearner, _equal
from src.adapters.clm_head_probe import CLMHeadProbe
from src.adapters.clm_state_only_catalog_learner import CLMStateOnlyCatalogLearner
import src.app.clm_workflow_training as workflow
from src.core.clm_catalog import CLMCatalogTargets
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_head import CLMHeadPolicy, CLMTargets, FrozenEmbeddingBatch
from src.core.clm_probe import FrozenProbeEmbeddingBatch
from src.core.clm_training_observation import CLMGradientObservation, CLMObservedUpdate
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from test_clm_catalog_learning import parts, identity, batch, owner
from test_clm_workflow_passes import make_case


def learner(kind, maximum=3):
    classes = {
        "paired": CLMHeadLearner,
        "catalog": CLMCatalogLearner,
        "state_only": CLMStateOnlyCatalogLearner,
    }
    return classes[kind](
        *parts(torch),
        identity=identity(),
        policy=CLMHeadPolicy(projection_dim=2, max_updates=maximum, max_rows=12, max_restores=0),
    )


def packet(kind):
    values, targets = batch(torch)
    if kind == "paired":
        return FrozenEmbeddingBatch(values.identity, values.states, values.actions[:2]), CLMTargets(
            (0, 1)
        )
    return values, targets


@pytest.mark.parametrize("kind", ["paired", "catalog", "state_only"])
def test_should_return_existing_norm_with_exact_default_state_work_loss_and_rng(kind, monkeypatch):
    ordinary, observed = learner(kind), learner(kind)
    values, targets = packet(kind)
    rng = torch.get_rng_state().clone()
    real_clip, calls = torch.nn.utils.clip_grad_norm_, []

    def clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        raw = [v for p in parameters for v in p.grad.detach().flatten().tolist()]
        expected = sqrt(fsum(v * v for v in raw))
        result = real_clip(parameters, *args, **kwargs)
        calls.append((float(result), expected, len(parameters)))
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    original_fields = set(vars(observed))
    for update in (1, 2):
        loss = ordinary.train_batch(values, targets)
        result = observed.train_batch_observed(values, targets)
        assert type(loss) is TrainingDiagnostic and type(result) is CLMObservedUpdate
        assert result.diagnostic == loss
        item = result.observation
        assert item.committed_update == update and item.training_rows == 2
        assert item.gradient_tensor_count == (1 if kind == "state_only" else 3)
        actual, expected, _ = calls[-1]
        assert item.pre_clip_l2_norm == actual == calls[-2][0]
        assert item.pre_clip_l2_norm > 0.0
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
        assert _equal(vars(ordinary._capture()), vars(observed._capture()))
        assert ordinary.work == observed.work
        assert set(vars(observed)) == original_fields  # No hidden observation history.
        assert torch.equal(torch.get_rng_state(), rng)
        json.dumps(asdict(result), allow_nan=False)
    assert len(calls) == 4 and observed.work["backward_returns"] == 2
    with pytest.raises(FrozenInstanceError):
        result.observation.pre_clip_l2_norm = 9.0


def test_should_record_zero_gradient_without_fabricating_unpaid_or_skipped_update():
    source = learner("catalog")
    values, _ = packet("catalog")
    values = replace(
        values, actions=torch.tensor([[1.0, 0.0], [1.0, 0.0]]), candidate_ids=("a", "b")
    )
    result = source.train_batch_observed(values, CLMCatalogTargets(("a", "b"), (0, 1)))
    assert result.observation.pre_clip_l2_norm == 0.0
    assert (
        source.work["backward_returns"]
        == source.work["optimizer_returns"]
        == source.work["commits"]
        == 1
    )


@pytest.mark.parametrize("scope", ["all", "state_only"])
def test_should_observe_disposable_probe_with_original_source_preserved(scope):
    source, _ = owner(torch)
    source_state, work, rng = source._capture(), source.work, torch.get_rng_state().clone()
    probe = CLMCatalogProbe(
        source,
        policy=replace(source.policy, max_updates=6, max_restores=0),
        max_readouts=6,
        parameter_scope=scope,
    )
    ordinary = CLMCatalogProbe(
        source,
        policy=replace(source.policy, max_updates=6, max_restores=0),
        max_readouts=6,
        parameter_scope=scope,
    )
    values, targets = batch(torch)
    probe_batch = FrozenProbeCatalogBatch(
        values.identity, values.states, values.actions, values.candidate_ids
    )
    result = probe.train_batch_observed(probe_batch, targets)
    assert result.diagnostic == ordinary.train_batch(probe_batch, targets)
    assert _equal(vars(probe._learner._capture()), vars(ordinary._learner._capture()))
    assert probe.work == ordinary.work
    assert result.observation.committed_update == 1
    assert result.observation.gradient_tensor_count == (1 if scope == "state_only" else 3)
    assert probe.work["commits"] == 1
    assert source.work == work and _equal(vars(source._capture()), vars(source_state))
    assert torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(ValueError):
        probe.train_batch_observed(values, targets)
    assert probe.work["commits"] == 1


def test_should_preserve_paired_disposable_probe_state_and_role_admission():
    source = learner("paired")
    source_state, work, rng = source._capture(), source.work, torch.get_rng_state().clone()
    policy = replace(source.policy, max_updates=6)
    ordinary = CLMHeadProbe(source, policy=policy, max_readouts=6)
    observed = CLMHeadProbe(source, policy=policy, max_readouts=6)
    values, targets = packet("paired")
    probe_batch = FrozenProbeEmbeddingBatch(values.identity, values.states, values.actions)
    result = observed.train_batch_observed(probe_batch, targets)
    assert result.diagnostic == ordinary.train_batch(probe_batch, targets)
    assert result.observation.gradient_tensor_count == 3
    assert _equal(vars(observed._learner._capture()), vars(ordinary._learner._capture()))
    assert observed.work == ordinary.work
    assert source.work == work and _equal(vars(source._capture()), vars(source_state))
    assert torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(ValueError):
        observed.train_batch_observed(values, targets)
    assert observed.work["commits"] == 1


@pytest.mark.parametrize("flag", [None, 0, 1, "yes"])
def test_should_refuse_non_boolean_flag_without_paid_native_update(flag):
    source = learner("catalog")
    values, targets = packet("catalog")
    work = source.work
    with pytest.raises(ValueError, match="boolean"):
        source.train_batch(values, targets, record_gradient_norm=flag)
    assert source.work == work


@pytest.mark.parametrize(
    "field,value",
    [
        ("committed_update", True),
        ("training_rows", 0),
        ("gradient_tensor_count", 1.0),
        ("pre_clip_l2_norm", -1.0),
        ("pre_clip_l2_norm", float("nan")),
        ("pre_clip_l2_norm", 1),
        ("clip_limit", 2.0),
        ("definition", "loss"),
    ],
)
def test_should_refuse_invalid_scalar_observation(field, value):
    options: dict[str, Any] = dict(
        committed_update=1, training_rows=2, gradient_tensor_count=3, pre_clip_l2_norm=0.0
    )
    options[field] = value
    with pytest.raises(ValueError):
        CLMGradientObservation(**options)


def test_should_keep_paid_native_failure_without_returning_or_storing_observation(monkeypatch):
    source = learner("catalog")
    values, targets = packet("catalog")
    before = source._capture()

    def fail(*args, **kwargs):
        raise RuntimeError("controlled optimizer failure")

    monkeypatch.setattr(torch.optim.AdamW, "step", fail)
    with pytest.raises(TrainingUpdateFailed):
        source.train_batch_observed(values, targets)
    assert source.work["backward_returns"] == source.work["optimizer_attempts"] == 1
    assert source.work["optimizer_returns"] == source.completed_updates == 0
    assert source.work["unreturned_work_unknown"] is True
    assert _equal(vars(source._capture()), vars(before))
    with pytest.raises(ValueError):
        source.train_batch_observed(values, targets)


def test_should_bind_eight_observations_to_original_delayed_receipts_and_same_state():
    ordinary, values, targets, _ = make_case()
    observed, observed_values, observed_targets, _ = make_case()
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    workflow.fit_catalog_workflow(ordinary, values, targets, first)
    workflow.fit_catalog_workflow(
        observed, observed_values, observed_targets, second, record_gradient_norm=True
    )
    result = deepcopy(second["training"])
    norms = result.pop("gradient_observations")
    assert len(norms) == 8 and [r["committed_update"] for r in norms] == list(range(1, 9))
    assert [r["applied_at"] for r in result["receipts"]] == list(range(1, 16, 2))
    assert result.pop("gradient_observation_definition") == norms[0]["definition"]
    assert result == first["training"]
    assert _equal(vars(ordinary._capture()), vars(observed._capture()))
    assert ordinary.work == observed.work


@pytest.mark.parametrize(
    "fault",
    [
        "value",
        "list_owner",
        "row_owner",
        "returned_value",
        "returned_owner",
        "save",
        "budget",
        "features",
    ],
)
def test_should_preserve_trusted_observation_and_stop_before_second_paid_update(monkeypatch, fault):
    source, values, targets, _ = make_case()
    report: dict[str, Any] = {}
    captured: dict[str, Any] = {}
    real_inbox = workflow.ExperienceInbox

    def inbox(port, **options):
        result = real_inbox(port, **options)
        captured.update(port=port, inbox=result)
        return result

    monkeypatch.setattr(workflow, "ExperienceInbox", inbox)
    changed = False

    def save():
        nonlocal changed
        record = report.get("training", {})
        if changed or record.get("committed_updates") != 1:
            return
        changed = True
        if fault == "value":
            record["gradient_observations"][0]["pre_clip_l2_norm"] = -1.0
        elif fault == "list_owner":
            record["gradient_observations"] = deepcopy(record["gradient_observations"])
        elif fault == "row_owner":
            record["gradient_observations"][0] = deepcopy(record["gradient_observations"][0])
        elif fault == "returned_value":
            object.__setattr__(captured["port"].last_update.observation, "pre_clip_l2_norm", -1.0)
        elif fault == "returned_owner":
            captured["port"].last_update = deepcopy(captured["port"].last_update)
        elif fault == "save":
            raise OSError("controlled saved observation failure")
        elif fault == "budget":
            object.__setattr__(captured["inbox"]._budget.budget, "max_training_updates", 9)
        elif fault == "features":
            values.states.add_(0.1)

    with pytest.raises((ValueError, OSError)):
        workflow.fit_catalog_workflow(
            source, values, targets, report, record_gradient_norm=True, save=save
        )
    assert changed and source.work["commits"] == 1
    prefix = report["preserved_training_prefix"]
    assert prefix["committed_updates"] == len(prefix["gradient_observations"]) == 1
    assert prefix["gradient_observations"][0]["pre_clip_l2_norm"] >= 0.0
    with pytest.raises(ValueError):
        workflow.fit_catalog_workflow(source, values, targets, {}, record_gradient_norm=True)
    assert source.work["commits"] == 1


def test_should_keep_unreturned_native_commit_paid_without_inventing_norm(monkeypatch):
    source, values, targets, _ = make_case()
    real = source.train_batch_observed
    report: dict[str, Any] = {}

    def lost(*args):
        real(*args)
        raise RuntimeError("controlled lost observed return")

    monkeypatch.setattr(source, "train_batch_observed", lost)
    with pytest.raises((ValueError, RuntimeError)):
        workflow.fit_catalog_workflow(source, values, targets, report, record_gradient_norm=True)
    assert source.work["commits"] == 1
    assert report["training"]["gradient_observations"] == []
    assert report["training"]["committed_updates"] == 0


def test_should_keep_saved_before_default_workflow_report_exact():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/clm-gradient-observation-20261025/workflow_before.py"
    )
    spec = importlib.util.spec_from_file_location("before_gradient_workflow", path)
    assert spec is not None and spec.loader is not None
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    source, values, targets, _ = make_case()
    other, other_values, other_targets, _ = make_case()
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    previous.fit_catalog_workflow(source, values, targets, first)
    workflow.fit_catalog_workflow(other, other_values, other_targets, second)
    assert first == second and "gradient_observations" not in first["training"]


@pytest.mark.parametrize("flag", [0, None, "yes"])
def test_should_refuse_invalid_workflow_opt_in_before_native_access(flag):
    with pytest.raises(ValueError, match="boolean"):
        workflow.fit_catalog_workflow(None, None, None, {}, record_gradient_norm=flag)
