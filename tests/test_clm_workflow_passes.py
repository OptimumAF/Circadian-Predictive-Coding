"""Bounded TRAIN dose and original authority on deterministic literal heads."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import src.app.clm_workflow_training as workflow
from src.adapters.clm_catalog_learner import CLMCatalogLearner
from src.adapters.clm_head_learner import _equal
from src.core.clm_catalog import CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_head import CLMHeadPolicy, FrozenEncoderIdentity


def make_case(passes=8, *, rows=12, restores=0):
    def linear(values):
        layer = torch.nn.Linear.__new__(torch.nn.Linear)
        torch.nn.Module.__init__(layer)
        layer.in_features = layer.out_features = 2
        layer.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        layer.register_parameter("bias", None)
        return layer

    heads = (
        linear([[1.0, 0.2], [-0.1, 0.9]]),
        linear([[0.8, -0.2], [0.3, 1.1]]),
        torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32)),
    )
    heads[0].eval()
    identity = FrozenEncoderIdentity("fake", "literal", "literal", "last", "workflow-passes", 2)
    learner = CLMCatalogLearner(
        *heads,
        identity=identity,
        policy=CLMHeadPolicy(
            projection_dim=2, max_updates=passes, max_rows=rows, max_restores=restores
        ),
    )
    batch = FrozenCatalogTrainingBatch(
        identity,
        torch.tensor(
            [[(i + 1) / 13, (-1) ** i * (i + 2) / 14] for i in range(12)],
            dtype=torch.float32,
        ),
        torch.tensor([[0.9, 0.1], [0.2, 1.0], [-0.6, 0.4], [0.5, -0.2], [-0.3, -0.8]]),
        workflow.ACTIVE_IDS,
    )
    targets = CLMCatalogTargets(workflow.ACTIVE_IDS, workflow.expected_targets())
    return learner, batch, targets, heads


def assert_loaded_unchanged(heads):
    assert heads[0].weight.detach().tolist() == torch.tensor([[1.0, 0.2], [-0.1, 0.9]]).tolist()
    assert heads[1].weight.detach().tolist() == torch.tensor([[0.8, -0.2], [0.3, 1.1]]).tolist()
    assert heads[2].item() == torch.tensor(0.2).item()
    assert not heads[0].training and heads[1].training
    assert all(p.grad is None for head in heads[:2] for p in head.parameters())
    assert heads[2].grad is None


def assert_paid(learner, record, passes):
    assert record["complete"] is True
    assert record["committed_updates"] == learner.completed_updates == learner.generation == passes
    assert record["logical_clock"] == 2 * passes - 1
    assert (
        learner.training_examples
        == record["learner_work"]["training_rows_attempted"]
        == 12 * passes
    )
    assert record["permissions"]["training"] is True
    assert len(record["receipts"]) == passes
    work = learner.work
    for key in (
        "train_attempts",
        "backward_returns",
        "optimizer_returns",
        "scheduler_returns",
        "commits",
    ):
        assert work[key] == passes
    assert work["catalog_candidate_rows_attempted"] == 5 * passes
    assert work["catalog_logit_values_attempted"] == 60 * passes
    assert work["prediction_attempts"] == work["restore_attempts"] == 0
    assert work["unreturned_work_unknown"] is False
    for index, receipt in enumerate(record["receipts"]):
        assert receipt["sample_id"] == f"workflow-pass{index}"
        assert receipt["observed_at"] == 2 * index
        assert receipt["applied_at"] == 2 * index + 1


@pytest.mark.parametrize("passes", [None, False, True, 7, 33, 8.0, "32", 10**10])
def test_should_reject_invalid_passes_before_native_access(passes):
    with pytest.raises(ValueError, match="integer from 8 through 32"):
        workflow.fit_catalog_workflow(SimpleNamespace(), None, None, {}, passes=passes)


@pytest.mark.parametrize(
    "maximum,requested,rows,restores",
    [(8, 32, 12, 0), (32, 8, 12, 0), (8, 8, 13, 0), (8, 8, 12, 1)],
)
def test_should_require_fresh_matching_policy_before_work(maximum, requested, rows, restores):
    learner, batch, targets, _ = make_case(maximum, rows=rows, restores=restores)
    before = learner.work
    report: dict[str, Any] = {}
    with pytest.raises(ValueError, match="fresh exact"):
        workflow.fit_catalog_workflow(learner, batch, targets, report, passes=requested)
    assert learner.work == before and report == {}


def test_should_preserve_default_eight_pass_schema_and_full_native_state():
    rng = torch.get_rng_state().clone()
    first, batch, targets, heads = make_case()
    second, other_batch, other_targets, other_heads = make_case()
    a: dict[str, Any] = {}
    b: dict[str, Any] = {}
    default = workflow.fit_catalog_workflow(first, batch, targets, a)
    explicit = workflow.fit_catalog_workflow(second, other_batch, other_targets, b, passes=8)
    assert_paid(first, default, 8)
    assert default == explicit and "requested_passes" not in default
    assert _equal(vars(first._capture()), vars(second._capture()))
    assert torch.equal(rng, torch.get_rng_state())
    assert_loaded_unchanged(heads)
    assert_loaded_unchanged(other_heads)
    before = first.work
    with pytest.raises(ValueError, match="fresh exact"):
        workflow.fit_catalog_workflow(first, batch, targets, {}, passes=8)
    assert first.work == before


def test_should_equal_direct_thirty_two_step_native_schedule_without_reentry():
    rng = torch.get_rng_state().clone()
    learner, batch, targets, heads = make_case(32)
    reference, other_batch, other_targets, other_heads = make_case(32)
    record = workflow.fit_catalog_workflow(learner, batch, targets, {}, passes=32)
    for _ in range(32):
        reference.train_batch(other_batch, other_targets)
    assert_paid(learner, record, 32)
    assert record["requested_passes"] == 32
    assert learner.work == reference.work
    assert _equal(vars(learner._capture()), vars(reference._capture()))
    assert torch.equal(rng, torch.get_rng_state())
    assert_loaded_unchanged(heads)
    assert_loaded_unchanged(other_heads)
    before = learner.work
    with pytest.raises(ValueError, match="fresh exact"):
        workflow.fit_catalog_workflow(learner, batch, targets, {}, passes=32)
    assert learner.work == before


@pytest.mark.parametrize(
    "change",
    [
        "in_place_limits",
        "replaced_limits",
        "clock_owner",
        "start",
        "tick",
        "record",
        "counter",
        "policy",
    ],
)
def test_should_refuse_changed_authority_after_one_paid_update(monkeypatch, change):
    learner, batch, targets, _ = make_case()
    report: dict[str, Any] = {}
    captured: dict[str, Any] = {}
    original_budget, original_clock = workflow.ToyBudgetSession, workflow.LogicalClock

    def budget(*args, **kwargs):
        captured["budget"] = original_budget(*args, **kwargs)
        return captured["budget"]

    def clock():
        captured["clock"] = original_clock()
        return captured["clock"]

    monkeypatch.setattr(workflow, "ToyBudgetSession", budget)
    monkeypatch.setattr(workflow, "LogicalClock", clock)
    changed = False

    def save():
        nonlocal changed
        record = report["training"]
        if changed or record["committed_updates"] != 1:
            return
        changed = True
        session = captured["budget"]
        if change == "in_place_limits":
            object.__setattr__(session.budget, "max_training_updates", 64)
        elif change == "replaced_limits":
            session.budget = deepcopy(session.budget)
        elif change == "clock_owner":
            session.clock = lambda: 0.0
        elif change == "start":
            session.started_at += 1
        elif change == "tick":
            captured["clock"].advance_to(99)
        elif change == "record":
            report["training"] = deepcopy(record)
        elif change == "counter":
            learner._work["train_attempts"] += 1
        elif change == "policy":
            object.__setattr__(learner._policy, "max_updates", 32)

    with pytest.raises(ValueError, match="changed|paid prefix"):
        workflow.fit_catalog_workflow(learner, batch, targets, report, save=save)
    assert changed and learner.completed_updates == 1
    assert learner.work["optimizer_returns"] == 1
    prefix = report["preserved_training_prefix"]
    assert prefix["complete"] is False and prefix["committed_updates"] == 1
    assert len(prefix["receipts"]) == 1 and prefix["logical_clock"] == 1
    assert prefix["learner_work"]["train_attempts"] == 1
    assert captured["budget"].updates_completed == 1


def test_should_preserve_failed_boundary_prefix_and_deny_fresh_reentry():
    learner, batch, targets, _ = make_case()
    report: dict[str, Any] = {}
    failed = False

    def save():
        nonlocal failed
        if report["training"]["committed_updates"] == 1 and not failed:
            failed = True
            raise OSError("controlled save boundary failure")

    with pytest.raises(OSError, match="controlled save"):
        workflow.fit_catalog_workflow(learner, batch, targets, report, save=save)
    assert report["training"]["complete"] is False
    assert report["preserved_training_prefix"]["committed_updates"] == 1
    assert len(report["preserved_training_prefix"]["receipts"]) == 1
    assert learner.completed_updates == learner.work["optimizer_returns"] == 1
    before = learner.work
    with pytest.raises(ValueError, match="fresh exact"):
        workflow.fit_catalog_workflow(learner, batch, targets, {})
    assert learner.work == before
