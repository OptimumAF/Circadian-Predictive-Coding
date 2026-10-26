"""Paired replay candidates on fixed literal heads, no encoder or scientific run."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("torch")

from src.adapters.clm_head_learner import CLMHeadLearner, OBJECTIVE, _equal, _params
from src.adapters.clm_replay_candidate import CLMReplayCandidate
from src.app.clm_replay_candidate import CLMReplayRecord, run_clm_replay_candidate
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_head import (
    CLMHeadPolicy,
    CLMTargets,
    FrozenChoiceBatch,
    FrozenEmbeddingBatch,
    FrozenEncoderIdentity,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from test_clm_catalog_learning import parts


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def source_case(torch):
    identity = FrozenEncoderIdentity("literal", "paired-replay", "literal", "last", "paired", 2)
    policy = CLMHeadPolicy(2, 2, max_rows=2, max_restores=0)
    source = CLMHeadLearner(*parts(torch), identity=identity, policy=policy)
    batch = FrozenEmbeddingBatch(
        identity, torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[1.0, 0.5], [0.2, 1.0]])
    )
    targets = CLMTargets((0, 1))
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    inbox = ExperienceInbox(
        source, clock=clock, budget=budget, learner_version="source-v2", max_experiences=2
    )
    records = []
    for index in range(2):
        clock.advance_to(2 * index)
        e = Experience(
            f"original-{index}",
            "source",
            2 * index,
            "model-v2",
            batch,
            "train",
            ExperiencePermissions(training=True, replay=True),
        )
        inbox.record_experience(e)
        assert inbox.drain(max_updates=1) == ()
        clock.advance_to(2 * index + 1)
        label = LabelArrival(
            f"label-original-{index}", e.sample_id, "source", clock.now(), "model-v2", targets
        )
        inbox.record_label(label)
        (receipt,) = inbox.drain(max_updates=1)
        records.append(CLMReplayRecord(e, label, receipt, "v2"))
    readout = FrozenChoiceBatch(identity, batch.states.clone(), batch.actions.clone())
    return dict(
        source=source,
        policy=policy,
        history=tuple(records),
        readout=readout,
        source_state=source._capture(),
        source_work=source.work,
        rng=torch.get_rng_state().clone(),
        source_clock=clock,
        source_budget=budget,
    )


def invoke(
    case,
    *,
    factory=CLMReplayCandidate,
    history=None,
    plan=(0, 1),
    before_step=lambda: None,
    save=None,
):
    report: dict[str, Any] = {}
    holder: dict[str, Any] = {}

    def create(*args, **kwargs):
        candidate = factory(*args, **kwargs)
        holder["candidate"] = candidate
        return candidate

    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=4), lambda: 0.0)
    run_clm_replay_candidate(
        case["source"],
        case["history"] if history is None else history,
        plan,
        case["readout"],
        report,
        policy=case["policy"],
        source_version="source-v2",
        current_policy="v2",
        at=4,
        candidate_factory=create,
        budget=budget,
        replay_budget=replay,
        before_step=before_step,
        save=(lambda: None) if save is None else lambda: save(report, holder, budget, replay),
    )
    return dict(report=report, candidate=holder["candidate"], budget=budget, replay=replay)


@pytest.fixture(scope="module")
def completed(source_case):
    return invoke(source_case)


def test_should_replay_original_train_through_inbox_and_keep_origin_and_consumption_clocks(
    completed, source_case
):
    r = completed["report"]
    assert r["complete"] is True and r["construction_attempts"] == r["construction_returns"] == 1
    assert r["replay_rows_attempted"] == r["replay_budget_rows"] == 4
    assert r["update_budget_commits"] == 2
    assert [v["original_key"] for v in r["attempts"]] == [
        ["source", "original-0"],
        ["source", "original-1"],
    ]
    assert [v["original_update"] for v in r["attempts"]] == [1, 2]
    assert [v["original_event_id"] for v in r["attempts"]] == [
        "label-original-0",
        "label-original-1",
    ]
    for index, receipt in enumerate(r["receipts"]):
        assert receipt["sample_id"] == f"clm-replay-{index}"
        assert receipt["observed_at"] == receipt["arrived_at"] == receipt["applied_at"] == 4
        assert (
            receipt["update_number"] == index + 1
            and receipt["diagnostic"]["definition"] == OBJECTIVE
        )
    assert [origin["receipt"]["applied_at"] for origin in r["origins"]] == [1, 3]
    assert [v["phase"] for v in r["readouts"]] == ["before", "after"]
    assert r["candidate_work"]["commits"] == r["candidate_work"]["backward_returns"] == 2
    assert r["candidate_work"]["snapshot_attempts"] == r["candidate_work"]["restore_attempts"] == 0


def test_should_preserve_complete_source_work_state_and_caller_rng(completed, source_case, torch):
    assert _equal(vars(source_case["source"]._capture()), vars(source_case["source_state"]))
    assert source_case["source"].work == source_case["source_work"]
    assert torch.equal(torch.get_rng_state(), source_case["rng"])
    assert source_case["source_clock"].now() == 3
    assert source_case["source_budget"].updates_completed == 2
    child = completed["candidate"]._learner
    parent = source_case["source"]
    assert child._bundle[3] is not parent._bundle[3] and child._bundle[4] is not parent._bundle[4]
    assert child._rng.data_ptr() != parent._rng.data_ptr()
    for a, b in zip(_params(parent._bundle), _params(child._bundle), strict=True):
        assert a is not b and a.data_ptr() != b.data_ptr()
    assert completed["report"]["copy_work"]["source_capture_returns"] > 2
    assert completed["report"]["copy_work"]["source_capture_tensor_elements"] > 0
    assert completed["report"]["copy_work"]["child_rng_copy_elements"] == parent._rng.numel()


def test_should_start_with_exact_warm_values_and_fresh_native_state(source_case, torch):
    child = CLMReplayCandidate(source_case["source"], policy=source_case["policy"])
    assert child.completed_updates == child.generation == child.training_examples == 0
    assert child.work["train_attempts"] == 0 and child._learner._bundle[3].state == {}
    assert child._learner._bundle[4]._step_count == 1
    for a, b in zip(
        _params(source_case["source"]._bundle), _params(child._learner._bundle), strict=True
    ):
        assert torch.equal(a, b) and a is not b and a.data_ptr() != b.data_ptr()
        assert b.grad is None
    assert torch.equal(child._learner._rng, source_case["source"]._rng)
    assert child._learner._rng.data_ptr() != source_case["source"]._rng.data_ptr()
    assert source_case["source"].work == source_case["source_work"]


def test_should_match_ordinary_fresh_optimizer_native_paired_updates(completed, source_case, torch):
    parent = source_case["source"]
    with torch.random.fork_rng(devices=[]):
        control = CLMHeadLearner(
            *parent._bundle[:3], identity=parent.identity, policy=source_case["policy"]
        )
        for parameter in _params(control._bundle):
            parameter.grad = None
        control._rng = parent._rng.clone()
        losses = [
            control.train_batch(v.experience.features, v.label.targets).value
            for v in source_case["history"]
        ]
    assert losses == [v["diagnostic"]["value"] for v in completed["report"]["receipts"]]
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            _params(control._bundle), _params(completed["candidate"]._learner._bundle), strict=True
        )
    )
    assert _equal(
        control._bundle[3].state_dict(), completed["candidate"]._learner._bundle[3].state_dict()
    )
    assert (
        control._bundle[4].state_dict() == completed["candidate"]._learner._bundle[4].state_dict()
    )


def test_should_compute_fresh_projection_readouts_after_weights_change(
    completed, source_case, torch
):
    from torch.nn import functional as F

    state, action, scale = completed["candidate"]._learner._bundle[:3]
    with torch.no_grad():
        expected = (
            scale.exp().clamp(max=100.0)
            * F.normalize(state(source_case["readout"].states), dim=-1)
            @ F.normalize(action(source_case["readout"].actions), dim=-1).t()
        ).softmax(dim=-1)
    before, after = completed["report"]["readouts"]
    assert before["probabilities"] != after["probabilities"]
    assert torch.allclose(expected, torch.tensor(after["probabilities"]), atol=1e-7, rtol=0)
    assert completed["candidate"].work["choice_prediction_returns"] == 2
    with pytest.raises(ValueError, match="exhausted"):
        completed["candidate"].predict_choices(source_case["readout"])
    assert not any(
        hasattr(completed["candidate"], key) for key in ("snapshot_state", "restore_state", "fork")
    )


@pytest.mark.parametrize(
    "fault",
    [
        "consent",
        "role",
        "label-role",
        "future",
        "policy",
        "receipt-key",
        "receipt-clock",
        "receipt-objective",
        "receipt-learner",
        "target",
        "identity",
        "duplicate",
    ],
)
def test_should_refuse_unadmitted_history_before_candidate_construction(
    source_case, monkeypatch, fault
):
    old = source_case["history"][0]
    if fault == "duplicate":
        history = (old, old)
    else:
        record = deepcopy(old)
        if fault == "consent":
            object.__setattr__(
                record.experience, "permissions", ExperiencePermissions(training=True)
            )
        elif fault == "role":
            object.__setattr__(record.experience, "role", "outer_selection")
            object.__setattr__(record.experience, "permissions", ExperiencePermissions())
        elif fault == "label-role":
            object.__setattr__(record.label, "role", "probe_train")
        elif fault == "future":
            object.__setattr__(record.receipt, "applied_at", 5)
        elif fault == "policy":
            object.__setattr__(record, "policy_version", "v1")
        elif fault == "receipt-key":
            object.__setattr__(record.receipt, "sample_id", "foreign")
        elif fault == "receipt-clock":
            object.__setattr__(record.receipt, "arrived_at", 0)
        elif fault == "receipt-objective":
            object.__setattr__(record.receipt.diagnostic, "definition", "catalog")
        elif fault == "receipt-learner":
            object.__setattr__(record.receipt, "learner_version", "foreign")
        elif fault == "target":
            object.__setattr__(record.label, "targets", CLMTargets((0, 1, 2)))
        else:
            object.__setattr__(record.experience.features.identity, "revision", "foreign")
        history = (record, source_case["history"][1])
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("unadmitted candidate constructed")

    with pytest.raises(ValueError):
        invoke(source_case, history=history, factory=forbidden)
    assert not calls


@pytest.mark.parametrize(
    "kwargs",
    [dict(max_updates=10), dict(max_restores=1), dict(projection_dim=3), dict(learning_rate=0.01)],
)
def test_should_preserve_native_source_settings_and_refuse_invalid_candidate_policy(
    source_case, kwargs
):
    with pytest.raises(ValueError):
        CLMReplayCandidate(source_case["source"], policy=replace(source_case["policy"], **kwargs))


def test_should_refuse_probe_catalog_and_nested_candidate_sources_before_copy(
    source_case, completed
):
    from src.adapters.clm_catalog_learner import CLMCatalogLearner
    from src.adapters.clm_head_probe import CLMHeadProbe

    parent = source_case["source"]
    probe = CLMHeadProbe(parent, policy=source_case["policy"], max_readouts=2)
    catalog = CLMCatalogLearner(
        *parent._bundle[:3], identity=parent.identity, policy=source_case["policy"]
    )
    for foreign in (probe, catalog, completed["candidate"]):
        with pytest.raises(ValueError, match="exact paired"):
            CLMReplayCandidate(foreign, policy=source_case["policy"])


def test_should_keep_native_failure_replay_charge_and_original_source(
    source_case, torch, monkeypatch
):
    holder: dict[str, Any] = {}

    def factory(*args, **kwargs):
        child = CLMReplayCandidate(*args, **kwargs)
        holder["child"] = child
        return child

    def fail(*args, **kwargs):
        raise RuntimeError("controlled replay gradient failure")

    report: dict[str, Any] = {}
    update = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=4), lambda: 0.0)
    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd, "backward", fail)
        with pytest.raises(Exception, match="controlled replay gradient failure"):
            run_clm_replay_candidate(
                source_case["source"],
                source_case["history"],
                (0, 1),
                source_case["readout"],
                report,
                policy=source_case["policy"],
                source_version="source-v2",
                current_policy="v2",
                at=4,
                candidate_factory=factory,
                budget=update,
                replay_budget=replay,
            )
    assert (
        report["complete"] is False
        and report["replay_rows_attempted"] == replay.replay_examples_completed == 2
    )
    assert update.updates_completed == 0 and report["receipts"] == []
    assert report["candidate_work"]["train_attempts"] == 1
    assert report["candidate_work"]["unreturned_work_unknown"] is True
    with pytest.raises(ValueError, match="stopped"):
        holder["child"].train_batch(
            source_case["history"][0].experience.features, source_case["history"][0].label.targets
        )
    assert _equal(vars(source_case["source"]._capture()), vars(source_case["source_state"]))


@pytest.mark.parametrize("fault", ["history", "budget", "policy", "prefix"])
def test_should_reject_callback_changes_before_more_native_work(source_case, monkeypatch, fault):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("corruption reached factory")

    history = source_case["history"]
    policy = deepcopy(source_case["policy"])
    report: dict[str, Any] = {}
    update = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=4), lambda: 0.0)

    def change():
        if fault == "history":
            monkeypatch.setattr(history[0], "policy_version", "changed")
        elif fault == "budget":
            monkeypatch.setattr(update, "updates_completed", 1)
        elif fault == "policy":
            object.__setattr__(policy, "max_updates", 3)
        else:
            report["plan"] = [1, 0]

    # Frozen records need controlled object-level mutation rather than normal setattr.
    if fault == "history":
        history = (deepcopy(history[0]), history[1])

        def change():
            object.__setattr__(history[0], "policy_version", "changed")

    with pytest.raises(ValueError):
        run_clm_replay_candidate(
            source_case["source"],
            history,
            (0, 1),
            source_case["readout"],
            report,
            policy=policy,
            source_version="source-v2",
            current_policy="v2",
            at=4,
            candidate_factory=forbidden,
            budget=update,
            replay_budget=replay,
            before_step=change,
        )
    assert calls == [] and replay.replay_examples_completed == 0


@pytest.mark.parametrize("fault", ["shared", "combined", "spent"])
def test_should_refuse_renewed_or_combined_budget_authority_before_factory(source_case, fault):
    update = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=4), lambda: 0.0)
    if fault == "shared":
        replay = update
    elif fault == "combined":
        update = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=2, max_replay_examples=4), lambda: 0.0
        )
    else:
        replay.record_replay(2)

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted budgets reached factory")

    with pytest.raises(ValueError):
        run_clm_replay_candidate(
            source_case["source"],
            source_case["history"],
            (0, 1),
            source_case["readout"],
            {},
            policy=source_case["policy"],
            source_version="source-v2",
            current_policy="v2",
            at=4,
            candidate_factory=forbidden,
            budget=update,
            replay_budget=replay,
        )
