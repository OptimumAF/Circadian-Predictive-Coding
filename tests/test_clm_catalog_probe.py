"""Disposable catalog probes with explicit roles and causal, separately paid work."""

from dataclasses import replace
from typing import Any, Literal, cast

import pytest

pytest.importorskip("torch")

import src.adapters.clm_catalog_learner as catalog
import src.adapters.clm_catalog_probe as probe_adapter
import src.adapters.clm_head_learner as native
from src.adapters.clm_catalog_probe import CLMCatalogProbe
from src.adapters.clm_head_probe import CLMHeadProbe
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_catalog import CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_head import CLMTargets, FrozenChoiceBatch, FrozenEmbeddingBatch
from src.core.clm_probe import FrozenProbeChoiceBatch, FrozenProbeEmbeddingBatch
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingUpdateFailed
from test_clm_catalog_learning import batch as catalog_batch
from test_clm_catalog_learning import owner

IDS = ("alpha", "beta", "gamma")


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def records(torch, identity):
    training = FrozenProbeCatalogBatch(
        identity,
        torch.tensor([[0.2, 0.8], [-0.6, 0.4]], dtype=torch.float32),
        torch.tensor([[0.4, 0.9], [-0.7, 0.2], [0.8, -0.3]], dtype=torch.float32),
        IDS,
    )
    choice = FrozenProbeChoiceBatch(
        identity,
        torch.tensor([[0.3, 0.7]], dtype=torch.float32),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-0.5, 0.5]], dtype=torch.float32),
    )
    return training, CLMCatalogTargets(IDS, (1, 1)), choice


@pytest.fixture(scope="module")
def probe_case(torch):
    source, source_parts = owner(torch, maximum=2)
    source_batch, source_targets = catalog_batch(torch)
    source.train_batch(source_batch, source_targets)
    source.train_batch(source_batch, source_targets)
    training, targets, choice = records(torch, source.identity)
    warm = source.predict_choices(FrozenChoiceBatch(source.identity, choice.states, choice.actions))
    state = source._capture()
    object.__setattr__(state, "cpu_rng", torch.Generator().manual_seed(99).get_state())
    # The objective-specific public record is required even for this RNG-only restore.
    source.restore_state(catalog.CLMCatalogSnapshot(**vars(state)))
    birth, source_work = source._capture(), source.work
    source_owners = tuple(source._owners)
    source_storage = tuple(source._storage)
    rng = torch.get_rng_state().clone()
    policy = replace(source.policy, max_updates=2, max_restores=0)
    probe = CLMCatalogProbe(source, policy=policy, max_readouts=3)
    fresh = probe._learner._capture()
    fresh_work = probe.work
    initial = probe.predict_choices(choice)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    permissions = ExperiencePermissions(training=True, replay=False, evaluation=False)
    # Inbox updates use train_batch; probes intentionally omit its broader
    # NativeLearner annotation's predict/snapshot/restore capabilities.
    inbox: ExperienceInbox[Any, Any, Any, Any] = ExperienceInbox(
        cast(Any, probe),
        clock=clock,
        budget=budget,
        learner_version="catalog-probe",
        max_experiences=2,
        training_role="probe_train",
    )
    for index in range(2):
        observed = 2 * index
        clock.advance_to(observed)
        sample = f"probe-{index}"
        inbox.record_experience(
            Experience(sample, "probe", observed, "source", training, "probe_train", permissions)
        )
        assert inbox.drain(max_updates=1) == ()
        assert probe.completed_updates == index
        clock.advance_to(observed + 1)
        inbox.record_label(
            LabelArrival(
                "label-" + sample, sample, "probe", observed + 1, "source", targets, "probe_train"
            )
        )
        assert len(inbox.drain(max_updates=1)) == 1
        role = "probe_train" if index == 0 else "outer_selection"
        probe.predict_choices(
            FrozenProbeChoiceBatch(choice.identity, choice.states, choice.actions, role)
        )
    return dict(
        source=source,
        source_parts=source_parts,
        source_batch=source_batch,
        source_targets=source_targets,
        birth=birth,
        source_work=source_work,
        source_owners=source_owners,
        source_storage=source_storage,
        rng=rng,
        policy=policy,
        probe=probe,
        fresh=fresh,
        fresh_work=fresh_work,
        initial=initial,
        warm=warm,
        training=training,
        targets=targets,
        choice=choice,
        inbox=inbox,
        budget=budget,
        permissions=permissions,
    )


def require_source_unchanged(torch, case):
    source = case["source"]
    assert native._equal(vars(case["birth"]), vars(source._capture()))
    assert source.work == case["source_work"] and tuple(source._storage) == case["source_storage"]
    assert all(a is b for a, b in zip(source._owners, case["source_owners"], strict=True))
    assert torch.equal(torch.get_rng_state(), case["rng"])


def test_should_warm_start_from_spent_catalog_weights_with_fresh_native_optimizer(
    torch, probe_case
):
    case = probe_case
    source, fresh = case["birth"], case["fresh"]
    assert source.completed_updates == 2 and case["source_work"]["train_attempts"] == 2
    assert native._equal(fresh.state_head, source.state_head)
    assert native._equal(fresh.action_head, source.action_head)
    assert torch.equal(fresh.logit_scale, source.logit_scale)
    assert fresh.modes == source.modes and torch.equal(fresh.cpu_rng, source.cpu_rng)
    assert not torch.equal(fresh.cpu_rng, case["rng"])
    assert source.optimizer_state["state"] and not fresh.optimizer_state["state"]
    assert all(gradient is not None for gradient in source.gradients)
    assert all(gradient is None for gradient in fresh.gradients)
    assert source.scheduler_state["last_epoch"] == 2 and fresh.scheduler_state["last_epoch"] == 0
    assert fresh.completed_updates == fresh.training_examples == 0
    assert all(value == 0 for value in case["fresh_work"].values())
    torch.testing.assert_close(case["initial"], case["warm"], rtol=0, atol=0)
    require_source_unchanged(torch, case)


def test_should_own_separate_catalog_parameters_modules_schedule_and_private_rng(torch, probe_case):
    case, probe = probe_case, probe_case["probe"]
    source, owned = case["source"], probe._learner
    assert not native._equal(case["fresh"].state_head, owned._bundle[0].state_dict())
    assert not native._equal(case["fresh"].action_head, owned._bundle[1].state_dict())
    for original, copied in zip(source._bundle[:2], owned._bundle[:2], strict=True):
        assert original is not copied
        assert not (
            {id(module) for module in original.modules()}
            & {id(module) for module in copied.modules()}
        )
        for a, b in zip(original.parameters(), copied.parameters(), strict=True):
            assert a is not b and a.data_ptr() != b.data_ptr()
            assert a.grad is not b.grad and a.grad.data_ptr() != b.grad.data_ptr()
    assert source._bundle[3] is not owned._bundle[3] and source._bundle[4] is not owned._bundle[4]
    assert source._rng is not owned._rng and source._rng.data_ptr() != owned._rng.data_ptr()
    assert all(value is not source for value in vars(probe).values())
    assert probe.source_completed_updates == 2 and probe.source_training_examples == 4
    assert probe.source_generation == source.generation == 3
    assert probe.completed_updates == probe.generation == 2 and probe.training_examples == 4
    require_source_unchanged(torch, case)


def test_should_charge_rectangular_repeated_label_updates_only_after_delayed_probe_labels(
    torch, probe_case
):
    case, work = probe_case, probe_case["probe"].work
    assert case["training"].states.shape == (2, 2) and case["training"].actions.shape == (3, 2)
    assert case["targets"].candidate_indices == (1, 1)
    assert work["train_attempts"] == work["commits"] == work["backward_returns"] == 2
    assert work["training_rows_attempted"] == 4
    assert work["catalog_candidate_rows_attempted"] == 6
    assert work["catalog_logit_values_attempted"] == 12
    assert work["choice_prediction_attempts"] == work["choice_prediction_returns"] == 3
    assert work["prediction_rows_attempted"] == 3 and work["choice_candidate_rows_attempted"] == 9
    assert work["snapshot_attempts"] == work["restore_attempts"] == 0
    assert work["unreturned_work_unknown"] is False
    receipts = case["inbox"].applied_updates
    assert [
        (row.observed_at, row.arrived_at, row.applied_at, row.update_number) for row in receipts
    ] == [(0, 1, 1, 1), (2, 3, 3, 2)]
    assert all(row.diagnostic.definition == catalog.OBJECTIVE for row in receipts)
    assert case["budget"].updates_completed == 2 and case["budget"].replay_examples_completed == 0
    assert case["permissions"] == ExperiencePermissions(training=True)
    require_source_unchanged(torch, case)


def test_should_keep_probe_caps_paid_and_expose_no_install_restore_or_chaining(probe_case):
    case, probe = probe_case, probe_case["probe"]
    before = probe.work
    with pytest.raises(ValueError):
        probe.train_batch(case["training"], case["targets"])
    with pytest.raises(ValueError):
        probe.predict_choices(case["choice"])
    for name in ("snapshot_state", "restore_state", "fork", "export", "install"):
        assert not hasattr(probe, name)
    with pytest.raises(ValueError):
        CLMCatalogProbe(probe, policy=case["policy"], max_readouts=1)
    detached = probe.work
    detached.clear()
    policy = probe.policy
    object.__setattr__(policy, "max_updates", 3)
    assert probe.work == before and probe.policy == case["policy"]


@pytest.fixture(scope="module")
def fresh_probe(probe_case):
    return CLMCatalogProbe(probe_case["source"], policy=probe_case["policy"], max_readouts=2)


@pytest.mark.parametrize(
    "fault",
    (
        "train_catalog",
        "paired",
        "probe_paired",
        "role",
        "identity",
        "extra",
        "order",
        "target_type",
        "target_order",
        "target_count",
        "index",
        "rows",
        "candidates",
        "dtype",
        "nonfinite",
    ),
)
def test_should_refuse_unbound_catalog_probe_training_before_any_paid_work(
    torch, probe_case, fresh_probe, fault
):
    training, targets, _ = records(torch, probe_case["source"].identity)
    if fault == "train_catalog":
        training = FrozenCatalogTrainingBatch(
            training.identity, training.states, training.actions, IDS
        )
    elif fault == "paired":
        training = FrozenEmbeddingBatch(training.identity, training.states, training.actions[:2])
    elif fault == "probe_paired":
        training = FrozenProbeEmbeddingBatch(
            training.identity, training.states, training.actions[:2]
        )
    elif fault == "role":
        object.__setattr__(training, "role", "train")
    elif fault == "identity":
        object.__setattr__(training, "identity", replace(training.identity, revision="foreign"))
    elif fault == "extra":
        object.__setattr__(training, "label", 1)
    elif fault == "order":
        object.__setattr__(training, "candidate_ids", tuple(reversed(IDS)))
    elif fault == "target_type":
        targets = CLMTargets((1, 1))
    elif fault == "target_order":
        targets = CLMCatalogTargets(tuple(reversed(IDS)), (1, 1))
    elif fault == "target_count":
        targets = CLMCatalogTargets(IDS, (1, 1, 1))
    elif fault == "index":
        object.__setattr__(targets, "candidate_indices", (True, 1))
    elif fault == "rows":
        object.__setattr__(training, "states", training.states[:1])
    elif fault == "candidates":
        object.__setattr__(training, "actions", training.actions[:2])
    elif fault == "dtype":
        object.__setattr__(training, "actions", training.actions.double())
    else:
        training.states[0, 0] = float("nan")
    before = fresh_probe.work
    with pytest.raises(ValueError):
        fresh_probe.train_batch(training, targets)
    assert fresh_probe.work == before
    require_source_unchanged(torch, probe_case)


@pytest.mark.parametrize(
    "fault",
    ("normal", "final", "identity", "extra", "candidates", "queries", "wide", "dtype", "nonfinite"),
)
def test_should_refuse_invalid_probe_readout_without_spending_readout_allowance(
    torch, probe_case, fresh_probe, fault
):
    _, _, choice = records(torch, probe_case["source"].identity)
    if fault == "normal":
        choice = FrozenChoiceBatch(choice.identity, choice.states, choice.actions)
    elif fault == "final":
        object.__setattr__(choice, "role", "final_test")
    elif fault == "identity":
        object.__setattr__(
            choice, "identity", replace(choice.identity, tokenizer_revision="foreign")
        )
    elif fault == "extra":
        object.__setattr__(choice, "target", "alpha")
    elif fault == "candidates":
        object.__setattr__(choice, "actions", choice.actions[:1])
    elif fault == "queries":
        object.__setattr__(choice, "states", choice.states[:0])
    elif fault == "wide":
        object.__setattr__(choice, "states", choice.states[:, :1])
    elif fault == "dtype":
        object.__setattr__(choice, "states", choice.states.double())
    else:
        choice.actions[0, 0] = float("inf")
    before = fresh_probe.work
    with pytest.raises(ValueError):
        fresh_probe.predict_choices(choice)
    assert fresh_probe.work == before
    require_source_unchanged(torch, probe_case)


@pytest.mark.parametrize("role", ("train", "probe_test", "outer_selection", "final_test"))
def test_should_refuse_nontraining_roles_at_probe_catalog_record_birth(probe_case, role):
    with pytest.raises(ValueError):
        FrozenProbeCatalogBatch(probe_case["source"].identity, None, None, IDS, role)


@pytest.mark.parametrize(
    "fault",
    (
        "horizon10",
        "horizon17",
        "restore",
        "lr",
        "weight_decay",
        "rows",
        "projection",
        "parameters",
        "extra",
        "readouts0",
        "readouts65",
        "readouts_bool",
    ),
)
def test_should_reject_changed_probe_recipe_before_native_constructor(
    probe_case, monkeypatch, fault
):
    policy, readouts = replace(probe_case["policy"]), 1
    if fault == "horizon10":
        policy = replace(policy, max_updates=10)
    elif fault == "horizon17":
        policy = replace(policy, max_updates=17)
    elif fault == "restore":
        policy = replace(policy, max_restores=1)
    elif fault == "lr":
        policy = replace(policy, learning_rate=0.002)
    elif fault == "weight_decay":
        policy = replace(policy, weight_decay=0.02)
    elif fault == "rows":
        policy = replace(policy, max_rows=3)
    elif fault == "projection":
        policy = replace(policy, projection_dim=3)
    elif fault == "parameters":
        policy = replace(policy, max_parameters=128)
    elif fault == "extra":
        object.__setattr__(policy, "new_field", True)
    elif fault == "readouts0":
        readouts = 0
    elif fault == "readouts65":
        readouts = 65
    else:
        readouts = True

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid catalog probe reached native constructor")

    monkeypatch.setattr(probe_adapter, "_ProbeCatalogLearner", forbidden)
    before = probe_case["source"].work
    with pytest.raises(ValueError):
        CLMCatalogProbe(probe_case["source"], policy=policy, max_readouts=readouts)
    assert probe_case["source"].work == before


def test_should_preserve_source_checks_when_catalog_probe_construction_fails(
    torch, probe_case, monkeypatch
):
    source, calls = probe_case["source"], []
    require = source._require_original

    def observed(owner, before):
        calls.append(True)
        return require(owner, before)

    def fail(*args, **kwargs):
        raise RuntimeError("catalog probe constructor did not return")

    monkeypatch.setattr(source, "_require_original", observed)
    monkeypatch.setattr(probe_adapter, "_ProbeCatalogLearner", fail)
    with pytest.raises(RuntimeError, match="did not return"):
        CLMCatalogProbe(source, policy=probe_case["policy"], max_readouts=1)
    assert calls == [True]
    require_source_unchanged(torch, probe_case)


@pytest.mark.parametrize("operation", ("train", "choices"))
def test_should_keep_uncertain_catalog_probe_costs_and_poison_without_changing_source(
    torch, probe_case, monkeypatch, operation
):
    case = probe_case
    probe = CLMCatalogProbe(case["source"], policy=case["policy"], max_readouts=1)

    def fail(*args, **kwargs):
        raise RuntimeError("catalog probe native return lost")

    if operation == "train":
        monkeypatch.setattr(catalog, "catalog_head_loss", fail)
        with pytest.raises(TrainingUpdateFailed) as error:
            probe.train_batch(case["training"], case["targets"])
        assert error.value.definition == catalog.OBJECTIVE and error.value.work == probe.work
        assert probe.work["train_attempts"] == probe.work["loss_forward_attempts"] == 1
        assert probe.work["training_rows_attempted"] == 2
        assert probe.work["catalog_candidate_rows_attempted"] == 3
        assert probe.work["catalog_logit_values_attempted"] == 6
        assert probe.work["loss_forward_returns"] == probe.work["backward_attempts"] == 0
    else:
        monkeypatch.setattr(probe._learner, "_project", fail)
        with pytest.raises(RuntimeError, match="return lost"):
            probe.predict_choices(case["choice"])
        assert probe.work["choice_prediction_attempts"] == 1
        assert probe.work["choice_prediction_returns"] == 0
        assert probe.work["prediction_rows_attempted"] == 1
        assert probe.work["choice_candidate_rows_attempted"] == 3
        assert probe.work["catalog_candidate_rows_attempted"] == 0
    paid = probe.work
    assert paid["unreturned_work_unknown"] is True and paid["commits"] == 0
    assert probe.completed_updates == probe.training_examples == 0
    with pytest.raises(ValueError):
        probe.train_batch(case["training"], case["targets"])
    with pytest.raises(ValueError):
        probe.predict_choices(case["choice"])
    assert probe.work == paid
    require_source_unchanged(torch, case)


@pytest.fixture(scope="module")
def paired_source(probe_case):
    source = probe_case["source"]
    return native.CLMHeadLearner(
        *source._bundle[:3], identity=source.identity, policy=source.policy
    )


@pytest.mark.parametrize("fault", ("paired", "subclass", "probe"))
def test_should_reject_every_nonexact_catalog_source_before_any_construction(
    probe_case, paired_source, fault
):
    if fault == "paired":
        source = paired_source
    elif fault == "subclass":

        class Derived(catalog.CLMCatalogLearner):
            pass

        source = Derived.__new__(Derived)
    else:
        source = probe_case["probe"]
    with pytest.raises(ValueError):
        CLMCatalogProbe(source, policy=probe_case["policy"], max_readouts=1)
    with pytest.raises(ValueError):
        CLMHeadProbe(probe_case["source"], policy=probe_case["policy"], max_readouts=1)


@pytest.mark.parametrize("operation", ("train", "paired", "choice"))
def test_should_keep_catalog_benchmark_admission_strict_for_probe_features(probe_case, operation):
    source, before = probe_case["source"], probe_case["source"].work
    with pytest.raises(ValueError):
        if operation == "train":
            source.train_batch(probe_case["training"], probe_case["targets"])
        elif operation == "paired":
            source.predict(probe_case["training"])
        else:
            source.predict_choices(probe_case["choice"])
    assert source.work == before


@pytest.mark.parametrize("fault", ("consent", "benchmark_role", "budget"))
def test_should_refuse_unconsented_or_misbound_probe_inbox_before_native_work(
    torch, probe_case, fresh_probe, fault
):
    training, _, _ = records(torch, probe_case["source"].identity)
    clock = LogicalClock()
    limit = 3 if fault == "budget" else 2
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=limit), lambda: 0.0)
    before = fresh_probe.work
    with pytest.raises(ValueError):
        inbox = ExperienceInbox(
            fresh_probe,
            clock=clock,
            budget=budget,
            learner_version="catalog-probe-refusal",
            max_experiences=2,
            training_role="probe_train",
        )
        permissions = ExperiencePermissions(training=fault != "consent")
        role: Literal["train", "probe_train"] = (
            "train" if fault == "benchmark_role" else "probe_train"
        )
        inbox.record_experience(
            Experience("refused", "probe", 0, "source", training, role, permissions)
        )
    assert fresh_probe.work == before and budget.updates_completed == 0
    require_source_unchanged(torch, probe_case)
