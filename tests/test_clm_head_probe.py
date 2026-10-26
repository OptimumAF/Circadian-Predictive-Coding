"""Disposable warm-weight CLM probe boundary; literal tiny heads, no encoder."""

from dataclasses import replace

import pytest

pytest.importorskip("torch")

import src.adapters.clm_head_learner as native
import src.adapters.clm_head_probe as probe_adapter
from src.adapters.clm_head_probe import CLMHeadProbe
from src.core.clm_head import CLMTargets, FrozenChoiceBatch, FrozenEmbeddingBatch
from src.core.clm_probe import FrozenProbeChoiceBatch, FrozenProbeEmbeddingBatch
from src.core.learner_ports import TrainingUpdateFailed
from test_clm_temporal_sequence import owner


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def records(torch, identity):
    source_batch = FrozenEmbeddingBatch(
        identity,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.8, 0.1], [0.2, 1.0]]),
    )
    training = FrozenProbeEmbeddingBatch(
        identity,
        torch.tensor([[0.2, 0.8], [-0.6, 0.4]]),
        torch.tensor([[0.4, 0.9], [-0.7, 0.2]]),
    )
    choice = FrozenProbeChoiceBatch(
        identity,
        torch.tensor([[0.3, 0.7]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-0.5, 0.5]]),
    )
    return source_batch, training, choice


@pytest.fixture(scope="module")
def probe_case(torch):
    declared = native.FrozenEncoderIdentity("fake", "literal", "literal", "last", "owned-probe", 2)
    source, *_ = owner(torch, declared, maximum=2)
    source_batch, training, choice = records(torch, declared)
    targets = CLMTargets((0, 1))
    source.train_batch(source_batch, targets)
    source.train_batch(source_batch, targets)
    normal_choice = FrozenChoiceBatch(declared, choice.states, choice.actions)
    warm_probability = source.predict_choices(normal_choice)
    # A private RNG different from the caller makes the copy assertion non-vacuous.
    saved = source._capture()
    object.__setattr__(saved, "cpu_rng", torch.Generator().manual_seed(99).get_state())
    source.restore_state(saved)
    birth = source._capture()
    source_work = source.work
    source_owners = tuple(source._owners)
    rng = torch.get_rng_state().clone()
    policy = replace(source.policy, max_updates=2, max_restores=0)
    probe = CLMHeadProbe(source, policy=policy, max_readouts=3)
    engine = probe._learner
    fresh = engine._capture()
    fresh_work = probe.work
    initial_probability = probe.predict_choices(choice)
    first = probe.train_batch(training, targets)
    probe.predict_choices(
        FrozenProbeChoiceBatch(declared, choice.states, choice.actions, "probe_train")
    )
    second = probe.train_batch(training, targets)
    probe.predict_choices(
        FrozenProbeChoiceBatch(declared, choice.states, choice.actions, "outer_selection")
    )
    return dict(
        source=source,
        source_batch=source_batch,
        source_birth=birth,
        source_work=source_work,
        source_owners=source_owners,
        rng=rng,
        probe=probe,
        fresh=fresh,
        fresh_work=fresh_work,
        warm=warm_probability,
        initial=initial_probability,
        training=training,
        choice=choice,
        targets=targets,
        policy=policy,
        diagnostics=(first, second),
    )


def require_source_unchanged(torch, case):
    source = case["source"]
    assert native._equal(vars(case["source_birth"]), vars(source._capture()))
    assert source.work == case["source_work"]
    assert all(left is right for left, right in zip(source._owners, case["source_owners"]))
    assert torch.equal(case["rng"], torch.get_rng_state())


def test_should_start_from_spent_source_weights_with_fresh_optimizer_gradients_and_schedule(
    torch, probe_case
):
    case, fresh = probe_case, probe_case["fresh"]
    source = case["source_birth"]
    assert case["source"].completed_updates == 2 and case["source_work"]["train_attempts"] == 2
    assert native._equal(fresh.state_head, source.state_head)
    assert native._equal(fresh.action_head, source.action_head)
    assert torch.equal(fresh.logit_scale, source.logit_scale)
    assert fresh.modes == source.modes and torch.equal(fresh.cpu_rng, source.cpu_rng)
    assert not torch.equal(source.cpu_rng, case["rng"])
    assert source.optimizer_state["state"] and not fresh.optimizer_state["state"]
    assert all(gradient is not None for gradient in source.gradients)
    assert all(gradient is None for gradient in fresh.gradients)
    assert fresh.scheduler_state["last_epoch"] == 0 and source.scheduler_state["last_epoch"] == 2
    assert fresh.completed_updates == fresh.training_examples == 0
    assert all(value == 0 for value in case["fresh_work"].values())
    torch.testing.assert_close(case["initial"], case["warm"], rtol=0, atol=0)
    require_source_unchanged(torch, case)


def test_should_keep_native_objective_and_independent_owned_state(torch, probe_case):
    case, probe = probe_case, probe_case["probe"]
    engine, source = probe._learner, case["source"]
    for original, owned in zip(source._bundle[:2], engine._bundle[:2]):
        assert original is not owned
        for a, b in zip(original.parameters(), owned.parameters()):
            assert a is not b and a.data_ptr() != b.data_ptr()
            assert a.grad is not b.grad and a.grad.data_ptr() != b.grad.data_ptr()
    assert engine._bundle[3] is not source._bundle[3]
    assert engine._bundle[4] is not source._bundle[4]
    assert engine._rng is not source._rng and engine._rng.data_ptr() != source._rng.data_ptr()
    assert probe.completed_updates == 2 and probe.training_examples == 4
    assert probe.source_completed_updates == 2 and probe.source_training_examples == 4
    assert probe.source_generation == 3 and probe.generation == 2
    assert all(
        item.definition == native.OBJECTIVE and item.value >= 0 for item in case["diagnostics"]
    )
    assert (
        probe.work["commits"]
        == probe.work["backward_returns"]
        == probe.work["optimizer_returns"]
        == 2
    )
    assert probe.work["training_rows_attempted"] == 4
    assert not native._equal(engine._bundle[0].state_dict(), source._bundle[0].state_dict())
    require_source_unchanged(torch, case)


def test_should_bound_readouts_across_all_admitted_roles_without_refunding_work(torch, probe_case):
    case, probe = probe_case, probe_case["probe"]
    paid = probe.work
    assert paid["choice_prediction_attempts"] == paid["choice_prediction_returns"] == 3
    assert paid["prediction_rows_attempted"] == 3 and paid["choice_candidate_rows_attempted"] == 9
    with pytest.raises(ValueError):
        probe.predict_choices(case["choice"])
    with pytest.raises(ValueError):
        probe.train_batch(case["training"], case["targets"])
    policy = probe.policy
    object.__setattr__(policy, "max_updates", 3)
    detached = probe.work
    detached.clear()
    assert probe.policy == case["policy"] and probe.work == paid
    require_source_unchanged(torch, case)


def test_should_not_offer_snapshot_install_or_source_chaining(probe_case):
    probe = probe_case["probe"]
    for name in ("snapshot_state", "restore_state", "fork", "export"):
        assert not hasattr(probe, name)
    with pytest.raises(ValueError):
        CLMHeadProbe(probe, policy=probe_case["policy"], max_readouts=1)


@pytest.fixture(scope="module")
def fresh_probe(probe_case):
    return CLMHeadProbe(probe_case["source"], policy=probe_case["policy"], max_readouts=2)


@pytest.mark.parametrize(
    "fault",
    [
        "normal_record",
        "role",
        "identity",
        "extra",
        "rows",
        "dtype",
        "nonfinite",
        "target_count",
        "no_targets",
    ],
)
def test_should_refuse_bad_probe_training_before_native_work(torch, probe_case, fresh_probe, fault):
    case = probe_case
    batch = FrozenProbeEmbeddingBatch(
        case["training"].identity, case["training"].states.clone(), case["training"].actions.clone()
    )
    targets = case["targets"]
    if fault == "normal_record":
        batch = case["source_batch"]
    elif fault == "role":
        object.__setattr__(batch, "role", "train")
    elif fault == "identity":
        object.__setattr__(batch, "identity", replace(batch.identity, revision="foreign"))
    elif fault == "extra":
        object.__setattr__(batch, "undeclared_label", "not admitted")
    elif fault == "rows":
        object.__setattr__(batch, "states", batch.states[:1])
        object.__setattr__(batch, "actions", batch.actions[:1])
    elif fault == "dtype":
        object.__setattr__(batch, "states", batch.states.double())
    elif fault == "nonfinite":
        batch.states[0, 0] = float("nan")
    elif fault == "target_count":
        targets = CLMTargets((0, 1, 2))
    else:
        targets = None
    before = fresh_probe.work
    with pytest.raises(ValueError):
        fresh_probe.train_batch(batch, targets)
    assert fresh_probe.work == before
    require_source_unchanged(torch, case)


@pytest.mark.parametrize(
    "fault",
    [
        "normal_record",
        "final",
        "identity",
        "extra",
        "one_candidate",
        "empty_queries",
        "wide",
        "dtype",
        "nonfinite",
    ],
)
def test_should_refuse_bad_probe_choice_without_consuming_readout_allowance(
    torch, probe_case, fresh_probe, fault
):
    choice: FrozenProbeChoiceBatch | FrozenChoiceBatch = FrozenProbeChoiceBatch(
        probe_case["choice"].identity,
        probe_case["choice"].states.clone(),
        probe_case["choice"].actions.clone(),
    )
    if fault == "normal_record":
        choice = FrozenChoiceBatch(choice.identity, choice.states, choice.actions)
    elif fault == "final":
        object.__setattr__(choice, "role", "final_test")
    elif fault == "identity":
        object.__setattr__(
            choice, "identity", replace(choice.identity, tokenizer_revision="foreign")
        )
    elif fault == "extra":
        object.__setattr__(choice, "label", "not admitted")
    elif fault == "one_candidate":
        object.__setattr__(choice, "actions", choice.actions[:1])
    elif fault == "empty_queries":
        object.__setattr__(choice, "states", choice.states[:0])
    elif fault == "wide":
        object.__setattr__(
            choice, "states", torch.cat((choice.states, choice.states[:, :1]), dim=1)
        )
    elif fault == "dtype":
        object.__setattr__(choice, "actions", choice.actions.double())
    else:
        choice.actions[0, 0] = float("inf")
    before = fresh_probe.work
    with pytest.raises(ValueError):
        fresh_probe.predict_choices(choice)
    assert fresh_probe.work == before
    require_source_unchanged(torch, probe_case)


@pytest.mark.parametrize("operation", ["train", "paired_predict", "choices"])
def test_should_keep_normal_learner_admission_strict_before_spent_budget_checks(
    probe_case, operation
):
    case, source = probe_case, probe_case["source"]
    before = source.work
    with pytest.raises(ValueError, match="expected exact complete Frozen"):
        if operation == "train":
            source.train_batch(case["training"], case["targets"])
        elif operation == "paired_predict":
            source.predict(case["training"])
        else:
            source.predict_choices(case["choice"])
    assert source.work == before
    with pytest.raises(ValueError, match="physical CLM update allowance exhausted"):
        source.train_batch(case["source_batch"], case["targets"])
    assert source.work == before


@pytest.mark.parametrize(
    "kind,role",
    [
        (FrozenProbeEmbeddingBatch, "train"),
        (FrozenProbeEmbeddingBatch, "probe_test"),
        (FrozenProbeEmbeddingBatch, "final_test"),
        (FrozenProbeChoiceBatch, "train"),
        (FrozenProbeChoiceBatch, "final_test"),
        (FrozenProbeChoiceBatch, "inner_guard"),
    ],
)
def test_should_reject_unadmitted_roles_at_probe_record_birth(probe_case, kind, role):
    with pytest.raises(ValueError):
        kind(probe_case["source"].identity, None, None, role)


@pytest.mark.parametrize(
    "fault",
    [
        "horizon10",
        "horizon17",
        "restore",
        "lr",
        "rows",
        "projection",
        "extra",
        "readouts0",
        "readouts65",
        "readouts_bool",
    ],
)
def test_should_reject_invalid_probe_limits_before_native_constructor(
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
    elif fault == "rows":
        policy = replace(policy, max_rows=3)
    elif fault == "projection":
        policy = replace(policy, projection_dim=3)
    elif fault == "extra":
        object.__setattr__(policy, "new_field", True)
    elif fault == "readouts0":
        readouts = 0
    elif fault == "readouts65":
        readouts = 65
    else:
        readouts = True

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid declaration reached native construction")

    monkeypatch.setattr(probe_adapter, "_ProbeHeadLearner", forbidden)
    before = probe_case["source"].work
    with pytest.raises(ValueError):
        CLMHeadProbe(probe_case["source"], policy=policy, max_readouts=readouts)
    assert probe_case["source"].work == before


def test_should_check_original_source_even_when_probe_constructor_does_not_return(
    torch, probe_case, monkeypatch
):
    source, observed = probe_case["source"], []
    require_original = source._require_original

    def require(owner, before):
        observed.append(True)
        return require_original(owner, before)

    def fail(*args, **kwargs):
        raise RuntimeError("native probe construction did not return")

    monkeypatch.setattr(source, "_require_original", require)
    monkeypatch.setattr(probe_adapter, "_ProbeHeadLearner", fail)
    with pytest.raises(RuntimeError, match="construction did not return"):
        CLMHeadProbe(source, policy=probe_case["policy"], max_readouts=1)
    assert observed == [True]
    require_source_unchanged(torch, probe_case)


@pytest.mark.parametrize("operation", ["train", "choice"])
def test_should_preserve_paid_uncertain_probe_costs_and_poison_without_source_changes(
    torch, probe_case, monkeypatch, operation
):
    case = probe_case
    probe = CLMHeadProbe(case["source"], policy=case["policy"], max_readouts=1)

    def fail(*args, **kwargs):
        raise RuntimeError("probe native return lost")

    if operation == "train":
        monkeypatch.setattr(native, "clm_head_loss", fail)
        with pytest.raises(TrainingUpdateFailed) as error:
            probe.train_batch(case["training"], case["targets"])
        assert error.value.work == probe.work
        assert probe.work["train_attempts"] == probe.work["loss_forward_attempts"] == 1
        assert probe.work["training_rows_attempted"] == 2
        assert probe.work["loss_forward_returns"] == probe.work["backward_attempts"] == 0
    else:
        monkeypatch.setattr(probe._learner, "_project", fail)
        with pytest.raises(RuntimeError, match="return lost"):
            probe.predict_choices(case["choice"])
        assert probe.work["choice_prediction_attempts"] == 1
        assert probe.work["prediction_rows_attempted"] == 1
        assert probe.work["choice_candidate_rows_attempted"] == 3
        assert probe.work["choice_prediction_returns"] == 0
    paid = probe.work
    assert paid["unreturned_work_unknown"] is True
    assert probe.completed_updates == probe.training_examples == paid["commits"] == 0
    with pytest.raises(ValueError):
        probe.train_batch(case["training"], case["targets"])
    with pytest.raises(ValueError):
        probe.predict_choices(case["choice"])
    assert probe.work == paid
    require_source_unchanged(torch, case)
