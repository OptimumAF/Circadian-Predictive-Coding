"""Cue routing against unchanged native heads; fixtures are not game outcomes."""

from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from src.adapters import minigrid_context_torch_feature_learner as port
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.app.coupled_attribution import AttributionBatch, AttributionStream, _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingUpdateFailed
from src.core.minigrid_context_encoding import context_public_history
from src.shared.torch_runtime import require_torch
from test_invariant_torch_feature_learner import data
from test_torch_feature_learner import NAMES, assert_tree, tensors

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def policy(method="bp"):
    return port.ContextTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 4, 17, num_classes=7))


def projected(x):
    return np.array(
        [context_public_history(tuple(float(v) for v in row))[:296] for row in x], dtype=np.float64
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_match_both_native_heads_full_state_and_owned_continuation(method):
    torch = require_torch()
    global_rng, numpy_rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    selected = policy(method)
    owner = port.ContextTorchFeatureLearner(selected)
    reference = tuple(TorchFeatureLearner(selected.native_policy) for _ in range(2))
    x, y = data()
    before_arrays = pickle.dumps((x, y))
    for row in (1, 0, 1):  # blue, red, blue; each routed update stays homogeneous.
        cue = 0 if row == 1 else 1
        assert owner.train_batch(x[[row]], y[[row]]) == reference[cue].train_batch(
            projected(x[[row]]), y[[row]]
        )
        for actual, expected_owner in zip(owner.snapshot_state().heads, reference):
            assert_tree(torch, actual, expected_owner.snapshot_state())
        probabilities = np.concatenate(
            [reference[1].predict(projected(x[[0]])), reference[0].predict(projected(x[[1]]))]
        )
        np.testing.assert_array_equal(owner.predict(x), probabilities)
    saved = owner.snapshot_state()
    assert (saved.completed_updates, saved.training_examples) == (3, 3)
    assert [h.completed_updates for h in saved.heads] == [2, 1]
    if method in ("neutral_cpc", "gated_cpc"):
        # A noninitial LOCAL split RNG proves full owned-state continuation.
        head = deepcopy(saved.heads[0].head_state)
        generator = torch.Generator(device="cpu")
        generator.set_state(head["split_generator_state"])
        torch.rand(5, generator=generator)
        head["split_generator_state"] = generator.get_state().clone()
        saved = replace(saved, heads=(replace(saved.heads[0], head_state=head), saved.heads[1]))
        owner.restore_state(saved)
        reference[0].restore_state(saved.heads[0])
    child = owner.fork()
    restored = port.ContextTorchFeatureLearner(selected)
    restored.restore_state(saved)
    for copy in (child.snapshot_state(), restored.snapshot_state()):
        assert_tree(torch, copy, saved)
        assert all(
            a.data_ptr() != b.data_ptr()
            for a, b in zip(tensors(torch, copy), tensors(torch, saved), strict=True)
        )
    continuation = reference[0].train_batch(projected(x[[1]]), y[[1]])
    assert child.train_batch(x[[1]], y[[1]]) == restored.train_batch(x[[1]], y[[1]]) == continuation
    assert_tree(torch, child.snapshot_state(), restored.snapshot_state())
    assert_tree(torch, child.snapshot_state().heads[0], reference[0].snapshot_state())
    exposed = child.snapshot_state()
    tensors(torch, exposed)[0].zero_()
    assert_tree(torch, child.snapshot_state().heads[0], reference[0].snapshot_state())
    assert owner.completed_updates == 3 and child.completed_updates == 4
    assert pickle.dumps((x, y)) == before_arrays
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert pickle.dumps(np.random.get_state()) == numpy_rng


def test_should_match_all_initial_parameters_and_keep_pc_neutral_parameter_paths_exact():
    torch = require_torch()
    owners = {method: port.ContextTorchFeatureLearner(policy(method)) for method in METHODS}
    base = owners["bp"].snapshot_state().heads[0]
    for owner in owners.values():
        for head in owner.snapshot_state().heads:
            assert all(torch.equal(head.head_state[n], base.head_state[n]) for n in NAMES)
    x, y = data(variant=True)
    for row in (1, 0, 1):
        assert owners["pc"].train_batch(x[[row]], y[[row]]) == owners["neutral_cpc"].train_batch(
            x[[row]], y[[row]]
        )
        for a, b in zip(
            owners["pc"].snapshot_state().heads, owners["neutral_cpc"].snapshot_state().heads
        ):
            assert all(torch.equal(a.head_state[n], b.head_state[n]) for n in NAMES)
        np.testing.assert_array_equal(owners["pc"].predict(x), owners["neutral_cpc"].predict(x))


@pytest.mark.parametrize(
    "fault", ["encoding", "head_policy", "missing", "counter", "extra", "failure", "tensor"]
)
def test_should_refuse_corrupt_complete_snapshot_before_native_staging(monkeypatch, fault):
    owner = port.ContextTorchFeatureLearner(policy())
    saved = owner.snapshot_state()
    if fault == "encoding":
        object.__setattr__(saved.policy, "encoding_id", "unbound")
    elif fault == "head_policy":
        object.__setattr__(saved.heads[0], "policy", replace(saved.heads[0].policy, seed=18))
    elif fault == "missing":
        object.__setattr__(saved, "heads", saved.heads[:1])
    elif fault == "counter":
        object.__setattr__(saved.heads[0], "completed_updates", 1)
    elif fault == "extra":
        object.__setattr__(saved.heads[0], "extra", 1)
    elif fault == "failure":
        object.__setattr__(saved, "failed_work", (2, 0, 0, 0, 0, 0, 0, 0, True))
    else:
        saved.heads[0].head_state[NAMES[0]] = saved.heads[0].head_state[NAMES[0]][:1]
    monkeypatch.setattr(TorchFeatureLearner, "__init__", lambda *a: pytest.fail("native staged"))
    with pytest.raises(ValueError):
        owner.restore_state(saved)
    assert owner.completed_updates == 0


@pytest.mark.parametrize("fault", ["mixed", "grid", "labels", "dtype", "nan", "target_type"])
def test_should_refuse_entire_bad_batch_before_native_model_or_call(monkeypatch, fault):
    owner = port.ContextTorchFeatureLearner(policy())
    before = owner.snapshot_state()
    x, y = data()
    if fault != "mixed":
        x, y = x[[0]], y[[0]]
    if fault == "grid":
        x[0, 0] = 0.123
    elif fault == "labels":
        y[0, 0] = 7
    elif fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "nan":
        x[0, 5] = np.nan
    elif fault == "target_type":
        y = y.tolist()
    monkeypatch.setattr(TorchFeatureLearner, "__init__", lambda *a: pytest.fail("native staged"))
    with pytest.raises(ValueError):
        owner.train_batch(x, y)
    assert _same_value(owner.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["raise", "input", "source", "counter", "replacement", "caller"])
def test_should_keep_paid_failed_attempt_and_known_prefix_then_refuse_erasure(monkeypatch, fault):
    torch = require_torch()
    owner = port.ContextTorchFeatureLearner(policy())
    x, y = data()
    owner.train_batch(x[[1]], y[[1]])
    before = owner.snapshot_state()
    call_x = x[[0]]
    actual = TorchFeatureLearner.train_batch

    def fail(native, design, labels):
        result = actual(native, design, labels)
        if fault == "raise":
            raise RuntimeError("deliberate unreturned native update")
        if fault == "input":
            design.fill(0)
        elif fault == "source":
            owner._state.heads[1].head_state[NAMES[0]].zero_()
        elif fault == "counter":
            native._payload = (*native._payload[:2], 2, 2)
        elif fault == "replacement":
            owner._state = port.copy_context_torch_feature_snapshot(owner._state)
        else:
            call_x.fill(0)
        return result

    monkeypatch.setattr(TorchFeatureLearner, "train_batch", fail)
    with pytest.raises(TrainingUpdateFailed) as caught:
        owner.train_batch(call_x, y[[0]])
    failed = owner.snapshot_state()
    assert (
        failed.failed_work is not None and failed.completed_updates == before.completed_updates == 1
    )
    for a, b in zip(failed.heads, before.heads):
        assert_tree(torch, a, b)
    assert caught.value.work["native_calls_attempted"] == 1
    assert caught.value.work["native_rows_attempted"] == 1
    assert caught.value.work["failure_work_may_be_unobserved"] is True
    with pytest.raises(ValueError):
        owner.restore_state(before)
    child = owner.fork()
    for denied in (owner, child):
        with pytest.raises(TrainingUpdateFailed):
            denied.train_batch(x[[1]], y[[1]])
        with pytest.raises(TrainingUpdateFailed):
            denied.predict(x)


@pytest.mark.parametrize("fault", ["source", "staged"])
def test_should_refuse_prediction_side_effects_and_keep_bounded_failure(monkeypatch, fault):
    owner = port.ContextTorchFeatureLearner(policy())
    before, actual = owner.snapshot_state(), TorchFeatureLearner.predict
    x, _ = data()

    def read(native, features):
        values = actual(native, features)
        if fault == "source":
            owner._state.heads[1].head_state[NAMES[0]].zero_()
        else:
            with require_torch().no_grad():
                native._head.weight_hidden_output.add_(1)
        return values

    monkeypatch.setattr(TorchFeatureLearner, "predict", read)
    with pytest.raises(TrainingUpdateFailed) as caught:
        owner.predict(x[[0]])
    assert caught.value.work["prediction_operation"] is True
    assert caught.value.work["native_calls_returned"] == 1
    assert _same_value(owner.snapshot_state().heads, before.heads)


def test_should_compose_original_delay_one_training_without_mutating_source_actor_or_rng():
    source = port.ContextTorchFeatureLearner(policy())
    before = source.snapshot_state()
    x, y = data()
    training = tuple(
        AttributionBatch("train", "misleading-red", (f"row-{i}",), x[[i]], y[[i]], num_classes=7)
        for i in (1, 0)
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    stream = AttributionStream(
        source,
        training,
        (),
        checkpoint_updates=(0, 1, 2),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    stream.run_until(2)
    state = stream.candidate_snapshot()
    report = stream.finish()
    assert (state.completed_updates, state.training_examples) == (2, 2)
    assert [h.completed_updates for h in state.heads] == [1, 1]
    assert report["training_examples"] == 2 and report["score_forward_calls"] == 0
    assert [
        (r["observed_at"], r["arrived_at"], r["applied_at"]) for r in report["applied_updates"]
    ] == [(0, 1, 1), (1, 2, 2)]
    assert _same_value(source.snapshot_state(), before)
    assert _same_value(stream._runtime.actor.snapshot_state().state, before)


def test_should_keep_disposable_fresh_probe_separate_from_source_and_checkpoint():
    from src.app.fresh_task_probe import run_fresh_task_probe

    source = port.ContextTorchFeatureLearner(policy())
    checkpoint = source.snapshot_state()
    x, y = data(variant=True)
    train = tuple(
        AttributionBatch("probe_train", "reserved", (f"probe-{i}",), x[[i]], y[[i]], num_classes=7)
        for i in (1, 0)
    )
    heldout = (
        AttributionBatch("probe_test", "reserved", ("probe-test",), x[[0]], y[[0]], num_classes=7),
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    report = run_fresh_task_probe(
        source,
        checkpoint,
        train,
        heldout,
        budget=budget,
        checkpoint_updates=(0, 1, 2),
        timer=lambda: 0.0,
    )
    assert report["status"] == "complete" and report["completed_updates"] == 2
    assert report["completed_training_examples"] == 2
    assert _same_value(source.snapshot_state(), checkpoint)


@pytest.mark.parametrize("fault", ["width", "classes", "encoding", "extra"])
def test_should_refuse_incompatible_policy_before_native_creation(monkeypatch, fault):
    selected = policy().native_policy
    if fault == "width":
        selected = replace(selected, feature_dim=148)
    elif fault == "classes":
        selected = replace(selected, num_classes=2)
    elif fault == "extra":
        object.__setattr__(selected, "extra", 1)
    monkeypatch.setattr(TorchFeatureLearner, "__init__", lambda *a: pytest.fail("native staged"))
    with pytest.raises(ValueError):
        port.ContextTorchFeatureLearner(
            port.ContextTorchFeaturePolicy(
                selected, "bad" if fault == "encoding" else port.CONTEXT_HISTORY_ENCODING_ID
            )
        )
