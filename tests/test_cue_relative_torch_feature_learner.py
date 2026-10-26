"""Distinct cue-relative native bindings with unchanged four-method mathematics.

One invocation plans24 native updates48rows and18 predictions36rows, including
direct projected references and owned continuation. No game or teacher executes.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any
from time import monotonic

import numpy as np
import pytest

from src.adapters import cue_relative_torch_feature_learner as bound
from src.adapters import torch_feature_gradient_diagnostics as old_gradients
from src.adapters.cue_relative_torch_feature_learner import (
    CueRelativeTorchFeatureLearner,
    CueRelativeTorchFeaturePolicy,
    CueRelativeTorchFeatureSnapshot,
    copy_cue_relative_torch_feature_policy,
    copy_cue_relative_torch_feature_snapshot,
)
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.core.minigrid_cue_relative_encoding import (
    CUE_RELATIVE_HISTORY_ENCODING_ID,
    cue_relative_public_history,
)
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_invariant_torch_feature_learner import data
from test_minigrid_invariant_encoding import change_pixel
from test_torch_feature_learner import NAMES, assert_tree, tensors
from test_torch_feature_gradient_diagnostics import pair
from test_torch_feature_rehearsal_pilot import forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def native_policy(method="bp", **options):
    return TorchFeaturePolicy(
        **dict(method=method, feature_dim=296, hidden_dim=8, seed=17, num_classes=7, **options)
    )


def policy(method="bp"):
    return CueRelativeTorchFeaturePolicy(native_policy(method))


def projected(features):
    return np.array(
        [cue_relative_public_history(tuple(float(value) for value in row)) for row in features],
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def torch():
    value = require_torch()
    value.set_num_threads(1)
    if value.get_num_interop_threads() != 1:
        value.set_num_interop_threads(1)
    return value


@pytest.fixture(scope="module")
def native_work():
    patch = pytest.MonkeyPatch()
    train, predict = TorchFeatureLearner.train_batch, TorchFeatureLearner.predict
    work = dict(updates=0, training_rows=0, predictions=0, prediction_rows=0)

    def update(owner, features, targets):
        work["updates"] += 1
        work["training_rows"] += len(features)
        assert work["updates"] <= 32 and work["training_rows"] <= 64
        return train(owner, features, targets)

    def read(owner, features):
        work["predictions"] += 1
        work["prediction_rows"] += len(features)
        assert work["predictions"] <= 24 and work["prediction_rows"] <= 48
        return predict(owner, features)

    patch.setattr(TorchFeatureLearner, "train_batch", update)
    patch.setattr(TorchFeatureLearner, "predict", read)
    yield work
    patch.undo()


@pytest.fixture(autouse=True)
def record_native_work(native_work, record_property):
    before = native_work.copy()
    yield
    for name, value in native_work.items():
        record_property(f"cue_relative_native_{name}", value - before[name])


@pytest.mark.parametrize("method", METHODS)
def test_should_match_projected_native_full_state_and_owned_continuation(torch, method):
    selected = native_policy(method)
    declared = CueRelativeTorchFeaturePolicy(selected)
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate()
    wrapped, reference = CueRelativeTorchFeatureLearner(declared), TorchFeatureLearner(selected)
    x, y = data()
    variant, _ = data(variant=True)
    inputs = pickle.dumps((x, variant, y))
    assert declared.method == "public_cue_relative_" + method
    assert (declared.feature_dim, declared.hidden_dim, declared.num_classes) == (296, 8, 7)
    assert declared.encoding_id == CUE_RELATIVE_HISTORY_ENCODING_ID != INVARIANT_HISTORY_ENCODING_ID
    assert wrapped.train_batch(x, y) == reference.train_batch(projected(x), y)
    saved = wrapped.snapshot_state()
    assert type(saved) is CueRelativeTorchFeatureSnapshot
    assert saved.completed_updates == 1 and saved.training_examples == 2
    assert_tree(torch, saved.native_state, reference.snapshot_state())
    np.testing.assert_array_equal(wrapped.predict(variant), reference.predict(projected(variant)))
    if method in ("neutral_cpc", "gated_cpc"):
        head = deepcopy(saved.native_state.head_state)
        generator = torch.Generator(device="cpu")
        generator.set_state(head["split_generator_state"])
        torch.rand(5, generator=generator)
        head["split_generator_state"] = generator.get_state().clone()
        saved = replace(saved, native_state=replace(saved.native_state, head_state=head))
        wrapped.restore_state(saved)
        reference.restore_state(saved.native_state)
    child, restored = wrapped.fork(), CueRelativeTorchFeatureLearner(policy(method))
    restored.restore_state(saved)
    assert child is not wrapped and child._native is not wrapped._native
    for owned in (child.snapshot_state(), restored.snapshot_state()):
        assert_tree(torch, owned, saved)
        assert all(
            a.data_ptr() != b.data_ptr()
            for a, b in zip(tensors(torch, owned), tensors(torch, saved), strict=True)
        )
    reference_diagnostic = reference.train_batch(projected(variant), y)
    assert child.train_batch(variant, y) == restored.train_batch(variant, y) == reference_diagnostic
    assert_tree(torch, child.snapshot_state(), restored.snapshot_state())
    assert_tree(torch, child.snapshot_state().native_state, reference.snapshot_state())
    np.testing.assert_array_equal(child.predict(x), restored.predict(x))
    assert (
        wrapped.completed_updates == 1
        and child.completed_updates == restored.completed_updates == 2
    )
    assert_tree(torch, wrapped.snapshot_state(), saved)
    exposed = child.snapshot_state()
    tensors(torch, exposed)[0].zero_()
    object.__setattr__(declared.native_policy, "seed", 18)
    assert child.policy.native_policy.seed == wrapped.policy.native_policy.seed == 17
    assert_tree(torch, child.snapshot_state().native_state, reference.snapshot_state())
    assert pickle.dumps((x, variant, y)) == inputs
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]


def test_should_keep_original_pc_and_neutral_cpc_parameters_and_predictions_matched(torch):
    pc, neutral = (
        CueRelativeTorchFeatureLearner(policy(method)) for method in ("pc", "neutral_cpc")
    )
    x, y = data()
    assert pc.train_batch(x, y) == neutral.train_batch(x, y)
    for name in NAMES:
        assert torch.equal(
            pc.snapshot_state().native_state.head_state[name],
            neutral.snapshot_state().native_state.head_state[name],
        )
    np.testing.assert_array_equal(pc.predict(x), neutral.predict(x))


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "subclass",
        "float32",
        "width",
        "late_nan",
        "late_grid",
        "late_direction",
        "late_missing_cue",
        "late_key_color",
    ],
)
def test_should_preflight_every_raw_row_before_native_train_or_prediction(
    torch, monkeypatch, fault
):
    learner = CueRelativeTorchFeatureLearner(policy())
    before = learner.snapshot_state()
    x: Any
    x, y = data()
    if fault == "list":
        x = x.tolist()
    elif fault == "subclass":
        x = x.view(ArraySubclass)
    elif fault == "float32":
        x = x.astype(np.float32)
    elif fault == "width":
        x = x[:, :148]
    elif fault == "late_nan":
        x[1, -1] = float("nan")
    elif fault == "late_grid":
        x[1, 148 + 144] += 0.01
    elif fault == "late_direction":
        x[1, -1] += 0.01
    else:
        half = 0 if fault == "late_missing_cue" else 148
        raw = tuple(float(value) for value in x[1, half : half + 148])
        changed = change_pixel(raw, 1 if half == 0 else 48, (3, 0, 0) if half == 0 else (5, 1, 0))
        x[1, half : half + 148] = changed
    monkeypatch.setattr(TorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(TorchFeatureLearner, "predict", forbidden)
    with pytest.raises(ValueError):
        learner.train_batch(x, y)
    with pytest.raises(ValueError):
        learner.predict(x)
    assert_tree(torch, learner.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault",
    [
        "native",
        "old_family",
        "tag",
        "wrapper_extra",
        "policy_extra",
        "native_extra",
        "bool_clock",
        "native_method",
        "nonfinite",
        "optimizer_policy",
    ],
)
def test_should_refuse_foreign_or_corrupted_snapshots_atomically(torch, fault):
    learner = CueRelativeTorchFeatureLearner(policy())
    before = learner.snapshot_state()
    incoming: Any = learner.snapshot_state()
    if fault == "native":
        incoming = incoming.native_state
    elif fault == "old_family":
        incoming = InvariantTorchFeatureSnapshot(
            InvariantTorchFeaturePolicy(native_policy()), incoming.native_state
        )
    elif fault == "tag":
        object.__setattr__(incoming.policy, "encoding_id", INVARIANT_HISTORY_ENCODING_ID)
    elif fault in ("wrapper_extra", "policy_extra", "native_extra"):
        target = (
            incoming
            if fault == "wrapper_extra"
            else incoming.policy
            if fault == "policy_extra"
            else incoming.native_state
        )
        object.__setattr__(target, "foreign", 1)
    elif fault == "bool_clock":
        object.__setattr__(incoming.native_state, "completed_updates", False)
    elif fault == "native_method":
        incoming = replace(
            incoming, native_state=TorchFeatureLearner(native_policy("pc")).snapshot_state()
        )
    elif fault == "nonfinite":
        incoming.native_state.head_state[NAMES[0]][0, 0] = float("nan")
    else:
        incoming.native_state.optimizer_state["param_groups"][0]["lr"] = 0.1
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        learner.restore_state(incoming)
    assert_tree(torch, learner.snapshot_state(), before)


def test_should_preserve_exact_old_family_and_old_reader_admission(torch, monkeypatch):
    fresh = CueRelativeTorchFeatureLearner(policy()).snapshot_state()
    foreign: Any = fresh
    old = InvariantTorchFeatureLearner(InvariantTorchFeaturePolicy(native_policy()))
    before = old.snapshot_state()
    with pytest.raises(ValueError):
        old.restore_state(foreign)
    with pytest.raises(ValueError):
        TorchFeatureLearner(native_policy()).restore_state(foreign)
    monkeypatch.setattr(old_gradients, "require_torch", forbidden)
    monkeypatch.setattr(old_gradients, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        old_gradients.diagnose_prior_gradients(fresh, pair())
    assert_tree(torch, old.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault", ["width", "classes", "old_tag", "native_extra", "old_policy", "policy_subclass"]
)
def test_should_refuse_invalid_exact_policies_before_constructing_native_heads(monkeypatch, fault):
    monkeypatch.setattr(bound, "TorchFeatureLearner", forbidden)
    if fault == "old_policy":
        foreign: Any = InvariantTorchFeaturePolicy(native_policy())
        with pytest.raises(ValueError):
            CueRelativeTorchFeatureLearner(foreign)
        return
    if fault == "policy_subclass":

        class Subclass(CueRelativeTorchFeaturePolicy):
            pass

        with pytest.raises(ValueError):
            CueRelativeTorchFeatureLearner(Subclass(native_policy()))
        return
    selected = TorchFeaturePolicy(
        "bp", 148 if fault == "width" else 296, 8, 17, num_classes=5 if fault == "classes" else 7
    )
    if fault == "native_extra":
        object.__setattr__(selected, "foreign", 1)
    with pytest.raises(ValueError):
        CueRelativeTorchFeaturePolicy(
            selected,
            INVARIANT_HISTORY_ENCODING_ID
            if fault == "old_tag"
            else CUE_RELATIVE_HISTORY_ENCODING_ID,
        )


def test_should_copy_policy_and_complete_snapshot_without_exposing_native_storage(torch):
    declared = policy()
    copied = copy_cue_relative_torch_feature_policy(declared)
    assert (
        copied == declared
        and copied is not declared
        and copied.native_policy is not declared.native_policy
    )
    learner = CueRelativeTorchFeatureLearner(declared)
    snapshot = learner.snapshot_state()
    owned = copy_cue_relative_torch_feature_snapshot(snapshot)
    assert_tree(torch, owned, snapshot)
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(tensors(torch, owned), tensors(torch, snapshot), strict=True)
    )
    tensors(torch, owned)[0].fill_(123)
    object.__setattr__(copied.native_policy, "seed", 18)
    assert learner.policy.native_policy.seed == 17
    assert_tree(torch, learner.snapshot_state(), snapshot)


def test_should_run_existing_delayed_training_only_stream_with_distinct_snapshot_family(torch):
    source = CueRelativeTorchFeatureLearner(policy())
    initial = deepcopy(source.snapshot_state())
    features, targets = data()
    training = tuple(
        AttributionBatch(
            "train",
            "A",
            tuple(f"cue-relative:{wake}:{row}" for row in range(2)),
            features.copy(),
            targets.copy(),
            None,
            7,
        )
        for wake in range(2)
    )
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=10), monotonic
    )
    stream = AttributionStream(
        source,
        training,
        (),
        checkpoint_updates=(0, 1, 2),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    stream.run_until(1)
    stream.run_until(2)
    result = stream.finish()
    assert result["evaluation_scope"] == "training_only"
    assert result["completed_updates"] == budget.updates_completed == 2
    assert result["training_examples"] == 4 and result["score_forward_calls"] == 0
    assert result["checkpoints"] == [] and budget.replay_examples_completed == 0
    assert len(result["applied_updates"]) == 2
    assert all(r["arrived_at"] == r["observed_at"] + 1 for r in result["applied_updates"])
    candidate = stream._runtime._candidate.snapshot_state()
    assert type(candidate) is CueRelativeTorchFeatureSnapshot
    assert candidate.completed_updates == 2 and candidate.training_examples == 4
    assert_tree(torch, source.snapshot_state(), initial)
    assert_tree(torch, stream._runtime.actor.snapshot_state().state, initial)
