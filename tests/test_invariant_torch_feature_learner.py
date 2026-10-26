"""Encoding-bound neural ports against unchanged native heads, without game physics.

One file invocation plans exactly 36 native updates / 72 rows: 28/56 parity and
continuation, 4/8 PC-neutral control, 4/8 existing stream/probe composition.
The independent hard ceiling is 64 native calls / 1024 rows; no scientific run.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
    copy_invariant_torch_feature_policy,
    copy_invariant_torch_feature_snapshot,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_invariant_encoding import (
    INVARIANT_HISTORY_ENCODING_ID,
    canonicalize_public_history,
)
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_encoding import frame
from test_torch_feature_learner import NAMES, assert_tree, tensors


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
PLANNED_NATIVE_UPDATES = 36
PLANNED_NATIVE_ROWS = 72


@pytest.fixture(scope="module")
def torch():
    return require_torch()


@pytest.fixture(scope="module", autouse=True)
def native_work_budget():
    """Count every native port update, including reference and disposable owners."""
    original = TorchFeatureLearner.train_batch
    work = dict(calls=0, rows=0)
    patch = pytest.MonkeyPatch()

    def bounded(owner, features, targets):
        work["calls"] += 1
        work["rows"] += len(features)
        assert work["calls"] <= 64 and work["rows"] <= 1024
        return original(owner, features, targets)

    patch.setattr(TorchFeatureLearner, "train_batch", bounded)
    yield work
    patch.undo()
    assert work["calls"] <= 64 and work["rows"] <= 1024


def native_policy(method="bp", **options):
    values = dict(method=method, feature_dim=296, hidden_dim=8, seed=17, num_classes=7)
    values.update(options)
    return TorchFeaturePolicy(**values)


def policy(method="bp"):
    return InvariantTorchFeaturePolicy(native_policy(method))


def data(*, variant=False):
    rows = []
    for cue in (0, 2):
        first = frame(
            wall_color=1 if variant else 5,
            direction=2 if variant else 0,
            cue_color=cue,
            key_color=cue,
        )
        current = frame(
            wall_color=4 if variant else 5,
            direction=3 if variant else 1,
            cue_color=cue,
            key_color=cue,
            door_state=0,
        )
        rows.append(first + current)
    return np.array(rows, dtype=np.float64), np.array([[6.0], [2.0]], dtype=np.float64)


def canonical(features):
    return np.array(
        [canonicalize_public_history(tuple(float(v) for v in row)) for row in features],
        dtype=np.float64,
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_match_manual_canonical_native_updates_full_state_rng_and_owned_continuation(
    torch, method
):
    selected = native_policy(method)
    declared = InvariantTorchFeaturePolicy(selected)
    global_rng = torch.get_rng_state().clone()
    numpy_rng = pickle.dumps(np.random.get_state())
    wrapped = InvariantTorchFeatureLearner(declared)
    reference = TorchFeatureLearner(selected)
    x, y = data()
    variant, _ = data(variant=True)
    before = pickle.dumps((x, variant, y))
    assert declared.method == "public_invariant_" + method
    assert (declared.feature_dim, declared.hidden_dim, declared.num_classes) == (296, 8, 7)
    for update in (1, 2):
        raw = x if update == 1 else variant
        actual = wrapped.train_batch(raw, y)
        expected = reference.train_batch(canonical(raw), y)
        assert actual == expected
        state = wrapped.snapshot_state()
        assert type(state) is InvariantTorchFeatureSnapshot
        assert state.policy.encoding_id == INVARIANT_HISTORY_ENCODING_ID
        assert (state.completed_updates, state.training_examples) == (update, 2 * update)
        assert_tree(torch, state.native_state, reference.snapshot_state())
        np.testing.assert_array_equal(
            wrapped.predict(variant), reference.predict(canonical(variant))
        )
    saved = wrapped.snapshot_state()
    if method in ("neutral_cpc", "gated_cpc"):
        generator = torch.Generator(device="cpu")
        generator.set_state(saved.native_state.head_state["split_generator_state"])
        torch.rand(5, generator=generator)
        head = deepcopy(saved.native_state.head_state)
        head["split_generator_state"] = generator.get_state().clone()
        saved = replace(saved, native_state=replace(saved.native_state, head_state=head))
        wrapped.restore_state(saved)
        reference.restore_state(saved.native_state)
    child = wrapped.fork()
    restored = InvariantTorchFeatureLearner(policy(method))
    restored.restore_state(saved)
    assert child is not wrapped
    for copy in (child.snapshot_state(), restored.snapshot_state()):
        assert_tree(torch, copy, saved)
        left, right = tensors(torch, copy), tensors(torch, saved)
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(left, right, strict=True))
    continuation = reference.train_batch(canonical(x), y)
    assert child.train_batch(x, y) == restored.train_batch(x, y) == continuation
    assert_tree(torch, child.snapshot_state(), restored.snapshot_state())
    assert_tree(torch, child.snapshot_state().native_state, reference.snapshot_state())
    assert (
        wrapped.completed_updates == 2
        and child.completed_updates == restored.completed_updates == 3
    )
    exposed = child.snapshot_state()
    tensors(torch, exposed)[0].zero_()
    object.__setattr__(declared.native_policy, "seed", 18)
    assert child.policy.native_policy.seed == wrapped.policy.native_policy.seed == 17
    assert_tree(torch, child.snapshot_state().native_state, reference.snapshot_state())
    assert pickle.dumps((x, variant, y)) == before
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert pickle.dumps(np.random.get_state()) == numpy_rng


def test_should_keep_pc_and_neutral_cpc_parameter_predictions_matched_with_the_existing_control(
    torch,
):
    pc, neutral = (InvariantTorchFeatureLearner(policy(method)) for method in ("pc", "neutral_cpc"))
    x, y = data(variant=True)
    for _ in range(2):
        assert pc.train_batch(x, y) == neutral.train_batch(x, y)
        for name in NAMES:
            assert torch.equal(
                pc.snapshot_state().native_state.head_state[name],
                neutral.snapshot_state().native_state.head_state[name],
            )
        np.testing.assert_array_equal(pc.predict(x), neutral.predict(x))
    assert (pc.completed_updates, neutral.completed_updates) == (2, 2)


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize(
    "fault", ["list", "subclass", "float32", "width", "late_nan", "late_grid", "late_direction"]
)
def test_should_validate_all_public_rows_before_calling_native_training_or_prediction(
    torch, monkeypatch, fault
):
    learner = InvariantTorchFeatureLearner(policy())
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
        x[1, 148 + 3] += 0.01
    else:
        x[1, -1] += 0.01

    def forbidden(*args, **options):
        pytest.fail("malformed projection reached a native learner call")

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
        "raw",
        "encoding",
        "wrapper_extra",
        "policy_extra",
        "native_extra",
        "bool_clock",
        "native_method",
        "nonfinite_tensor",
        "optimizer_policy",
    ],
)
def test_should_refuse_mismatched_complete_state_atomically_without_reinterpreting_raw_checkpoints(
    torch, fault
):
    learner = InvariantTorchFeatureLearner(policy())
    before = learner.snapshot_state()
    incoming: Any = learner.snapshot_state()
    if fault == "raw":
        incoming = incoming.native_state
    elif fault == "encoding":
        object.__setattr__(incoming.policy, "encoding_id", "raw_history")
    elif fault == "wrapper_extra":
        object.__setattr__(incoming, "extra", 1)
    elif fault == "policy_extra":
        object.__setattr__(incoming.policy, "extra", 1)
    elif fault == "native_extra":
        object.__setattr__(incoming.native_state, "extra", 1)
    elif fault == "bool_clock":
        object.__setattr__(incoming.native_state, "completed_updates", False)
    elif fault == "native_method":
        incoming = replace(
            incoming, native_state=TorchFeatureLearner(native_policy("pc")).snapshot_state()
        )
    elif fault == "nonfinite_tensor":
        incoming.native_state.head_state[NAMES[0]][0, 0] = float("nan")
    else:
        incoming.native_state.optimizer_state["param_groups"][0]["lr"] = 0.1
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        learner.restore_state(incoming)
    assert_tree(torch, learner.snapshot_state(), before)
    invalid_bound: Any = before
    with pytest.raises(ValueError):
        TorchFeatureLearner(native_policy()).restore_state(invalid_bound)


@pytest.mark.parametrize("fault", ["width", "classes", "encoding", "native_extra"])
def test_should_validate_exact_bound_policies_before_loading_a_native_head(fault):
    selected: Any = (
        native_policy(feature_dim=148)
        if fault == "width"
        else native_policy(num_classes=5)
        if fault == "classes"
        else native_policy()
    )
    encoding = "different" if fault == "encoding" else INVARIANT_HISTORY_ENCODING_ID
    if fault == "native_extra":
        object.__setattr__(selected, "extra", 1)
    with pytest.raises(ValueError):
        InvariantTorchFeaturePolicy(selected, encoding)


def test_should_own_exact_policy_records_and_refuse_subclass_or_snapshot_copy_corruption(torch):
    selected = policy()
    copied = copy_invariant_torch_feature_policy(selected)
    assert copied is not selected and copied.native_policy is not selected.native_policy
    assert copied == selected
    learner = InvariantTorchFeatureLearner(selected)
    state = learner.snapshot_state()
    owned = copy_invariant_torch_feature_snapshot(state)
    assert_tree(torch, owned, state)
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(tensors(torch, owned), tensors(torch, state), strict=True)
    )

    class PolicySubclass(InvariantTorchFeaturePolicy):
        pass

    invalid: Any = PolicySubclass(native_policy())
    with pytest.raises(ValueError):
        InvariantTorchFeatureLearner(invalid)
    object.__setattr__(state.policy, "encoding_id", "unbound")
    with pytest.raises(ValueError):
        copy_invariant_torch_feature_snapshot(state)
    assert learner.policy.encoding_id == INVARIANT_HISTORY_ENCODING_ID


def batch(role, identity, cue, *, wall=5):
    initial = frame(wall_color=wall, direction=2, cue_color=cue, key_color=cue)
    current = frame(wall_color=wall, direction=3, cue_color=cue, key_color=cue, door_state=0)
    return AttributionBatch(
        role,
        "A",
        (identity + ":0", identity + ":1"),
        np.array([initial + initial, initial + current], dtype=np.float64),
        np.array([[6.0], [2.0]], dtype=np.float64),
        num_classes=7,
    )


def test_should_compose_the_existing_delayed_train_stream_and_a_separate_owned_fresh_probe(torch):
    source = InvariantTorchFeatureLearner(policy("pc"))
    initial = source.snapshot_state()
    training = (batch("train", "wake0", 0), batch("train", "wake1", 2, wall=1))
    data_before = pickle.dumps(training)
    states = []
    wake_budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)

    def observe(point, state, packet):
        states.append((point, state))

    result = run_attribution_stream(
        source,
        training,
        (),
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=wake_budget,
        observe_checkpoint=observe,
        training_only=True,
    )
    assert result["completed_updates"] == wake_budget.updates_completed == 2
    assert result["training_examples"] == 4 and result["score_forward_calls"] == 0
    assert [r["arrived_at"] for r in result["applied_updates"]] == [1, 2]
    assert_tree(torch, source.snapshot_state(), initial)
    assert pickle.dumps(training) == data_before
    assert [point for point, _ in states] == [0, 2]
    checkpoint = states[-1][1]
    checkpoint_before = deepcopy(checkpoint)
    probe_train = (
        batch("probe_train", "probe0", 0, wall=2),
        batch("probe_train", "probe1", 2, wall=3),
    )
    probe_test = (batch("probe_test", "probe-test", 0, wall=4),)
    reserved_before = pickle.dumps((probe_train, probe_test))
    probe_budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)

    def boundary():
        assert_tree(torch, source.snapshot_state(), initial)
        assert_tree(torch, states[-1][1], checkpoint_before)
        assert pickle.dumps((probe_train, probe_test)) == reserved_before

    report = run_fresh_task_probe(
        source,
        checkpoint,
        probe_train,
        probe_test,
        budget=probe_budget,
        checkpoint_updates=(0, 1, 2),
        timer=lambda: 0.0,
        decision_diagnostics=True,
        check_boundary=boundary,
    )
    assert report["status"] == "complete" and report["completed_updates"] == 2
    assert (
        report["completed_training_examples"] == 4 and report["score_forward_calls_completed"] == 3
    )
    assert report["decision_diagnostics_enabled"] is True
    assert (
        wake_budget is not probe_budget
        and wake_budget.updates_completed == probe_budget.updates_completed == 2
    )
    assert_tree(torch, source.snapshot_state(), initial)
    assert checkpoint.completed_updates == 2 and checkpoint.training_examples == 4
