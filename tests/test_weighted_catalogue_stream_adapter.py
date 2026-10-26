"""Owned joint TRAIN wakes through the existing weighted delayed stream.

Native reference calls and discarded candidate work are charged separately. These
are bounded engineering checks without probes, game physics or scientific claims.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_catalogue_weighted as catalogue_native
from src.adapters import torch_pc_weighted as pc_native
from src.adapters import torch_train_weighted_learner as adapter
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.minigrid_train_catalogue import compose_catalogue_training
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app import coupled_attribution as stream_app
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_class_weights import episode_class_weights
from src.core.learner_ports import TrainingDiagnostic
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from src.shared.torch_runtime import require_torch
from test_torch_catalogue_weighted import arrays, owner, sources
from test_torch_feature_learner import assert_tree, tensors


@pytest.fixture(scope="module")
def torch():
    return require_torch()


@pytest.fixture(autouse=True)
def work_accounting(monkeypatch, record_property, torch):
    work: dict[str, int] = {}

    def instrument(kind, name, label, rows, nominal=None):
        original = getattr(kind, name)

        def invoke(*args, **kwargs):
            count = rows(args, kwargs)
            for suffix, value in (("attempts", 1), ("rows_attempted", count)):
                key = label + "_" + suffix
                work[key] = work.get(key, 0) + value
            if nominal is not None:
                key = label + "_nominal_latent_row_iterations"
                work[key] = work.get(key, 0) + nominal(args, kwargs)
            result = original(*args, **kwargs)
            for suffix, value in (("returns", 1), ("rows_returned", count)):
                key = label + "_" + suffix
                work[key] = work.get(key, 0) + value
            return result

        monkeypatch.setattr(kind, name, invoke)
        return invoke

    counted = instrument(
        catalogue_native,
        "train_weighted_catalogue",
        "catalogue",
        lambda a, k: sum(len(item.transitions) for item in a[1].episodes),
    )
    monkeypatch.setattr(adapter, "train_weighted_catalogue", counted)
    instrument(
        adapter.TorchTrainWeightedLearner, "train_batch", "stream_train", lambda a, k: len(a[1])
    )
    instrument(InvariantTorchFeatureLearner, "fork", "candidate_or_stream_fork", lambda a, k: 0)
    instrument(catalogue_native, "_stage_bp", "numeric_bp", lambda a, k: len(a[2]))
    instrument(
        pc_native,
        "class_weighted_pc_update",
        "numeric_pc",
        lambda a, k: len(a[0]),
        lambda a, k: len(a[0]) * k["inference_steps"],
    )
    instrument(torch.optim.SGD, "step", "sgd", lambda a, k: 0)
    instrument(TorchFeatureLearner, "predict", "predict", lambda a, k: len(a[1]))
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        instrument(kind, "_apply_training_gradients", "commit", lambda a, k: len(a[1]))
    yield work
    # Catalogue admission/candidate and native commit counts overlap numerical
    # work. Only numeric_bp + numeric_pc count independent numerical calls.
    for name, value in work.items():
        record_property(name, value)


def policy(method="bp", steps=2):
    return InvariantTorchFeaturePolicy(
        TorchFeaturePolicy(method, 296, 8, 17, inference_steps=steps, num_classes=7)
    )


def catalogue():
    packets = list(sources())
    packet = packets[-1]
    labels = (1, 4, 1, 4, 1, 4, 1, 4, 6)
    packets[-1] = replace(
        packet,
        transitions=tuple(
            replace(step, action=action)
            for step, action in zip(packet.transitions, labels, strict=True)
        ),
    )
    return catalogue_native.prepare_train_catalogue(tuple(packets))


def learner(method="bp", steps=2, updates=3, value=None):
    return adapter.TorchTrainWeightedLearner.from_catalogue(
        policy(method, steps), catalogue() if value is None else value, updates
    )


@pytest.mark.parametrize("updates", [1, 128])
def test_should_admit_boundary_update_limits_without_repeating_catalogue_metadata(updates):
    source = learner("pc", updates=updates)
    assert source._catalogue_updates == updates
    assert source._episodes == ()
    assert len(source._catalogue.episodes) == len(source._catalogue.source_spans) == 6
    assert source._catalogue_rows == 54
    assert source.completed_updates == source.training_examples == 0


@pytest.mark.parametrize(
    "method,steps", [("bp", 2), ("pc", 2), ("pc", 64), ("neutral_cpc", 2), ("neutral_cpc", 64)]
)
def test_should_match_full_native_weighted_catalogue_reference_through_exhaustion(
    torch, method, steps
):
    value = catalogue()
    source, reference = learner(method, steps, 2, value), owner(method, steps)
    initial = source.snapshot_state()
    for count in (1, 2):
        actual = source.train_batch(*arrays(value))
        expected = catalogue_native.train_weighted_catalogue(
            reference, value, catalogue_native.catalogue_class_weights(value)
        )
        assert actual == expected
        assert actual.definition == (
            catalogue_native.TORCH_CATALOGUE_WEIGHTED_CE_ID
            if method == "bp"
            else catalogue_native.TORCH_CATALOGUE_WEIGHTED_PC_ID
        )
        assert_tree(torch, source.snapshot_state(), reference.snapshot_state())
        assert source.completed_updates == count and source.training_examples == count * 54
    assert initial.completed_updates == initial.training_examples == 0
    with pytest.raises(ValueError, match="exhausted"):
        source.train_batch(*arrays(value))


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc"])
def test_should_own_original_spans_and_use_one_fixed_aggregate_weight_rule(
    torch, monkeypatch, method
):
    value = catalogue()
    saved = deepcopy(value)
    source = learner(method, updates=2, value=value)
    assert source._episodes == () and len(source._catalogue.episodes) == 6
    object.__setattr__(value.episodes[-1], "role", "final_test")
    object.__setattr__(value.source_spans[-1], "context", "foreign")
    observed = []
    update = adapter.train_weighted_catalogue

    def observe(candidate, packet, weights):
        observed.append((deepcopy(packet), weights))
        return update(candidate, packet, weights)

    monkeypatch.setattr(adapter, "train_weighted_catalogue", observe)
    for _ in range(2):
        features, targets = arrays(saved)
        source.train_batch(features, targets)
        features.fill(-1)
        targets.fill(6)
    weights = catalogue_native.catalogue_class_weights(saved)
    assert weights != episode_class_weights(saved.episodes[0])
    assert len(observed) == 2
    assert all(packet == saved and mass == weights for packet, mass in observed)
    assert all(packet.episodes is not saved.episodes for packet, _ in observed)
    assert source.training_examples == 108


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc"])
def test_should_fork_deepcopy_and_restore_exact_catalogue_prefix_without_aliasing(torch, method):
    value, source = catalogue(), learner(method)
    initial = source.snapshot_state()
    source.train_batch(*arrays(value))
    saved = source.snapshot_state()
    children = (source.fork(), deepcopy(source))
    for child in children:
        assert child._learner is not source._learner
        assert child._learner._native is not source._learner._native
        assert child._catalogue is not source._catalogue
        assert child._catalogue.episodes is not source._catalogue.episodes
        assert child._catalogue.episodes[0] is not source._catalogue.episodes[0]
        assert child._catalogue.source_spans[0] is not source._catalogue.source_spans[0]
        assert_tree(torch, child.snapshot_state(), saved)
        child.train_batch(*arrays(value))
    assert_tree(torch, children[0].snapshot_state(), children[1].snapshot_state())
    assert_tree(torch, source.snapshot_state(), saved)
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(
            tensors(torch, source.snapshot_state()),
            tensors(torch, children[0].snapshot_state()),
            strict=True,
        )
    )
    source.restore_state(initial)
    source.train_batch(*arrays(value))
    assert_tree(torch, source.snapshot_state(), saved)
    object.__setattr__(children[0]._catalogue.episodes[-1], "role", "final_test")
    with pytest.raises(ValueError):
        children[0].snapshot_state()
    assert_tree(torch, source.snapshot_state(), saved)


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc"])
def test_should_release_three_joint_wakes_through_existing_delayed_train_stream(
    torch, monkeypatch, method
):
    value, source = catalogue(), learner(method)
    packets = compose_catalogue_training(value.episodes, 3).training
    original_packets = pickle.dumps(packets)
    initial = source.snapshot_state()
    streams, states = [], []
    stream = stream_app.AttributionStream

    def track(*args, **kwargs):
        result = stream(*args, **kwargs)
        streams.append(result)
        return result

    monkeypatch.setattr(stream_app, "AttributionStream", track)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)

    def observe(point, state, packet):
        states.append((point, state))
        assert_tree(torch, source.snapshot_state(), initial)

    report = stream_app.run_attribution_stream(
        source,
        packets,
        (),
        checkpoint_updates=(0, 1, 2, 3),
        label_delay=1,
        budget=budget,
        observe_checkpoint=observe,
        training_only=True,
    )
    assert report["completed_updates"] == budget.updates_completed == 3
    assert report["training_examples"] == 162 and report["score_forward_calls"] == 0
    assert report["evaluation_scope"] == "training_only"
    assert [point for point, _ in states] == [0, 1, 2, 3]
    receipts = report["applied_updates"]
    assert [row["arrived_at"] for row in receipts] == [1, 2, 3]
    assert [row["update_number"] for row in receipts] == [1, 2, 3]
    expected_id = (
        catalogue_native.TORCH_CATALOGUE_WEIGHTED_CE_ID
        if method == "bp"
        else catalogue_native.TORCH_CATALOGUE_WEIGHTED_PC_ID
    )
    assert all(row["diagnostic"]["definition"] == expected_id for row in receipts)
    assert_tree(torch, source.snapshot_state(), initial)
    assert_tree(torch, streams[0]._runtime.actor.snapshot_state().state, initial)
    assert states[-1][1].completed_updates == 3 and states[-1][1].training_examples == 162
    assert pickle.dumps(packets) == original_packets


@pytest.mark.parametrize(
    "fault",
    [
        "zero",
        "bool",
        "float",
        "too_many",
        "active",
        "unbound",
        "recurrent",
        "foreign",
        "untyped",
        "reserved",
        "span",
        "span_bool",
        "source",
        "extra",
    ],
)
def test_should_refuse_invalid_initial_catalogue_configuration_before_model_construction(
    monkeypatch, fault
):
    value, candidate = catalogue(), policy()
    updates: Any = 3
    if fault == "zero":
        updates = 0
    elif fault == "bool":
        updates = True
    elif fault == "float":
        updates = 3.0
    elif fault == "too_many":
        updates = 129
    elif fault == "active":
        candidate = policy("gated_cpc")
    elif fault == "unbound":
        candidate = TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7)
    elif fault == "recurrent":
        candidate = TorchRecurrentPolicy(296, 8, 7, 17)
    elif fault == "foreign":
        candidate = object()
    elif fault == "untyped":
        value = value.episodes
    elif fault == "reserved":
        object.__setattr__(value.episodes[-1], "role", "probe_train")
    elif fault in ("span", "span_bool"):
        object.__setattr__(value.source_spans[0], "start", False if fault == "span_bool" else 1)
    elif fault == "source":
        object.__setattr__(value.episodes[-1].transitions[-1], "terminal", False)
    else:
        object.__setattr__(value, "extra", 1)
    monkeypatch.setattr(
        adapter,
        "InvariantTorchFeatureLearner",
        lambda *a, **k: pytest.fail("invalid catalogue allocated native model"),
    )
    with pytest.raises(ValueError):
        adapter.TorchTrainWeightedLearner.from_catalogue(candidate, value, updates)


@pytest.mark.parametrize("fault", ["features", "labels", "order", "dtype", "list", "shape", "nan"])
def test_should_refuse_unmatched_arrays_before_any_catalogue_numerical_work(
    torch, work_accounting, fault
):
    value, source = catalogue(), learner("pc")
    before = source.snapshot_state()
    features, targets = arrays(value)
    if fault == "features":
        features[0, 0] += 0.1
    elif fault == "labels":
        targets[0, 0] = 1
    elif fault == "order":
        features, targets = features[::-1].copy(), targets[::-1].copy()
    elif fault == "dtype":
        features = features.astype(np.float32)
    elif fault == "list":
        features = features.tolist()
    elif fault == "shape":
        targets = targets[:, 0]
    else:
        features[0, 0] = np.nan
    with pytest.raises(ValueError):
        source.train_batch(features, targets)
    assert_tree(torch, source.snapshot_state(), before)
    assert work_accounting.get("catalogue_attempts", 0) == 0


@pytest.mark.parametrize(
    "fault",
    [
        "catalogue",
        "role",
        "label",
        "next",
        "extra",
        "objective",
        "policy",
        "owner",
        "state",
        "witness",
        "updates",
        "rows",
        "span",
        "span_owner",
        "payload",
        "updates_bool",
        "rows_float",
        "limit_bool",
        "row_limit_float",
    ],
)
def test_should_refuse_changed_owned_schedule_metadata_before_numerical_work(
    torch, work_accounting, fault
):
    source = learner("pc")
    if fault == "catalogue":
        source._catalogue = deepcopy(source._catalogue)
    elif fault == "role":
        object.__setattr__(source._catalogue.episodes[-1], "role", "final_test")
    elif fault == "label":
        object.__setattr__(source._catalogue.episodes[-1].transitions[-1], "action", True)
    elif fault == "next":
        row = source._catalogue.episodes[-1].transitions[-1]
        object.__setattr__(row, "next_observation", (0.0,) * 296)
    elif fault == "extra":
        object.__setattr__(source._catalogue.episodes[-1], "extra", 1)
    elif fault == "objective":
        source._objective_id = "ordinary_ce"
    elif fault == "policy":
        object.__setattr__(source._policy.native_policy, "learning_rate", 0.2)
    elif fault == "owner":
        source._learner = owner("pc")
    elif fault == "witness":
        source._catalogue_witness = deepcopy(source._catalogue_witness)
    elif fault == "updates":
        source._catalogue_updates = 128
    elif fault == "rows":
        source._catalogue_rows = 53
    elif fault == "span":
        object.__setattr__(source._catalogue.source_spans[0], "start", False)
    elif fault == "span_owner":
        object.__setattr__(
            source._catalogue, "source_spans", deepcopy(source._catalogue.source_spans)
        )
    elif fault == "payload":
        source._catalogue_span_payload = bytes(bytearray(source._catalogue_span_payload))
    elif fault == "updates_bool":
        source._catalogue_updates = True
    elif fault == "rows_float":
        source._catalogue_rows = 54.0
    elif fault == "limit_bool":
        source._catalogue_limit = True
    elif fault == "row_limit_float":
        source._catalogue_row_limit = 54.0
    else:
        with torch.no_grad():
            source._learner._native._head.bias_output[0, 0] += 0.125
    with pytest.raises(ValueError):
        source.train_batch(*arrays(catalogue()))
    assert work_accounting.get("catalogue_attempts", 0) == 0


@pytest.mark.parametrize("fault", ["negative", "bool", "beyond", "rows", "policy"])
def test_should_refuse_non_catalogue_restore_prefix_without_publication(torch, fault):
    source = learner("bp", updates=2)
    before = source.snapshot_state()
    if fault in ("negative", "bool"):
        # Deliberately corrupt received evidence after construction; otherwise
        # the dataclass constructor refuses it before the restore port is tested.
        requested = deepcopy(before)
        object.__setattr__(
            requested.native_state, "completed_updates", -1 if fault == "negative" else False
        )
    elif fault in ("beyond", "rows"):
        value = catalogue()
        if fault == "rows":
            # A valid native53-row state must still refuse a54-row schedule.
            value = catalogue_native.prepare_train_catalogue(
                value.episodes[:-1] + (sources(6, 8)[-1],)
            )
        reference = owner("bp")
        for _ in range(3 if fault == "beyond" else 1):
            catalogue_native.train_weighted_catalogue(
                reference, value, catalogue_native.catalogue_class_weights(value)
            )
        requested = reference.snapshot_state()
    else:
        requested = replace(before, policy=policy("pc"))
    with pytest.raises(ValueError):
        source.restore_state(requested)
    assert_tree(torch, source.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["native", "objective", "counter", "input", "admitted_span"])
def test_should_discard_faulted_candidates_and_preserve_paid_attempts(
    torch, monkeypatch, work_accounting, fault
):
    value, source = catalogue(), learner("pc")
    before = source.snapshot_state()
    features, targets = arrays(value)
    original = adapter.train_weighted_catalogue
    primary = RuntimeError("injected weighted catalogue candidate failure")

    def inject(candidate, packet, weights):
        if fault == "native":
            raise primary
        result = original(candidate, packet, weights)
        if fault == "objective":
            return TrainingDiagnostic("wrong_objective", result.value)
        if fault == "counter":
            head, optimizer, count, rows = candidate._native._payload
            candidate._native._payload = (head, optimizer, count, rows + 1)
        elif fault == "input":
            features[0, 0] += 0.1
        else:
            object.__setattr__(packet.source_spans[0], "context", "changed")
        return result

    monkeypatch.setattr(adapter, "train_weighted_catalogue", inject)
    with pytest.raises((ValueError, RuntimeError)) as error:
        source.train_batch(features, targets)
    if fault == "native":
        assert error.value is primary
    assert_tree(torch, source.snapshot_state(), before)
    assert work_accounting.get("numeric_pc_returns", 0) == (0 if fault == "native" else 1)


@pytest.mark.parametrize(
    "fault", ["parameter", "gradient", "payload", "policy", "source", "torch", "mode", "span"]
)
def test_should_preserve_foreign_changes_without_rollback(torch, monkeypatch, fault):
    source = learner("pc")
    donor = source._learner
    native = donor._native
    initial_rng, initial_mode = torch.get_rng_state().clone(), torch.is_grad_enabled()
    observed = []
    original = adapter.train_weighted_catalogue

    def inject(candidate, packet, weights):
        result = original(candidate, packet, weights)
        if fault == "parameter":
            with torch.no_grad():
                native._head.bias_output[0, 0] += 0.125
        elif fault == "gradient":
            native._head.bias_output.grad = torch.zeros_like(native._head.bias_output)
        elif fault == "payload":
            native._payload = tuple(list(native._payload))
        elif fault == "policy":
            donor._policy = replace(donor._policy)
        elif fault == "source":
            object.__setattr__(source._catalogue.episodes[-1], "role", "final_test")
        elif fault == "torch":
            torch.rand(1)
        elif fault == "mode":
            torch.set_grad_enabled(not initial_mode)
        else:
            object.__setattr__(source._catalogue.source_spans[0], "context", "foreign")
        observed.append(
            (
                donor.snapshot_state(),
                native._payload,
                donor._policy,
                native._head.bias_output.grad,
                torch.get_rng_state().clone(),
                torch.is_grad_enabled(),
            )
        )
        return result

    monkeypatch.setattr(adapter, "train_weighted_catalogue", inject)
    try:
        with pytest.raises(ValueError):
            source.train_batch(*arrays(catalogue()))
        assert len(observed) == 1
        state, payload, bound_policy, gradient, rng, mode = observed[0]
        assert_tree(torch, donor.snapshot_state(), state)
        assert source._learner is donor and native._payload is payload
        assert donor._policy is bound_policy and native._head.bias_output.grad is gradient
        assert torch.equal(torch.get_rng_state(), rng) and torch.is_grad_enabled() == mode
        if fault == "source":
            assert source._catalogue.episodes[-1].role == "final_test"
        elif fault == "span":
            assert source._catalogue.source_spans[0].context == "foreign"
    finally:
        torch.set_rng_state(initial_rng)
        torch.set_grad_enabled(initial_mode)


@pytest.mark.parametrize("fault", ["changed", "shared"])
def test_should_refuse_wrong_candidate_before_catalogue_work(
    torch, monkeypatch, work_accounting, fault
):
    source = learner("pc")
    before = source.snapshot_state()
    candidate = source._learner.fork()
    if fault == "changed":
        with torch.no_grad():
            candidate._native._head.bias_output[0, 0] += 0.125
    else:
        candidate._native = source._learner._native
    monkeypatch.setattr(source._learner, "fork", lambda: candidate)
    with pytest.raises(ValueError):
        source.train_batch(*arrays(catalogue()))
    assert_tree(torch, source.snapshot_state(), before)
    assert work_accounting.get("catalogue_attempts", 0) == 0


def test_should_refuse_changed_restore_input_before_publication(torch, monkeypatch):
    source = learner("pc")
    before = source.snapshot_state()
    requested = deepcopy(before)
    restore = adapter.feature_native._restore_components

    def inject(snapshot):
        result = restore(snapshot)
        requested.native_state.head_state["bias_output"][0, 0] += 0.125
        return result

    monkeypatch.setattr(adapter.feature_native, "_restore_components", inject)
    with pytest.raises(ValueError):
        source.restore_state(requested)
    assert_tree(torch, source.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["source", "rng"])
def test_should_refuse_foreign_change_during_catalogue_fork_copy(torch, monkeypatch, fault):
    source = learner("pc")
    before = source._learner.snapshot_state()
    original_rng = torch.get_rng_state().clone()
    original = adapter._copy_catalogue
    changed = []

    def inject(value):
        result = original(value)
        if fault == "source":
            object.__setattr__(source._catalogue.episodes[-1], "role", "final_test")
        else:
            torch.rand(1)
        changed.append(torch.get_rng_state().clone())
        return result

    monkeypatch.setattr(adapter, "_copy_catalogue", inject)
    try:
        with pytest.raises(ValueError):
            source.fork()
        assert len(changed) == 1
        assert_tree(torch, source._learner.snapshot_state(), before)
        assert torch.equal(torch.get_rng_state(), changed[0])
        if fault == "source":
            assert source._catalogue.episodes[-1].role == "final_test"
    finally:
        torch.set_rng_state(original_rng)


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc"])
def test_should_predict_without_training_and_refuse_recurrent_sessions(
    torch, work_accounting, method
):
    value, source = catalogue(), learner(method)
    before = source.snapshot_state()
    probabilities = source.predict(arrays(value)[0])
    assert probabilities.shape == (54, 7) and np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        source.start_episode()
    assert_tree(torch, source.snapshot_state(), before)
    assert work_accounting.get("catalogue_attempts", 0) == 0
