"""Weighted joint TRAIN wakes retain real source spans and full native state.

These are bounded engineering checks, with no environment, probes or experiment.
Every candidate, ordinary reference and numerical oracle is charged separately.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import torch_catalogue_weighted as weighted
from src.adapters import torch_feature_learner as native
from src.adapters import torch_pc_weighted as pc_weighted
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.minigrid_train_catalogue import compose_catalogue_training
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.core.categorical_pc_weighted import class_weighted_pc_update
from src.core.demonstration_class_weights import episode_class_weights
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_encoding import frame
from test_torch_feature_learner import NAMES, assert_tree


ONES = (1.0,) * 7
WEIGHTS = (0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


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
            work[label + "_attempts"] = work.get(label + "_attempts", 0) + 1
            work[label + "_rows_attempted"] = work.get(label + "_rows_attempted", 0) + count
            if nominal is not None:
                key = label + "_nominal_latent_row_iterations"
                work[key] = work.get(key, 0) + nominal(args, kwargs)
            result = original(*args, **kwargs)
            work[label + "_returns"] = work.get(label + "_returns", 0) + 1
            work[label + "_rows_returned"] = work.get(label + "_rows_returned", 0) + count
            return result

        monkeypatch.setattr(kind, name, invoke)

    instrument(
        weighted,
        "train_weighted_catalogue",
        "weighted",
        lambda a, k: sum(len(item.transitions) for item in a[1].episodes),
    )
    instrument(TorchFeatureLearner, "train_batch", "ordinary", lambda a, k: len(a[1]))
    instrument(TorchFeatureLearner, "predict", "predict", lambda a, k: len(a[1]))
    instrument(torch.optim.SGD, "step", "sgd", lambda a, k: 0)
    instrument(
        pc_weighted,
        "class_weighted_pc_update",
        "proposal",
        lambda a, k: len(a[0]),
        lambda a, k: len(a[0]) * k["inference_steps"],
    )
    if hasattr(weighted, "class_weighted_pc_update"):
        instrument(
            weighted,
            "class_weighted_pc_update",
            "catalogue_proposal",
            lambda a, k: len(a[0]),
            lambda a, k: len(a[0]) * k["inference_steps"],
        )
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        instrument(
            kind,
            "train_step",
            "native_pc",
            lambda a, k: len(a[1]),
            lambda a, k: len(a[1]) * a[4],
        )
        instrument(kind, "_apply_training_gradients", "commit", lambda a, k: len(a[1]))
    yield work
    for name, value in work.items():
        record_property(name, value)


def owner(method="bp", steps=2):
    return InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(
            TorchFeaturePolicy(method, 296, 8, 17, inference_steps=steps, num_classes=7)
        )
    )


def sources(count=6, rows=9):
    result = []
    for identity in range(count):
        initial = frame(wall_color=identity % 6, direction=2, cue_color=2, key_color=2)
        current = [initial] + [
            frame(wall_color=index % 6, direction=index % 4, door_state=index % 3)
            for index in range(1, rows + 1)
        ]
        observations = [initial + item for item in current]
        actions = (2, 2, 3, 2, 5, 0, 2, 2, 6, 1, 4, 2, 2, 2, 2, 6)
        result.append(
            DemonstrationEpisode(
                f"catalogue-TRAIN-{identity}",
                "ABC"[identity % 3],
                "train",
                tuple(
                    DemonstrationTransition(
                        index,
                        observations[index],
                        actions[index],
                        float(index == rows - 1),
                        observations[index + 1],
                        index == rows - 1,
                        7,
                    )
                    for index in range(rows)
                ),
                7,
            )
        )
    return tuple(result)


def catalogue():
    return weighted.prepare_train_catalogue(sources())


def arrays(value):
    batch = compose_catalogue_training(value.episodes, 1).training[0]
    assert batch.episode is None and batch.context == "train-catalogue"
    return batch.features, batch.targets


def test_should_own_six_original_sources_and_exact_half_open_spans():
    packets = sources()
    value = weighted.prepare_train_catalogue(packets)
    assert value.episodes == packets and value.episodes is not packets
    expected = compose_catalogue_training(packets, 1)
    assert value.source_spans == expected.source_spans
    assert [(span.start, span.stop) for span in value.source_spans] == [
        (index * 9, (index + 1) * 9) for index in range(6)
    ]
    for original, owned in zip(packets, value.episodes, strict=True):
        assert original is not owned and original.transitions is not owned.transitions
    object.__setattr__(packets[0], "episode_id", "foreign-source")
    assert value.episodes[0].episode_id == "catalogue-TRAIN-0"


def test_should_derive_fixed_aggregate_target_mass_without_reserved_or_absent_labels():
    packets = sources(2)
    second = replace(
        packets[1], transitions=tuple(replace(item, action=0) for item in packets[1].transitions)
    )
    value = weighted.prepare_train_catalogue((packets[0], second))
    counts = [0] * 7
    for packet in value.episodes:
        for item in packet.transitions:
            counts[item.action] += 1
    present = sum(count > 0 for count in counts)
    expected = tuple(18 / (present * count) if count else 1.0 for count in counts)
    assert weighted.catalogue_class_weights(value) == expected
    assert expected[1] == expected[4] == 1.0
    assert expected != episode_class_weights(packets[0])


@pytest.mark.parametrize(
    "method,steps", [("bp", 2), ("pc", 2), ("pc", 64), ("neutral_cpc", 2), ("neutral_cpc", 64)]
)
def test_should_preserve_all_one_full_native_state_and_weighted_fork_restore(torch, method, steps):
    learner, reference, value = owner(method, steps), owner(method, steps), catalogue()
    for _ in range(2):
        actual = weighted.train_weighted_catalogue(learner, value, ONES)
        ordinary = reference.train_batch(*arrays(value))
        expected_id = (
            weighted.TORCH_CATALOGUE_WEIGHTED_CE_ID
            if method == "bp"
            else weighted.TORCH_CATALOGUE_WEIGHTED_PC_ID
        )
        assert actual.definition == expected_id and actual.definition != ordinary.definition
        assert_tree(torch, learner.snapshot_state(), reference.snapshot_state())
    saved = learner.snapshot_state()
    child, restored = learner.fork(), owner(method, steps)
    restored.restore_state(saved)
    for item in (learner, child, restored):
        weighted.train_weighted_catalogue(item, value, WEIGHTS)
    assert_tree(torch, child.snapshot_state(), learner.snapshot_state())
    assert_tree(torch, restored.snapshot_state(), learner.snapshot_state())
    assert learner.completed_updates == 3 and learner.training_examples == 162


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_match_weighted_pc_oracle_and_preserve_legacy_history(
    torch, method, record_property
):
    learner, value = owner(method), catalogue()
    before = learner.snapshot_state().native_state
    raw, targets = arrays(value)
    features = native._features(learner._features(raw), 296)
    proposal = class_weighted_pc_update(
        features,
        torch.from_numpy(targets[:, 0].astype(np.int64)),
        cast(tuple[Any, Any, Any, Any], tuple(before.head_state[name] for name in NAMES)),
        class_weights=WEIGHTS,
        learning_rate=0.03,
        inference_steps=2,
        inference_learning_rate=0.04,
        role="train",
    )
    record_property("oracle_proposal_attempts", 1)
    record_property("oracle_proposal_returns", 1)
    record_property("oracle_proposal_rows_attempted", 54)
    record_property("oracle_proposal_rows_returned", 54)
    record_property("oracle_proposal_nominal_latent_row_iterations", 108)
    diagnostic = weighted.train_weighted_catalogue(learner, value, WEIGHTS)
    after = learner.snapshot_state().native_state
    assert diagnostic.value == proposal.energy
    for name, parameter in zip(NAMES, proposal.parameters, strict=True):
        assert torch.equal(after.head_state[name], parameter)
    assert after.optimizer_state is None and after.head_state["traffic_steps"] == 1
    if method == "neutral_cpc":
        assert after.head_state["energy_history"] == [proposal.legacy_diagnostic]
        assert (
            after.head_state["wake_examples"] == 54 and after.head_state["steps_since_sleep"] == 1
        )


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc"])
def test_should_ignore_absent_weights_when_observed_target_mass_is_uniform(torch, method):
    learner, reference, value = owner(method), owner(method), catalogue()
    weights = (3.0, 1e-6, 3.0, 3.0, 1e6, 3.0, 3.0)
    weighted.train_weighted_catalogue(learner, value, weights)
    reference.train_batch(*arrays(value))
    assert_tree(torch, learner.snapshot_state(), reference.snapshot_state())


def test_should_match_independent_numpy_weighted_bp_objective_and_gradients(torch):
    learner, value = owner(), catalogue()
    raw, targets = arrays(value)
    features = learner._features(raw).astype(np.float64)
    parameters = learner.snapshot_state().native_state.head_state
    values = {name: parameters[name].numpy().astype(np.float64) for name in NAMES}
    hidden = np.tanh(features @ values[NAMES[0]] + values[NAMES[1]])
    logits = hidden @ values[NAMES[2]] + values[NAMES[3]]
    mass = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = mass / mass.sum(axis=1, keepdims=True)
    labels = targets[:, 0].astype(np.int64)
    coefficients = np.array(WEIGHTS)[labels]
    coefficients /= coefficients.sum()
    expected_loss = -np.sum(coefficients * np.log(probabilities[np.arange(54), labels]))
    delta = probabilities.copy()
    delta[np.arange(54), labels] -= 1
    delta *= coefficients[:, None]
    hidden_error = (delta @ values[NAMES[2]].T) * (1 - hidden**2)
    gradients = (
        features.T @ hidden_error,
        hidden_error.sum(axis=0, keepdims=True),
        hidden.T @ delta,
        delta.sum(axis=0, keepdims=True),
    )
    diagnostic = weighted.train_weighted_catalogue(learner, value, WEIGHTS)
    assert diagnostic.value == pytest.approx(expected_loss, rel=2e-6, abs=2e-7)
    after = learner.snapshot_state().native_state.head_state
    for name, gradient in zip(NAMES, gradients, strict=True):
        np.testing.assert_allclose(
            after[name].numpy(), values[name] - 0.03 * gradient, rtol=3e-5, atol=3e-7
        )


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc"])
def test_should_preserve_source_serving_donor_and_rng_under_supported_mode(torch, method):
    donor, value = owner(method), catalogue()
    before = donor.snapshot_state()
    child = donor.fork()
    source_bytes = pickle.dumps(value)
    rng = (torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate())
    with torch.no_grad():
        if method == "bp":
            with pytest.raises(ValueError):
                weighted.train_weighted_catalogue(child, value, WEIGHTS)
        else:
            weighted.train_weighted_catalogue(child, value, WEIGHTS)
        assert not torch.is_grad_enabled()
    assert_tree(torch, donor.snapshot_state(), before)
    assert child.completed_updates == (0 if method == "bp" else 1)
    assert pickle.dumps(value) == source_bytes
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc"])
def test_should_refuse_inference_mode_before_candidate(torch, monkeypatch, method):
    learner, value = owner(method), catalogue()
    monkeypatch.setattr(learner, "snapshot_state", lambda: pytest.fail("mode reached snapshot"))
    with torch.inference_mode(), pytest.raises(ValueError):
        weighted.train_weighted_catalogue(learner, value, ONES)


@pytest.mark.parametrize(
    "weights",
    [[1.0] * 7, (1.0,) * 6, (True,) + (1.0,) * 6, (0.0,) * 7, (float("nan"),) * 7, (1e7,) * 7],
)
def test_should_refuse_invalid_weights_before_candidate(monkeypatch, weights):
    learner, value = owner("pc"), catalogue()
    monkeypatch.setattr(learner, "snapshot_state", lambda: pytest.fail("weights reached snapshot"))
    with pytest.raises(ValueError):
        weighted.train_weighted_catalogue(learner, value, weights)


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "list",
        "seven",
        "too_many_rows",
        "duplicate",
        "final_test",
        "probe_train",
        "probe_test",
        "incomplete",
        "extra",
        "initial",
        "late_next",
        "width",
        "classes",
    ],
)
def test_should_refuse_reserved_or_corrupt_sources_without_training(fault):
    packets: Any = sources()
    if fault == "empty":
        packets = ()
    elif fault == "list":
        packets = list(packets)
    elif fault == "seven":
        packets = sources(7)
    elif fault == "too_many_rows":
        packets = sources(6, 10)
    elif fault == "duplicate":
        packets = (packets[0], packets[0])
    elif fault in ("final_test", "probe_train", "probe_test"):
        packets = (replace(packets[0], role=fault),)
    elif fault == "incomplete":
        object.__setattr__(packets[-1].transitions[-1], "terminal", False)
    elif fault == "extra":
        object.__setattr__(packets[-1].transitions[-1], "extra", 1)
    elif fault == "width":
        row = packets[-1].transitions[-1]
        object.__setattr__(row, "next_observation", row.next_observation[:148])
    elif fault == "classes":
        object.__setattr__(packets[-1], "num_classes", 8)
    elif fault in ("initial", "late_next"):
        row = packets[-1].transitions[-1]
        field = "observation" if fault == "initial" else "next_observation"
        values = list(getattr(row, field))
        values[0 if fault == "initial" else -1] = 0.123
        object.__setattr__(row, field, tuple(values))
    with pytest.raises(ValueError):
        weighted.prepare_train_catalogue(packets)


@pytest.mark.parametrize(
    "fault", ["span_id", "span_start", "span_stop", "span_context", "span_bool", "source", "extra"]
)
def test_should_refuse_replaced_catalogue_evidence_before_snapshot(monkeypatch, fault):
    learner, value = owner("pc"), catalogue()
    if fault.startswith("span_"):
        span = value.source_spans[-1]
        field = {
            "span_id": "episode_id",
            "span_start": "start",
            "span_stop": "stop",
            "span_context": "context",
            "span_bool": "start",
        }[fault]
        replacement = (
            "replaced"
            if field in ("episode_id", "context")
            else False
            if fault == "span_bool"
            else 0
        )
        object.__setattr__(span, field, replacement)
    elif fault == "source":
        object.__setattr__(value.episodes[-1], "episode_id", "replaced")
    else:
        object.__setattr__(value, "extra", 1)
    monkeypatch.setattr(learner, "snapshot_state", lambda: pytest.fail("evidence reached snapshot"))
    with pytest.raises(ValueError):
        weighted.train_weighted_catalogue(learner, value, WEIGHTS)
    with pytest.raises(ValueError):
        weighted.catalogue_class_weights(value)


@pytest.mark.parametrize("kind", ["active", "raw", "foreign", "policy"])
def test_should_refuse_unbound_owners_before_snapshot(monkeypatch, kind):
    learner: Any = (
        owner("gated_cpc")
        if kind == "active"
        else TorchFeatureLearner(TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7))
        if kind == "raw"
        else object()
        if kind == "foreign"
        else owner("pc")
    )
    if kind == "policy":
        learner._native._policy = replace(learner._native._policy, learning_rate=0.2)
    if hasattr(learner, "snapshot_state"):
        monkeypatch.setattr(
            learner, "snapshot_state", lambda: pytest.fail("owner reached snapshot")
        )
    with pytest.raises(ValueError):
        weighted.train_weighted_catalogue(learner, catalogue(), ONES)


def install_after_candidate(monkeypatch, torch, method, callback):
    kind = (
        torch.optim.SGD
        if method == "bp"
        else PredictiveCodingHead
        if method == "pc"
        else CircadianPredictiveCodingHead
    )
    name = "step" if method == "bp" else "_apply_training_gradients"
    original = getattr(kind, name)

    def invoke(*args, **kwargs):
        result = original(*args, **kwargs)
        callback()
        return result

    monkeypatch.setattr(kind, name, invoke)


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc"])
def test_should_publish_nothing_after_actual_candidate_failure(torch, monkeypatch, method):
    learner, value = owner(method), catalogue()
    before, payload = learner.snapshot_state(), learner._native._payload
    primary = RuntimeError("injected catalogue candidate failure")

    def fail():
        raise primary

    install_after_candidate(monkeypatch, torch, method, fail)
    with pytest.raises(RuntimeError) as failure:
        weighted.train_weighted_catalogue(learner, value, WEIGHTS)
    assert failure.value is primary and learner._native._payload is payload
    monkeypatch.undo()
    assert_tree(torch, learner.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault",
    [
        "parameter",
        "parameter_identity",
        "gradient",
        "counter",
        "payload",
        "native_owner",
        "invariant_policy",
        "native_policy",
        "source",
        "span",
        "torch",
        "numpy",
        "python",
        "mode",
    ],
)
def test_should_refuse_foreign_changes_without_rolling_them_back(torch, monkeypatch, fault):
    learner, value = owner("pc"), catalogue()
    rng = (torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate())
    grad_mode = torch.is_grad_enabled()
    observed = []

    def mutate():
        donor = learner._native
        if fault == "parameter":
            with torch.no_grad():
                donor._head.weight_feature_hidden[0, 0] += 0.125
        elif fault == "parameter_identity":
            donor._head.weight_feature_hidden = donor._head.weight_feature_hidden.clone()
        elif fault == "gradient":
            donor._head.weight_feature_hidden.grad = torch.zeros_like(
                donor._head.weight_feature_hidden
            )
        elif fault == "counter":
            head, optimizer, updates, examples = donor._payload
            donor._payload = (head, optimizer, updates + 1, examples + 54)
        elif fault == "payload":
            donor._payload = tuple(list(donor._payload))
        elif fault == "native_owner":
            learner._native = owner("pc")._native
            donor = learner._native
        elif fault == "invariant_policy":
            learner._policy = replace(learner._policy)
        elif fault == "native_policy":
            donor._policy = replace(donor._policy)
        elif fault == "source":
            object.__setattr__(value.episodes[0], "episode_id", "foreign-source")
        elif fault == "span":
            object.__setattr__(value.source_spans[0], "episode_id", "foreign-span")
        elif fault == "torch":
            torch.rand(1)
        elif fault == "numpy":
            np.random.random()
        elif fault == "mode":
            torch.set_grad_enabled(not grad_mode)
        else:
            random.random()
        observed.append(
            (
                # A foreign counter replacement deliberately disagrees with
                # native traffic history. Record raw evidence without asking
                # the snapshot validator to certify this corrupted state.
                None if fault == "counter" else learner.snapshot_state(),
                donor._payload,
                pickle.dumps(value),
                torch.get_rng_state().clone(),
                pickle.dumps(np.random.get_state()),
                random.getstate(),
                donor._head.weight_feature_hidden.grad,
                donor,
                learner._policy,
                donor._policy,
                torch.is_grad_enabled(),
            )
        )

    if fault == "mode":
        original = weighted._stage_pc

        def stage(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result

        monkeypatch.setattr(weighted, "_stage_pc", stage)
    else:
        install_after_candidate(monkeypatch, torch, "pc", mutate)
    try:
        with pytest.raises(ValueError):
            weighted.train_weighted_catalogue(learner, value, WEIGHTS)
        assert len(observed) == 1
        state, payload, source, trng, nrng, prng, gradient, donor, policy, native_policy, mode = (
            observed[0]
        )
        if fault == "counter":
            assert learner._native.completed_updates == 1
            assert learner._native.training_examples == 54
            with pytest.raises(ValueError):
                learner.snapshot_state()
        else:
            assert_tree(torch, learner.snapshot_state(), state)
        assert learner._native._payload is payload and pickle.dumps(value) == source
        assert learner._native._head.weight_feature_hidden.grad is gradient
        assert (
            learner._native is donor
            and learner._policy is policy
            and donor._policy is native_policy
        )
        assert torch.equal(torch.get_rng_state(), trng)
        assert pickle.dumps(np.random.get_state()) == nrng and random.getstate() == prng
        # Native PC commits run inside their own no_grad context. Its ordinary
        # unwind restores the caller's mode; only the explicit post-stage mode
        # injection is a foreign change that must remain observable here.
        assert torch.is_grad_enabled() == (mode if fault == "mode" else grad_mode)
    finally:
        torch.set_rng_state(rng[0])
        np.random.set_state(rng[1])
        random.setstate(rng[2])
        torch.set_grad_enabled(grad_mode)


@pytest.mark.parametrize("field,index,replacement", [("start", 0, False), ("stop", -1, 54.0)])
def test_should_refuse_equal_value_span_primitive_replacements(
    monkeypatch, field, index, replacement
):
    learner, value = owner("pc"), catalogue()
    object.__setattr__(value.source_spans[index], field, replacement)
    monkeypatch.setattr(learner, "snapshot_state", lambda: pytest.fail("span reached snapshot"))
    with pytest.raises(ValueError, match="exact string identities and integer bounds"):
        weighted.train_weighted_catalogue(learner, value, WEIGHTS)
    with pytest.raises(ValueError):
        weighted.catalogue_class_weights(value)


def test_should_preserve_pc_neutral_parameters_and_energy_over_weighted_continuation(torch):
    pc, neutral, value = owner("pc"), owner("neutral_cpc"), catalogue()
    for update in (1, 2):
        left = weighted.train_weighted_catalogue(pc, value, WEIGHTS)
        right = weighted.train_weighted_catalogue(neutral, value, WEIGHTS)
        assert left == right
        for name in NAMES:
            assert torch.equal(
                getattr(pc._native._head, name), getattr(neutral._native._head, name)
            )
        assert (
            pc.snapshot_state().completed_updates
            == neutral.snapshot_state().completed_updates
            == update
        )
