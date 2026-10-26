"""Versioned teacher targets at the owned coherent NumPy learner boundary.

Tiny native references establish the actual mixed objective and wake-only EMA.
Malformed and stale records are checked before any target forward or update.
"""

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from typing import Any

import numpy as np
import pytest

import src.adapters.slow_teacher_learner as teacher_module
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.matched_backprop_learner import MatchedBackpropLearner
from src.adapters.slow_teacher_learner import SlowTeacherLearner
from src.core.circadian_predictive_coding import replay_sample_id
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import (
    evaluate_coupled,
    feedforward_latents,
    predict_coupled,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.matched_backprop import MatchedBackpropNetwork, evaluate_matched_backprop
from src.core.slow_teacher import SlowTeacherPolicy, TeacherTargets


METHODS = ("bp", "pc", "neutral", "active")
SCHEMA = "fixed_fixture_features_v1"


def batch():
    return np.array([[0.7, -0.2], [-0.3, 0.8]]), np.array([[0.0], [1.0]])


def policy(**changes):
    return SlowTeacherPolicy("fixture-teacher", SCHEMA, **changes)


def student(method):
    base = CoupledPredictiveCodingNetwork(2, (3,), seed=23, precisions=(0.7,))
    if method == "bp":
        return MatchedBackpropLearner(
            MatchedBackpropNetwork(base.snapshot_parameters()), learning_rate=0.03
        )
    if method != "pc":
        base = CoupledCircadianPredictiveCodingNetwork(
            2,
            (3,),
            seed=23,
            precisions=(0.7,),
            config=(
                CoupledCircadianConfig.matched_pc_control()
                if method == "neutral"
                else CoupledCircadianConfig(chemical_buildup_rate=0.3, plasticity_sensitivity=2.0)
            ),
        )
    return CoupledLearner(base, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.04)


def arrays(parameters):
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def assert_tree(actual, expected):
    assert type(actual) is type(expected)
    if type(expected) is np.ndarray:
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected) and not isinstance(expected, type):
        for field in fields(expected):
            assert_tree(getattr(actual, field.name), getattr(expected, field.name))
    elif type(expected) in (tuple, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_tree(left, right)
    elif type(expected) is dict:
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree(actual[key], expected[key])
    else:
        assert actual == expected


def targets(learner, x, y, **changes):
    options = dict(role="train", feature_schema=SCHEMA, output_schema="binary_probability_v1")
    options.update(changes)
    return learner.make_teacher_targets(x, y, **options)


def forbidden(*args, **kwargs):
    raise AssertionError("forbidden forward or native update")


@pytest.mark.parametrize(
    "change",
    [
        {"teacher_id": ""},
        {"teacher_id": "t" * 129},
        {"feature_schema": ""},
        {"feature_schema": "\u03bb" * 65},
        {"output_schema": "unrecognized"},
        {"mixture_weight": True},
        {"mixture_weight": -0.1},
        {"mixture_weight": 1.1},
        {"mixture_weight": float("nan")},
        {"ema_decay": True},
        {"ema_decay": -0.1},
        {"ema_decay": 1.1},
    ],
)
def test_should_refuse_invalid_teacher_policy(change):
    arguments: dict[str, Any] = dict(teacher_id="fixture-teacher", feature_schema=SCHEMA)
    arguments.update(change)
    with pytest.raises(ValueError):
        SlowTeacherPolicy(**arguments)


def test_should_refuse_unsupported_student_types_before_calling_their_ports():
    class UnqualifiedStudent(CoupledLearner):
        def fork(self):
            raise AssertionError("an unsupported subclass must not be forked")

    unsupported = object.__new__(UnqualifiedStudent)
    with pytest.raises(ValueError):
        SlowTeacherLearner(unsupported, policy())


@pytest.mark.parametrize("method", METHODS)
def test_should_keep_zero_weight_wake_and_replay_exactly_on_the_native_path(monkeypatch, method):
    x, y = batch()
    original = student(method)
    before = original.snapshot_state()
    direct = original.fork()
    wrapped = SlowTeacherLearner(original, policy(mixture_weight=0.0))
    # A disabled teacher has no target-forward or EMA work at either call site.
    monkeypatch.setattr(teacher_module, "predict_coupled", forbidden)
    monkeypatch.setattr(teacher_module, "_ema_parameters", forbidden)
    expected = direct.train_batch(x, y)
    assert wrapped.train_batch(x, y) == expected
    expected = direct.train_batch(x, y)
    assert wrapped.train_distillation_batch(x, y, None) == expected
    assert_tree(wrapped.snapshot_state().student_state, direct.snapshot_state())
    assert_tree(original.snapshot_state(), before)
    assert wrapped.wake_updates == wrapped.replay_updates == 1
    assert wrapped.teacher_revision == wrapped.teacher_storage_bytes == 0


@pytest.mark.parametrize("method", METHODS)
def test_should_train_the_native_mixed_target_and_average_parameters_only_after_wake(method):
    x, y = batch()
    original = student(method)
    direct = original.fork()
    caller_policy = policy(mixture_weight=0.25, ema_decay=0.5)
    wrapped = SlowTeacherLearner(original, caller_policy)
    object.__setattr__(caller_policy, "teacher_id", "changed-caller-policy")
    returned_policy = wrapped.teacher_policy
    object.__setattr__(returned_policy, "feature_schema", "changed-returned-policy")
    assert wrapped.teacher_policy == policy(mixture_weight=0.25, ema_decay=0.5)
    initial = wrapped.snapshot_state()
    assert wrapped.teacher_storage_bytes == sum(
        a.nbytes for a in arrays(initial.teacher_parameters)
    )
    assert wrapped.teacher_storage_bytes == 104
    wake_diagnostic = direct.train_batch(x, y)
    assert wrapped.train_batch(x, y) == wake_diagnostic
    wake = wrapped.snapshot_state()
    assert wake.teacher_parameters is not None
    for actual, previous, current in zip(
        arrays(wake.teacher_parameters),
        arrays(initial.teacher_parameters),
        arrays(direct.snapshot_state().parameters),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, 0.5 * previous + 0.5 * current)
        assert not np.shares_memory(actual, previous)
        assert not np.shares_memory(actual, current)
    expected_q = predict_coupled(wake.teacher_parameters, x)
    record = targets(wrapped, x, y)
    assert record.teacher_revision == record.student_updates == 1
    assert record.sample_ids == tuple(replay_sample_id(a[None], b[None]) for a, b in zip(x, y))
    np.testing.assert_array_equal(record.probabilities, expected_q)
    assert_tree(wrapped.snapshot_state(), wake)
    mixed = 0.75 * y + 0.25 * expected_q
    expected_diagnostic = direct.train_batch(x, mixed)
    diagnostic = wrapped.train_distillation_batch(x, y, record)
    assert diagnostic.value == expected_diagnostic.value
    assert diagnostic.definition == (
        "numpy_matched_binary_bce_slow_teacher_mixture_preupdate_v1"
        if method == "bp"
        else "numpy_coupled_binary_energy_slow_teacher_mixture_v1"
    )
    after = wrapped.snapshot_state()
    assert_tree(after.student_state, direct.snapshot_state())
    np.testing.assert_array_equal(wrapped.predict(x), direct.predict(x))
    assert_tree(after.teacher_parameters, wake.teacher_parameters)
    assert wrapped.teacher_revision == wrapped.wake_updates == wrapped.replay_updates == 1
    record.probabilities.fill(999)
    assert_tree(wrapped.snapshot_state(), after)


def test_should_preserve_affine_bce_identity_for_all_parameters_and_latent_gradients():
    x, y = batch()
    parameters = student("pc").snapshot_state().parameters
    latent = tuple(z + 0.1 for z in feedforward_latents(parameters, x))
    q = np.array([[0.2], [0.65]])
    weight = 0.25
    mixed = (1 - weight) * y + weight * q
    hard = evaluate_coupled(parameters, x, y, latent, (0.7,))
    soft = evaluate_coupled(parameters, x, q, latent, (0.7,))
    result = evaluate_coupled(parameters, x, mixed, latent, (0.7,))
    assert result.energy == pytest.approx((1 - weight) * hard.energy + weight * soft.energy)
    for actual, left, right in zip(
        (*arrays(result.parameter_gradients), *result.latent_gradients),
        (*arrays(hard.parameter_gradients), *hard.latent_gradients),
        (*arrays(soft.parameter_gradients), *soft.latent_gradients),
        strict=True,
    ):
        np.testing.assert_allclose(
            actual, (1 - weight) * left + weight * right, rtol=1e-13, atol=1e-15
        )
    bp = [evaluate_matched_backprop(parameters, x, value) for value in (y, q, mixed)]
    assert bp[2].loss == pytest.approx((1 - weight) * bp[0].loss + weight * bp[1].loss)
    for actual, left, right in zip(
        arrays(bp[2].parameter_gradients),
        arrays(bp[0].parameter_gradients),
        arrays(bp[1].parameter_gradients),
        strict=True,
    ):
        np.testing.assert_allclose(
            actual, (1 - weight) * left + weight * right, rtol=1e-13, atol=1e-15
        )


@pytest.mark.parametrize(
    "fault",
    [
        "format",
        "teacher_revision",
        "student_updates",
        "id",
        "dtype",
        "shape",
        "rows",
        "range",
        "nan",
    ],
)
def test_should_refuse_malformed_teacher_packets(fault):
    x, y = batch()
    arguments: dict[str, Any] = dict(
        format_version=1,
        policy=policy(),
        teacher_revision=0,
        student_updates=0,
        sample_ids=tuple(replay_sample_id(a[None], b[None]) for a, b in zip(x, y)),
        probabilities=np.array([[0.2], [0.8]]),
    )
    if fault == "format":
        arguments["format_version"] = True
    elif fault == "teacher_revision":
        arguments["teacher_revision"] = True
    elif fault == "student_updates":
        arguments["student_updates"] = -1
    elif fault == "id":
        arguments["sample_ids"] = ("bad", "bad")
    elif fault == "dtype":
        arguments["probabilities"] = np.array([[0.2], [0.8]], dtype=np.float32)
    elif fault == "shape":
        arguments["probabilities"] = np.array([0.2, 0.8])
    elif fault == "rows":
        arguments["probabilities"] = np.zeros((129, 1))
        arguments["sample_ids"] = ("0" * 64,) * 129
    else:
        arguments["probabilities"][0, 0] = 1.1 if fault == "range" else np.nan
    with pytest.raises(ValueError):
        TeacherTargets(**arguments)


@pytest.mark.parametrize(
    "fault", ["features", "labels", "teacher", "student", "policy", "probabilities"]
)
def test_should_reject_stale_or_foreign_target_bindings_before_forward_or_native(
    monkeypatch, fault
):
    x, y = batch()
    learner = SlowTeacherLearner(student("pc"), policy())
    record = targets(learner, x, y)
    before = learner.snapshot_state()
    if fault == "features":
        x = x[::-1].copy()
    elif fault == "labels":
        y = 1 - y
    elif fault == "teacher":
        record = replace(record, teacher_revision=1, student_updates=1)
    elif fault == "student":
        record = replace(record, student_updates=1)
    elif fault == "policy":
        record = replace(record, policy=replace(record.policy, teacher_id="another-teacher"))
    else:
        record.probabilities[0, 0] = np.nan
    monkeypatch.setattr(teacher_module, "predict_coupled", forbidden)
    monkeypatch.setattr(CoupledLearner, "train_batch", forbidden)
    with pytest.raises(ValueError):
        learner.train_distillation_batch(x, y, record)
    assert_tree(learner.snapshot_state(), before)


@pytest.mark.parametrize(
    "changes",
    [{"role": "final_test"}, {"feature_schema": "different"}, {"output_schema": "logits"}],
)
def test_should_check_role_encoder_and_output_identity_before_teacher_forward(monkeypatch, changes):
    x, y = batch()
    learner = SlowTeacherLearner(student("pc"), policy())
    before = learner.snapshot_state()
    monkeypatch.setattr(teacher_module, "predict_coupled", forbidden)
    with pytest.raises(ValueError):
        targets(learner, x, y, **changes)
    assert_tree(learner.snapshot_state(), before)


@pytest.mark.parametrize("method", METHODS)
def test_should_continue_complete_forks_and_restores_without_aliasing_teacher_or_student(method):
    x, y = batch()
    learner = SlowTeacherLearner(student(method), policy())
    learner.train_batch(x, y)
    learner.train_distillation_batch(x, y, targets(learner, x, y))
    saved = learner.snapshot_state()
    fork = learner.fork()
    assert fork is not learner
    assert_tree(fork.snapshot_state(), saved)
    assert learner.train_batch(x, 1 - y) == fork.train_batch(x, 1 - y)
    continued = learner.snapshot_state()
    assert_tree(fork.snapshot_state(), continued)
    learner.restore_state(saved)
    assert_tree(learner.snapshot_state(), saved)
    learner.train_batch(x, 1 - y)
    assert_tree(learner.snapshot_state(), continued)
    assert saved.teacher_parameters is not None
    saved.teacher_parameters.output_weight.fill(999)
    saved.student_state.parameters.output_bias.fill(999)
    assert_tree(learner.snapshot_state(), continued)
    assert_tree(fork.snapshot_state(), continued)


@pytest.mark.parametrize(
    "fault", ["format", "policy", "binding", "student", "teacher", "wake", "replay", "revision"]
)
def test_should_reject_incomplete_or_incompatible_restore_atomically(fault):
    learner = SlowTeacherLearner(student("active"), policy())
    before = learner.snapshot_state()
    broken = deepcopy(before)
    if fault == "format":
        object.__setattr__(broken, "format_version", 2)
    elif fault == "policy":
        object.__setattr__(broken, "policy", replace(broken.policy, ema_decay=0.3))
    elif fault == "binding":
        broken.student_binding["family"] = "foreign"
    elif fault == "student":
        broken.student_state.parameters.output_bias.fill(np.nan)
    elif fault == "teacher":
        assert broken.teacher_parameters is not None
        broken.teacher_parameters.output_weight.fill(np.nan)
    elif fault == "wake":
        object.__setattr__(broken, "wake_updates", 1)
    elif fault == "replay":
        object.__setattr__(broken, "replay_updates", True)
    else:
        object.__setattr__(broken, "teacher_revision", 1)
    with pytest.raises(ValueError):
        learner.restore_state(broken)
    assert_tree(learner.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["native", "ema", "ema_result"])
def test_should_keep_live_native_and_teacher_state_unchanged_after_staged_failure(
    monkeypatch, fault
):
    x, y = batch()
    learner = SlowTeacherLearner(student("active"), policy())
    before = learner.snapshot_state()
    if fault == "native":
        train = CoupledLearner.train_batch

        def fail_after_native(self, features, labels):
            train(self, features, labels)
            raise RuntimeError("after native update")

        monkeypatch.setattr(CoupledLearner, "train_batch", fail_after_native)
    else:
        ema = teacher_module._ema_parameters

        def fail_ema(*args, **kwargs):
            if fault == "ema":
                raise RuntimeError("EMA unavailable")
            value = ema(*args, **kwargs)
            value.output_weight.fill(np.nan)
            return value

        monkeypatch.setattr(teacher_module, "_ema_parameters", fail_ema)
    with pytest.raises((RuntimeError, ValueError)):
        learner.train_batch(x, y)
    assert_tree(learner.snapshot_state(), before)
