"""Bounded callback policy binding and same-process handoff using fake learners."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import pickle
from types import SimpleNamespace
from typing import Any

import pytest

import src.adapters.coupled_checkpoint_callbacks as callbacks_module
from src.adapters.coupled_checkpoint_callbacks import CoupledCheckpointCallbacks
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.adapters.coupled_learners import CoupledLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_learner_state import CoupledLearnerCheckpoint, CoupledUpdatePolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.resource_sharing import SharingLimits
from test_coupled_checkpoint_codec import assert_same, snapshot
from test_native_transfer_session import advance

BINDING = CodecBinding("a" * 64, "b" * 64)
LIMITS = CodecLimits(32768, 8192, 8, 64)


def policy(guarded=True):
    return CoupledUpdatePolicy(
        0.025, 3, 0.07, CoupledInferencePolicy(1, 0.1, None) if guarded else None
    )


def payload(native, configured=None, *, binding=BINDING, limits=LIMITS):
    return CoupledLearnerCheckpointCodec().encode(
        CoupledLearnerCheckpoint(1, policy() if configured is None else configured, native),
        binding=binding,
        limits=limits,
    )


def policy_hash(configured):
    return sha256(
        json.dumps(
            asdict(configured), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def trusted(configured):
    learner: Any = object.__new__(CoupledLearner)
    learner._learning_rate = configured.learning_rate
    learner._inference_steps = configured.inference_steps
    learner._inference_learning_rate = configured.inference_learning_rate
    learner._inference_policy = deepcopy(configured.inference_policy)
    return learner


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
def test_should_own_complete_controls_and_forward_exact_bounded_bytes_without_native_work(
    monkeypatch, guarded, kind
):
    configured, binding, limits = policy(guarded), deepcopy(BINDING), deepcopy(LIMITS)
    expected_policy = deepcopy(configured)
    native = snapshot(kind, "F")
    expected = payload(native, expected_policy)
    callback = CoupledCheckpointCallbacks(configured, binding=binding, limits=limits)
    events: list[Any] = []
    sentinel = object()

    def restore(raw, **kwargs):
        events.append((raw, kwargs))
        assert raw == expected
        assert kwargs == dict(
            binding=BINDING, limits=LIMITS, expected_sha256=sha256(expected).hexdigest()
        )
        return sentinel

    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    object.__setattr__(configured, "learning_rate", 0.5)
    if configured.inference_policy is not None:
        object.__setattr__(configured.inference_policy, "gradient_tolerance", 0.9)
    object.__setattr__(binding, "source_sha256", "f" * 64)
    object.__setattr__(limits, "max_encoded_bytes", 1)
    assert callback.state_digest(native) == sha256(expected).hexdigest()
    assert events == []
    assert callback.policy_digest(trusted(expected_policy)) == policy_hash(expected_policy)
    assert callback.build(native) is sentinel and len(events) == 1
    assert_same(native, snapshot(kind, "F"))


@pytest.mark.parametrize(
    "fault",
    [
        "policy_mapping",
        "policy_bool",
        "policy_extra",
        "guard_extra",
        "binding_mapping",
        "binding_missing",
        "limits_mapping",
        "limits_bool",
        "policy_byte_cap",
    ],
)
def test_should_validate_constructor_metadata_without_model_or_encoder_calls(monkeypatch, fault):
    configured: Any = policy()
    binding: Any = deepcopy(BINDING)
    limits: Any = deepcopy(LIMITS)
    if fault == "policy_mapping":
        configured = asdict(configured)
    elif fault == "policy_bool":
        object.__setattr__(configured, "learning_rate", True)
    elif fault == "policy_extra":
        object.__setattr__(configured, "extra", 1)
    elif fault == "guard_extra":
        object.__setattr__(configured.inference_policy, "extra", 1)
    elif fault == "binding_mapping":
        binding = asdict(binding)
    elif fault == "binding_missing":
        object.__delattr__(binding, "source_sha256")
    elif fault == "limits_mapping":
        limits = asdict(limits)
    elif fault == "limits_bool":
        object.__setattr__(limits, "max_array_bytes", True)
    else:
        limits = replace(limits, max_encoded_bytes=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("constructor reached native envelope or model")

    monkeypatch.setattr(CoupledLearnerCheckpointCodec, "encode", forbidden)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", forbidden)
    with pytest.raises(ValueError):
        CoupledCheckpointCallbacks(configured, binding=binding, limits=limits)


@pytest.mark.parametrize("kind", ["mapping", "subclass"])
def test_should_refuse_untrusted_policy_probe_family_before_method_access(kind):
    callback = CoupledCheckpointCallbacks(policy(), binding=BINDING, limits=LIMITS)

    def forbidden(*args, **kwargs):
        raise AssertionError("wrong family reached policy getter")

    class Other(CoupledLearner):
        checkpoint_policy = forbidden

    learner = (
        SimpleNamespace(checkpoint_policy=forbidden) if kind == "mapping" else object.__new__(Other)
    )
    with pytest.raises(ValueError):
        callback.policy_digest(learner)


@pytest.mark.parametrize(
    "field", ["learning_rate", "inference_steps", "inference_learning_rate", "inference_policy"]
)
def test_should_refuse_wrong_complete_policy_without_accessing_model(field):
    expected = policy()
    values = dict(
        learning_rate=0.05, inference_steps=4, inference_learning_rate=0.08, inference_policy=None
    )
    changed = replace(expected, **{field: values[field]})
    callback = CoupledCheckpointCallbacks(expected, binding=BINDING, limits=LIMITS)
    learner = trusted(changed)
    learner._model = SimpleNamespace(
        snapshot_state=lambda: pytest.fail("policy probe accessed model")
    )
    with pytest.raises(ValueError):
        callback.policy_digest(learner)


@pytest.mark.parametrize("kind", ["policy", "guard", "binding", "limits"])
def test_should_revalidate_tampered_owned_records_before_any_public_callback_work(
    monkeypatch, kind
):
    callback = CoupledCheckpointCallbacks(policy(), binding=BINDING, limits=LIMITS)
    records = list(vars(callback).values())
    if kind == "policy":
        object.__setattr__(
            next(value for value in records if type(value) is CoupledUpdatePolicy),
            "learning_rate",
            False,
        )
    elif kind == "guard":
        object.__setattr__(
            next(value for value in records if type(value) is CoupledUpdatePolicy).inference_policy,
            "min_steps",
            True,
        )
    elif kind == "binding":
        object.__setattr__(
            next(value for value in records if type(value) is CodecBinding), "policy_sha256", False
        )
    else:
        object.__setattr__(
            next(value for value in records if type(value) is CodecLimits),
            "max_encoded_bytes",
            False,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("tampered configuration reached callback work")

    monkeypatch.setattr(CoupledLearnerCheckpointCodec, "encode", forbidden)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", forbidden)
    for operation in (
        lambda: callback.state_digest(snapshot()),
        lambda: callback.build(snapshot()),
        lambda: callback.policy_digest(trusted(policy())),
    ):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize(
    "field,value",
    [("max_dimension", 1), ("max_layers", 1), ("max_array_bytes", 1), ("max_encoded_bytes", 512)],
)
def test_should_refuse_native_bounds_before_any_restore_model_callback(monkeypatch, field, value):
    callback = CoupledCheckpointCallbacks(
        policy(), binding=BINDING, limits=replace(LIMITS, **{field: value})
    )
    touched = []

    def forbidden(*args, **kwargs):
        touched.append(True)
        raise AssertionError("native bound reached model")

    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", forbidden)
    with pytest.raises(ValueError):
        callback.state_digest(snapshot())
    with pytest.raises(ValueError):
        callback.build(snapshot())
    assert touched == []


class FakeCoupledLearner:
    def __init__(self, native, configured):
        self.native, self.configured = deepcopy(native), deepcopy(configured)
        self.trains = 0

    def fork(self):
        return FakeCoupledLearner(self.native, self.configured)

    def checkpoint_policy(self):
        return deepcopy(self.configured)

    def snapshot_state(self):
        return deepcopy(self.native)

    def restore_state(self, native):
        self.native = deepcopy(native)

    def predict(self, features):
        return [[float(self.native.parameters.output_bias[0, 0])]]

    def train_batch(self, features, targets):
        self.native = advance(self.native)
        self.trains += 1
        return TrainingDiagnostic("fake_coupled_update_v1", 0.75)


def test_should_use_actual_controller_for_wrong_policy_refusal_owned_transfer_delayed_continuation_and_original_quotas(
    monkeypatch,
):
    configured = policy()
    source = FakeCoupledLearner(snapshot(), configured)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    old = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=3,
        max_consolidations=0,
    )
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(old, gate)
    for index, at in enumerate((3, 7, 9)):
        sample = f"s{index}"
        old.record_experience(
            Experience(
                sample, "e1", 0, "actor-0", [index], "train", ExperiencePermissions(training=True)
            )
        )
        old.record_label(LabelArrival(f"label-{sample}", sample, "e1", at, "actor-0", [index]))
    clock.advance_to(3)
    assert [update.sample_id for update in shared.train_ready().updates] == ["s0"]
    gate.pause()
    callbacks = CoupledCheckpointCallbacks(configured, binding=BINDING, limits=LIMITS)
    monkeypatch.setattr(callbacks_module, "CoupledLearner", FakeCoupledLearner)
    attempts: list[Any] = []
    observations: list[Any] = []
    actor_before = pickle.dumps(old.actor.snapshot_state())

    def restore(raw, **kwargs):
        captured = CoupledLearnerCheckpointCodec().decode(raw, **kwargs)
        observations.append(shared.predict([0]).prediction)
        selected = replace(captured.policy, learning_rate=0.05) if not attempts else captured.policy
        model = FakeCoupledLearner(captured.native_state, selected)
        attempts.append(model)
        return model

    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    control = CandidateCheckpointController(
        shared,
        build_learner=callbacks.build,
        state_digest=callbacks.state_digest,
        policy_digest=callbacks.policy_digest,
        max_prepare_attempts=2,
    )
    token = control.capture()
    inbox_before = pickle.dumps(old._inbox.capture_cursor())

    def captured():
        return pickle.dumps(
            (
                old.candidate_snapshot(),
                old._inbox.capture_cursor(),
                gate.snapshot(),
                {key: value for key, value in vars(budget).items() if key != "clock"},
            )
        )

    before = captured()
    with pytest.raises(ValueError):
        control.restore(token)
    assert shared._runtime is old and not old._retired and control._attempts == 1
    assert captured() == before
    new = control.restore(token)
    assert new is shared._runtime and new is not old and old._retired
    assert (
        new.actor is old.actor
        and new._budget is budget
        and new._inbox._clock is clock
        and shared._sharing is gate
    )
    assert new.applied_updates == old.applied_updates and control._attempts == 2
    assert new._inbox.capture_cursor().completed_updates == 1
    assert pickle.dumps(new._inbox.capture_cursor()) == inbox_before
    assert payload(new.candidate_snapshot().state) == payload(advance(snapshot()))
    with pytest.raises(ValueError):
        control.restore(token)
    with pytest.raises(ValueError):
        old.train_ready()
    assert len(attempts) == 2 and all(model.trains == 0 for model in attempts)
    assert len(observations) == 2 and observations[0] == observations[1]
    assert pickle.dumps(new.actor.snapshot_state()) == actor_before
    gate.resume()
    assert shared.train_ready().updates == ()
    clock.advance_to(7)
    assert [update.sample_id for update in shared.train_ready().updates] == ["s1"]
    assert payload(new.candidate_snapshot().state) == payload(advance(advance(snapshot())))
    clock.advance_to(9)
    assert shared.train_ready().updates == ()
    assert budget.updates_completed == gate.snapshot().admitted_updates == 2
    with pytest.raises(ToyExecutionStopped):
        budget.before_update()
    assert pickle.dumps(new.actor.snapshot_state()) == actor_before and source.trains == 0
