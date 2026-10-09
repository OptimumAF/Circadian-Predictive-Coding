"""Tiny actual managed CPC wake integration;run only after declared native gate."""

from contextlib import ExitStack
import json
import sys
from unittest.mock import patch
import numpy as np
import pytest

from src.adapters.numpy_learners import CircadianLearner
from src.adapters.numpy_replay_origins import (
    replay_model_reference,
    retained_replay_references,
    replay_copy_bytes,
    replay_payload_references,
    replay_payload_fingerprint,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
)
from src.core.data_lifecycle import LifecycleLimits
from src.core.experience import LogicalClock, Experience, ExperiencePermissions, LabelArrival
from src.core.replay_origin import ReplayOriginPorts
from src.core.resource_sharing import SharingLimits
from test_managed_experience import declaration
from test_managed_replay_origins import LIMITS, WINDOW


@pytest.fixture(scope="module", autouse=True)
def native_work(tmp_path_factory):
    work = dict(
        model_constructors=0,
        learner_constructors=0,
        forks=0,
        wakes=0,
        inference_steps_requested=0,
        stores=0,
        predictions=0,
        array_copies=0,
    )
    target = tmp_path_factory.mktemp("native-origin-work") / "native-origin-work.json"
    previous = sys.getprofile()

    def counted(name, original):
        def call(*args, **kwargs):
            work[name] += 1
            if name == "wakes":
                work["inference_steps_requested"] += kwargs.get(
                    "inference_steps", args[4] if len(args) > 4 else 0
                )
            return original(*args, **kwargs)

        return call

    def profile(frame, event, function):
        if (
            event == "c_call"
            and frame.f_code.co_filename.endswith("circadian_predictive_coding.py")
            and getattr(function, "__name__", None) == "copy"
            and isinstance(getattr(function, "__self__", None), np.ndarray)
        ):
            work["array_copies"] += 1
        if previous is not None:
            previous(frame, event, function)

    with ExitStack() as stack:
        for cls, method, name in [
            (CircadianPredictiveCodingNetwork, "__init__", "model_constructors"),
            (CircadianLearner, "__init__", "learner_constructors"),
            (CircadianLearner, "fork", "forks"),
            (CircadianPredictiveCodingNetwork, "train_epoch", "wakes"),
            (CircadianPredictiveCodingNetwork, "_store_replay_snapshot", "stores"),
            (CircadianPredictiveCodingNetwork, "predict_proba", "predictions"),
        ]:
            stack.enter_context(patch.object(cls, method, counted(name, getattr(cls, method))))
        sys.setprofile(profile)
        try:
            yield
        finally:
            sys.setprofile(previous)
            target.write_text(json.dumps(work, indent=2), encoding="utf8")
            assert (
                work["model_constructors"] <= 3
                and work["learner_constructors"] <= 9
                and work["forks"] <= 6
            )
            assert work["wakes"] <= 4 and work["inference_steps_requested"] <= 8
            assert work["stores"] <= 4 and work["predictions"] <= 4 and work["array_copies"] <= 64


def setup(seconds=None):
    clock = LogicalClock()
    seconds = seconds if seconds is not None else [0.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1), lambda: seconds[0]
    )
    model = CircadianPredictiveCodingNetwork(
        3, 2, seed=23, min_hidden_dim=2, circadian_config=CircadianConfig(sleep_mode="disabled")
    )
    model.configure_replay_retention(ReplayRetentionBudget(2, 64))
    learner = CircadianLearner(
        model, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )
    runtime = ActorShadowRuntime(
        learner,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=4,
    )
    shared = ResourceSharedRuntime(
        runtime, ServingPriorityGate(SharingLimits(1, 8, 1), resource_available=lambda: True)
    )
    owner = ManagedExperienceOwner(shared, limits=LifecycleLimits(4, 4))
    ports = ReplayOriginPorts(
        replay_model_reference,
        retained_replay_references,
        replay_copy_bytes,
        replay_payload_references,
        replay_payload_fingerprint,
    )
    ledger = ManagedReplayOrigins(owner, ports, LIMITS, WINDOW)
    return owner, ledger, runtime, clock, budget


def queue(owner, clock, sample="s1", subject="person-1"):
    owner.declare(declaration(sample, subject))
    owner.record_experience(
        Experience(
            sample,
            "e1",
            1,
            "actor-0",
            np.array([[1.0, 0.0, 0.0]]),
            "train",
            ExperiencePermissions(True, True),
        )
    )
    owner.record_label(
        LabelArrival("label-" + sample, sample, "e1", 3, "actor-0", np.array([[0.0]]))
    )
    clock.advance_to(3)


def test_should_bind_real_managed_duplicate_copy_to_new_original_consent():
    owner, ledger, runtime, clock, budget = setup()
    fields = tuple(vars(runtime._candidate._model))
    queue(owner, clock)
    first = ledger.train_ready().updates[0]
    assert ledger.origins()[0].update_number == first.update_number == 1
    queue(owner, clock, "s2", "person-2")
    second = ledger.train_ready().updates[0]
    owner.opt_out("person-1")
    assert ledger.origins()[0].key == ("e1", "s2")
    assert ledger.origins()[0].subject_id == "person-2" and second.update_number == 2
    assert budget.updates_completed == runtime._candidate._model._epoch_count == 2
    assert len(runtime._candidate._model._replay_memory) == 1
    assert tuple(vars(runtime._candidate._model)) == fields
    assert ledger.accounting().records_created == 2 and ledger.accounting().live_records == 1


def test_should_refuse_mutated_real_retained_payload_without_rebuilding_origin():
    owner, ledger, runtime, clock, budget = setup()
    queue(owner, clock)
    ledger.train_ready()
    runtime._candidate._model._replay_memory[0].input_batch[0, 0] = 0.0
    with pytest.raises(ValueError, match="integrity changed"):
        ledger.origins()
    assert budget.updates_completed == 1 and ledger.accounting().records_created == 1


def test_should_keep_real_native_work_and_receipt_on_post_update_budget_failure(monkeypatch):
    seconds = [0.0]
    owner, ledger, runtime, clock, budget = setup(seconds)
    queue(owner, clock)
    original = CircadianLearner.train_batch

    def train(learner, features, targets):
        result = original(learner, features, targets)
        seconds[0] = 2.0
        return result

    monkeypatch.setattr(CircadianLearner, "train_batch", train)
    with pytest.raises(ToyExecutionStopped):
        ledger.train_ready()
    assert runtime._candidate._model._epoch_count == budget.updates_completed == 1
    assert (
        runtime.applied_updates[0].update_number == 1
        and len(runtime._candidate._model._replay_memory) == 1
    )
    assert budget.started_at == 0.0 and budget.last_clock == 2.0
    with pytest.raises(ValueError, match="uncertain"):
        ledger.origins()
