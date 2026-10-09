"""Matched inner-guard evaluation, role timing, ownership and negative evidence."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import pickle
from typing import Any

import numpy as np
import pytest

from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.core.actor_ports import CandidateState, VersionedState
from src.core.promotion_guard import GuardBatch, PromotionPolicy


class ScalarLearner:
    def __init__(self, state):
        self.state = list(state)

    def train_batch(self, features, targets):
        raise AssertionError("promotion guard never trains")

    def predict(self, features):
        return [self.state[features[0]]]

    def snapshot_state(self):
        return self.state[:]

    def restore_state(self, state):
        self.state = list(state)


class Clock:
    def __init__(self):
        self.time = 0.0

    def __call__(self):
        self.time += 0.01
        return self.time


def snapshots():
    return VersionedState("actor-0", [0.5, 0.8]), CandidateState(
        "actor-0", "candidate-0", [0.75, 0.8], 1, 0, 0
    )


def guards():
    return (
        GuardBatch("new", (("e-new", "s1"),), 2, [0], [1.0], "inner_guard"),
        GuardBatch("old", (("e-old", "s1"),), 1, [1], [1.0], "inner_guard"),
    )


def policy():
    return PromotionPolicy(
        "one_minus_absolute_error_v1",
        "fixture_state_bytes_v1",
        0.6,
        0.1,
        0.05,
        0.1,
        100,
        ("allow",),
    )


def make_evaluator(**changes):
    arguments: dict[str, Any] = dict(
        build_learner=ScalarLearner,
        utility=lambda predictions, targets: 1.0 - abs(predictions[0] - targets[0]),
        actions=lambda prediction: ("allow",) if prediction[0] <= 1.0 else ("unsafe",),
        state_valid=lambda state: all(np.isfinite(state)),
        prediction_valid=lambda prediction: all(np.isfinite(prediction)),
        resource_bytes=lambda state: 16,
        state_digest=lambda state: sha256(pickle.dumps(state)).hexdigest(),
        clock=Clock(),
    )
    arguments.update(changes)
    return PromotionGuardEvaluator(**arguments)


def evaluate(
    evaluator=None, actor=None, candidate=None, new=None, old=None, now=2, train_ids=frozenset()
):
    original_actor, original_candidate = snapshots()
    new_guard, old_guard = guards()
    return (evaluator or make_evaluator()).evaluate(
        policy(),
        actor or original_actor,
        candidate or original_candidate,
        new or new_guard,
        old or old_guard,
        now=now,
        training_ids=train_ids,
    )


def test_should_measure_same_new_and_old_inputs_and_metric_for_actor_and_candidate():
    scored = []

    def metric(prediction, targets):
        scored.append((tuple(prediction), tuple(targets)))
        return 1.0 - abs(prediction[0] - targets[0])

    report = evaluate(make_evaluator(utility=metric))
    assert report.decision.accepted and report.decision.reasons == ()
    assert report.evidence.actor_new_utility == 0.5
    assert report.evidence.candidate_new_utility == 0.75
    assert report.evidence.actor_old_utility == report.evidence.candidate_old_utility == 0.8
    assert [item[1] for item in scored] == [(1.0,)] * 4
    assert report.actor_version == "actor-0" and report.learner_version == "candidate-0"
    assert report.candidate_revision == (1, 0, 0)
    assert report.policy == policy() and report.assessed_at == 2


class SealedPayload:
    def __deepcopy__(self, memo):
        raise AssertionError("unauthorized guard payload was copied")


@pytest.mark.parametrize("role", ["train", "outer_selection", "final_test"])
def test_should_refuse_non_inner_roles_before_payload_factory_or_probe_access(role):
    def forbidden(value):
        raise AssertionError("metadata gate must precede callbacks")

    new, _ = guards()
    denied = replace(new, role=role, features=SealedPayload(), targets=SealedPayload())
    with pytest.raises(ValueError, match="inner_guard"):
        evaluate(make_evaluator(build_learner=forbidden, state_digest=forbidden), new=denied)


@pytest.mark.parametrize(
    "case",
    ["future", "guard_overlap", "training_overlap", "base_version", "task_overlap", "revision"],
)
def test_should_refuse_chronology_identity_or_base_conflicts_before_payload_access(case):
    new, old = guards()
    candidate = snapshots()[1]
    new = replace(new, features=SealedPayload(), targets=SealedPayload())
    train: frozenset[tuple[str, str]] = frozenset()
    if case == "future":
        new = replace(new, labels_arrived_at=3)
    elif case == "guard_overlap":
        old = replace(old, sample_keys=new.sample_keys)
    elif case == "training_overlap":
        train = frozenset(new.sample_keys)
    elif case == "base_version":
        candidate = replace(candidate, actor_version="stale")
    elif case == "task_overlap":
        old = replace(old, task_id=new.task_id)
    else:
        candidate = replace(candidate, consolidations_completed=1)
    with pytest.raises(ValueError):
        evaluate(new=new, old=old, candidate=candidate, train_ids=train)


def test_should_detach_all_model_input_target_and_callback_values():
    actor, candidate = snapshots()
    new, old = guards()
    original = pickle.dumps((actor, candidate, new, old))
    retained = []

    def builder(state):
        retained.append(state)
        learner = ScalarLearner(state)
        state[:] = [99, 99]
        return learner

    def metric(prediction, targets):
        value = 1.0 - abs(prediction[0] - targets[0])
        prediction[:] = [99]
        targets[:] = [99]
        return value

    report = evaluate(
        make_evaluator(build_learner=builder, utility=metric),
        actor=actor,
        candidate=candidate,
        new=new,
        old=old,
    )
    assert report.decision.accepted
    assert pickle.dumps((actor, candidate, new, old)) == original
    assert len(retained) == 2 and report.evidence.actions_safe


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"actions": lambda prediction: ("unsafe",)}, "action_safety"),
        ({"actions": lambda prediction: ()}, "action_safety"),
        ({"resource_bytes": lambda state: 101}, "resource"),
        ({"prediction_valid": lambda prediction: False}, "numerical_validity"),
        ({"utility": lambda prediction, targets: float("nan")}, "numerical_validity"),
    ],
)
def test_should_reject_measured_bad_candidates_with_explicit_reasons(changes, reason):
    report = evaluate(make_evaluator(**changes))
    assert not report.decision.accepted and reason in report.decision.reasons


def test_should_reject_numerically_bad_state_before_building_predictors():
    def forbidden(state):
        raise AssertionError("invalid numerical state must not reach factory")

    report = evaluate(make_evaluator(state_valid=lambda state: False, build_learner=forbidden))
    assert not report.decision.accepted and "numerical_validity" in report.decision.reasons
    assert report.evidence.candidate_new_utility is None


def test_should_reject_actual_prediction_latency_without_moving_the_limit():
    class SlowClock:
        time = 0.0

        def __call__(self):
            self.time += 0.2
            return self.time

    report = evaluate(make_evaluator(clock=SlowClock()))
    assert report.evidence.candidate_max_prediction_seconds > report.policy.max_prediction_seconds
    assert not report.decision.accepted and "latency" in report.decision.reasons


@pytest.mark.parametrize(
    "changes",
    [
        {"state_digest": lambda state: "unbound"},
        {"resource_bytes": lambda state: True},
        {"prediction_valid": lambda prediction: 1},
        {"actions": lambda prediction: ["allow"]},
    ],
)
def test_should_refuse_malformed_native_probe_outputs(changes):
    with pytest.raises(ValueError):
        evaluate(make_evaluator(**changes))


def test_should_reject_prediction_that_changes_model_state_during_assessment():
    class MutatingLearner(ScalarLearner):
        def predict(self, features):
            result = super().predict(features)
            self.state[0] += 0.001
            return result

    report = evaluate(make_evaluator(build_learner=MutatingLearner))
    assert not report.decision.accepted and "numerical_validity" in report.decision.reasons


@pytest.mark.parametrize("readings", [[1.0, 0.0], [float("nan")], [True]])
def test_should_refuse_invalid_or_backwards_observation_clock(readings):
    values = iter(readings)
    with pytest.raises(ValueError, match="clock"):
        evaluate(make_evaluator(clock=lambda: next(values)))


def test_should_propagate_restore_failure_without_changing_original_snapshots():
    class FailedRestore(ScalarLearner):
        def restore_state(self, state):
            self.state[:] = [99, 99]
            raise RuntimeError("failed restore")

    actor, candidate = snapshots()
    original = pickle.dumps((actor, candidate))
    with pytest.raises(RuntimeError, match="failed restore"):
        evaluate(make_evaluator(build_learner=FailedRestore), actor=actor, candidate=candidate)
    assert pickle.dumps((actor, candidate)) == original


def test_should_refuse_a_factory_that_returns_the_same_mutable_predictor():
    shared = ScalarLearner([0.5, 0.8])
    with pytest.raises(ValueError, match="independent"):
        evaluate(make_evaluator(build_learner=lambda state: shared))


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_record_equal_native_guard_utility_as_negative_gain_evidence_without_tuning(kind):
    from test_experience_inbox import make_native_pair

    _, source = make_native_pair(kind)
    snapshot = source.snapshot_state()
    original = pickle.dumps(snapshot)
    actor = VersionedState("actor-0", deepcopy(snapshot))
    candidate = CandidateState("actor-0", "candidate-0", deepcopy(snapshot), 0, 0, 0)

    def builder(state):
        return source.fork()

    evaluator = PromotionGuardEvaluator(
        build_learner=builder,
        utility=lambda predictions, targets: float(np.mean((predictions >= 0.5) == targets)),
        actions=lambda predictions: tuple("allow" for _ in predictions),
        state_valid=lambda state: all(
            np.all(np.isfinite(value))
            for value in state.state.values()
            if isinstance(value, np.ndarray)
        ),
        prediction_valid=lambda predictions: bool(np.all(np.isfinite(predictions))),
        resource_bytes=lambda state: len(pickle.dumps(state)),
        state_digest=lambda state: sha256(pickle.dumps(state)).hexdigest(),
        clock=Clock(),
    )
    new = GuardBatch(
        "new",
        (("new", "1"), ("new", "2")),
        1,
        np.array([[0.3, -0.2], [-0.5, 0.4]]),
        np.array([[1.0], [0.0]]),
        "inner_guard",
    )
    old = GuardBatch(
        "old",
        (("old", "1"), ("old", "2")),
        1,
        np.array([[0.1, 0.6], [-0.7, -0.1]]),
        np.array([[1.0], [0.0]]),
        "inner_guard",
    )
    declared = PromotionPolicy(
        "shared_binary_accuracy_v1",
        "pickle_snapshot_bytes_v1",
        0.0,
        0.1,
        0.0,
        0.1,
        1000000,
        ("allow",),
    )
    report = evaluator.evaluate(
        declared, actor, candidate, new, old, now=1, training_ids=frozenset()
    )
    assert report.evidence.actor_new_utility == report.evidence.candidate_new_utility
    assert report.evidence.actor_old_utility == report.evidence.candidate_old_utility
    assert not report.decision.accepted and "new_task_gain" in report.decision.reasons
    assert report.evidence.candidate_resource_bytes == len(original)
    assert pickle.dumps(source.snapshot_state()) == original
