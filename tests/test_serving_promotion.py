"""Guarded complete serving transitions, ownership and deterministic concurrency."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import pickle
from threading import Event

import numpy as np
import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.experience import LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.serving_ports import ServingConfiguration
from test_promotion_guard_evaluation import ScalarLearner, guards, make_evaluator, policy


def digest(state):
    return sha256(pickle.dumps(state)).hexdigest()


def native_state_valid(snapshot):
    def finite(value):
        if isinstance(value, np.ndarray):
            return bool(np.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        if isinstance(value, (int, float, np.number)):
            return bool(np.isfinite(value))
        return True

    return finite(snapshot.state)


class Source(ScalarLearner):
    def fork(self):
        return Source(self.state)


def preparation_failure_builder(builder):
    """Two valid guard forks, then the deliberately broken preparation fork.

    All calls use one configured factory; the broken behavior is intentional
    failure injection, outside the normal same-native-policy port contract.
    """
    calls = 0

    def build(state):
        nonlocal calls
        calls += 1
        return builder(state) if calls % 3 == 0 else ScalarLearner(state)

    return build


def bind_preparation_builder(controller, builder):
    factory = preparation_failure_builder(builder)
    controller._build = controller._evaluator._build = factory


def setup(*, evaluator=None, declared=None, builder=ScalarLearner):
    source = Source([0.5, 0.8])
    actor = PromotableActor(
        source,
        version="actor-0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=digest,
        metadata={"route": ["old"]},
    )
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0),
        actor=actor,
    )
    runtime.consolidate(
        "new", lambda state: ConsolidatedState([0.75, 0.8], TrainingDiagnostic("fixture", 0.0))
    )
    factory = builder if builder is ScalarLearner else preparation_failure_builder(builder)
    evaluator = evaluator or make_evaluator(build_learner=factory)
    controller = ServingPromotionController(
        actor,
        policy=declared or policy(),
        evaluator=evaluator,
        build_learner=evaluator._build,
        state_digest=digest,
        max_pending=4,
    )
    return actor, runtime, controller


def prepare(controller, runtime, **kwargs):
    new, old = guards()
    return controller.prepare(runtime, new, old, now=2, training_ids=frozenset(), **kwargs)


def test_should_commit_all_supported_state_and_restore_exact_prior_bundle():
    actor, runtime, controller = setup()
    actor.serve([0], now=1)
    before = actor.serving_snapshot()
    ticket = prepare(controller, runtime, metadata={"route": ["new"]})
    receipt = controller.commit(runtime, ticket)
    after = actor.serving_snapshot()
    assert after.generation == 1 and after.actor_version == "candidate-0"
    assert after.model_state == [0.75, 0.8] and after.cache == {}
    assert after.metadata == {"route": ["new"]}
    assert actor.serve([0], now=2).prediction == [0.75]
    controller.rollback(receipt)
    restored = actor.serving_snapshot()
    assert restored == replace(before, generation=2)
    assert actor.serve([0], now=1).prediction == [0.5]
    assert runtime.candidate_snapshot().actor_version == "actor-0"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"min_new_utility": 0.9}, "new_task_utility"),
        ({"min_new_gain": 0.4}, "new_task_gain"),
        ({"max_prediction_seconds": 0.0}, "latency"),
        ({"max_resource_bytes": 0}, "resource"),
        ({"allowed_actions": ("other",)}, "action_safety"),
    ],
)
def test_should_refuse_configured_guard_failures_without_serving_change(changes, reason):
    actor, runtime, controller = setup(declared=replace(policy(), **changes))
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match=reason):
        prepare(controller, runtime)
    assert actor.serving_snapshot() == before


@pytest.mark.parametrize("kind", ["numerical", "retention"])
def test_should_refuse_bad_state_and_old_retention(kind):
    actor, runtime, controller = setup()
    state = [float("nan"), 0.8] if kind == "numerical" else [0.75, 0.1]
    runtime.consolidate(
        "bad", lambda _: ConsolidatedState(state, TrainingDiagnostic("fixture", 0.0))
    )
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="numerical_validity|old_task_retention"):
        prepare(controller, runtime)
    assert actor.serving_snapshot() == before


@pytest.mark.parametrize("role", ["train", "outer_selection", "final_test"])
def test_should_refuse_non_inner_roles_without_serving_or_payload_access(role):
    actor, runtime, controller = setup()
    new, old = guards()
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="inner_guard"):
        controller.prepare(runtime, replace(new, role=role), old, now=2, training_ids=frozenset())
    assert actor.serving_snapshot() == before


def test_should_refuse_changed_candidate_revision_even_if_model_bytes_are_equal():
    actor, runtime, controller = setup()
    ticket = prepare(controller, runtime)
    runtime.consolidate(
        "same", lambda state: ConsolidatedState(state, TrainingDiagnostic("fixture", 0.0))
    )
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="candidate"):
        controller.commit(runtime, ticket)
    assert actor.serving_snapshot() == before


def test_should_refuse_changed_candidate_model_bytes_even_if_revision_is_equal():
    actor, runtime, controller = setup()
    ticket = prepare(controller, runtime)
    runtime._candidate.restore_state([0.7, 0.8])  # simulate a broken trusted port
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="candidate"):
        controller.commit(runtime, ticket)
    assert actor.serving_snapshot() == before


def test_should_refuse_forged_cross_runtime_and_consumed_tickets():
    actor, runtime, controller = setup()
    ticket = prepare(controller, runtime)
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="ticket"):
        controller.commit(runtime, replace(ticket))
    _, other, _ = setup()
    with pytest.raises(ValueError, match="runtime"):
        controller.commit(other, ticket)
    assert actor.serving_snapshot() == before
    controller.commit(runtime, ticket)
    with pytest.raises(ValueError, match="ticket"):
        controller.commit(runtime, ticket)


def test_should_refuse_stale_generation_after_commit_and_rollback_aba():
    actor, runtime, controller = setup()
    first, stale = prepare(controller, runtime), prepare(controller, runtime)
    receipt = controller.commit(runtime, first)
    controller.rollback(receipt)
    before = actor.serving_snapshot()
    assert before.actor_version == stale.report.actor_version and before.generation == 2
    with pytest.raises(ValueError, match="generation"):
        controller.commit(runtime, stale)
    assert actor.serving_snapshot() == before
    with pytest.raises(ValueError, match="rollback"):
        controller.rollback(receipt)


def test_should_refuse_modified_policy_or_decision_in_issued_report():
    actor, runtime, controller = setup()
    ticket = prepare(controller, runtime)
    before = actor.serving_snapshot()
    object.__setattr__(
        ticket, "report", replace(ticket.report, policy=replace(policy(), min_new_gain=0.0))
    )
    with pytest.raises(ValueError, match="report"):
        controller.commit(runtime, ticket)
    assert actor.serving_snapshot() == before


def test_should_refuse_mutated_prepared_predictor():
    built = []

    def build(state):
        model = ScalarLearner(state)
        built.append(model)
        return model

    actor, runtime, controller = setup(builder=build)
    ticket = prepare(controller, runtime)
    built[-1].state[:] = [99, 99]
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="prepared"):
        controller.commit(runtime, ticket)
    assert actor.serving_snapshot() == before


def test_should_leave_actor_unchanged_after_partial_preparation_restore_failure():
    class Broken(ScalarLearner):
        def restore_state(self, state):
            self.state[0] = state[0]
            raise RuntimeError("partially restored preparation")

    actor, runtime, controller = setup(builder=Broken)
    before = actor.serving_snapshot()
    with pytest.raises(RuntimeError, match="partially restored"):
        prepare(controller, runtime)
    assert actor.serving_snapshot() == before
    assert actor.predict([0]).prediction == [0.5]


def test_should_detach_metadata_predictions_model_and_cache_snapshots():
    actor, runtime, controller = setup()
    metadata = {"route": ["new"]}
    ticket = prepare(controller, runtime, metadata=metadata)
    metadata["route"].append("mutation")
    controller.commit(runtime, ticket)
    output = actor.serve([0], now=2)
    output.prediction[:] = [99]
    output.metadata["route"].append("mutation")
    snapshot = actor.serving_snapshot()
    snapshot.model_state[:] = [99, 99]
    snapshot.cache[digest([0])].prediction[:] = [99]
    snapshot.metadata["route"].append("mutation")
    assert actor.serve([0], now=2).prediction == [0.75]
    assert actor.serving_snapshot().metadata == {"route": ["new"]}


def test_should_enforce_cache_ttl_capacity_clock_and_feature_identity():
    actor, _, _ = setup()
    assert not actor.serve([0], now=1).cache_hit
    assert actor.serve([0], now=3).cache_hit
    assert not actor.serve([0], now=4).cache_hit  # expiry boundary
    actor.serve([1], now=4)
    actor.serve([0, "different-input"], now=4)
    assert len(actor.serving_snapshot().cache) == 2
    assert digest([0]) not in actor.serving_snapshot().cache
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="backwards"):
        actor.serve([0], now=3)
    assert actor.serving_snapshot() == before


def test_should_bound_pending_tickets_and_allow_explicit_discard():
    _, runtime, controller = setup()
    tickets = [prepare(controller, runtime) for _ in range(4)]
    with pytest.raises(ValueError, match="quota"):
        prepare(controller, runtime)
    controller.discard(tickets[0])
    prepare(controller, runtime)


def test_should_refuse_modified_ticket_generation_and_rollback_receipt():
    actor, runtime, controller = setup()
    ticket = prepare(controller, runtime)
    object.__setattr__(ticket, "actor_generation", 99)
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="generation"):
        controller.commit(runtime, ticket)
    assert actor.serving_snapshot() == before
    valid = prepare(controller, runtime)
    receipt = controller.commit(runtime, valid)
    object.__setattr__(receipt, "generation", 99)
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="modified"):
        controller.rollback(receipt)
    assert actor.serving_snapshot() == before


def test_should_refuse_changed_actor_bytes_without_publication():
    actor, runtime, controller = setup()
    ticket = prepare(controller, runtime)
    actor._slot.bundle.learner.restore_state([0.4, 0.8])  # broken trusted ownership
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="actor model state"):
        controller.commit(runtime, ticket)
    assert actor.serving_snapshot() == before


def test_should_refuse_builders_returning_live_actor_or_candidate_before_restore():
    for owner in ("actor", "candidate"):
        actor, runtime, controller = setup()
        bind_preparation_builder(
            controller,
            lambda state: actor._slot.bundle.learner if owner == "actor" else runtime._candidate,
        )
        before = actor.serving_snapshot()
        candidate = runtime.candidate_snapshot()
        with pytest.raises(ValueError, match="independently owned"):
            prepare(controller, runtime)
        assert actor.serving_snapshot() == before
        assert runtime.candidate_snapshot() == candidate


def test_should_refuse_builder_reusing_a_prepared_model():
    model = ScalarLearner([0.5, 0.8])
    actor, runtime, controller = setup(builder=lambda state: model)
    ticket = prepare(controller, runtime)
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="independently owned"):
        prepare(controller, runtime)
    assert actor.serving_snapshot() == before
    controller.commit(runtime, ticket)


def test_should_refuse_restore_that_does_not_reproduce_complete_candidate_state():
    class Incomplete(ScalarLearner):
        def restore_state(self, state):
            self.state = [state[0], -1]

    actor, runtime, controller = setup(builder=Incomplete)
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="complete candidate"):
        prepare(controller, runtime)
    assert actor.serving_snapshot() == before


def test_should_retain_completed_ticket_after_prepublication_digest_failure():
    actor, runtime, controller = setup()
    ticket = prepare(controller, runtime)
    original = controller._digest
    controller._digest = lambda state: "invalid"
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="SHA256"):
        controller.commit(runtime, ticket)
    assert actor.serving_snapshot() == before
    controller._digest = original
    controller.commit(runtime, ticket)


def test_should_refuse_actual_training_sample_overlap_even_if_caller_omits_it():
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival

    actor, runtime, controller = setup()
    runtime._candidate.train_batch = lambda x, y: TrainingDiagnostic("fixture", 0.0)
    runtime.record_experience(
        Experience("s1", "e-new", 1, "actor-0", [0], "train", ExperiencePermissions(training=True))
    )
    runtime.record_label(LabelArrival("label", "s1", "e-new", 2, "actor-0", [1]))
    runtime._inbox._clock.advance_to(2)
    runtime.train_ready()
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="disjoint"):
        prepare(controller, runtime)
    assert actor.serving_snapshot() == before


@pytest.mark.parametrize("kind", ["copy", "factory", "predict"])
def test_should_preserve_published_bundle_after_copy_factory_or_read_failure(kind):
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy metadata")

    actor, runtime, controller = setup()
    before = actor.serving_snapshot()
    if kind == "copy":
        with pytest.raises(RuntimeError, match="copy"):
            prepare(controller, runtime, metadata={"route": CannotCopy()})
    elif kind == "factory":

        def fail(state):
            raise RuntimeError("factory failed")

        bind_preparation_builder(controller, fail)
        with pytest.raises(RuntimeError, match="factory"):
            prepare(controller, runtime)
    else:
        original = actor._slot.bundle.learner.predict

        def fail(features):
            raise RuntimeError("predict failed")

        actor._slot.bundle.learner.predict = fail
        with pytest.raises(RuntimeError, match="predict"):
            actor.serve([0], now=2)
        actor._slot.bundle.learner.predict = original
    assert actor.serving_snapshot() == before


@pytest.mark.parametrize("bad", [(0, 2), (3, 0), (True, 2), (-1, 2)])
def test_should_refuse_unsupported_cache_configuration(bad):
    with pytest.raises(ValueError):
        ServingConfiguration(*bad)


def test_should_refuse_bad_feature_digest_without_publishing_cache_state():
    actor, _, _ = setup()
    actor._feature_digest = lambda features: "bad"
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="SHA256"):
        actor.serve([0], now=2)
    assert actor.serving_snapshot() == before


def test_should_refuse_nested_controller_and_candidate_callbacks_without_deadlock():
    actor, runtime, controller = setup()

    def build(state):
        with pytest.raises(ValueError, match="busy"):
            prepare(controller, runtime)
        with pytest.raises(ValueError, match="busy"):
            runtime.candidate_snapshot()
        return ScalarLearner(state)

    bind_preparation_builder(controller, build)
    ticket = prepare(controller, runtime)
    controller.commit(runtime, ticket)


def test_should_refuse_commit_while_candidate_training_gate_is_held():
    actor, runtime, controller = setup()
    ticket = prepare(controller, runtime)
    entered, release = Event(), Event()

    def lease(snapshot):
        entered.set()
        assert release.wait(3)

    before = actor.serving_snapshot()
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = pool.submit(runtime.with_candidate_snapshot, lease)
        try:
            assert entered.wait(3)
            with pytest.raises(ValueError, match="busy"):
                controller.commit(runtime, ticket)
            assert actor.serving_snapshot() == before
        finally:
            release.set()
        work.result(timeout=3)
    controller.commit(runtime, ticket)


def test_should_keep_prior_cache_updates_at_actual_commit_boundary():
    actor, runtime, controller = setup()
    ticket = prepare(controller, runtime)
    actor.serve([1], now=2)  # cache evolves after approval; model/generation stay equal
    before = actor.serving_snapshot()
    receipt = controller.commit(runtime, ticket)
    controller.rollback(receipt)
    assert actor.serving_snapshot() == replace(before, generation=2)


def test_should_refuse_different_measured_and_prepared_native_builders():
    actor, runtime, controller = setup()
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="identical native builder"):
        ServingPromotionController(
            actor,
            policy=policy(),
            evaluator=make_evaluator(),
            build_learner=lambda state: ScalarLearner(state),
            state_digest=digest,
        )
    ticket = prepare(controller, runtime)
    controller._build = lambda state: ScalarLearner(state)
    with pytest.raises(ValueError, match="builder"):
        controller.commit(runtime, ticket)
    with pytest.raises(ValueError, match="builder"):
        prepare(controller, runtime)
    assert actor.serving_snapshot() == before


def test_should_refuse_rebased_candidate_and_stale_rollback_after_new_commit():
    actor, runtime, controller = setup()
    receipt = controller.commit(runtime, prepare(controller, runtime))
    assert runtime.candidate_snapshot().actor_version == "actor-0"
    with pytest.raises(ValueError, match="stale"):
        prepare(controller, runtime)
    # New generation starts from the newly served native model, not the old inbox.
    source = Source(actor.snapshot_state().state)
    fresh = ActorShadowRuntime(
        source,
        actor_version="candidate-0",
        candidate_version="candidate-1",
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0),
        actor=actor,
    )
    fresh.consolidate(
        "new", lambda _: ConsolidatedState([0.9, 0.8], TrainingDiagnostic("fixture", 0.0))
    )
    next_receipt = controller.commit(fresh, prepare(controller, fresh))
    before = actor.serving_snapshot()
    with pytest.raises(ValueError, match="rollback"):
        controller.rollback(receipt)
    assert actor.serving_snapshot() == before
    controller.rollback(next_receipt)
    assert actor.version == "candidate-0"


def test_should_keep_serving_available_during_native_preparation():
    entered, release = Event(), Event()

    def build(state):
        entered.set()
        assert release.wait(3)
        return ScalarLearner(state)

    actor, runtime, controller = setup(builder=build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(prepare, controller, runtime)
        try:
            assert entered.wait(3)
            assert pool.submit(actor.predict, [0]).result(timeout=3).prediction == [0.5]
            with pytest.raises(ValueError, match="busy"):
                runtime.candidate_snapshot()
        finally:
            release.set()
        future.result(timeout=3)


@pytest.mark.parametrize("operation", ["commit", "rollback"])
def test_should_serialize_inflight_complete_serving_frame_with_bundle_swap(operation):
    actor, runtime, controller = setup()
    ticket = prepare(controller, runtime, metadata={"route": ["new"]})
    receipt = controller.commit(runtime, ticket) if operation == "rollback" else None
    entered, release, attempted = Event(), Event(), Event()
    original = actor._slot.bundle.learner.predict

    def predict(features):
        entered.set()
        assert release.wait(3)
        return original(features)

    actor._slot.bundle.learner.predict = predict

    def transition():
        attempted.set()
        return (
            controller.commit(runtime, ticket)
            if operation == "commit"
            else controller.rollback(receipt)
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(actor.serve, [0], now=2)
        try:
            assert entered.wait(3)
            swap = pool.submit(transition)
            assert attempted.wait(3) and not swap.done()
        finally:
            release.set()
        frame = read.result(timeout=3)
        swap.result(timeout=3)
    expected = (
        ("actor-0", 0, [0.5], {"route": ["old"]})
        if operation == "commit"
        else ("candidate-0", 1, [0.75], {"route": ["new"]})
    )
    assert (frame.actor_version, frame.generation, frame.prediction, frame.metadata) == expected
    assert actor.serving_snapshot().generation == (1 if operation == "commit" else 2)


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_promote_and_restore_complete_fixed_native_model_and_serving_cache(kind):
    from test_experience_inbox import make_native_pair
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival
    from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
    from src.core.promotion_guard import GuardBatch, PromotionPolicy

    _, source = make_native_pair(kind)
    direct = source.fork()
    features, targets = np.array([[0.3, -0.2], [-0.5, 0.4]]), np.array([[1.0], [0.0]])
    clock = LogicalClock()
    actor = PromotableActor(
        source,
        version="actor-0",
        configuration=ServingConfiguration(5, 2),
        feature_digest=digest,
        metadata={"route": ["old"]},
    )
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0),
        actor=actor,
    )
    runtime.record_experience(
        Experience(
            "s1", "train", 1, "actor-0", features, "train", ExperiencePermissions(training=True)
        )
    )
    runtime.record_label(LabelArrival("label", "s1", "train", 2, "actor-0", targets))
    clock.advance_to(2)
    runtime.train_ready()  # one fixture native wake per kind, no sleep
    build = lambda state: source.fork()
    # Fixed permissive transaction fixture policy; no advantage or deployment claim.
    declared = PromotionPolicy(
        "binary_accuracy_v1",
        "serialized_native_state_bytes_v1",
        0.0,
        -1.0,
        1.0,
        1.0,
        1048576,
        ("0", "1"),
    )
    evaluator = PromotionGuardEvaluator(
        build_learner=build,
        utility=lambda p, t: float(np.mean((p >= 0.5) == t)),
        actions=lambda p: tuple(str(int(v)) for v in (p >= 0.5).ravel()),
        state_valid=native_state_valid,
        prediction_valid=lambda p: bool(np.isfinite(p).all()),
        resource_bytes=lambda state: len(pickle.dumps(state)),
        state_digest=digest,
        clock=lambda: 0.0,
    )
    controller = ServingPromotionController(
        actor,
        policy=declared,
        evaluator=evaluator,
        build_learner=build,
        state_digest=digest,
        max_pending=2,
    )
    original = actor.serve(features, now=2).prediction  # 1 native prediction
    before = actor.serving_snapshot()
    new = GuardBatch("new", (("new", "s1"), ("new", "s2")), 2, features, targets, "inner_guard")
    old = replace(new, task_id="old", sample_keys=(("old", "s1"), ("old", "s2")))
    ticket = controller.prepare(
        runtime, new, old, now=2, training_ids=frozenset(), metadata={"route": ["new"]}
    )  # 4 native predictions
    receipt = controller.commit(runtime, ticket)
    assert digest(actor.snapshot_state().state) == digest(runtime.candidate_snapshot().state)
    actor.serve(features, now=2)  # 1 native prediction
    controller.rollback(receipt)
    restored = actor.serving_snapshot()
    assert digest(restored.model_state) == digest(before.model_state)
    assert digest(restored.cache) == digest(before.cache) and restored.metadata == before.metadata
    np.testing.assert_array_equal(actor.serve(features, now=2).prediction, original)  # cache hit
    np.testing.assert_array_equal(
        actor.predict(features).prediction, direct.predict(features)
    )  # 2 native predictions
    assert restored.generation == 2 and restored.actor_version == "actor-0"
