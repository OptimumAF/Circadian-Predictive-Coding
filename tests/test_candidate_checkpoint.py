"""Complete owner transfer without native replay or cumulative quota renewal."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from threading import Event

import pytest

from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.experience import LabelArrival
from src.shared.process_memory import ProcessRssSampler
from test_actor_shadow import PausableLearner, consolidate, queue
from test_resource_sharing import shared


def digest(state):
    return hashlib.sha256(pickle.dumps(state)).hexdigest()


def controller(wrapper, **kwargs):
    return CandidateCheckpointController(
        wrapper,
        build_learner=kwargs.pop("build_learner", lambda state: PausableLearner(state)),
        state_digest=digest,
        policy_digest=kwargs.pop("policy_digest", lambda learner: digest("fixed-fixture-policy")),
        **kwargs,
    )


def prepared():
    wrapper, old, clock, gate, source, budget = shared()
    queue(old)
    queue(old, "s2", [9, 9])
    clock.advance_to(3)
    old.train_ready(max_updates=1)
    old.record_label(LabelArrival("label-future", "future", "e1", 7, "actor-0", [4, 4]))
    gate.pause()
    return controller(wrapper), wrapper, old, clock, gate, source, budget


def test_should_transfer_complete_history_retire_old_owner_and_resume_without_replay():
    control, wrapper, old, clock, gate, source, budget = prepared()
    old.consolidate("sleep-ok", consolidate)
    actor_before = old.actor.snapshot_state()
    checkpoint = control.capture()
    view = control.inspect(checkpoint)
    view.inbox.experiences[0].features.append("mutation")
    new = control.restore(checkpoint)
    assert new is wrapper._runtime and new is not old and new.actor is old.actor
    assert new._budget is budget and new._inbox._clock is clock
    assert wrapper._sharing is gate and gate.snapshot().paused
    assert new.applied_updates == old.applied_updates
    assert new.applied_consolidations == old.applied_consolidations
    assert "mutation" not in new._inbox.capture_cursor().experiences[0].features
    assert new.candidate_snapshot().state == [9, 9]
    assert source.value == [2, 2] and new.actor.snapshot_state() == actor_before
    with pytest.raises(ValueError, match="retired"):
        old.train_ready()
    with pytest.raises(ValueError, match="retired"):
        old.with_candidate_snapshot(lambda state: state)
    with pytest.raises(ValueError, match="duplicate"):
        queue(new)
    gate.resume()
    assert [u.sample_id for u in wrapper.train_ready().updates] == ["s2"]
    assert budget.updates_completed == 2 and wrapper.train_ready().updates == ()
    assert [u.update_number for u in new.applied_updates] == [1, 2]


@pytest.mark.parametrize("kind", ["foreign", "copy", "used"])
def test_should_refuse_checkpoint_without_current_single_use_owner_authority(kind):
    control, wrapper, old, _, _, _, _ = prepared()
    token = control.capture()
    if kind == "foreign":
        other = controller(shared()[0])
        with pytest.raises(ValueError, match="checkpoint"):
            other.restore(token)
    elif kind == "copy":
        from copy import copy

        with pytest.raises(ValueError, match="checkpoint"):
            control.restore(copy(token))
    else:
        control.restore(token)
        with pytest.raises(ValueError, match="checkpoint"):
            control.restore(token)


@pytest.mark.parametrize(
    "mutation", ["registration", "attempt", "poll", "policy", "budget", "generation"]
)
def test_should_refuse_stale_candidate_work_policy_budget_or_actor_checkpoint(mutation):
    control, wrapper, old, _, gate, _, budget = prepared()
    token = control.capture()
    if mutation == "registration":
        old.record_label(LabelArrival("extra", "extra", "e1", 8, "actor-0", [3, 3]))
    elif mutation == "attempt":
        with pytest.raises(RuntimeError):
            old.consolidate("failed", lambda state: (_ for _ in ()).throw(RuntimeError("fail")))
    elif mutation == "poll":
        wrapper.train_ready()
    elif mutation == "policy":
        control._policy = lambda learner: digest("changed")
    elif mutation == "budget":
        budget.budget = ToyExecutionBudget(max_training_updates=99)
    else:
        old.actor._version = "actor-1"
    with pytest.raises(ValueError, match="stale|changed"):
        control.restore(token)
    assert wrapper._runtime is old and not old._retired and gate.snapshot().paused


@pytest.mark.parametrize("kind", ["native", "cursor", "attempts", "receipt"])
def test_should_refuse_corrupt_complete_checkpoint_and_leave_live_candidate(kind):
    control, wrapper, old, _, _, _, _ = prepared()
    old.consolidate("sleep", consolidate)
    token = control.capture()
    data = control._pending[token]
    before = old.candidate_snapshot()
    if kind == "native":
        data.view.state[0] = 99
    elif kind == "cursor":
        object.__setattr__(data.view.inbox, "completed_updates", 99)
    elif kind == "attempts":
        object.__setattr__(data.view, "attempted_ids", ())
    else:
        object.__setattr__(data.view.consolidations[0], "attempt_number", 9)
    with pytest.raises(ValueError, match="checkpoint|cursor|consolidation|changed"):
        control.restore(token)
    assert wrapper._runtime is old and old.candidate_snapshot() == before


@pytest.mark.parametrize("kind", ["unpaused", "serving", "training", "candidate"])
def test_should_refuse_busy_capture_before_native_copy_and_preserve_live_owner(kind):
    control, _, old, _, gate, _, _ = prepared()
    if kind == "unpaused":
        gate.resume()
        with pytest.raises(ValueError, match="paused"):
            control.capture()
        return
    if kind == "candidate":
        with old._exclusive(), pytest.raises(ValueError, match="busy"):
            control.capture()
        return
    if kind == "serving":
        with gate.serving(), pytest.raises(ValueError, match="busy"):
            control.capture()
        return
    gate._training = True
    try:
        with pytest.raises(ValueError, match="busy"):
            control.capture()
    finally:
        gate._training = False


@pytest.mark.parametrize("failure", ["builder", "restore", "digest", "alias", "policy"])
def test_should_keep_old_owner_on_failed_independent_native_preparation(failure):
    wrapper, old, clock, gate, _, budget = shared()
    queue(old)
    clock.advance_to(3)
    gate.pause()

    def build(state):
        if failure == "builder":
            raise RuntimeError("builder failed")
        if failure == "alias":
            return old._candidate
        learner = PausableLearner(state)
        if failure == "restore":
            learner.phase = "restore_error"
        if failure == "digest":
            setattr(learner, "snapshot_state", lambda: [99, 99])
        if failure == "policy":
            setattr(learner, "changed", True)
        return learner

    control = controller(
        wrapper,
        build_learner=build,
        policy_digest=lambda learner: digest(getattr(learner, "changed", False)),
    )
    token = control.capture()
    before = old.candidate_snapshot(), old._inbox.capture_cursor(), gate.snapshot()
    with pytest.raises((RuntimeError, ValueError)):
        control.restore(token)
    assert wrapper._runtime is old and not old._retired and budget.updates_completed == 0
    assert (old.candidate_snapshot(), old._inbox.capture_cursor(), gate.snapshot()) == before


def test_should_preserve_failed_consolidation_attempts_stopped_state_and_quotas():
    control, wrapper, old, _, gate, _, budget = prepared()
    with pytest.raises(RuntimeError):
        old.consolidate("failed", lambda state: (_ for _ in ()).throw(RuntimeError("fail")))
    old.consolidate("ok", consolidate)
    token = control.capture()
    new = control.restore(token)
    with pytest.raises(ValueError, match="duplicate"):
        new.consolidate("failed", consolidate)
    with pytest.raises(ValueError, match="quota"):
        new.consolidate("another", consolidate)
    assert new.candidate_snapshot().consolidation_attempts == 2
    assert new.applied_consolidations[0].attempt_number == 2
    assert budget.updates_completed == 1 and gate.snapshot().admitted_updates == 0
    new._candidate.train_batch = lambda x, y: (_ for _ in ()).throw(RuntimeError("uncertain"))
    gate.resume()
    with pytest.raises(RuntimeError):
        wrapper.train_ready()
    gate.pause()
    stopped = control.capture()
    retired = control.restore(stopped)
    assert retired._stopped and retired._inbox.stopped
    with pytest.raises(ValueError, match="stopped"):
        retired.train_ready()


def test_should_bound_pending_and_preparation_attempts_without_retry_loop():
    control, _, _, _, _, _, _ = prepared()
    first = control.capture()
    second = control.capture()
    with pytest.raises(ValueError, match="capacity"):
        control.capture()
    control.discard(second)
    assert control.capture() is not first
    control._build = lambda state: (_ for _ in ()).throw(RuntimeError("failure"))
    # Changing a configured builder invalidates captured checkpoints before callbacks.
    with pytest.raises(ValueError, match="changed"):
        control.restore(first)


def test_should_preserve_original_elapsed_time_and_rss_sampler_without_budget_reset():
    wall = [0.0]
    rss = [10]
    sampler = ProcessRssSampler(read_rss_bytes=lambda: rss[0])
    sampler.sample()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=10, max_process_rss_bytes=100),
        lambda: wall[0],
    )
    budget.attach_memory(sampler)
    from test_actor_shadow import setup
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.core.resource_sharing import SharingLimits

    old, clock, _, _ = setup(budget=budget)
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(old, gate)
    control = controller(wrapper)
    gate.pause()
    wall[0] = 4
    token = control.capture()
    peak = sampler.snapshot().peak_bytes
    wall[0] = 8
    rss[0] = 20
    new = control.restore(token)
    assert new._budget is budget and budget.clock() == 8 and budget.started_at == 0
    assert budget.process_rss_sampler is sampler and sampler.snapshot().peak_bytes >= peak
    queue(new)
    clock.advance_to(3)
    gate.resume()
    wall[0] = 10
    with pytest.raises(ToyExecutionStopped, match="max_wall_seconds"):
        wrapper.train_ready()
    assert budget.updates_completed == 0 and gate.snapshot().admitted_updates == 1


def test_should_keep_serving_available_while_preparing_and_refuse_resume_or_reentry():
    entered, release = Event(), Event()
    control, wrapper, old, _, gate, _, _ = prepared()
    original = control._build

    def build(state):
        entered.set()
        assert release.wait(3)
        return original(state)

    control._build = build
    token = control.capture()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(control.restore, token)
        try:
            assert entered.wait(3)
            assert wrapper.predict(["x"]).prediction == [2, 2]
            with pytest.raises(ValueError, match="checkpoint"):
                gate.resume()
            with pytest.raises(ValueError, match="busy"):
                old.record_label(LabelArrival("extra", "extra", "e1", 4, "actor-0", [3, 3]))
        finally:
            release.set()
        assert pending.result(timeout=3) is wrapper._runtime


def test_should_refuse_unsupported_event_clock_without_claiming_durable_recovery():
    control, _, old, _, _, _, _ = prepared()

    class OtherClock:
        def now(self):
            return 3

    old._inbox._clock = OtherClock()
    with pytest.raises(ValueError, match="LogicalClock"):
        control.capture()


def test_should_charge_failed_preparation_attempts_and_never_renew_controller_quota():
    wrapper, _, _, gate, _, _ = shared()
    gate.pause()
    calls = []

    def build(state):
        calls.append(state)
        raise RuntimeError("failed preparation")

    control = controller(wrapper, build_learner=build, max_prepare_attempts=2)
    token = control.capture()
    for _ in range(2):
        with pytest.raises(RuntimeError):
            control.restore(token)
    with pytest.raises(ValueError, match="quota"):
        control.restore(token)
    control.discard(token)
    token = control.capture()
    with pytest.raises(ValueError, match="quota"):
        control.restore(token)
    assert len(calls) == 2 and not wrapper._runtime._retired


def test_should_preserve_promotable_bundle_and_invalidate_old_pending_promotion_ticket():
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.core.resource_sharing import SharingLimits
    from test_serving_promotion import setup, prepare
    from test_promotion_guard_evaluation import ScalarLearner

    actor, old, promotion = setup()
    actor.serve([0], now=1)
    before = actor.serving_snapshot()
    ticket = prepare(promotion, old)
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    gate.pause()
    wrapper = ResourceSharedRuntime(old, gate)
    control = controller(wrapper, build_learner=ScalarLearner)
    new = control.restore(control.capture())
    assert actor.serving_snapshot() == before
    with pytest.raises(ValueError, match="retired"):
        promotion.commit(old, ticket)
    with pytest.raises(ValueError, match="different"):
        promotion.commit(new, ticket)
    assert actor.serving_snapshot() == before
    fresh = prepare(promotion, new)
    promotion.commit(new, fresh)
    assert actor.serving_snapshot().generation == 1


def test_should_refuse_actor_generation_aba_even_when_its_version_returns_to_base():
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.core.resource_sharing import SharingLimits
    from test_serving_promotion import setup, prepare
    from test_promotion_guard_evaluation import ScalarLearner

    actor, old, promotion = setup()
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    gate.pause()
    wrapper = ResourceSharedRuntime(old, gate)
    control = controller(wrapper, build_learner=ScalarLearner)
    token = control.capture()
    receipt = promotion.commit(old, prepare(promotion, old))
    promotion.rollback(receipt)
    assert actor.version == "actor-0"
    with pytest.raises(ValueError, match="generation"):
        control.restore(token)
    assert wrapper._runtime is old and not old._retired


@pytest.mark.parametrize("mutation", ["restore_policy", "digest_config", "resource", "clock"])
def test_should_refuse_changed_native_policy_probe_resource_or_clock_without_publication(mutation):
    wrapper, old, _, gate, _, budget = shared()
    gate.pause()

    def build(state):
        learner = PausableLearner(state)
        if mutation == "restore_policy":
            native_restore = learner.restore_state

            def restore(value):
                native_restore(value)
                setattr(learner, "changed", True)

            setattr(learner, "restore_state", restore)
        return learner

    control = controller(
        wrapper,
        build_learner=build,
        policy_digest=lambda learner: digest(getattr(learner, "changed", False)),
    )
    token = control.capture()
    if mutation == "digest_config":
        control._digest = lambda state: digest(state)
    if mutation == "resource":
        gate._resource = lambda: True
    if mutation == "clock":
        budget.clock = lambda: 0.0
    with pytest.raises(ValueError, match="changed|equality"):
        control.restore(token)
    assert wrapper._runtime is old and not old._retired


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_preserve_exact_both_native_continuation_across_complete_handoff(kind):
    import numpy as np
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.core.experience import Experience, ExperiencePermissions, LogicalClock
    from src.core.resource_sharing import SharingLimits
    from test_experience_inbox import make_native_pair

    _, source = make_native_pair(kind)
    actors = []
    for interrupted in (False, True):
        clock = LogicalClock()
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
        runtime = ActorShadowRuntime(
            source,
            actor_version="actor-0",
            candidate_version="candidate-0",
            clock=clock,
            budget=budget,
        )
        gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
        wrapper = ResourceSharedRuntime(runtime, gate)
        for sample, at in (("first", 3), ("future", 5)):
            runtime.record_experience(
                Experience(
                    sample,
                    "e1",
                    1,
                    "actor-0",
                    np.array([[0.3, -0.2], [-0.5, 0.4]]),
                    "train",
                    ExperiencePermissions(training=True),
                )
            )
            runtime.record_label(
                LabelArrival(
                    "label-" + sample, sample, "e1", at, "actor-0", np.array([[1.0], [0.0]])
                )
            )
        actor_before = pickle.dumps(runtime.actor.snapshot_state())
        clock.advance_to(3)
        wrapper.train_ready()
        if interrupted:
            gate.pause()

            def policy(learner):
                return digest(
                    tuple(
                        (n, getattr(learner, n))
                        for n in ("_learning_rate", "_inference_steps", "_inference_learning_rate")
                        if hasattr(learner, n)
                    )
                )

            control = CandidateCheckpointController(
                wrapper,
                build_learner=lambda state: source.fork(),
                state_digest=digest,
                policy_digest=policy,
            )
            runtime = control.restore(control.capture())
            gate.resume()
        clock.advance_to(5)
        wrapper.train_ready()
        assert pickle.dumps(runtime.actor.snapshot_state()) == actor_before
        assert budget.updates_completed == gate.snapshot().admitted_updates == 2
        actors.append((pickle.dumps(runtime.candidate_snapshot().state), runtime.applied_updates))
    assert actors[0] == actors[1]


@pytest.mark.parametrize("mutation", ["gate", "actor", "history", "consolidation_limit"])
def test_should_bind_actual_serving_owner_gate_and_complete_current_histories(mutation):
    from src.app.actor_shadow import StableActor
    from src.app.resource_sharing import ServingPriorityGate

    control, wrapper, old, _, gate, _, _ = prepared()
    token = control.capture()
    if mutation == "gate":
        clone = ServingPriorityGate(gate.snapshot().limits, resource_available=gate._resource)
        clone.pause()
        wrapper._sharing = clone
    if mutation == "actor":
        old._actor = StableActor(PausableLearner(), version="actor-0")
    if mutation == "history":
        old._inbox._labels.pop(("e1", "future"))
        old._inbox._event_ids.remove("label-future")
    if mutation == "consolidation_limit":
        old._consolidation_limit += 1
    with pytest.raises(ValueError, match="changed|stale"):
        control.restore(token)
    assert wrapper._runtime is old and not old._retired
