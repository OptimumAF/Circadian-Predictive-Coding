"""Deterministic actor availability at partial candidate mutation boundaries."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import pickle
from threading import Event

import numpy as np
import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.actor_ports import ConsolidatedState
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic


class PausableLearner:
    """Share test synchronization only; each fork owns its mutable model list."""

    def __init__(self, value=None, entered=None, release=None, phase=None):
        self.value = [2, 2] if value is None else list(value)
        self.entered, self.release, self.phase = entered, release, phase

    def fork(self):
        return PausableLearner(self.value, self.entered, self.release, self.phase)

    def _pause(self, phase):
        if self.phase == phase:
            self.entered.set()
            assert self.release.wait(3), "test did not release native operation"

    def train_batch(self, features, targets):
        self.value[0] = targets[0]
        self._pause("train")
        self.value[1] = targets[1]
        return TrainingDiagnostic("fixture_native_v1", float(sum(self.value)))

    def predict(self, features):
        features.append("predict mutation")
        return self.value

    def snapshot_state(self):
        return self.value[:]

    def restore_state(self, state):
        self.value[0] = state[0]
        self._pause("restore")
        if self.phase == "restore_error":
            raise RuntimeError("partially restored")
        self.value[1] = state[1]


def setup(source=None, budget=None, max_consolidations=2):
    source = source or PausableLearner()
    budget = budget or ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=4,
        max_consolidations=max_consolidations,
    )
    return runtime, clock, source, budget


def queue(runtime, sample_id="s1", targets=None):
    runtime.record_experience(
        Experience(
            sample_id, "e1", 1, "actor-0", ["x"], "train", ExperiencePermissions(training=True)
        )
    )
    runtime.record_label(
        LabelArrival("label-" + sample_id, sample_id, "e1", 3, "actor-0", targets or [8, 8])
    )


def consolidate(state):
    state[:] = [9, 9]
    return ConsolidatedState(state, TrainingDiagnostic("fixture_consolidation_v1", 18.0))


@pytest.mark.parametrize("phase", ["train", "restore"])
def test_should_serve_complete_actor_version_while_candidate_is_partially_mutated(phase):
    entered, release = Event(), Event()
    runtime, clock, source, _ = setup(
        PausableLearner(entered=entered, release=release, phase=phase)
    )
    queue(runtime)
    clock.advance_to(3)
    before = runtime.actor.snapshot_state()
    with ThreadPoolExecutor(max_workers=4) as pool:
        future = pool.submit(
            runtime.train_ready
            if phase == "train"
            else lambda: runtime.consolidate("sleep-1", consolidate)
        )
        try:
            assert entered.wait(3), "candidate never reached partial native mutation"
            reads = [pool.submit(runtime.actor.predict, ["input"]) for _ in range(3)]
            outputs = [read.result(timeout=3) for read in reads]
            assert all(
                out.actor_version == "actor-0" and out.prediction == [2, 2] for out in outputs
            )
            assert runtime.actor.snapshot_state() == before
            with pytest.raises(ValueError, match="busy"):
                runtime.candidate_snapshot()
            with pytest.raises(ValueError, match="busy"):
                runtime.train_ready()
        finally:
            release.set()
        future.result(timeout=3)
    assert runtime.actor.snapshot_state() == before and source.value == [2, 2]
    assert runtime.candidate_snapshot().state == ([8, 8] if phase == "train" else [9, 9])


def test_should_detach_actor_predictions_inputs_and_all_returned_states():
    runtime, _, source, _ = setup()
    features = ["input"]
    result = runtime.actor.predict(features)
    result.prediction[:] = [99, 99]
    actor = runtime.actor.snapshot_state()
    actor.state[:] = [99, 99]
    candidate = runtime.candidate_snapshot()
    candidate.state[:] = [99, 99]
    source.value[:] = [99, 99]
    assert features == ["input"]
    assert runtime.actor.predict(features).prediction == [2, 2]
    assert runtime.candidate_snapshot().state == [2, 2]
    assert runtime.actor.version == "actor-0" and candidate.learner_version == "candidate-0"


def test_should_preserve_arrival_duplicate_role_and_version_rules_on_shadow_training():
    runtime, clock, _, budget = setup()
    queue(runtime)
    clock.advance_to(2)
    assert runtime.train_ready() == () and runtime.candidate_snapshot().state == [2, 2]
    with pytest.raises(ValueError, match="duplicate"):
        queue(runtime)
    held = Experience(
        "held", "e1", 1, "actor-0", ["x"], "final_test", ExperiencePermissions(evaluation=True)
    )
    with pytest.raises(ValueError, match="training permission"):
        runtime.record_experience(held)
    runtime.record_label(LabelArrival("other-label", "other", "e1", 3, "wrong-version", [8, 8]))
    with pytest.raises(ValueError, match="version"):
        runtime.record_experience(
            replace(
                held,
                sample_id="other",
                role="train",
                permissions=ExperiencePermissions(training=True),
            )
        )
    clock.advance_to(3)
    (update,) = runtime.train_ready()
    assert update.learner_version == "candidate-0" and update.actor_version == "actor-0"
    assert budget.updates_completed == 1 and runtime.actor.predict([]).prediction == [2, 2]


@pytest.mark.parametrize("bad", [True, -1, 0.5])
def test_should_reject_invalid_consolidation_quota(bad):
    with pytest.raises(ValueError, match="max_consolidations"):
        setup(max_consolidations=bad)


def test_should_bound_lifetime_consolidation_ids_attempts_and_detach_callback_state():
    runtime, _, _, _ = setup()
    retained = []

    def operation(state):
        retained.append(state)
        return consolidate(state)

    first = runtime.consolidate("sleep-1", operation)
    retained[0][:] = [99, 99]
    assert first.actor_version == "actor-0" and first.learner_version == "candidate-0"
    assert runtime.candidate_snapshot().state == [9, 9]
    with pytest.raises(ValueError, match="duplicate"):
        runtime.consolidate("sleep-1", operation)

    def failed(state):
        state[:] = [99, 99]
        raise RuntimeError("detached failure")

    with pytest.raises(RuntimeError, match="detached failure"):
        runtime.consolidate("sleep-2", failed)
    with pytest.raises(ValueError, match="quota"):
        runtime.consolidate("sleep-3", operation)
    assert len(runtime.applied_consolidations) == 1
    assert runtime.candidate_snapshot().consolidation_attempts == 2
    assert runtime.candidate_snapshot().state == [9, 9]


def test_should_reject_nested_candidate_operations_without_blocking_actor_reads():
    runtime, _, _, _ = setup()

    def operation(state):
        assert runtime.actor.predict([]).prediction == [2, 2]
        with pytest.raises(ValueError, match="busy"):
            runtime.consolidate("nested", consolidate)
        with pytest.raises(ValueError, match="busy"):
            queue(runtime)
        return consolidate(state)

    runtime.consolidate("sleep-1", operation)
    assert runtime.candidate_snapshot().consolidation_attempts == 1


def test_should_stop_candidate_after_partial_restore_failure_but_keep_actor_available():
    runtime, _, _, _ = setup(PausableLearner(phase="restore_error"))
    with pytest.raises(RuntimeError, match="partially restored"):
        runtime.consolidate("sleep-1", consolidate)
    for action in [runtime.train_ready, lambda: runtime.consolidate("sleep-2", consolidate)]:
        with pytest.raises(ValueError, match="stopped"):
            action()
    assert runtime.actor.predict([]).prediction == [2, 2]
    assert runtime.applied_consolidations == ()


def test_should_propagate_uncertain_wake_failure_to_candidate_consolidation_gate():
    class FailingLearner(PausableLearner):
        def fork(self):
            return FailingLearner(self.value)

        def train_batch(self, features, targets):
            self.value[0] = 99
            raise RuntimeError("partial wake")

    runtime, clock, _, _ = setup(FailingLearner())
    queue(runtime)
    clock.advance_to(3)
    with pytest.raises(RuntimeError, match="partial wake"):
        runtime.train_ready()
    with pytest.raises(ValueError, match="stopped"):
        runtime.consolidate("sleep-1", consolidate)
    assert runtime.actor.predict([]).prediction == [2, 2]


def test_should_stop_after_partial_wake_cancellation_without_hiding_the_exception():
    class Cancelled(BaseException):
        pass

    class CancelledLearner(PausableLearner):
        def fork(self):
            return CancelledLearner(self.value)

        def train_batch(self, features, targets):
            self.value[0] = 99
            raise Cancelled("native cancelled")

    runtime, clock, _, _ = setup(CancelledLearner())
    queue(runtime)
    clock.advance_to(3)
    with pytest.raises(Cancelled, match="native cancelled"):
        runtime.train_ready()
    with pytest.raises(ValueError, match="stopped"):
        runtime.consolidate("sleep-1", consolidate)
    assert runtime.actor.predict([]).prediction == [2, 2]


def test_should_charge_malformed_transform_results_without_changing_candidate_state():
    runtime, _, _, _ = setup(max_consolidations=1)
    with pytest.raises(ValueError, match="ConsolidatedState"):
        runtime.consolidate("sleep-1", lambda state: state)
    assert runtime.candidate_snapshot().state == [2, 2]
    with pytest.raises(ValueError, match="duplicate"):
        runtime.consolidate("sleep-1", consolidate)
    with pytest.raises(ValueError, match="quota"):
        runtime.consolidate("sleep-2", consolidate)


def test_should_deny_zero_consolidation_quota_and_invalid_events_before_callbacks():
    runtime, _, _, _ = setup(max_consolidations=0)

    def forbidden(state):
        raise AssertionError("callback must not be reached")

    with pytest.raises(ValueError, match="quota"):
        runtime.consolidate("sleep-1", forbidden)
    with pytest.raises(ValueError, match="event_id"):
        runtime.consolidate("", forbidden)
    with pytest.raises(ValueError, match="callable"):
        runtime.consolidate("sleep-2", None)
    assert runtime.candidate_snapshot().consolidation_attempts == 0


def test_should_refuse_one_cached_fork_reused_for_both_actor_and_candidate():
    class CachedFork(PausableLearner):
        cached = PausableLearner()

        def fork(self):
            return self.cached

    with pytest.raises(ValueError, match="distinct"):
        setup(CachedFork())


def test_should_keep_pre_update_budget_stop_pending_and_actor_usable():
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    runtime, clock, _, _ = setup(budget=budget)
    queue(runtime)
    clock.advance_to(3)
    for _ in range(2):
        with pytest.raises(ToyExecutionStopped):
            runtime.train_ready()
    assert runtime.applied_updates == ()
    assert runtime.candidate_snapshot().state == [2, 2]
    assert runtime.actor.predict([]).prediction == [2, 2]


@pytest.mark.parametrize("phase", ["train", "consolidate"])
def test_should_record_complete_work_before_a_late_budget_stop(phase):
    class Clock:
        time = 0.0

        def __call__(self):
            return self.time

    wall = Clock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_wall_seconds=1.0), wall)
    runtime, clock, _, _ = setup(budget=budget)
    if phase == "train":

        class LateTargets(list):
            def __getitem__(self, index):
                wall.time = 2.0
                return super().__getitem__(index)

        queue(runtime, targets=LateTargets([8, 8]))
        clock.advance_to(3)
        action = runtime.train_ready
    else:

        def late(state):
            wall.time = 2.0
            return consolidate(state)

        action = lambda: runtime.consolidate("sleep-1", late)
    with pytest.raises(ToyExecutionStopped):
        action()
    assert runtime.actor.predict([]).prediction == [2, 2]
    assert len(runtime.applied_updates if phase == "train" else runtime.applied_consolidations) == 1
    assert runtime.candidate_snapshot().state == ([8, 8] if phase == "train" else [9, 9])


def test_should_refuse_a_consolidation_precheck_without_consuming_its_id_or_quota():
    class Clock:
        time = 0.0

        def __call__(self):
            return self.time

    wall = Clock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_wall_seconds=1.0), wall)
    runtime, _, _, _ = setup(budget=budget)
    wall.time = 2.0
    with pytest.raises(ToyExecutionStopped):
        runtime.consolidate("sleep-1", consolidate)
    assert runtime.candidate_snapshot().consolidation_attempts == 0
    assert runtime.actor.predict([]).prediction == [2, 2]


@pytest.mark.parametrize("phase", ["pre", "train", "consolidate"])
def test_should_enforce_attached_sampled_rss_at_native_boundaries(phase):
    from src.shared.process_memory import ProcessRssSampler

    measured = [100]
    with ProcessRssSampler(interval_seconds=60.0, read_rss_bytes=lambda: measured[0]) as sampler:
        budget = ToyBudgetSession(ToyExecutionBudget(max_process_rss_bytes=150), lambda: 0.0)
        budget.attach_memory(sampler)
        runtime, clock, _, _ = setup(budget=budget)
        if phase == "consolidate":

            def late(state):
                measured[0] = 200
                return consolidate(state)

            action = lambda: runtime.consolidate("sleep-1", late)
        else:

            class LateTargets(list):
                def __getitem__(self, index):
                    measured[0] = 200
                    return super().__getitem__(index)

            queue(runtime, targets=LateTargets([8, 8]))
            clock.advance_to(3)
            if phase == "pre":
                measured[0] = 200
            action = runtime.train_ready
        with pytest.raises(ToyExecutionStopped) as caught:
            action()
        assert caught.value.stop.reason == "max_process_rss_bytes"
        assert runtime.actor.predict([]).prediction == [2, 2]
        expected = 0 if phase == "pre" else 1
        assert (
            len(
                runtime.applied_consolidations
                if phase == "consolidate"
                else runtime.applied_updates
            )
            == expected
        )


@pytest.mark.parametrize(
    "limits", [{"max_hidden_width": 4}, {"max_replay_examples": 2}, {"max_process_rss_bytes": 1024}]
)
def test_should_refuse_uncomposed_resource_limits_before_forking(limits):
    class NoFork(PausableLearner):
        def fork(self):
            raise AssertionError("unsupported resource limit must precede fork")

    with pytest.raises(ValueError, match="width|replay|RSS"):
        setup(NoFork(), ToyBudgetSession(ToyExecutionBudget(**limits), lambda: 0.0))


def test_should_reject_nonowned_forks_and_equal_or_empty_version_names():
    class SameFork(PausableLearner):
        def fork(self):
            return self

    with pytest.raises(ValueError, match="distinct"):
        setup(SameFork())
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    for actor, candidate in [("", "candidate-0"), ("same", "same")]:
        with pytest.raises(ValueError):
            ActorShadowRuntime(
                PausableLearner(),
                actor_version=actor,
                candidate_version=candidate,
                clock=LogicalClock(),
                budget=budget,
            )


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_fork_complete_native_policy_and_train_only_the_shadow_with_exact_state_parity(kind):
    from test_experience_inbox import make_native_pair

    features = np.array([[0.3, -0.2], [-0.5, 0.4]])
    targets = np.array([[1.0], [0.0]])
    native, learner = make_native_pair(kind)
    learner.train_batch(features, targets)
    if kind == "backprop":
        native.train_epoch(features, targets, 0.03)
    else:
        native.train_epoch(features, targets, 0.03, 2, 0.2)
    source_bytes = pickle.dumps(learner.snapshot_state())
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    runtime = ActorShadowRuntime(
        learner,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
    )
    actor_bytes = pickle.dumps(runtime.actor.snapshot_state())
    runtime.record_experience(
        Experience(
            "s1", "e1", 1, "actor-0", features, "train", ExperiencePermissions(training=True)
        )
    )
    runtime.record_label(LabelArrival("label-1", "s1", "e1", 3, "actor-0", targets))
    clock.advance_to(3)
    (update,) = runtime.train_ready()
    expected = (
        native.train_epoch(features, targets, 0.03)
        if kind == "backprop"
        else native.train_epoch(features, targets, 0.03, 2, 0.2)
    )
    assert update.diagnostic.value == (expected.loss if kind == "backprop" else expected.energy)
    assert pickle.dumps(runtime.candidate_snapshot().state.state) == pickle.dumps(native.__dict__)
    assert pickle.dumps(learner.snapshot_state()) == source_bytes
    assert pickle.dumps(runtime.actor.snapshot_state()) == actor_bytes
    np.testing.assert_array_equal(
        runtime.actor.predict(features).prediction, learner.predict(features)
    )
    candidate_bytes = pickle.dumps(runtime.candidate_snapshot())
    learner.train_batch(features, targets)
    assert pickle.dumps(runtime.actor.snapshot_state()) == actor_bytes
    assert pickle.dumps(runtime.candidate_snapshot()) == candidate_bytes


def test_should_apply_real_native_cpc_consolidation_to_detached_shadow_state_only():
    from src.adapters.numpy_learners import CircadianLearner
    from src.core.circadian_predictive_coding import (
        CircadianConfig,
        CircadianPredictiveCodingNetwork,
    )

    config = CircadianConfig(
        sleep_mode="components",
        sleep_enable_split=False,
        sleep_enable_prune=False,
        sleep_enable_replay=False,
        sleep_enable_homeostasis=True,
        homeostatic_downscale_factor=0.8,
    )
    native = CircadianPredictiveCodingNetwork(
        2, 4, seed=23, circadian_config=config, min_hidden_dim=4, max_hidden_dim=4
    )
    learner = CircadianLearner(
        native, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
    )
    runtime, _, _, _ = setup(source=learner)
    before = pickle.dumps(runtime.candidate_snapshot().state)
    actor_before = pickle.dumps(runtime.actor.snapshot_state())
    control = deepcopy(native)
    expected = control.sleep_event(max_replay_examples=0, max_hidden_width=4)
    assert expected.performed
    entered, release = Event(), Event()

    def native_sleep(snapshot):
        model = deepcopy(native)
        model.restore_state(snapshot)
        result = model.sleep_event(max_replay_examples=0, max_hidden_width=4)
        assert result.performed
        entered.set()
        assert release.wait(3), "test did not release detached native sleep"
        return ConsolidatedState(
            model.snapshot_state(),
            TrainingDiagnostic("fixture_native_sleep_performed_v1", float(result.performed)),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(runtime.consolidate, "sleep-1", native_sleep)
        try:
            assert entered.wait(3)
            served = pool.submit(runtime.actor.predict, np.zeros((1, 2))).result(timeout=3)
            assert served.actor_version == "actor-0"
            np.testing.assert_array_equal(served.prediction, learner.predict(np.zeros((1, 2))))
            assert pickle.dumps(runtime.actor.snapshot_state()) == actor_before
        finally:
            release.set()
        future.result(timeout=3)
    after = runtime.candidate_snapshot().state
    assert pickle.dumps(after) != before and pickle.dumps(after.state) == pickle.dumps(
        control.__dict__
    )
    assert after.state["_sleep_events"] == 1
    np.testing.assert_array_equal(
        after.state["weight_input_hidden"], native.weight_input_hidden * 0.8
    )
    assert not np.array_equal(after.state["weight_input_hidden"], native.weight_input_hidden)
    assert pickle.dumps(runtime.actor.snapshot_state()) == actor_before
    assert pickle.dumps(learner.snapshot_state().state) == pickle.dumps(native.__dict__)
