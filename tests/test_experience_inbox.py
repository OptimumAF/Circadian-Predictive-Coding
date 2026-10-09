"""Arrival, role, ownership and failure boundaries around the shared native step."""

from dataclasses import replace
import pickle

import numpy as np
import pytest

from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic


class RecordingLearner:
    def __init__(self):
        self.calls = []

    def train_batch(self, features: list[str], targets: list[str]) -> TrainingDiagnostic:
        self.calls.append((tuple(features), tuple(targets)))
        features.append("learner mutation")
        targets.append("learner mutation")
        return TrainingDiagnostic("fixture_native_v1", -1.0)

    def predict(self, features: list[str]) -> list[str]:
        return features[:]

    def snapshot_state(self) -> int:
        return len(self.calls)

    def restore_state(self, state: int) -> None:
        del self.calls[state:]


def experience(sample_id="s1", observed_at=1, episode_id="e1"):
    return Experience(
        sample_id,
        episode_id,
        observed_at,
        "actor-0",
        [sample_id],
        "train",
        ExperiencePermissions(training=True),
    )


def label(sample_id="s1", arrived_at=3, episode_id="e1", event_id=None):
    return LabelArrival(
        event_id or "label-" + episode_id + "-" + sample_id,
        sample_id,
        episode_id,
        arrived_at,
        "actor-0",
        ["target-" + sample_id],
    )


def setup(clock=None, budget=None, learner=None, max_experiences=8):
    clock = clock or LogicalClock()
    learner = learner or RecordingLearner()
    budget = budget or ToyBudgetSession(ToyExecutionBudget(max_training_updates=8), lambda: 0.0)
    inbox = ExperienceInbox(
        learner,
        clock=clock,
        budget=budget,
        learner_version="candidate-0",
        max_experiences=max_experiences,
    )
    return inbox, clock, learner, budget


def test_should_wait_for_both_source_and_label_arrival_without_updating():
    inbox, clock, learner, budget = setup()
    inbox.record_experience(experience(observed_at=3))
    inbox.record_label(label(arrived_at=5))
    clock.advance_to(2)
    assert inbox.drain() == ()
    clock.advance_to(4)
    assert inbox.drain() == ()
    assert learner.calls == [] and budget.updates_completed == 0
    clock.advance_to(5)
    (update,) = inbox.drain()
    assert update.sample_id == "s1" and update.applied_at == update.arrived_at == 5
    assert update.actor_version == "actor-0" and update.learner_version == "candidate-0"
    assert budget.updates_completed == 1 and inbox.drain() == ()


@pytest.mark.parametrize("reverse", [False, True])
def test_should_buffer_label_first_delivery_and_order_each_drain_by_arrival_then_identity(reverse):
    inbox, clock, learner, _ = setup()
    records = [
        (experience("z", 1), label("z", 4)),
        (experience("b", 1), label("b", 2)),
        (experience("a", 1), label("a", 2)),
    ]
    for source, target in reversed(records) if reverse else records:
        inbox.record_label(target)
        inbox.record_experience(source)
    clock.advance_to(5)
    assert [r.sample_id for r in inbox.drain()] == ["a", "b", "z"]
    assert [x[0] for x in learner.calls] == [("a",), ("b",), ("z",)]


def test_should_apply_late_delivery_at_current_time_without_reordering_old_updates():
    inbox, clock, _, _ = setup()
    clock.advance_to(5)
    inbox.record_experience(experience("new"))
    inbox.record_label(label("new", 4))
    inbox.drain()
    inbox.record_label(label("old", 2))
    inbox.record_experience(experience("old"))
    (update,) = inbox.drain()
    assert update.arrived_at == 2 and update.applied_at == 5
    assert [x.sample_id for x in inbox.applied_updates] == ["new", "old"]


@pytest.mark.parametrize("when", ["pending", "applied"])
def test_should_reject_duplicate_source_label_and_event_ids_without_extra_updates(when):
    inbox, clock, learner, _ = setup()
    inbox.record_experience(experience())
    inbox.record_label(label())
    if when == "applied":
        clock.advance_to(3)
        inbox.drain()
    with pytest.raises(ValueError, match="duplicate"):
        inbox.record_experience(experience())
    with pytest.raises(ValueError, match="duplicate"):
        inbox.record_label(label())
    with pytest.raises(ValueError, match="duplicate"):
        inbox.record_label(label("other", event_id="label-e1-s1"))
    assert len(learner.calls) == (1 if when == "applied" else 0)


@pytest.mark.parametrize("label_first", [False, True])
def test_should_refuse_version_or_chronology_conflicts_before_registration(label_first):
    inbox, _, learner, _ = setup()
    source, target = experience(observed_at=2), label(arrived_at=3)
    if label_first:
        inbox.record_label(target)
        with pytest.raises(ValueError, match="version"):
            inbox.record_experience(replace(source, model_version="different"))
        inbox.record_experience(source)
    else:
        inbox.record_experience(source)
        with pytest.raises(ValueError, match="version"):
            inbox.record_label(replace(target, model_version="different"))
        with pytest.raises(ValueError, match="before source"):
            inbox.record_label(replace(target, arrived_at=1))
        inbox.record_label(target)
    assert learner.calls == []


def test_should_keep_episodes_separate_and_bound_identity_history_including_consumed_events():
    inbox, clock, learner, _ = setup(max_experiences=2)
    for episode in ["a", "b"]:
        inbox.record_experience(experience(episode_id=episode))
        inbox.record_label(label(episode_id=episode))
    clock.advance_to(3)
    assert [r.episode_id for r in inbox.drain()] == ["a", "b"]
    with pytest.raises(ValueError, match="capacity"):
        inbox.record_experience(experience("other"))
    assert len(learner.calls) == 2


class SealedPayload:
    def __deepcopy__(self, memo):
        raise AssertionError("held-out payload must not be copied")


@pytest.mark.parametrize("role", ["inner_guard", "outer_selection", "final_test"])
def test_should_refuse_held_out_data_before_copy_or_native_call(role):
    inbox, _, learner, _ = setup()
    with pytest.raises(ValueError, match="training permission"):
        inbox.record_experience(
            replace(
                experience(),
                role=role,
                features=SealedPayload(),
                permissions=ExperiencePermissions(evaluation=True),
            )
        )
    with pytest.raises(ValueError, match="train role"):
        inbox.record_label(replace(label(), role=role, targets=SealedPayload()))
    assert learner.calls == []


def test_should_copy_accepted_inputs_and_detach_arguments_given_to_the_learner():
    inbox, clock, learner, _ = setup()
    source, target = experience(), label()
    inbox.record_experience(source)
    inbox.record_label(target)
    source.features.append("caller mutation")
    target.targets.append("caller mutation")
    clock.advance_to(3)
    inbox.drain()
    assert learner.calls == [(("s1",), ("target-s1",))]
    assert source.features == ["s1", "caller mutation"]
    assert target.targets == ["target-s1", "caller mutation"]


def test_should_record_completed_native_work_before_a_post_update_budget_stop():
    times = iter([0.0, 0.0, 2.0])
    budget = ToyBudgetSession(ToyExecutionBudget(max_wall_seconds=1.0), lambda: next(times))
    inbox, clock, learner, _ = setup(budget=budget)
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.advance_to(3)
    with pytest.raises(ToyExecutionStopped):
        inbox.drain()
    assert len(learner.calls) == budget.updates_completed == 1
    assert inbox.applied_updates[0].diagnostic == TrainingDiagnostic("fixture_native_v1", -1.0)
    assert inbox.drain() == ()


def test_should_keep_a_pre_update_refusal_pending_without_calling_the_learner():
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    inbox, clock, learner, _ = setup(budget=budget)
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.advance_to(3)
    for _ in range(2):
        with pytest.raises(ToyExecutionStopped):
            inbox.drain()
    assert learner.calls == [] and inbox.applied_updates == ()


def test_should_stop_after_a_learner_error_that_may_have_partially_changed_state():
    class FailingLearner(RecordingLearner):
        def train_batch(self, features, targets):
            self.calls.append((features, targets))
            raise RuntimeError("partial mutation")

    inbox, clock, learner, _ = setup(learner=FailingLearner())
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.advance_to(3)
    with pytest.raises(RuntimeError, match="partial mutation"):
        inbox.drain()
    with pytest.raises(ValueError, match="stopped"):
        inbox.drain()
    assert len(learner.calls) == 1


def test_should_stop_when_native_partial_work_raises_the_budget_exception_type():
    refusal = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    try:
        refusal.before_update()
    except ToyExecutionStopped as error:
        native_error = error

    class FailingLearner(RecordingLearner):
        def train_batch(self, features, targets):
            self.calls.append((features, targets))
            raise native_error

    inbox, clock, learner, budget = setup(learner=FailingLearner())
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.advance_to(3)
    with pytest.raises(ToyExecutionStopped):
        inbox.drain()
    with pytest.raises(ValueError, match="stopped"):
        inbox.drain()
    assert len(learner.calls) == 1 and budget.updates_completed == 0


def test_should_bound_orphan_label_history_and_refuse_missing_training_permission():
    inbox, _, learner, _ = setup(max_experiences=1)
    inbox.record_label(label())
    with pytest.raises(ValueError, match="capacity"):
        inbox.record_label(label("other"))
    with pytest.raises(ValueError, match="training permission"):
        inbox.record_experience(replace(experience(), permissions=ExperiencePermissions()))
    inbox.record_experience(experience())
    assert learner.calls == []


def test_should_reject_a_backwards_injected_clock_before_any_update():
    class Clock:
        value = 5

        def now(self):
            return self.value

    clock = Clock()
    inbox, _, learner, _ = setup(clock=clock)
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.value = 4
    with pytest.raises(ValueError, match="backwards"):
        inbox.drain()
    assert learner.calls == []


def test_should_reject_nested_drain_before_a_second_native_call():
    class NestedLearner(RecordingLearner):
        def train_batch(self, features, targets):
            inbox.drain()
            return super().train_batch(features, targets)

    inbox, clock, learner, _ = setup(learner=NestedLearner())
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.advance_to(3)
    with pytest.raises(ValueError, match="nested"):
        inbox.drain()
    assert learner.calls == []


def make_native_pair(kind):
    from src.adapters.numpy_learners import BackpropLearner, CircadianLearner
    from src.core.backprop_mlp import BackpropMLP
    from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork

    if kind == "backprop":
        native = BackpropMLP(2, 4, seed=23)
        return native, BackpropLearner(native, learning_rate=0.03)
    circadian = CircadianPredictiveCodingNetwork(2, 4, seed=23, min_hidden_dim=4, max_hidden_dim=4)
    return circadian, CircadianLearner(
        circadian, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
    )


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_use_the_same_arrival_boundary_for_both_native_learners_with_exact_parity(kind):
    features = np.array([[0.3, -0.2], [-0.5, 0.4]])
    targets = np.array([[1.0], [0.0]])
    native, learner = make_native_pair(kind)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    inbox = ExperienceInbox(learner, clock=clock, budget=budget, learner_version="candidate-0")
    original = pickle.dumps(learner.snapshot_state().state)
    inbox.record_experience(replace(experience(), features=features))
    inbox.record_label(replace(label(), targets=targets))
    assert inbox.drain() == () and pickle.dumps(learner.snapshot_state().state) == original
    clock.advance_to(3)
    (update,) = inbox.drain()
    expected = (
        native.train_epoch(features, targets, 0.03)
        if kind == "backprop"
        else native.train_epoch(features, targets, 0.03, 2, 0.2)
    )
    assert update.diagnostic.value == (expected.loss if kind == "backprop" else expected.energy)
    assert pickle.dumps(learner.snapshot_state().state) == pickle.dumps(native.__dict__)
    assert budget.updates_completed == 1
