"""Disposable probe wake keeps its role while fixed history remains TRAIN-only."""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle

import numpy as np
import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.experience_inbox import ExperienceInbox
from src.app.stream_rehearsal import StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from test_coverage_stream_rehearsal import Model, SCHEMA
from test_demonstration_episode import episode
from test_fixed_training_stream_rehearsal import fixed_bank, policy


def probe_batches(count=4):
    result = []
    for index in range(count):
        packet = episode(
            dim=2,
            steps=2,
            episode_id=f"fresh-{index}",
            context="F",
            role="probe_train",
        )
        features = np.array([row.observation for row in packet.transitions], dtype=np.float64)
        targets = np.array([[row.action] for row in packet.transitions], dtype=np.float64)
        result.append(
            AttributionBatch(
                "probe_train",
                "F",
                tuple(f"{packet.episode_id}:step:{row.step}" for row in packet.transitions),
                features,
                targets,
                packet,
            )
        )
    return tuple(result)


def setup(*, count=4, max_updates=None, helper_role="probe_train", **faults):
    source = Model()
    source.control.update(faults)
    wake = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=count if max_updates is None else max_updates,
            max_wall_seconds=10,
        ),
        lambda: 0.0,
    )
    replay = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=8, max_wall_seconds=10), lambda: 0.0
    )
    helper = StreamRehearsal(
        source,
        wake,
        policy(),
        replay_budget=replay,
        total_updates=count,
        input_dim=2,
        episode_schema=SCHEMA,
        fixed_training_episodes=fixed_bank(),
        wake_role=helper_role,
    )
    return source, wake, replay, helper


def stream(source, wake, helper, training=None, **changes):
    selected = probe_batches() if training is None else training
    options = dict(
        checkpoint_updates=(0, len(selected)),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
        training_role="probe_train",
    )
    options.update(changes)
    return AttributionStream(source, selected, (), **options)


def test_should_keep_probe_roles_and_complete_old_train_bank_through_actual_updates():
    source, wake, replay, helper = setup(mutate_inputs=True)
    training = probe_batches()
    original = pickle.dumps(training)
    run = stream(source, wake, helper, training)
    run.run_until(4)
    result = run.finish()

    assert pickle.dumps(training) == original
    assert result["evaluation_scope"] == "training_only"
    assert result["training_role"] == "probe_train"
    assert result["training_examples"] == 8
    assert wake.updates_completed == 4 and wake.replay_examples_completed == 0
    assert replay.updates_completed == 0 and replay.replay_examples_completed == 8
    assert source.state[0] == run._runtime.actor.snapshot_state().state[0] == 0
    assert run._runtime._candidate.snapshot_state()[0] == 8
    assert len(source.control["calls"]) == 8
    assert all(batch.role == batch.episode.role == "probe_train" for batch in run._training)
    inbox = run._runtime._inbox
    assert {record.role for record in inbox._experiences.values()} == {"probe_train"}
    assert {record.role for record in inbox._labels.values()} == {"probe_train"}
    for receipt in inbox.applied_updates:
        key = receipt.episode_id, receipt.sample_id
        assert inbox._experiences[key].role == inbox._labels[key].role == "probe_train"
        assert receipt.episode_id in {batch.episode.episode_id for batch in training}
        assert receipt.event_id == inbox._labels[key].event_id
    assert len(inbox.applied_updates) == 4
    expected = [asdict(packet) for packet in fixed_bank()]
    for event in result["rehearsal"]["events"]:
        assert event["selected_episodes"] == expected
        assert event["charged_exposures"] == 4
        assert event["attempted_updates"] == event["returned_updates"] == 2
    assert result["rehearsal"]["fixed_bank"]["episode_ids"] == ["old-A", "old-B"]


def test_should_resume_original_probe_scope_budget_clock_and_preceding_replay_success():
    source, wake, replay, helper = setup()
    run = stream(source, wake, helper)
    run.run_until(2)
    previous = deepcopy(helper.report["events"][0])
    original = (id(wake), id(replay), wake.started_at, replay.started_at, id(run._clock))
    pause = run.pause()
    assert pause._snapshot["authority"]["training_role"] == "probe_train"
    cursor = pause._snapshot["inbox_cursor"]
    assert {record.role for record in cursor.experiences} == {"probe_train"}
    assert {record.role for record in cursor.labels} == {"probe_train"}
    run.resume(pause)
    assert helper.report["events"][0] == previous
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 4
    run.run_until(4)
    result = run.finish()
    assert original == (id(wake), id(replay), wake.started_at, replay.started_at, id(run._clock))
    assert result["training_role"] == "probe_train"
    assert result["rehearsal"]["events"][0] == previous
    assert wake.updates_completed == 4 and replay.replay_examples_completed == 8
    assert len(run._runtime.applied_consolidations) == 2
    assert len(source.control["calls"]) == 8


@pytest.mark.parametrize("role", ["train", "final_test", "probe_test"])
def test_should_refuse_mixed_complete_episode_roles_before_any_work(role):
    source, wake, replay, helper = setup()
    training = list(probe_batches())
    packet = replace(training[1].episode, role=role)
    training[1] = replace(training[1], role=role, episode=packet)
    with pytest.raises(ValueError):
        stream(source, wake, helper, tuple(training))
    assert source.control == {"calls": [], "forks": 0}
    assert wake.updates_completed == replay.replay_examples_completed == 0


@pytest.mark.parametrize("fault", ["scope", "budget", "too_many", "metadata", "observer"])
def test_should_refuse_unbounded_or_nonisolated_probe_stream_before_forks(fault):
    count = 17 if fault == "too_many" else 4
    source, wake, replay, helper = setup(count=count, max_updates=5 if fault == "budget" else None)
    training = probe_batches(count)
    changes = {}
    if fault == "scope":
        changes["training_only"] = False
    elif fault == "metadata":
        training = tuple(replace(batch, episode=None) for batch in training)
    elif fault == "observer":
        changes["observe_checkpoint"] = lambda *_: pytest.fail("probe observer ran")
    with pytest.raises(ValueError):
        stream(source, wake, helper, training, **changes)
    assert source.control == {"calls": [], "forks": 0}
    assert wake.updates_completed == replay.replay_examples_completed == 0


def test_should_refuse_probe_wake_bound_to_ordinary_train_rehearsal():
    source, wake, replay, helper = setup(helper_role="train")
    with pytest.raises(ValueError):
        stream(source, wake, helper)
    assert source.control == {"calls": [], "forks": 0}
    assert wake.updates_completed == replay.replay_examples_completed == 0


def test_should_keep_ordinary_stream_default_from_admitting_probe_training():
    source, wake, replay, helper = setup()
    with pytest.raises(ValueError):
        AttributionStream(
            source,
            probe_batches(),
            (),
            checkpoint_updates=(0, 4),
            label_delay=1,
            budget=wake,
            rehearsal=helper,
            training_only=True,
        )
    assert source.control == {"calls": [], "forks": 0}
    assert wake.updates_completed == replay.replay_examples_completed == 0


def test_should_refuse_final_evaluation_batches_in_probe_mode_before_any_fork():
    source, wake, replay, helper = setup()
    final = probe_batches(1)[0]
    packet = replace(final.episode, episode_id="final-only", role="final_test")
    final = replace(
        final,
        role="final_test",
        episode=packet,
        row_ids=tuple(f"final-only:step:{row.step}" for row in packet.transitions),
    )
    with pytest.raises(ValueError):
        AttributionStream(
            source,
            probe_batches(),
            (final,),
            checkpoint_updates=(0, 4),
            label_delay=1,
            budget=wake,
            rehearsal=helper,
            training_only=True,
            training_role="probe_train",
        )
    assert source.control == {"calls": [], "forks": 0}
    assert wake.updates_completed == replay.replay_examples_completed == 0


@pytest.mark.parametrize("fault", ["authority", "source", "label"])
def test_should_refuse_relabelled_pause_evidence_and_keep_preceding_paid_work(fault):
    source, wake, replay, helper = setup()
    run = stream(source, wake, helper)
    run.run_until(2)
    previous = deepcopy(helper.report["events"][0])
    pause = run.pause()
    if fault == "authority":
        pause._snapshot["authority"]["training_role"] = "train"
    else:
        cursor = pause._snapshot["inbox_cursor"]
        record = cursor.experiences[0] if fault == "source" else cursor.labels[0]
        object.__setattr__(record, "role", "train")
    with pytest.raises(ValueError):
        run.resume(pause)
    assert helper.report["events"][0] == previous
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 4
    assert len(source.control["calls"]) == 4


@pytest.mark.parametrize("owner", ["stream", "runtime", "inbox", "rehearsal"])
def test_should_refuse_changed_role_authority_before_more_native_work(owner):
    source, wake, replay, helper = setup()
    run = stream(source, wake, helper)
    run.run_until(2)
    preceding = deepcopy(run._runtime.applied_consolidations)
    selected = {
        "stream": run,
        "runtime": run._runtime,
        "inbox": run._runtime._inbox,
        "rehearsal": helper,
    }[owner]
    setattr(selected, "_wake_role" if owner == "rehearsal" else "_training_role", "train")
    with pytest.raises(ValueError):
        run.run_until(4)
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 4
    assert len(source.control["calls"]) == 4
    assert run._runtime.applied_consolidations == preceding


def test_should_refuse_changed_future_probe_episode_before_more_work():
    source, wake, replay, helper = setup()
    run = stream(source, wake, helper)
    run.run_until(2)
    object.__setattr__(run._training[2].episode, "role", "train")
    with pytest.raises(ValueError):
        run.run_until(4)
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 4
    assert len(source.control["calls"]) == 4


class SealedPayload:
    def __deepcopy__(self, memo):
        pytest.fail("refused role payload was copied")


@pytest.mark.parametrize("probe_mode", [False, True])
@pytest.mark.parametrize("role", ["inner_guard", "outer_selection", "final_test", "probe_test"])
def test_should_refuse_reserved_labels_in_every_inbox_mode_before_payload_access(probe_mode, role):
    source = Model()
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    inbox = ExperienceInbox(
        source,
        clock=clock,
        budget=budget,
        learner_version="candidate",
        max_experiences=1,
        **({"training_role": "probe_train"} if probe_mode else {}),
    )
    with pytest.raises(ValueError):
        inbox.record_label(
            LabelArrival("label", "sample", "episode", 0, "actor", SealedPayload(), role)
        )
    assert source.control["calls"] == [] and budget.updates_completed == 0


@pytest.mark.parametrize("probe_mode", [False, True])
def test_should_keep_opposite_training_role_out_of_inbox_and_runtime(probe_mode):
    configured = "probe_train" if probe_mode else "train"
    rejected = "train" if probe_mode else "probe_train"
    source = Model()
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor",
        candidate_version="candidate",
        clock=clock,
        budget=budget,
        max_experiences=1,
        training_role=configured,
    )
    with pytest.raises(ValueError):
        runtime.record_experience(
            Experience(
                "sample",
                "episode",
                0,
                "actor",
                SealedPayload(),
                rejected,
                ExperiencePermissions(training=True),
            )
        )
    with pytest.raises(ValueError):
        runtime.record_label(
            LabelArrival("label", "sample", "episode", 0, "actor", SealedPayload(), rejected)
        )
    assert source.control["calls"] == [] and budget.updates_completed == 0


@pytest.mark.parametrize("field", ["replay", "evaluation"])
def test_should_refuse_probe_permission_escalation_before_any_payload_access(field):
    permissions = {"training": True, field: True}
    with pytest.raises(ValueError):
        Experience(
            "sample",
            "episode",
            0,
            "actor",
            SealedPayload(),
            "probe_train",
            ExperiencePermissions(**permissions),
        )


def test_should_keep_default_capture_fields_and_refuse_probe_managed_enrollment():
    from src.app.managed_experience import ManagedExperienceOwner
    from src.app.managed_record_capture import INBOX_FIELDS, RUNTIME_FIELDS
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.core.data_lifecycle import LifecycleLimits
    from src.core.resource_sharing import SharingLimits

    for role in ("train", "probe_train"):
        source = Model()
        clock = LogicalClock()
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
        runtime = ActorShadowRuntime(
            source,
            actor_version="actor",
            candidate_version="candidate",
            clock=clock,
            budget=budget,
            max_experiences=1,
            training_role=role,
        )
        if role == "train":
            assert vars(runtime).keys() == RUNTIME_FIELDS
            assert vars(runtime._inbox).keys() == INBOX_FIELDS
        else:
            shared = ResourceSharedRuntime(
                runtime,
                gate=ServingPriorityGate(SharingLimits(1, 1, 1), resource_available=lambda: True),
            )
            with pytest.raises(ValueError, match="TRAIN owners"):
                ManagedExperienceOwner(shared, limits=LifecycleLimits(1, 1))
            assert runtime._inbox._registration_guard is None
            assert runtime._inbox._training_guard is None
        assert source.control["calls"] == [] and budget.updates_completed == 0
