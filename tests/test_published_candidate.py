"""Real ownership/guard/cursor transitions on scripted native learner ports."""

from copy import copy
from dataclasses import asdict, fields, replace

import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import (
    Experience,
    ExperiencePermissions,
    LabelArrival,
    LogicalClock,
    AppliedExperience,
)
from src.core.inbox_cursor import InboxCursor, GenerationInboxCursor, validate_inbox_cursor
from src.core.learner_ports import TrainingDiagnostic
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import ServingConfiguration
from test_promotion_guard_evaluation import ScalarLearner, guards, make_evaluator, policy
from test_serving_promotion import digest

WORK = dict(owners=0, snapshots=0, trains=0, train_rows=0, predicts=0, predict_rows=0)


class Native(ScalarLearner):
    """Small literal state, no solver/forward/physics/random initialization."""

    def __init__(self, state):
        WORK["owners"] += 1
        assert WORK["owners"] <= 300
        super().__init__(state)

    def fork(self):
        return Native(self.state)

    def snapshot_state(self):
        WORK["snapshots"] += 1
        assert WORK["snapshots"] <= 12000
        return super().snapshot_state()

    def train_batch(self, features, targets):
        WORK["trains"] += 1
        WORK["train_rows"] += len(features)
        assert WORK["trains"] <= 120 and WORK["train_rows"] <= 1800
        self.state[0] = targets[0]
        return TrainingDiagnostic("scripted_no_numeric_work", 0.0)

    def predict(self, x):
        WORK["predicts"] += 1
        WORK["predict_rows"] += 1
        assert WORK["predicts"] <= 400
        return super().predict(x)


def queue(runtime, clock, index, target):
    version = runtime._base_actor_version
    runtime.record_experience(
        Experience(
            f"s{index}",
            f"e{index}",
            clock.now(),
            version,
            [0],
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.record_label(
        LabelArrival(f"l{index}", f"s{index}", f"e{index}", clock.now(), version, [target])
    )


def publish(controller, owner):
    ticket = controller.prepare(owner, *guards(), now=2, training_ids=frozenset())
    return controller.commit(owner, ticket)


@pytest.fixture
def case():
    source = Native([0.5, 0.8])
    actor = PromotableActor(
        source,
        version="a0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=digest,
        metadata={"scope": "literal"},
    )
    clock = LogicalClock(1)
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=4, max_wall_seconds=90), lambda: 0.0
    )
    old = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        actor=actor,
        max_experiences=4,
        max_consolidations=0,
    )
    gate = ServingPriorityGate(SharingLimits(1, 4, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(old, gate)
    queue(old, clock, 0, 0.75)
    assert len(shared.train_ready().updates) == 1
    evaluator = make_evaluator(build_learner=Native)
    controller = ServingPromotionController(
        actor, policy=policy(), evaluator=evaluator, build_learner=Native, state_digest=digest
    )
    receipt = publish(controller, old)
    gate.pause()
    return dict(
        source=source,
        actor=actor,
        clock=clock,
        budget=budget,
        old=old,
        gate=gate,
        shared=shared,
        controller=controller,
        receipt=receipt,
    )


def start(box):
    return box["controller"].start_candidate(box["shared"], box["receipt"], candidate_version="c1")


def test_should_start_from_current_actor_keep_charges_and_deploy_the_next_candidate(case):
    old = case["old"]
    before = old._inbox.capture_cursor()
    actor_before = case["actor"].serving_snapshot()
    gate_before = case["gate"].snapshot()
    new = start(case)
    assert case["shared"]._runtime is new and new.actor is case["actor"]
    assert new._budget is case["budget"] and new._inbox._clock is case["clock"]
    assert new._inbox._capacity == 3 and old._inbox._capacity == 4
    assert new.candidate_snapshot().state == actor_before.model_state == [0.75, 0.8]
    assert (
        new.candidate_snapshot().actor_version == "c0"
        and new.candidate_snapshot().learner_version == "c1"
    )
    assert (
        case["actor"].serving_snapshot() == actor_before and case["gate"].snapshot() == gate_before
    )
    case["gate"].resume()
    queue(new, case["clock"], 1, 0.95)
    result = case["shared"].train_ready()
    assert result.updates[0].update_number == 2
    assert case["budget"].updates_completed == 2 and case["gate"].snapshot().admitted_updates == 2
    assert case["shared"].predict([0]).prediction == [0.75]
    receipt = publish(case["controller"], new)
    assert receipt.actor_version == "c1" and receipt.generation == 2
    assert case["shared"].predict([0]).prediction == [0.95]
    assert old._inbox.capture_cursor() == before  # old total freezes at1, not live2
    assert case["source"].state == [0.5, 0.8]
    with pytest.raises(ValueError, match="retired"):
        old.train_ready()


def test_should_capture_and_checkpoint_generation_offset_without_renumbering_prior_work(case):
    new = start(case)
    initial = new._inbox.capture_cursor()
    assert type(initial) is GenerationInboxCursor and initial.format_version == 3
    assert initial.completed_before_generation == initial.completed_updates == 1
    assert not initial.applied
    case["gate"].resume()
    queue(new, case["clock"], 1, 0.95)
    case["shared"].train_ready()
    case["gate"].pause()
    cursor = new._inbox.capture_cursor()
    assert cursor.applied[0].update_number == cursor.completed_updates == 2
    validate_inbox_cursor(cursor)
    control = CandidateCheckpointController(
        case["shared"],
        build_learner=Native,
        state_digest=digest,
        policy_digest=lambda m: digest("literal"),
    )
    token = control.capture()
    restored = control.restore(token)
    assert restored._inbox.capture_cursor() == cursor
    assert restored._inbox._generation_start_updates == 1
    assert restored._budget is case["budget"]


@pytest.mark.parametrize(
    "fault",
    [
        "copy",
        "receipt",
        "not_paused",
        "private",
        "pending",
        "stopped",
        "role",
        "managed",
        "action",
        "consent",
        "capacity",
        "version",
    ],
)
def test_should_refuse_unsupported_or_unpublished_state_before_builder(case, fault):
    receipt = case["receipt"]
    owner = case["old"]
    if fault == "copy":
        receipt = copy(receipt)
    elif fault == "receipt":
        object.__setattr__(receipt, "generation", 99)
    elif fault == "not_paused":
        case["gate"].resume()
    elif fault == "private":
        owner._candidate.state[0] = 0.9
    elif fault == "pending":
        owner.record_experience(
            Experience(
                "extra", "extra", 1, "a0", [0], "train", ExperiencePermissions(training=True)
            )
        )
    elif fault == "stopped":
        owner._stopped = True
    elif fault == "role":
        owner._training_role = "probe_train"
    elif fault == "managed":
        case["actor"]._payload_registry._lifecycle = object()
    elif fault == "action":
        case["shared"]._action_envelope = object()
    elif fault == "consent":
        owner._inbox._training_guard = lambda *a: True
    elif fault == "capacity":
        owner._inbox._capacity = 1
    elif fault == "version":
        owner._candidate_version = "changed"
    count = WORK["owners"]
    with pytest.raises(ValueError):
        case["controller"].start_candidate(case["shared"], receipt, candidate_version="c1")
    assert WORK["owners"] == count and not owner._retired
    assert case["shared"]._runtime is owner and case["budget"].updates_completed == 1


@pytest.mark.parametrize(
    "fault", ["budget", "clock", "receipt", "gate", "actor", "model", "builder_fail"]
)
def test_should_refuse_callback_changes_keep_old_owner_and_consume_failed_attempt(case, fault):
    controller = case["controller"]
    original = controller._build

    def build(state):
        model = original(state)
        if fault == "budget":
            case["budget"].budget = ToyExecutionBudget(max_training_updates=99)
        elif fault == "clock":
            case["old"]._inbox._clock = LogicalClock(1)
        elif fault == "receipt":
            object.__setattr__(
                next(iter(case["old"]._inbox._applied.values())), "sample_id", "changed"
            )
        elif fault == "gate":
            case["gate"]._admitted = 0
        elif fault == "actor":
            case["actor"]._slot.bundle.cache["changed"] = object()
        elif fault == "model":
            model.restore_state = lambda state: None
            model.state = [0.0, 0.0]
        else:
            raise RuntimeError("lost builder return")
        return model

    controller._build = controller._evaluator._build = build
    with pytest.raises((ValueError, RuntimeError)):
        start(case)
    assert case["shared"]._runtime is case["old"] and not case["old"]._retired
    assert case["budget"].updates_completed == 1
    count = WORK["owners"]
    with pytest.raises(ValueError, match="attempt already consumed"):
        start(case)
    assert WORK["owners"] == count


@pytest.mark.parametrize(
    "change",
    [
        {"format_version": 1},
        {"completed_before_generation": 0},
        {"completed_before_generation": True},
        {"completed_before_generation": 2},
        {"completed_updates": 2},
    ],
)
def test_should_refuse_false_generation_offsets_and_totals(change):
    cursor: GenerationInboxCursor = GenerationInboxCursor(
        3, "c1", 3, (), (), (), 1, False, 1, (), 1
    )
    with pytest.raises(ValueError):
        replace(cursor, **change)


def test_should_preserve_exact_default_cursor_fields_and_format(case):
    cursor = case["old"]._inbox.capture_cursor()
    assert type(cursor) is InboxCursor and cursor.format_version == 1
    assert "completed_before_generation" not in asdict(cursor)
    assert [f.name for f in fields(InboxCursor)] == [
        "format_version",
        "learner_version",
        "capacity",
        "experiences",
        "labels",
        "applied",
        "last_tick",
        "stopped",
        "completed_updates",
        "erased",
    ]


@pytest.mark.parametrize(
    "change", [{"update_number": 1}, {"learner_version": "c0"}, {"actor_version": "a0"}]
)
def test_should_refuse_generation_receipt_renumbering_or_relabeling(change):
    source = Experience("s1", "e1", 1, "c0", [0], "train", ExperiencePermissions(training=True))
    label = LabelArrival("l1", "s1", "e1", 1, "c0", [0.9])
    receipt = AppliedExperience(
        "s1", "e1", "l1", "c0", "c1", 1, 1, 1, 2, TrainingDiagnostic("literal", 0.0)
    )
    cursor: GenerationInboxCursor = GenerationInboxCursor(
        3, "c1", 3, (source,), (label,), (receipt,), 1, False, 2, (), 1
    )
    validate_inbox_cursor(cursor)
    with pytest.raises(ValueError):
        replace(cursor, applied=(replace(receipt, **change),))


def test_should_refuse_busy_actor_at_final_publication_without_partial_retirement(case):
    controller = case["controller"]
    original = controller._build
    # Existing actor lock is nonreentrant. Hold it only at final publication,
    # after ordinary state checks, by intercepting the nonblocking lease port.
    from contextlib import contextmanager

    @contextmanager
    def busy():
        with case["actor"]._read_gate:
            with PromotableActor._payload_exclusive(case["actor"]):
                yield

    case["actor"]._payload_exclusive = busy
    with pytest.raises(ValueError, match="busy"):
        start(case)
    assert not case["old"]._retired and case["shared"]._runtime is case["old"]
    assert controller._build is original
