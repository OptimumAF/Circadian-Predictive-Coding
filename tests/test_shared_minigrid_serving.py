"""Real sharing/runtime/actor ports with scripted models and literal game frames."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import pickle
from threading import Event
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.adapters.minigrid_serving import run_promotable_minigrid_episode
from src.adapters.promotable_minigrid_binding import bind_promotable_minigrid
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import _same_value
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import ServingConfiguration
from test_affine_reference_readout import literal_state
from test_context_linear_causal_games import PhaseEnvironment
from test_minigrid_learning_pilot import Environment

WORK = dict(
    owners=0,
    snapshots=0,
    trains=0,
    train_rows=0,
    predicts=0,
    predict_rows=0,
    episodes=0,
    steps=0,
    forbidden=0,
)


def digest(value):
    return sha256(pickle.dumps(value)).hexdigest()


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = {}
    init = ContextLinearLearner.__init__
    snapshot = ContextLinearLearner.snapshot_state

    def empty(owner):
        WORK["owners"] += 1
        assert WORK["owners"] <= 160
        init(owner)

    def read(owner):
        WORK["snapshots"] += 1
        assert WORK["snapshots"] <= 6000
        return snapshot(owner)

    def train(owner, x, y):
        WORK["trains"] += 1
        WORK["train_rows"] += len(x)
        assert WORK["trains"] <= 24 and WORK["train_rows"] <= 216
        callback = work.get("on_train")
        if callback:
            callback(owner)
        owner._state = literal_state(owner._state.completed_updates + 1)
        return TrainingDiagnostic("literal_no_fit", 0.0)

    def predict(owner, x):
        WORK["predicts"] += 1
        WORK["predict_rows"] += len(x)
        assert WORK["predicts"] <= 128 and WORK["predict_rows"] <= 512
        callback = work.get("on_predict")
        if callback:
            callback(owner)
        if work.get("predict_error"):
            raise RuntimeError("lost scripted prediction return")
        result = np.zeros((len(x), 7), dtype=np.float64)
        result[:, 2] = 1
        return result

    def forbidden(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("genuine numerical or physical work forbidden")

    class LiteralEnvironment(PhaseEnvironment):
        def __init__(self, *args, **kwargs):
            WORK["episodes"] += 1
            assert WORK["episodes"] <= 24
            super().__init__(*args, **kwargs)

        def step(self, action):
            WORK["steps"] += 1
            assert WORK["steps"] <= 64
            return super().step(action)

    monkeypatch.setattr(ContextLinearLearner, "__init__", empty)
    monkeypatch.setattr(ContextLinearLearner, "snapshot_state", read)
    monkeypatch.setattr(ContextLinearLearner, "train_batch", train)
    monkeypatch.setattr(ContextLinearLearner, "predict", predict)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", LiteralEnvironment)
    source = ContextLinearLearner()
    source._state = literal_state(2)  # Literal metadata; no trained result is asserted.
    actor = PromotableActor(
        source,
        version="a0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=digest,
        metadata={"route": "shared-game"},
    )
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        actor=actor,
        max_experiences=4,
        max_consolidations=0,
    )
    gate = ServingPriorityGate(SharingLimits(1, 2, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(runtime, gate)
    work.update(
        source=source,
        actor=actor,
        clock=clock,
        budget=budget,
        runtime=runtime,
        gate=gate,
        shared=shared,
    )
    return work


def pending(case):
    x = np.zeros((9, 296), dtype=np.float64)
    y = np.full((9, 1), 2.0, dtype=np.float64)
    case["runtime"].record_experience(
        Experience(
            "pending", "literal-source", 0, "a0", x, "train", ExperiencePermissions(training=True)
        )
    )
    case["runtime"].record_label(
        LabelArrival("label-pending", "pending", "literal-source", 0, "a0", y)
    )


def reader(case):
    return bind_promotable_minigrid(
        case["actor"], episode_id="shared-direct", shared_runtime=case["shared"]
    )


def play(case, **options):
    defaults = dict(
        cue="blue",
        episode_id="shared-game",
        budget=ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=0,
                max_wall_seconds=150,
                max_process_rss_bytes=512 * 1024**2,
            ),
            lambda: 0.0,
        ),
        check_boundary=lambda: None,
        shared_runtime=case["shared"],
    )
    defaults.update(options)
    return run_promotable_minigrid_episode(case["actor"], **defaults)


def test_should_use_both_real_public_ports_and_hold_original_request_lease(case, monkeypatch):
    actor_predict = PromotableActor.predict
    shared_predict = ResourceSharedRuntime.predict
    calls = []

    def through_actor(owner, x):
        assert case["gate"].snapshot().active_serving_requests == 1
        calls.append("actor")
        return actor_predict(owner, x)

    def through_shared(owner, x):
        calls.append("shared")
        return shared_predict(owner, x)

    monkeypatch.setattr(PromotableActor, "predict", through_actor)
    monkeypatch.setattr(ResourceSharedRuntime, "predict", through_shared)
    before = case["actor"].serving_snapshot()
    result = play(case)
    assert result["status"] == "complete" and calls == ["shared", "actor"] * 2
    assert _same_value(case["actor"].serving_snapshot(), before)
    assert case["gate"].snapshot().active_serving_requests == 0
    assert result["serving"]["sharing"]["initial"]["admitted_updates"] == 0
    assert result["serving"]["sharing"]["latest"]["active_serving_requests"] == 0
    assert case["environments"][0].closed


def test_should_defer_ready_update_during_prediction_then_charge_once_after_request(case):
    pending(case)
    polls = []

    def attempt(owner):
        assert case["gate"].snapshot().active_serving_requests == 1
        polls.append(case["shared"].train_ready())

    case["on_predict"] = attempt
    pinned = reader(case)
    before = case["runtime"].candidate_snapshot()
    pinned.predict(np.zeros((1, 296), dtype=np.float64))
    assert polls[0].deferred_reason == "serving_active" and polls[0].updates == ()
    assert case["budget"].updates_completed == 0 and case["runtime"].candidate_snapshot() == before
    case.pop("on_predict")
    update = case["shared"].train_ready()
    assert len(update.updates) == 1 and update.updates[0].sample_id == "pending"
    pinned.check()
    observed = pinned.report()["sharing"]
    assert observed["latest"]["admitted_updates"] == 1
    assert observed["latest"]["deferrals"] == (("serving_active", 1),)
    assert observed["native_progress_latest"][0] == 1
    assert case["budget"].updates_completed == 1 and case["gate"].snapshot().admitted_updates == 1
    assert case["shared"].train_ready().updates == ()
    assert case["gate"].snapshot().admitted_updates == 1


@pytest.mark.parametrize("fault", ["paused", "quota", "resource"])
def test_should_preserve_existing_training_denials_and_keep_shared_prediction_available(
    case, fault
):
    pending(case)
    if fault == "paused":
        case["gate"].pause()
    elif fault == "quota":
        case["gate"]._admitted = 2
    else:
        case["gate"]._resource = lambda: False
    pinned = reader(case)
    result = case["shared"].train_ready()
    expected = dict(paused="paused", quota="work_quota", resource="resource_unavailable")[fault]
    assert result.deferred_reason == expected and not result.updates
    pinned.predict(np.zeros((1, 296), dtype=np.float64))
    assert dict(pinned.report()["sharing"]["latest"]["deferrals"])[expected] == 1
    assert case["budget"].updates_completed == 0


def test_should_refuse_capacity_before_backend_and_release_ordinary_owned_request(case):
    pinned = reader(case)
    before = WORK["predicts"]
    with case["gate"].serving(), pytest.raises(ValueError, match="capacity"):
        pinned.predict(np.zeros((1, 296), dtype=np.float64))
    assert WORK["predicts"] == before and case["gate"].snapshot().active_serving_requests == 0
    assert pinned.predictions_attempted == 1 and pinned.predictions_returned == 0 and pinned.stopped


def test_should_release_request_and_close_literal_environment_after_lost_return(case):
    case["predict_error"] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        play(case)
    assert case["gate"].snapshot().active_serving_requests == 0
    assert case["environments"][0].closed
    assert caught.value.report["serving"]["predictions_attempted"] == 1
    assert caught.value.report["serving"]["predictions_returned"] == 0


@pytest.mark.parametrize(
    "fault", ["wrapper", "actor", "role", "limits", "counters", "deferrals", "budget"]
)
def test_should_refuse_foreign_or_malformed_sharing_before_actor_snapshot(case, monkeypatch, fault):
    if fault == "wrapper":
        shared = object()
    else:
        shared = case["shared"]
        if fault == "actor":
            shared._runtime._actor = object()
        elif fault == "role":
            shared._runtime._training_role = "probe_train"
        elif fault == "limits":
            object.__setattr__(case["gate"]._limits, "max_serving_requests", True)
        elif fault == "counters":
            case["gate"]._admitted = -1
        elif fault == "deferrals":
            case["gate"]._deferrals = {"foreign": 1}
        else:
            shared._runtime._budget.budget = replace(case["budget"].budget, max_training_updates=1)

    def forbidden(*args):
        pytest.fail("invalid sharing reached actor snapshot")

    monkeypatch.setattr(PromotableActor, "serving_snapshot_if_idle", forbidden)
    with pytest.raises(ValueError):
        bind_promotable_minigrid(case["actor"], episode_id="invalid", shared_runtime=shared)


@pytest.mark.parametrize(
    "fault",
    ["runtime", "gate", "limits", "resource", "budget", "budget_config", "clock", "inbox_role"],
)
def test_should_stop_on_equal_replacement_or_changed_control_before_prediction(case, fault):
    pinned = reader(case)
    if fault == "runtime":
        case["shared"]._runtime = object()
    elif fault == "gate":
        case["shared"]._sharing = ServingPriorityGate(
            SharingLimits(1, 2, 1), resource_available=lambda: True
        )
    elif fault == "limits":
        case["gate"]._limits = replace(case["gate"]._limits)
    elif fault == "resource":
        case["gate"]._resource = lambda: True
    elif fault == "budget":
        case["runtime"]._budget = ToyBudgetSession(case["budget"].budget, lambda: 0.0)
    elif fault == "budget_config":
        case["budget"].budget = replace(case["budget"].budget)
    elif fault == "clock":
        case["budget"].clock = lambda: 0.0
    else:
        case["runtime"]._inbox._training_role = "probe_train"
    before = WORK["predicts"]
    with pytest.raises(ValueError):
        pinned.predict(np.zeros((1, 296), dtype=np.float64))
    assert pinned.stopped and WORK["predicts"] == before


@pytest.mark.parametrize("fault", ["admitted", "deferrals", "budget_progress"])
def test_should_refuse_counter_rollback_and_keep_last_valid_observation(case, fault):
    case["gate"]._admitted = 1
    case["gate"]._deferrals = {"paused": 1}
    case["budget"].updates_completed = 1
    pinned = reader(case)
    if fault == "admitted":
        case["gate"]._admitted = 0
    elif fault == "deferrals":
        case["gate"]._deferrals = {}
    else:
        case["budget"].updates_completed = 0
    with pytest.raises(ValueError, match="backwards"):
        pinned.check()
    observed = pinned.report()["sharing"]
    assert observed["latest"]["admitted_updates"] == 1 and observed["last_readback_error"]


def test_should_refuse_post_snapshot_control_mutation_before_shared_callback(case, monkeypatch):
    pinned = reader(case)
    snapshot = ServingPriorityGate.snapshot

    def changed(owner):
        result = snapshot(owner)
        owner._resource = lambda: True
        return result

    monkeypatch.setattr(ServingPriorityGate, "snapshot", changed)
    with pytest.raises(ValueError, match="changed"):
        pinned.predict(np.zeros((1, 296), dtype=np.float64))
    assert pinned.predictions_attempted == 0


def test_should_serve_pinned_actor_during_partial_private_update_without_candidate_reads(case):
    pending(case)
    entered, release = Event(), Event()

    def pause(owner):
        owner._state = replace(owner._state, training_examples=999)
        entered.set()
        assert release.wait(3), "test did not release private update"

    case["on_train"] = pause
    before = case["actor"].serving_snapshot()
    pinned = reader(case)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(case["shared"].train_ready)
        try:
            assert entered.wait(3)
            assert case["gate"].snapshot().training_active
            values = pinned.predict(np.zeros((1, 296), dtype=np.float64))
            assert np.argmax(values) == 2 and _same_value(case["actor"].serving_snapshot(), before)
        finally:
            release.set()
        result = future.result(timeout=3)
    assert len(result.updates) == 1 and case["budget"].updates_completed == 1
    pinned.check()


def test_should_keep_default_report_and_prediction_route_without_sharing(case):
    pinned = bind_promotable_minigrid(case["actor"], episode_id="ordinary")
    pinned.predict(np.zeros((1, 296), dtype=np.float64))
    assert "sharing" not in pinned.report() and case["gate"].snapshot().active_serving_requests == 0


def test_should_reject_actor_slot_swap_even_when_shared_owner_stays_same(case):
    pinned = reader(case)
    case["actor"]._slot = replace(case["actor"]._slot)
    with pytest.raises(ValueError, match="changed"):
        pinned.check()


def test_should_observe_pause_resume_and_genuine_deferrals_without_erasing_them(case):
    pending(case)
    pinned = reader(case)
    case["gate"].pause()
    assert case["shared"].train_ready().deferred_reason == "paused"
    pinned.check()
    assert pinned.report()["sharing"]["latest"]["paused"]
    case["gate"].resume()
    pinned.predict(np.zeros((1, 296), dtype=np.float64))
    assert dict(pinned.report()["sharing"]["latest"]["deferrals"])["paused"] == 1
    assert case["budget"].updates_completed == 0


@pytest.mark.parametrize("fault", ["resource", "clock", "actor_slot"])
def test_should_pin_controls_before_first_public_gate_snapshot(case, monkeypatch, fault):
    snapshot = ServingPriorityGate.snapshot

    def changed(owner):
        result = snapshot(owner)
        if fault == "resource":
            owner._resource = lambda: True
        elif fault == "clock":
            case["budget"].clock = lambda: 0.0
        else:
            case["actor"]._slot = replace(case["actor"]._slot)
        return result

    def forbidden(*args):
        pytest.fail("changed shared admission reached public actor snapshot")

    monkeypatch.setattr(ServingPriorityGate, "snapshot", changed)
    monkeypatch.setattr(PromotableActor, "serving_snapshot_if_idle", forbidden)
    with pytest.raises(ValueError, match="changed"):
        reader(case)
