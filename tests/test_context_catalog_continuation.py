"""Real replay/serving/pause/resume with literal TRAIN source, no study payload."""

from copy import deepcopy
from dataclasses import replace

import pytest

from scripts import context_head_live as live
from scripts.context_catalog_control import ContextCatalogControl
from src.adapters.context_serving import snapshot_digest
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import _same_value
from src.app.resource_sharing import ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.reference_controller import ArrivedTrainingFeedback, ReferenceObservation
from src.core.resource_sharing import SharingLimits
from src.core.ticket_workflow import ACTION_IDS
from src.shared.process_memory import ProcessRssSampler
import test_context_catalog_control as fixture_module
from test_context_catalog_control import initial, request, fitted
from test_context_catalog_replay import prototype, CountedContext

source = fixture_module.source


class DeferredContext(CountedContext):
    def fork(self):
        child = DeferredContext(self.policy)
        child.restore_state(self.snapshot_state())
        child.after_replay = getattr(self, "after_replay", lambda: None)
        return child

    def train_batch(self, features, targets):
        result = super().train_batch(features, targets)
        if len(features) == 6:
            self.after_replay()
        return result


@pytest.fixture
def case():
    state = initial()
    control = ContextCatalogControl(request(state), base_snapshot=state, resume_wake=True)
    control.require_source(state, state, digest=snapshot_digest, same_state=_same_value)
    old = prototype()
    template = prototype(("legacy", "password"))
    targets = template.targets[:3].copy()
    targets[1, 0] = ACTION_IDS.index("retire")
    new = replace(
        template,
        context="new-v1",
        row_ids=tuple(f"update-0-{i}" for i in range(3)),
        features=template.features[:3].copy(),
        targets=targets,
    )
    control.prepare_packets((new, old))
    clock = LogicalClock()
    wake = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=1, max_wall_seconds=30), lambda: 0.0
    )
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        control.prepare(wake, sampler)
        learner = DeferredContext(state.policy)
        learner.restore_state(state)
        gate = ServingPriorityGate(
            SharingLimits(1, control.admissions, 1), resource_available=lambda: True
        )
        lease = gate.serving()
        learner.after_replay = lambda: lease.__enter__()
        runtime = ActorShadowRuntime(
            learner,
            actor_version="qualified-12",
            candidate_version="c0",
            clock=clock,
            budget=wake,
            max_experiences=1,
            max_consolidations=2,
        )
        shared = control.bind(runtime, learner, gate)
        runtime.record_experience(
            Experience(
                "batch-0",
                new.context,
                0,
                "qualified-12",
                new.features,
                "train",
                ExperiencePermissions(training=True),
            )
        )
        runtime.record_label(
            LabelArrival("label-0", "batch-0", new.context, 1, "qualified-12", new.targets)
        )
        clock.advance_to(1)
        data = dict(
            control=control,
            shared=shared,
            runtime=runtime,
            gate=gate,
            clock=clock,
            wake=wake,
            learner=learner,
            state=state,
            lease=lease,
        )
        yield data
        if gate.snapshot().active_serving_requests:
            lease.__exit__(None, None, None)


def choose(c, tick, *, clean=False):
    c["clock"].advance_to(tick)
    feedback = ArrivedTrainingFeedback(
        "train", "batch-0", "qualified-12", 0, 1, 0 if clean else 1, 3
    )
    observation = ReferenceObservation(tick, 14 - tick, 1, 1, "qualified-12", feedback)
    return c["control"].choose(observation, c["shared"].adaptation_snapshot().progress)


def poll(c, tick, *, clean=False):
    selected = choose(c, tick, clean=clean)
    result = c["control"].train_action(selected.action_id)
    c["control"].observe_poll(selected.action_id, result)
    return result


def arm(c):
    result = poll(c, 1)
    assert result.training.deferred_reason == "serving_active"
    assert result.replay_receipt is c["runtime"].applied_consolidations[0]
    assert c["wake"].updates_completed == 0
    c["lease"].__exit__(None, None, None)
    return result


def test_should_finish_only_original_wake_after_serving_pause_and_resume(case):
    c = case
    before = (
        id(c["wake"].budget),
        c["wake"].started_at,
        id(c["control"].replay_budget.budget),
        c["control"].replay_budget.started_at,
    )
    first = arm(c)
    c["gate"].pause()
    deferred = poll(c, 2, clean=True)
    assert deferred.action_id == "wake1" and deferred.training.deferred_reason == "paused"
    assert deferred.replay_receipt is None and deferred.training.updates == ()
    assert len(c["runtime"].applied_consolidations) == 1
    c["gate"].resume()
    final = poll(c, 3, clean=True)
    assert len(final.training.updates) == 1 and final.replay_receipt is None
    assert final.training.updates[0].applied_at == 3
    state = c["runtime"].candidate_snapshot().state
    assert state.completed_updates == 14 and state.training_examples == 81
    assert state.heads[1] == c["state"].heads[1]
    assert c["learner"].snapshot_state() == c["state"]
    assert c["runtime"].actor.snapshot_state().state == c["state"]
    assert c["control"].replay_budget.replay_examples_completed == 6
    assert c["wake"].updates_completed == 1
    assert before == (
        id(c["wake"].budget),
        c["wake"].started_at,
        id(c["control"].replay_budget.budget),
        c["control"].replay_budget.started_at,
    )
    assert first.replay_receipt is c["runtime"].applied_consolidations[0]
    snapshot = c["control"].capture()["continuation"]
    assert snapshot["pending"] is None and snapshot["completed"] == snapshot["armed"] == 1
    assert snapshot["deferred_polls"] == 2 and not snapshot["awaiting_poll"]
    assert c["gate"].snapshot().deferrals == (("paused", 1), ("serving_active", 1))


@pytest.mark.parametrize(
    "fault",
    [
        "pending",
        "poll",
        "receipt",
        "bank",
        "admissions",
        "deferrals",
        "shared",
        "option",
        "native",
        "progress",
        "role",
        "tick",
    ],
)
def test_should_refuse_changed_preceding_success_before_more_work(case, fault):
    c = case
    returned = arm(c)
    if fault == "pending":
        c["control"]._continuation._pending = tuple(list(c["control"]._continuation._pending))
    elif fault == "poll":
        object.__setattr__(returned, "requested_wake_updates", 0)
    elif fault == "receipt":
        c["runtime"]._consolidations[0] = replace(returned.replay_receipt)
    elif fault == "bank":
        c["control"].replay._bank = deepcopy(c["control"].replay._bank)
    elif fault == "admissions":
        c["gate"]._admitted += 1
    elif fault == "deferrals":
        c["gate"]._deferrals["paused"] = 1
    elif fault == "shared":
        c["control"]._continuation._shared = deepcopy(object())
    elif fault == "option":
        c["control"].resume_wake = False
    elif fault == "native":
        c["runtime"]._candidate.train_batch(prototype().features[:3], prototype().targets[:3])
    elif fault == "progress":
        c["shared"]._action_progress = replace(c["shared"]._action_progress, native_attempts=3)
    elif fault == "role":
        c["runtime"]._training_role = "probe_train"
    before = (c["wake"].updates_completed, c["control"].replay_budget.replay_examples_completed)
    with pytest.raises(ValueError):
        choose(c, 1 if fault == "tick" else 2, clean=True)
    assert before == (
        c["wake"].updates_completed,
        c["control"].replay_budget.replay_examples_completed,
    )


@pytest.mark.parametrize(
    "fault", ["missing", "replaced", "mutated", "duplicate", "wrong_action", "extra_execution"]
)
def test_should_require_exact_observed_return_once(case, fault):
    c = case
    selected = choose(c, 1)
    returned = c["control"].train_action(selected.action_id)
    if fault == "missing":
        with pytest.raises(ValueError, match="not observed"):
            choose(c, 2)
        return
    if fault == "replaced":
        returned = replace(returned)
    elif fault == "mutated":
        object.__setattr__(returned, "progress", replace(returned.progress, decisions_returned=99))
    elif fault == "duplicate":
        c["control"].observe_poll(selected.action_id, returned)
    elif fault == "extra_execution":
        with pytest.raises(ValueError):
            c["control"].train_action(selected.action_id)
        return
    with pytest.raises(ValueError):
        c["control"].observe_poll(
            "wait" if fault == "wrong_action" else selected.action_id, returned
        )
    assert (
        c["wake"].updates_completed == 0
        and c["control"].replay_budget.replay_examples_completed == 6
    )


@pytest.mark.parametrize("value", [True, 1, None, "yes"])
def test_should_refuse_invalid_or_missing_explicit_mode_before_io(monkeypatch, value):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        live.run_case("unused", resume_catalog_wake=value)


@pytest.mark.parametrize("application", [False, True])
def test_should_complete_original_selected_action_through_existing_worker(source, application):
    options = (
        dict(catalog_policy=fitted(source))
        if application
        else dict(catalog_trial=request(source), catalog_base=source)
    )
    result = live.run_case("unused", resume_catalog_wake=True, **options)
    assert result["complete"], result.get("error")
    history = result["catalog_control"]["native_history"]
    assert history["replay_updates"] == history["wake_updates"] == 1
    assert history["replay_rows"] == 6 and history["wake_rows"] == 3
    assert result["actor_unchanged"] and result["source_unchanged"]
    assert result["catalog_control"]["continuation"]["pending"] is None
    assert len(result["live_serving"]["records"]) == 14


def test_should_preserve_default_catalog_capture_without_continuation():
    control = ContextCatalogControl(request(), base_snapshot=initial())
    assert "continuation" not in control.capture()


def test_should_defer_remaining_wake_when_resource_unavailable_without_replay(case):
    c = case
    arm(c)
    c["gate"]._resource = lambda: False
    result = poll(c, 2, clean=True)
    assert result.training.deferred_reason == "resource_unavailable"
    assert result.replay_receipt is None and result.training.updates == ()
    assert c["control"].replay_budget.replay_examples_completed == 6
    c["gate"]._resource = lambda: True
    assert len(poll(c, 3, clean=True).training.updates) == 1


@pytest.mark.parametrize("fault", ["progress", "deferral", "action", "owner", "clock"])
def test_should_refuse_changes_between_selection_and_execution(case, fault):
    c = case
    arm(c)
    decision = choose(c, 2, clean=True)
    if fault == "progress":
        c["shared"]._action_progress = replace(c["shared"]._action_progress, decisions_returned=20)
    elif fault == "deferral":
        c["gate"]._deferrals["paused"] = 1
    elif fault == "owner":
        c["control"]._continuation = object()
    elif fault == "clock":
        c["clock"].advance_to(3)
    with pytest.raises(ValueError):
        c["control"].train_action("replay6-wake1" if fault == "action" else decision.action_id)
    assert (
        c["wake"].updates_completed == 0
        and c["control"].replay_budget.replay_examples_completed == 6
    )


def test_should_preserve_paid_failed_native_return_and_refuse_continuation(case, monkeypatch):
    c = case
    original = DeferredContext.train_batch

    def failed(owner, features, targets):
        original(owner, features, targets)
        raise RuntimeError("lost replay native return")

    monkeypatch.setattr(DeferredContext, "train_batch", failed)
    decision = choose(c, 1)
    with pytest.raises(RuntimeError, match="lost replay"):
        c["control"].train_action(decision.action_id)
    assert c["control"].replay_budget.replay_examples_completed == 6
    assert c["runtime"].applied_consolidations == () and c["wake"].updates_completed == 0
    with pytest.raises(ValueError):
        choose(c, 2)
    assert c["control"].capture()["continuation"]["pending"] is None
