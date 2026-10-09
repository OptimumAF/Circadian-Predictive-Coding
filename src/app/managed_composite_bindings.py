"""Validate original enrolled sources and retained pending observations.

Inputs are already leased original owners and bounded complete projections.
Outputs are refusals or permission to continue capture. No model/probe callbacks,
additional leases, native work, publication or restore authorization.
"""

from dataclasses import replace
from math import isfinite
import re
from types import MethodType

from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import (
    CandidateCheckpoint,
    CandidateCheckpointController,
    _Pending as CheckpointPending,
    _validate_view,
    _integrity,
)
from src.app.managed_composite_sources import SCHEMAS
from src.app.experience_inbox import ExperienceInbox
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import (
    ServingPromotionController,
    _Pending as PromotionPending,
)
from src.core.experience import require_tick, require_identifier, Experience
from src.core.promotion_guard import decide_promotion
from src.core.serving_ports import PreparedPromotion, PromotionReceipt
from src.shared.process_memory import ProcessRssSegment


def require_bound_method(value, original) -> None:
    """Compare installed bound-method authority without invoking either callback."""
    if (
        type(value) is not MethodType
        or type(original) is not MethodType
        or value.__self__ is not original.__self__
        or value.__func__ is not original.__func__
    ):
        raise ValueError("retained inbox guard differs from original consent/copy owner")


def _schema(value, kind) -> None:
    spec = SCHEMAS.get(kind)
    if spec is None or type(value) is not kind or vars(value).keys() != spec.keys():
        raise ValueError("retained original source differs from complete supported schema")


def require_payload_events(owner, experiences, labels) -> None:
    for event in (*experiences, *labels):
        declaration = owner._catalog.get(event.key)
        if declaration is None or event.key in owner._revoked_keys:
            raise ValueError("retained payload lacks original live consent")
        owner._limits.require_supported(declaration)
        if declaration.provenance.subject_id in owner._opted_out:
            raise ValueError("retained payload subject opted out")
        if event.role != "train":
            raise ValueError("retained payload has forbidden evaluation role")
        if type(event) is Experience and not (
            event.permissions.training and event.permissions.replay
        ):
            raise ValueError("retained source lacks training/replay permission")


def _runtime(owner, life, runtime) -> None:
    _schema(runtime, ActorShadowRuntime)
    inbox = runtime._inbox
    _schema(inbox, ExperienceInbox)
    require_identifier(runtime._candidate_version, "retained candidate version")
    require_identifier(inbox._learner_version, "retained inbox version")
    require_tick(runtime._revision, "retained original revision")
    if (
        runtime._actor is not life._actor
        or runtime._payload_lineage is not life._lineage
        or runtime._budget is not life._budget
        or inbox._clock is not life._clock
        or inbox._budget is not life._budget
        or inbox._learner is not runtime._candidate
        or inbox._learner_version != runtime._candidate_version
    ):
        raise ValueError("retained runtime differs from original actor/lineage/budget/clock")
    require_bound_method(inbox._registration_guard, owner._require_registration)
    require_bound_method(inbox._training_guard, owner._eligible)
    if life._copy_budget is None:
        if inbox._payload_copy_guard is not None:
            raise ValueError("retained inbox has foreign copy authority")
    else:
        require_bound_method(inbox._payload_copy_guard, life._before_native_update)
    if type(runtime._retired) is not bool:
        raise ValueError("retained retirement requires an exact flag")
    historical = inbox._historical_completed_updates
    if runtime._retired:
        require_tick(historical, "retired original completed updates")
        if historical > life._budget.updates_completed:
            raise ValueError("retired inbox work exceeds original cumulative budget")
    elif historical is not None:
        raise ValueError("current retained runtime has a retired inbox ledger")


def require_composite_bindings(owner, life, live, limits) -> None:
    """Check references before zero-copy projection can read an unleased owner."""
    runtimes = {id(value): value for _, value in live if type(value) is ActorShadowRuntime}
    for _, holder in live:
        _schema(holder, type(holder))
        if type(holder) is ActorShadowRuntime:
            _runtime(owner, life, holder)
        elif type(holder) is CandidateCheckpointController:
            _checkpoint_bindings(holder, life, runtimes, limits)
        elif type(holder) is ServingPromotionController:
            _promotion_bindings(holder, life, runtimes, limits)


def _history(controller, limits) -> None:
    require_tick(controller._limit, "retained pending capacity")
    if (
        controller._limit == 0
        or type(controller._pending) is not dict
        or len(controller._pending) > min(controller._limit, limits.records.max_records)
    ):
        raise ValueError("retained pending history exceeds original capacity")


def _checkpoint_bindings(controller, life, runtimes, limits) -> None:
    _history(controller, limits)
    require_tick(controller._attempt_limit, "retained preparation capacity")
    require_tick(controller._attempts, "retained preparation attempts")
    if controller._attempt_limit == 0 or controller._attempts > controller._attempt_limit:
        raise ValueError("retained checkpoint preparation allowance is corrupt")
    for token, pending in controller._pending.items():
        if type(token) is not CandidateCheckpoint or vars(token):
            raise ValueError("retained checkpoint token has unsupported fields/type")
        _schema(pending, CheckpointPending)
        if runtimes.get(id(pending.owner)) is not pending.owner:
            raise ValueError("pending checkpoint runtime is not originally enrolled and leased")
        if (
            pending.budget is not life._budget
            or pending.wall_clock is not life._wall_clock
            or pending.event_clock is not life._clock
            or pending.sampler is not life._sampler
            or pending.progress is not life._progress
            or pending.resource is not life._resource
            or pending.sharing_gate is not life._sharing
            or pending.actor is not life._actor
            or pending.build is not controller._build
            or pending.policy is not controller._policy
            or pending.state_probe is not controller._digest
        ):
            raise ValueError("pending checkpoint original source/resource/probe changed")
        require_tick(pending.revision, "retained checkpoint revision")
        if pending.revision > pending.owner._revision:
            raise ValueError("retained checkpoint revision is ahead of original owner")
        _sampler_observation(pending, life)


def _sampler_observation(pending, life) -> None:
    if life._sampler is None:
        if pending.sampler_state is not None:
            raise ValueError("pending checkpoint has foreign sampler observation")
        return
    state = pending.sampler_state
    if type(state) is not tuple or len(state) != 2 or state[0] is not life._sampler.read_rss_bytes:
        raise ValueError("pending checkpoint original sampler reader changed")
    old = state[1]
    if type(old) is not ProcessRssSegment:
        raise ValueError("pending checkpoint requires exact original sampler segment")
    if vars(old).keys() != {"pid", "start_bytes", "peak_bytes", "sample_count", "interval_seconds"}:
        raise ValueError("pending original sampler segment fields changed")
    if (
        type(old.interval_seconds) not in (int, float)
        or not isfinite(old.interval_seconds)
        or old.interval_seconds <= 0
    ):
        raise ValueError("pending original sampler interval is invalid")
    for value in (old.pid, old.start_bytes, old.peak_bytes, old.sample_count):
        require_tick(value, "pending original sampler observation")
    now = life._budget.process_rss_segment
    if (
        now is None
        or old.pid != now.pid
        or old.start_bytes != now.start_bytes
        or old.interval_seconds != now.interval_seconds
        or not old.start_bytes <= old.peak_bytes <= now.peak_bytes
        or not 1 <= old.sample_count <= now.sample_count
    ):
        raise ValueError("pending checkpoint sampler observation is not original monotone history")


def _promotion_bindings(controller, life, runtimes, limits) -> None:
    _history(controller, limits)
    _schema(controller._evaluator, PromotionGuardEvaluator)
    if controller._evaluator._build is not controller._build:
        raise ValueError("retained promotion builder differs from original evaluator")
    for token, pending in controller._pending.items():
        if type(token) is not PreparedPromotion or vars(token).keys() != {
            "actor_generation",
            "report",
        }:
            raise ValueError("retained promotion token has unsupported fields/type")
        _schema(pending, PromotionPending)
        if runtimes.get(id(pending.runtime)) is not pending.runtime:
            raise ValueError("pending promotion runtime is not originally enrolled and leased")
        if pending.builder is not controller._build:
            raise ValueError("pending promotion original builder changed")
        require_tick(pending.actor_generation, "retained promotion generation")
        if pending.actor_generation > life._actor._slot.generation:
            raise ValueError("retained promotion generation is ahead of original actor")
    if controller._latest is not None:
        latest = controller._latest
        if (
            type(latest) is not tuple
            or len(latest) != 2
            or any(type(v) is not PromotionReceipt for v in latest)
        ):
            raise ValueError("retained rollback observation requires original receipt pair")
        for receipt in latest:
            if vars(receipt).keys() != {"generation", "previous_version", "actor_version"}:
                raise ValueError("retained rollback receipt fields changed")
            require_tick(receipt.generation, "retained rollback generation")
            require_identifier(receipt.previous_version, "retained rollback version")
            require_identifier(receipt.actor_version, "retained rollback version")
        if vars(latest[0]) != vars(latest[1]):
            raise ValueError("original rollback receipt differs from retained observation")


def require_bounded_pending_payloads(owner, life, live) -> None:
    """Only after full graph bounds, check pure view/report relations and consent.

    Why: restore's current-state check rejects legitimate stale retained histories
    and invokes native callbacks/leases. Capture validates observations, not reuse.
    """
    for _, holder in live:
        if type(holder) is CandidateCheckpointController:
            for pending in holder._pending.values():
                view = pending.view
                _validate_view(view)
                _budget_history(view, life)
                if (
                    view.actor_version != pending.owner._base_actor_version
                    or view.learner_version != pending.owner._candidate_version
                    or view.event_tick > life._clock._time
                ):
                    raise ValueError("retained checkpoint original version/chronology differs")
                for digest in (pending.integrity, pending.native_digest, pending.policy_digest):
                    if type(digest) is not str or re.fullmatch("[0-9a-f]{64}", digest) is None:
                        raise ValueError("retained checkpoint digest format changed")
                if _integrity(view) != pending.integrity:
                    raise ValueError("retained complete checkpoint observation changed")
                require_payload_events(owner, view.inbox.experiences, view.inbox.labels)
        elif type(holder) is ServingPromotionController:
            for token, promotion_pending in holder._pending.items():
                holder._require_ticket(
                    token
                )  # Pure original token/report checks; no callback/lease.
                report = promotion_pending.report
                replace(report.policy)
                replace(report.evidence)
                if report.decision != decide_promotion(report.policy, report.evidence):
                    raise ValueError("retained promotion decision differs from recorded evidence")
                for version in (report.actor_version, report.learner_version):
                    require_identifier(version, "retained promotion version")
                if (
                    report.actor_version != promotion_pending.runtime._base_actor_version
                    or report.learner_version != promotion_pending.runtime._candidate_version
                ):
                    raise ValueError("retained promotion original candidate version differs")


def _budget_history(view, life) -> None:
    state = view.budget_state
    fields = (SCHEMAS[type(life._budget)].keys() - {"clock", "progress", "process_rss_sampler"}) | {
        "progress_state"
    }
    if type(state) is not dict or state.keys() != fields:
        raise ValueError("retained checkpoint budget history fields changed")
    if state["budget"] != life._budget_policy or state["started_at"] != life._budget.started_at:
        raise ValueError("retained checkpoint original budget policy/origin changed")
    last = state["last_clock"]
    if (
        type(last) not in (int, float)
        or not isfinite(last)
        or not state["started_at"] <= last <= life._budget.last_clock
    ):
        raise ValueError("retained checkpoint budget chronology is not original history")
    for field in ("updates_completed", "replay_examples_completed"):
        require_tick(state[field], "retained checkpoint cumulative work")
        if state[field] > getattr(life._budget, field):
            raise ValueError("retained checkpoint work exceeds original cumulative budget")
    if view.inbox.completed_updates != state["updates_completed"]:
        raise ValueError("retained checkpoint inbox/budget work differs")
    progress = state["progress_state"]
    if life._progress is None:
        if progress is not None:
            raise ValueError("retained checkpoint has foreign progress history")
    elif type(progress) is not dict or progress.keys() != SCHEMAS[type(life._progress)].keys():
        raise ValueError("retained checkpoint progress history fields changed")
