"""Start an ordinary TRAIN candidate from the just-published actor.

Inputs: original controller/publication/shared owner. Output: one independently
owned candidate and a retired preceding ledger. Original resource authorities
and cumulative charges persist; no unpublished state rebasing, consent transfer,
managed lifecycle, checkpoint token, training or promotion is supplied.
"""

from copy import deepcopy

from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import _budget_state, _integrity, _sampler_state
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import budget_binding
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import PromotableActor, _digest
from src.core.experience import LogicalClock, require_identifier
from src.core.inbox_cursor import GenerationInboxCursor
from src.shared.process_memory import ProcessRssSampler


def _ordinary(controller, shared, publication, version, *, consumed=False):
    if (
        type(shared) is not ResourceSharedRuntime
        or type(shared._sharing) is not ServingPriorityGate
    ):
        raise ValueError("candidate start requires the exact original sharing wrapper/gate")
    owner, gate = shared._runtime, shared._sharing
    if gate._paused is not True or gate._retention_hold is not None:
        raise ValueError("candidate start requires explicit ordinary training pause")
    if type(owner) is not ActorShadowRuntime or type(owner.actor) is not PromotableActor:
        raise ValueError("candidate start requires exact actor/shadow and promotable actor")
    if owner.actor is not controller._actor or owner.actor._payload_registry._lifecycle is not None:
        raise ValueError("candidate start requires this controller's unmanaged actor")
    if (
        owner._training_role != "train"
        or owner._original_training_role != "train"
        or type(owner._inbox._clock) is not LogicalClock
        or owner._consolidation_limit != 0
        or owner._consolidations
        or owner._attempted_ids
        or controller._pending
    ):
        raise ValueError(
            "candidate start requires ordinary quiescent TRAIN without pending tickets or sleep"
        )
    if (
        shared._action_envelope is not None
        or shared._action_pin is not None
        or any(name in vars(shared) for name in ("_action_replay", "_action_sleep_quota"))
        or any(
            getattr(owner._inbox, name) is not None
            for name in ("_registration_guard", "_training_guard", "_payload_copy_guard")
        )
    ):
        raise ValueError("candidate start does not transfer action/replay/consent guard authority")
    if (
        controller._latest is None
        or publication is not controller._latest[0]
        or not _same_value(publication, controller._latest[1])
    ):
        raise ValueError("candidate start requires the original latest publication receipt")
    if not consumed and getattr(controller, "_candidate_start_publication", None) is publication:
        raise ValueError("candidate start attempt already consumed for this publication")
    require_identifier(version, "new candidate version")
    if version in (owner._base_actor_version, owner._candidate_version, publication.actor_version):
        raise ValueError("new candidate version must be distinct")
    if (
        publication.previous_version != owner._base_actor_version
        or publication.actor_version != owner._candidate_version
    ):
        raise ValueError("publication does not belong to the preceding candidate")
    return owner, gate


def start_published_candidate(controller, shared, publication, version):
    with controller._exclusive():
        owner, gate = _ordinary(controller, shared, publication, version)
        with owner._exclusive(), gate._checkpoint_lease():
            owner._require_open()
            actor, budget, clock = owner.actor, owner._budget, owner._inbox._clock
            ActorShadowRuntime._validate_budget(budget)
            if (
                budget.process_rss_sampler is not None
                and type(budget.process_rss_sampler) is not ProcessRssSampler
            ):
                raise ValueError("candidate start requires the original supported RSS sampler")
            pins = budget_binding(budget)
            sampler_pin = _sampler_state(budget.process_rss_sampler)
            controls = _budget_state(budget)
            resource, gate_limits, gate_lock = gate._resource, gate._limits, gate._gate
            authority, action_limits = shared._action_authority, deepcopy(shared._action_limits)
            if (
                len(authority) != 3
                or any(a is not b for a, b in zip(authority, (owner, gate, budget), strict=True))
                or (gate.snapshot().limits, budget.budget) != shared._action_limits
            ):
                raise ValueError("candidate start requires original resource authorities")
            slot = actor._slot
            actor_view = actor.serving_snapshot()
            candidate_digest = _digest(controller._digest, owner._candidate.snapshot_state())
            if (actor_view.actor_version, actor_view.generation) != (
                publication.actor_version,
                publication.generation,
            ) or candidate_digest != _digest(controller._digest, actor_view.model_state):
                raise ValueError(
                    "candidate start refuses stale publication or unpublished private advance"
                )
            cursor = owner._inbox.capture_cursor()
            count = len(cursor.applied)
            remaining = cursor.capacity - count
            if (
                not count
                or cursor.erased
                or cursor.stopped
                or remaining <= 0
                or len(cursor.experiences) != count
                or len(cursor.labels) != count
            ):
                raise ValueError(
                    "candidate start requires a complete consumed ledger and remaining capacity"
                )
            cursor_pin = _integrity(cursor)
            gate_view = gate.snapshot()
            latest = controller._latest
            build, digest, evaluator = controller._build, controller._digest, controller._evaluator
            if evaluator._build is not build:
                raise ValueError("candidate start requires the original guard/preparation builder")

            def check():
                _ordinary(controller, shared, publication, version, consumed=True)
                if (
                    shared._runtime is not owner
                    or shared._sharing is not gate
                    or shared._action_authority is not authority
                    or shared._action_limits != action_limits
                    or owner._budget is not budget
                    or owner._inbox._budget is not budget
                    or owner._inbox._clock is not clock
                    or gate._resource is not resource
                    or gate._limits is not gate_limits
                    or gate._gate is not gate_lock
                    or budget_binding(budget) != pins
                    or controller._latest is not latest
                    or not _same_value(publication, latest[1])
                    or controller._build is not build
                    or controller._digest is not digest
                    or controller._evaluator is not evaluator
                    or evaluator._build is not build
                    or controller._candidate_start_publication is not publication
                ):
                    raise ValueError(
                        "candidate start original owner, publication or controls changed"
                    )
                current, before = _budget_state(budget), deepcopy(controls)
                if current["last_clock"] < before["last_clock"]:
                    raise ValueError("candidate start clock moved backwards")
                for row in (current, before):
                    row.pop("last_clock")
                    row.pop("process_rss_segment")
                    if row["progress_state"] is not None:
                        row["progress_state"].pop("process_rss_segment", None)
                if current != before or gate.snapshot() != gate_view:
                    raise ValueError(
                        "candidate start cumulative work, limits or sharing state changed"
                    )
                if sampler_pin is not None:
                    old_reader, old_rss = sampler_pin
                    reader, rss = _sampler_state(budget.process_rss_sampler)
                    if (
                        reader is not old_reader
                        or rss.pid != old_rss.pid
                        or rss.start_bytes != old_rss.start_bytes
                        or rss.interval_seconds != old_rss.interval_seconds
                        or rss.peak_bytes < old_rss.peak_bytes
                        or rss.sample_count < old_rss.sample_count
                    ):
                        raise ValueError(
                            "candidate start original RSS sampler changed or moved backwards"
                        )
                owner._require_open()
                if (
                    _integrity(owner._inbox._borrow_cursor()) != cursor_pin
                    or _digest(digest, owner._candidate.snapshot_state()) != candidate_digest
                    or not _same_value(actor.serving_snapshot(), actor_view)
                ):
                    raise ValueError("candidate start prior ledger, private state or actor changed")

            # Charge preparation before any builder callback. One receipt/slot,
            # no retry chain or new numerical allowance on failure.
            controller._candidate_start_publication = publication
            check()
            budget.before_final()
            model = build(deepcopy(actor_view.model_state))
            if (
                owner._owns_candidate(model)
                or actor._owns(model)
                or not all(
                    callable(getattr(model, name, None))
                    for name in ("train_batch", "predict", "snapshot_state", "restore_state")
                )
            ):
                raise ValueError(
                    "candidate start builder must return an independent complete learner"
                )
            model.restore_state(deepcopy(actor_view.model_state))
            if _digest(digest, model.snapshot_state()) != candidate_digest:
                raise ValueError("candidate start prepared state differs from published weights")
            check()
            next_cursor: GenerationInboxCursor = GenerationInboxCursor(
                3,
                version,
                remaining,
                (),
                (),
                (),
                cursor.last_tick,
                False,
                cursor.completed_updates,
                (),
                cursor.completed_updates,
            )
            prepared = owner._prepare_handoff(
                model,
                next_cursor,
                actor_version=publication.actor_version,
                candidate_version=version,
            )
            new_authority = (prepared, gate, budget)
            budget.before_final()
            check()
            if _digest(digest, model.snapshot_state()) != candidate_digest:
                raise ValueError("candidate start prepared state changed before publication")
            with actor._payload_exclusive():
                if actor._slot is not slot:
                    raise ValueError("candidate start actor generation changed before publication")
                # All callbacks/copies/validation precede this publication point.
                owner._inbox._retire_ledger()
                owner._retired = True
                shared._runtime = prepared
                shared._action_authority = new_authority
            return prepared
