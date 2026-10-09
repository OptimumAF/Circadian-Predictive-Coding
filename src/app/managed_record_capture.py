"""Detach paired records under one original nonblocking owner interval.

Inputs: installed managed owner and independent aggregate/UTF8 limits. Output:
complete paired metadata and original live references. No native payload, clock
read, model/measurement callback, worker, byte encoding, publication or restore.
"""

from copy import deepcopy

from src.app.actor_shadow import ActorShadowRuntime
from src.app.experience_inbox import ExperienceInbox
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_lifecycle_capture import _lease_managed_lifecycle
from src.app.resource_sharing import ResourceSharedRuntime
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.managed_lifecycle_state import AuthorityReference, LifecycleCaptureLimits
from src.core.managed_record_state import (
    ManagedRecordCapture,
    ManagedRecordMetadata,
    RuntimeRecordObservation,
    validate_managed_record_capture,
)

RUNTIME_FIELDS = frozenset(
    {
        "_actor",
        "_base_actor_version",
        "_candidate",
        "_candidate_version",
        "_inbox",
        "_budget",
        "_consolidation_limit",
        "_attempted_ids",
        "_consolidations",
        "_stopped",
        "_retired",
        "_revision",
        "_write_gate",
        "_payload_ready",
        "_payload_lineage",
    }
)
INBOX_FIELDS = frozenset(
    {
        "_learner",
        "_clock",
        "_budget",
        "_learner_version",
        "_capacity",
        "_experiences",
        "_labels",
        "_event_ids",
        "_applied",
        "_erased",
        "_historical_completed_updates",
        "_draining",
        "_stopped",
        "_last_time",
        "_registration_guard",
        "_training_guard",
        "_payload_copy_guard",
    }
)


def _runtime_schema(runtime) -> None:
    if type(runtime) is not ActorShadowRuntime or vars(runtime).keys() != RUNTIME_FIELDS:
        raise ValueError("paired capture requires complete original runtime schema")
    inbox = runtime._inbox
    if type(inbox) is not ExperienceInbox or vars(inbox).keys() != INBOX_FIELDS:
        raise ValueError("paired capture requires complete original inbox schema")


def _source(runtime, limits, life) -> None:
    _runtime_schema(runtime)
    inbox = runtime._inbox
    if (
        type(runtime._budget) is not ToyBudgetSession
        or inbox._budget is not runtime._budget
        or inbox._learner is not runtime._candidate
        or type(inbox._learner_version) is not str
        or inbox._learner_version != runtime._candidate_version
        or inbox._draining is not False
    ):
        raise ValueError("paired original inbox learner/budget/version or quiescence changed")
    histories = (runtime._attempted_ids, runtime._consolidations)
    if type(histories[0]) is not set or type(histories[1]) is not list:
        raise ValueError("unsupported original consolidation storage")
    owner = life.owner
    size = (
        1
        + sum(map(len, histories))
        + sum(
            map(
                len,
                (
                    owner.catalog,
                    owner.opted_out,
                    owner.revoked_keys,
                    owner.declaration_ticks,
                    owner.declaration_seconds,
                    life.registry.holders,
                ),
            )
        )
    )
    if size > limits.max_records:
        raise ValueError("paired aggregate histories exceed independent capacity")
    # Bound IDs before the leased cursor helper sorts the original attempt set.
    for value in (runtime._base_actor_version, runtime._candidate_version, *histories[0]):
        if type(value) is not str or len(value) > limits.max_identifier_bytes:
            raise ValueError("oversized original paired identity")


def _observation(runtime, registry) -> RuntimeRecordObservation:
    enrollments = [
        number
        for number, (kind, reference) in registry._holders.items()
        if kind == "candidate" and reference() is runtime
    ]
    if len(enrollments) != 1:
        raise ValueError("current original runtime requires one retained enrollment")
    inbox = runtime._inbox
    actor = runtime.actor
    # The original actor lease is already held. PromotableActor.version would
    # reenter its nonreentrant read gate; read the same current slot directly.
    serving_version = (
        actor._slot.bundle.version if type(actor) is PromotableActor else actor._version
    )
    return RuntimeRecordObservation(
        runtime._base_actor_version,
        serving_version,
        runtime._candidate_version,
        runtime._revision,
        runtime._consolidation_limit,
        runtime._stopped,
        runtime._retired,
        runtime._payload_ready,
        runtime._budget.updates_completed,
        inbox._completed_updates(),
        inbox._last_time,
        inbox._stopped,
        enrollments[0],
    )


def _authority(runtime):
    inbox = runtime._inbox
    refs = {
        "runtime.root": runtime,
        "runtime.actor": runtime.actor,
        "runtime.candidate": runtime._candidate,
        "runtime.inbox": inbox,
        "runtime.budget": runtime._budget,
        "runtime.lineage": runtime._payload_lineage,
        "runtime.gate": runtime._write_gate,
        "inbox.learner": inbox._learner,
        "inbox.clock": inbox._clock,
        "inbox.budget": inbox._budget,
        "shared.runtime": runtime,
    }
    return tuple(AuthorityReference(path, value) for path, value in sorted(refs.items()))


def capture_managed_records(
    owner: ManagedExperienceOwner, *, limits: LifecycleCaptureLimits
) -> ManagedRecordCapture:
    """Capture both complete records without reacquiring the held runtime lease."""
    if (
        type(owner) is not ManagedExperienceOwner
        or type(owner._shared) is not ResourceSharedRuntime
    ):
        raise ValueError("paired capture requires original installed shared owner")
    _runtime_schema(owner._shared._runtime)
    with _lease_managed_lifecycle(owner, limits=limits) as life:
        runtime = owner._shared._runtime
        _source(runtime, limits, life.metadata)
        result = ManagedRecordCapture(
            ManagedRecordMetadata(
                1,
                _observation(runtime, owner._lifecycle._registry),
                runtime._read_consolidation_cursor(),
                life.metadata,
            ),
            tuple(sorted(life.authority + _authority(runtime), key=lambda item: item.path)),
        )
        validate_managed_record_capture(result, limits)
        # Why: a single copy preserves aliases across both metadata components;
        # original locks, ports, clocks, models and owners never enter deepcopy.
        detached = ManagedRecordCapture(deepcopy(result.metadata), result.authority)
        validate_managed_record_capture(detached, limits)
        return detached
