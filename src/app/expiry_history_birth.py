"""Original local expiry birth metadata; no ports, locks or portable authority.

The lifecycle owns one bounded scalar birth snapshot. Its original opt-in ledger
shares that exact tuple and its already existing weak history reference. Capture
validates these explicitly ephemeral fields but never serializes or restores
them. Metadata comparison cannot enroll a new observer or authorize raw access.
"""

import sys
from math import isfinite
from typing import Any
from weakref import ReferenceType

from src.app.erased_inbox_origins import _counter, _schema
from src.app.toy_execution_budget import ToyExecutionBudget
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits


EPHEMERAL_LIFECYCLE_FIELDS = frozenset(
    {"_expiry_history_birth", "_expiry_history_on", "_expiry_authority"}
)


def expiry_policy_birth(policy: Any) -> tuple:
    """Snapshot fixed original policy values before any constructor clock port."""
    _schema(policy, DataRetentionPolicy)
    _schema(policy.holders, PayloadOwnershipLimits)
    values = (
        policy.max_lifetime_ingress_bytes,
        policy.max_retention_ticks,
        policy.max_checkpoint_pending,
        policy.max_checkpoint_preparations,
        policy.max_promotion_pending,
        policy.holders.max_live_holders,
        policy.holders.max_lifetime_enrollments,
    )
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError("expiry birth requires original nonnegative policy counters")
    copies = policy.owned_payload_copies
    if copies is not None:
        _schema(copies, PayloadCopyLimits)
        if type(copies.max_lifetime_owned_bytes) is not int or copies.max_lifetime_owned_bytes < 0:
            raise ValueError("expiry birth requires original nonnegative copy allowance")
    seconds = policy.max_retention_seconds
    if seconds is not None and (
        (type(seconds) is not int and type(seconds) is not float)
        or not isfinite(seconds)
    ):
        raise ValueError("expiry birth requires bounded finite original retention time")
    DataRetentionPolicy.__post_init__(policy)
    return (
        id(policy),
        id(policy.holders),
        id(copies),
        values,
        None if copies is None else copies.max_lifetime_owned_bytes,
        seconds,
    )


def expiry_authority_birth(life: Any, original_policy: tuple) -> tuple:
    """Bind the fixed lifecycle roots, gates and port identities without calls."""
    from src.app.actor_shadow import ActorShadowRuntime, StableActor
    from src.app.experience_inbox import ExperienceInbox
    from src.app.managed_experience import ManagedExperienceOwner
    from src.app.payload_copy_budget import PayloadCopyBudget
    from src.app.payload_ownership import PayloadOwnershipRegistry
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.serving_promotion import PromotableActor
    from src.app.toy_execution_budget import ToyBudgetSession
    from src.core.experience import LogicalClock

    for name, kinds in (
        ("_owner", (ManagedExperienceOwner,)),
        ("_shared", (ResourceSharedRuntime,)),
        ("_actor", (StableActor, PromotableActor)),
        ("_registry", (PayloadOwnershipRegistry,)),
        ("_clock", (LogicalClock,)),
        ("_budget", (ToyBudgetSession,)),
        ("_sharing", (ServingPriorityGate,)),
        ("_copy_budget", (PayloadCopyBudget, type(None))),
    ):
        if not any(type(getattr(life, name)) is kind for kind in kinds):
            raise ValueError("expiry original root requires exact supported app type")
    if type(life._shared._runtime) is not ActorShadowRuntime:
        raise ValueError("expiry original runtime requires exact supported app type")
    runtime = life._shared._runtime
    if (
        type(runtime._inbox) is not ExperienceInbox
        or life._owner._shared is not life._shared
        or runtime._actor is not life._actor
        or runtime._budget is not life._budget
        or runtime._inbox._clock is not life._clock
        or life._actor._payload_registry is not life._registry
    ):
        raise ValueError("expiry original runtime/owner relationships changed")
    policy = expiry_policy_birth(life._policy)
    if policy != original_policy:
        raise ValueError("expiry original policy changed during lifecycle construction")
    for value, kind in (
        (life._budget_policy, ToyExecutionBudget),
        (life._owner._limits, LifecycleLimits),
        (life._sharing_limits, SharingLimits),
    ):
        _schema(value, kind)
    configuration = tuple(
        tuple(vars(value).values())
        for value in (life._budget_policy, life._owner._limits, life._sharing_limits)
    )
    _require_configuration(configuration)
    roots = (
        life,
        life._owner,
        life._shared,
        life._actor,
        life._registry,
        life._lineage,
        life._clock,
        life._budget,
        life._sharing,
        life._budget_policy,
        life._sharing_limits,
        life._time_gate,
        life._owner._gate,
        life._registry._gate,
        life._sharing._gate,
        life._copy_budget,
        None if life._copy_budget is None else life._copy_budget._gate,
        None if life._copy_budget is None else life._copy_budget._limits,
        life._owner._limits,
    )
    ports = (
        life._measure,
        life._footprint,
        life._erase,
        life._wall_clock,
        life._progress,
        life._sampler,
        life._resource,
        life._auxiliary_bytes,
        life._checkpoint_bytes,
        life._growth_bytes,
        life._prepare_bytes,
        life._prediction_bytes,
    )
    return (
        policy,
        tuple(id(value) for value in roots),
        tuple(id(value) for value in ports),
        configuration,
    )


def _require_configuration(value: Any, *, bounded: bool = False) -> None:
    if type(value) is not tuple or len(value) != 3:
        raise ValueError("expiry original configuration birth schema changed")
    for row in value:
        if type(row) is not tuple or len(row) > 32:
            raise ValueError("expiry original configuration exceeds its scalar bound")
        for item in row:
            if item is None or type(item) is bool:
                continue
            if type(item) is int:
                if not bounded or item.bit_length() <= 63:
                    continue
            elif type(item) is float and isfinite(item):
                continue
            raise ValueError("expiry original configuration requires bounded exact scalars")


def _require_birth_tuple(value: Any, *, bounded: bool = False) -> None:
    # Exact primitive grammar precedes equality, so foreign operands cannot run
    # comparison callbacks during source capture or the final publication proof.
    if type(value) is not tuple or len(value) != 4:
        raise ValueError("expiry requires its original bounded scalar birth tuple")
    policy, roots, ports, configuration = value
    _require_configuration(configuration, bounded=bounded)
    if type(policy) is not tuple or len(policy) != 6:
        raise ValueError("expiry original policy birth schema changed")
    ids, values, copies, seconds = policy[:3], policy[3], policy[4], policy[5]
    for item in ids:
        _counter(item)
    if type(values) is not tuple or len(values) != 7:
        raise ValueError("expiry original policy value schema changed")
    for item in values:
        if type(item) is not int or item < 0 or (bounded and item.bit_length() > 63):
            raise ValueError("expiry original policy value schema changed")
    if copies is not None:
        if type(copies) is not int or copies < 0 or (bounded and copies.bit_length() > 63):
            raise ValueError("expiry original copy value schema changed")
    if seconds is not None and (
        (type(seconds) is not int and type(seconds) is not float)
        or (bounded and type(seconds) is int and seconds.bit_length() > 63)
        or not isfinite(seconds)
    ):
        raise ValueError("expiry original elapsed birth value changed")
    for items, size in ((roots, 19), (ports, 12)):
        if type(items) is not tuple or len(items) != size:
            raise ValueError("expiry original root/port birth schema changed")
        for item in items:
            _counter(item)


def require_expiry_birth_source(life: Any) -> None:
    """Validate explicitly ephemeral capture fields; acquire no ledger gate."""
    from src.app.managed_data_lifecycle import ManagedDataLifecycle
    from src.app.managed_inbox_origins import ManagedInboxOrigins
    from src.app.managed_replay_origins import ManagedReplayOrigins

    if type(life) is not ManagedDataLifecycle or type(life._expiry_history_on) is not bool:
        raise ValueError("expiry birth requires its exact original lifecycle/ON marker")
    original = life._expiry_authority
    # Birth/capture retain the original valid integer domain. Automatic cleanup
    # observation separately applies its finite proof bound without changing
    # enrollment, ordinary access, policy values or portable capture validation.
    _require_birth_tuple(original)
    if expiry_authority_birth(life, original[0]) != original:
        raise ValueError("expiry original lifecycle policy/roots/ports changed")
    if (
        life._shared._sharing is not life._sharing
        or life._sharing._resource is not life._resource
        or life._budget.clock is not life._wall_clock
        or life._budget.progress is not life._progress
        or life._budget.process_rss_sampler is not life._sampler
        or life._budget.budget is not life._budget_policy
        or life._sharing._limits is not life._sharing_limits
    ):
        raise ValueError("expiry original current authority relationships changed")
    birth = life._expiry_history_birth
    if not life._expiry_history_on:
        if birth is not None:
            raise ValueError("expiry OFF lifecycle cannot carry enrolled history")
        return
    if type(birth) is not ReferenceType:
        raise ValueError("expiry ON lifecycle lost its original weak history birth")
    history = birth()
    if type(history) is not ManagedInboxOrigins or type(history._ledger) is not ReferenceType:
        raise ValueError("expiry original history or weak ledger expired")
    ledger = history._ledger()
    if (
        type(ledger) is not ManagedReplayOrigins
        or ledger._expiry_authority_birth is not original
        or ledger._inbox_history_birth is not birth
        or ledger._original_inbox_origins is not birth
        or ledger._inbox_origins is not history
        or type(ledger._lifecycle) is not ReferenceType
        or ledger._lifecycle() is not life
        or type(ledger._owner) is not ReferenceType
        or ledger._owner() is not life._owner
    ):
        raise ValueError("expiry original history/lifecycle birth peer changed")


def bind_expiry_history(life: Any, ledger: Any) -> None:
    """Only final fresh original ledger construction installs the weak bridge."""
    from src.app.managed_replay_origins import ManagedReplayOrigins

    caller = sys._getframe(1)
    if (
        type(ledger) is not ManagedReplayOrigins
        or caller.f_code is not ManagedReplayOrigins.__init__.__code__
        or caller.f_locals.get("self") is not ledger
    ):
        raise ValueError("expiry history enrollment requires original ledger construction")
    require_expiry_birth_source(life)
    if (
        life._expiry_history_on
        or life._expiry_history_birth is not None
        or ledger._expiry_authority_birth is not None
        or ledger._rows
        or ledger._last_work
        or ledger._pending is not None
        or life._budget.updates_completed
        or type(ledger._inbox_history_birth) is not ReferenceType
        or ledger._inbox_history_birth is not ledger._original_inbox_origins
        or ledger._inbox_history_birth() is not ledger._inbox_origins
        or ledger._lifecycle is None
        or ledger._lifecycle() is not life
    ):
        raise ValueError("expiry history requires fresh original ON birth, without reenrollment")
    # Why: reuse the already committed weak birth and scalar tuple; capture and
    # cleanup cannot manufacture either of these original peer identities.
    ledger._expiry_authority_birth = life._expiry_authority
    life._expiry_history_birth = ledger._inbox_history_birth
    life._expiry_history_on = True
