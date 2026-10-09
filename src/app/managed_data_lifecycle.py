"""Conservative all-owner cleanup under original local consent/resource authority.

Policies bound lifetime ingress bytes, declaration age and holder/copy capacities.
Cleanup clears all delivered raw records; it does not unlearn model parameters,
overwrite RAM or erase caller snapshots/callback-owned data. No worker or IO.
"""

from dataclasses import replace
from math import isfinite
from threading import Lock
from typing import Any, Callable

from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_experience import ManagedExperienceOwner
from src.app.experience_inbox import ExperienceInbox
from src.app.payload_copy_budget import PayloadCopyBudget
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_retention import DataCleanupReport, DataRetentionPolicy
from src.core.experience import (
    Experience,
    LabelArrival,
    LogicalClock,
    SampleKey,
    require_identifier,
    require_tick,
)


class ManagedDataLifecycle:
    def __init__(
        self,
        owner: ManagedExperienceOwner,
        *,
        policy: DataRetentionPolicy,
        measure_payload_bytes: Callable[[object], int] | None = None,
        native_footprint: Callable[[object], ReplayPayloadErasure] | None = None,
        native_erase: Callable[[object], ReplayPayloadErasure] | None = None,
        measure_auxiliary_bytes=None,
        measure_checkpoint_bytes=None,
        native_growth_bytes=None,
        prepare_model_bytes=None,
        prediction_cache_bytes=None,
    ) -> None:
        if (
            type(owner) is not ManagedExperienceOwner
            or type(policy) is not DataRetentionPolicy
            or not all(callable(f) for f in (measure_payload_bytes, native_footprint, native_erase))
        ):
            raise ValueError(
                "managed lifecycle requires an original owner, typed policy and supported ports"
            )
        from src.app.expiry_history_birth import expiry_authority_birth, expiry_policy_birth

        policy_birth = expiry_policy_birth(policy)
        self._expiry_history_birth = None
        self._expiry_history_on = False
        self._owner, self._policy = owner, policy
        self._shared = owner._shared
        self._actor = self._shared._runtime.actor
        self._registry = self._actor._payload_registry
        self._lineage = self._shared._runtime._payload_lineage
        self._clock = self._shared._runtime._inbox._clock
        self._sharing = self._shared._sharing
        self._budget = self._shared._runtime._budget
        self._budget_policy, self._sharing_limits = self._budget.budget, self._sharing._limits
        self._wall_clock, self._progress, self._sampler = (
            self._budget.clock,
            self._budget.progress,
            self._budget.process_rss_sampler,
        )
        self._resource = self._sharing._resource
        if type(self._clock) is not LogicalClock:
            raise ValueError("managed retention requires the original LogicalClock")
        assert (
            measure_payload_bytes is not None
            and native_footprint is not None
            and native_erase is not None
        )
        self._measure, self._footprint, self._erase = (
            measure_payload_bytes,
            native_footprint,
            native_erase,
        )
        self._admitted_bytes = 0
        self._last_tick = 0
        self._failed = False
        self._retention_driver: Any = None
        self._time_gate = Lock()
        self._last_seconds: float | None = None
        self._auxiliary_started_at: float | None = None
        self._retention_fault = False
        self._copy_budget = (
            None
            if policy.owned_payload_copies is None
            else PayloadCopyBudget(policy.owned_payload_copies)
        )
        self._auxiliary_bytes, self._checkpoint_bytes = (
            measure_auxiliary_bytes,
            measure_checkpoint_bytes,
        )
        self._growth_bytes, self._prepare_bytes, self._prediction_bytes = (
            native_growth_bytes,
            prepare_model_bytes,
            prediction_cache_bytes,
        )
        if self._copy_budget is not None and not all(
            callable(p)
            for p in (
                measure_auxiliary_bytes,
                measure_checkpoint_bytes,
                native_growth_bytes,
                prepare_model_bytes,
                prediction_cache_bytes,
            )
        ):
            raise ValueError("owned payload bytes require all supported copy measurement ports")
        # Why: fixed local roots/limits precede optional opaque clock and holder
        # callbacks. The original opt-in ledger later shares this same snapshot.
        self._expiry_authority = expiry_authority_birth(self, policy_birth)
        self._last_tick = self._clock.now()
        from src.app.expiry_history_birth import require_expiry_birth_source

        require_expiry_birth_source(self)
        if policy.max_retention_seconds is not None:
            if not callable(measure_auxiliary_bytes):
                raise ValueError("elapsed retention requires supported auxiliary measurement")
            self._elapsed()
        self._bind()
        require_expiry_birth_source(self)

    def _bind(self) -> None:
        owner, policy = self._owner, self._policy
        with owner._operation(), self._registry._lease() as groups:
            if (
                owner._lifecycle is not None
                or self._registry._lifecycle is not None
                or owner._catalog
            ):
                raise ValueError("lifecycle requires a fresh unconfigured original manager")
            holders = self._registry._live()
            for _, kind, holder in holders:
                self._require_holder(kind, holder)
            if (
                len(holders) > policy.holders.max_live_holders
                or self._registry._total > policy.holders.max_lifetime_enrollments
            ):
                raise ValueError("existing ownership exceeds declared holder capacities")
            if self._registry._limits is not None and self._registry._limits != policy.holders:
                raise ValueError("original holder allowance cannot be renewed")
            for model in self._models(groups):
                if self._describe(model).payload_bytes:
                    raise ValueError("managed lifecycle requires fresh replay buffers")
            for group in groups:
                for inbox in group.references.inboxes:
                    if type(inbox) is not ExperienceInbox:
                        raise ValueError("unsupported retained inbox")
                    if inbox._experiences or inbox._labels or inbox._applied:
                        raise ValueError("managed lifecycle requires fresh inboxes")
            if self._copy_budget is not None:
                self._copy_budget.reserve(self._retained_bytes(groups))
                for group in groups:
                    for inbox in group.references.inboxes:
                        assert isinstance(inbox, ExperienceInbox)
                        inbox._payload_copy_guard = self._before_native_update
            if policy.max_retention_seconds is not None:
                for group in groups:
                    for auxiliary in group.references.auxiliary:
                        self._retain_auxiliary(auxiliary)
            self._registry._limits = policy.holders
            self._registry._lifecycle = self
            owner._lifecycle = self

    @property
    def admitted_payload_bytes(self) -> int:
        return self._admitted_bytes

    def _require_holder(self, kind, holder) -> None:
        if kind == "actor":
            if holder is not self._actor or type(holder) not in (StableActor, PromotableActor):
                raise ValueError("unsupported actor ownership")
        elif kind == "candidate":
            if (
                type(holder) is not ActorShadowRuntime
                or holder._payload_lineage is not self._lineage
            ):
                raise ValueError("candidate belongs to another lifecycle lineage")
            if holder._budget is not self._budget or holder._inbox._clock is not self._clock:
                raise ValueError("candidate original budget/clock authority changed")
        elif kind == "checkpoint":
            if (
                type(holder) is not CandidateCheckpointController
                or holder._shared is not self._shared
                or holder._shared._runtime._payload_lineage is not self._lineage
            ):
                raise ValueError("checkpoint differs from original sharing authority or lineage")
            if (
                holder._limit > self._policy.max_checkpoint_pending
                or holder._attempt_limit > self._policy.max_checkpoint_preparations
            ):
                raise ValueError("checkpoint capacity exceeds retention policy")
        elif kind == "promotion":
            if type(holder) is not ServingPromotionController or holder._actor is not self._actor:
                raise ValueError("unsupported promotion owner")
            if holder._limit > self._policy.max_promotion_pending:
                raise ValueError("promotion capacity exceeds retention policy")
        else:
            raise ValueError("unsupported lifecycle holder")

    def _tick(self) -> int:
        runtime = self._shared._runtime
        if (
            runtime._payload_lineage is not self._lineage
            or runtime._inbox._clock is not self._clock
            or runtime.actor is not self._actor
            or runtime._budget is not self._budget
            or self._shared._sharing is not self._sharing
            or self._sharing._resource is not self._resource
            or self._budget.clock is not self._wall_clock
            or self._budget.progress is not self._progress
            or self._budget.process_rss_sampler is not self._sampler
            or self._budget.budget is not self._budget_policy
            or self._sharing._limits is not self._sharing_limits
        ):
            raise ValueError("original lifecycle lineage/clock/budget/sharing authority changed")
        now = self._clock.now()
        require_tick(now, "retention clock")
        if now < max(self._last_tick, runtime._inbox._last_time):
            raise ValueError("retention clock moved backwards")
        self._last_tick = now
        return now

    def _require_live(self, key: SampleKey) -> None:
        if self._failed:
            raise ValueError("payload cleanup failed; retry cleanup before access")
        self._require_retention_driver()
        if self._expired(key, self._tick(), self._elapsed()):
            raise ValueError("declared payload retention has expired; cleanup required")

    def _require_access(self) -> None:
        self._require_access_clock(self._elapsed)

    def _require_access_leased(self) -> None:
        """Internal common-source coordinator already owns the original time gate."""
        self._require_access_clock(self._elapsed_leased)

    def _require_access_clock(self, elapsed) -> None:
        if self._failed:
            raise ValueError("payload cleanup failed; retry cleanup before access")
        self._require_retention_driver()
        now, seconds = self._tick(), elapsed()
        if any(
            k not in self._owner._revoked_keys and self._expired(k, now, seconds)
            for k, at in self._owner._declaration_ticks.items()
        ) or self._auxiliary_expired(seconds):
            raise ValueError("declared payload retention has expired; cleanup required")

    def _elapsed(self):
        if self._policy.max_retention_seconds is None:
            return None
        with self._time_gate:
            return self._elapsed_leased()

    def _elapsed_leased(self):
        if self._policy.max_retention_seconds is None:
            return None
        try:
            if self._budget.clock is not self._wall_clock:
                raise ValueError("original retention clock changed")
            value = self._wall_clock()
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or value < 0
                or (self._last_seconds is not None and value < self._last_seconds)
            ):
                raise ValueError("original elapsed retention clock invalid or backwards")
            self._last_seconds = float(value)
            return self._last_seconds
        except BaseException:
            self._retention_fault = True
            raise

    def _declaration_anchor(self):
        return self._tick(), self._elapsed()

    def _expired(self, key, now, seconds):
        limit = self._policy.max_retention_seconds
        return now >= self._owner._declaration_ticks[key] + self._policy.max_retention_ticks or (
            seconds is not None
            and limit is not None
            and seconds >= self._owner._declaration_seconds[key] + limit
        )

    def _auxiliary_expired(self, seconds):
        limit = self._policy.max_retention_seconds
        return (
            seconds is not None
            and limit is not None
            and self._auxiliary_started_at is not None
            and seconds >= self._auxiliary_started_at + limit
        )

    def _anchor_auxiliary(self):
        seconds = self._elapsed()
        if seconds is not None:
            with self._time_gate:
                if self._auxiliary_started_at is None:
                    self._auxiliary_started_at = seconds

    def _require_retention_driver(self):
        if self._retention_fault:
            raise ValueError("retention clock fault requires cleanup")
        if self._retention_driver is not None and self._retention_driver._state in (
            "stopping",
            "stopped",
            "exhausted",
            "failed",
        ):
            raise ValueError("retention driver terminal; raw admission/access is closed")

    def _due(self):
        with self._owner._operation():
            now, seconds = self._tick(), self._elapsed()
            keys = tuple(
                sorted(
                    k
                    for k in self._owner._declaration_ticks
                    if k not in self._owner._revoked_keys and self._expired(k, now, seconds)
                )
            )
            return keys, self._auxiliary_expired(seconds)

    def _expire_all(self):
        keys = tuple(sorted(k for k in self._owner._catalog if k not in self._owner._revoked_keys))
        return self._cleanup(keys, "expired", allow_empty=True)

    def _admit_event(self, event) -> None:
        inbox = self._shared._runtime._inbox
        if (type(event) is Experience and event.key in inbox._experiences) or (
            type(event) is LabelArrival
            and (event.key in inbox._labels or event.event_id in inbox._event_ids)
        ):
            return  # original inbox will refuse duplicate metadata before copying
        payload = event.features if type(event) is Experience else event.targets
        size = self._measure(payload)
        require_tick(size, "ingress payload bytes")
        if self._admitted_bytes + size > self._policy.max_lifetime_ingress_bytes:
            raise ValueError("lifetime ingress byte allowance exhausted")
        self._reserve_bytes(size)
        self._admitted_bytes += size  # charge attempts before opaque copying; never refund

    def _reserve_bytes(self, size) -> None:
        if self._copy_budget is not None:
            self._copy_budget.reserve(size)

    @staticmethod
    def _port_bytes(port, *args) -> int:
        size = port(*args)
        require_tick(size, "measured owned payload bytes")
        return size

    def _before_native_update(self, source, label) -> None:
        self._require_access()
        self._reserve_bytes(
            self._port_bytes(
                self._growth_bytes, self._shared._runtime._candidate, source.features, label.targets
            )
        )

    def _inbox_bytes(self, inbox) -> int:
        return sum(
            self._port_bytes(self._measure, s.features) for s in inbox._experiences.values()
        ) + sum(self._port_bytes(self._measure, t.targets) for t in inbox._labels.values())

    def _before_checkpoint_capture(self, owner) -> None:
        if self._copy_budget is not None:
            self._require_access()
            self._reserve_bytes(
                self._describe(owner._candidate).payload_bytes + self._inbox_bytes(owner._inbox)
            )

    def _before_checkpoint_restore(self, builder, view) -> None:
        if self._copy_budget is not None:
            self._require_access()
            inbox_bytes = sum(
                self._port_bytes(self._measure, s.features) for s in view.inbox.experiences
            ) + sum(self._port_bytes(self._measure, t.targets) for t in view.inbox.labels)
            self._reserve_bytes(
                self._port_bytes(self._prepare_bytes, builder, view.state) + inbox_bytes
            )

    def _before_promotion(self, builder, state, metadata) -> None:
        if self._copy_budget is not None:
            self._require_access()
            self._reserve_bytes(
                self._port_bytes(self._prepare_bytes, builder, state)
                + self._port_bytes(self._auxiliary_bytes, metadata)
            )
        if self._policy.max_retention_seconds is not None:
            self._retain_auxiliary(metadata)

    def _retain_auxiliary(self, auxiliary) -> None:
        if type(auxiliary) is not dict:
            raise ValueError("unsupported retained auxiliary data")
        self._port_bytes(self._auxiliary_bytes, auxiliary)
        # Age bounds owned content, including scalar metadata and empty arrays.
        # Array bytes are a separate quota and cannot establish data presence.
        if auxiliary:
            self._anchor_auxiliary()

    def _validate_cache_input(self, features) -> None:
        if self._policy.max_retention_seconds is not None:
            self._require_access()
        if self._copy_budget is not None:
            self._require_access()
            self._port_bytes(self._auxiliary_bytes, features)

    def _before_cache_write(self, model, features):
        if self._policy.max_retention_seconds is not None:
            self._anchor_auxiliary()
        if self._copy_budget is None:
            return None
        size = self._port_bytes(self._prediction_bytes, model, features)
        self._reserve_bytes(size)
        return size

    def _validate_cache_output(self, prediction, bound) -> None:
        if self._policy.max_retention_seconds is not None:
            self._require_access()
        if bound is not None and self._port_bytes(self._auxiliary_bytes, prediction) > bound:
            raise ValueError("native prediction exceeded its declared owned cache byte bound")

    def _retained_bytes(self, groups) -> int:
        models = self._models(groups)
        inboxes = {id(i): i for g in groups for i in g.references.inboxes}
        snapshots = {id(s): s for g in groups for s in g.references.snapshots}
        auxiliary = {id(a): a for g in groups for a in g.references.auxiliary}
        return (
            sum(self._describe(m).payload_bytes for m in models)
            + sum(self._inbox_bytes(i) for i in inboxes.values())
            + sum(self._port_bytes(self._checkpoint_bytes, s) for s in snapshots.values())
            + sum(self._port_bytes(self._auxiliary_bytes, a) for a in auxiliary.values())
        )

    def payload_byte_snapshot(self):
        if self._copy_budget is None:
            raise ValueError("owned payload byte policy is not configured")
        with self._owner._operation(), self._registry._lease() as groups:
            self._tick()
            return self._copy_budget.snapshot(self._retained_bytes(groups))

    def _describe(self, model) -> ReplayPayloadErasure:
        result = self._footprint(model)
        if type(result) is not ReplayPayloadErasure:
            raise ValueError("native footprint requires exact erasure counts")
        return result

    @staticmethod
    def _models(groups):
        models = {id(model): model for group in groups for model in group.references.models}
        return tuple(models.values())

    def delete(self, keys: tuple[SampleKey, ...]) -> DataCleanupReport:
        return self._cleanup(keys, "deleted")

    def opt_out(self, subject_id: str) -> DataCleanupReport:
        require_identifier(subject_id, "subject_id")
        keys = tuple(
            sorted(
                k for k, d in self._owner._catalog.items() if d.provenance.subject_id == subject_id
            )
        )
        if not keys:
            raise ValueError("unknown subject")
        return self._cleanup(keys, "opt_out", subject_id)

    def expire(self) -> DataCleanupReport:
        from src.app.expiry_inbox_observation import (
            ExpiryObservationError,
            _ExpiryObservation,
            expiry_observation,
        )

        observation = expiry_observation(self)
        _ExpiryObservation.__enter__(observation)
        proof, gate, context, access = (
            observation.proof,
            observation.gate,
            observation.context,
            observation.close_access,
        )
        active = observation.active
        thread = observation.thread
        fault = False
        try:
            keys, auxiliary = self._due()
            if not keys and not auxiliary:
                report = DataCleanupReport("expired", (), (), 0, 0, 0, 0)
            else:
                report = self._cleanup(
                    keys, "expired", allow_empty=auxiliary, expiry=(observation, proof)
                )
                fault = observation.fault is not False or (active and proof is None)
        finally:
            # Lexical original gate/tokens survive mutable observer fields and
            # releases never replace a primary native/holder cleanup exception.
            _ExpiryObservation.close(observation, gate, context, access, thread)
        if fault:
            raise ExpiryObservationError("expiry-observation-refused", report)
        return report

    def retry_cleanup(self) -> DataCleanupReport:
        if not self._failed:
            raise ValueError("no failed cleanup to retry")
        return self._cleanup(tuple(sorted(self._owner._revoked_keys)), "deleted")

    def _cleanup(
        self, keys, reason, subject_id=None, allow_empty=False, expiry=None
    ) -> DataCleanupReport:
        if not (allow_empty and keys == ()):
            self._validate_keys(keys)
        with (
            self._owner._operation(),
            self._registry._lease() as groups,
            self._shared._sharing._checkpoint_lease(),
        ):
            if expiry is not None:
                from src.app.expiry_inbox_observation import _ExpiryObservation

                _ExpiryObservation.before_cleanup(expiry[0], expiry[1], groups)
            self._tick()
            holders = self._registry._live()
            for _, kind, holder in holders:
                self._require_holder(kind, holder)
            models = self._models(groups)
            for model in models:
                self._describe(model)
            for group in groups:
                for auxiliary in group.references.auxiliary:
                    if type(auxiliary) is not dict:
                        raise ValueError("unsupported retained auxiliary data")
            plans = self._prepare_inboxes(groups, reason)
            revoked = set(keys) | {key for _, inbox_keys, _, _ in plans for key in inbox_keys}
            self._owner._revoked_keys.update(revoked)
            if subject_id is not None:
                self._owner._opted_out.add(subject_id)
            checkpoints, promotions = self._invalidate(holders)
            try:
                erased = self._clear_payloads(models, plans, groups)
                self._failed = False
                self._retention_fault = False
                self._auxiliary_started_at = None
            except BaseException:
                self._failed = True
                for _, kind, holder in holders:
                    if kind == "candidate":
                        assert isinstance(holder, ActorShadowRuntime)
                        holder._stopped = True
                raise
            staged = None if expiry is None else expiry[0].staged
            report = DataCleanupReport(
                reason,
                tuple(sorted(keys)),
                tuple(sorted(revoked)),
                erased,
                len(plans),
                checkpoints,
                promotions,
            )
            if expiry is not None:
                _ExpiryObservation.finish(expiry[0], expiry[1], report, staged, groups)
            return report

    def _clear_payloads(self, models, plans, groups) -> int:
        erased = 0
        for model in models:
            result = self._erase(model)
            if type(result) is not ReplayPayloadErasure or self._describe(model).payload_bytes:
                raise ValueError("native erasure did not clear its replay payloads")
            erased += result.snapshots
        for inbox, inbox_keys, prepared, _ in plans:
            inbox._commit_erased_history(inbox_keys, prepared)
        for group in groups:
            for auxiliary in group.references.auxiliary:
                assert isinstance(auxiliary, dict)  # validated before revocation
                auxiliary.clear()
        if type(self._actor) is PromotableActor:
            slot = self._actor._slot
            self._actor._slot = replace(slot, generation=slot.generation + 1, previous=None)
        return erased

    def _validate_keys(self, keys) -> None:
        if type(keys) is not tuple or not keys:
            raise ValueError("cleanup requires immutable nonempty keys")
        for key in keys:
            if type(key) is not tuple or len(key) != 2:
                raise ValueError("cleanup requires episode/sample keys")
            for value in key:
                require_identifier(value, "cleanup identity")
            if key not in self._owner._catalog:
                raise ValueError("unknown cleanup identity")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate cleanup identity")

    def _prepare_inboxes(self, groups, reason):
        plans = []
        seen = set()
        for group in groups:
            for inbox in group.references.inboxes:
                if id(inbox) in seen:
                    continue
                seen.add(id(inbox))
                inbox._require_history_indexes()
                keys = tuple(sorted(inbox._experiences.keys() | inbox._labels.keys()))
                if any(k not in self._owner._catalog for k in keys):
                    raise ValueError("inbox contains undeclared lifecycle data")
                if keys:
                    now = inbox._read_clock()
                    plans.append(
                        (inbox, keys, inbox._prepare_erased_history(keys, now, reason), now)
                    )
        return plans

    @staticmethod
    def _invalidate(holders):
        checkpoints = promotions = 0
        for _, kind, holder in holders:
            if kind == "checkpoint":
                checkpoints += len(holder._pending)
                holder._pending.clear()
            if kind == "promotion":
                promotions += len(holder._pending)
                holder._pending.clear()
                holder._latest = None
            if kind == "candidate":
                holder._revision += 1
        return checkpoints, promotions
