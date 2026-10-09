"""Capture complete retained state after original access and copy admission.

All native/inbox/cache arrays enter one deepcopy, preserving graph aliases.
Independent ports preflight bounded supported state before copying. No model
method, encoding, owner construction, publication, budget refund or restore.
"""

from contextlib import ExitStack

from src.app.managed_composite_sources import SourceReader
from src.app.managed_composite_bindings import (
    require_composite_bindings,
    require_bounded_pending_payloads,
    require_payload_events,
)
from src.app.managed_lifecycle_capture import _lease_lifecycle_sources, _metadata, _authority
from src.app.managed_record_capture import _source, _observation, _authority as record_authority
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.replay_capture_origins import lease_replay_capture
from src.core.experience import require_tick
from src.core.managed_composite_state import (
    CompositeCaptureLimits,
    ManagedCompositeState,
    ManagedCompositeCapture,
    NativeProjectionPort,
    CompositePreflightPort,
    CompositeCopyPort,
)
from src.core.managed_record_state import (
    ManagedRecordMetadata,
    ManagedRecordCapture,
    validate_managed_record_capture,
    RECORD_AUTHORITY_PATHS,
)
from src.core.payload_ownership import PayloadReferences, PayloadHolderMetadata, OwnedPayloadGroup


def _retained_sources(lifecycle, owner, limits):
    live = []
    for number, (kind, reference) in lifecycle._registry._holders.items():
        holder = reference()
        if holder is not None:
            live.append((number, holder))
    # Validate pending owners before holder ports can dereference their payloads.
    require_composite_bindings(owner, lifecycle, live, limits)
    groups = []
    for number, holder in live:
        kind = lifecycle._registry._holders[number][0]
        refs = holder._payload_references()
        if type(refs) is not PayloadReferences:
            raise ValueError("unsupported complete retained payload reference port")
        groups.append(OwnedPayloadGroup(PayloadHolderMetadata(number, kind, True), refs))
    return live, tuple(groups)


def _require_payload_consent(owner, groups):
    for group in groups:
        for inbox in group.references.inboxes:
            inbox._require_history_indexes()
            if inbox._draining:
                raise ValueError("retained inbox is draining")
            require_payload_events(owner, inbox._experiences.values(), inbox._labels.values())


def capture_managed_composite(
    owner,
    *,
    limits: CompositeCaptureLimits,
    project_native: NativeProjectionPort,
    preflight: CompositePreflightPort,
    copy_state: CompositeCopyPort,
    replay_inventory,
    replay_origins=None,
) -> ManagedCompositeCapture:
    """Detach all supported retained sources during one original owner interval."""
    if type(limits) is not CompositeCaptureLimits or not all(
        callable(port) for port in (project_native, preflight, copy_state)
    ):
        raise ValueError("complete capture requires independent typed limits and ports")
    limits.__post_init__()
    with _lease_lifecycle_sources(owner, limits=limits.records) as (life, enrollments, reserve):
        sharing = life._sharing
        if sharing._training or sharing._serving or sharing._checkpointing:
            raise PayloadOwnershipBusy("complete payload capture requires quiescent sharing")
        if not sharing._paused and sharing._retention_hold is None:
            raise ValueError("complete payload capture requires original paused training")
        life._require_access_leased()
        with ExitStack() as stack:
            observe = None
            if life._sampler is not None:
                try:
                    observe = stack.enter_context(life._sampler._lease_observation())
                except ValueError as error:
                    raise PayloadOwnershipBusy(str(error)) from error
            life._budget._before_final_leased(observe)
            live, groups = _retained_sources(life, owner, limits)
            _require_payload_consent(owner, groups)
            replay_check = stack.enter_context(
                lease_replay_capture(owner, groups, limits, replay_inventory, replay_origins)
            )
            reader = SourceReader(project_native, limits)
            holders = tuple(
                (number, reader.record(holder, f"holder.{number}")) for number, holder in live
            )
            shared = reader.record(sharing, "composite.sharing")
            budget = reader.record(life._budget, "composite.budget")
            clock = reader.record(life._clock, "composite.clock")
            payload_bytes = 0  # Bound the complete graph before original measurement ports.

            def assemble():
                records = ManagedRecordMetadata(
                    1,
                    _observation(owner._shared._runtime, life._registry),
                    owner._shared._runtime._read_consolidation_cursor(),
                    _metadata(owner, life, enrollments),
                )
                _source(owner._shared._runtime, limits.records, records.lifecycle)
                authority = tuple(
                    sorted(
                        _authority(owner, life)
                        + record_authority(owner._shared._runtime)
                        + tuple(reader.authority),
                        key=lambda item: item.path,
                    )
                )
                validate_managed_record_capture(
                    ManagedRecordCapture(
                        records,
                        tuple(item for item in authority if item.path in RECORD_AUTHORITY_PATHS),
                    ),
                    limits.records,
                )
                return ManagedCompositeState(
                    1, records, holders, shared, budget, clock, payload_bytes
                ), authority

            state, authority = assemble()
            preflight(state, limits)
            require_bounded_pending_payloads(owner, life, live)
            replay_check()
            payload_bytes = life._retained_bytes(groups)
            require_tick(payload_bytes, "original retained payload copy bytes")
            require_composite_bindings(owner, life, live, limits)
            if reserve is not None:
                reserve(payload_bytes)
            state, authority = assemble()  # Observe charges only after original admission.
            preflight(state, limits)
            replay_check()
            memo: dict[int, object] = {}
            detached = copy_state(state, limits, memo)
            preflight(detached, limits)
            life._require_access_leased()
            life._budget._before_final_leased(observe)
            require_composite_bindings(owner, life, live, limits)
            require_bounded_pending_payloads(owner, life, live)
            _require_payload_consent(owner, groups)
            replay_check()
            budget, clock = reader.refresh_observations(life._budget, life._clock)
            state, authority = assemble()
            preflight(state, limits)
            replay_check()
            # The existing memo returns every already detached payload object;
            # only fresh scalar/observation records are copied during this pass.
            detached = copy_state(state, limits, memo)
            preflight(detached, limits)
            replay_check()
            return ManagedCompositeCapture(detached, authority)
