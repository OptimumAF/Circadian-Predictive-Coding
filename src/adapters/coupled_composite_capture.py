"""Capture complete coupled managed graphs after original RAW-copy admission.

Usage: pause the original owner's shared learner, then call
``capture_coupled_managed_composite(owner, limits=limits)``. Current/previous
serving bundles, candidate, pending checkpoints/promotions and original resource
observations share one bounded graph copy. No native methods, replay, IO, model
construction, restoration or portable authority is provided.
"""

from dataclasses import fields
from types import MappingProxyType
from typing import Any

from src.adapters.coupled_composite_native import (
    LEARNER_FIELDS,
    LEARNER_KIND,
    MODEL_SCHEMAS,
    project_coupled_native,
    validate_coupled_model,
    validate_coupled_policy,
    validate_coupled_snapshot,
)
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.numpy_composite_capture import (
    GraphPreflight,
    BackpropSnapshot,
    CircadianNetworkSnapshot,
    _preflight_checkpoint_inbox,
)
from src.app.candidate_checkpoint import CandidateCheckpointView
from src.app.managed_composite_capture import capture_managed_composite
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledSnapshot, CoupledWakeState
from src.core.managed_composite_state import (
    CompositeCaptureLimits,
    ManagedCompositeCapture,
    ManagedCompositeState,
    SourceRecord,
)
from src.core.payload_ownership import OwnedPayloadGroup, PayloadReferences


class _CoupledGraphPreflight(GraphPreflight):
    _record_types = (
        GraphPreflight._record_types - {BackpropSnapshot, CircadianNetworkSnapshot}
    ) | {
        CoupledSnapshot,
        CoupledParameters,
        CoupledWakeState,
        CoupledInferencePolicy,
        CoupledCircadianConfig,
    }
    _source_schemas = MappingProxyType(
        {**GraphPreflight._source_schemas, LEARNER_KIND: LEARNER_FIELDS, **MODEL_SCHEMAS}
    )

    def source(self, value: SourceRecord) -> None:
        if value.kind in MODEL_SCHEMAS:
            validate_coupled_model(value, self.limits)
        elif value.kind == LEARNER_KIND:
            data = dict(value.fields)
            validate_coupled_policy(data)
            model = data["_model"]
            if type(model) is not SourceRecord:
                raise ValueError("coupled learner requires complete native model source record")
            validate_coupled_model(model, self.limits)
        else:
            super().source(value)

    def _validate_native_record(self, value: Any) -> None:
        if type(value) is CoupledSnapshot:
            validate_coupled_snapshot(value, self.limits)

    def _native_input_dim(self, native: Any) -> int:
        if type(native) is not SourceRecord:
            raise ValueError("coupled runtime requires complete native model projection")
        return validate_coupled_model(native, self.limits)

    def _preflight_checkpoint(self, view: CandidateCheckpointView) -> None:
        input_dim = validate_coupled_snapshot(view.state, self.limits)
        _preflight_checkpoint_inbox(view, self.limits, input_dim)


def preflight_coupled_composite(
    state: ManagedCompositeState, limits: CompositeCaptureLimits
) -> None:
    _CoupledGraphPreflight(limits).preflight(state)


def copy_coupled_composite(
    state: ManagedCompositeState, limits: CompositeCaptureLimits, memo: dict[int, object]
) -> ManagedCompositeState:
    return _CoupledGraphPreflight(limits).copy(state, memo)


def coupled_replay_inventory(groups: Any, limits: CompositeCaptureLimits) -> tuple[Any, ...]:
    """Validate every original holder; coupled native families have no replay rows."""
    if type(groups) is not tuple or len(groups) > limits.records.max_records:
        raise ValueError("coupled replay holder inventory exceeds original capture bound")
    entries: dict[int, tuple[object, tuple[()]]] = {}
    visited = 0
    for group in groups:
        if type(group) is not OwnedPayloadGroup or vars(group).keys() != {"holder", "references"}:
            raise ValueError("coupled inventory requires exact owned payload groups")
        refs = group.references
        if type(refs) is not PayloadReferences or vars(refs).keys() != {
            "models",
            "inboxes",
            "snapshots",
            "auxiliary",
        }:
            raise ValueError("coupled inventory requires exact payload reference fields")
        if any(
            type(items) is not tuple or len(items) > limits.max_nodes
            for items in (refs.models, refs.inboxes, refs.snapshots, refs.auxiliary)
        ):
            raise ValueError("coupled inventory reference collections exceed exact bounds")
        for holder in refs.models + refs.snapshots:
            visited += 1
            if visited > limits.max_nodes:
                raise ValueError("coupled native inventory exceeds original capture bound")
            original = holder
            if type(holder) is CandidateCheckpointView:
                if vars(holder).keys() != {field.name for field in fields(CandidateCheckpointView)}:
                    raise ValueError("coupled inventory checkpoint view fields changed")
                holder = holder.state
                original = holder
            if type(holder) is CoupledLearner:
                project_coupled_native(holder, "replay_inventory", limits)
                original = vars(holder)["_model"]
            elif type(holder) is CoupledSnapshot:
                validate_coupled_snapshot(holder, limits)
            else:
                raise ValueError(
                    "coupled replay inventory requires an exact supported native holder"
                )
            entries[id(original)] = (original, ())
    return tuple(entries.values())


def capture_coupled_managed_composite(
    owner: Any, *, limits: CompositeCaptureLimits
) -> ManagedCompositeCapture:
    """Detach complete coupled state under the existing owner's original leases."""
    return capture_managed_composite(
        owner,
        limits=limits,
        project_native=project_coupled_native,
        preflight=preflight_coupled_composite,
        copy_state=copy_coupled_composite,
        replay_inventory=coupled_replay_inventory,
    )
