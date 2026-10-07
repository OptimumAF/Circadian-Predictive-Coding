"""Closed raw fact schemas for independent confirmation validation.

Existing dataclass annotations describe JSON scalar/container types. Explicit
TypedDicts close the older dictionary-based combined/parent records. Inputs
are decoded facts; no instances, data, scores or IO are constructed. Unsupported
annotations fail rather than permitting open Any fields.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from math import isfinite
from types import UnionType
from typing import (
    Any,
    Literal,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from src.app.continual_confirmation_json import integer, object_fields, require
from src.app.continual_gating_pilot import GatingMethodResult, GatingSeedResult
from src.app.continual_replay_factor_pilot import ReplayMethodResult, ReplaySeedResult
from src.app.continual_schedule_factor_preflight import ScheduleSeedFacts
from src.app.continual_sleep_factor_preflight import SleepSeedFacts
from src.core.circadian_predictive_coding import ReplayRetentionSnapshot
from src.core.controlled_parent_selection import ParentSelectionState
from src.core.neuron_adaptation import NeuronLineageSnapshot
from src.core.sleep_clocks import SleepClockSnapshot
from src.core.sleep_telemetry import SleepEventTelemetry


class MethodSchema(TypedDict):
    name: str
    initial_parameter_sha256: str
    after_a_parameter_sha256: str
    final_parameter_sha256: str
    width_initial: int
    width_final: int
    width_peak: int
    parameters_initial: int
    parameters_final: int
    parameters_peak: int
    wake_updates: int
    wake_presentations: int
    wake_inference_loops: int
    wake_example_inference_iterations: int
    applied_replay_updates: int
    rejected_executed_replay_updates: int
    own_sleep_attempts: int
    committed_sleep_events: int
    guard_evaluations: int
    guard_examples: int
    final_clocks: SleepClockSnapshot | None
    retention: ReplayRetentionSnapshot | None
    final_state_sha256: str | None


class CombinedMethodSchema(MethodSchema):
    applied_replay_presentations: int
    applied_replay_inference_loops: int
    rejected_replay_inference_loops: int


class ParentMethodSchema(MethodSchema):
    after_a_state_sha256: str | None
    final_selector: ParentSelectionState | None


class WakeSchema(TypedDict):
    name: str
    width: int
    parameters: int
    parameter_sha256: str
    minimum_plasticity: float | None
    reward_scale: float | None


class ParentWakeSchema(WakeSchema):
    state_sha256: str | None


class CombinedDecisionSchema(TypedDict):
    name: str
    event: SleepEventTelemetry
    state_sha256_before: str
    state_sha256_after: str
    parameter_sha256_before: str
    parameter_sha256_after: str
    lineage_before: NeuronLineageSnapshot
    lineage_after: NeuronLineageSnapshot
    clocks_before: SleepClockSnapshot
    clocks_after: SleepClockSnapshot
    proposed_replay_ids: list[str]
    applied_replay_ids: list[str]


class ParentStateSchema(TypedDict):
    state_sha256: str
    parameter_sha256: str
    lineage: NeuronLineageSnapshot
    clocks: SleepClockSnapshot
    selector: ParentSelectionState


class ParentDecisionSchema(TypedDict):
    name: str
    requested_add_count: int
    split_scores: list[float]
    event: SleepEventTelemetry
    before: ParentStateSchema
    proposed: ParentStateSchema
    after: ParentStateSchema


class OpportunitySchema(TypedDict):
    phase: str
    epoch: int
    global_epoch: int
    train_role_hash: str
    retained_order_ids: list[str]
    retained_examples: int
    retained_bytes: int
    after_epoch_parameter_sha256: dict[str, str]
    after_epoch_state_sha256: dict[str, str]
    after_epoch_widths: dict[str, int]


class CombinedOpportunitySchema(OpportunitySchema):
    wake: list[WakeSchema]
    decisions: list[CombinedDecisionSchema]
    selected_ids: list[str]
    applied_control_replay_ids: dict[str, list[str]]


class ParentOpportunitySchema(OpportunitySchema):
    wake: list[ParentWakeSchema]
    decisions: list[ParentDecisionSchema]


class SeedSchema(TypedDict):
    seed: int
    role_hashes: dict[str, str]
    role_counts: dict[str, int]
    executed_optimizer_updates: int
    final_released: bool


class CombinedSeedSchema(SeedSchema):
    methods: list[CombinedMethodSchema]
    opportunities: list[CombinedOpportunitySchema]


class ParentSeedSchema(SeedSchema):
    methods: list[ParentMethodSchema]
    opportunities: list[ParentOpportunitySchema]


_SCHEMAS = {
    "gating": GatingSeedResult,
    "replay": ReplaySeedResult,
    "sleep": SleepSeedFacts,
    "schedule": ScheduleSeedFacts,
    "combined": CombinedSeedSchema,
    "parent": ParentSeedSchema,
}
_OMITTED = {
    GatingMethodResult: {"development"},
    ReplayMethodResult: {"development"},
    SleepEventTelemetry: {"durations"},
}


def verify_raw_schema(value: dict[str, Any], family: str) -> None:
    _verify_type(value, _SCHEMAS[family], f"{family} raw facts")


def _verify_type(value: Any, annotation: Any, context: str) -> None:
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is int:
        integer(value, context)
    elif annotation is float:
        require(type(value) in (int, float) and isfinite(value), f"{context} number type differs")
    elif annotation in (str, bool, type(None)):
        require(type(value) is annotation, f"{context} scalar type differs")
    elif origin in (Union, UnionType):
        _verify_union(value, args, context)
    elif origin is Literal:
        require(
            any(type(value) is type(expected) and value == expected for expected in args),
            f"{context} literal differs",
        )
    elif origin in (list, tuple):
        _verify_sequence(value, args, context)
    elif origin is dict:
        require(type(value) is dict, f"{context} mapping type differs")
        for key, item in value.items():
            _verify_type(key, args[0], f"{context} key")
            _verify_type(item, args[1], f"{context}/{key}")
    elif isinstance(annotation, type) and (is_dataclass(annotation) or is_typeddict(annotation)):
        hints = get_type_hints(annotation)
        hints = {
            key: hint for key, hint in hints.items() if key not in _OMITTED.get(annotation, set())
        }
        row = object_fields(value, set(hints), context)
        for key, hint in hints.items():
            _verify_type(row[key], hint, f"{context}/{key}")
    else:
        raise ValueError(f"confirmation JSON unsupported raw schema annotation: {annotation}")


def _verify_union(value: Any, annotations: tuple[Any, ...], context: str) -> None:
    for annotation in annotations:
        try:
            _verify_type(value, annotation, context)
            return
        except ValueError:
            continue
    raise ValueError(f"confirmation JSON {context} union type/schema differs")


def _verify_sequence(value: Any, annotations: tuple[Any, ...], context: str) -> None:
    require(type(value) is list, f"{context} JSON sequence type differs")
    if len(annotations) == 1 or annotations[-1] is Ellipsis:
        for item in value:
            _verify_type(item, annotations[0], context)
    else:
        require(len(value) == len(annotations), f"{context} tuple length differs")
        for item, annotation in zip(value, annotations, strict=True):
            _verify_type(item, annotation, context)
