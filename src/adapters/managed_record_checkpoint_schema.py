"""Complete paired wire schema and raw relationships before materialization.

Uses trusted internal schema descriptors for all component fields and immutable
aliases. No wire data selects code. No live reference, port, clock, array or IO.
"""

from math import isfinite
from typing import Any

from src.adapters.lifecycle_checkpoint_schema import SCHEMAS, _walk, preflight_metadata
from src.core.actor_ports import AppliedConsolidation
from src.core.consolidation_codec_policy import ConsolidationCodecPolicy
from src.core.consolidation_cursor import ConsolidationCursor
from src.core.learner_ports import TrainingDiagnostic
from src.core.lifecycle_codec_policy import LifecycleCodecPolicy
from src.core.managed_lifecycle_state import LifecycleMetadata
from src.core.managed_record_codec_policy import ManagedRecordCodecPolicy
from src.core.managed_record_state import ManagedRecordMetadata, RuntimeRecordObservation


def _diagnostic_value(value, limits) -> None:
    if (
        type(value) not in (int, float)
        or (type(value) is int and value.bit_length() > 1024)
        or not isfinite(value)
    ):
        raise ValueError("paired diagnostic requires supported finite native numeric value")


def _inbox_tick(value, limits) -> None:
    if type(value) is not int or not -1 <= value < 2**63:
        raise ValueError("paired inbox tick differs from original supported observation")


PAIR_SCHEMAS = {
    **SCHEMAS,
    ManagedRecordMetadata: dict(
        format_version="version",
        owner=RuntimeRecordObservation,
        consolidation=ConsolidationCursor,
        lifecycle=LifecycleMetadata,
    ),
    RuntimeRecordObservation: dict(
        base_actor_version="string",
        serving_actor_version="string",
        learner_version="string",
        revision="counter",
        consolidation_limit="counter",
        stopped="bool",
        retired="bool",
        payload_ready="bool",
        budget_updates="counter",
        inbox_completed_updates="counter",
        inbox_last_tick=_inbox_tick,
        inbox_stopped="bool",
        enrollment="positive_counter",
    ),
    ConsolidationCursor: dict(
        format_version="version",
        actor_version="string",
        learner_version="string",
        consolidation_limit="counter",
        attempted_ids=("sequence", "string"),
        consolidations=("sequence", AppliedConsolidation),
        stopped="bool",
        retired="bool",
        revision="counter",
        payload_ready="bool",
    ),
    AppliedConsolidation: dict(
        event_id="string",
        actor_version="string",
        learner_version="string",
        attempt_number="positive_counter",
        diagnostic=TrainingDiagnostic,
    ),
    TrainingDiagnostic: dict(definition="string", value=_diagnostic_value),
    ConsolidationCodecPolicy: dict(
        consolidation_limit="counter", max_identifier_bytes="positive_counter"
    ),
    ManagedRecordCodecPolicy: dict(
        lifecycle=LifecycleCodecPolicy, consolidation=ConsolidationCodecPolicy
    ),
}


def _consolidation(data, policy) -> None:
    attempted, receipts = data["attempted_ids"], data["consolidations"]
    if (
        data["consolidation_limit"] != policy.consolidation_limit
        or len(attempted) > policy.consolidation_limit
        or len(receipts) > len(attempted)
        or attempted != sorted(set(attempted))
        or data["actor_version"] == data["learner_version"]
    ):
        raise ValueError("paired consolidation differs from original full allowance/history")
    seen, previous = set(), 0
    for row in receipts:
        if (
            row["event_id"] not in attempted
            or row["event_id"] in seen
            or not previous < row["attempt_number"] <= len(attempted)
            or row["actor_version"] != data["actor_version"]
            or row["learner_version"] != data["learner_version"]
        ):
            raise ValueError("paired original receipt identities/versions/attempt gaps differ")
        seen.add(row["event_id"])
        previous = row["attempt_number"]


def _coupling(data) -> None:
    owner, cursor, life = data["owner"], data["consolidation"], data["lifecycle"]
    holders = [
        item for item in life["registry"]["holders"] if item["enrollment"] == owner["enrollment"]
    ]
    if (
        owner["base_actor_version"] != cursor["actor_version"]
        or owner["learner_version"] != cursor["learner_version"]
        or any(
            owner[name] != cursor[name]
            for name in (
                "revision",
                "consolidation_limit",
                "stopped",
                "retired",
                "payload_ready",
            )
        )
        or owner["budget_updates"] < owner["inbox_completed_updates"]
        or len(holders) != 1
        or holders[0]["kind"] != "candidate"
        or holders[0]["ready"] is not owner["payload_ready"]
        or ((life["lifecycle"]["failed"] or owner["inbox_stopped"]) and not cursor["stopped"])
    ):
        raise ValueError("paired cross-component original observations disagree")


def preflight_pair(data, policy, canonical):
    limits = policy.lifecycle.capture_limits
    nodes: list[tuple[str, type, Any]] = []
    _walk(data, ManagedRecordMetadata, limits, nodes, "metadata", schemas=PAIR_SCHEMAS)
    life, cursor = data["lifecycle"], data["consolidation"]
    owner = life["owner"]
    count = (
        1
        + sum(
            len(owner[name])
            for name in (
                "catalog",
                "opted_out",
                "revoked_keys",
                "declaration_ticks",
                "declaration_seconds",
            )
        )
        + len(life["registry"]["holders"])
        + len(cursor["attempted_ids"])
        + len(cursor["consolidations"])
    )
    if count > limits.max_records:
        raise ValueError("paired wire exceeds original aggregate record capacity")
    preflight_metadata(life, policy.lifecycle, canonical)
    _consolidation(cursor, policy.consolidation)
    _coupling(data)
    return nodes
