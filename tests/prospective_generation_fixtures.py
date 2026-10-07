"""Complete invented declarations, never actual seeds, arrivals or row assignments."""

from typing import Any

from src.app.prospective_confirmation_design import fixed_prospective_design
from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot
from src.core.prospective_role_requests import ProspectiveRoleDeclaration
from src.core.seed_stream_screening import (
    DerivedSeedStream,
    EvidenceIdentity,
    confirmation_seed_streams,
)


def generation_inputs() -> tuple[
    dict[str, Any],
    EvidenceIdentity,
    tuple[ReplicaSeedBinding, ...],
    tuple[DerivedSeedStream, ...],
]:
    design = fixed_prospective_design()
    slots = tuple(ReplicaSlot(**row) for row in design["replica_slots"])
    groups = tuple(dict.fromkeys(slot.source_group for slot in slots))
    bindings = tuple(
        ReplicaSeedBinding(slot, 1_000_000 + 20_000 * groups.index(slot.source_group))
        for slot in slots
    )
    streams = tuple(
        stream
        for group in groups
        for stream in confirmation_seed_streams(
            next(row.base_seed for row in bindings if row.slot.source_group == group)
        )
    )
    return design, EvidenceIdentity(123, "a" * 64), bindings, streams


def concrete_role_declarations(
    design: dict[str, Any], bindings: tuple[ReplicaSeedBinding, ...]
) -> tuple[ProspectiveRoleDeclaration, ...]:
    """Supply claimed after-arrival metadata, without claiming the recipe ran.

    Why this: contiguous invented positions allow full partition/shared-view
    validation under raising source/RNG guards. They are not the live splitter's
    realized positions and cannot demonstrate actual arrival or seeded assignment.
    """
    bases = {row.slot: row.base_seed for row in bindings}
    roles = []
    for row in design["role_requirements"]:
        slot = ReplicaSlot(**row["slot"])
        phase, role = row["phase"], row["role"]
        starts = (
            {"train": 0, "inner_guard": 72, "outer_selection": 96}
            if phase == "a"
            else {"train": 60, "inner_guard": 96, "outer_selection": 108}
        )
        start = 0 if role == "final_test" else starts[role]
        namespace = "final" if role == "final_test" else "development"
        ids = tuple(
            f"phase_{phase}/seed_{bases[slot]}/{namespace}/{index}"
            for index in range(start, start + row["expected_count"])
        )
        roles.append(
            ProspectiveRoleDeclaration(
                slot,
                phase,
                role,
                row["expected_count"],
                ids,
                row["source_available_at"],
                row["labels_available_at"],
                row["allowed_use"],
            )
        )
    return tuple(roles)
