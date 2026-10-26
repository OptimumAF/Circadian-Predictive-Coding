"""Fixed stage-balanced source-row indices for existing A15 TRAIN packets.

Inputs are a qualified-boundary readiness binding and a bounded update count.
Outputs own the original binding and immutable index packets, never a reordered
trajectory or new labels. No model, array coercion, RNG, label release or IO.
"""

from dataclasses import dataclass
from typing import Any

from src.core.pick_cube_train_context import pick_cube_training_context
from src.core.readiness_stage_diagnostics import ReadinessStageBinding

DEFINITION = "readiness_stage_balanced_a15_packets_v1"
STAGE_ORDER = ("align", "descend", "grasp", "lift", "settle")
ROWS_PER_STAGE = 3
ROWS_PER_PACKET = 15
MAX_PACKETS = 384


def _packet_count(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_PACKETS:
        raise ValueError("readiness packet count must be a native integer in1..384")
    return value


def _own_binding(binding: ReadinessStageBinding) -> ReadinessStageBinding:
    if type(binding) is not ReadinessStageBinding:
        raise ValueError("readiness packets require an exact ReadinessStageBinding")
    # Reuse original flag-before-copy validation and complete episode semantics.
    owned = ReadinessStageBinding(binding.episode, binding.stages)
    episode = owned.episode
    if (
        episode.role != "train"
        or episode.context != pick_cube_training_context(17)
        or episode.episode_id
        not in ("pick-cube-readiness-teacher-17", "pick-cube-observed-grasp-teacher-17")
        or len(episode.transitions) != ROWS_PER_PACKET
    ):
        raise ValueError("readiness packets require the complete original warm17 TRAIN15 binding")
    if any(stage not in owned.stages for stage in STAGE_ORDER):
        raise ValueError("stage-balanced packets require all five original recorded stages")
    return owned


def _stage_indices(binding: ReadinessStageBinding) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {stage: [] for stage in STAGE_ORDER}
    for index, stage in enumerate(binding.stages):
        grouped[stage].append(index)
    return {stage: tuple(indices) for stage, indices in grouped.items()}


def _expected_indices(binding: ReadinessStageBinding, count: int) -> tuple[tuple[int, ...], ...]:
    grouped = _stage_indices(binding)
    # Why: cycle only actual source rows. Rare stages repeat without inventing
    # states; their encoded causal history remains the original row's history.
    return tuple(
        tuple(
            grouped[stage][(ROWS_PER_STAGE * packet + offset) % len(grouped[stage])]
            for stage in STAGE_ORDER
            for offset in range(ROWS_PER_STAGE)
        )
        for packet in range(count)
    )


@dataclass(frozen=True)
class ReadinessPacketPlan:
    binding: ReadinessStageBinding
    indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        binding = _own_binding(self.binding)
        if type(self.indices) is not tuple:
            raise ValueError("readiness packet indices must be an exact tuple")
        count = _packet_count(len(self.indices))
        if any(
            type(packet) is not tuple
            or len(packet) != ROWS_PER_PACKET
            or any(type(index) is not int or not 0 <= index < ROWS_PER_PACKET for index in packet)
            for packet in self.indices
        ):
            raise ValueError("each readiness packet requires15 original native integer indices")
        expected = _expected_indices(binding, count)
        if self.indices != expected:
            raise ValueError("readiness packet indices differ from the fixed stage-cycling rule")
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "indices", expected)


def prepare_readiness_packets(
    binding: ReadinessStageBinding, total_updates: int
) -> ReadinessPacketPlan:
    """Keep15rows/update: three per fixed stage, deterministic across updates."""
    count = _packet_count(total_updates)
    owned = _own_binding(binding)
    return ReadinessPacketPlan(owned, _expected_indices(owned, count))


def readiness_packet_metadata(plan: ReadinessPacketPlan) -> dict[str, Any]:
    """Revalidate the full supplied plan and return detached bounded metadata."""
    if type(plan) is not ReadinessPacketPlan:
        raise ValueError("expected an exact ReadinessPacketPlan")
    owned = ReadinessPacketPlan(plan.binding, plan.indices)
    grouped = _stage_indices(owned.binding)
    count = len(owned.indices)
    selected = count * ROWS_PER_PACKET
    unique = len({index for packet in owned.indices for index in packet})
    return dict(
        definition=DEFINITION,
        source_episode_id=owned.binding.episode.episode_id,
        context=owned.binding.episode.context,
        role=owned.binding.episode.role,
        source_rows=ROWS_PER_PACKET,
        source_stages=tuple(stage for stage in owned.binding.stages),
        stage_order=STAGE_ORDER,
        stage_source_rows={stage: len(grouped[stage]) for stage in STAGE_ORDER},
        rows_per_packet=ROWS_PER_PACKET,
        rows_per_stage=ROWS_PER_STAGE,
        planned_packets=count,
        unique_training_rows_actual=unique,
        repeated_training_rows=selected - unique,
        planned_stage_exposures={stage: ROWS_PER_STAGE * count for stage in STAGE_ORDER},
        retained_index_numeric_bytes=selected * 8,
        source_row_classification_visits=ROWS_PER_PACKET,
        source_index_selections=selected,
        index_writes=selected,
        metadata_union_index_visits=selected,
        scope="logical packet construction and retained int64 index payload; validation, recomputation, ownership copies, Python containers and RSS are additional",
    )
