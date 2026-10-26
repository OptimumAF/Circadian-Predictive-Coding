"""Compose complete public TRAIN sources into bounded joint array wakes.

Inputs are causal raw296/K7 TRAIN episodes and a fixed wake count. Outputs are
owned attribution batches, original source spans and explicit repeated exposure/
array costs. Existing native learners retain projection and objective ownership;
the stream owns delayed label release. The optional fixed arrival recipe reuses
one or explicitly two earlier TRAIN packets per public cue; an explicit ten-arrival
bound also permits three. No native training, evaluation, clock,
RNG, filesystem or new orchestration belongs here.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.app.coupled_attribution import AttributionBatch
from src.core.demonstration_episode import DemonstrationEpisode
from src.core.demonstration_schedule_witness import DemonstrationScheduleWitness
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.core.minigrid_cue_relative_encoding import cue_relative_public_history
from src.core.minigrid_context_encoding import context_public_history

CATALOGUE_DEFINITION = "minigrid_joint_train_catalogue_v1"
MAX_INPUT_ARRAY_BYTES = 16 * 1024**2


@dataclass(frozen=True)
class CatalogueSourceSpan:
    episode_id: str
    context: str
    start: int
    stop: int


@dataclass(frozen=True)
class CatalogueTrainingPlan:
    training: tuple[AttributionBatch, ...]
    source_spans: tuple[CatalogueSourceSpan, ...]
    source_rows: int
    row_exposures: int
    input_array_bytes: int
    source_template_array_bytes: int
    validation_projection_rows: int
    validation_projected_array_equivalent_bytes: int


@dataclass(frozen=True)
class ArrivedCueCatalogueWake:
    observed_at: int
    arrived_at: int
    cue: str
    current_episode_id: str
    retained_episode_id: str | None
    retained_arrived_at: int | None
    current_rows: int
    replay_rows: int
    source_spans: tuple[CatalogueSourceSpan, ...]
    bank_episode_ids_after: tuple[str, ...]
    bank_rows_after: int
    # Legacy scalar fields above identify the most recent prior packet;
    # these complete tuples describe every replay source, oldest first.
    retained_episode_ids: tuple[str, ...] = ()
    retained_arrival_ticks: tuple[int, ...] = ()


@dataclass(frozen=True)
class ArrivedCueCataloguePlan:
    training: tuple[AttributionBatch, ...]
    wakes: tuple[ArrivedCueCatalogueWake, ...]
    arrival_row_exposures: int
    replay_row_exposures: int
    fit_row_exposures: int
    input_array_bytes: int
    peak_joint_template_array_bytes: int
    peak_bank_rows: int
    peak_bank_observation_target_float64_equivalent_bytes: int
    retained_per_cue: int = 1


def _validate_sources(
    episodes: tuple[DemonstrationEpisode, ...], updates: int, *, cue_relative: bool = False
):
    if type(cue_relative) is not bool:
        raise ValueError("cue_relative must be an exact bool")
    if type(updates) is not int or not 1 <= updates <= 128:
        raise ValueError("catalogue updates must be an exact integer in1..128")
    if type(episodes) is not tuple or not 1 <= len(episodes) <= 16:
        raise ValueError("catalogue requires an exact tuple of1..16 complete TRAIN sources")
    witness = DemonstrationScheduleWitness(episodes)
    rows = sum(len(episode.transitions) for episode in episodes)
    if (
        rows > 128
        or len({episode.episode_id for episode in episodes}) != len(episodes)
        or any(
            episode.num_classes != 7 or len(episode.transitions[0].observation) != 296
            for episode in episodes
        )
    ):
        raise ValueError("catalogue requires distinct TRAIN296/K7 sources with at most128 rows")
    if rows * 297 * 8 * updates > MAX_INPUT_ARRAY_BYTES:
        raise ValueError("catalogue retained input arrays exceed16MiB")
    # Why: validate the complete causal schema, including unused next views,
    # before allocating transport arrays. Native learners still project raw inputs.
    project = cue_relative_public_history if cue_relative else canonicalize_public_history
    for episode in episodes:
        initial = episode.transitions[0].observation[:148]
        if initial != episode.transitions[0].observation[148:]:
            raise ValueError("catalogue initial history must equal its first public frame")
        for step in episode.transitions:
            if step.observation[:148] != initial or step.next_observation[:148] != initial:
                raise ValueError("catalogue causal initial public frame changed within episode")
            project(step.observation)
            project(step.next_observation)
    witness.check(episodes)
    return witness, rows


def compose_catalogue_training(
    episodes: tuple[DemonstrationEpisode, ...], updates: int, *, cue_relative: bool = False
) -> CatalogueTrainingPlan:
    """Repeat every original current-view/action row in each joint TRAIN wake.

    Transport IDs distinguish exposures, not novel observations. Context and
    episode provenance remain metadata in half-open source spans; no fictitious
    trajectory is created by joining episodes. Returned arrays are mutable owned
    copies, disjoint between wakes and from the caller. The existing stream will
    own its own copies as usual; those additional allocations are not counted here.
    """
    if type(cue_relative) is not bool:
        raise ValueError("cue_relative must be an exact bool")
    # Why: admission uses the chosen representation, while transport stays raw;
    # the bound learner must apply the non-idempotent cue transform exactly once.
    witness, rows = (
        _validate_sources(episodes, updates, cue_relative=True)
        if cue_relative
        else _validate_sources(episodes, updates)
    )
    spans, offset = [], 0
    features, targets, ids = [], [], []
    for episode in episodes:
        end = offset + len(episode.transitions)
        spans.append(CatalogueSourceSpan(episode.episode_id, episode.context, offset, end))
        for step in episode.transitions:
            features.append(step.observation)
            targets.append((float(step.action),))
            ids.append(f"{episode.episode_id}:step:{step.step}")
        offset = end
    inputs = np.array(features, dtype=np.float64)
    labels = np.array(targets, dtype=np.float64)
    # Why: these are non-recurrent independent rows, not one complete episode.
    # The transport context never enters the native feature arrays.
    batches = tuple(
        AttributionBatch(
            role="train",
            context="train-catalogue",
            row_ids=tuple(f"catalogue:{wake}:{identity}" for identity in ids),
            features=inputs.copy(),
            targets=labels.copy(),
            episode=None,
            num_classes=7,
        )
        for wake in range(updates)
    )
    witness.check(episodes)
    template_bytes = inputs.nbytes + labels.nbytes
    return CatalogueTrainingPlan(
        training=batches,
        source_spans=tuple(spans),
        source_rows=rows,
        row_exposures=rows * updates,
        input_array_bytes=template_bytes * updates,
        source_template_array_bytes=template_bytes,
        validation_projection_rows=2 * rows,
        validation_projected_array_equivalent_bytes=2 * rows * 296 * 8,
    )


def compose_arrived_cue_catalogue(
    episodes: tuple[DemonstrationEpisode, ...],
    *,
    retained_per_cue: int = 1,
    max_arrivals: int = 6,
) -> ArrivedCueCataloguePlan:
    """Delay-one recipe: bounded earlier episodes plus current, fitted jointly.

    Application holds the private finite plan. These are independent row arrays,
    not joined trajectories or a live replay service. The default six-arrival
    bound permits at most two earlier packets/cue. Explicit max_arrivals=10
    permits three, at most54bank rows and36rows per fit. Default retention stays
    one packet/cue (18rows per fit). No future labels reach a wake.
    Numeric bank equivalents include current+next observations and targets;
    episode objects are shared with the source schedule, not new numeric banks.
    """
    # Why: larger recurrence is opt-in; existing callers retain six-arrival
    # admission and do not silently gain a larger data or replay allowance.
    if type(max_arrivals) is not int or max_arrivals not in (6, 10):
        raise ValueError("max_arrivals must be an exact integer6 or10")
    maximum_retained = 3 if max_arrivals == 10 else 2
    if type(retained_per_cue) is not int or not 1 <= retained_per_cue <= maximum_retained:
        raise ValueError(f"retained_per_cue must be an exact integer1..{maximum_retained}")
    if type(episodes) is not tuple or not 1 <= len(episodes) <= max_arrivals:
        raise ValueError(f"arrived cue catalogue requires1..{max_arrivals} complete TRAIN packets")
    witness, _ = _validate_sources(episodes, 1)
    if any(len(episode.transitions) != 9 for episode in episodes):
        raise ValueError("arrived cue catalogue requires exactly nine rows per packet")
    previous: list[list[tuple[int, DemonstrationEpisode]]] = [[], []]
    training, wakes = [], []
    peak_bank_rows = peak_template_bytes = input_bytes = 0
    for tick, episode in enumerate(episodes):
        witness.check(episodes)
        cue = int(context_public_history(episode.transitions[0].observation)[297])
        retained = tuple(previous[cue])
        old_arrivals = tuple(item[0] + 1 for item in retained)
        if any(arrival > tick for arrival in old_arrivals):
            raise ValueError("retained cue labels have not arrived by observation time")
        sources = tuple(item[1] for item in retained) + (episode,)
        # Raw296 observations plus one target, nine rows/source, float64. The
        # cumulative bound protects all owned wake arrays, not only one template.
        next_array_bytes = len(sources) * 9 * 297 * 8
        if input_bytes + next_array_bytes > MAX_INPUT_ARRAY_BYTES:
            raise ValueError("arrived catalogue cumulative input arrays exceed16MiB")
        joint = compose_catalogue_training(sources, 1)
        witness.check(episodes)
        batch = joint.training[0]
        # Why: repeated rows are new exposures, not new observations. Keep each
        # wake's transport IDs unique; source spans preserve original packet IDs.
        training.append(
            replace(batch, row_ids=tuple(f"arrived:{tick}:{key}" for key in batch.row_ids))
        )
        previous[cue] = [*retained, (tick, episode)][-retained_per_cue:]
        bank = tuple(item[1] for group in previous for item in group)
        bank_rows = sum(len(item.transitions) for item in bank)
        peak_bank_rows = max(peak_bank_rows, bank_rows)
        peak_template_bytes = max(peak_template_bytes, joint.source_template_array_bytes)
        input_bytes += joint.input_array_bytes
        wakes.append(
            ArrivedCueCatalogueWake(
                tick,
                tick + 1,
                ("blue", "red")[cue],
                episode.episode_id,
                None if not retained else retained[-1][1].episode_id,
                None if not retained else old_arrivals[-1],
                9,
                9 * len(retained),
                joint.source_spans,
                tuple(item.episode_id for item in bank),
                bank_rows,
                tuple(item[1].episode_id for item in retained),
                old_arrivals,
            )
        )
    witness.check(episodes)
    arrival_rows = len(episodes) * 9
    replay_rows = sum(wake.replay_rows for wake in wakes)
    return ArrivedCueCataloguePlan(
        tuple(training),
        tuple(wakes),
        arrival_rows,
        replay_rows,
        arrival_rows + replay_rows,
        input_bytes,
        peak_template_bytes,
        peak_bank_rows,
        peak_bank_rows * (2 * 296 + 1) * 8,
        retained_per_cue,
    )
