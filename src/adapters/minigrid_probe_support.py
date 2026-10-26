"""Describe TRAIN input bounds and finite probe support without learner work.

Inputs are actual128 raw nine-row TRAIN packets, or explicitly the six fixed
original TRAIN sources, and the fixed16+2 raw probe cohort. Outputs are detached
per-episode raw/canonical296 support counts and
scoped logical key costs. Labels only measure support agreement; they never
construct features. This helper supplies no novelty/generalization proof, model,
update authority, physics, final-data access, persistence or probe selection.
"""

from time import monotonic
from dataclasses import asdict
import math
import numpy as np
from typing import Any

from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.adapters.minigrid_train_catalogue import _validate_sources
from src.core.categorical_input_bounds import categorical_input_bounds
from src.core.demonstration_episode import DemonstrationEpisode
from src.core.minigrid_public_history import encode_public_history_episode
from src.core.minigrid_public_history import PublicInitialFrameHistory
from src.core.minigrid_context_encoding import context_public_history, CONTEXT_HISTORY_ENCODING_ID
from src.core.demonstration_episode import episode_arrays
from src.core.demonstration_schedule_witness import DemonstrationScheduleWitness
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.core.minigrid_cue_relative_encoding import (
    CUE_RELATIVE_HISTORY_ENCODING_ID,
    cue_relative_public_history,
)


def _validate_packet(packet: AttributionBatch, role: str, seen: set[str]) -> None:
    if type(packet) is not AttributionBatch:
        raise ValueError("probe support requires exact AttributionBatch records")
    _validate_batch(packet, role, seen)
    if (
        packet.features.shape != (9, 148)
        or packet.num_classes != 7
        or packet.episode is None
        or len(packet.episode.transitions) != 9
    ):
        raise ValueError("probe support requires complete raw nine-row148/K7 packets")


def _probe_pair(data) -> tuple[str, str]:
    if (
        type(data) is not tuple
        or len(data) != 2
        or type(data[0]) is not tuple
        or type(data[1]) is not tuple
        or len(data[0]) != 16
        or len(data[1]) != 2
    ):
        raise ValueError("probe support requires the fixed16+2 packet tuple")
    if any(type(packet) is not AttributionBatch for packets in data for packet in packets):
        raise ValueError("probe support requires exact AttributionBatch records")
    pair = tuple(packet.context for packet in data[1])
    if pair not in (("E-blue", "E-red"), ("F-blue", "F-red")):
        raise ValueError("probe support requires one fixed E or F cue pair")
    if tuple(packet.context for packet in data[0]) != pair * 8:
        raise ValueError("probe support requires uniform alternating E or F contexts")
    return pair[0], pair[1]


def _rows(packet: AttributionBatch):
    if packet.episode is None:
        raise ValueError("probe support requires complete episode metadata")
    episode = encode_public_history_episode(packet.episode)
    rows = tuple(step.observation for step in episode.transitions)
    canonical = tuple(canonicalize_public_history(row) for row in rows)
    return rows, canonical, tuple(step.action for step in episode.transitions)


def _table_summary(table) -> dict[str, int]:
    pairs = sum(len(labels) for labels in table.values())
    return dict(
        unique_input_keys=len(table),
        conflicting_training_keys=sum(len(labels) > 1 for labels in table.values()),
        distinct_input_label_pairs=pairs,
        logical_key_payload_bytes=len(table) * 2368,
        logical_label_payload_bytes=pairs * 8,
    )


def _support(rows, labels, table) -> dict[str, int]:
    matched = sum(row in table for row in rows)
    label_matches = sum(
        label in table.get(row, ()) for row, label in zip(rows, labels, strict=True)
    )
    return dict(
        matched_input_rows=matched,
        unmatched_input_rows=len(rows) - matched,
        matched_label_rows=label_matches,
        unsupported_label_rows=len(rows) - label_matches,
        conflicting_training_key_rows=sum(len(table.get(row, ())) > 1 for row in rows),
    )


def summarize_probe_support(
    training_packets, probe_data, *, catalogue_sources: bool = False
) -> dict[str, Any]:
    """Measure actual exposed keys; return all18 probe cells without filtering."""
    started = monotonic()
    if type(catalogue_sources) is not bool:
        raise ValueError("catalogue_sources must be an exact bool")
    count = 6 if catalogue_sources else 128
    if type(training_packets) is not tuple or len(training_packets) != count:
        if catalogue_sources:
            raise ValueError("catalogue support requires all six original TRAIN source episodes")
        raise ValueError("probe support requires all actual128 TRAIN packets")
    _probe_pair(probe_data)
    seen: set[str] = set()
    for index, packet in enumerate(training_packets):
        _validate_packet(packet, "train", seen)
        if packet.context not in ("A", "B", "C_train"):
            raise ValueError("probe support TRAIN contexts must be A/B/C_train")
        if catalogue_sources and (
            packet.episode is None
            or packet.context != ("A", "B", "C_train")[index // 2]
            or packet.episode.episode_id
            != f"minigrid-source-{packet.context}-{('blue', 'red')[index % 2]}"
        ):
            raise ValueError("catalogue support requires the fixed six-source context/cue order")
    for packets, role in zip(probe_data, ("probe_train", "probe_test"), strict=True):
        for packet in packets:
            _validate_packet(packet, role, seen)
    tables: tuple[dict[Any, set[int]], dict[Any, set[int]]] = ({}, {})
    context_rows: dict[str, int] = {}
    for packet in training_packets:
        raw, canonical, labels = _rows(packet)
        context_rows[packet.context] = context_rows.get(packet.context, 0) + 9
        for rows, table in zip((raw, canonical), tables, strict=True):
            for row, label in zip(rows, labels, strict=True):
                table.setdefault(row, set()).add(label)
    reports = []
    for packets in probe_data:
        for packet in packets:
            raw, canonical, labels = _rows(packet)
            reports.append(
                dict(
                    episode_id=packet.episode.episode_id,
                    role=packet.role,
                    context=packet.context,
                    rows=9,
                    raw=_support(raw, labels, tables[0]),
                    canonical=_support(canonical, labels, tables[1]),
                )
            )
    all_packets = (*training_packets, *probe_data[0], *probe_data[1])
    summaries = tuple(_table_summary(table) for table in tables)
    return dict(
        schema_id="minigrid_probe_input_support_v1",
        status="complete",
        training=dict(
            packets=count,
            rows=count * 9,
            context_rows=context_rows,
            raw=summaries[0],
            canonical=summaries[1],
        ),
        probe=dict(training_packets=16, evaluation_packets=2, rows=162, packets=reports),
        all_canonical_inputs_unseen=all(
            record["canonical"]["unmatched_input_rows"] == 9 for record in reports
        ),
        work=dict(
            native_updates=0,
            native_score_forwards=0,
            history_rows=(count + 18) * 9,
            canonical_projection_rows=(count + 18) * 9,
            canonical_projected_array_equivalent_bytes=(count + 18) * 9 * 2368,
            input_array_bytes=sum(
                packet.features.nbytes + packet.targets.nbytes for packet in all_packets
            ),
            training_logical_key_payload_bytes=sum(
                value["logical_key_payload_bytes"] for value in summaries
            ),
            training_logical_label_payload_bytes=sum(
                value["logical_label_payload_bytes"] for value in summaries
            ),
            elapsed_seconds=monotonic() - started,
        ),
        memory_scope="input arrays are existing caller cohorts; keys/label sets and complete causal history records transient; logical payload counts only float64 key coordinates/int64 distinct labels, not Python objects/copies/next-frame records; actual peak RSS/failed allocations additional or unknown",
    )


TRAIN_BOUNDS_DEFINITION = "minigrid_catalogue_train_input_bounds_v1"
CUE_RELATIVE_TRAIN_BOUNDS_DEFINITION = "minigrid_cue_relative_catalogue_train_input_bounds_v1"


def describe_train_fit_bounds(*, cue_relative: bool = False) -> dict[str, Any]:
    """No data/model access: state the empirical normalized-simplex question."""
    if type(cue_relative) is not bool:
        raise ValueError("cue_relative must be an exact bool")
    report = dict(
        definition=TRAIN_BOUNDS_DEFINITION,
        role="train",
        source_episodes=6,
        rows=54,
        inputs="original causal public296 current rows; raw and canonical exact float64 numerical keys; signed zeros equal",
        accuracy="sum maximum class frequencies per identical key /54; attainable by an unrestricted deterministic lookup",
        ce="conditional label entropy per key /54; normalized-simplex unweighted probability CE floor; absent classes cost0",
        gaps="original unweighted row accuracy ceiling minus observed accuracy; observed CE minus normalized floor; signed, not clamped",
        scope="empirical TRAIN row bounds only; not H8 attainability, convergence, exact-episode, live, retention or generalization bounds; probability validation allows1e-6 sum roundoff",
        native_updates=0,
        native_score_forwards=0,
        fit_calls=0,
        input_bound_reductions=2,
        input_bound_rows=108,
        admission_projection_rows=108,
        bound_projection_rows=54,
    )
    if cue_relative:
        report.update(
            definition=CUE_RELATIVE_TRAIN_BOUNDS_DEFINITION,
            encoding_id=CUE_RELATIVE_HISTORY_ENCODING_ID,
            inputs="original causal public296 current rows; raw and cue-relative canonical exact float64 numerical keys; signed zeros equal",
            input_contract="raw public296; apply cue-relative projection exactly once; never project already encoded input",
            representation_scope="retained initial cue makes key-color remapping bijective relative to wall/direction canonical inputs; no learnability or convergence proof",
        )
    return report


def summarize_train_fit_bounds(
    episodes: tuple[DemonstrationEpisode, ...], *, cue_relative: bool = False
) -> dict[str, Any]:
    """Read only the fixed complete TRAIN catalogue; no probe/final argument exists."""
    if type(cue_relative) is not bool:
        raise ValueError("cue_relative must be an exact bool")
    if type(episodes) is not tuple or len(episodes) != 6:
        raise ValueError("TRAIN input bounds require exactly six original source episodes")
    # Reuse full role/schema/current+next-view admission before allocating arrays.
    witness, rows = (
        _validate_sources(episodes, 1, cue_relative=True)
        if cue_relative
        else _validate_sources(episodes, 1)
    )
    expected = tuple((context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red"))
    if rows != 54 or any(
        episode.context != context
        or episode.episode_id != f"minigrid-source-{context}-{cue}"
        or len(episode.transitions) != 9
        for episode, (context, cue) in zip(episodes, expected, strict=True)
    ):
        raise ValueError(
            "TRAIN input bounds require original ordered A/B/C blue/red nine-row sources"
        )
    started = monotonic()
    raw = np.array(
        [step.observation for episode in episodes for step in episode.transitions], dtype=np.float64
    )
    project = cue_relative_public_history if cue_relative else canonicalize_public_history
    canonical = np.array(
        [project(tuple(float(value) for value in row)) for row in raw],
        dtype=np.float64,
    )
    targets = np.array(
        [step.action for episode in episodes for step in episode.transitions], dtype=np.float64
    ).reshape(-1, 1)
    raw_bounds = categorical_input_bounds(raw, targets, 7)
    witness.check(episodes)
    canonical_bounds = categorical_input_bounds(canonical, targets, 7)
    elapsed = monotonic() - started
    witness.check(episodes)
    protocol = (
        describe_train_fit_bounds(cue_relative=True)
        if cue_relative
        else describe_train_fit_bounds()
    )
    report = dict(
        definition=protocol["definition"],
        status="complete",
        role="train",
        source_episode_ids=[episode.episode_id for episode in episodes],
        raw=asdict(raw_bounds),
        canonical=asdict(canonical_bounds),
        source_unchanged=True,
        protocol=protocol,
        work=dict(
            native_updates=0,
            native_score_forwards=0,
            fit_calls=0,
            input_bound_reductions=2,
            input_bound_rows=108,
            admission_projection_rows=108,
            bound_projection_rows=54,
            projected_array_equivalent_bytes=162 * 296 * 8,
            input_array_bytes=raw.nbytes + canonical.nbytes + targets.nbytes,
            logical_key_payload_bytes=(
                raw_bounds.unique_input_keys + canonical_bounds.unique_input_keys
            )
            * 296
            * 8,
            logical_count_payload_bytes=(
                raw_bounds.unique_input_keys + canonical_bounds.unique_input_keys
            )
            * 7
            * 8,
            elapsed_seconds=elapsed,
            elapsed_seconds_scope="after initial complete admission through reductions/closing clock, before final witness check; enclosing invocation wall/RSS includes all admission/copy/validation cost",
            scope="completed reductions only; current/next admission and owned arrays counted; Python/transient tuples/validation/failed work additional or unknown, measured wall/RSS outside numeric counts",
        ),
    )
    if cue_relative:
        report.update(
            encoding_id=CUE_RELATIVE_HISTORY_ENCODING_ID,
            input_contract=protocol["input_contract"],
        )
    return report


def validate_train_fit_bounds_option(flag: bool, joint_catalogue: bool) -> None:
    if type(flag) is not bool or (flag and joint_catalogue is not True):
        raise ValueError("train_fit_bounds requires an exact bool and joint_catalogue=True")


def describe_context_train_bounds() -> dict[str, Any]:
    """Describe a readonly extension of the original fixed TRAIN bounds."""
    return dict(
        definition="minigrid_context_train_alignment_bounds_v1",
        role="train",
        encoding_id=CONTEXT_HISTORY_ENCODING_ID,
        source_episodes=6,
        rows=54,
        reductions=4,
        reduction_rows=162,
        canonical_projection_rows=216,
        context_projection_rows=54,
        scope="exact TRAIN input conflicts and current observation/action alignment; unrestricted lookup ceilings do not prove linear attainability, convergence or live success",
        native_updates=0,
        native_score_forwards=0,
        fit_calls=0,
    )


def summarize_context_train_bounds(episodes: tuple[DemonstrationEpisode, ...]) -> dict[str, Any]:
    """Reuse catalogue bounds, grouping only by the initial public cue.

    Inputs are the original six causal raw296 TRAIN sources. Outputs add owned
    row provenance, conflict witnesses and per-head bounds. No model, new labels,
    saved checkpoint, next-view input, game or fitting callback is accepted.
    """
    witness = DemonstrationScheduleWitness(episodes)
    original = summarize_train_fit_bounds(episodes)
    groups: dict[str, list[tuple[tuple[float, ...], int, dict[str, Any]]]] = {"blue": [], "red": []}
    for episode in episodes:
        features, targets = episode_arrays(episode)
        history = PublicInitialFrameHistory()
        history.reset(episode.transitions[0].observation[:148])
        for index, step in enumerate(episode.transitions):
            row = tuple(float(v) for v in features[index])
            # Why: exercise the same reset/current-history port used by live
            # scoring; a next-view shift must fail even if shapes still match.
            if row != history.encode(step.observation[148:]) or targets[index, 0] != step.action:
                raise ValueError("TRAIN current observation/action alignment changed")
            projected = context_public_history(row)
            cue = "blue" if projected[-2:] == (1.0, 0.0) else "red"
            groups[cue].append(
                (
                    projected[:296],
                    step.action,
                    dict(
                        episode_id=episode.episode_id,
                        step=step.step,
                        action=step.action,
                        public_cue=cue,
                        named_cue_matches=episode.episode_id.endswith(f"-{cue}"),
                    ),
                )
            )
    heads = {}
    for cue, rows in groups.items():
        if not rows:
            raise ValueError("context TRAIN bounds require both public cue heads")
        inputs = np.array([row[0] for row in rows], dtype=np.float64)
        targets = np.array([[row[1]] for row in rows], dtype=np.float64)
        bounds = categorical_input_bounds(inputs, targets, 7)
        keys: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        for key, _, provenance in rows:
            keys.setdefault(key, []).append(provenance)
        conflicts = [value for value in keys.values() if len({r["action"] for r in value}) > 1]
        heads[cue] = dict(bounds=asdict(bounds), rows=[row[2] for row in rows], conflicts=conflicts)
    witness.check(episodes)
    return dict(
        definition=describe_context_train_bounds()["definition"],
        status="complete",
        role="train",
        protocol=describe_context_train_bounds(),
        original_bounds=original,
        heads=heads,
        current_action_alignment=True,
        source_unchanged=True,
        work=dict(
            native_updates=0,
            native_score_forwards=0,
            fit_calls=0,
            input_bound_reductions=4,
            input_bound_rows=162,
            canonical_projection_rows=216,
            context_projection_rows=54,
        ),
    )


def fit_gap_outcomes(bounds: dict[str, Any], points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Relate existing equal-row TRAIN scores to a theoretical input-key bound."""
    if (
        bounds.get("definition")
        not in (TRAIN_BOUNDS_DEFINITION, CUE_RELATIVE_TRAIN_BOUNDS_DEFINITION)
        or (
            bounds.get("definition") == CUE_RELATIVE_TRAIN_BOUNDS_DEFINITION
            and bounds.get("encoding_id") != CUE_RELATIVE_HISTORY_ENCODING_ID
        )
        or bounds.get("status") != "complete"
        or bounds.get("role") != "train"
        or bounds["canonical"]["rows"] != 54
    ):
        raise ValueError("fit gaps require complete fixed TRAIN input bounds")
    ceiling = bounds["canonical"]["row_accuracy_upper_bound"]
    floor = bounds["canonical"]["normalized_ce_lower_bound"]
    result = []
    for point in points:
        # All six episodes have nine rows; equal context/cue means equal row mass.
        accuracy = sum(point[context]["accuracy"] for context in ("A", "B", "C_train")) / 3
        ce = sum(point[context]["ce"] for context in ("A", "B", "C_train")) / 3
        if not math.isfinite(accuracy) or not math.isfinite(ce) or not 0 <= accuracy <= 1 or ce < 0:
            raise ValueError("fit gaps require finite unweighted TRAIN scores")
        result.append(
            dict(
                updates=point["updates"],
                row_accuracy_upper_bound=ceiling,
                normalized_ce_lower_bound=floor,
                observed_accuracy=accuracy,
                observed_ce=ce,
                accuracy_gap=ceiling - accuracy,
                ce_gap=ce - floor,
            )
        )
    return result
