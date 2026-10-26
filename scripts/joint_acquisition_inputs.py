"""Strict saved-receipt intake for the fixed joint-acquisition CLI profile.

Own six bounded original byte payloads, reuse teacher qualification and domain
records, and retain provenance/work metadata. No projection, label generation,
model, simulator, training, authority conversion or artifact-module import occurs.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.run_native_stream_demo import _read_readiness_teacher
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.curated_readiness_corrections import (
    CuratedReadinessBatch,
    CuratedReadinessLabel,
    DEFINITION as CURATED_DEFINITION,
    PROVENANCE as CURATED_PROVENANCE,
    GEOMETRY_DEFINITION,
)
from src.core.pick_cube_action_history import _native_action
from src.core.pick_cube_corrections import _UNITS
from src.core.pick_cube_readiness_corrections import (
    GoalConsistentObservedGraspCorrectionBatch,
    ReadinessCorrectionLabel,
    GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
)
from src.core.pick_cube_readiness_teacher import GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
from src.core.pick_cube_train_context import pick_cube_training_context
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.readiness_stage_diagnostics import ReadinessStageBinding

LIMIT = 1024**2


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON receipt key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON receipt constant: " + value)


def _read(path):
    if not isinstance(path, Path):
        raise ValueError("joint receipt inputs must be Paths")
    with path.open("rb") as handle:
        payload = handle.read(LIMIT + 1)
    if len(payload) > LIMIT:
        raise ValueError("joint receipt exceeds1MiB: " + str(path))
    value = json.loads(payload, object_pairs_hook=_unique, parse_constant=_constant)
    if type(value) is not dict:
        raise ValueError("joint receipt must be a JSON object")
    _json(value)  # Also refuse overflowing finite-literal syntax such as1e999.
    return payload, value


def _same(actual, expected, message):
    if _json(actual) != _json(expected):
        raise ValueError(message)


def _episode(value):
    spec = value["spec"]
    episode = ContinuousEpisode(
        value["episode_id"],
        value["context"],
        value["role"],
        ContinuousActionSpec(
            spec["observation_dim"],
            tuple(spec["action_units"]),
            tuple(spec["lower_bounds"]),
            tuple(spec["upper_bounds"]),
        ),
        tuple(
            ContinuousTransition(
                row["step"],
                tuple(row["observation"]),
                tuple(row["action"]),
                row["reward"],
                tuple(row["next_observation"]),
                row["terminated"],
                row["truncated"],
            )
            for row in value["transitions"]
        ),
    )
    _same(asdict(episode), value, "native episode changed during exact typed decoding")
    for row in episode.transitions:
        _native_action(row.action)
    return episode


def _native(raw, seed, updates, training_rows):
    context = pick_cube_training_context(seed)
    if (
        raw["complete"] is not True
        or raw.get("error") is not None
        or raw.get("traceback") is not None
        or type(raw["seed"]) is not int
        or raw["seed"] != seed
        or (raw["task"], raw["controller"], raw["role"], raw["context"], raw["policy"])
        != ("PickCube-v1", "pd_ee_delta_pos", "train", context, "frozen_continuous_bc")
        or any(
            raw[key] is not True
            for key in ("frozen_state_exact", "original_owner_exact", "action_history_exact")
        )
        or any(
            type(raw[key]) is not int or raw[key] != value
            for key, value in (
                ("source_updates", updates),
                ("source_training_rows", training_rows),
                ("optimizer_updates", 0),
                ("creation_attempts", 1),
                ("creation_returns", 1),
            )
        )
        or raw["explicit_reset"] != f"seed{seed}_if_construction_reaches_port_reset"
        or any(
            raw.get(key) is not None
            for key in (
                "guard_seed",
                "probe_test_seed",
                "scoring_only",
                "mass_context",
                "lighting_context",
                "camera_context",
                "observation_recipe",
                "visual_work",
                "readiness_roll_in",
                "action_decoding",
            )
        )
    ):
        raise ValueError("unsupported original native TRAIN receipt or source counters")
    episode = _episode(raw["episode"])
    if (
        episode.episode_id != f"pick-cube-smoke-{seed}"
        or episode.context != context
        or episode.role != "train"
        or len(episode.transitions) != 50
        or episode.spec.observation_dim != 42
        or episode.spec.action_units != _UNITS
        or episode.spec.lower_bounds != (-1.0,) * 4
        or episode.spec.upper_bounds != (1.0,) * 4
    ):
        raise ValueError("native receipt requires the original fifty-row raw42 TRAIN source")
    work = raw["work"]
    for key, value in (
        ("step_attempts", 50),
        ("completed_steps", 50),
        ("completed_control_physics_steps", 250),
        ("possible_failed_physics_steps", 0),
    ):
        if type(work[key]) is not int or work[key] != value:
            raise ValueError("native source paid control/physics work differs")
    _same(
        work,
        dict(
            step_attempts=50,
            completed_steps=50,
            completed_control_physics_steps=250,
            possible_failed_physics_steps=0,
        ),
        "native work record fields changed",
    )
    rewards = tuple(row.reward for row in episode.transitions)
    if (
        any(value not in (0.0, 1.0) for value in rewards)
        or type(raw["sparse_return"]) not in (int, float)
        or raw["sparse_return"] != sum(rewards)
        or type(raw["any_success"]) is not bool
        or raw["any_success"] is not any(value > 0 for value in rewards)
        or type(raw["final_success"]) is not bool
        or raw["final_success"] is not (rewards[-1] > 0)
    ):
        raise ValueError("native source success metadata differs from sparse rewards")
    history = raw["action_history"]
    closing = 0
    for row in episode.transitions:
        closing = min(6, closing + 1) if row.action[3] < 0 else 0
    _same(
        history["recipe"],
        asdict(PickCubeTaskSpaceHistoryRecipe()),
        "native source history recipe changed",
    )
    _same(
        history["state"],
        dict(previous_action=episode.transitions[-1].action, closing_count=closing),
        "native source executed history state changed",
    )
    _same(
        history["work"],
        dict(
            encoding_attempts=50,
            encoding_returns=50,
            committed_controls=50,
            source_values_validated=2100,
            model_values_emitted=600,
            model_numeric_bytes_emitted=4800,
        ),
        "native source history work changed",
    )
    if history["pending"] is not False:
        raise ValueError("native source contains a pending history input")
    return episode


def _corrections(raw):
    episode = _native(raw, 19, 16, 296)
    for key in ("observed_grasp_corrections", "goal_consistent_corrections"):
        if key in raw:
            _same(raw[key], True, "selected correction mode marker changed: " + key)
    query = raw["corrections"]
    if (
        query.get("error") is not None
        or query.get("traceback") is not None
        or ("complete" in query and query["complete"] is not True)
    ):
        raise ValueError("original correction query contains failed or incomplete evidence")
    expected = dict(
        provenance=GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
        train_seed=19,
        teacher_definition=GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
        geometry_attempts=50,
        geometry_returns=50,
        teacher_action_attempts=50,
        teacher_actions=50,
        teacher_decision_steps=50,
    )
    for key, value in expected.items():
        _same(query[key], value, "original correction query version/count changed: " + key)
    labels = tuple(
        ReadinessCorrectionLabel(
            row["step"], tuple(row["observation"]), tuple(row["target"]), row["stage"]
        )
        for row in query["labels"]
    )
    batch = GoalConsistentObservedGraspCorrectionBatch(
        19, episode, labels, tuple(query["observed_grasp"])
    )
    for label, raw_label in zip(labels, query["labels"], strict=True):
        _same(
            dict(
                asdict(label),
                observed_at=label.observed_at,
                labels_arrived_at=label.labels_arrived_at,
            ),
            raw_label,
            "original B label/arrival metadata changed",
        )
    _same(
        query["batch"],
        dict(train_seed=19, episode_id=episode.episode_id, rows=50, labels_arrived_at=99),
        "original correction batch summary changed",
    )
    batch.training_pairs(now=99)
    return batch


def _curated(source, export, payload):
    episode = _native(source, 17, 24, 816)
    for key in ("observed_grasp_corrections", "goal_consistent_corrections"):
        if key in source:
            _same(source[key], False, "curated source contains an online correction mode")
    if source.get("corrections") is not None or source["sparse_return"] != 0.0:
        raise ValueError("curated source cannot contain online correction labels")
    if (
        export["complete"] is not True
        or export["source_bytes_unchanged"] is not True
        or export.get("error") is not None
        or export.get("traceback") is not None
        or type(export["source_sha256"]) is not str
        or export["source_sha256"] != hashlib.sha256(payload).hexdigest()
    ):
        raise ValueError("curation export does not bind the exact supplied source bytes")
    if type(export["source_path"]) is not str or export["early_release_refused"] is not True:
        raise ValueError("curation source-path metadata or original early release refusal changed")
    for key in (
        "model_initializations",
        "model_updates",
        "model_forwards",
        "model_gradients",
        "new_simulator_controls",
        "new_probe_reads",
        "new_final_reads",
    ):
        if key in export:
            _same(export[key], 0, "original curation zero-work metadata changed: " + key)
    report = export["curation"]
    expected = dict(
        complete=True,
        source_episode_id=episode.episode_id,
        source_context=episode.context,
        source_role="train",
        source_available_at=100,
        started_at=100,
        source_observed_at=[None] * 50,
        teacher_definition=CURATED_DEFINITION,
        provenance=CURATED_PROVENANCE,
        geometry_definition=GEOMETRY_DEFINITION,
        teacher_queries_attempted=50,
        teacher_queries_returned=50,
        work=dict(
            geometry_attempts=50,
            geometry_returns=50,
            source_values_validated=2100,
            teacher_state_steps=50,
        ),
    )
    for key, value in expected.items():
        _same(report[key], value, "original curation metadata/work changed: " + key)
    if report.get("error") is not None:
        raise ValueError("curation export contains a failed partial query")
    labels = tuple(
        CuratedReadinessLabel(
            row["step"],
            tuple(row["observation"]),
            tuple(row["target"]),
            row["stage"],
            row["source_observed_at"],
            row["queried_at"],
            row["arrived_at"],
        )
        for row in report["labels"]
    )
    batch = CuratedReadinessBatch(17, episode, labels, 100)
    _same(
        [asdict(label) for label in batch.labels],
        report["labels"],
        "curated labels changed during exact typed decoding",
    )
    for i, label in enumerate(batch.labels):
        if (
            label.source_observed_at is not None
            or label.queried_at != 100 + 2 * i
            or label.arrived_at != 101 + 2 * i
        ):
            raise ValueError("curation original unknown clocks or fixed query arrivals changed")
    batch.training_pairs(now=199)
    original_work = dict(
        new_teacher_initializations=1,
        teacher_queries_attempted=50,
        teacher_queries_returned=50,
        geometry_rows=50,
        geometry_source_values=2100,
        relative_geometry_values=450,
        relative_geometry_numeric_bytes=3600,
        target_values=200,
        target_numeric_bytes=1600,
        model_projection_rows=50,
        projection_raw_views=100,
        projection_raw_values=4200,
        projection_model_values=1200,
        projection_model_numeric_bytes=9600,
        history_advances=50,
    )
    _same(export["work"], original_work, "original curation export work changed")
    if "grasp_positive_source_rows" in export:
        _same(
            export["grasp_positive_source_rows"],
            sum(row.observation[18] == 1.0 for row in episode.transitions),
            "curation grasp-positive count changed",
        )
    _same(
        export["stage_coverage"],
        {
            stage: sum(label.stage == stage for label in labels)
            for stage in ("align", "descend", "grasp", "lift", "settle")
        },
        "curation stage coverage differs",
    )
    _same(
        export["target_differs_from_command_rows"],
        sum(
            label.target != row.action
            for label, row in zip(labels, episode.transitions, strict=True)
        ),
        "curation actual-command difference count changed",
    )
    if "model_pairs" in export:
        pairs = export["model_pairs"]
        for key, value in dict(
            role="train",
            release_at=199,
            source_steps=list(range(50)),
            targets=[list(label.target) for label in labels],
            actual_actions=[list(row.action) for row in episode.transitions],
            projection="pick_cube_official_state42_taskspace7_executed_history5_v1",
        ).items():
            _same(pairs[key], value, "historical curation model-pair metadata changed: " + key)
        # Historical feature payloads are finite bounded JSON, never training inputs.
        # The typed workflow reprojects original raw episodes with actual commands.
    return batch


def _summary(raw, excluded):
    return {key: deepcopy(value) for key, value in raw.items() if key not in excluded}


def read_joint_acquisition_inputs(
    warm_path: Path,
    new_path: Path,
    corrections_path: Path,
    curated_source_path: Path,
    curated_labels_path: Path,
    retained_path: Path,
):
    """Read exactly six bounded byte payloads; return five owned typed inputs and provenance."""
    paths = (
        warm_path,
        new_path,
        corrections_path,
        curated_source_path,
        curated_labels_path,
        retained_path,
    )
    kinds = (
        "warm_teacher",
        "new_teacher",
        "corrections",
        "curated_source",
        "curated_labels",
        "retained_teacher",
    )
    try:
        loaded = tuple(_read(path) for path in paths)
        payloads = tuple(item[0] for item in loaded)
        raw = tuple(item[1] for item in loaded)
        if any(value.get("train_request_error") is not None for value in raw):
            raise ValueError("joint receipt contains a failed TRAIN request")
        warm, wm = _read_readiness_teacher(warm_path, 17, observed_grasp=True, _payload=payloads[0])
        new, nm = _read_readiness_teacher(new_path, 19, _payload=payloads[1])
        retained, rm = _read_readiness_teacher(
            retained_path, 20, observed_grasp=True, _payload=payloads[5]
        )
        for episode, original in ((warm, raw[0]), (new, raw[1]), (retained, raw[5])):
            _same(
                asdict(episode),
                original["candidate_episode"],
                "qualified teacher episode changed during exact typed decoding",
            )
        corrections = _corrections(raw[2])
        curated = _curated(raw[3], raw[4], payloads[3])
        if tuple(
            map(
                len,
                (
                    warm.transitions,
                    new.transitions,
                    corrections.labels,
                    curated.labels,
                    retained.transitions,
                ),
            )
        ) != (15, 22, 50, 50, 15):
            raise ValueError("joint acquisition receipt counts must be15/22/50/50/15")
        bindings = (
            ReadinessStageBinding(warm, tuple(wm["stages"])),
            ReadinessStageBinding(new, tuple(nm["stages"])),
        )
        inputs = (
            *bindings,
            corrections,
            curated,
            ReadinessStageBinding(retained, tuple(rm["stages"])),
        )
        metadata: dict[str, Any] = dict(
            schema_version=1,
            protocol_id="joint_acquisition_receipts_v1",
            files=[
                dict(
                    kind=kind,
                    path=str(path),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    source_file_bytes=len(payload),
                    per_file_byte_limit=LIMIT,
                )
                for kind, path, payload in zip(kinds, paths, payloads, strict=True)
            ],
            teachers=dict(warm=wm, new=nm, retained=rm),
            corrections=_summary(raw[2]["corrections"], {"labels"}),
            correction_source=_summary(raw[2], {"episode", "corrections"}),
            curated_source=_summary(raw[3], {"episode"}),
            curation=_summary(raw[4], {"curation", "model_pairs"}),
        )
        metadata["curation"]["curation"] = _summary(raw[4]["curation"], {"labels", "batch"})
        if len(_json(metadata).encode("utf8")) > 256 * 1024:
            raise ValueError("joint acquisition receipt metadata exceeds256KiB")
        return inputs, deepcopy(metadata)
    except (
        KeyError,
        TypeError,
        AttributeError,
        UnicodeError,
        RecursionError,
        OverflowError,
    ) as error:
        raise ValueError("malformed fixed joint acquisition receipt") from error
