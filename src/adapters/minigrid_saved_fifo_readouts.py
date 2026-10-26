"""Read the missing fixed TRAIN cells from five actual FIFO36 checkpoints.

Inputs are decoded, owned historical snapshots and original teacher DTOs. Outputs
are separately charged readouts; historical failures stay failures. This adapter
never trains, restores an old budget, teaches a live actor, selects or publishes.
"""

from copy import deepcopy
from dataclasses import asdict
from math import isfinite
import pickle
import random
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    copy_invariant_memory_snapshot,
)
from src.adapters.minigrid_learning_pilot import _batch, _signature, _check_signature
from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    _return_budget_guard,
    evaluate_invariant_fifo_checkpoint,
)
from src.adapters.minigrid_saved_fifo_inputs import SavedFifoInputs
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.coupled_attribution import _same_value
from src.core.demonstration_episode import copy_episode
from src.core.fifo_action_memory import FifoActionMemoryPolicy
from src.core.minigrid_key_door import predefined_layouts
from src.shared.process_memory import ProcessRssSampler

BODY_SECONDS = 120.0
RSS_LIMIT = 768 * 1024**2
PAIRS = tuple((context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red"))
POINTS = (0, 32, 64, 96, 128)


def describe_protocol() -> dict[str, Any]:
    return dict(
        protocol_id="minigrid_saved_fifo36_pending_train_readouts_v1",
        classification="exploratory_known_train_saved_count_control",
        seed=17,
        source_points=list(POINTS),
        new_training_updates=0,
        new_training_rows=0,
        replay_examples=0,
        teacher_episodes=0,
        planned_work=dict(
            raw_score_forwards=12,
            raw_score_rows=108,
            live_episodes=16,
            maximum_live_predictions=256,
            maximum_environment_steps=256,
            maximum_total_prediction_calls=268,
            maximum_total_prediction_rows=364,
        ),
        limits=dict(body_seconds=BODY_SECONDS, process_rss_bytes=RSS_LIMIT, cpu_threads=1, gpu=0),
        limitations="Known TRAIN recall only. Original incomplete/spent run remains incomplete; supplied JSON is not authenticated historical provenance. No neural, heldout or CPC advantage claim.",
    )


def _admit(inputs: SavedFifoInputs) -> None:
    if type(inputs) is not SavedFifoInputs:
        raise ValueError("saved FIFO readout requires exact decoded inputs")
    score_slots = tuple((point, *pair) for point in (96, 128) for pair in PAIRS)
    game_slots = tuple(
        (point, *pair)
        for point, pairs in ((64, PAIRS[2:]), (96, PAIRS), (128, PAIRS))
        for pair in pairs
    )
    completed = tuple(
        (point, *pair)
        for point, pairs in ((0, PAIRS), (32, PAIRS), (64, PAIRS[:2]))
        for pair in pairs
    )
    if (
        inputs.score_slots != score_slots
        or inputs.game_slots != game_slots
        or inputs.completed_games != completed
        or inputs.censored_games != ((64, "B", "blue"),)
        or inputs.origin_status != "incomplete"
        or inputs.original_training_updates != 128
        or inputs.original_training_rows != 1152
    ):
        raise ValueError("saved FIFO pending cells differ from the frozen original incomplete case")
    if type(inputs.snapshots) is not tuple or len(inputs.snapshots) != 5:
        raise ValueError("five complete FIFO36 snapshots required")
    policy = FifoActionMemoryPolicy(
        feature_dim=296, num_classes=7, max_entries=36, max_payload_bytes=87264
    )
    for point, state in zip(POINTS, inputs.snapshots, strict=True):
        owned = copy_invariant_memory_snapshot(state)
        if (
            owned.policy.native_policy != policy
            or owned.completed_updates != point
            or owned.training_examples != 9 * point
        ):
            raise ValueError("saved FIFO checkpoint policy or native clock differs")
    if type(inputs.episodes) is not tuple or len(inputs.episodes) != 6:
        raise ValueError("six original TRAIN episodes required")
    for episode, (context, cue) in zip(inputs.episodes, PAIRS, strict=True):
        owned_episode = copy_episode(episode)
        if (
            owned_episode.episode_id != f"minigrid-source-{context}-{cue}"
            or owned_episode.context != context
            or owned_episode.role != "train"
            or owned_episode.num_classes != 7
            or len(owned_episode.transitions) != 9
            or any(len(t.observation) != 148 for t in owned_episode.transitions)
        ):
            raise ValueError("original FIFO TRAIN episode differs from the fixed cohort")


def _validate_result(result: Any, point: int, scores: tuple, games: tuple) -> None:
    if (
        type(result) is not dict
        or result.get("status") != "complete"
        or result.get("state_unchanged") is not True
        or result.get("updates") != point
    ):
        raise ValueError("preceding saved FIFO readout did not complete with preserved state")
    if (
        tuple((row.get("context"), row.get("cue")) for row in result.get("scores", ())) != scores
        or tuple((row.get("context"), row.get("cue")) for row in result.get("episodes", ()))
        != games
        or any(
            row.get("status") != "complete" or row.get("state_unchanged") is not True
            for row in result["episodes"]
        )
    ):
        raise ValueError("saved FIFO readout returned different or incomplete selected cells")


def run_pilot(inputs: SavedFifoInputs, *, max_seconds: float = BODY_SECONDS) -> dict[str, Any]:
    """Compose the existing fixed evaluator with a new zero-update allowance."""
    if (
        type(max_seconds) not in (int, float)
        or not isfinite(max_seconds)
        or not 0 < max_seconds <= BODY_SECONDS
    ):
        raise ValueError("saved FIFO max_seconds must be finite in (0, 120]")
    _admit(inputs)
    started = monotonic()
    original = pickle.dumps(inputs, protocol=5)
    owned = deepcopy(inputs)
    sources = tuple(_batch(episode) for episode in owned.episodes)
    signature = _signature(sources)
    source = InvariantActionMemoryLearner(owned.snapshots[0].policy)
    python_rng = random.getstate()
    numpy_rng = pickle.dumps(np.random.get_state(), protocol=5)
    report: dict[str, Any] = dict(
        protocol=describe_protocol(),
        status="running",
        reason=None,
        readouts=[],
        origin=dict(
            status=owned.origin_status,
            reason=owned.origin_reason,
            elapsed_seconds=owned.original_elapsed_seconds,
            training_updates=owned.original_training_updates,
            training_rows=owned.original_training_rows,
            budget_status="spent; never restored",
        ),
        completed_games=[list(slot) for slot in owned.completed_games],
        censored_games=[list(slot) for slot in owned.censored_games],
        new_training_updates=0,
        new_training_rows=0,
        replay_examples=0,
        scientific_advantage_verified=False,
    )
    metadata = {
        key: value for key, value in report.items() if key not in ("readouts", "status", "reason")
    }
    metadata_bytes = pickle.dumps(metadata, protocol=5)
    metadata_ids = {key: id(value) for key, value in metadata.items()}
    readouts: list[dict[str, Any]] = report["readouts"]
    protected: list[tuple[dict[str, Any], bytes]] = []
    pending: dict[str, Any] | None = None
    opaque_active = False
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=float(max_seconds),
            max_process_rss_bytes=RSS_LIMIT,
        ),
        monotonic,
    )
    budget_guard = None
    sampler = None

    def evidence_guard() -> None:
        if (
            pickle.dumps(inputs, protocol=5) != original
            or pickle.dumps(owned, protocol=5) != original
        ):
            raise ValueError("saved FIFO input evidence changed")
        if not _check_signature(sources, signature):
            raise ValueError("original FIFO TRAIN source packets changed")
        if (
            random.getstate() != python_rng
            or pickle.dumps(np.random.get_state(), protocol=5) != numpy_rng
        ):
            raise ValueError("saved FIFO readout changed global RNG state")
        if (
            any(
                key not in report or id(report[key]) != identity
                for key, identity in metadata_ids.items()
            )
            or pickle.dumps({key: report[key] for key in metadata}, protocol=5) != metadata_bytes
        ):
            raise ValueError("saved FIFO original provenance or protocol changed")
        if report["readouts"] is not readouts or len(readouts) != len(protected):
            raise ValueError("preceding FIFO readout publication was replaced")
        for actual, (expected, payload) in zip(readouts, protected, strict=True):
            if actual is not expected or pickle.dumps(actual, protocol=5) != payload:
                raise ValueError("preceding FIFO readout was replaced or corrupted")
        if budget_guard is not None:
            budget_guard()

    def guard() -> None:
        evidence_guard()
        budget.before_final()
        evidence_guard()

    elapsed = None
    observed_rss = None
    try:
        with ProcessRssSampler() as active_sampler:
            sampler = active_sampler
            budget.attach_memory(sampler)
            budget_guard = _return_budget_guard(budget)
            for point, state in zip(POINTS, owned.snapshots, strict=True):
                scores = tuple(
                    (context, cue) for p, context, cue in owned.score_slots if p == point
                )
                games = tuple((context, cue) for p, context, cue in owned.game_slots if p == point)
                if not scores and not games:
                    continue
                guard()
                opaque_active = True
                try:
                    result = evaluate_invariant_fifo_checkpoint(
                        source,
                        state,
                        updates=point,
                        layouts=predefined_layouts()[:3],
                        role="train",
                        sources=sources,
                        budget=budget,
                        check_boundary=guard,
                        score_pairs=scores,
                        game_pairs=games,
                    )
                except MiniGridEvaluationFailed as error:
                    pending = deepcopy(error.report)
                    raise
                opaque_active = False
                pending = deepcopy(result)  # Paid result survives a subsequent guard failure.
                guard()
                _validate_result(result, point, scores, games)
                readouts.append(result)
                protected.append((result, pickle.dumps(result, protocol=5)))
                pending = None
                guard()
            guard()
        # Why: shutdown, RSS readback and the final clock are also opaque ports.
        # Seal after them before rendering private paid records as a success.
        guard()
        if not _same_value(source.snapshot_state(), owned.snapshots[0]):
            raise ValueError("saved FIFO initial source changed during closing readback")
        evidence_guard()
        observed_rss = sampler.snapshot()
        elapsed = monotonic() - started
        evidence_guard()
        if not isfinite(elapsed) or elapsed < 0 or elapsed >= max_seconds:
            raise ValueError("saved FIFO whole readout wall allowance exhausted")
        if observed_rss.peak_bytes > RSS_LIMIT:
            raise ValueError("saved FIFO final sampled RSS allowance exhausted")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
        # The budget records a returned RSS segment before refusing its ceiling.
        # Reuse that paid observation without invoking a broken source again.
        if observed_rss is None:
            observed_rss = budget.process_rss_segment
    report.update(pickle.loads(metadata_bytes))
    if pending is not None:
        report["failed_or_unpublished_readout"] = pending
    # Restore the accepted private prefix if an opaque port corrupted its public copy.
    report["readouts"] = [pickle.loads(payload) for _, payload in protected]
    reports = report["readouts"] + ([pending] if pending is not None else [])
    names = (
        "raw_score_forwards_attempted",
        "raw_score_forwards_completed",
        "raw_score_rows_attempted",
        "raw_score_rows_completed",
        "live_prediction_calls_attempted",
        "live_prediction_calls_returned",
        "environment_resets_attempted",
        "environment_resets_completed",
        "environment_steps_attempted",
        "environment_steps_returned",
    )
    report["work"] = {
        name: sum(item.get("work", {}).get(name, 0) for item in reports) for name in names
    }
    report["work"]["failure_work_may_be_unobserved"] = opaque_active or any(
        item.get("work", {}).get("failure_work_may_be_unobserved", False) for item in reports
    )
    report["new_training_updates"] = budget.updates_completed
    report["replay_examples"] = budget.replay_examples_completed
    report["total_seconds"] = elapsed
    report["observed_process_rss"] = asdict(observed_rss) if observed_rss is not None else None
    return report
