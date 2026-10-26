"""Replay two released TRAIN episodes through the existing delayed-label runtime.

Inputs are complete continuous episodes; output is a bounded JSON-compatible
integration report. Warm state permits serving before new labels arrive. No IO,
simulator, promotion, final/probe evaluation or durable checkpoint lives here.
"""

from dataclasses import asdict
from time import monotonic
from typing import Any
from collections.abc import Callable

import numpy as np

from src.adapters.numpy_continuous_bc import fit_training_standardization
from src.adapters.numpy_nearest_continuous_bc import (
    OBJECTIVE,
    NearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_demonstration import (
    ContinuousEpisode,
    continuous_training_pairs,
    copy_continuous_episode,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.nearest_continuous_bc import NearestPrediction


def _same_state(a: NearestContinuousBCSnapshot, b: NearestContinuousBCSnapshot) -> bool:
    return (
        a.policy == b.policy
        and a.completed_updates == b.completed_updates
        and a.training_examples == b.training_examples
        and np.array_equal(a.prototype_features, b.prototype_features)
        and np.array_equal(a.prototype_actions, b.prototype_actions)
    )


def _training_episodes(
    warm: ContinuousEpisode, new: ContinuousEpisode
) -> tuple[ContinuousEpisode, ContinuousEpisode]:
    # Why: reject held-out roles/configuration before fitting or creating learners.
    if any(
        type(episode) is not ContinuousEpisode or episode.role != "train" for episode in (warm, new)
    ):
        raise ValueError("continuous stream requires two exact released TRAIN episodes")
    warm, new = copy_continuous_episode(warm), copy_continuous_episode(new)
    if warm.spec != new.spec:
        raise ValueError("continuous stream episodes must have the same spec")
    if warm.episode_id == new.episode_id:
        raise ValueError("continuous stream episodes require distinct identities")
    if len(warm.transitions) + len(new.transitions) > 256:
        raise ValueError("continuous stream exceeds256 prototype rows")
    return warm, new


def _work_report(
    initial: NearestContinuousBCSnapshot,
    new_rows: int,
    warm_diagnostic: TrainingDiagnostic,
    runtime_updates: int,
    warm_prediction: NearestPrediction,
    new_prediction: NearestPrediction,
) -> dict[str, Any]:
    """Separate returned candidate work from derived actor counts; no native calls."""
    warm_rows = initial.training_examples
    actor_rows, candidate_rows = 4 * new_rows, warm_rows + new_rows
    feature_dim, action_dim = (
        initial.policy.spec.observation_dim,
        len(initial.policy.spec.action_units),
    )
    return dict(
        fit_attempts=1,
        fit_returns=1,
        fit_rows=warm_rows,
        fit_feature_values=warm_rows * feature_dim,
        fit_numeric_bytes=feature_dim * 16,
        warm_append_attempts=1,
        warm_append_returns=1,
        warm_append_rows=warm_rows,
        warm_diagnostic=asdict(warm_diagnostic),
        runtime_append_returns=runtime_updates,
        runtime_append_rows=new_rows,
        source_rows_ingested=warm_rows + new_rows,
        replayed_training_rows=0,
        optimizer_updates=0,
        actor_prediction_calls=4,
        actor_prediction_rows=actor_rows,
        actor_prediction_bytes=actor_rows * action_dim * 8,
        actor_distance_count_source="derived_from_exact_unchanged_typed_actor_and_query_shapes",
        actor_distance_candidates=actor_rows * warm_rows,
        actor_distance_feature_values=actor_rows * warm_rows * feature_dim,
        candidate_prediction_calls=2,
        candidate_prediction_rows=candidate_rows,
        candidate_prediction_bytes=candidate_rows * action_dim * 8,
        candidate_distance_candidates=(
            warm_prediction.distance_candidates + new_prediction.distance_candidates
        ),
        candidate_distance_feature_values=(
            warm_prediction.distance_feature_values + new_prediction.distance_feature_values
        ),
        normalization_calls=8,
        normalization_rows=warm_rows + new_rows + actor_rows + candidate_rows,
    )


def _memory_report(initial: NearestContinuousBCSnapshot, new_rows: int) -> dict[str, Any]:
    """Per-owner payload arithmetic; never claim a whole-process memory total."""
    warm_rows = initial.training_examples
    candidate_rows = warm_rows + new_rows
    feature_dim = initial.policy.spec.observation_dim
    row_bytes = (feature_dim + len(initial.policy.spec.action_units)) * 8
    return dict(
        source_prototype_bytes=warm_rows * row_bytes,
        actor_prototype_bytes=warm_rows * row_bytes,
        candidate_prototype_bytes=candidate_rows * row_bytes,
        readout_prototype_bytes=candidate_rows * row_bytes,
        prototype_growth_bytes=new_rows * row_bytes,
        statistics_bytes_per_owner=feature_dim * 16,
        inbox_feature_target_bytes=new_rows * row_bytes,
        scope="numeric payloads per named owner; excludes snapshots, Python and temporaries",
    )


def run_continuous_demo(
    warm_episode: ContinuousEpisode,
    new_episode: ContinuousEpisode,
    *,
    additional_episodes: tuple[ContinuousEpisode, ...] = (),
    checkpoint_observer: Callable | None = None,
    fixed_input: bool = False,
) -> dict[str, Any]:
    """Hold the new episode's labels until tick2; append once to a private candidate.

    A packet is one arrived array-pair batch, not one training call per action.
    Logical arrival is simulated from already saved TRAIN labels. The source,
    stable actor and candidate own separate native arrays and frozen statistics.
    Explicit fixed inputs use the existing sequence route without fitting.
    """
    if type(fixed_input) is not bool:
        raise ValueError("fixed_input must be an exact bool")
    if type(additional_episodes) is not tuple or len(additional_episodes) > 2:
        raise ValueError("additional_episodes must be an exact tuple with0..2 episodes")
    if checkpoint_observer is not None and not callable(checkpoint_observer):
        raise ValueError("checkpoint_observer must be callable or None")
    if fixed_input or additional_episodes or checkpoint_observer is not None:
        from src.adapters.continuous_stream_sequence import run_sequence

        if fixed_input:
            return run_sequence(
                warm_episode,
                (new_episode, *additional_episodes),
                checkpoint_observer,
                fixed_input=True,
            )
        return run_sequence(warm_episode, (new_episode, *additional_episodes), checkpoint_observer)
    started = monotonic()
    warm, new = _training_episodes(warm_episode, new_episode)
    fit_started = monotonic()
    statistics = fit_training_standardization(warm)
    fit_elapsed = monotonic() - fit_started
    source = NumpyNearestContinuousBCLearner(NearestContinuousBCPolicy(warm.spec, statistics))
    append_started = monotonic()
    warm_diagnostic = source.train_episode(warm)
    append_elapsed = monotonic() - append_started
    initial = source.snapshot_state()
    warm_rows, new_rows = len(warm.transitions), len(new.transitions)
    warm_features, _ = continuous_training_pairs(warm, role="train")
    new_features, new_actions = continuous_training_pairs(new, role="train")
    x_warm = np.asarray(warm_features, dtype=np.float64)
    x_new = np.asarray(new_features, dtype=np.float64)
    y_new = np.asarray(new_actions, dtype=np.float64)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), monotonic)
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=1,
        max_consolidations=0,
    )
    events: list[dict[str, Any]] = []
    update_elapsed = 0.0
    for tick in range(3):
        clock.advance_to(tick)
        # Serve before this tick's update. The saved label is not in the inbox
        # before its logical release; possessing it offline grants no early update.
        served = runtime.actor.predict(x_new)
        if tick == 0:
            runtime.record_experience(
                Experience(
                    "new-episode-pairs",
                    new.episode_id,
                    0,
                    "actor-0",
                    x_new,
                    "train",
                    ExperiencePermissions(training=True),
                )
            )
        if tick == 2:
            runtime.record_label(
                LabelArrival(
                    "new-episode-label",
                    "new-episode-pairs",
                    new.episode_id,
                    2,
                    "actor-0",
                    y_new,
                )
            )
        update_started = monotonic()
        applied = runtime.train_ready(max_updates=1)
        update_elapsed += monotonic() - update_started
        candidate = runtime.candidate_snapshot()
        expected_rows = warm_rows if tick < 2 else warm_rows + new_rows
        if candidate.state.training_examples != expected_rows or (
            tick < 2 and not _same_state(initial, candidate.state)
        ):
            raise RuntimeError("candidate changed before declared label arrival")
        events.append(
            dict(
                tick=tick,
                label_available=tick == 2,
                actor_version=served.actor_version,
                actor_predictions=served.prediction.tolist(),
                applied=[asdict(item) for item in applied],
                candidate_prototype_rows=candidate.state.training_examples,
                candidate_publications=candidate.state.completed_updates,
                budget_updates=budget.updates_completed,
            )
        )

    final = runtime.candidate_snapshot()
    actor_after = runtime.actor.predict(x_new)
    no_more_updates = runtime.train_ready(max_updates=1) == ()
    after_poll = runtime.candidate_snapshot()
    checks = dict(
        source_full_state_exact=_same_state(initial, source.snapshot_state()),
        actor_full_state_exact=_same_state(initial, runtime.actor.snapshot_state().state),
        frozen_statistics_exact=final.state.policy == initial.policy,
        a_prefix_exact=(
            np.array_equal(final.state.prototype_features[:warm_rows], initial.prototype_features)
            and np.array_equal(final.state.prototype_actions[:warm_rows], initial.prototype_actions)
        ),
        post_poll_no_update=no_more_updates,
        post_poll_candidate_exact=_same_state(final.state, after_poll.state),
        actor_predictions_exact=all(
            event["actor_predictions"] == actor_after.prediction.tolist() for event in events
        ),
    )
    if not all(checks.values()) or budget.updates_completed != 1:
        raise RuntimeError("continuous stream ownership/arrival/budget invariant failed")
    applied_updates = runtime.applied_updates
    if (
        len(applied_updates) != 1
        or applied_updates[0].diagnostic.definition != OBJECTIVE
        or applied_updates[0].diagnostic.value != new_rows
        or final.state.completed_updates != 2
    ):
        raise RuntimeError("continuous stream must publish exactly one new pair append")
    # Read the candidate through an owned complete native state, without exposing
    # the runtime's mutable learner or pretending the JSON report is a checkpoint.
    readout = NumpyNearestContinuousBCLearner(initial.policy)
    readout.restore_state(final.state)
    warm_prediction = readout.predict_with_work(x_warm)
    new_prediction = readout.predict_with_work(x_new)
    checks["readout_full_state_exact"] = _same_state(final.state, readout.snapshot_state())
    if not checks["readout_full_state_exact"]:
        raise RuntimeError("continuous readout changed full native state")
    return dict(
        schema_version=1,
        protocol_id="continuous_nearest_delayed_stream_v1",
        scope="saved_train_integration",
        complete=True,
        settings=dict(
            label_delay_ticks=2,
            warm_rows=warm_rows,
            new_rows=new_rows,
            max_training_updates=1,
            max_experiences=1,
            max_prototypes=256,
            promotion_calls=0,
            simulator_calls=0,
            optimizer_updates=0,
        ),
        experiment_state=dict(
            format_version=1, completed_updates=budget.updates_completed, next_tick=3
        ),
        model_state=dict(
            snapshot_type="NearestContinuousBCSnapshot",
            durable_checkpoint=False,
            actor_version=final.actor_version,
            candidate_version=final.learner_version,
            source_publications=initial.completed_updates,
            candidate_publications=final.state.completed_updates,
        ),
        source_episodes=dict(
            warm=warm.episode_id,
            new=new.episode_id,
            role="train",
            warm_context=warm.context,
            new_context=new.context,
        ),
        policy=asdict(initial.policy),
        checks=checks,
        events=events,
        applied=[asdict(item) for item in runtime.applied_updates],
        readouts=dict(
            actor_after_new=actor_after.prediction.tolist(),
            candidate_warm=warm_prediction.actions.tolist(),
            candidate_new=new_prediction.actions.tolist(),
            candidate_warm_indices=list(warm_prediction.selected_indices),
            candidate_new_indices=list(new_prediction.selected_indices),
        ),
        work=_work_report(
            initial,
            new_rows,
            warm_diagnostic,
            budget.updates_completed,
            warm_prediction,
            new_prediction,
        ),
        memory=_memory_report(initial, new_rows),
        timings=dict(
            fit_seconds=fit_elapsed,
            warm_append_seconds=append_elapsed,
            runtime_drain_seconds=update_elapsed,
            total_seconds=monotonic() - started,
        ),
    )
