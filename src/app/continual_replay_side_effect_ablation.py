"""Compare two replay side-effect contracts on one fixed matched schedule.

Inputs are a bounded v9 matched-control manifest and two versioned side-effect
choices. All eight trials remain unscored until role, work, and baseline parity
checks pass. This module does not tune settings, persist files, or select a winner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.app import continual_matched_replay_outcomes as outcomes
from src.app import continual_matched_replay_schedule as schedule
from src.app.continual_checkpoint import continual_config_digest
from src.app.continual_matched_replay_runner import (
    MatchedReplayTrainingResult,
    replay_side_effect_training_identity,
    run_matched_replay_training,
)
from src.core.circadian_predictive_coding import WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY
from src.infra.continual_roles import release_final_test


SIDE_EFFECT_ABLATION_PROTOCOL = "continual_replay_side_effect_ablation_v10"


@dataclass(frozen=True)
class ReplaySideEffectAblationManifest:
    """Bind retention policies, seeds, budgets, and side effects before training."""

    matched: outcomes.MatchedReplayOutcomeManifest
    side_effect_policies: tuple[str, str] = ("historical", WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY)
    protocol_id: str = SIDE_EFFECT_ABLATION_PROTOCOL


@dataclass(frozen=True)
class ReplaySideEffectOutcome:
    side_effect_policy: str
    retention: tuple[outcomes.MatchedReplayPolicyOutcome, ...]


@dataclass(frozen=True)
class ReplaySideEffectAblationResult:
    protocol_id: str
    manifest: ReplaySideEffectAblationManifest
    manifest_digest: str
    outcomes: tuple[ReplaySideEffectOutcome, ...]


def run_replay_side_effect_ablation(
    manifest: ReplaySideEffectAblationManifest,
) -> ReplaySideEffectAblationResult:
    """Score all matched controls only after a common global final seal."""
    _validate_manifest(manifest)
    matched = manifest.matched
    schedules = tuple(outcomes._schedule_manifest(matched, policy) for policy in matched.policies)
    trained = tuple(
        tuple(
            tuple(
                run_matched_replay_training(bound, seed=seed, side_effect_policy=side_policy)
                for seed in matched.seeds
            )
            for bound in schedules
        )
        for side_policy in manifest.side_effect_policies
    )
    for side_policy, groups in zip(manifest.side_effect_policies, trained, strict=True):
        for bound, trials in zip(schedules, groups, strict=True):
            for seed, trial in zip(matched.seeds, trials, strict=True):
                schedule_digest = schedule.MatchedReplayScheduleSession(
                    bound, seed=seed
                ).manifest_digest
                protocol_id, digest = replay_side_effect_training_identity(
                    schedule_digest, side_policy
                )
                if (
                    trial.pending.state.circadian_model.get_replay_side_effect_policy()
                    != side_policy
                ):
                    raise ValueError("replay side-effect model policy differs")
                outcomes._validate_unscored_trial(
                    bound,
                    seed,
                    trial,
                    training_protocol=protocol_id,
                    training_digest=digest,
                )
    _validate_matched_work(trained)

    # Why this: no final label becomes available while another policy/seed
    # still trains, and every final role is compared before scoring starts.
    released = tuple(
        tuple(
            tuple(
                (
                    release_final_test(trial.pending.phase_a),
                    release_final_test(trial.pending.phase_b),
                )
                for trial in trials
            )
            for trials in groups
        )
        for groups in trained
    )
    first = released[0]
    for other in released[1:]:
        for left, right in zip(first, other, strict=True):
            outcomes._validate_matched_final_roles(matched.seeds, (left, right))
    for group in released:
        outcomes._validate_matched_final_roles(matched.seeds, group)

    scored = tuple(
        ReplaySideEffectOutcome(
            side_policy,
            tuple(
                outcomes._score_policy(matched, policy, trials, roles)
                for policy, trials, roles in zip(matched.policies, groups, role_groups, strict=True)
            ),
        )
        for side_policy, groups, role_groups in zip(
            manifest.side_effect_policies, trained, released, strict=True
        )
    )
    return ReplaySideEffectAblationResult(
        SIDE_EFFECT_ABLATION_PROTOCOL,
        manifest,
        continual_config_digest(manifest, matched.seeds),
        scored,
    )


def _validate_manifest(manifest: ReplaySideEffectAblationManifest) -> None:
    if (
        type(manifest) is not ReplaySideEffectAblationManifest
        or manifest.protocol_id != SIDE_EFFECT_ABLATION_PROTOCOL
        or manifest.side_effect_policies != ("historical", WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY)
    ):
        raise ValueError("side-effect ablation requires historical and wake-only v1 policies")
    outcomes._validate_manifest(manifest.matched)
    training = manifest.matched.arrived.training
    if len(manifest.matched.seeds) != 2 or training.phase_a_epochs + training.phase_b_epochs > 4:
        raise ValueError("side-effect ablation exceeds the declared two-seed/four-epoch budget")


def _validate_matched_work(
    trained: tuple[tuple[tuple[MatchedReplayTrainingResult, ...], ...], ...],
) -> None:
    historical, wake_only = trained
    for old_group, new_group in zip(historical, wake_only, strict=True):
        for old, new in zip(old_group, new_group, strict=True):
            if old.boundaries != new.boundaries:
                raise ValueError("side-effect ablation applied replay work differs")
            for name in (
                "backprop_model",
                "predictive_model",
                "backprop_after_a",
                "predictive_after_a",
            ):
                old_model = getattr(old.pending.state, name)
                new_model = getattr(new.pending.state, name)
                old_arrays = (
                    *old_model._hidden_weights,
                    *old_model._hidden_biases,
                    old_model.weight_hidden_output,
                    old_model.bias_output,
                    *old_model._traffic_sums,
                )
                new_arrays = (
                    *new_model._hidden_weights,
                    *new_model._hidden_biases,
                    new_model.weight_hidden_output,
                    new_model.bias_output,
                    *new_model._traffic_sums,
                )
                if (
                    len(old_arrays) != len(new_arrays)
                    or any(
                        not np.array_equal(left, right)
                        for left, right in zip(old_arrays, new_arrays, strict=True)
                    )
                    or old_model._traffic_steps != new_model._traffic_steps
                ):
                    raise ValueError("side-effect ablation baseline state differs")
