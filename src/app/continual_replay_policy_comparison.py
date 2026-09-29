"""Matched ordinary A→B comparison of bounded replay retention policies.

Inputs are one predeclared v6 arrived-role setting, seed tuple, and two
policies. Outputs retain every policy/seed result and measured replay IDs.
This module does not select a winner or read final data during training.
Checkpoint persistence and preflight live in separate app modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import pickle

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_checkpoint import continual_config_digest
from src.app.continual_replay_policy_checkpoint import ReplayPolicyCheckpointStore
from src.app.continual_shift_benchmark import ContinualShiftAggregate
from src.core.replay_retention import ReplayExposureSnapshot, ReplayRetentionPolicy


REPLAY_POLICY_COMPARISON_PROTOCOL = "continual_replay_policy_comparison_v8"


@dataclass(frozen=True)
class ReplayPolicyComparisonManifest:
    """Freeze one role setting and ordered policies/seeds before source access."""

    arrived: arrived.ContinualArrivedRolesConfig
    seeds: tuple[int, ...]
    policies: tuple[ReplayRetentionPolicy, ReplayRetentionPolicy]
    protocol_id: str = REPLAY_POLICY_COMPARISON_PROTOCOL


@dataclass(frozen=True)
class ReplayPolicySeedExposure:
    """Cumulative opt-in audit at A and B; B includes any A replay exposure."""

    phase_a: ReplayExposureSnapshot
    after_b: ReplayExposureSnapshot
    observed_ids: tuple[str, ...]
    observed_duplicate_ids: tuple[str, ...]
    observed_duplicate_occurrences: int


@dataclass(frozen=True)
class ReplayPolicySeedResult:
    arrived: arrived.ContinualArrivedSeedResult
    exposure: ReplayPolicySeedExposure
    baseline_state_digests: dict[str, str]


@dataclass(frozen=True)
class ReplayPolicyResult:
    policy: ReplayRetentionPolicy
    seeds: tuple[ReplayPolicySeedResult, ...]
    aggregate: ContinualShiftAggregate


@dataclass(frozen=True)
class ReplayPolicyComparisonResult:
    protocol_id: str
    manifest: ReplayPolicyComparisonManifest
    manifest_digest: str
    policies: tuple[ReplayPolicyResult, ...]


def _validate_manifest(manifest: ReplayPolicyComparisonManifest) -> None:
    if (
        type(manifest) is not ReplayPolicyComparisonManifest
        or manifest.protocol_id != REPLAY_POLICY_COMPARISON_PROTOCOL
        or type(manifest.seeds) is not tuple
        or not manifest.seeds
        or any(type(seed) is not int for seed in manifest.seeds)
        or len(set(manifest.seeds)) != len(manifest.seeds)
    ):
        raise ValueError("v8 manifest requires nonempty unique integer seeds and its protocol")
    if (
        type(manifest.policies) is not tuple
        or len(manifest.policies) != 2
        or type(manifest.policies[0]) is not ReplayRetentionPolicy
        or type(manifest.policies[1]) is not ReplayRetentionPolicy
        or tuple(policy.name for policy in manifest.policies) != ("recent_fifo", "seeded_reservoir")
    ):
        raise ValueError("v8 manifest requires ordered FIFO and seeded reservoir policies")
    arrived._validate_arrived_config(manifest.arrived, list(manifest.seeds))


def _baseline_state_digests(pending: arrived._PendingSeed) -> dict[str, str]:
    state = pending.state
    return {
        name: sha256(pickle.dumps(model, protocol=5)).hexdigest()
        for name, model in (
            ("backprop_after_a", state.backprop_after_a),
            ("predictive_after_a", state.predictive_after_a),
            ("backprop_after_b", state.backprop_model),
            ("predictive_after_b", state.predictive_model),
        )
    }


def _exposure(pending: arrived._PendingSeed) -> ReplayPolicySeedExposure:
    phase_a = pending.state.circadian_after_a.get_replay_exposure()
    after_b = pending.state.circadian_model.get_replay_exposure()
    if (
        not set(phase_a.observed_ids).issubset(after_b.observed_ids)
        or not set(phase_a.duplicate_ids).issubset(after_b.duplicate_ids)
        or not set(phase_a.exposed_ids).issubset(after_b.exposed_ids)
        or phase_a.duplicate_occurrences > after_b.duplicate_occurrences
        or phase_a.replay_updates > after_b.replay_updates
    ):
        raise ValueError("v8 replay exposure regressed between arrived phases")
    return ReplayPolicySeedExposure(
        phase_a=phase_a,
        after_b=after_b,
        observed_ids=after_b.observed_ids,
        observed_duplicate_ids=after_b.duplicate_ids,
        observed_duplicate_occurrences=after_b.duplicate_occurrences,
    )


def run_replay_policy_comparison(
    manifest: ReplayPolicyComparisonManifest,
    *,
    checkpoint_store: ReplayPolicyCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
) -> ReplayPolicyComparisonResult:
    """Train all policy/seed runs before releasing any final-test role."""
    _validate_manifest(manifest)
    if resume_from_checkpoint and checkpoint_store is None:
        raise ValueError("v8 policy resume requires a checkpoint store")
    # Why this: a single frozen input binds both policies and every seed;
    # all training ends before either policy can receive a final-test role.
    digest = continual_config_digest(manifest, manifest.seeds)
    if checkpoint_store is None:
        pending_by_policy = tuple(
            tuple(
                arrived._train_arrived_seed(manifest.arrived, seed, retention_policy=policy)
                for seed in manifest.seeds
            )
            for policy in manifest.policies
        )
    else:
        from src.app.continual_replay_policy_resume import run_completed_policy_checkpoints

        pending_by_policy = run_completed_policy_checkpoints(
            manifest, digest, checkpoint_store, resume_from_checkpoint
        )
    measured = tuple(
        tuple((_exposure(item), _baseline_state_digests(item)) for item in pending)
        for pending in pending_by_policy
    )
    policy_results = []
    for policy, pending, observations in zip(
        manifest.policies, pending_by_policy, measured, strict=True
    ):
        scored = arrived._build_arrived_result(
            manifest.arrived, list(manifest.seeds), list(pending)
        )
        policy_results.append(
            ReplayPolicyResult(
                policy=policy,
                seeds=tuple(
                    ReplayPolicySeedResult(item, exposure, baseline_digests)
                    for item, (exposure, baseline_digests) in zip(
                        scored.seed_results, observations, strict=True
                    )
                ),
                aggregate=scored.aggregate,
            )
        )
    return ReplayPolicyComparisonResult(
        protocol_id=REPLAY_POLICY_COMPARISON_PROTOCOL,
        manifest=manifest,
        manifest_digest=digest,
        policies=tuple(policy_results),
    )
