"""Bind one fixed v14 study to the first versioned run manifest.

Inputs are the already trained, globally scored v14 records, their exact
JSON bytes, and execution facts captured before training. Output is a
validated P5.1 manifest. This use case does not write files, choose an
arm, change v14 training, or supply missing provenance by inference.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

from src.app.comparison_scope import (
    NUMPY_BACKPROP_ALGORITHM_ID,
    NUMPY_CIRCADIAN_ALGORITHM_ID,
    NUMPY_PC_ALGORITHM_ID,
)
from src.app.continual_arrived_benchmark import (
    _PHASE_A_SPLIT_SEED_OFFSET,
    _PHASE_B_EXPOSURE_SEED_OFFSET,
    _PHASE_B_SPLIT_SEED_OFFSET,
)
from src.app.continual_trigger_replay_outcomes import (
    TRIGGER_REPLAY_OUTCOMES_PROTOCOL,
    TriggerReplayComparison,
)
from src.app.continual_trigger_replay_schedule import (
    TRIGGER_REPLAY_OPPORTUNITIES_PROTOCOL,
    fixed_trigger_replay_manifest,
)
from src.app.continual_trigger_replay_training_study import (
    TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL,
    TriggerReplayTrainingStudy,
)
from src.core.run_manifest import RUN_MANIFEST_SCHEMA_ID, RunEnvironment, validate_run_manifest


_ROLES = ("train", "inner_guard", "outer_selection", "final_test")


def _read_fixed_payloads(
    study: TriggerReplayTrainingStudy,
    comparison: TriggerReplayComparison,
    training_bytes: bytes,
    outcomes_bytes: bytes,
    config: dict[str, Any],
) -> None:
    try:
        training = json.loads(training_bytes)
        outcomes = json.loads(outcomes_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v14 training or outcome payload is invalid JSON") from exc
    if (
        not isinstance(training, dict)
        or training.get("protocol_id") != TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL
        or training.get("manifest_digest") != study.manifest_digest
        or training.get("resolved_manifest") != config
        or not isinstance(training.get("rows"), list)
        or [(row.get("seed"), row.get("arm")) for row in training["rows"]]
        != [(trial.seed, trial.arm) for trial in study.trials]
    ):
        raise ValueError("v14 training payload differs from the fixed study")
    if (
        not isinstance(outcomes, dict)
        or outcomes.get("protocol_id") != TRIGGER_REPLAY_OUTCOMES_PROTOCOL
        or outcomes.get("manifest_digest") != comparison.manifest_digest
        or outcomes.get("manifest") != config
        or outcomes != json.loads(json.dumps(asdict(comparison), sort_keys=True, allow_nan=False))
    ):
        raise ValueError("v14 outcome payload differs from the scored comparison")


def _collect_role_hashes(
    study: TriggerReplayTrainingStudy, comparison: TriggerReplayComparison
) -> dict[str, dict[str, dict[str, str]]]:
    records: dict[str, dict[str, dict[str, str]]] = {}
    for seed in study.manifest.seeds:
        matched = [
            (trial, outcome)
            for trial, outcome in zip(study.trials, comparison.outcomes, strict=True)
            if trial.seed == seed
        ]
        if len(matched) != len(study.manifest.arms):
            raise ValueError("v14 run manifest lacks a seed/arm cell")
        phases: dict[str, dict[str, str]] = {}
        for phase in ("a", "b"):
            hashes = []
            for trial, outcome in matched:
                pending = trial.pending.phase_a if phase == "a" else trial.pending.phase_b
                final = dict(outcome.final_role_hashes)
                hashes.append(
                    {
                        **{role: pending.split_hashes[role] for role in _ROLES[:-1]},
                        "final_test": final[phase],
                    }
                )
            if any(candidate != hashes[0] for candidate in hashes[1:]):
                raise ValueError("v14 source role hashes differ across trigger arms")
            phases[phase] = hashes[0]
        records[str(seed)] = phases
    return records


def _seed_map(seeds: tuple[int, ...]) -> dict[str, dict[str, int]]:
    # Why this: these are the source/role and model constructor streams
    # actually used by v14; storing only the two base seeds hides offsets.
    return {
        str(seed): {
            "phase_a_source": seed,
            "phase_b_source": seed + 101,
            "phase_a_roles": seed + _PHASE_A_SPLIT_SEED_OFFSET,
            "phase_b_exposure": seed + _PHASE_B_EXPOSURE_SEED_OFFSET,
            "phase_b_roles": seed + _PHASE_B_SPLIT_SEED_OFFSET,
            "backprop": seed,
            "predictive_coding": seed + 1,
            "circadian": seed + 2,
            "circadian_local_rng": seed + 10_003,
        }
        for seed in seeds
    }


def build_v14_run_manifest(
    run_id: str,
    environment: RunEnvironment,
    study: TriggerReplayTrainingStudy,
    comparison: TriggerReplayComparison,
    training_bytes: bytes,
    outcomes_bytes: bytes,
) -> dict[str, Any]:
    """Validate complete v14 identities and assemble explicit run facts."""
    if (
        study.manifest != fixed_trigger_replay_manifest()
        or comparison.manifest != study.manifest
        or study.protocol_id != TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL
        or comparison.protocol_id != TRIGGER_REPLAY_OUTCOMES_PROTOCOL
        or comparison.manifest_digest != study.manifest_digest
        or [(item.seed, item.arm) for item in comparison.outcomes]
        != [(item.seed, item.arm) for item in study.trials]
    ):
        raise ValueError("v14 run manifest study and outcome identities differ")
    config = json.loads(json.dumps(asdict(study.manifest), sort_keys=True, allow_nan=False))
    _read_fixed_payloads(study, comparison, training_bytes, outcomes_bytes, config)
    manifest: dict[str, Any] = {
        "schema_id": RUN_MANIFEST_SCHEMA_ID,
        "run_id": run_id,
        "status": "completed",
        "protocol_versions": {
            "source": TRIGGER_REPLAY_OPPORTUNITIES_PROTOCOL,
            "training": TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL,
            "outcomes": TRIGGER_REPLAY_OUTCOMES_PROTOCOL,
        },
        "algorithm_versions": {
            "backprop": NUMPY_BACKPROP_ALGORITHM_ID,
            "predictive_coding": NUMPY_PC_ALGORITHM_ID,
            "circadian_predictive_coding": NUMPY_CIRCADIAN_ALGORITHM_ID,
        },
        "source": dict(environment.source),
        "resolved_config": config,
        "config_sha256": study.manifest_digest,
        "seed_map": _seed_map(study.manifest.seeds),
        "dataset_role_names": list(_ROLES),
        "dataset_split_hashes": _collect_role_hashes(study, comparison),
        "pretrained_weights": {
            "state": "not_applicable",
            "sha256": None,
            "reason": "The fixed synthetic v14 NumPy models use seeded initialization.",
        },
        "dependency_versions": dict(environment.dependency_versions),
        "hardware": dict(environment.hardware),
        "precision": {"inputs": "float64", "parameters": "float64", "metrics": "float64"},
        "determinism": {
            "numpy_rng": "local numpy.default_rng streams derived in seed_map",
            "python_hash_seed": environment.python_hash_seed,
            "scope": "same environment; cross-platform bitwise identity is not claimed",
        },
        "timing_scope": {
            "mode": "not_measured",
            "includes": [],
            "excludes": ["source construction", "training", "final scoring", "serialization"],
        },
        "files": {
            "training": {
                "path": "training.json",
                "sha256": sha256(training_bytes).hexdigest(),
                "protocol_id": TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL,
            },
            "outcomes": {
                "path": "outcomes.json",
                "sha256": sha256(outcomes_bytes).hexdigest(),
                "protocol_id": TRIGGER_REPLAY_OUTCOMES_PROTOCOL,
            },
        },
    }
    validate_run_manifest(manifest)
    return manifest
