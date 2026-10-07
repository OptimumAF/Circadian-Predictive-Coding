"""P5.1 manifest records explicit provenance before any run is publishable."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from src.core.run_manifest import serialize_run_manifest, validate_run_manifest


_DIGEST = "a" * 64


def _valid_manifest() -> dict[str, object]:
    return {
        "schema_id": "circadian_run_manifest_v1",
        "run_id": "p51-v14-fixture",
        "status": "completed",
        "protocol_versions": {
            "source": "continual_trigger_opportunities_v14",
            "training": "continual_trigger_replay_train_only_v14",
            "outcomes": "continual_trigger_replay_outcomes_v14",
        },
        "algorithm_versions": {"backprop": "numpy_backprop_tanh_mlp_v1"},
        "source": {
            "commit_sha": "b" * 40,
            "dirty": True,
            "status_sha256": _DIGEST,
            "tracked_diff_sha256": _DIGEST,
            "workspace_sha256": _DIGEST,
            "untracked_file_count": 3,
            "unavailable_reason": None,
        },
        "resolved_config": {"protocol_id": "continual_trigger_opportunities_v14", "seeds": [47]},
        "config_sha256": _DIGEST,
        "seed_map": {"47": {"phase_a_source": 47, "backprop": 47}},
        "dataset_role_names": ["train", "inner_guard", "outer_selection", "final_test"],
        "dataset_split_hashes": {
            "47": {
                phase: {
                    role: _DIGEST
                    for role in ("train", "inner_guard", "outer_selection", "final_test")
                }
                for phase in ("a", "b")
            }
        },
        "pretrained_weights": {
            "state": "not_applicable",
            "sha256": None,
            "reason": "Synthetic NumPy model initialized from a seed.",
        },
        "dependency_versions": {"python": "3.11.9", "numpy": "2.4.0"},
        "hardware": {
            "system": "Windows",
            "release": "11",
            "machine": "AMD64",
            "processor": "Fixture CPU",
            "processor_unavailable_reason": None,
            "logical_cpu_count": 8,
            "compute_device": "cpu",
        },
        "precision": {"inputs": "float64", "parameters": "float64", "metrics": "float64"},
        "determinism": {
            "numpy_rng": "local default_rng seeds",
            "python_hash_seed": None,
            "scope": "same environment; cross-platform bitwise equality is not claimed",
        },
        "timing_scope": {"mode": "not_measured", "includes": [], "excludes": ["training"]},
        "files": {
            "training": {
                "path": "training.json",
                "sha256": _DIGEST,
                "protocol_id": "continual_trigger_replay_train_only_v14",
            },
            "outcomes": {
                "path": "outcomes.json",
                "sha256": _DIGEST,
                "protocol_id": "continual_trigger_replay_outcomes_v14",
            },
        },
    }


def test_should_accept_complete_dirty_clean_and_explicit_missing_git_records() -> None:
    dirty = _valid_manifest()
    validate_run_manifest(dirty)

    clean = deepcopy(dirty)
    clean["source"]["dirty"] = False  # type: ignore[index]
    clean["source"]["status_sha256"] = sha256(b"").hexdigest()  # type: ignore[index]
    clean["source"]["tracked_diff_sha256"] = sha256(b"").hexdigest()  # type: ignore[index]
    clean["source"]["untracked_file_count"] = 0  # type: ignore[index]
    validate_run_manifest(clean)

    unavailable = deepcopy(dirty)
    unavailable["source"] = {
        "commit_sha": None,
        "dirty": None,
        "status_sha256": None,
        "tracked_diff_sha256": None,
        "workspace_sha256": None,
        "untracked_file_count": None,
        "unavailable_reason": "Git metadata is unavailable in this source snapshot.",
    }
    validate_run_manifest(unavailable)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("run_id", "../escape"),
        ("status", "succeeded-ish"),
        ("config_sha256", "short"),
        ("dependency_versions", {}),
    ],
)
def test_should_reject_invalid_required_identity(field: str, bad_value: object) -> None:
    manifest = _valid_manifest()
    manifest[field] = bad_value
    with pytest.raises(ValueError):
        validate_run_manifest(manifest)


def test_should_reject_missing_field_nonfinite_config_or_false_completion() -> None:
    missing = _valid_manifest()
    del missing["pretrained_weights"]
    with pytest.raises(ValueError, match="fields"):
        validate_run_manifest(missing)

    nonfinite = _valid_manifest()
    nonfinite["resolved_config"]["learning_rate"] = float("nan")  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        validate_run_manifest(nonfinite)

    false_complete = _valid_manifest()
    false_complete["files"] = {}
    with pytest.raises(ValueError, match="completed"):
        validate_run_manifest(false_complete)


def test_should_reject_missing_role_hash_unsafe_output_and_unexplained_absence() -> None:
    missing_role = _valid_manifest()
    del missing_role["dataset_split_hashes"]["47"]["a"]["final_test"]  # type: ignore[index]
    with pytest.raises(ValueError, match="role"):
        validate_run_manifest(missing_role)

    unsafe = _valid_manifest()
    unsafe["files"]["outcomes"]["path"] = "../outcomes.json"  # type: ignore[index]
    with pytest.raises(ValueError, match="path"):
        validate_run_manifest(unsafe)

    absent = _valid_manifest()
    absent["source"]["commit_sha"] = None  # type: ignore[index]
    with pytest.raises(ValueError, match="Git"):
        validate_run_manifest(absent)

    false_clean = _valid_manifest()
    false_clean["source"]["dirty"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="clean Git"):
        validate_run_manifest(false_clean)

    unknown_processor = _valid_manifest()
    unknown_processor["hardware"]["processor"] = None  # type: ignore[index]
    with pytest.raises(ValueError, match="processor unavailable"):
        validate_run_manifest(unknown_processor)


def test_should_serialize_stable_finite_lf_json() -> None:
    payload = serialize_run_manifest(_valid_manifest())
    assert payload.endswith(b"\n") and b"\r\n" not in payload
    assert json.loads(payload)["run_id"] == "p51-v14-fixture"
    assert serialize_run_manifest(_valid_manifest()) == payload
