"""The opt-in P5.1 bundle records one fixed study without changing v14 JSON."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts.run_continual_trigger_replay_outcomes import serialize_outcome_comparison
from scripts.run_continual_trigger_replay_training import serialize_training_study
from scripts.run_versioned_v14_bundle import main, run_versioned_v14_bundle
from src.app.continual_trigger_replay_outcomes import (
    TriggerReplayComparison,
    score_trigger_replay_training_study,
)
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.app.continual_trigger_replay_training_study import (
    TriggerReplayTrainingStudy,
    run_trigger_replay_training_study,
)
from src.app.versioned_v14_run import build_v14_run_manifest
from src.core.run_manifest import RunEnvironment
from src.infra.versioned_run_files import verify_run_bundle, write_run_bundle


@pytest.fixture(scope="module")
def fixed_materials() -> tuple[TriggerReplayTrainingStudy, TriggerReplayComparison, bytes, bytes]:
    study = run_trigger_replay_training_study(fixed_trigger_replay_manifest())
    training = serialize_training_study(study).encode()
    comparison = score_trigger_replay_training_study(study)
    return study, comparison, training, serialize_outcome_comparison(comparison).encode()


def _environment() -> RunEnvironment:
    return RunEnvironment(
        source={
            "commit_sha": "c" * 40,
            "dirty": True,
            "status_sha256": "d" * 64,
            "tracked_diff_sha256": "e" * 64,
            "workspace_sha256": "f" * 64,
            "untracked_file_count": 2,
            "unavailable_reason": None,
        },
        dependency_versions={"python": "3.11.9", "numpy": "2.4.0"},
        hardware={
            "system": "Windows",
            "release": "11",
            "machine": "AMD64",
            "processor": "Fixture CPU",
            "processor_unavailable_reason": None,
            "logical_cpu_count": 8,
            "compute_device": "cpu",
        },
        python_hash_seed=None,
    )


def _build_manifest(
    fixed_materials: tuple[TriggerReplayTrainingStudy, TriggerReplayComparison, bytes, bytes],
) -> dict[str, Any]:
    study, comparison, training, outcomes = fixed_materials
    return build_v14_run_manifest(
        "p51-v14-fixture", _environment(), study, comparison, training, outcomes
    )


def test_should_bind_all_required_provenance_and_original_v14_bytes(
    fixed_materials: tuple[TriggerReplayTrainingStudy, TriggerReplayComparison, bytes, bytes],
) -> None:
    manifest = _build_manifest(fixed_materials)
    _, _, training, outcomes = fixed_materials
    assert (
        sha256(training).hexdigest()
        == "174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324"
    )
    assert (
        sha256(outcomes).hexdigest()
        == "ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f"
    )
    assert manifest["status"] == "completed"
    assert manifest["source"]["dirty"] is True
    assert manifest["seed_map"]["47"]["phase_b_source"] == 148
    assert manifest["seed_map"]["53"]["circadian"] == 55
    for seed in ("47", "53"):
        for phase in ("a", "b"):
            assert set(manifest["dataset_split_hashes"][seed][phase]) == {
                "train",
                "inner_guard",
                "outer_selection",
                "final_test",
            }


def test_should_reject_payload_or_study_identity_drift(
    fixed_materials: tuple[TriggerReplayTrainingStudy, TriggerReplayComparison, bytes, bytes],
) -> None:
    study, comparison, training, outcomes = fixed_materials
    forged = training.replace(
        b"continual_trigger_replay_train_only_v14", b"forged_train_protocol_v99", 1
    )
    with pytest.raises(ValueError, match="training payload"):
        build_v14_run_manifest(
            "p51-v14-fixture", _environment(), study, comparison, forged, outcomes
        )


def test_should_write_once_and_reject_tampered_or_missing_files(
    tmp_path: Path,
    fixed_materials: tuple[TriggerReplayTrainingStudy, TriggerReplayComparison, bytes, bytes],
) -> None:
    manifest = _build_manifest(fixed_materials)
    _, _, training, outcomes = fixed_materials
    run_path = write_run_bundle(tmp_path, manifest, training, outcomes)
    verified = verify_run_bundle(run_path)
    assert verified == manifest
    assert (run_path / "training.json").read_bytes() == training
    assert (run_path / "outcomes.json").read_bytes() == outcomes
    with pytest.raises(FileExistsError):
        write_run_bundle(tmp_path, manifest, training, outcomes)

    (run_path / "outcomes.json").write_bytes(outcomes + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_run_bundle(run_path)
    (run_path / "outcomes.json").write_bytes(outcomes)
    (run_path / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest"):
        verify_run_bundle(run_path)


def test_should_reject_partial_cells_even_when_file_checksum_is_updated(
    tmp_path: Path,
    fixed_materials: tuple[TriggerReplayTrainingStudy, TriggerReplayComparison, bytes, bytes],
) -> None:
    manifest = _build_manifest(fixed_materials)
    _, _, training, outcomes = fixed_materials
    run_path = write_run_bundle(tmp_path, manifest, training, outcomes)
    partial = json.loads(training)
    partial["rows"].pop()
    partial_bytes = (json.dumps(partial, indent=2, sort_keys=True) + "\n").encode()
    manifest["files"]["training"]["sha256"] = sha256(partial_bytes).hexdigest()
    (run_path / "training.json").write_bytes(partial_bytes)
    (run_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cells"):
        verify_run_bundle(run_path)


def test_should_refuse_existing_run_id_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "p51-v14-fixture"
    (tmp_path / run_id).mkdir()
    monkeypatch.setattr(
        "scripts.run_versioned_v14_bundle.run_trigger_replay_training_study",
        lambda *_: (_ for _ in ()).throw(AssertionError("trained before exclusive path check")),
    )
    with pytest.raises(FileExistsError):
        run_versioned_v14_bundle(tmp_path, run_id, Path.cwd())


def test_should_emit_one_actual_opt_in_bundle_with_tracked_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_path = run_versioned_v14_bundle(tmp_path, "p51-v14-integration", Path.cwd())
    manifest = verify_run_bundle(run_path)
    assert manifest["run_id"] == "p51-v14-integration"
    assert manifest["source"]["commit_sha"] is not None
    assert json.loads((run_path / "outcomes.json").read_text(encoding="utf-8"))["protocol_id"] == (
        "continual_trigger_replay_outcomes_v14"
    )
    monkeypatch.setattr(sys, "argv", ["run_versioned_v14_bundle", "--verify-run", str(run_path)])
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_should_reject_source_change_during_training_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _environment()
    changed = _environment()
    changed.source["workspace_sha256"] = "0" * 64
    observations = iter((before, changed))
    monkeypatch.setattr(
        "scripts.run_versioned_v14_bundle.capture_run_environment",
        lambda *_: next(observations),
    )
    with pytest.raises(ValueError, match="source changed"):
        run_versioned_v14_bundle(tmp_path, "p51-v14-drift", Path.cwd())
    assert not (tmp_path / "p51-v14-drift").exists()
