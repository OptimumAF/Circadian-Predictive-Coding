"""Run the fixed v14 study once and write an opt-in P5.1 provenance bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from scripts.run_continual_trigger_replay_outcomes import serialize_outcome_comparison
from scripts.run_continual_trigger_replay_training import serialize_training_study
from src.app.continual_trigger_replay_outcomes import score_trigger_replay_training_study
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.app.continual_trigger_replay_runner import run_trigger_replay_training
from src.app.continual_trigger_replay_training_study import (
    preflight_trigger_replay_training_study,
    run_trigger_replay_training_study,
)
from src.app.v14_measured_observations import serialize_wake_diagnostic_study
from src.app.v14_trial_checkpoint import (
    build_v14_trial_checkpoint,
    expected_v14_cells,
    study_from_v14_trial_checkpoint,
    v14_resume_protocol_sha256,
    validate_v14_trial_checkpoint,
)
from src.app.versioned_v14_run import build_v14_run_manifest
from src.app.continual_checkpoint import continual_config_digest
from src.core.run_manifest import validate_run_id
from src.infra.run_environment import capture_run_environment
from src.infra.measured_observation_files import (
    DIAGNOSTIC_FILE,
    MEASUREMENT_DIRECTORY,
    verify_wake_diagnostic_sidecar,
    write_wake_diagnostic_sidecar,
)
from src.infra.v14_resume_files import V14ResumeFiles, V14ResumeState
from src.infra.versioned_run_files import verify_run_bundle, write_run_bundle


def run_versioned_v14_bundle(
    output_root: str | Path,
    run_id: str,
    repository: str | Path,
    *,
    capture_wake_diagnostics: bool = False,
    resumable: bool = False,
    resume: bool = False,
) -> Path:
    """Refuse an occupied ID, capture source, then emit one complete run."""
    safe_id = validate_run_id(run_id)
    destination = Path(output_root) / safe_id
    if resumable and resume:
        raise ValueError("choose either a new resumable run or resume an existing one")
    if not resume and destination.exists():
        raise FileExistsError(f"run directory already exists: {destination}")
    if not resumable and not resume and V14ResumeFiles(output_root, safe_id).directory.exists():
        raise FileExistsError("v14 run ID has an unfinished resume state")
    if resumable or resume:
        files = V14ResumeFiles(output_root, safe_id)
        with files.claim():
            return _run_checked_resume(
                files, destination, repository, capture_wake_diagnostics, resume
            )
    before = capture_run_environment(repository)
    study = run_trigger_replay_training_study(
        fixed_trigger_replay_manifest(), capture_wake_diagnostics=capture_wake_diagnostics
    )
    training = serialize_training_study(study).encode("utf-8")
    diagnostics = serialize_wake_diagnostic_study(study) if capture_wake_diagnostics else None
    comparison = score_trigger_replay_training_study(study)
    outcomes = serialize_outcome_comparison(comparison).encode("utf-8")
    after = capture_run_environment(repository)
    if before.source != after.source:
        raise ValueError("repository source changed during the v14 run")
    manifest = build_v14_run_manifest(safe_id, before, study, comparison, training, outcomes)
    written = write_run_bundle(output_root, manifest, training, outcomes)
    if diagnostics is not None:
        # Why this: the sidecar must not exist until all final roles were
        # globally sealed, scored, and published as a completed bundle.
        write_wake_diagnostic_sidecar(written, diagnostics)
    return written


def _validate_resume_state(
    state: V14ResumeState,
    environment: dict[str, object],
    config_sha256: str,
    protocol_sha256: str,
    capture_wake_diagnostics: bool,
    cells: tuple[tuple[int, str], ...],
) -> None:
    index = state.next_trial_index
    if (
        state.environment != environment
        or state.config_sha256 != config_sha256
        or state.protocol_sha256 != protocol_sha256
        or state.capture_wake_diagnostics is not capture_wake_diagnostics
        or not 0 <= index <= len(cells)
        or state.next_cell != (cells[index] if index < len(cells) else None)
        or (index == 0) != (state.checkpoint_file is None)
        or (index == 0) != (state.checkpoint_sha256 is None)
    ):
        raise ValueError("v14 resume source, config, protocol, capture, or cursor differs")


def _run_checked_resume(
    files: V14ResumeFiles,
    destination: Path,
    repository: str | Path,
    capture_wake_diagnostics: bool,
    resume: bool,
) -> Path:
    """Continue only a verified unscored prefix; publish after the global gate."""
    manifest = fixed_trigger_replay_manifest()
    cells = expected_v14_cells(manifest)
    before = capture_run_environment(repository)
    source_digest = before.source.get("workspace_sha256")
    if type(source_digest) is not str:
        raise ValueError("v14 resume requires an available Git workspace SHA-256")
    config_sha256 = continual_config_digest(manifest, manifest.seeds)
    protocol_sha256 = v14_resume_protocol_sha256()
    environment = asdict(before)
    if resume:
        state = files.load()
        _validate_resume_state(
            state, environment, config_sha256, protocol_sha256, capture_wake_diagnostics, cells
        )
        if state.next_trial_index:
            checkpoint = files.load_checkpoint(state)
            validate_v14_trial_checkpoint(
                checkpoint, manifest, source_digest, capture_wake_diagnostics
            )
            if (
                checkpoint.next_trial_index != state.next_trial_index
                or checkpoint.next_cell != state.next_cell
            ):
                raise ValueError("v14 resume state and checkpoint cursor differ")
            trials = checkpoint.trials
        else:
            trials = ()
        if state.status == "completed":
            if state.next_trial_index != len(cells):
                raise ValueError("completed v14 resume state lacks all trial checkpoints")
            verify_run_bundle(destination)
            if capture_wake_diagnostics:
                verify_wake_diagnostic_sidecar(destination)
            return destination
        state = files.mark(state, "incomplete")
    else:
        if destination.exists():
            raise FileExistsError(f"run directory already exists: {destination}")
        state = files.create(
            environment, config_sha256, protocol_sha256, capture_wake_diagnostics, cells[0]
        )
        trials = ()
    try:
        for seed, arm in cells[len(trials) :]:
            trial = run_trigger_replay_training(
                manifest,
                seed=seed,
                arm=arm,
                capture_wake_diagnostics=capture_wake_diagnostics,
            )
            candidate = (*trials, trial)
            checkpoint = build_v14_trial_checkpoint(
                manifest, source_digest, capture_wake_diagnostics, candidate
            )
            state = files.commit(state, checkpoint)
            trials = candidate
        checkpoint = build_v14_trial_checkpoint(
            manifest, source_digest, capture_wake_diagnostics, trials
        )
        study = study_from_v14_trial_checkpoint(checkpoint, manifest)
        preflight_trigger_replay_training_study(study)
        training = serialize_training_study(study).encode("utf-8")
        diagnostics = serialize_wake_diagnostic_study(study) if capture_wake_diagnostics else None
        comparison = score_trigger_replay_training_study(study)
        outcomes = serialize_outcome_comparison(comparison).encode("utf-8")
        after = capture_run_environment(repository)
        if before != after:
            raise ValueError("repository or execution environment changed during the v14 run")
        bundle = build_v14_run_manifest(files.run_id, before, study, comparison, training, outcomes)
        if destination.exists():
            if (
                verify_run_bundle(destination) != bundle
                or (destination / "training.json").read_bytes() != training
                or (destination / "outcomes.json").read_bytes() != outcomes
            ):
                raise FileExistsError("existing v14 bundle differs from resumed output")
            written = destination
        else:
            written = write_run_bundle(files.output_root, bundle, training, outcomes)
        if diagnostics is not None:
            sidecar = written / MEASUREMENT_DIRECTORY
            if sidecar.exists():
                verify_wake_diagnostic_sidecar(written)
                if (sidecar / DIAGNOSTIC_FILE).read_bytes() != diagnostics:
                    raise FileExistsError("existing v14 diagnostics differ from resumed output")
            else:
                write_wake_diagnostic_sidecar(written, diagnostics)
        files.mark(state, "completed")
        return written
    except BaseException as exc:
        status = "canceled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        files.mark(state, status, type(exc).__name__)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run-id", help="new lowercase run slug")
    action.add_argument("--verify-run", type=Path, help="verify an existing v14 bundle")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--capture-wake-diagnostics",
        action="store_true",
        help="save separately versioned real wake metrics after completed scoring",
    )
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--resumable", action="store_true", help="start a checked trial cursor")
    lifecycle.add_argument("--resume", action="store_true", help="continue a checked trial cursor")
    arguments = parser.parse_args()
    if arguments.verify_run is not None:
        if arguments.resumable or arguments.resume:
            parser.error("resume flags require --run-id")
        manifest = verify_run_bundle(arguments.verify_run)
        print(json.dumps({"run": str(arguments.verify_run), "status": manifest["status"]}))
        return
    repository = Path(__file__).resolve().parents[1]
    destination = run_versioned_v14_bundle(
        arguments.output_root,
        arguments.run_id,
        repository,
        capture_wake_diagnostics=arguments.capture_wake_diagnostics,
        resumable=arguments.resumable,
        resume=arguments.resume,
    )
    payload = (destination / "manifest.json").read_bytes()
    print(json.dumps({"run": str(destination), "manifest_sha256": sha256(payload).hexdigest()}))


if __name__ == "__main__":
    main()
