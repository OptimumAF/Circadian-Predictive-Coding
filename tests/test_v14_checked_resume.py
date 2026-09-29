"""Interrupted v14 trials resume before the one global final-role seal."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import cast

import pytest

from scripts import run_versioned_v14_bundle as bundle_script
from src.app.continual_trigger_replay_outcomes import TriggerReplayComparison
from src.app.continual_trigger_replay_runner import (
    TriggerReplayTrainingResult,
    run_trigger_replay_training,
)
from src.app.continual_trigger_replay_schedule import TriggerReplayOpportunityManifest
from src.app.continual_trigger_replay_training_study import TriggerReplayTrainingStudy
from src.app.v14_trial_checkpoint import V14TrialPrefixCheckpoint
from src.infra.measured_observation_files import (
    DIAGNOSTIC_FILE,
    MEASUREMENT_DIRECTORY,
    verify_wake_diagnostic_sidecar,
)
from src.infra.v14_resume_files import V14ResumeFiles
from src.infra.versioned_run_files import verify_run_bundle


@pytest.fixture(scope="module")
def fresh_bytes(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, bytes, bytes]:
    root = tmp_path_factory.mktemp("v14-fresh")
    run = bundle_script.run_versioned_v14_bundle(
        root, "p53-fresh", Path.cwd(), capture_wake_diagnostics=True, resumable=True
    )
    assert V14ResumeFiles(root, "p53-fresh").load().status == "completed"
    verify_run_bundle(run)
    verify_wake_diagnostic_sidecar(run)
    return (
        (run / "training.json").read_bytes(),
        (run / "outcomes.json").read_bytes(),
        (run / MEASUREMENT_DIRECTORY / DIAGNOSTIC_FILE).read_bytes(),
    )


def _interrupt_after(
    monkeypatch: pytest.MonkeyPatch, completed: int, error: type[BaseException] = RuntimeError
) -> None:
    original = run_trigger_replay_training
    calls = 0

    def interrupted(
        manifest: TriggerReplayOpportunityManifest,
        *,
        seed: int,
        arm: str,
        capture_wake_diagnostics: bool = False,
    ) -> TriggerReplayTrainingResult:
        nonlocal calls
        calls += 1
        if calls > completed:
            raise error("interrupted trial")
        return original(
            manifest, seed=seed, arm=arm, capture_wake_diagnostics=capture_wake_diagnostics
        )

    monkeypatch.setattr(bundle_script, "run_trigger_replay_training", interrupted)


@pytest.mark.parametrize("completed", [1, 3])
def test_should_resume_only_missing_trials_and_repeat_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_bytes: tuple[bytes, bytes, bytes],
    completed: int,
) -> None:
    run_id = f"p53-resume-{completed}"
    _interrupt_after(monkeypatch, completed)
    with pytest.raises(RuntimeError, match="interrupted trial"):
        bundle_script.run_versioned_v14_bundle(
            tmp_path, run_id, Path.cwd(), capture_wake_diagnostics=True, resumable=True
        )
    files = V14ResumeFiles(tmp_path, run_id)
    assert not (tmp_path / run_id).exists()
    assert files.load().status == "failed"
    assert files.load().next_trial_index == completed
    assert files.load().next_cell == ((47, "adaptive") if completed == 1 else (53, "periodic"))

    trained: list[tuple[int, str]] = []
    original = run_trigger_replay_training

    def counting(
        manifest: TriggerReplayOpportunityManifest,
        *,
        seed: int,
        arm: str,
        capture_wake_diagnostics: bool = False,
    ) -> TriggerReplayTrainingResult:
        trained.append((seed, arm))
        return original(
            manifest, seed=seed, arm=arm, capture_wake_diagnostics=capture_wake_diagnostics
        )

    monkeypatch.setattr(bundle_script, "run_trigger_replay_training", counting)
    release = bundle_script.score_trigger_replay_training_study

    def checked_release(study: TriggerReplayTrainingStudy) -> TriggerReplayComparison:
        assert files.load().next_trial_index == 6
        return release(study)

    monkeypatch.setattr(bundle_script, "score_trigger_replay_training_study", checked_release)
    run = bundle_script.run_versioned_v14_bundle(
        tmp_path, run_id, Path.cwd(), capture_wake_diagnostics=True, resume=True
    )
    assert (
        trained
        == [(seed, arm) for seed in (47, 53) for arm in ("periodic", "adaptive", "no_sleep")][
            completed:
        ]
    )
    assert files.load().status == "completed"
    stored = files.load_checkpoint(files.load())
    assert all(
        trial.pending.phase_a._source is None and trial.pending.phase_b._source is None
        for trial in stored.trials
    )
    verify_run_bundle(run)
    verify_wake_diagnostic_sidecar(run)
    assert (
        (run / "training.json").read_bytes(),
        (run / "outcomes.json").read_bytes(),
        (run / MEASUREMENT_DIRECTORY / DIAGNOSTIC_FILE).read_bytes(),
    ) == fresh_bytes
    assert [sha256(item).hexdigest() for item in fresh_bytes] == [
        "174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324",
        "ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f",
        "2d6b6e4b0bc86a3dc292e7f04ad82816ae698dd43b9e63a5fceb41fe85f49f7c",
    ]


def test_should_record_cancellation_without_public_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _interrupt_after(monkeypatch, 1, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt, match="interrupted trial"):
        bundle_script.run_versioned_v14_bundle(tmp_path, "p53-cancel", Path.cwd(), resumable=True)
    state = V14ResumeFiles(tmp_path, "p53-cancel").load()
    assert state.status == "canceled" and state.next_trial_index == 1
    assert not (tmp_path / "p53-cancel").exists()


@pytest.mark.parametrize(
    "changed",
    ["source", "config_sha256", "protocol_sha256", "capture_wake_diagnostics", "next_cell"],
)
def test_should_reject_drift_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    _interrupt_after(monkeypatch, 1)
    with pytest.raises(RuntimeError):
        bundle_script.run_versioned_v14_bundle(tmp_path, "p53-drift", Path.cwd(), resumable=True)
    files = V14ResumeFiles(tmp_path, "p53-drift")
    state = files.load()
    if changed == "source":
        environment = dict(state.environment)
        source = dict(cast(dict[str, object], environment["source"]))
        source["workspace_sha256"] = "b" * 64
        environment["source"] = source
        state = replace(state, environment=environment)
    elif changed == "capture_wake_diagnostics":
        state = replace(state, capture_wake_diagnostics=True)
    elif changed == "next_cell":
        state = replace(state, next_cell=(53, "periodic"))
    elif changed == "config_sha256":
        state = replace(state, config_sha256="b" * 64)
    else:
        state = replace(state, protocol_sha256="b" * 64)
    files.write(state)
    monkeypatch.setattr(
        bundle_script,
        "run_trigger_replay_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trained before validation")),
    )
    with pytest.raises(ValueError, match="v14 resume"):
        bundle_script.run_versioned_v14_bundle(tmp_path, "p53-drift", Path.cwd(), resume=True)


def test_should_reject_changed_checkpoint_bytes_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _interrupt_after(monkeypatch, 1)
    with pytest.raises(RuntimeError):
        bundle_script.run_versioned_v14_bundle(tmp_path, "p53-checksum", Path.cwd(), resumable=True)
    files = V14ResumeFiles(tmp_path, "p53-checksum")
    state = files.load()
    assert state.checkpoint_file is not None
    with (files.directory / state.checkpoint_file).open("ab") as output:
        output.write(b"changed")
    monkeypatch.setattr(
        bundle_script,
        "run_trigger_replay_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trained before validation")),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        bundle_script.run_versioned_v14_bundle(tmp_path, "p53-checksum", Path.cwd(), resume=True)


@pytest.mark.parametrize("forgery", ["role", "replay", "work"])
def test_should_reject_rehashed_forged_trial_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forgery: str
) -> None:
    _interrupt_after(monkeypatch, 1)
    with pytest.raises(RuntimeError):
        bundle_script.run_versioned_v14_bundle(tmp_path, "p53-forged", Path.cwd(), resumable=True)
    files = V14ResumeFiles(tmp_path, "p53-forged")
    state = files.load()
    checkpoint = files.load_checkpoint(state)
    trial = checkpoint.trials[0]
    if forgery == "role":
        access = trial.pending.audit.accesses[0]
        trial.pending.audit.accesses[0] = replace(access, role="final_test")
    elif forgery == "replay":
        opportunity = trial.opportunities[3]
        changed = replace(opportunity, selected_ids=("f" * 64,))
        trial = replace(
            trial, opportunities=(*trial.opportunities[:3], changed, *trial.opportunities[4:])
        )
    else:
        opportunity = trial.opportunities[3]
        work = replace(opportunity.applied_by_method[0], optimizer_updates=9)
        changed = replace(opportunity, applied_by_method=(work, *opportunity.applied_by_method[1:]))
        trial = replace(
            trial, opportunities=(*trial.opportunities[:3], changed, *trial.opportunities[4:])
        )
    forged: V14TrialPrefixCheckpoint = replace(checkpoint, trials=(trial,))
    files.commit(state, forged)
    monkeypatch.setattr(
        bundle_script,
        "run_trigger_replay_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trained before validation")),
    )
    with pytest.raises(ValueError, match="v14 train-only|v14 periodic"):
        bundle_script.run_versioned_v14_bundle(tmp_path, "p53-forged", Path.cwd(), resume=True)


def test_should_refuse_stale_checkpoint_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _interrupt_after(monkeypatch, 3)
    with pytest.raises(RuntimeError):
        bundle_script.run_versioned_v14_bundle(tmp_path, "p53-stale", Path.cwd(), resumable=True)
    files = V14ResumeFiles(tmp_path, "p53-stale")
    state = files.load()
    old = next(files.directory.glob("checkpoint-01-*.bin"))
    files.write(
        replace(
            state, checkpoint_file=old.name, checkpoint_sha256=sha256(old.read_bytes()).hexdigest()
        )
    )
    monkeypatch.setattr(
        bundle_script,
        "run_trigger_replay_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trained before validation")),
    )
    with pytest.raises(ValueError, match="checkpoint identity"):
        bundle_script.run_versioned_v14_bundle(tmp_path, "p53-stale", Path.cwd(), resume=True)


def test_should_not_treat_partial_state_as_public_result(tmp_path: Path) -> None:
    files = V14ResumeFiles(tmp_path, "p53-incomplete")
    with files.claim():
        state = files.create(
            {"source": {"workspace_sha256": "a" * 64}}, "b" * 64, "c" * 64, False, (47, "periodic")
        )
    assert state.status == "incomplete"
    assert not (tmp_path / "p53-incomplete").exists()
    assert json.loads(files.state_path.read_text(encoding="utf-8"))["status"] == "incomplete"
    with pytest.raises(FileExistsError, match="unfinished resume state"):
        bundle_script.run_versioned_v14_bundle(tmp_path, "p53-incomplete", Path.cwd())


def test_should_finish_existing_bundle_after_sidecar_failure_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "p53-sidecar-retry"
    original = bundle_script.write_wake_diagnostic_sidecar
    monkeypatch.setattr(
        bundle_script,
        "write_wake_diagnostic_sidecar",
        lambda *args: (_ for _ in ()).throw(RuntimeError("sidecar interruption")),
    )
    with pytest.raises(RuntimeError, match="sidecar interruption"):
        bundle_script.run_versioned_v14_bundle(
            tmp_path, run_id, Path.cwd(), capture_wake_diagnostics=True, resumable=True
        )
    run = tmp_path / run_id
    files = V14ResumeFiles(tmp_path, run_id)
    assert files.load().status == "failed" and files.load().next_trial_index == 6
    assert verify_run_bundle(run)["status"] == "completed"
    assert not (run / MEASUREMENT_DIRECTORY).exists()
    original_training = (run / "training.json").read_bytes()
    original_outcomes = (run / "outcomes.json").read_bytes()

    monkeypatch.setattr(bundle_script, "write_wake_diagnostic_sidecar", original)
    resumed = bundle_script.run_versioned_v14_bundle(
        tmp_path, run_id, Path.cwd(), capture_wake_diagnostics=True, resume=True
    )
    assert resumed == run and files.load().status == "completed"
    assert (run / "training.json").read_bytes() == original_training
    assert (run / "outcomes.json").read_bytes() == original_outcomes
    verify_wake_diagnostic_sidecar(run)

    files.mark(files.load(), "failed", "InjectedFailure")
    (run / "training.json").write_bytes(original_training + b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        bundle_script.run_versioned_v14_bundle(
            tmp_path, run_id, Path.cwd(), capture_wake_diagnostics=True, resume=True
        )
    assert (run / "training.json").read_bytes() == original_training + b"changed"
