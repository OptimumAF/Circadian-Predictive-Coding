"""Fixed v8 policy CLI output and unscored-checkpoint smoke.

The paired policy rows are retained without selecting a winner.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import run_continual_replay_policy_smoke as policy_cli
from src.infra.circadian_checkpoint_files import TrustedLocalReplayPolicyCheckpointStore


REPOSITORY = Path(__file__).resolve().parents[1]
DEVELOPMENT_ROLES = {
    f"phase_{phase}_{role}"
    for phase in ("a", "b")
    for role in ("train", "inner_guard", "outer_selection")
}


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite policy artifact value: {value}")


def _assert_policy_artifacts(result: dict[str, Any], checkpoint_path: Path) -> None:
    checkpoint = TrustedLocalReplayPolicyCheckpointStore(checkpoint_path).load()
    assert checkpoint.manifest_digest == result["manifest_digest"]
    assert checkpoint.next_trial_index == len(checkpoint.unscored_seeds) == 4
    assert checkpoint.active is None
    for record in checkpoint.unscored_seeds:
        assert set(dict(record.arrived.role_hashes)) == DEVELOPMENT_ROLES
        assert all(event.role != "final_test" for event in record.arrived.role_accesses)
    first, second = result["policies"]
    for fifo, reservoir in zip(first["seeds"], second["seeds"], strict=True):
        assert fifo["arrived"]["seed"] == reservoir["arrived"]["seed"]
        assert fifo["arrived"]["role_hashes"] == reservoir["arrived"]["role_hashes"]
        assert fifo["baseline_state_digests"] == reservoir["baseline_state_digests"]
        for row in (fifo, reservoir):
            arrived = row["arrived"]
            assert set(arrived["role_hashes"]) == DEVELOPMENT_ROLES | {
                "phase_a_final_test",
                "phase_b_final_test",
            }
            assert [
                event["completed_epoch"]
                for event in arrived["metrics"]["circadian_predictive_coding"]["sleep_events"]
            ] == [1, 2, 3, 4]
            assert any(
                event["outcome"] in {"accepted", "rolled_back"}
                for event in arrived["metrics"]["circadian_predictive_coding"]["sleep_events"]
            )
            retention = arrived["metrics"]["replay_retention"]
            assert retention["budget_examples"] == 4
            assert retention["budget_bytes"] == 96
            assert all(retention[phase]["example_count"] <= 4 for phase in ("phase_a", "phase_b"))
            assert row["exposure"]["after_b"]["replay_updates"] > 0


def test_should_refuse_occupied_v8_result_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "occupied.json"
    result_path.write_bytes(b"prior result")
    monkeypatch.setattr(sys, "argv", ["policy", "--result", str(result_path)])
    monkeypatch.setattr(
        policy_cli,
        "run_replay_policy_comparison",
        lambda *_args, **_kwargs: pytest.fail("trained before occupied result preflight"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        policy_cli.main()
    assert result_path.read_bytes() == b"prior result"


def test_should_refuse_occupied_checkpoint_on_fresh_v8_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "new-result.json"
    checkpoint_path = tmp_path / "occupied.checkpoint"
    checkpoint_path.write_bytes(b"prior checkpoint")
    monkeypatch.setattr(
        sys,
        "argv",
        ["policy", "--result", str(result_path), "--checkpoint", str(checkpoint_path)],
    )
    monkeypatch.setattr(
        policy_cli,
        "run_replay_policy_comparison",
        lambda *_args, **_kwargs: pytest.fail("trained before occupied checkpoint preflight"),
    )
    with pytest.raises(FileExistsError, match="checkpoint.*already exists"):
        policy_cli.main()
    assert checkpoint_path.read_bytes() == b"prior checkpoint"
    assert not result_path.exists()


def test_should_reject_shared_v8_result_and_checkpoint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_path = tmp_path / "shared-output"
    monkeypatch.setattr(
        sys, "argv", ["policy", "--result", str(shared_path), "--checkpoint", str(shared_path)]
    )
    monkeypatch.setattr(
        policy_cli,
        "run_replay_policy_comparison",
        lambda *_args, **_kwargs: pytest.fail("trained with aliased output paths"),
    )
    with pytest.raises(SystemExit) as rejected:
        policy_cli.main()
    assert rejected.value.code == 2
    assert not shared_path.exists()


def test_should_publish_v8_policy_result_and_unscored_checkpoint(tmp_path: Path) -> None:
    result_path = tmp_path / "policy-result.json"
    checkpoint_path = tmp_path / "policy.checkpoint"
    command = [
        sys.executable,
        "-m",
        "scripts.run_continual_replay_policy_smoke",
        "--result",
        str(result_path),
        "--checkpoint",
        str(checkpoint_path),
    ]
    completed = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=60, check=False
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    summary = json.loads(completed.stdout, parse_constant=_reject_nonfinite)
    assert isinstance(result, dict)
    assert summary["result"] == str(result_path)
    assert summary["sha256"] == sha256(result_path.read_bytes()).hexdigest()
    assert summary["manifest_digest"] == result["manifest_digest"]
    assert (
        result["protocol_id"]
        == result["manifest"]["protocol_id"]
        == "continual_replay_policy_comparison_v8"
    )
    assert result["manifest"]["seeds"] == [17, 19]
    assert [row["policy"]["name"] for row in result["policies"]] == [
        "recent_fifo",
        "seeded_reservoir",
    ]
    assert len(summary["scores"]) == 4
    assert {(row["policy"], row["seed"]) for row in summary["scores"]} == {
        (policy, seed) for policy in ("recent_fifo", "seeded_reservoir") for seed in (17, 19)
    }
    _assert_policy_artifacts(result, checkpoint_path)

    original = result_path.read_bytes()
    duplicate = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=30, check=False
    )
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr
    assert result_path.read_bytes() == original

    resumed_path = tmp_path / "resumed-result.json"
    checkpoint_bytes = checkpoint_path.read_bytes()
    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_continual_replay_policy_smoke",
            "--result",
            str(resumed_path),
            "--checkpoint",
            str(checkpoint_path),
            "--resume",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (
        json.loads(resumed_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
        == result
    )
    assert checkpoint_path.read_bytes() == checkpoint_bytes
