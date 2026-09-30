"""Public fixed v7 request, unscored checkpoint, and final-role artifact smoke.

The predeclared synthetic request is a protocol check, not a policy ranking.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from src.infra.circadian_checkpoint_files import TrustedLocalArrivedSelectionCheckpointStore


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE = "scripts.run_continual_arrived_confirmation"
DEVELOPMENT_ROLES = {
    f"phase_{phase}_{role}"
    for phase in ("a", "b")
    for role in ("train", "inner_guard", "outer_selection")
}


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite confirmation artifact value: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert isinstance(record, dict)
    return record


def _run_cli(*arguments: str, timeout: int = 210) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", MODULE, *arguments],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _assert_order_artifacts(name: str, order: dict[str, Any], checkpoint_dir: Path) -> None:
    checkpoint_path = checkpoint_dir / f"{name}.ckpt"
    assert checkpoint_path.is_file()
    checkpoint = TrustedLocalArrivedSelectionCheckpointStore(checkpoint_path).load()
    assert checkpoint.stage == order["checkpoint_stage"] == "frozen"
    assert checkpoint.manifest_digest == order["checkpoint_manifest_digest"]
    assert checkpoint.freeze is not None
    assert checkpoint.active_v6 is None
    assert checkpoint.seeds == (17, 19)
    assert len(checkpoint.completed_candidates) == 2
    assert order["checkpoint_matches_ordinary"] is True
    assert order["interruptions"] == ["phase_a_wake", "phase_b_wake"]
    assert order["model_updates"] == {"ordinary": 24, "checkpoint_resume": 24}
    assert all(len(reads) == 8 for reads in order["final_source_reads"].values())
    for candidate in checkpoint.completed_candidates:
        assert len(candidate.unscored_seeds) == 2
        for seed in candidate.unscored_seeds:
            assert set(dict(seed.role_hashes)) == DEVELOPMENT_ROLES
            assert all(access.role != "final_test" for access in seed.role_accesses)
            assert len(seed.sleep_events) == 2
            assert all(event.trigger_reason == "periodic" for event in seed.sleep_events)
            assert all(event.outcome in {"accepted", "rolled_back"} for event in seed.sleep_events)
    result = order["result"]
    assert len(result["trials"]) == 12
    assert len(result["final_seed_results"]) == 2
    assert len(order["seed_differences"]) == 2
    assert all(
        event["role"] != "final_test"
        for trial in result["trials"]
        for event in trial["role_accesses"]
    )
    for seed in result["final_seed_results"]:
        assert set(seed["role_hashes"]) == DEVELOPMENT_ROLES | {
            "phase_a_final_test",
            "phase_b_final_test",
        }
        assert all(
            event["role"] == "final_test" and event["event"] == "global_freeze"
            for event in seed["final_role_accesses"]
        )


def test_should_keep_fixed_v7_confirmation_unscored_until_final_release(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    checkpoint_dir = tmp_path / "checkpoints"
    prepared = _run_cli("prepare", "--request", str(request_path))
    assert prepared.returncode == 0, prepared.stderr
    assert prepared.stdout.strip() == str(request_path)
    request = _read_json(request_path)
    assert request["protocol_id"] == "continual_arrived_outer_selection_v7"
    assert request["seeds"] == [17, 19]
    assert request["budget"]["total_model_updates"] == 96
    assert set(request["candidates_by_order"]) == {"forward", "reverse"}
    assert not result_path.exists()
    assert not checkpoint_dir.exists()

    completed = _run_cli(
        "run",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
    )
    assert completed.returncode == 0, completed.stderr
    report = _read_json(result_path)
    printed = json.loads(completed.stdout, parse_constant=_reject_nonfinite)
    assert printed == {"result": str(result_path), "checks": report["checks"]}
    assert report["schema"] == "continual_arrived_confirmation_result_v1"
    assert report["request_sha256"] == sha256(request_path.read_bytes()).hexdigest()
    assert report["checks"] == {
        "model_order_isolated": True,
        "total_model_updates": 96,
        "final_source_sealed_until_freeze": True,
    }
    assert set(report["orders"]) == {"forward", "reverse"}
    for name, order in report["orders"].items():
        _assert_order_artifacts(name, order, checkpoint_dir)

    originals = {path: path.read_bytes() for path in (result_path, *checkpoint_dir.glob("*.ckpt"))}
    duplicate = _run_cli(
        "run",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        timeout=30,
    )
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr
    assert {path: path.read_bytes() for path in originals} == originals
