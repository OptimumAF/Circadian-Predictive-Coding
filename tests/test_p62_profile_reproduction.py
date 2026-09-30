"""Frozen profile requests and complete artifact audits precede P6.2 scores."""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from scripts import run_p62_profile_reproduction as study
from scripts import verify_p62_profile_repeats as repeat_verifier
from src.app.continual_shift_benchmark import ContinualShiftConfig


def test_should_keep_current_profile_sources_and_policies_frozen() -> None:
    assert study.require_frozen_sources() == study.SOURCE_SHA256
    assert set(study.PROFILE_BUDGET_SECONDS) == {
        "baseline",
        "strength-case",
        "hardest-case",
    }
    for profile in study.PROFILE_BUDGET_SECONDS:
        assert study.digest_json(study.profile_policy(profile)) == study.POLICY_SHA256[profile]
    assert study.SEEDS == (3, 7, 11, 19, 23, 31, 37)


def test_should_save_exact_request_before_child_and_refuse_occupied_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "new"
    paths = study.artifact_paths(output_dir, "baseline")
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert paths["request"].is_file()
        request = study.read_finite_json(paths["request"])
        assert request["schema_id"] == "p62_profile_request_v1"
        assert request["protocol_id"] == "continual_validation_v1"
        assert request["seeds"] == list(study.SEEDS)
        assert request["source_sha256"] == study.SOURCE_SHA256
        assert request["circadian_policy_sha256"] == study.POLICY_SHA256["baseline"]
        assert request["wall_limit_seconds"] == kwargs["timeout"] == 240
        assert request["command"] == command
        assert all(Path(path).is_absolute() for path in (command[-5], command[-3], command[-1]))
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, "finished stdout", "")

    monkeypatch.setattr(study.subprocess, "run", fake_run)
    monkeypatch.setattr(
        study,
        "verify_artifacts",
        lambda profile, files, stdout, policy: {
            "schema_id": "p62_profile_audit_v1",
            "profile": profile,
            "stdout": stdout,
        },
    )

    audit = study.run_profile("baseline", output_dir)

    assert audit["profile"] == "baseline"
    assert audit["stdout"] == "finished stdout"
    assert audit["elapsed_seconds"] >= 0
    assert len(observed) == 1
    assert study.read_finite_json(paths["audit"])["profile"] == "baseline"
    with pytest.raises(FileExistsError, match="already exists"):
        study.run_profile("baseline", output_dir)
    assert len(observed) == 1


def test_should_record_timeout_without_a_false_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "timeout"
    paths = study.artifact_paths(output_dir, "baseline")

    def timeout(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial")

    monkeypatch.setattr(study.subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match="wall limit"):
        study.run_profile("baseline", output_dir)

    failure = study.read_finite_json(paths["failure"])
    assert failure["reason"] == "wall_limit"
    assert failure["stdout"] == "partial"
    assert paths["request"].is_file()
    assert not paths["audit"].exists()


def synthetic_result() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(json.dumps(asdict(ContinualShiftConfig())))
    policy = study.profile_policy("baseline")
    assert config["circadian_config"] == policy
    metric_values = {metric: 0.5 for metric in study.SHIFT_METRICS}
    rows = []
    for seed in study.SEEDS:
        rows.append(
            {
                "seed": seed,
                "split_hashes": {
                    role: f"{index:064x}"
                    for index, role in enumerate(sorted(study.ROLE_NAMES), start=1)
                },
                "backprop": dict(metric_values),
                "predictive_coding": dict(metric_values),
                "circadian_predictive_coding": {
                    **metric_values,
                    "sleep_event_count": 0,
                    "total_splits": 0,
                    "total_prunes": 0,
                    "hidden_dim_end": 12,
                },
            }
        )
    aggregate: dict[str, Any] = {"run_count": 7}
    for name in study.CONTINUAL_MODEL_ORDER:
        aggregate[name] = {
            f"{kind}_{metric}": 0.5 if kind == "mean" else 0.0
            for metric in study.SHIFT_METRICS
            for kind in ("mean", "std")
        }
    aggregate["circadian_predictive_coding"].update(
        mean_sleep_event_count=0.0,
        mean_total_splits=0.0,
        mean_total_prunes=0.0,
        mean_hidden_dim_end=12.0,
    )
    result = {
        "config": config,
        "seeds": list(study.SEEDS),
        "seed_results": rows,
        "aggregate": aggregate,
    }
    resolved = {
        "schema_id": "continual_resolved_config_v1",
        "preset": "baseline",
        "seeds": list(study.SEEDS),
        "overrides": {},
        "config": config,
    }
    return result, resolved, policy


def write_synthetic_artifacts(
    tmp_path: Path, result: dict[str, Any], resolved: dict[str, Any]
) -> dict[str, Path]:
    paths = study.artifact_paths(tmp_path, "baseline")
    paths["request"].write_text("{}\n", encoding="utf-8")
    hashes_by_seed = {row["seed"]: row["split_hashes"] for row in result["seed_results"]}
    metrics = (
        "A_pre=0.500+/-0.000, A_post=0.500+/-0.000, "
        "B_post=0.500+/-0.000, retention=0.500+/-0.000, "
        "balanced=0.500+/-0.000"
    )
    paths["text"].write_text(
        "Continual Shift Benchmark\n"
        "Protocol: continual_validation_v1\n"
        "Circadian sleep mode: legacy\n"
        f"Seeds: {list(study.SEEDS)}\n"
        f"Split hashes by seed: {hashes_by_seed}\n"
        "Phase B train fraction: 0.14\n"
        f"Backprop: {metrics}\n"
        f"Predictive coding: {metrics}\n"
        f"Circadian predictive coding: {metrics}, "
        "sleep_events=0.00, splits=0.00, prunes=0.00, hidden_end=12.00\n",
        encoding="utf-8",
    )
    paths["result"].write_text(json.dumps(result), encoding="utf-8")
    paths["config"].write_text(json.dumps(resolved), encoding="utf-8")
    return paths


def test_should_verify_complete_profile_artifacts_and_reject_metric_drift(tmp_path: Path) -> None:
    result, resolved, policy = synthetic_result()
    paths = write_synthetic_artifacts(tmp_path, result, resolved)
    stdout = paths["text"].read_text(encoding="utf-8")

    audit = study.verify_artifacts("baseline", paths, stdout, policy)

    assert audit["result_count"] == 7
    assert set(audit["artifact_sha256"]) == {"request", "text", "result", "config"}
    incorrect_text = stdout.replace("Backprop: A_pre=0.500", "Backprop: A_pre=0.501")
    paths["text"].write_text(incorrect_text, encoding="utf-8")
    with pytest.raises(ValueError, match="text/JSON summary differs"):
        study.verify_artifacts("baseline", paths, incorrect_text, policy)
    paths["text"].write_text(stdout, encoding="utf-8")
    result["seed_results"][0]["backprop"]["balanced_score"] = 0.8
    paths["result"].write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate differs"):
        study.verify_artifacts("baseline", paths, stdout, policy)


def test_should_reject_nonfinite_json_even_when_parser_accepts_large_exponent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad.json"
    source.write_text('{"metric": 1e309}', encoding="utf-8")

    with pytest.raises(ValueError, match="nonfinite"):
        study.read_finite_json(source)


def test_should_exclude_only_measured_sleep_durations_from_repeat_digest() -> None:
    first: dict[str, Any] = {
        "seed_results": [
            {
                "circadian_predictive_coding": {
                    "sleep_events": [
                        {
                            "outcome": "applied",
                            "durations": {"attempt_seconds": 0.1, "core_seconds": 0.09},
                        }
                    ]
                }
            }
        ]
    }
    second = deepcopy(first)
    second["seed_results"][0]["circadian_predictive_coding"]["sleep_events"][0]["durations"][
        "attempt_seconds"
    ] = 0.2

    changed = repeat_verifier.differences(first, second)

    assert len(changed) == 1
    assert repeat_verifier.is_measured_duration(changed[0])
    assert repeat_verifier.deterministic_digest(first) == repeat_verifier.deterministic_digest(
        second
    )

    second["seed_results"][0]["circadian_predictive_coding"]["sleep_events"][0]["outcome"] = (
        "rejected"
    )
    changed = repeat_verifier.differences(first, second)
    assert any(not repeat_verifier.is_measured_duration(path) for path in changed)
    assert repeat_verifier.deterministic_digest(first) != repeat_verifier.deterministic_digest(
        second
    )

    second = deepcopy(first)
    second["seed_results"][0]["circadian_predictive_coding"]["sleep_events"][0]["durations"][
        "new_deterministic_field"
    ] = 42
    assert repeat_verifier.deterministic_digest(first) != repeat_verifier.deterministic_digest(
        second
    )
