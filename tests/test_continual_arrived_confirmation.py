"""A fixed v7 request must survive both A/B resumes before final reporting."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from scripts.run_continual_arrived_confirmation import (
    prepare_request,
    run_confirmation,
)


def test_should_require_the_saved_unchanged_request_before_training(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    prepare_request(request_path)

    with pytest.raises(FileExistsError):
        prepare_request(request_path)
    with pytest.raises(ValueError, match="predeclared"):
        run_confirmation(tmp_path / "missing.json", tmp_path / "result.json", tmp_path / "ckpt")

    request_path.write_text(request_path.read_text(encoding="utf-8").replace("17", "23", 1))
    with pytest.raises(ValueError, match="predeclared"):
        run_confirmation(request_path, tmp_path / "result.json", tmp_path / "ckpt")
    assert not (tmp_path / "ckpt").exists()


def test_should_keep_all_trials_and_match_ab_resume_in_both_orders(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    prepare_request(request_path)

    report = run_confirmation(request_path, result_path, tmp_path / "ckpt")

    assert result_path.exists()
    assert report["request_sha256"] == sha256(request_path.read_bytes()).hexdigest()
    assert report["checks"]["model_order_isolated"] is True
    assert report["checks"]["total_model_updates"] == 96
    assert set(report["orders"]) == {"forward", "reverse"}
    for order in report["orders"].values():
        assert order["checkpoint_matches_ordinary"] is True
        assert order["model_updates"] == {"ordinary": 24, "checkpoint_resume": 24}
        assert order["interruptions"] == ["phase_a_wake", "phase_b_wake"]
        assert len(order["final_source_reads"]["ordinary"]) == 8
        assert len(order["final_source_reads"]["checkpoint_resume"]) == 8
        assert len(order["result"]["trials"]) == 12
        assert len(order["result"]["final_seed_results"]) == 2
        assert len(order["seed_differences"]) == 2
        assert {
            (trial["candidate_id"], trial["seed"], trial["method"])
            for trial in order["result"]["trials"]
        } == {
            (candidate, seed, method)
            for candidate in ("default", "lower_rate")
            for seed in (17, 19)
            for method in ("backprop", "predictive_coding", "circadian_predictive_coding")
        }
        for trial in order["result"]["trials"]:
            assert trial["train_updates"] == 2
            assert trial["train_examples_seen"] > 0
            assert trial["outer_examples_scored"] > 0
            if trial["method"] == "circadian_predictive_coding":
                for phase in ("replay_phase_a", "replay_phase_b"):
                    assert trial[phase]["example_count"] <= 4
                    assert trial[phase]["retained_bytes"] <= 96
        for seed_result, differences in zip(
            order["result"]["final_seed_results"], order["seed_differences"], strict=True
        ):
            roles = seed_result["role_ids"]
            for phase in ("a", "b"):
                phase_roles = [
                    set(roles[f"phase_{phase}_{role}"])
                    for role in ("train", "inner_guard", "outer_selection", "final_test")
                ]
                assert sum(map(len, phase_roles)) == len(set().union(*phase_roles))
            scores = seed_result["metrics"]
            circadian = scores["circadian_predictive_coding"]["balanced_score"]
            assert differences == {
                "seed": seed_result["seed"],
                "circadian_minus_backprop_balanced": (
                    circadian - scores["backprop"]["balanced_score"]
                ),
                "circadian_minus_predictive_balanced": (
                    circadian - scores["predictive_coding"]["balanced_score"]
                ),
            }
