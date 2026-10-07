"""Freeze the next matched-head study from one development-only cost probe.

This writes a request only. It neither trains heads nor opens final test.
The later selection and confirmation runners must restore this exact file.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import profile_cifar_representative_feasibility as probe  # noqa: E402
from scripts import run_cifar_pretrained_validation as prior_study  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402


PROBE_RESULT_PATH = probe.RESULT_PATH
PROBE_RESULT_SHA256 = "a1180542abd4b1e51cd545022f6b501842ea3a340609e9888d38d9c35bb08649"
REQUEST_PATH = REPO_ROOT / "data" / "cifar-representative-study-v1-request.json"
SELECTION_SEED = 179
CONFIRMATION_SEEDS = (181, 191, 193)


def _read_verified_probe() -> tuple[dict[str, Any], str]:
    if not PROBE_RESULT_PATH.is_file():
        raise FileNotFoundError("representative study requires the saved cost probe")
    raw = PROBE_RESULT_PATH.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != PROBE_RESULT_SHA256:
        raise ValueError("saved representative cost probe changed")
    result: dict[str, Any] = json.loads(raw)
    _, request_digest = probe._read_request(probe.REQUEST_PATH)
    measurement = result["measurement"]
    if (
        result["request_sha256"] != request_digest
        or measurement["image_size"] != 224
        or measurement["seed"] != 173
        or measurement["final_test_constructions"] != 0
        or measurement["final_test_iterations"] != 0
        or {role: row["examples"] for role, row in measurement["roles"].items()}
        != {"train": 4096, "guard": 512, "validation": 512}
        or set(measurement["split_hashes"]) != set(probe.ROLE_ORDER)
        or measurement["elapsed_seconds"] > probe.PROBE_TIMEOUT_SECONDS
        or any(
            row["utilization_percent"] > probe.MAX_UTILIZATION_PERCENT
            or row["free_mib"] < probe.MIN_FREE_MIB
            for row in result["quiet_window"]
        )
    ):
        raise ValueError("saved cost probe failed a development-only budget gate")
    return result, digest


def _base_config() -> ResNet50BenchmarkConfig:
    return replace(
        prior_study._base_config(),
        seed=SELECTION_SEED,
        device="cuda",
        image_size=224,
        dataset_train_subset_size=16384,
        dataset_guard_subset_size=2048,
        dataset_validation_subset_size=2048,
        dataset_test_subset_size=4096,
        batch_size=32,
    )


def _study_request(probe_result: dict[str, Any], probe_digest: str) -> dict[str, Any]:
    base = _base_config()
    candidate_grid = prior_study._candidates(base)
    measured = probe_result["measurement"]
    projected_feature_seconds = measured["setup_seconds"] + 4 * sum(
        row["seconds"] for row in measured["roles"].values()
    )
    return {
        "schema": "cifar_representative_matched_heads_request_v1",
        "source": probe_result["source_hashes"],
        "probe_result_sha256": probe_digest,
        "probe_request_sha256": probe_result["request_sha256"],
        "probe_feature_seconds": measured["elapsed_seconds"],
        "projected_feature_seconds_per_seed": projected_feature_seconds,
        "projection_scope": "four_times_observed_role_counts_at_same_image_size_not_a_guarantee",
        "base_config": asdict(base),
        "candidate_grid": {
            name: [asdict(candidate) for candidate in options]
            for name, options in candidate_grid.items()
        },
        "selection_seeds": [SELECTION_SEED],
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "candidates_per_head": 2,
        "selection_objective": "mean_outer_validation_accuracy_first_candidate_on_exact_tie",
        "selection_limit_seconds": 180,
        "scopes": {
            "fixed_data": {"epochs": 1, "limit_seconds": 240},
            "wall_time": {
                "per_head_seconds": 5.0,
                "epoch_cap": 1000,
                "limit_seconds": 240,
            },
            "capacity_memory": {
                "fixed_hidden_width": 16,
                "fresh_child_per_head": True,
                "epochs": 1,
                "limit_seconds": 600,
                "rss_scope": "process_train_observed_absolute_peak",
                "cuda_scope": "child_torch_allocator_peak",
            },
        },
        "confirmation_total_limit_seconds": 1080,
        "quiet_window": {
            "readings": 3,
            "spacing_seconds": 5,
            "max_utilization_percent": 10,
            "min_free_mib": 5120,
        },
        "final_test_policy": "no_iteration_until_all_equal_trials_and_choices_are_frozen",
        "outcomes": [
            "all_selection_attempts_and_trials",
            "per_seed_final_accuracy_and_dispersion",
            "fixed_data_work_and_relaxation",
            "deadline_work_stop_and_overshoot",
            "isolated_rss_and_cuda_allocator_scopes",
            "role_feature_backbone_and_initial_hashes",
            "every_negative_or_failed_result",
        ],
        "interpretation_limit": (
            "CIFAR-10 subset with resized 224-pixel inputs and frozen ImageNet backbone; "
            "not a full-data or end-to-end backbone ranking"
        ),
    }


def prepare_request(path: Path = REQUEST_PATH) -> dict[str, Any]:
    probe_result, probe_digest = _read_verified_probe()
    request = _study_request(probe_result, probe_digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
    return request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=REQUEST_PATH)
    args = parser.parse_args()
    request = prepare_request(args.request)
    print(
        json.dumps(
            {
                "request": str(args.request),
                "selection_seeds": request["selection_seeds"],
                "confirmation_seeds": request["confirmation_seeds"],
                "projected_feature_seconds_per_seed": request["projected_feature_seconds_per_seed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
