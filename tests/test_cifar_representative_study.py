"""The larger matched study freezes seeds, trials, scopes, and limits first."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import prepare_cifar_representative_study as study


def test_should_freeze_equal_trial_study_from_development_cost_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = {
        "source_hashes": {"archive_sha256": "archive", "weight_sha256": "weight"},
        "request_sha256": "probe-request",
        "measurement": {
            "elapsed_seconds": 9.5,
            "setup_seconds": 1.4,
            "roles": {
                "train": {"seconds": 6.5},
                "guard": {"seconds": 0.8},
                "validation": {"seconds": 0.8},
            },
        },
    }
    monkeypatch.setattr(study, "_read_verified_probe", lambda: (probe, "probe-result"))
    path = tmp_path / "request.json"

    request = study.prepare_request(path)

    assert json.loads(path.read_text(encoding="utf-8")) == request
    assert request["base_config"]["image_size"] == 224
    assert request["base_config"]["dataset_train_subset_size"] == 16384
    assert request["base_config"]["dataset_guard_subset_size"] == 2048
    assert request["base_config"]["dataset_validation_subset_size"] == 2048
    assert request["selection_seeds"] == [179]
    assert request["confirmation_seeds"] == [181, 191, 193]
    assert set(request["candidate_grid"]) == {
        "backprop_mlp",
        "predictive_coding",
        "circadian_predictive_coding",
    }
    assert all(len(options) == 2 for options in request["candidate_grid"].values())
    assert request["projected_feature_seconds_per_seed"] == pytest.approx(33.8)
    assert request["scopes"]["wall_time"]["per_head_seconds"] == 5.0
    assert request["final_test_policy"].startswith("no_iteration_until")
    with pytest.raises(FileExistsError):
        study.prepare_request(path)
