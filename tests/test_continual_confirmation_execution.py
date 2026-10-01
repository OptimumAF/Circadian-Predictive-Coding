"""Frozen request metadata is validated without a source, model or score."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_confirmation_execution import execution_request, verify_execution_request
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


@pytest.fixture
def bindings() -> dict[str, Any]:
    return {
        "manifest": fixed_confirmation_manifest(),
        "scope_sha256": "a" * 64,
        "source_sha256": {"src/app/example.py": "b" * 64},
        "adapter_sha256": "c" * 64,
        "command": ["python", "-m", "scripts.run_p67_confirmation_training", "--worker"],
        "environment": {
            "python_version": "3.14.7",
            "numpy_version": "2.4.6",
            "platform": "fixture",
            "processor": "fixture",
        },
    }


def _forbid(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("execution metadata opened a source/model/train/score")


def test_should_bind_whole_frozen_scope_without_sources_or_models(
    bindings: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(model, "__init__", _forbid)
        monkeypatch.setattr(model, "train_epoch", _forbid)
        monkeypatch.setattr(model, "predict_proba", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    request = execution_request(started_utc="2026-09-30T12:00:00+00:00", **bindings)
    verify_execution_request(request, **bindings)
    assert request["summary"]["cell_count"] == 560
    assert request["summary"]["family_seed_instances"] == 60
    assert request["limits"] == {
        "max_optimizer_updates": 16000,
        "wall_limit_seconds": 600,
        "max_process_rss_bytes": 512 * 1024 * 1024,
        "rss_interval_seconds": 0.005,
    }
    assert request["outer_selection_scored"] is False and request["final_released"] is False


@pytest.mark.parametrize(
    "corruption",
    [
        "extra",
        "source",
        "adapter",
        "scope",
        "command",
        "environment",
        "manifest",
        "summary",
        "limit",
        "bool_limit",
        "float_summary",
        "final",
        "outer",
        "protocol",
        "digest",
    ],
)
def test_should_reject_any_changed_request_field(bindings: dict[str, Any], corruption: str) -> None:
    request = deepcopy(execution_request(started_utc="2026-09-30T12:00:00+00:00", **bindings))
    if corruption == "extra":
        request["extra"] = 1
    elif corruption == "source":
        request["source_sha256"]["src/app/example.py"] = "d" * 64
    elif corruption in {"adapter", "scope", "digest"}:
        field = {
            "adapter": "adapter_sha256",
            "scope": "scope_record_sha256",
            "digest": "manifest_sha256",
        }[corruption]
        request[field] = "d" * 64
    elif corruption == "command":
        request["command"].append("--seed=101")
    elif corruption == "environment":
        request["environment"]["numpy_version"] = "changed"
    elif corruption == "manifest":
        request["manifest"]["families"].pop()
    elif corruption in {"summary", "float_summary"}:
        request["summary"]["cell_count"] = 559 if corruption == "summary" else 560.0
    elif corruption in {"limit", "bool_limit"}:
        request["limits"]["max_optimizer_updates"] = 16001 if corruption == "limit" else True
    elif corruption in {"final", "outer"}:
        request["final_released" if corruption == "final" else "outer_selection_scored"] = True
    else:
        request["protocol_id"] = "changed"
    with pytest.raises(ValueError, match="execution request"):
        verify_execution_request(request, **bindings)


@pytest.mark.parametrize(
    "timestamp", [None, 123, "garbage", "2026-09-30T12:00:00", "2026-09-30T12:00:00+02:00"]
)
def test_should_reject_missing_malformed_or_nonutc_time(
    bindings: dict[str, Any], timestamp: Any
) -> None:
    with pytest.raises(ValueError, match="time"):
        execution_request(started_utc=timestamp, **bindings)


def test_should_reject_partial_manifest_before_sources(bindings: dict[str, Any]) -> None:
    bindings["manifest"] = replace(bindings["manifest"], families=bindings["manifest"].families[:1])
    with pytest.raises(ValueError, match="frozen complete scope"):
        execution_request(started_utc="2026-09-30T12:00:00+00:00", **bindings)


def test_should_reject_numerically_equal_but_differently_typed_manifest(
    bindings: dict[str, Any],
) -> None:
    bindings["manifest"] = replace(bindings["manifest"], wall_limit_seconds=600.0)
    with pytest.raises(ValueError, match="execution manifest.*type"):
        execution_request(started_utc="2026-09-30T12:00:00+00:00", **bindings)
