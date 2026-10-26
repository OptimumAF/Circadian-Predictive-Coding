"""Aggregated198 orchestration against scalar stage doubles; zero learner arithmetic.

Fifteen route cases and thirteen pre-IO flag refusals. Every opaque return is retained;
negative endpoint scores stay valid. The shared helper preserves legacy defaults.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts import push_cube_continuation_study as study
from src.adapters import numpy_continuous_bc as bp
from src.core import continuous_bc as core
from test_push_cube_continuation_study import run_routing_case, source as source, forbidden


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts: dict[str, int] = defaultdict(int)
    with pytest.MonkeyPatch.context() as patch:
        for name in ("__init__", "predict", "train_batch", "train_head_gauss_newton_batch", "fork"):
            patch.setattr(bp.NumpyContinuousBCLearner, name, forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        patch.setattr(core, "_forward", forbidden)
        patch.setattr(core, "continuous_bc_loss_and_gradients", forbidden)
        yield counts
    if os.environ.get("ACS_STUDY_FIXTURE_LEDGER"):
        with Path(os.environ["ACS_STUDY_FIXTURE_LEDGER"]).open("x", encoding="utf8") as stream:
            json.dump(dict(counts), stream, sort_keys=True)
            stream.write("\n")
    assert counts["routing_source_calls"] == counts["routing_acquisition_calls"] == 15
    assert counts["routing_longer_calls"] == counts["routing_correction_fit_calls"] == 15
    assert counts["routing_continuation_calls"] == 58
    assert counts["routing_fit_calls"] == 51
    assert counts["routing_push_calls"] == 5
    assert counts["routing_pick_calls"] == 6


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "balanced_readout",
        "aggregated120",
        "incomplete",
        "receipt",
        "counter",
        "scope",
        "prior",
        "packet",
        "aggregated128",
        "paid_readout",
    ],
)
def test_should_keep_all198_and_stop_before_native_on_paid_or_reconstruction_refusal(
    tmp_path, monkeypatch, source, ledger, fault
):
    result = run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        longer_fit=True,
        native_endpoint=True,
        correction_fit=True,
        aggregated_correction_fit=True,
        aggregated_fault=fault,
    )
    if fault is None:
        assert result["native_push"]["final_success"] is False
        assert all(item["final_success"] is False for item in result["pick"])
        assert result["complete"] is True
        assert result["aggregated_continuation"]["native_updates"][0]["objective_row_weights"][
            :51
        ] == [49.0] + [1.0] * 49 + [49.0]
    elif fault in ("prior", "packet"):
        assert result["refused_return"]["stage"] == "aggregated_continuation"
        assert result["refused_return"]["report"]["rows"] == 198
    elif fault in ("incomplete", "receipt", "counter", "scope", "aggregated128", "paid_readout"):
        assert (
            "aggregated_continuation" in result and result["aggregated_continuation"]["rows"] == 198
        )
    elif fault == "balanced_readout":
        assert "balanced_readout120" in result and "aggregated_readout120" not in result
    else:
        assert "aggregated_readout120" in result and "aggregated_continuation" not in result


@pytest.mark.parametrize("fault", ["push_incomplete", "pick_binding", "pick_reset", "prior_native"])
def test_should_keep_paid_native_prefix_and_refuse_next_scene_on128_binding_failure(
    tmp_path, monkeypatch, source, ledger, fault
):
    result = run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        longer_fit=True,
        native_endpoint=True,
        correction_fit=True,
        aggregated_correction_fit=True,
        endpoint_fault=fault,
    )
    assert "native_push" in result and result["complete"] is False
    if fault in ("push_incomplete", "prior_native"):
        assert result["pick"] == []
    else:
        assert len(result["pick"]) == 1
    assert result["aggregated_continuation"]["final_counts"] == dict(
        updates=128, training_rows=12600
    )


@pytest.mark.parametrize(
    "fault",
    [
        "nonbool",
        "profile",
        "longer",
        "native",
        "correction",
        "training_diagnostics",
        "diagnostics_only",
        "reserved_probes",
        "push_corrections",
        "correction_readouts",
        "balanced_correction_fit",
        "zero_velocity_fit",
        "balanced_push_corrections",
    ],
)
def test_should_refuse_wrong_aggregated_mode_before_io(tmp_path, monkeypatch, fault):
    options: dict[str, Any] = dict(
        profile="bp_mixed_retention",
        longer_fit=True,
        native_endpoint=True,
        correction_fit=True,
        aggregated_correction_fit=True,
    )
    if fault == "nonbool":
        options["aggregated_correction_fit"] = 1
    elif fault == "profile":
        options["profile"] = "bp_mixed"
    elif fault in ("longer", "native", "correction"):
        options[
            {"longer": "longer_fit", "native": "native_endpoint", "correction": "correction_fit"}[
                fault
            ]
        ] = False
    else:
        options[fault] = True
    monkeypatch.setattr(study, "read_reused_control", forbidden)
    monkeypatch.setattr(study, "read_push_training", forbidden)
    with pytest.raises(ValueError):
        study.run_study(tmp_path / "run.json", **options)
    assert not (tmp_path / "run.json").exists()
