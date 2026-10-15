"""Fixed protocol, complete seed accounting and failure retention; no steps."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from src.adapters import linear_scaling_pilot as pilot
from src.app.linear_scaling_summary import KEYS, summarize_scaling
from src.shared.process_memory import ProcessRssSampler


def _rows():
    return tuple(
        dict(
            seed=s,
            width=w,
            hidden_layers=d,
            scheme=m,
            status="complete",
            committed_updates=6,
            checkpoints=[{"updates": i} for i in range(7)],
            heldout_half_mse=[float(s) / (i + 1) for i in range(7)],
        )
        for s, w, d, m in KEYS
    )


def test_should_build_repeatable_disjoint_real_regression_batches():
    train, final = pilot.build_batches()
    again = pilot.build_batches()
    assert (train.role, final.role) == ("train", "final_test")
    assert train.features.shape == (16, 2) and final.features.shape == (64, 2)
    assert not set(train.row_ids) & set(final.row_ids)
    for first, second in zip((train, final), again):
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(
            first.targets, first.features[:, [0]] - 0.5 * first.features[:, [1]]
        )
        assert not np.shares_memory(first.features, second.features)
    for left in (train.features, train.targets):
        for right in (final.features, final.targets):
            assert not np.shares_memory(left, right)


def test_should_retain_all_ordered_seeds_in_fixed_summaries():
    summaries = summarize_scaling(_rows())
    assert len(summaries) == 27
    for row in summaries:
        assert [x["seed"] for x in row["initial"]["observations"]] == [23, 41, 59]
        assert row["initial"]["mean"] == 41.0
        assert row["final"]["mean"] == pytest.approx(41 / 7)
        assert row["final"]["marginal_interval"] is not None


def test_should_leave_incomplete_vector_mean_and_interval_unavailable():
    rows = list(_rows())
    rows[0].update(status="failed", reason="numeric refusal", heldout_half_mse=None)
    summary = summarize_scaling(tuple(rows))[0]["final"]
    assert summary["mean"] is None and summary["marginal_interval"] is None
    assert summary["observed_seed_count"] == 2
    assert summary["observations"][0] == {"seed": 23, "value": None, "reason": "numeric refusal"}


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "boolean", "nan", "negative", "checkpoint", "reason", "commits"],
)
def test_should_refuse_invalid_or_lost_run_records(change):
    rows = list(deepcopy(_rows()))
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[1] = rows[0]
    elif change == "boolean":
        rows[0]["seed"] = True
    elif change == "nan":
        rows[0]["heldout_half_mse"][3] = float("nan")
    elif change == "negative":
        rows[0]["heldout_half_mse"][6] = -1.0
    elif change == "checkpoint":
        rows[0]["checkpoints"][2]["updates"] = True
    elif change == "reason":
        rows[0].update(status="failed")
    else:
        rows[0]["committed_updates"] = True
    with pytest.raises(ValueError):
        summarize_scaling(tuple(rows))


def test_should_report_zero_variance_without_interval():
    rows = _rows()
    for row in rows:
        row["heldout_half_mse"] = [1.0] * 7
    result = summarize_scaling(rows)[0]["final"]
    assert result["mean"] == 1.0 and result["marginal_interval"] is None
    assert result["interval_status"] == "zero_observed_variance"


def test_should_keep_all_failures_without_training(monkeypatch):
    monkeypatch.setattr(
        pilot, "ProcessRssSampler", lambda: ProcessRssSampler(read_rss_bytes=lambda: 1000)
    )
    calls = []

    def refuse(row, started, sampler):
        calls.append((row["seed"], row["width"], row["hidden_layers"], row["scheme"]))
        row.update(status="failed", reason="test refusal")

    monkeypatch.setattr(pilot, "_run_one", refuse)
    report = pilot.run_pilot()
    assert tuple(calls) == KEYS and len(report["runs"]) == 81
    assert report["status"] == "incomplete" and report["within_resources"]
    assert len(report["summaries"]) == 27
    assert all(s["final"]["mean"] is None for s in report["summaries"])


def test_should_preserve_unreached_rows_when_memory_reader_fails(monkeypatch):
    def fail():
        raise OSError("test memory unavailable")

    monkeypatch.setattr(pilot, "ProcessRssSampler", lambda: ProcessRssSampler(read_rss_bytes=fail))
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and len(report["runs"]) == 81
    assert all(r["status"] == "not_run" for r in report["runs"])
    assert report["global_errors"] and report["observed_process_rss"] is None


def test_should_record_exact_factory_state_and_controls_without_kernel(monkeypatch):
    def fake(initial, train, heldout, **kwargs):
        assert kwargs["updates"] == 6 and kwargs["learning_rate"] == 0.01
        assert asdict(initial.config) == dict(
            scheme="mean_field", input_dim=2, width=8, hidden_layers=2, gamma0=1.0
        )
        return dict(
            status="failed",
            reason="injected",
            attempted_updates=0,
            committed_updates=0,
            checkpoints=[],
            heldout_half_mse=None,
        )

    monkeypatch.setattr(pilot, "run_linear_scaling_case", fake)
    row: dict[str, Any] = dict(
        seed=23,
        width=8,
        hidden_layers=2,
        scheme="mean_field",
        status="not_run",
        attempted_updates=0,
        committed_updates=0,
        checkpoints=[],
        heldout_half_mse=None,
    )
    with ProcessRssSampler(read_rss_bytes=lambda: 1000) as sampler:
        pilot._run_one(row, pilot.monotonic(), sampler)
    state = row["initial_state"]
    assert state["forward_scales"] == [1 / np.sqrt(2), 1 / np.sqrt(8), 1 / 8]
    for raw, effective, factor in zip(
        state["raw_weights"], state["effective_weights"], state["forward_scales"]
    ):
        np.testing.assert_array_equal(np.array(raw) * factor, effective)
    assert row["status"] == "failed" and row["work"]["attempted_step_calls"] == 0


def test_should_describe_cli_without_any_training():
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_linear_scaling_pilot", "--describe"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    protocol = json.loads(completed.stdout)
    assert protocol == pilot.describe_protocol()
    assert protocol["resources"]["training_updates"] == 486
