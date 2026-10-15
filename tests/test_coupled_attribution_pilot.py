"""Data/factory/aggregation and CLI controls without executing the full pilot."""

from copy import deepcopy
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import src.adapters.coupled_attribution_pilot as pilot
from src.app.coupled_attribution_summary import (
    CHECKPOINTS,
    CONTEXTS,
    METHODS,
    PLANNED_SEEDS,
    STREAMS,
    summarize_attribution,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.shared.process_memory import ProcessRssSegment


def complete_rows():
    rows = []
    for seed, stream, method in product(PLANNED_SEEDS, STREAMS, METHODS):
        value = 0.5 + PLANNED_SEEDS.index(seed) * 0.1 + (0.2 if method == "bp" else 0)
        points = [
            {"updates": updates, "scores": {c: {"bce": value, "accuracy": 0.5} for c in CONTEXTS}}
            for updates in CHECKPOINTS
        ]
        rows.append(
            {
                "seed": seed,
                "stream": stream,
                "method": method,
                "status": "complete",
                "stream_result": {"checkpoints": points},
            }
        )
    return rows


def test_should_generate_fixed_matched_stream_inputs_and_disjoint_heldout():
    stationary, heldout = pilot.build_stream("stationary")
    recurring, other_heldout = pilot.build_stream("recurring")
    again, _ = pilot.build_stream("stationary")
    assert len(stationary) == len(recurring) == 18
    assert [b.context for b in recurring] == ["A"] * 6 + ["B"] * 6 + ["A"] * 6
    assert [b.context for b in stationary] == ["A"] * 18
    for left, right, repeated in zip(stationary, recurring, again):
        np.testing.assert_array_equal(left.features, right.features)
        np.testing.assert_array_equal(left.features, repeated.features)
        for batch in (left, right):
            assert batch.role == "train" and batch.features.shape == (8, 2)
            np.testing.assert_array_equal(
                batch.targets, batch.features[:, [CONTEXTS.index(batch.context)]] >= 0
            )
    train_ids = {i for b in stationary for i in b.row_ids}
    test_ids = {i for b in heldout for i in b.row_ids}
    assert len(train_ids) == 144 and len(test_ids) == 128 and not train_ids & test_ids
    for a, b in zip(heldout, other_heldout):
        assert a.role == "final_test" and a.features.shape == (64, 2)
        np.testing.assert_array_equal(a.features, b.features)
        for train in stationary:
            assert not np.shares_memory(a.features, train.features)
            assert not np.shares_memory(a.targets, train.targets)


def test_should_construct_exact_matched_owned_initial_tensors_without_updates():
    initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
    learners = [pilot.build_learner(method, initial) for method in METHODS]
    reference = asdict(initial)
    for learner in learners:
        np.testing.assert_equal(asdict(learner.snapshot_state().parameters), reference)
    initial.output_weight.fill(999)
    for learner in learners:
        np.testing.assert_equal(asdict(learner.snapshot_state().parameters), reference)
    with pytest.raises(ValueError):
        pilot.build_stream("selected_best")
    with pytest.raises(ValueError):
        pilot.build_learner("selected_best", initial)


def test_should_summarize_all_fixed_seeds_and_keep_paired_direction():
    rows = complete_rows()
    summaries = summarize_attribution(tuple(rows))
    assert len(summaries) == 224
    entry = next(s for s in summaries if s["comparison"] == "pc" and s["metric"] == "bce")
    summary = entry["summary"]
    assert [o["seed"] for o in summary["observations"]] == [23, 41, 59]
    assert summary["mean"] == pytest.approx(0.6)
    assert summary["marginal_interval"]["lower"] == pytest.approx(0.6 - 4.303 * 0.1 / np.sqrt(3))
    assert summary["marginal_interval"]["upper"] == pytest.approx(0.6 + 4.303 * 0.1 / np.sqrt(3))
    paired = next(s for s in summaries if s["comparison"] == "pc_minus_bp" and s["metric"] == "bce")
    assert paired["difference_direction"] == "left_minus_right"
    assert paired["summary"]["mean"] == pytest.approx(-0.2)
    neutral = next(s for s in summaries if s["comparison"] == "neutral_cpc_minus_pc")
    assert neutral["summary"]["interval_status"] == "zero_observed_variance"
    assert neutral["summary"]["marginal_interval"] is None


def test_should_retain_failed_seed_and_suppress_its_complete_interval():
    rows = complete_rows()
    failed = next(
        r
        for r in rows
        if r["seed"] == 41 and r["stream"] == "recurring" and r["method"] == "gated_cpc"
    )
    failed.update(status="failed", reason="fixed failure control")
    summaries = summarize_attribution(tuple(rows))
    entry = next(
        s
        for s in summaries
        if s["stream"] == "recurring" and s["comparison"] == "gated_cpc_minus_neutral_cpc"
    )
    summary = entry["summary"]
    assert summary["observed_seed_count"] == 2 and summary["planned_seed_count"] == 3
    assert summary["mean"] is None and summary["marginal_interval"] is None
    assert summary["observations"][1] == {
        "seed": 41,
        "value": None,
        "reason": "fixed failure control",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "lost",
        "duplicate",
        "boolseed",
        "badstream",
        "boolpoint",
        "missingpoint",
        "nan",
        "missingcontext",
    ],
)
def test_should_refuse_incomplete_or_malformed_complete_records(mutation):
    rows = complete_rows()
    if mutation == "lost":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif mutation == "boolseed":
        rows[0]["seed"] = True
    elif mutation == "badstream":
        rows[0]["stream"] = []
    elif mutation == "boolpoint":
        rows[0]["stream_result"]["checkpoints"][0]["updates"] = False
    elif mutation == "missingpoint":
        rows[0]["stream_result"]["checkpoints"].pop()
    elif mutation == "nan":
        rows[0]["stream_result"]["checkpoints"][0]["scores"]["A"]["bce"] = float("nan")
    else:
        del rows[0]["stream_result"]["checkpoints"][0]["scores"]["B"]
    with pytest.raises(ValueError):
        summarize_attribution(tuple(rows))


class FakeSampler:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def snapshot(self):
        return ProcessRssSegment(1, 100, 200, 2, 0.005)


def test_should_keep_every_run_when_orchestration_has_failures(monkeypatch):
    calls = []

    def fail(seed, stream, method, *_):
        calls.append((seed, stream, method))
        return {
            "seed": seed,
            "stream": stream,
            "method": method,
            "status": "failed",
            "reason": "injected",
            "committed_updates": 0,
        }

    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "_run_one", fail)
    report = pilot.run_pilot()
    assert calls == list(product(PLANNED_SEEDS, STREAMS, METHODS))
    assert report["status"] == "incomplete" and len(report["runs"]) == 24
    assert all(s["summary"]["mean"] is None for s in report["summaries"])


def test_should_keep_unreached_rows_if_sampler_fails(monkeypatch):
    class MissingSampler(FakeSampler):
        def __enter__(self):
            raise OSError("RSS unavailable control")

        def snapshot(self):
            raise RuntimeError("no samples")

    monkeypatch.setattr(pilot, "ProcessRssSampler", MissingSampler)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and not report["within_resources"]
    assert len(report["runs"]) == 24 and all(r["status"] == "not_run" for r in report["runs"])
    assert len(report["global_errors"]) == 2


def test_should_describe_fixed_protocol_cli_without_training():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_attribution", "--describe"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    protocol = json.loads(result.stdout)
    assert result.stderr == "" and protocol == pilot.describe_protocol()
    assert protocol["resources"]["training_updates"] == 432
    assert protocol["resources"]["training_updates"] < 512
    assert protocol["seeds"] == [23, 41, 59]
    assert protocol["checkpoint_updates"] == [0, 6, 12, 18]
