"""Frozen protocol, complete seed vectors and failure orchestration; no pilot."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters import nonlinear_scaling_pilot as pilot
from src.app.nonlinear_scaling_summary import (
    CHECKPOINTS,
    METHODS,
    SEEDS,
    planned_keys,
    summarize_nonlinear_scaling,
)
from src.core.local_pilot_budget import PilotRequest, validate_local_pilot_request
from src.shared.process_memory import ProcessRssSegment


def complete_rows() -> list[dict]:
    rows = []
    for width, depth, scheme, seed, method in planned_keys():
        value = 0.5 + SEEDS.index(seed) * 0.1 + (0.2 if method == "bp" else 0.0)
        points = [
            {"updates": update, "scores": {"stationary": {"bce": value, "accuracy": 0.5}}}
            for update in CHECKPOINTS
        ]
        state = SimpleNamespace(raw_parameters={"fixed": seed})
        rows.append(
            dict(
                width=width,
                depth=depth,
                scheme=scheme,
                seed=seed,
                method=method,
                status="complete",
                reason=None,
                committed_updates=3,
                case_result={
                    "stream_result": {"checkpoints": points},
                    "initial_state": state,
                    "final_state": state,
                    "diagnostics": [],
                },
            )
        )
    return rows


def test_should_freeze_small_protocol_and_independent_matched_data() -> None:
    protocol = pilot.describe_protocol()
    assert len(planned_keys()) == len(set(planned_keys())) == protocol["planned_runs"] == 96
    assert protocol["planned_training_updates"] == 288
    assert (
        validate_local_pilot_request(PilotRequest(pilot.RESOURCES)).resources.training_updates
        == 288
    )
    training, final = pilot.build_data()
    repeated, again = pilot.build_data()
    assert len(training) == 3 and len(final) == 1
    all_ids = [row for batch in (*training, *final) for row in batch.row_ids]
    assert len(all_ids) == len(set(all_ids)) == 88
    for left, right in zip((*training, *final), (*repeated, *again)):
        np.testing.assert_array_equal(left.features, right.features)
        np.testing.assert_array_equal(
            left.targets, left.features[:, [0]] + 0.25 * left.features[:, [1]] >= 0
        )
        assert not np.shares_memory(left.features, right.features)
    assert all(batch.role == "train" for batch in training) and final[0].role == "final_test"


@pytest.mark.parametrize("width,depth,scheme", [(2, 1, "fan_in"), (8, 3, "mean_field")])
def test_should_share_owned_exact_initial_tensors_factors_and_representation(
    width, depth, scheme
) -> None:
    learners = [pilot.build_learner(width, depth, scheme, 23, method) for method in METHODS]
    states = [learner.snapshot_state() for learner in learners]
    expected = asdict(states[0].raw_parameters)
    features = pilot.build_data()[0][0].features
    for state, learner in zip(states, learners):
        np.testing.assert_equal(asdict(state.raw_parameters), expected)
        assert state.config.forward_scales == states[0].config.forward_scales
        assert state.config.bias_scale == states[0].config.bias_scale
        np.testing.assert_array_equal(learner.predict(features), learners[0].predict(features))
    states[0].raw_parameters.output_weight[:] = 99
    for learner in learners:
        np.testing.assert_equal(asdict(learner.snapshot_state().raw_parameters), expected)
    with pytest.raises(ValueError):
        pilot.build_learner(width, depth, scheme, 1, "best_seed")


def test_should_keep_exact_seed_vectors_paired_direction_and_intervals() -> None:
    summaries = summarize_nonlinear_scaling(tuple(complete_rows()))
    assert len(summaries) == 448
    pc = next(s for s in summaries if s["comparison"] == "pc" and s["metric"] == "bce")
    assert pc["summary"]["mean"] == pytest.approx(0.6)
    assert [o["seed"] for o in pc["summary"]["observations"]] == [23, 41, 59]
    interval = pc["summary"]["marginal_interval"]
    assert interval["lower"] == pytest.approx(0.6 - 4.303 * 0.1 / np.sqrt(3))
    assert interval["upper"] == pytest.approx(0.6 + 4.303 * 0.1 / np.sqrt(3))
    paired = next(s for s in summaries if s["comparison"] == "pc_minus_bp" and s["metric"] == "bce")
    assert paired["difference_direction"] == "left_minus_right"
    assert paired["summary"]["mean"] == pytest.approx(-0.2)
    neutral = next(s for s in summaries if s["comparison"] == "neutral_cpc_minus_pc")
    assert neutral["summary"]["interval_status"] == "zero_observed_variance"
    assert neutral["summary"]["marginal_interval"] is None


def test_should_keep_failed_seed_and_suppress_complete_group_and_pair_estimates() -> None:
    rows = complete_rows()
    rows[2] = {**rows[2], "status": "failed", "reason": "injected", "case_result": None}
    summaries = summarize_nonlinear_scaling(tuple(rows))
    affected = [
        s
        for s in summaries
        if (s["width"], s["depth"], s["scheme"]) == (2, 1, "fan_in")
        and s["comparison"]
        in ("neutral_cpc", "neutral_cpc_minus_pc", "active_cpc_minus_neutral_cpc")
    ]
    assert len(affected) == 24
    assert all(
        s["summary"]["mean"] is None and s["summary"]["marginal_interval"] is None for s in affected
    )
    assert all(s["summary"]["observed_seed_count"] == 2 for s in affected)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "boolseed",
        "unknown",
        "reason",
        "score",
        "point",
        "partial",
        "badstream",
    ],
)
def test_should_refuse_incomplete_or_malformed_run_vectors(mutation) -> None:
    rows = complete_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif mutation == "boolseed":
        rows[0]["seed"] = True
    elif mutation == "unknown":
        rows[0]["method"] = "best"
    elif mutation == "reason":
        rows[0].update(status="failed", case_result=None)
    elif mutation == "score":
        rows[0]["case_result"]["stream_result"]["checkpoints"][0]["scores"]["stationary"]["bce"] = (
            float("nan")
        )
    elif mutation == "point":
        rows[0]["case_result"]["stream_result"]["checkpoints"][0] = None
    elif mutation == "partial":
        rows[0].update(status="failed", reason="injected")
    else:
        rows[0]["case_result"]["stream_result"] = None
    with pytest.raises(ValueError):
        summarize_nonlinear_scaling(tuple(rows))


class FakeSampler:
    def sample(self):
        return 2048

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return None

    def snapshot(self):
        return ProcessRssSegment(1, 1024, 2048, 1, 0.05)


def test_should_preserve_all_planned_rows_and_null_summaries_on_one_fake_case_failure(
    monkeypatch,
) -> None:
    rows = complete_rows()
    rows[0].update(
        status="failed", reason="injected factory", case_result=None, committed_updates=0
    )
    indexed = {key: row for key, row in zip(planned_keys(), rows)}
    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "_run_one", lambda key, started, sampler: deepcopy(indexed[key]))
    result = pilot.run_pilot()
    assert len(result["runs"]) == 96 and result["status"] == "incomplete"
    assert result["runs"][0]["reason"] == "injected factory"
    assert len(result["summaries"]) == 448


@pytest.mark.parametrize("phase", ["construction", "enter"])
def test_should_keep_unreached_rows_when_sampler_fails(monkeypatch, phase) -> None:
    class BrokenSampler(FakeSampler):
        def __init__(self):
            if phase == "construction":
                raise RuntimeError("injected sampler construction")

        def __enter__(self):
            raise RuntimeError("injected sampler enter")

    monkeypatch.setattr(pilot, "ProcessRssSampler", BrokenSampler)
    result = pilot.run_pilot()
    assert result["status"] == "incomplete" and len(result["runs"]) == 96
    assert all(
        row["status"] == "not_run" and row["committed_updates"] == 0 for row in result["runs"]
    )
    assert all(s["summary"]["mean"] is None for s in result["summaries"])
    assert any("injected sampler" in reason for reason in result["global_errors"])


def test_should_describe_cli_without_running_any_science() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_nonlinear_scaling_pilot", "--describe"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert not result.stderr
    protocol = json.loads(result.stdout)
    assert protocol["protocol_id"] == "nonlinear_raw_scaling_fixed_pilot_v1"
    assert protocol["planned_runs"] == 96 and protocol["planned_training_updates"] == 288


def test_should_keep_aggregation_inside_the_owned_memory_interval(monkeypatch) -> None:
    class TrackingSampler(FakeSampler):
        active = False

        def __enter__(self):
            self.active = True
            return self

        def __exit__(self, *arguments):
            self.active = False

        def sample(self):
            assert self.active
            return super().sample()

    sampler = TrackingSampler()
    indexed = dict(zip(planned_keys(), complete_rows()))
    summarize = pilot.summarize_nonlinear_scaling

    def observed_summary(rows):
        assert sampler.active
        return summarize(rows)

    monkeypatch.setattr(pilot, "ProcessRssSampler", lambda: sampler)
    monkeypatch.setattr(pilot, "_run_one", lambda key, started, memory: deepcopy(indexed[key]))
    monkeypatch.setattr(pilot, "summarize_nonlinear_scaling", observed_summary)
    assert pilot.run_pilot()["status"] == "complete"
    assert not sampler.active


def test_should_keep_cli_encoding_and_output_inside_the_shared_memory_interval(
    monkeypatch, capsys
) -> None:
    from scripts import run_nonlinear_scaling_pilot as cli

    class TrackingSampler(FakeSampler):
        active = False

        def __enter__(self):
            self.active = True
            return self

        def __exit__(self, *arguments):
            assert self.active
            self.active = False

        def sample(self):
            assert self.active
            return super().sample()

    sampler = TrackingSampler()
    encode = cli._encode

    def observed_encode(report):
        assert sampler.active
        return encode(report)

    def fake_pilot(**arguments):
        assert arguments["sampler"] is sampler and sampler.active
        return {"status": "complete"}

    monkeypatch.setattr(cli, "ProcessRssSampler", lambda: sampler)
    monkeypatch.setattr(cli, "run_pilot", fake_pilot)
    monkeypatch.setattr(cli, "_encode", observed_encode)
    monkeypatch.setattr(cli.sys, "argv", ["run_nonlinear_scaling_pilot"])
    cli.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "complete"
    assert json.loads(captured.err)["within_resources"] is True
    assert not sampler.active


@pytest.mark.parametrize("phase", ["construction", "enter"])
def test_should_retain_all_unstarted_rows_after_cli_sampler_failure(monkeypatch, capsys, phase):
    from scripts import run_nonlinear_scaling_pilot as cli

    calls = []

    class BrokenSampler(FakeSampler):
        def __init__(self):
            calls.append("construction")
            if phase == "construction":
                raise RuntimeError("injected startup")

        def __enter__(self):
            raise RuntimeError("injected startup")

    def forbidden_run(**arguments):
        pytest.fail("startup failure must not execute or retry pilot")

    monkeypatch.setattr(cli, "ProcessRssSampler", BrokenSampler)
    monkeypatch.setattr(cli, "run_pilot", forbidden_run)
    monkeypatch.setattr(cli.sys, "argv", ["run_nonlinear_scaling_pilot"])
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "incomplete" and len(report["runs"]) == 96
    assert all(row["status"] == "not_run" for row in report["runs"])
    assert json.loads(captured.err)["within_resources"] is False
    assert calls == ["construction"]


def test_should_charge_stdout_newline_to_the_output_cap(monkeypatch):
    from scripts import run_nonlinear_scaling_pilot as cli

    monkeypatch.setattr(cli, "RESOURCES", replace(pilot.RESOURCES, storage_bytes=2))
    with pytest.raises(RuntimeError, match="output allowance"):
        cli._encode({})


def test_should_charge_windows_line_expansion_before_output(monkeypatch):
    from scripts import run_nonlinear_scaling_pilot as cli

    monkeypatch.setattr(cli.os, "linesep", "\r\n")
    assert cli._output_bytes('{\n  "value": 1\n}') == len(b'{\r\n  "value": 1\r\n}\r\n')
    monkeypatch.setattr(cli, "RESOURCES", replace(pilot.RESOURCES, storage_bytes=18))
    with pytest.raises(RuntimeError, match="output allowance"):
        cli._encode({"value": 1})
