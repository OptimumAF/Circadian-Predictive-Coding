"""Composition correctness; native parity is not a learning-benefit assertion."""

from copy import deepcopy
import json
import pickle
import sys
from time import monotonic

import numpy as np
import pytest

import src.adapters.learnability_pilot as pilot
import scripts.run_learnability_pilot as cli
from src.adapters.coupled_learners import CoupledLearner
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)


def forbidden(*args, **kwargs):
    pytest.fail("an unselected operational path was reached")


def test_should_describe_frozen_work_without_building_data_models_or_running_probes(monkeypatch):
    monkeypatch.setattr(pilot, "build_reserved_data", forbidden)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    report = pilot.describe_protocol()
    assert report["seeds"] == [23]
    assert report["methods"] == ["bp", "pc", "neutral_cpc", "gated_cpc"]
    assert report["planned_work"]["total_updates"] == 136
    assert report["planned_work"]["probe_updates"] == 64
    assert report["probe_checkpoint_updates"] == [0, 1, 2, 4]
    assert report["resources"]["wall_seconds"] == 60
    assert report["resources"]["cpu_threads"] == 1
    assert set(report["unavailable_diagnostics"]) == {"importance", "neuron_age"}


def test_should_generate_reproducible_reserved_data_separate_from_benchmark_and_global_rng():
    before = pickle.dumps(np.random.get_state())
    training, evaluation, reference = pilot.build_reserved_data()
    again = pilot.build_reserved_data()
    benchmark, final = pilot.build_stream("recurring")
    assert pickle.dumps(np.random.get_state()) == before
    all_batches = (*training, *evaluation, *benchmark, *final)
    ids = [row_id for batch in all_batches for row_id in batch.row_ids]
    assert len(ids) == len(set(ids))
    arrays = [a for batch in all_batches for a in (batch.features, batch.targets)] + [reference]
    assert all(not np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])
    for left, right in zip((*training, *evaluation), (*again[0], *again[1]), strict=True):
        assert left.role in ("probe_train", "probe_test")
        np.testing.assert_array_equal(left.features, right.features)
        np.testing.assert_array_equal(left.targets, right.targets)
        np.testing.assert_array_equal(
            left.targets, (left.features[:, [0]] * left.features[:, [1]] >= 0)
        )
    np.testing.assert_array_equal(reference, again[2])


def test_should_probe_owned_native_pc_and_neutral_checkpoints_with_exact_curves_and_no_mutation():
    pc = CoupledPredictiveCodingNetwork(2, (2,), 23)
    neutral = CoupledCircadianPredictiveCodingNetwork(
        2, (2,), 0, config=CoupledCircadianConfig.matched_pc_control()
    )
    neutral.restore_parameters(pc.snapshot_parameters())
    training, evaluation, reference = pilot.build_reserved_data()
    observed = []
    for model in (pc, neutral):
        template = CoupledLearner(
            model, learning_rate=0.03, inference_steps=4, inference_learning_rate=0.2
        )
        state = template.snapshot_state()
        before = pickle.dumps(state)
        rows = pilot._empty_observations()
        observer = pilot._observer(
            template, training, evaluation, reference, rows, monotonic() + 60
        )
        observer(0, state, None)
        assert pickle.dumps(state) == before == pickle.dumps(template.snapshot_state())
        assert rows[0]["status"] == "complete" and rows[0]["probe"]["completed_updates"] == 4
        assert rows[0]["probe"]["attempted_updates"] == 4
        assert rows[0]["importance"] is rows[0]["neuron_age"] is None
        assert all(row["status"] == "not_run" for row in rows[1:])
        observed.append(rows[0])
    assert observed[0]["probe"]["checkpoints"] == observed[1]["probe"]["checkpoints"]
    assert observed[0]["feature_health"] == observed[1]["feature_health"]


@pytest.mark.parametrize("failure_point", ("constructor", "enter"))
def test_should_keep_all_planned_rows_when_global_setup_fails(monkeypatch, failure_point):
    class FailedSampler:
        def __init__(self):
            if failure_point == "constructor":
                raise RuntimeError("declared constructor failure")

        def __enter__(self):
            raise RuntimeError("declared setup failure")

        def __exit__(self, *args):
            return False

        def snapshot(self):
            raise RuntimeError("unavailable after failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", FailedSampler)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    result = pilot.run_pilot()
    assert result["status"] == "incomplete" and result["total_completed_updates"] == 0
    assert len(result["runs"]) == 4
    assert all(row["status"] == "not_run" and row["outcomes"] is None for row in result["runs"])
    assert sum(len(row["observations"]) for row in result["runs"]) == 16
    assert all(o["status"] == "not_run" for row in result["runs"] for o in row["observations"])
    assert result["pc_neutral_curves_exact"] is None


@pytest.mark.parametrize("action", ("describe", "run"))
def test_should_route_only_the_requested_cli_action(monkeypatch, capsys, action):
    report = {"status": "complete", "scope": "stub only"}
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(
        cli, "describe_protocol" if action == "describe" else "run_pilot", lambda: deepcopy(report)
    )
    monkeypatch.setattr(sys, "argv", ["learnability", f"--{action}"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize("args", ([], ["--run", "--describe"], ["--run", "--seed", "41"]))
def test_should_refuse_missing_action_conflicts_and_unfrozen_controls_before_callbacks(
    monkeypatch, args
):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["learnability", *args])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
