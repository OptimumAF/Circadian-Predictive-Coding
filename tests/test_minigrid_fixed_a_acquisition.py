"""Frozen A128 acquisition criteria and caller flow, without native work."""

from dataclasses import replace
import json
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_minigrid_history_evaluation_smoke as cli
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_key_door import predefined_layouts
from test_minigrid_learning_pilot import Learner, evaluator as evaluator
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


def record(method, point, *, gain=1.0, raw=True, goals=True):
    return dict(
        method=method,
        updates=point,
        role="train",
        status="complete",
        reason=None,
        scores=[
            dict(
                context="A",
                cue=cue,
                ce=2.0 - gain * point / 128,
                accuracy=float(point == 128 and raw),
                exact_sequence_accuracy=float(point == 128 and raw),
            )
            for cue in ("blue", "red")
        ],
        episodes=[
            dict(
                context="A",
                cue=cue,
                status="complete",
                success=point == 128 and goals,
                **{"return": float(point == 128 and goals)},
            )
            for cue in ("blue", "red")
        ]
        if point in (0, 128)
        else [],
    )


def all_records():
    return [record(method, point) for method in cli.shared.METHODS for point in cli.shared.POINTS]


def test_should_describe_fixed_dose_without_resources_and_preserve_default_false(monkeypatch):
    for name in (
        "require_torch",
        "ProcessRssSampler",
        "evaluate_checkpoint",
        "_score_checkpoint",
        "_history_learner",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    original = cli.describe_protocol()
    assert original == cli.describe_protocol(fixed_a_acquisition=False)
    assert (
        original["checkpoints"] == [0, 32]
        and original["planned_work"]["native_training_updates"] == 160
    )
    fixed = cli.describe_protocol(fixed_a_acquisition=True)
    assert json.loads(json.dumps(fixed, allow_nan=False)) == fixed
    assert fixed["protocol_id"] != original["protocol_id"]
    assert fixed["checkpoints"] == [0, 32, 64, 96, 128]
    assert fixed["planned_work"]["native_training_updates"] == 640
    assert fixed["planned_work"]["training_rows"] == 5760
    assert fixed["planned_work"]["raw_score_forwards"] == 50
    assert fixed["planned_work"]["raw_score_rows"] == 450
    assert fixed["planned_work"]["live_episodes"] == 20
    assert fixed["planned_work"]["max_total_physics"] == 416
    assert fixed["final_episodes"] == fixed["probe_updates"] == 0


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_nonboolean_fixed_mode_before_resources(monkeypatch, invalid):
    monkeypatch.setattr(cli, "require_torch", forbidden)
    monkeypatch.setattr(cli, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        cli.describe_protocol(fixed_a_acquisition=invalid)
    with pytest.raises(ValueError):
        cli.run_smoke(fixed_a_acquisition=invalid)


@pytest.mark.parametrize("describe", [False, True])
def test_should_route_explicit_fixed_cli_option_without_changing_default_call_shape(
    monkeypatch, capsys, describe
):
    monkeypatch.setattr(
        sys, "argv", ["fixed-a", "--fixed-a-acquisition"] + (["--describe"] if describe else [])
    )
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "describe_protocol", selected if describe else forbidden)
    monkeypatch.setattr(cli, "run_smoke", forbidden if describe else selected)
    cli.main()
    assert calls == [{"fixed_a_acquisition": True}]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


def test_should_build_unique_alternating_a_packets_without_touching_sources(evaluator):
    source_list = list(evaluator.sources)
    for index, code in enumerate((-0.5, 0.5)):
        original = source_list[index]
        episode = replace(
            original.episode,
            transitions=tuple(
                replace(
                    step,
                    observation=(code,) + step.observation[1:],
                    next_observation=(code,) + step.next_observation[1:],
                )
                for step in original.episode.transitions
            ),
        )
        values, targets = episode_arrays(episode)
        source_list[index] = replace(original, episode=episode, features=values, targets=targets)
    sources = tuple(source_list)
    before = pickle.dumps(sources)
    packets = cli._fixed_a_packets(sources)
    assert len(packets) == 128 and pickle.dumps(sources) == before
    assert len({packet.episode.episode_id for packet in packets}) == 128
    assert len({identity for packet in packets for identity in packet.row_ids}) == 1152
    assert all(
        packet.context == "A" and packet.role == packet.episode.role == "train"
        for packet in packets
    )
    for index, packet in enumerate(packets):
        source = sources[index % 2]
        assert packet.features.shape == (9, 148)
        np.testing.assert_array_equal(packet.features, source.features)
        np.testing.assert_array_equal(packet.targets, source.targets)
        assert not np.shares_memory(packet.features, source.features)
        assert not np.shares_memory(packet.targets, source.targets)
        assert packet.episode.episode_id == f"minigrid-fixed-a-training-{index}"
    assert all(not np.shares_memory(packets[0].features, packet.features) for packet in packets[1:])


@pytest.mark.parametrize("raw,goals", [(False, False), (True, False), (False, True), (True, True)])
def test_should_require_both_raw_sequences_and_both_native_goals(raw, goals):
    records = all_records()
    final = next(row for row in records if row["method"] == "bp" and row["updates"] == 128)
    final["scores"][1]["exact_sequence_accuracy"] = float(raw)
    final["scores"][1]["accuracy"] = float(raw)
    final["episodes"][1]["success"] = goals
    final["episodes"][1]["return"] = float(goals)
    result = cli._acquisition_outcomes(records)
    first = result["methods"][0]
    assert first["raw_both_cues_exact"] is raw
    assert first["live_both_cues_goals"] is goals
    assert first["joint_acquisition"] is (raw and goals)


@pytest.mark.parametrize("fault", ["missing", "failed", "whole_case_refused"])
def test_should_leave_unavailable_acquisition_and_gains_null(fault):
    records = all_records()
    if fault == "missing":
        records = [
            row for row in records if not (row["method"] == "gated_cpc" and row["updates"] == 64)
        ]
    elif fault == "failed":
        next(row for row in records if row["method"] == "gated_cpc" and row["updates"] == 64)[
            "status"
        ] = "failed"
    result = cli._acquisition_outcomes(records, complete=fault != "whole_case_refused")
    active = result["methods"][3]
    assert active["status"] == "unavailable"
    assert all(
        active[key] is None
        for key in (
            "raw_both_cues_exact",
            "live_both_cues_goals",
            "joint_acquisition",
            "ce_gain",
            "live_success_count_change",
            "mean_return_gain",
        )
    )
    assert all(
        value is None
        for comparison in result["signed_comparisons"].values()
        for value in comparison.values()
    )
    if fault == "whole_case_refused":
        assert all(row["status"] == "unavailable" for row in result["methods"])


def test_should_preserve_negative_ce_gain_and_active_minus_control_directions():
    gains = {"bp": 1.0, "pc": 2.0, "neutral_cpc": 3.0, "gated_cpc": -0.5, "recurrent_bc": 4.0}
    records = [
        record(method, point, gain=gains[method])
        for method in cli.shared.METHODS
        for point in cli.shared.POINTS
    ]
    result = cli._acquisition_outcomes(records)
    active = result["methods"][3]
    assert active["ce_gain"] == -0.5 and active["live_success_count_change"] == 2
    assert active["mean_return_gain"] == 1
    assert result["signed_comparisons"]["gated_cpc-minus-pc"]["ce_gain"] == -2.5
    assert result["signed_comparisons"]["gated_cpc-minus-neutral_cpc"]["ce_gain"] == -3.5


def fake_case(monkeypatch, evaluator, *, fault=None):
    trained, calls = [], []
    outputs: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(cli, "ProcessRssSampler", Sampler)

    def collect(check_boundary, reports):
        check_boundary()
        reports.extend(
            {"context": packet.context, "environment_steps": 9} for packet in evaluator.sources
        )
        return evaluator.sources

    def train(row, packets, sampler, deadline, check_boundary, **kwargs):
        check_boundary()
        method, width = row["method"], packets[0].features.shape[1]
        assert len(packets) == 128 and all(packet.context == "A" for packet in packets)
        assert kwargs["checkpoint_updates"] == cli.shared.POINTS
        assert kwargs["learner_factory"] is cli._history_learner
        policy = (
            TorchRecurrentPolicy(148, 8, 7, 17)
            if method == "recurrent_bc"
            else TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)
        )
        assert width == policy.feature_dim
        source = Learner(policy)
        states = []
        for point in cli.shared.POINTS:
            state = source.snapshot_state()
            state.completed_updates, state.training_examples = point, 9 * point
            states.append((point, state))
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=128), lambda: 0.0)
        budget.updates_completed = 128
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
            checkpoint_updates=list(cli.shared.POINTS),
            wake_calls_attempted=128,
            wake_calls_returned=128,
        )
        sealed = pickle.dumps(
            (
                source.snapshot_state(),
                states,
                budget.budget,
                budget.started_at,
                budget.updates_completed,
            )
        )

        def guard():
            assert (
                pickle.dumps(
                    (
                        source.snapshot_state(),
                        states,
                        budget.budget,
                        budget.started_at,
                        budget.updates_completed,
                    )
                )
                == sealed
            )

        arm = dict(row=row, source=source, states=states, budget=budget, guard=guard)
        trained.append(arm)
        return arm

    def read(source, state, **kwargs):
        assert len(trained) == 5 and all(arm["row"]["charged_updates"] == 128 for arm in trained)
        if fault == "prior" and outputs:
            outputs[0]["scores"][0]["ce"] = 999.0
        kwargs["check_boundary"]()
        method, point = kwargs["method"], kwargs["updates"]
        arm = next(arm for arm in trained if arm["row"]["method"] == method)
        assert source is arm["source"] and state is next(
            value for update, value in arm["states"] if update == point
        )
        assert kwargs["budget"] is arm["budget"]
        live = "layouts" in kwargs
        assert live == (point in (0, 128))
        if live:
            assert kwargs["layouts"] == predefined_layouts()[:1]
            assert kwargs["role"] == "train" and kwargs["public_history"] is True
        width = 148 if live or method == "recurrent_bc" else 296
        assert len(kwargs["sources"]) == 2 and all(
            packet.features.shape == (9, width) for packet in kwargs["sources"]
        )
        value = record(method, point)
        if fault == "raw" and point == 64:
            value.update(status="failed", reason="paid raw scorer failed")
            value["scores"] = value["scores"][:1]
        calls.append((method, point, live))
        outputs.append(value)
        return value

    monkeypatch.setattr(cli.shared, "_collect_sources", collect)
    monkeypatch.setattr(cli.shared, "_train_arm", train)
    monkeypatch.setattr(cli, "evaluate_checkpoint", read)
    monkeypatch.setattr(cli, "_score_checkpoint", read)
    return SimpleNamespace(trained=trained, calls=calls, outputs=outputs)


def test_should_finish_all_five_fixed_a_streams_before_fixed_checkpoint_readouts(
    monkeypatch, evaluator
):
    case = fake_case(monkeypatch, evaluator)
    report = cli.run_smoke(fixed_a_acquisition=True)
    assert report["status"] == "complete", report["reason"]
    assert case.calls == [
        (method, point, point in (0, 128))
        for method in cli.shared.METHODS
        for point in cli.shared.POINTS
    ]
    assert (
        len(report["evaluations"]) == 25
        and sum(len(row["scores"]) for row in report["evaluations"]) == 50
    )
    assert sum(len(row["episodes"]) for row in report["evaluations"]) == 20
    assert sum(row["charged_updates"] for row in report["methods"]) == 640
    assert sum(row["native_training_examples"] for row in report["methods"]) == 5760
    assert len(report["training_sources"]) == len(report["source_reports"]) == 6
    assert all(row["joint_acquisition"] is True for row in report["acquisition"]["methods"])
    assert not evaluator.work.get("trains") and not evaluator.work.get("predicts")
    assert not evaluator.work.get("environments")
    case.outputs[0]["scores"][0]["ce"] = 999
    assert report["evaluations"][0]["scores"][0]["ce"] == 2


@pytest.mark.parametrize("fault", ["prior", "raw"])
def test_should_retain_all_training_charges_and_paid_readout_prefix_when_later_work_refuses(
    monkeypatch, evaluator, fault
):
    case = fake_case(monkeypatch, evaluator, fault=fault)
    report = cli.run_smoke(fixed_a_acquisition=True)
    assert report["status"] == "incomplete" and report["reason"]
    assert (
        len(case.trained) == 5 and sum(row["charged_updates"] for row in report["methods"]) == 640
    )
    assert all(
        row["status"] == "unavailable" and row["joint_acquisition"] is None
        for row in report["acquisition"]["methods"]
    )
    if fault == "prior":
        assert "preceding paid evaluation" in report["reason"]
        assert len(case.calls) == len(report["evaluations"]) == 1
        assert report["evaluations"][0]["scores"][0]["ce"] == 2
    else:
        assert len(case.calls) == len(report["evaluations"]) == 3
        assert report["evaluations"][-1]["status"] == "failed"
        assert len(report["evaluations"][-1]["scores"]) == 1
