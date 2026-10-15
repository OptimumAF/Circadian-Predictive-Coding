"""Original elapsed and cumulative RSS limits through real handoff boundaries."""

from copy import deepcopy
from dataclasses import replace
import json
import sys
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_checkpoint_callbacks as callbacks_module
import src.adapters.coupled_handoff_resources as resources
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.app.candidate_checkpoint import CandidateCheckpointController
from test_coupled_repeated_handoff import FakeLearner, FakeModel
from test_native_transfer_session import snapshot


_CLI_NAMES = (
    "describe_example",
    "run_example",
    "describe_repeated_example",
    "run_repeated_example",
    "describe_resource_stop_example",
    "run_resource_stop_example",
)


@pytest.mark.parametrize("option", [None, "--repeat-handoffs", "--resource-stops"])
@pytest.mark.parametrize("mode", ["describe", "run"])
def test_should_route_only_the_selected_fixed_resource_protocol(monkeypatch, capsys, option, mode):
    import scripts.run_coupled_candidate_handoff as cli

    calls: list[Any] = []

    def forbidden():
        raise AssertionError("unselected workflow ran")

    def selected():
        calls.append((option, mode))
        return {"selection": (option, mode)}

    for name in _CLI_NAMES:
        monkeypatch.setattr(cli, name, forbidden)
    suffix = {
        None: "example",
        "--repeat-handoffs": "repeated_example",
        "--resource-stops": "resource_stop_example",
    }[option]
    monkeypatch.setattr(cli, f"{mode}_{suffix}", selected)
    monkeypatch.setattr(
        sys, "argv", ["handoff", f"--{mode}"] + ([] if option is None else [option])
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        json.dumps({"selection": (option, mode)})
    )
    assert calls == [(option, mode)]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--describe", "--run"],
        ["--resource-stops"],
        ["--describe", "--resource-stops", "--repeat-handoffs"],
        ["--run", "--resource-stops", "--max-wall-seconds", "9"],
        ["--run", "--resource-stops", "--max-rss", "4096"],
    ],
)
def test_should_refuse_missing_conflicting_and_unfrozen_resource_controls(monkeypatch, arguments):
    import scripts.run_coupled_candidate_handoff as cli

    def forbidden():
        raise AssertionError("invalid controls reached operational work")

    for name in _CLI_NAMES:
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["handoff", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_describe_injected_resources_without_constructing_operational_ports(monkeypatch):
    from src.adapters.coupled_candidate_handoff import describe_example

    def forbidden(*args, **kwargs):
        raise AssertionError("description reached an operational port")

    for name in ("_model", "CoupledLearner", "CandidateCheckpointController", "ProcessRssSampler"):
        monkeypatch.setattr(resources, name, forbidden)
    value, previous = resources.describe_resource_stop_example(), describe_example()
    for key in ("families", "inference_modes", "seed", "geometry", "training", "binding", "limits"):
        assert value[key] == previous[key]
    assert value["injected_clock"] is value["injected_rss"] is True
    assert value["protocol_id"] == "coupled_handoff_resources_engineering_v1"
    assert value["stop_modes"] == ["wall", "rss"]
    assert value["resource_sources"] == dict(
        clock="injected_logical_clock_seconds", rss="injected_reader_bytes"
    )
    assert value["label_arrivals"] == [3, 7, 9]
    assert value["runtime_update_limit"] == value["preparation_attempt_limit"] == 3
    assert value["handoff_cycles"] == 2
    assert value["sharing_limits"] == dict(
        max_serving_requests=2, max_admitted_updates=3, max_updates_per_poll=1
    )
    assert value["simulated_wall_limits"] == dict(wall=8.0, rss=100.0)
    assert value["injected_rss_start_bytes"] == 1024
    assert value["injected_rss_pressure_bytes"] == 4096
    assert value["injected_rss_limit_bytes"] == 2048
    assert value["native_training_calls"] == 72
    assert value["native_energy_evaluation_cap"] == 360
    assert value["native_constructor_cap"] == value["prediction_cap"] == 128
    assert value["guarded_inference_cap"] == 36
    assert value["process_launches"] == 0


def _assert_resource_progress(before, after, *, delivery_tick=None):
    """Sampler observations may grow; every original limit and origin must persist."""
    assert before["budget"]["started_at"] == after["budget"]["started_at"] == 0
    assert before["budget"]["budget"] == after["budget"]["budget"]
    assert before["budget"]["updates_completed"] == after["budget"]["updates_completed"]
    assert before["budget"]["last_clock"] <= after["budget"]["last_clock"]
    old, new = before["budget"]["process_rss_segment"], after["budget"]["process_rss_segment"]
    assert old["pid"] == new["pid"] and old["start_bytes"] == new["start_bytes"] == 1024
    assert old["interval_seconds"] == new["interval_seconds"]
    assert old["peak_bytes"] <= new["peak_bytes"]
    assert old["sample_count"] <= new["sample_count"]
    expected = deepcopy(before)
    if delivery_tick is not None:
        expected["cursor"]["last_tick"] = delivery_tick
        expected["consolidation_cursor"]["revision"] += 1
        expected["sharing"]["admitted_updates"] += 1
    for key in ("cursor", "sharing", "consolidation_cursor"):
        assert expected[key] == after[key]
    assert before["sampler"]["pid"] == after["sampler"]["pid"]
    assert before["sampler"]["start_bytes"] == after["sampler"]["start_bytes"] == 1024
    assert before["sampler"]["peak_bytes"] <= after["sampler"]["peak_bytes"]
    assert before["sampler"]["sample_count"] <= after["sampler"]["sample_count"]
    for key in before["budget"]:
        if key not in {"last_clock", "process_rss_segment"}:
            assert before["budget"][key] == after["budget"][key]


def test_should_preserve_original_resource_ports_and_stop_before_third_build_or_update(monkeypatch):
    FakeLearner.instances, FakeLearner.train_calls = [], 0
    controllers: list[Any] = []
    originals: list[Any] = []

    def model(family):
        native = snapshot(family)
        if native.wake_state is not None:
            native = replace(
                native,
                wake_state=replace(
                    native.wake_state,
                    chemical_state=np.zeros(2),
                    plasticity_state=np.ones(2),
                    wake_updates=0,
                ),
            )
        return FakeModel(native)

    def restore(raw, **kwargs):
        checkpoint = CoupledLearnerCheckpointCodec().decode(raw, **kwargs)
        policy = checkpoint.policy
        return FakeLearner(
            FakeModel(checkpoint.native_state),
            learning_rate=policy.learning_rate,
            inference_steps=policy.inference_steps,
            inference_learning_rate=policy.inference_learning_rate,
            inference_policy=policy.inference_policy,
        )

    def controller(shared, **kwargs):
        assert kwargs["max_pending"] == 1 and kwargs["max_prepare_attempts"] == 3
        runtime, budget = shared._runtime, shared._runtime._budget
        originals.append(
            (
                runtime.actor,
                budget,
                runtime._inbox._clock,
                budget.clock,
                budget.process_rss_sampler,
                budget.process_rss_sampler.read_rss_bytes,
                shared._sharing,
            )
        )
        actual = CandidateCheckpointController(shared, **kwargs)
        controllers.append(actual)
        return actual

    monkeypatch.setattr(resources, "_model", model)
    monkeypatch.setattr(resources, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(resources, "CandidateCheckpointController", controller)
    monkeypatch.setattr(callbacks_module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    result = resources.run_resource_stop_example()
    assert result["pc_neutral_parameters_exact"] is result["resource_stops_verified"] is True
    assert result["full_managed_process_recovery_verified"] is False
    assert len(result["cases"]) == len(controllers) == 12
    assert FakeLearner.train_calls == 72
    for row, control, original in zip(result["cases"], controllers, originals, strict=True):
        runtime, gate = control._shared._runtime, control._shared._sharing
        budget, sampler = runtime._budget, runtime._budget.process_rss_sampler
        assert all(
            left is right
            for left, right in zip(
                (
                    runtime.actor,
                    budget,
                    runtime._inbox._clock,
                    budget.clock,
                    sampler,
                    sampler.read_rss_bytes,
                    gate,
                ),
                original,
                strict=True,
            )
        )
        assert budget.started_at == 0 and budget.last_clock == 9
        assert budget.budget.max_training_updates == 3
        assert budget.budget.max_process_rss_bytes == 2048
        assert control._attempts == 3 and len(control._models) == 2 and len(control._pending) == 0
        assert row["actor_unchanged"] and row["complete_continuation_exact"]
        assert row["same_actor_budget_clock_gate"]
        assert row["resource_identity"]["original"] == row["resource_identity"]["current"]
        assert row["injected_clock"] is row["injected_rss"] is True
        assert len(row["cycles"]) == 2
        actor = json.loads(row["actor_a_utf8"])
        final = json.loads(row["final_c_utf8"])
        assert actor["policy"] == final["policy"] == row["policy"]
        for index, cycle in enumerate(row["cycles"]):
            assert cycle["tick"] == (3, 7)[index]
            assert cycle["preparation_attempts"] == cycle["prepared_models"] == index + 1
            assert cycle["pending_checkpoints"] == 0 and cycle["same_tick_updates"] == []
            assert len(cycle["updates"]) == 1
            update = cycle["updates"][0]
            assert update["sample_id"] == f"s{index}" and update["update_number"] == index + 1
            assert update["diagnostic"] == cycle["direct_diagnostic"]
            _assert_resource_progress(cycle["before"], cycle["after_handoff"])
            assert cycle["retired_owner_refusal"] and cycle["used_token_refusal"]
            native = json.loads(cycle["payload_utf8"])["native"]
            if row["family"] != "pc":
                assert native["wake_state"]["wake_updates"] == index + 3
        expected_reason = (
            "max_wall_seconds" if row["stop_mode"] == "wall" else "max_process_rss_bytes"
        )
        assert budget.budget.max_wall_seconds == (8 if row["stop_mode"] == "wall" else 100)
        preparation = row["preparation_stop"]
        _assert_resource_progress(preparation["before"], preparation["after"])
        assert preparation["attempts"] == 3 and preparation["models"] == 2
        assert preparation["pending_before_discard"] == 1
        assert preparation["pending_after_discard"] == 0
        for stop in (preparation["stop"], row["update_stop"]["stop"]):
            assert stop["reason"] == expected_reason
            assert stop["updates_completed"] == 2 and stop["elapsed_seconds"] == 9
            assert stop["status"] == "incomplete" and stop["checkpoint_position"] is None
        before, after = row["update_stop"]["before"], row["update_stop"]["after"]
        _assert_resource_progress(before, after, delivery_tick=9)
        assert before["sharing"]["admitted_updates"] == 2
        assert after["sharing"]["admitted_updates"] == 3
        assert row["final_quota"] == dict(updates=[], deferred_reason="work_quota")
        assert row["sharing_refusal"] == "work_quota"
        initial, final_sample = row["sampler_lineage"]["initial"], row["sampler_lineage"]["final"]
        assert initial["pid"] == final_sample["pid"]
        assert initial["start_bytes"] == final_sample["start_bytes"] == 1024
        assert initial["interval_seconds"] == final_sample["interval_seconds"]
        assert initial["sample_count"] < final_sample["sample_count"]
        assert initial["peak_bytes"] == 1024
        assert final_sample["peak_bytes"] == (1024 if row["stop_mode"] == "wall" else 4096)
        assert sampler.read_rss_bytes() == 1024 and sampler._thread is None
        assert row["final_reader_bytes"] == 1024 and row["same_original_resources"]
        assert row["final_budget"]["process_rss_segment"] == final_sample
        assert row["final_budget"]["updates_completed"] == 2
        assert row["final_sharing"]["admitted_updates"] == 3
        assert row["final_cursor"]["completed_updates"] == 2
        assert row["final_cursor"]["stopped"] is False
        assert [value["sample_id"] for value in row["final_cursor"]["applied"]] == ["s0", "s1"]
        assert len(row["final_cursor"]["experiences"]) == len(row["final_cursor"]["labels"]) == 3
        if row["family"] != "pc":
            assert actor["native"]["wake_state"]["wake_updates"] == 2
            assert final["native"]["wake_state"]["wake_updates"] == 4
        assert len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 65536
    for offset in (0, 3, 6, 9):
        for field in ("actor_a_utf8", "final_c_utf8"):
            frames = [
                json.loads(result["cases"][offset + i][field])["native"]["arrays"] for i in (0, 1)
            ]
            assert frames[0] == frames[1]
        for index in range(2):
            frames = [
                json.loads(result["cases"][offset + i]["cycles"][index]["payload_utf8"])["native"][
                    "arrays"
                ]
                for i in (0, 1)
            ]
            assert frames[0] == frames[1]
