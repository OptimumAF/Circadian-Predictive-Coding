"""Finite handoff endurance with actual owners and deterministic fake native ports."""

from copy import deepcopy
from dataclasses import replace
import json
import sys
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_checkpoint_callbacks as callbacks_module
import src.adapters.coupled_repeated_handoff as repeated
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.core.coupled_learner_state import CoupledUpdatePolicy
from test_coupled_repeated_handoff import FakeLearner, FakeModel
from test_native_transfer_session import snapshot


_ROUTES = {
    None: "example",
    "--repeat-handoffs": "repeated_example",
    "--resource-stops": "resource_stop_example",
    "--endurance": "endurance_example",
}


@pytest.mark.parametrize("option", list(_ROUTES))
@pytest.mark.parametrize("mode", ["describe", "run"])
def test_should_route_only_the_explicit_endurance_or_unchanged_cli_mode(
    monkeypatch, capsys, option, mode
):
    import scripts.run_coupled_candidate_handoff as cli

    calls: list[Any] = []

    def forbidden():
        raise AssertionError("unselected workflow ran")

    def selected():
        calls.append((option, mode))
        return {"selected": (option, mode)}

    for suffix in _ROUTES.values():
        for verb in ("describe", "run"):
            monkeypatch.setattr(cli, f"{verb}_{suffix}", forbidden)
    monkeypatch.setattr(cli, f"{mode}_{_ROUTES[option]}", selected)
    monkeypatch.setattr(
        sys, "argv", ["handoff", f"--{mode}"] + ([] if option is None else [option])
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        json.dumps({"selected": (option, mode)})
    )
    assert calls == [(option, mode)]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--endurance"],
        ["--describe", "--endurance", "--repeat-handoffs"],
        ["--run", "--endurance", "--resource-stops"],
        ["--run", "--endurance", "--cycles", "3"],
        ["--run", "--endurance", "--max-wall-seconds", "60"],
        ["--run", "--endurance", "--max-rss", "1024"],
    ],
)
def test_should_refuse_conflicting_or_unfrozen_endurance_controls(monkeypatch, arguments):
    import scripts.run_coupled_candidate_handoff as cli

    def forbidden():
        raise AssertionError("invalid controls reached operational workflow")

    for suffix in _ROUTES.values():
        for verb in ("describe", "run"):
            monkeypatch.setattr(cli, f"{verb}_{suffix}", forbidden)
    monkeypatch.setattr(sys, "argv", ["handoff", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_describe_the_fixed_horizon_without_reading_resources_or_models(monkeypatch):
    from src.adapters.coupled_candidate_handoff import describe_example

    def forbidden(*args, **kwargs):
        raise AssertionError("description reached operational work")

    for name in (
        "_model",
        "CoupledLearner",
        "CandidateCheckpointController",
        "ProcessRssSampler",
        "monotonic",
        "read_process_rss_bytes",
    ):
        monkeypatch.setattr(repeated, name, forbidden)
    value, old = repeated.describe_endurance_example(), describe_example()
    for key in ("families", "inference_modes", "seed", "geometry", "training", "binding", "limits"):
        assert value[key] == old[key]
    assert value["protocol_id"] == "coupled_handoff_endurance_engineering_v1"
    assert value["horizon"] == value["handoff_cycles"] == value["runtime_update_limit"] == 16
    assert value["sharing_limits"] == dict(
        max_serving_requests=2, max_admitted_updates=16, max_updates_per_poll=1
    )
    assert value["preparation_attempt_limit"] == 17
    assert value["label_arrivals"] == [3, *range(7, 38, 2)]
    assert value["native_training_calls"] == 204
    assert value["native_energy_evaluation_cap"] == 1020
    assert value["native_constructor_cap"] == 256
    assert value["prediction_cap"] == 340
    assert value["guarded_inference_cap"] == 102
    assert value["expected_native_energy_evaluations"] == 816
    assert value["expected_native_constructors"] == 228
    assert value["expected_native_predictions"] == 306
    assert value["resource_limits"] == dict(
        max_training_updates=16, max_wall_seconds=45.0, max_process_rss_bytes=512 * 1024**2
    )
    assert value["process_launches"] == 0


@pytest.mark.parametrize("reading", [None, 0, True, 512 * 1024**2 + 1])
def test_should_refuse_unavailable_or_over_limit_initial_rss_before_model_construction(
    monkeypatch, reading
):
    def forbidden(*args, **kwargs):
        raise AssertionError("resource preflight reached model work")

    monkeypatch.setattr(repeated, "_model", forbidden)
    monkeypatch.setattr(repeated, "CoupledLearner", forbidden)
    monkeypatch.setattr(repeated, "read_process_rss_bytes", lambda: reading)
    monkeypatch.setattr(repeated, "monotonic", lambda: 0.0)
    with pytest.raises((RuntimeError, ValueError)):
        repeated.run_endurance_example()


def test_should_refuse_initial_elapsed_limit_before_model_construction(monkeypatch):
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 45.0

    def forbidden(*args, **kwargs):
        raise AssertionError("elapsed preflight reached model work")

    monkeypatch.setattr(repeated, "_model", forbidden)
    monkeypatch.setattr(repeated, "CoupledLearner", forbidden)
    monkeypatch.setattr(repeated, "read_process_rss_bytes", lambda: 1024)
    monkeypatch.setattr(repeated, "monotonic", clock)
    with pytest.raises(RuntimeError):
        repeated.run_endurance_example()


def test_should_keep_the_original_wall_and_raw_rss_ports_after_global_rebinding(monkeypatch):
    reads: list[int] = []

    def wall():
        return 1.0

    def reader():
        reads.append(1024)
        return 1024

    def forbidden():
        raise AssertionError("resource setup retained a mutable global lookup")

    monkeypatch.setattr(repeated, "monotonic", wall)
    monkeypatch.setattr(repeated, "read_process_rss_bytes", reader)
    _, budget, sampler = repeated._resource_setup(16, True)
    assert sampler is not None and budget.clock is wall
    retained_reader = sampler.read_rss_bytes
    initial_reads = len(reads)
    monkeypatch.setattr(repeated, "monotonic", forbidden)
    monkeypatch.setattr(repeated, "read_process_rss_bytes", forbidden)
    budget.before_final()
    assert len(reads) == initial_reads + 1
    assert budget.clock is wall and sampler.read_rss_bytes is retained_reader
    assert budget.started_at == budget.last_clock == 1.0
    assert sampler.start_bytes == sampler.peak_bytes == 1024 and sampler._thread is None


@pytest.mark.parametrize(
    "path,value",
    [
        (("budget", "started_at"), 2.0),
        (("budget", "updates_completed"), 3),
        (("budget", "budget", "max_wall_seconds"), 46),
        (("sampler", "pid"), 2),
        (("sampler", "start_bytes"), 2048),
        (("sampler", "peak_bytes"), 1023),
        (("sampler", "sample_count"), 0),
        (("budget", "last_clock"), 0.0),
    ],
)
def test_should_refuse_resource_reset_or_tampering_in_full_handoff_evidence(path, value):
    policy = CoupledUpdatePolicy(0.025, 3, 0.07)
    segment = dict(pid=1, start_bytes=1024, peak_bytes=1024, sample_count=1, interval_seconds=0.005)
    before: dict[str, Any] = dict(
        native=snapshot(),
        cursor={"last_tick": 3},
        sharing={"admitted_updates": 1},
        consolidation_cursor={"revision": 1},
        sampler=segment,
        budget=dict(
            started_at=1.0,
            last_clock=1.0,
            updates_completed=1,
            budget=dict(max_wall_seconds=45),
            process_rss_segment=deepcopy(segment),
        ),
    )
    after = deepcopy(before)
    current: Any = after
    for component in path[:-1]:
        current = current[component]
    current[path[-1]] = value
    payload = repeated._payload(policy, before["native"])
    with pytest.raises(AssertionError):
        repeated._unchanged(policy, payload, before, after, "changed original resources")


def _assert_same_resources_with_cumulative_telemetry(before, after):
    old, new = deepcopy(before), deepcopy(after)
    assert new["budget"].pop("last_clock") >= old["budget"].pop("last_clock")
    for previous, current in (
        (old.pop("sampler"), new.pop("sampler")),
        (old["budget"].pop("process_rss_segment"), new["budget"].pop("process_rss_segment")),
    ):
        assert current.pop("peak_bytes") >= previous.pop("peak_bytes")
        assert current.pop("sample_count") >= previous.pop("sample_count")
        assert current == previous
    assert new == old


def test_should_complete_six_finite_endurance_cases_with_owned_original_resource_ports(monkeypatch):
    FakeLearner.instances, FakeLearner.train_calls = [], 0
    controllers: list[Any] = []
    originals: list[Any] = []
    clock_values: list[float] = []
    read_values: list[int] = []

    def wall():
        value = len(clock_values) * 0.001
        clock_values.append(value)
        return value

    def reader():
        value = 1024 + len(read_values)
        read_values.append(value)
        return value

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
        assert kwargs["max_pending"] == 1 and kwargs["max_prepare_attempts"] == 17
        runtime, budget = shared._runtime, shared._runtime._budget
        assert budget.clock is wall
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

    monkeypatch.setattr(repeated, "_model", model)
    monkeypatch.setattr(repeated, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(repeated, "CandidateCheckpointController", controller)
    monkeypatch.setattr(repeated, "monotonic", wall)
    monkeypatch.setattr(repeated, "read_process_rss_bytes", reader)
    monkeypatch.setattr(callbacks_module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    result = repeated.run_endurance_example()
    assert result["handoff_endurance_verified"] is result["pc_neutral_parameters_exact"] is True
    assert result["full_managed_process_recovery_verified"] is False
    assert len(result["cases"]) == len(controllers) == 6
    assert FakeLearner.train_calls == 204
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
        assert sampler._thread is None
        assert (
            budget.budget.max_wall_seconds == 45
            and budget.budget.max_process_rss_bytes == 512 * 1024**2
        )
        assert budget.budget.max_training_updates == 16
        assert control._attempts == len(control._models) == 17 and len(control._pending) == 0
        assert row["actor_unchanged"] and row["complete_continuation_exact"]
        assert row["preparation_attempts"] == row["prepared_models"] == 17
        assert row["pending_checkpoints"] == 0
        assert row["builds"] == ["wrong_policy"] + ["correct_policy"] * 16
        assert len(row["cycles"]) == 16
        actor = json.loads(row["actor_a_utf8"])
        final = json.loads(row["final_payload_utf8"])
        assert actor["policy"] == final["policy"] == row["policy"]
        for index, cycle in enumerate(row["cycles"]):
            assert cycle["cycle"] == index + 1 and cycle["tick"] == [3, *range(7, 36, 2)][index]
            assert cycle["preparation_attempts"] == cycle["prepared_models"] == index + 2
            assert cycle["pending_checkpoints"] == 0 and cycle["same_tick_updates"] == []
            assert len(cycle["updates"]) == 1
            update = cycle["updates"][0]
            assert update["sample_id"] == f"s{index}" and update["update_number"] == index + 1
            assert update["applied_at"] == cycle["tick"]
            assert update["diagnostic"] == cycle["direct_diagnostic"]
            assert all(flag is True for flag in cycle["source_assertions"].values())
            assert set(cycle["source_assertions"]) == {
                "complete_native_reference_exact",
                "complete_cursor_resources_preserved",
                "same_original_resources",
                "old_owner_retired",
                "token_consumed",
            }
            assert cycle["retired_owner_refusal"] and cycle["used_token_refusal"]
            assert "cursor" not in cycle and "before" not in cycle and "after_handoff" not in cycle
            observed = cycle["resources"]
            for stage in ("after_refusal", "after_handoff"):
                if observed[stage] is None:
                    assert stage == "after_refusal" and index > 0
                    continue
                old, new = observed["before"], observed[stage]
                _assert_same_resources_with_cumulative_telemetry(old, new)
                assert (
                    old["budget"]["started_at"] == new["budget"]["started_at"] == budget.started_at
                )
                assert old["budget"]["budget"] == new["budget"]["budget"]
                assert old["budget"]["last_clock"] <= new["budget"]["last_clock"]
                assert old["sampler"]["pid"] == new["sampler"]["pid"]
                assert old["sampler"]["start_bytes"] == new["sampler"]["start_bytes"]
                assert old["sampler"]["sample_count"] <= new["sampler"]["sample_count"]
                assert old["sampler"]["peak_bytes"] <= new["sampler"]["peak_bytes"]
            native = json.loads(cycle["payload_utf8"])["native"]
            if row["family"] != "pc":
                assert native["wake_state"]["wake_updates"] == index + 3
        assert row["preparation_quota"]["preparation_attempts"] == 17
        assert row["preparation_quota"]["prepared_models"] == 17
        assert row["preparation_quota"]["pending_before_discard"] == 1
        assert row["preparation_quota"]["pending_after_discard"] == 0
        _assert_same_resources_with_cumulative_telemetry(
            row["preparation_quota"]["before"], row["preparation_quota"]["after_refusal"]
        )
        assert row["preparation_quota"]["source_assertions"] == dict(
            complete_cursor_resources_preserved=True,
            same_original_resources=True,
            builder_not_called=True,
            token_discarded=True,
        )
        assert row["horizon"] == 16
        assert row["resource_sources"] == dict(
            clock="test_injected_clock",
            rss="test_injected_rss",
            injected_clock=True,
            injected_rss=True,
        )
        assert row["resource_identity"]["original"] == row["resource_identity"]["current"]
        initial, final_resources = row["resources"]["initial"], row["resources"]["final"]
        assert (
            initial["budget"]["started_at"]
            == final_resources["budget"]["started_at"]
            == budget.started_at
        )
        assert initial["sampler"]["start_bytes"] == final_resources["sampler"]["start_bytes"]
        assert initial["sampler"]["pid"] == final_resources["sampler"]["pid"]
        assert initial["sampler"]["sample_count"] < final_resources["sampler"]["sample_count"]
        assert initial["sampler"]["peak_bytes"] < final_resources["sampler"]["peak_bytes"]
        assert row["final_budget"]["started_at"] == budget.started_at
        assert row["final_budget"]["last_clock"] == budget.last_clock
        assert (
            row["final_budget"]["updates_completed"]
            == row["final_sharing"]["admitted_updates"]
            == 16
        )
        cursor = row["final_cursor"]
        assert len(cursor["experiences"]) == len(cursor["labels"]) == 17
        ordered = sorted(range(17), key=lambda i: (f"e{i}", f"s{i}"))
        arrivals = [3, *range(7, 38, 2)]
        for number, experience, label in zip(
            ordered, cursor["experiences"], cursor["labels"], strict=True
        ):
            assert experience["sample_id"] == label["sample_id"] == f"s{number}"
            assert experience["episode_id"] == label["episode_id"] == f"e{number}"
            assert experience["model_version"] == label["model_version"] == "a0"
            assert experience["role"] == label["role"] == "train"
            assert experience["observed_at"] == 0 and label["arrived_at"] == arrivals[number]
            assert label["event_id"] == f"l{number}"
            assert experience["features"] == [[0.2, -0.3], [-0.4, 0.6]]
            assert label["targets"] == [[1.0], [0.0]]
            assert experience["permissions"] == dict(training=True, replay=False, evaluation=False)
        assert [value["sample_id"] for value in cursor["applied"]] == [f"s{i}" for i in range(16)]
        assert (
            cursor["completed_updates"] == 16
            and cursor["last_tick"] == 37
            and cursor["stopped"] is False
        )
        assert (
            row["refusals"]["sharing"] == "work_quota"
            and row["refusals"]["budget"] == "max_training_updates"
        )
        if row["family"] != "pc":
            assert actor["native"]["wake_state"]["wake_updates"] == 2
            assert final["native"]["wake_state"]["wake_updates"] == 18
        assert len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 131072
    assert len(json.dumps(result, separators=(",", ":"), allow_nan=False).encode()) <= 1048576
    assert clock_values == sorted(clock_values) and read_values == sorted(read_values)
    for offset in (0, 3):
        for field in ("actor_a_utf8", "final_payload_utf8"):
            assert (
                json.loads(result["cases"][offset][field])["native"]["arrays"]
                == json.loads(result["cases"][offset + 1][field])["native"]["arrays"]
            )
        for index in range(16):
            assert (
                json.loads(result["cases"][offset]["cycles"][index]["payload_utf8"])["native"][
                    "arrays"
                ]
                == json.loads(result["cases"][offset + 1]["cycles"][index]["payload_utf8"])[
                    "native"
                ]["arrays"]
            )
