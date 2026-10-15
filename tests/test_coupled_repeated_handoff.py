"""Repeated handoff orchestration through real controllers and fake native ports."""

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
from src.core.learner_ports import TrainingDiagnostic
from test_native_transfer_session import advance, snapshot


class FakeModel:
    def __init__(self, native):
        self.native = deepcopy(native)

    def snapshot_state(self):
        return deepcopy(self.native)


class FakeLearner:
    instances: list[Any] = []
    train_calls = 0

    def __init__(
        self,
        model,
        *,
        learning_rate,
        inference_steps,
        inference_learning_rate,
        inference_policy=None,
    ):
        self.native = model.snapshot_state()
        self.policy = CoupledUpdatePolicy(
            learning_rate, inference_steps, inference_learning_rate, deepcopy(inference_policy)
        )
        self.trains = 0
        self.instances.append(self)

    def fork(self):
        return FakeLearner(
            FakeModel(self.native),
            learning_rate=self.policy.learning_rate,
            inference_steps=self.policy.inference_steps,
            inference_learning_rate=self.policy.inference_learning_rate,
            inference_policy=deepcopy(self.policy.inference_policy),
        )

    def checkpoint_policy(self):
        return deepcopy(self.policy)

    def snapshot_state(self):
        return deepcopy(self.native)

    def restore_state(self, native):
        self.native = deepcopy(native)

    def predict(self, features):
        return np.full(
            (len(features), 1), 0.5 + float(self.native.parameters.output_bias[0, 0]) / 10
        )

    def train_batch(self, features, targets):
        assert np.array_equal(features, [[0.2, -0.3], [-0.4, 0.6]])
        assert np.array_equal(targets, [[1.0], [0.0]])
        self.native = advance(self.native)
        self.trains += 1
        FakeLearner.train_calls += 1
        return TrainingDiagnostic("fake_repeated_coupled_update_v1", 0.75)


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("mode", ["describe", "run"])
def test_should_route_only_the_explicit_default_or_repeated_cli_function(
    monkeypatch, capsys, repeat, mode
):
    import scripts.run_coupled_candidate_handoff as cli

    expected = {"selected": (repeat, mode)}
    calls: list[Any] = []

    def selected():
        calls.append((repeat, mode))
        return expected

    def forbidden():
        raise AssertionError("CLI invoked an unselected product workflow")

    for name in (
        "describe_example",
        "run_example",
        "describe_repeated_example",
        "run_repeated_example",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    name = (
        ("describe_repeated_example" if repeat else "describe_example")
        if mode == "describe"
        else ("run_repeated_example" if repeat else "run_example")
    )
    monkeypatch.setattr(cli, name, selected)
    monkeypatch.setattr(
        sys, "argv", ["handoff", f"--{mode}"] + (["--repeat-handoffs"] if repeat else [])
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(expected))
    assert calls == [(repeat, mode)]


@pytest.mark.parametrize(
    "arguments", [[], ["--describe", "--run"], ["--repeat-handoffs"], ["--run", "--cycles", "2"]]
)
def test_should_refuse_missing_conflicting_or_unfrozen_cli_controls_before_any_work(
    monkeypatch, arguments
):
    import scripts.run_coupled_candidate_handoff as cli

    def forbidden():
        raise AssertionError("invalid CLI reached product work")

    for name in (
        "describe_example",
        "run_example",
        "describe_repeated_example",
        "run_repeated_example",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["handoff", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_describe_fixed_repeated_resources_without_operational_work(monkeypatch):
    from src.adapters.coupled_candidate_handoff import describe_example

    def forbidden(*args, **kwargs):
        raise AssertionError("description reached operational work")

    monkeypatch.setattr(repeated, "_model", forbidden)
    monkeypatch.setattr(repeated, "CandidateCheckpointController", forbidden)
    value = repeated.describe_repeated_example()
    previous = describe_example()
    for name in (
        "families",
        "inference_modes",
        "seed",
        "geometry",
        "training",
        "binding",
        "limits",
    ):
        assert value[name] == previous[name]
    assert value["protocol_id"] == "coupled_repeated_handoff_engineering_v1"
    assert value["label_arrivals"] == [3, 7, 9, 11]
    assert value["runtime_update_limit"] == value["handoff_cycles"] == 3
    assert value["sharing_limits"] == dict(
        max_serving_requests=2, max_admitted_updates=3, max_updates_per_poll=1
    )
    assert value["preparation_attempt_limit"] == 4
    assert value["native_training_calls"] == 48
    assert value["native_energy_evaluation_cap"] == 240
    assert value["native_constructor_cap"] == value["prediction_cap"] == 96
    assert value["guarded_inference_cap"] == 24 and value["process_launches"] == 0


def test_should_run_all_six_repeated_handoffs_through_real_controller_and_owned_fake_ports(
    monkeypatch,
):
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
        captured = CoupledLearnerCheckpointCodec().decode(raw, **kwargs)
        configured = captured.policy
        return FakeLearner(
            FakeModel(captured.native_state),
            learning_rate=configured.learning_rate,
            inference_steps=configured.inference_steps,
            inference_learning_rate=configured.inference_learning_rate,
            inference_policy=configured.inference_policy,
        )

    def controller(shared, **kwargs):
        assert kwargs["max_pending"] == 1 and kwargs["max_prepare_attempts"] == 4
        originals.append(
            (
                shared._runtime.actor,
                shared._runtime._budget,
                shared._runtime._inbox._clock,
                shared._sharing,
            )
        )
        actual = CandidateCheckpointController(shared, **kwargs)
        controllers.append(actual)
        return actual

    monkeypatch.setattr(repeated, "_model", model)
    monkeypatch.setattr(repeated, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    monkeypatch.setattr(repeated, "CandidateCheckpointController", controller)
    result = repeated.run_repeated_example()
    assert result["pc_neutral_parameters_exact"] is True
    assert result["repeated_handoff_verified"] is True
    assert result["full_managed_process_recovery_verified"] is False
    assert len(result["cases"]) == len(controllers) == 6
    assert FakeLearner.train_calls == 48
    for row, control, original in zip(result["cases"], controllers, originals, strict=True):
        runtime, gate = control._shared._runtime, control._shared._sharing
        assert all(
            left is right
            for left, right in zip(
                (runtime.actor, runtime._budget, runtime._inbox._clock, gate), original, strict=True
            )
        )
        assert row["builds"] == [
            "wrong_policy",
            "correct_policy",
            "correct_policy",
            "correct_policy",
        ]
        assert (
            row["preparation_attempts"]
            == row["prepared_models"]
            == control._attempts
            == len(control._models)
            == 4
        )
        assert row["pending_checkpoints"] == len(control._pending) == 0
        assert (
            row["same_actor_budget_clock_gate"]
            and row["actor_unchanged"]
            and row["complete_continuation_exact"]
        )
        assert len(row["cycles"]) == 3
        actor = json.loads(row["actor_a_utf8"])
        final = json.loads(row["final_d_utf8"])
        assert actor["policy"] == final["policy"] == row["policy"]
        for index, cycle in enumerate(row["cycles"]):
            assert cycle["cycle"] == index + 1 and cycle["tick"] == (3, 7, 9)[index]
            assert cycle["preparation_attempts"] == cycle["prepared_models"] == index + 2
            assert cycle["pending_checkpoints"] == 0 and cycle["same_tick_updates"] == []
            assert len(cycle["updates"]) == 1
            assert cycle["updates"][0]["sample_id"] == f"s{index}"
            assert cycle["updates"][0]["update_number"] == index + 1
            assert cycle["updates"][0]["diagnostic"] == cycle["direct_diagnostic"]
            assert (
                cycle["captured_cursor"]
                == cycle["before"]["cursor"]
                == cycle["after_handoff"]["cursor"]
            )
            for key in ("budget", "sharing", "consolidation_cursor"):
                assert cycle["before"][key] == cycle["after_handoff"][key]
            assert cycle["retired_owner_refusal"] and cycle["used_token_refusal"]
            if index == 0:
                assert cycle["after_refusal"] == cycle["before"]
            else:
                assert cycle["after_refusal"] is None
            native = json.loads(cycle["payload_utf8"])["native"]
            if row["family"] == "pc":
                assert native["wake_state"] is None
            else:
                assert native["wake_state"]["wake_updates"] == index + 3
        quota = row["preparation_quota"]
        assert quota["before"] == quota["after_refusal"]
        assert quota["preparation_attempts"] == quota["prepared_models"] == 4
        assert (
            quota["pending_before_discard"] == 1
            and quota["pending_after_discard"] == 0
            and quota["reason"]
        )
        assert (
            row["final_budget"]["updates_completed"]
            == row["final_sharing"]["admitted_updates"]
            == 3
        )
        assert (
            row["refusals"]["sharing"] == "work_quota"
            and row["refusals"]["budget"] == "max_training_updates"
        )
        assert [item["sample_id"] for item in row["final_cursor"]["applied"]] == ["s0", "s1", "s2"]
        if row["family"] != "pc":
            assert (
                actor["native"]["wake_state"]["wake_updates"] == 2
                and final["native"]["wake_state"]["wake_updates"] == 5
            )
        assert len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 65536
    for offset in (0, 3):
        for field in ("actor_a_utf8", "final_d_utf8"):
            pc, neutral = (
                json.loads(result["cases"][offset + index][field])["native"]["arrays"]
                for index in (0, 1)
            )
            assert pc == neutral
        for cycle in range(3):
            pc, neutral = (
                json.loads(result["cases"][offset + index]["cycles"][cycle]["payload_utf8"])[
                    "native"
                ]["arrays"]
                for index in (0, 1)
            )
            assert pc == neutral
