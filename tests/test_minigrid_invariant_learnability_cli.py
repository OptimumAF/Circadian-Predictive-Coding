"""Existing CLI explicitly selects bound learnability without changing defaults."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from test_minigrid_learning_pilot import evaluator as evaluator
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("describe", [False, True])
def test_should_route_bound_flag_only_to_existing_history_scenario(monkeypatch, capsys, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-minigrid-history-learnability", "--invariant-history"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [dict(public_history=True, invariant_history=True)]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--scenario", "torch-minigrid-learnability"],
        ["--scenario", "torch-minigrid-cue-key-door"],
        ["--scenario", "torch-minigrid-history-learnability", "--cue-gradient-geometry"],
    ],
)
def test_should_refuse_unimplemented_flag_combinations_before_selected_work(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--invariant-history", "--describe", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("corrupt_prior", [False, True])
def test_should_complete_bound_diagnostic_route_with_fake_ports_and_original_rnn(
    monkeypatch, evaluator, corrupt_prior
):
    from types import SimpleNamespace
    from copy import deepcopy
    from src.adapters import minigrid_learnability as pilot
    from src.adapters.torch_feature_learner import TorchFeaturePolicy
    from test_minigrid_history_route import history_environment
    from test_minigrid_learning_pilot import Learner

    case = history_environment(monkeypatch, evaluator)
    original = pilot._history_learner
    native_methods = ("bp", "pc", "neutral_cpc", "gated_cpc")
    selected = tuple("public_invariant_" + method for method in native_methods) + ("recurrent_bc",)

    def factory(method):
        if not method.startswith("public_invariant_"):
            return original(method)
        native = TorchFeaturePolicy(
            method.removeprefix("public_invariant_"), 296, 8, 17, num_classes=7
        )
        return Learner(
            SimpleNamespace(
                method=method,
                feature_dim=296,
                hidden_dim=8,
                num_classes=7,
                native_policy=native,
                encoding_id="minigrid_public_wall_direction_invariant_history_296_v1",
            )
        )

    # This fixture checks orchestration; native bound policy admission is covered
    # separately by true-state tests, not imitated by the fake state record.
    def accepted_training(row, packets, sampler, deadline, check_boundary, *, learner_factory):
        from time import monotonic
        from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

        check_boundary()
        source = learner_factory(row["method"])
        initial = source.snapshot_state()
        states = []
        for point in pilot.shared.POINTS:
            state = source.snapshot_state()
            state.completed_updates, state.training_examples = point, 9 * point
            states.append((point, state))
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=128, max_wall_seconds=max(0.0, deadline - monotonic())
            ),
            monotonic,
        )
        budget.updates_completed = 128
        stream = SimpleNamespace(completed_updates=128)
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
            checkpoint_updates=list(pilot.shared.POINTS),
            wake_calls_attempted=128,
            wake_calls_returned=128,
            stream_result=dict(
                completed_updates=128,
                training_examples=1152,
                evaluation_scope="training_only",
                checkpoints=[],
                score_forward_calls=0,
            ),
        )
        case.captured["streams"].append((stream, source, packets, budget))
        case.captured["finished"].append(stream)

        def completed():
            assert budget.updates_completed == 128 and budget.replay_examples_completed == 0
            assert pilot._same_state(source.snapshot_state(), initial)

        return dict(
            row=row, source=source, states=states, budget=budget, stream=stream, guard=completed
        )

    # Shared TRAIN mechanics already have accepted native integration evidence.
    # Stub only that unchanged work; exercise actual fit/probe/flat publication.
    monkeypatch.setattr(pilot.shared, "_train_arm", accepted_training)
    monkeypatch.setattr(pilot.shared, "_readback", lambda *args, **kwargs: None)
    monkeypatch.setattr(pilot, "_history_learner", factory)
    monkeypatch.setattr(
        pilot.shared,
        "_invariant_feature_counter",
        lambda state, point, method: pilot.shared._counter(state, point),
    )
    monkeypatch.setattr(pilot, "_invariant_snapshot", lambda state, point, method: deepcopy(state))
    if corrupt_prior:

        def corrupt(owner):
            if len(case.probes) == 2:
                original = case.rows[0]["probes"][0]
                assert type(original) is bytes
                case.rows[0]["probes"][0] = bytes(bytearray(original))

        case.work["on_train"] = corrupt
    report = pilot.run_pilot(public_history=True, invariant_history=True)
    if corrupt_prior:
        assert report["status"] == "incomplete" and report["comparisons"] is None
        first, second = report["methods"][0]["probes"]
        assert first["status"] == "complete" and first["charged_updates"] == 16
        assert second["status"] == "failed" and second["charged_updates"] == 1
        assert len(case.probes) == 2
        assert all(row["charged_updates"] == 128 for row in report["methods"])
        return
    assert report["status"] == "complete", report["reason"]
    assert tuple(row["method"] for row in report["methods"]) == selected
    assert len(case.captured["finished"]) == 5 and len(case.probes) == 10
    assert report["work"]["native_train_calls"] == 800  # fake calls only
    assert report["work"]["training_rows"] == 7200
    assert report["work"]["final_episodes"] == report["work"]["learner_live_episodes"] == 0
    assert set(report["comparisons"]) == {
        "public_invariant_gated_cpc-minus-public_invariant_pc",
        "public_invariant_gated_cpc-minus-public_invariant_neutral_cpc",
    }
    for row, (_, source, packets, budget) in zip(
        report["methods"], case.captured["streams"], strict=True
    ):
        assert budget.updates_completed == row["charged_updates"] == 128
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        width = 148 if row["method"] == "recurrent_bc" else 296
        assert source.policy.feature_dim == width
        assert all(packet.features.shape == (9, width) for packet in packets)
        assert [slot["charged_updates"] for slot in row["probes"]] == [16, 16]
        assert all(slot["source_unchanged"] and slot["state_unchanged"] for slot in row["probes"])
        assert row["final_evaluation"]["status"] == "not_run"
    assert not case.work.get("environments")
    assert report["scientific_advantage_verified"] is False
