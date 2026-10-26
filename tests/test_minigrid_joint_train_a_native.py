"""One full native S64 stream and genuine frozen readout with fake physics.

Reuses the accepted complete-route assertions; fake-game success is not evidence
of navigation or useful learning. No additional model is trained for this test.
"""

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_policy_evaluation as evaluation
from test_minigrid_invariant_evaluation import DiscreteEnvironment
from test_minigrid_learning_pilot import Environment
from test_minigrid_relax64_native import (
    test_should_complete_fixed64_joint_stream_and_disposable_f_return as exercise_route,
)


def test_should_run_real_joint64_stream_then_two_zero_update_a_games(monkeypatch):
    reports = []
    original = pilot.run_pilot

    def live_route(**options):
        result = original(**options, train_a_rollout=True)
        reports.append(result)
        return result

    monkeypatch.setattr(Environment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", DiscreteEnvironment)
    monkeypatch.setattr(pilot, "run_pilot", live_route)
    exercise_route(monkeypatch, "pc_relax64")
    assert len(reports) == 1
    report = reports[0]
    row = report["methods"][0]
    live = row["train_a_rollout"]
    assert live["status"] == "complete", live["reason"]
    assert live["charged_updates"] == live["replay_examples"] == 0
    assert live["budget_binding_preserved"] is True
    evaluated = live["report"]
    assert evaluated["method"] == "public_invariant_pc_relax64"
    assert evaluated["native_method"] == "pc"
    assert evaluated["native_training_examples"] == 6912
    assert evaluated["catalogue_rows_per_update"] == 54
    assert evaluated["scores"] == [] and evaluated["state_unchanged"] is True
    assert [
        (slot["context"], slot["cue"], slot["source_role"]) for slot in evaluated["episodes"]
    ] == [("A", "blue", "train"), ("A", "red", "train")]
    work = report["work"]
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["total_score_forwards_completed"] == 94
    assert work["total_prediction_forwards_completed"] == 98
    assert work["total_prediction_rows_completed"] == 850
    assert work["learner_live_episodes"] == 2
    assert work["live_prediction_forwards_completed"] == 4
    assert work["train_a_rollout"]["environment_steps_returned"] == 4
    assert evaluated["work"]["internal_projection_rows_returned"] == 4
    assert row["final_evaluation"]["status"] == "not_run"
