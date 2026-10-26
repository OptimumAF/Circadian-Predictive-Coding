"""Complete existing public joint TRAIN/probe composition with synthetic sources.

All actual native work is engineering, with no teacher/game/final evaluation.
The root's bounded observer charges weighted and ordinary updates separately.
"""

from copy import deepcopy

import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_weighted_catalogue_route as route
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.adapters.torch_feature_learner import TORCH_CE_ID, TORCH_PC_ENERGY_ID
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_run_weighted_joint_and_probes_with_exact_checkpoint_diagnostics(
    monkeypatch, method
):
    raw = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    reserved = raw_probe_data()
    arms, templates, calls = [], [], []
    train_arm, make_probe = pilot.shared._train_arm, route.probe_arm
    ordinary_train = InvariantTorchFeatureLearner.train_batch

    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic API gate forbids teacher/game physics")

    def arm(*args, **kwargs):
        assert kwargs["catalogue_rows"] == 54
        result = train_arm(*args, **kwargs)
        arms.append(result)
        return result

    def template(original, **kwargs):
        result = make_probe(original, **kwargs)
        assert type(result["source"]) is InvariantTorchFeatureLearner
        assert type(original["source"]) is TorchTrainWeightedLearner
        before = deepcopy(result["source"].snapshot_state())
        assert _same_state(before, original["source"].snapshot_state())
        templates.append((result["source"], before))
        return result

    def unweighted(owner, features, targets):
        result = ordinary_train(owner, features, targets)
        calls.append((owner, len(features), result.definition))
        return result

    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda boundary, reports: raw)
    monkeypatch.setattr(
        pilot, "_collect_probe_sources", lambda boundary, reports, **kwargs: reserved
    )
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", arm)
    monkeypatch.setattr(route, "probe_arm", template)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "train_batch", unweighted)
    report = pilot.run_pilot(
        public_history=True,
        invariant_history=True,
        qualification_method=method,
        joint_catalogue=True,
        weighted_catalogue=True,
        pc_update_diagnostics=True,
        distinct_probe=True,
        old_task_readouts=True,
        return_to_a=True,
    )
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["method"] == f"public_invariant_{method}"
    assert row["charged_updates"] == row["stream_result"]["completed_updates"] == 128
    assert row["native_training_examples"] == row["stream_result"]["training_examples"] == 6912
    assert row["partial_stream"]["committed_training_examples"] == 6912
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert "signed_metrics" not in row and "acquisition_conditions" not in row
    assert type(row["joint_catalogue_acquired"]) is bool
    assert [value["updates"] for value in row["readouts"]] == [0, 32, 64, 96, 128]
    for value in row["readouts"]:
        assert value["status"] == "complete" and len(value["scores"]) == 6
        assert value["expected_training_examples"] == 54 * value["updates"]
    for value in row["readouts"]:
        if value["updates"] not in (0, 128):
            assert "pc_update_diagnostics" not in value
            assert "pc_update_diagnostics" not in value["work"]
            continue
        records = value["pc_update_diagnostics"]
        assert len(records) == 1 and records[0]["context"] == "joint_train_catalogue"
        diagnostic = records[0]["diagnostic"]
        assert diagnostic["definition"] == "torch_invariant_weighted_pc_realized_update_v1"
        assert diagnostic["status"] == "complete" and diagnostic["method"] == method
        assert diagnostic["rows"] == 54 and diagnostic["relaxation"] == []
        assert diagnostic["source_episode_ids"] == records[0]["episode_ids"]
        assert len(diagnostic["class_weights"]) == 7
        assert (
            diagnostic["proposal"]["weighted_energy"]
            == diagnostic["fixed_state_weighted_energy_before"]
        )
        extra = value["work"]["pc_update_diagnostics"]
        assert extra["catalogue_calls_attempted"] == extra["catalogue_calls_completed"] == 1
        assert extra["projection_rows_completed"] == extra["train_rows_admitted"] == 54
        assert (
            extra["prior_forward_rows_completed"] == 108
            and extra["prior_backward_rows_completed"] == 54
        )
        assert extra["relaxation_forward_rows_completed"] == 0
        for key in ("proposal_calls", "fixed_state_energy_calls", "weight_derivations"):
            assert extra[key + "_attempted"] == extra[key + "_completed"] == 1
        assert (
            extra["proposal_rows_completed"]
            == extra["fixed_state_energy_rows_completed"]
            == extra["weight_rows_completed"]
            == 54
        )
        assert extra["proposal_nominal_latent_row_iterations_completed"] == 108
        assert extra["geometry_reductions_completed"] == 5
        assert (
            extra["native_train_calls"]
            == extra["optimizer_steps"]
            == extra["published_updates"]
            == 0
        )
    extra = report["work"]["pc_update_diagnostics"]
    assert row["work"]["pc_update_diagnostics"] == extra
    assert extra["catalogue_calls_attempted"] == extra["catalogue_calls_completed"] == 2
    assert extra["train_rows_admitted"] == extra["projection_rows_completed"] == 108
    assert (
        extra["prior_forward_rows_completed"] == 216
        and extra["prior_backward_rows_completed"] == 108
    )
    assert (
        extra["proposal_rows_completed"]
        == extra["fixed_state_energy_rows_completed"]
        == extra["weight_rows_completed"]
        == 108
    )
    assert extra["proposal_nominal_latent_row_iterations_completed"] == 216
    assert (
        extra["geometry_reductions_completed"] == 10
        and extra["failure_work_may_be_unobserved"] is False
    )
    for slot in row["probes"]:
        assert slot["status"] == "complete", slot["reason"]
        assert slot["charged_updates"] == slot["return_charged_updates"] == 16
        assert slot["expected_training_examples"] == 54 * slot["checkpoint_updates"]
        assert slot["source_unchanged"] is slot["state_unchanged"] is True
    assert len(arms) == 1 and len(templates) == 2 and len(calls) == 64
    ordinary_id = TORCH_CE_ID if method == "bp" else TORCH_PC_ENERGY_ID
    assert all(rows == 9 and definition == ordinary_id for _, rows, definition in calls)
    assert all(owner is not source for owner, _, _ in calls for source, _ in templates)
    for source, before in templates:
        assert _same_state(source.snapshot_state(), before)
        assert before.completed_updates == before.training_examples == 0
    arms[0]["guard"]()
    route.check_receipts(arms[0], method=method)
    initial = arms[0]["source"].snapshot_state()
    assert initial.completed_updates == initial.training_examples == 0
    assert _same_state(initial, arms[0]["stream"]._runtime.actor.snapshot_state().state)
    assert report["catalogue_inputs"]["row_exposures"] == 6912
    assert report["catalogue_inputs"]["input_array_bytes"] == 54 * 297 * 8 * 128
    support = report["probe_data"]["support"]
    assert support["work"]["canonical_projection_rows"] == 216
    assert support["all_canonical_inputs_unseen"] is True
    assert all(
        value["canonical"]["unmatched_input_rows"] == 9 for value in support["probe"]["packets"]
    )
    assert report["work"]["native_train_calls"] == 192
    assert report["work"]["training_rows"] == 7488
    assert report["work"]["total_score_forwards_completed"] == 94
    assert report["work"]["total_score_rows_completed"] == 846
    assert row["final_evaluation"]["status"] == "not_run"
    assert report["scientific_advantage_verified"] is False and report["comparisons"] is None
