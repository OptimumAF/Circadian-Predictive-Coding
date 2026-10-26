"""One private observed-contact update sequence with fixed TRAIN rehearsal.

Per platform: one D12/H2/A4 mixed BP initializer/32 normal values; warm1 plus
correction8 plus reference4 =13 updates/26 rows. Zero fits, PC work, external
predictions or scenes.13 forwards/26 rows,13 backwards/494 gradient values
(3952B),13 normalizations/26 rows/312 coordinates;1872 derived dense MACs.
All six tests inspect one module-cached actual fixture; no gain criterion.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.app.continuous_stream_rehearsal import ContinuousReplayPlan
from src.core import continuous_bc as core
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.pick_cube_action_history import action_history_training_episode
from src.core.pick_cube_action_history_corrections import action_history_correction_view
from src.core.pick_cube_readiness_corrections import (
    OBSERVED_GRASP_PROVENANCE,
    ObservedGraspCorrectionBatch,
)
from src.core.pick_cube_readiness_teacher import OBSERVED_GRASP_DEFINITION
from test_observed_grasp_corrections import RECIPE, assert_state, batch, raw


def reference():
    """A complete original TRAIN trace whose inputs/targets differ from corrections."""
    actions = ((-0.25, 0.5, 0.0, -1.0), (0.0, 0.0, 0.25, 1.0))
    episode = ContinuousEpisode(
        "reference-A",
        "pick-cube",
        "train",
        batch().episode.spec,
        tuple(
            ContinuousTransition(
                i,
                raw(i + 2),
                action,
                float(i == 1),
                raw(i + 3),
                i == 1,
                False,
            )
            for i, action in enumerate(actions)
        ),
    )
    return action_history_training_episode(episode, RECIPE)


def forbidden(*args, **kwargs):
    pytest.fail("observed/replay composition unexpectedly fitted statistics")


@pytest.fixture(scope="module")
def completed():
    corrections, bank = batch(), reference()
    raw_evidence, bank_evidence = asdict(corrections), asdict(bank)
    view = action_history_correction_view(corrections, RECIPE)
    owner = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(
            view.model.episode.spec,
            hidden_dim=2,
            seed=17,
            binary_gripper_axis=3,
        )
    )
    warm_x = np.asarray([row.observation for row in bank.transitions], dtype=np.float64)
    warm_y = np.asarray([row.action for row in bank.transitions], dtype=np.float64)
    owner.train_batch(warm_x, warm_y)
    initial = owner.snapshot_state()
    assert any(np.any(value != 0.0) for value in initial.velocity)
    captured: dict[str, Any] = {"trains": []}
    runtime_factory = demo.ActorShadowRuntime
    helper_factory = demo.ContinuousStreamRehearsal
    original_train = bp.NumpyContinuousBCLearner.train_batch

    def runtime(*args, **options):
        value = runtime_factory(*args, **options)
        captured.update(runtime=value, wake=options["budget"])
        return value

    def helper(*args, **options):
        value = helper_factory(*args, **options)
        captured.update(helper=value, replay=options["replay_budget"], helper_options=options)
        return value

    def training(model, features, targets):
        before = model.snapshot_state()
        result = original_train(model, features, targets)
        after = model.snapshot_state()
        captured["trains"].append(
            dict(
                features=features.copy(),
                targets=targets.copy(),
                diagnostic=asdict(result),
                before=(before.completed_updates, before.training_examples),
                after=(after.completed_updates, after.training_examples),
            )
        )
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(demo, "ContinuousStreamRehearsal", helper)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", training)
        patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
        result = demo.run_continuous_correction_update(
            owner,
            corrections,
            now=3,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
            replay_plan=ContinuousReplayPlan(2, 8),
            reference_episodes=(bank,),
        )
    assert result.report["complete"] is True, result.report
    return dict(
        owner=owner,
        initial=initial,
        corrections=corrections,
        bank=bank,
        raw_evidence=raw_evidence,
        bank_evidence=bank_evidence,
        result=result,
        **captured,
    )


def test_should_preserve_source_actor_policy_and_full_candidate_momentum_across_twelve_updates(
    completed,
):
    result, initial, runtime = completed["result"], completed["initial"], completed["runtime"]
    assert result.candidate is not None
    final = result.candidate.snapshot_state()
    assert final.policy == initial.policy and final.policy.binary_gripper_axis == 3
    assert final.policy.observation_standardization is None
    assert (initial.completed_updates, initial.training_examples) == (1, 2)
    assert (final.completed_updates, final.training_examples) == (13, 26)
    assert_state(initial, completed["owner"].snapshot_state())
    assert_state(initial, runtime.actor.snapshot_state().state)
    published = runtime.candidate_snapshot()
    assert_state(final, published.state)
    assert published.wake_updates == 8
    assert published.consolidation_attempts == published.consolidations_completed == 4
    assert any(
        not np.array_equal(a, b) for a, b in zip(initial.velocity, final.velocity, strict=True)
    )


def test_should_use_arrived_observed_targets_and_original_reference_actions_in_exact_cadence(
    completed,
):
    corrections, bank = completed["corrections"], completed["bank"]
    assert type(corrections) is ObservedGraspCorrectionBatch
    assert corrections.observed_grasp == (True, False)
    model = action_history_correction_view(corrections, RECIPE).model
    with pytest.raises(ValueError, match="arrived"):
        model.training_pairs(now=2)
    wake_x, wake_y = model.training_pairs(now=3)
    bank_x = tuple(row.observation for row in bank.transitions)
    bank_y = tuple(row.action for row in bank.transitions)
    trains = completed["trains"]
    assert len(trains) == 12
    for index, actual in enumerate(trains):
        expected_x, expected_y = (bank_x, bank_y) if index % 3 == 2 else (wake_x, wake_y)
        np.testing.assert_array_equal(actual["features"], expected_x)
        np.testing.assert_array_equal(actual["targets"], expected_y)
        assert actual["before"] == (index + 1, 2 * (index + 1))
        assert actual["after"] == (index + 2, 2 * (index + 2))
        assert actual["diagnostic"]["definition"] == core.MIXED_OBJECTIVE
    assert wake_x[1][-5:] == (0.5, 0.0, 0.0, 0.5, 0.0)
    assert bank_x[1][-5:] == (-0.25, 0.5, 0.0, -1.0, 1.0 / 6.0)
    assert wake_x != bank_x and wake_y != bank_y
    assert [row.action[3] for row in corrections.episode.transitions] == [0.5, -0.5]
    assert [label.target[3] for label in corrections.labels] == [-1.0, 1.0]
    assert asdict(corrections) == completed["raw_evidence"]
    assert asdict(bank) == completed["bank_evidence"]


def test_should_keep_replay_exposure_and_wake_update_authority_separate_and_permanently_charged(
    completed,
):
    wake, replay, helper = completed["wake"], completed["replay"], completed["helper"]
    assert wake is not replay and wake.budget is not replay.budget
    assert wake.budget.max_training_updates == 8 and wake.budget.max_replay_examples is None
    assert replay.budget.max_training_updates is None and replay.budget.max_replay_examples == 8
    assert (wake.updates_completed, wake.replay_examples_completed) == (8, 0)
    assert (replay.updates_completed, replay.replay_examples_completed) == (0, 8)
    assert completed["helper_options"]["expected_definition"] == core.MIXED_OBJECTIVE
    assert completed["helper_options"]["fixed_training_episodes"] == (completed["bank"],)
    report = helper.report
    assert (
        report["planned_events"]
        == report["attempted_updates"]
        == report["returned_updates"]
        == report["committed_updates"]
        == 4
    )
    assert report["charged_exposures"] == report["observed_budget_exposures"] == 8
    assert report["wake_updates_observed"] == report["total_updates"] == 8
    assert report["stopped"] is False and report["charge_count_unknown"] is False
    for index, event in enumerate(report["events"]):
        assert event["wake_updates"] == 2 * (index + 1)
        assert event["event_id"] == f"continuous-replay-{2 * (index + 1)}"
        assert event["status"] == "complete" and event["charged_exposures"] == 2
        assert (
            event["attempted_updates"]
            == event["returned_updates"]
            == event["committed_updates"]
            == 1
        )
        assert event["native_updates_after"] == 4 + 3 * index
        assert event["native_training_examples_after"] == 8 + 6 * index
        assert event["consolidation_attempt_consumed"] is True
    # Reading detached accounting cannot renew either original allowance.
    observed_again = helper.report
    assert observed_again["charged_exposures"] == 8
    assert wake.updates_completed == wake.budget.max_training_updates == 8
    assert replay.remaining_replay_examples == 0


def test_should_bind_original_feedback_provenance_clocks_and_both_kinds_of_native_receipt(
    completed,
):
    report, runtime = completed["result"].report, completed["runtime"]
    assert report["provenance"] == OBSERVED_GRASP_PROVENANCE
    assert report["teacher_definition"] == OBSERVED_GRASP_DEFINITION
    assert report["observed_grasp"] == [True, False]
    assert (
        report["binary_gripper_axis"] == 3 and report["training_objective"] == core.MIXED_OBJECTIVE
    )
    assert report["observed_at"] == 2 and report["labels_arrived_at"] == 3
    assert report["progress"]["updates_completed"] == 8
    assert report["progress"]["replay_examples_completed"] == 0
    assert report["progress"]["next_tick"] == 4
    assert report["applied_updates"] == [asdict(row) for row in runtime.applied_updates]
    assert report["applied_consolidations"] == [
        asdict(row) for row in runtime.applied_consolidations
    ]
    assert len(report["applied_updates"]) == 8 and len(report["applied_consolidations"]) == 4
    for index, receipt in enumerate(report["applied_updates"]):
        assert receipt["episode_id"] == "pick-cube-smoke-19"
        assert receipt["sample_id"] == f"corrective-batch-{index}"
        assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
        assert receipt["update_number"] == index + 1
        assert receipt["actor_version"] == "actor-0" and receipt["learner_version"] == "candidate-0"
        assert receipt["diagnostic"] == completed["trains"][index + index // 2]["diagnostic"]
    for index, receipt in enumerate(report["applied_consolidations"]):
        assert receipt["attempt_number"] == index + 1
        assert receipt["event_id"] == f"continuous-replay-{2 * (index + 1)}"
        assert receipt["diagnostic"] == completed["trains"][3 * index + 2]["diagnostic"]
    json.dumps(report, allow_nan=False)


def test_should_report_all_private_gradient_exposure_and_retained_memory_without_a_refit(completed):
    report = completed["result"].report
    work = report["work"]
    expected = dict(
        optimizer_updates=12,
        training_rows=24,
        unique_correction_rows=2,
        repeated_training_rows=14,
        reference_replay_rows=8,
        fit_calls=0,
        forward_passes=12,
        forward_rows=24,
        backward_passes=12,
        backward_rows=24,
        normalization_calls=12,
        normalization_rows=24,
        normalization_values=288,
        gradient_values=456,
        gradient_numeric_bytes=3648,
        dense_multiply_accumulates=1728,
        binary_gripper_loss_rows=24,
        model_numeric_bytes_per_owner=608,
        statistics_numeric_bytes_per_owner=0,
        named_model_owners=4,
        inbox_feature_target_numeric_bytes=2048,
        reference_bank_retained_rows=2,
        reference_bank_retained_numeric_bytes=464,
        replay_transient_model_owners=1,
        replay_transient_model_numeric_bytes_per_owner=608,
    )
    assert {name: work[name] for name in expected} == expected
    assert report["replay"]["bank_rows"] == 2
    assert report["replay"]["bank_episode_ids"] == ["reference-A"]
    assert report["replay"]["bank_contexts"] == ["pick-cube"]
    assert report["replay"]["retained_numeric_bytes"] == 464
    assert report["settings"]["reference_bank_rows"] == 2
    assert (
        report["settings"]["reference_replay_rows"]
        == report["settings"]["max_reference_replay_rows"]
        == 8
    )
    assert report["settings"]["planned_replay_events"] == 4
    assert report["settings"]["replay_plan"] == asdict(ContinuousReplayPlan(2, 8))
    projection = report["input_projection"]
    assert projection["projection"] == asdict(RECIPE)
    assert projection["source_spec"]["observation_dim"] == 42
    assert projection["model_spec"]["observation_dim"] == 12
    assert projection["source_episode_id"] == "pick-cube-smoke-19"


def test_should_detach_observed_reports_and_snapshots_without_erasing_paid_replay_history(
    completed,
):
    result, helper = completed["result"], completed["helper"]
    before = deepcopy(result.report)
    detached = helper.report
    detached["events"][0]["charged_exposures"] = 0
    detached["original_limits"]["replay"]["max_replay_examples"] = 1000
    snapshot = result.candidate.snapshot_state()
    snapshot.parameters[0].flat[0] += 10.0
    snapshot.velocity[0].flat[0] -= 10.0
    assert helper.report["events"][0]["charged_exposures"] == 2
    assert helper.report["original_limits"]["replay"]["max_replay_examples"] == 8
    assert result.report == before
    assert_state(result.candidate.snapshot_state(), completed["runtime"].candidate_snapshot().state)
    assert_state(completed["initial"], completed["owner"].snapshot_state())
    assert completed["replay"].replay_examples_completed == 8
