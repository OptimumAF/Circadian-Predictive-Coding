"""Arrived head updates plus independently charged fixed TRAIN rehearsal.

One new D3/H2/A4 initializer14 draws/warm SGD2rows. Three cached continuations:
27 GN attempts45rows,26 publications44rows,104 axis solves488 augmentedrows.
No author execution; stage ledger separately includes the old21-case cohort.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.app.continuous_stream_rehearsal import ContinuousReplayPlan
from src.core import continuous_bc as task
from src.core import continuous_head_gauss_newton as head
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_corrections import PROVENANCE
from test_continuous_head_corrections import assert_state, batch


def reference():
    return ContinuousEpisode(
        "gn-reference-A",
        "pick-cube",
        "train",
        batch().episode.spec,
        (
            ContinuousTransition(
                0, (-1.0, 0.0, 0.0), (0.25,) * 4, 1.0, (-0.5, 0.0, 0.0), True, False
            ),
        ),
    )


@pytest.fixture(scope="module")
def completed():
    corrections, bank = batch(), reference()
    raw, original_bank = asdict(corrections), asdict(bank)
    owner = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(
            corrections.episode.spec,
            hidden_dim=2,
            seed=17,
            observation_standardization=ObservationStandardization((0.0,) * 3, (1.0,) * 3, 2),
        )
    )
    features = np.asarray(
        [row.observation for row in corrections.episode.transitions], dtype=np.float64
    )
    owner.train_batch(features, np.full((2, 4), 0.25))
    initial = owner.snapshot_state()
    factory, helper_factory = demo.ActorShadowRuntime, demo.ContinuousStreamRehearsal
    train_method, propose = (
        bp.NumpyContinuousBCLearner.train_head_gauss_newton_batch,
        head.propose_head_gauss_newton,
    )
    record_work, clock = demo._record_head_work, demo.monotonic
    runs = {}
    for mode in ("success", "failed_replay", "late"):
        source = owner.fork()
        captured: dict[str, Any] = dict(trains=[], proposals=0, armed=False, mutations=0)

        def runtime(model, **options):
            value = factory(model, **options)
            captured.update(runtime=value, wake=options["budget"], source_facade=model)
            return value

        def helper(*args, **options):
            value = helper_factory(*args, **options)
            captured.update(helper=value, replay=options["replay_budget"], helper_options=options)
            return value

        def training(model, x, y):
            row = dict(features=x.copy(), targets=y.copy(), before=model.snapshot_state())
            captured["trains"].append(row)
            returned = train_method(model, x, y)
            row["after"] = model.snapshot_state()
            return returned

        def proposing(*args):
            captured["proposals"] += 1
            if mode == "failed_replay" and captured["proposals"] == 3:
                raise RuntimeError("first replay proposal stopped before core")
            return propose(*args)

        def work(*args, **options):
            returned = record_work(*args, **options)
            if mode == "late":
                captured["armed"] = True
            return returned

        def timer():
            returned = clock()
            if captured["armed"] and captured["mutations"] == 0:
                candidate = captured["runtime"]._candidate._learner
                captured["mutated_owner"] = candidate
                captured["before_corruption"] = candidate.snapshot_state()
                candidate._state.parameters[3][0] += 0.125
                captured["mutations"] += 1
            return returned

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(demo, "ActorShadowRuntime", runtime)
            patch.setattr(demo, "ContinuousStreamRehearsal", helper)
            patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", training)
            patch.setattr(head, "propose_head_gauss_newton", proposing)
            patch.setattr(demo, "_record_head_work", work)
            patch.setattr(demo, "monotonic", timer)
            result = demo.run_continuous_correction_update(
                source,
                corrections,
                now=3,
                head_gauss_newton=True,
                replay_plan=ContinuousReplayPlan(2, 4),
                reference_episodes=(bank,),
            )
        assert_state(initial, source.snapshot_state())
        assert (asdict(corrections), asdict(bank)) == (raw, original_bank)
        runs[mode] = result, captured
    return owner, corrections, bank, initial, runs


def test_should_continue_all_owned_state_with_eight_wakes_and_four_one_row_replays(completed):
    owner, _, _, initial, runs = completed
    result, captured = runs["success"]
    assert result.report["complete"] is True and result.candidate is not None
    final = result.candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (13, 22)
    assert final.policy == initial.policy
    for index in (0, 1):
        np.testing.assert_array_equal(final.parameters[index], initial.parameters[index])
        np.testing.assert_array_equal(final.velocity[index], initial.velocity[index])
    for index in (2, 3):
        np.testing.assert_array_equal(final.velocity[index], np.zeros_like(final.velocity[index]))
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, captured["runtime"].actor.snapshot_state().state)
    assert_state(final, captured["runtime"].candidate_snapshot().state)
    assert captured["wake"] is not captured["replay"]
    assert (
        captured["wake"].updates_completed == 8 and captured["wake"].replay_examples_completed == 0
    )
    assert (
        captured["replay"].updates_completed == 0
        and captured["replay"].replay_examples_completed == 4
    )
    assert captured["wake"].budget.max_training_updates == 8
    assert captured["replay"].budget.max_replay_examples == 4


def test_should_use_distinct_wake_and_reference_labels_in_the_actual_native_order(completed):
    _, corrections, bank, _, runs = completed
    result, captured = runs["success"]
    wake = corrections.training_pairs(now=3)
    replay = ((bank.transitions[0].observation,), (bank.transitions[0].action,))
    assert len(captured["trains"]) == 12
    cumulative = 2
    for index, row in enumerate(captured["trains"]):
        x, y = replay if index % 3 == 2 else wake
        np.testing.assert_array_equal(row["features"], x)
        np.testing.assert_array_equal(row["targets"], y)
        assert row["before"].completed_updates == index + 1
        assert row["before"].training_examples == cumulative
        cumulative += len(x)
        assert row["after"].training_examples == cumulative
    assert cumulative == 22
    assert [item["training_rows"] for item in result.report["native_updates"]] == [2, 2, 1] * 4
    assert [row["step"] for row in asdict(corrections)["labels"]] == [0, 1]


def test_should_preserve_original_label_clocks_and_separate_permanent_receipts(completed):
    _, corrections, bank, _, runs = completed
    result, captured = runs["success"]
    report = result.report
    assert report["provenance"] == PROVENANCE
    assert len(report["applied_updates"]) == 8 and len(report["applied_consolidations"]) == 4
    for event, receipt in zip(report["events"], report["applied_updates"], strict=True):
        assert event["success_validated"] is True and event["applied"] == [receipt]
        assert receipt["episode_id"] == corrections.episode.episode_id
        assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
        assert receipt["diagnostic"]["definition"] == task.OBJECTIVE
    replay = report["replay"]
    assert replay["bank_rows"] == 1 and replay["bank_episode_ids"] == [bank.episode_id]
    assert replay["charged_exposures"] == replay["observed_budget_exposures"] == 4
    assert (
        replay["attempted_updates"]
        == replay["returned_updates"]
        == replay["committed_updates"]
        == 4
    )
    assert replay["stopped"] is False and replay["charge_count_unknown"] is False
    assert report["progress"]["updates_completed"] == 8
    assert report["progress"]["replay_examples_completed"] == 0
    assert len(captured["runtime"].applied_updates) == 8
    assert len(captured["runtime"].applied_consolidations) == 4


def test_should_report_all_native_solver_rows_and_only_wake_repetition(completed):
    _, _, _, _, runs = completed
    result, _ = runs["success"]
    report = result.report
    assert report["protocol_id"] == "continuous_head_gauss_newton_reference_correction_8_v1"
    assert len(report["native_updates"]) == 12
    assert report["native_updates"] == report["sealed_native_updates"]
    work = report["work"]
    assert work["optimizer_updates"] == work["head_gauss_newton_updates"] == 12
    assert work["training_rows"] == 20 and work["reference_replay_rows"] == 4
    assert work["repeated_training_rows"] == 14 and work["unique_correction_rows"] == 2
    assert work["sgd_updates"] == work["training_gradient_evaluations"] == work["fit_calls"] == 0
    assert work["native_update_counts"]["published_updates"] == 12
    assert work["native_event_row_weights"]["published_updates"] == 20
    assert work["native_update_counts"]["solve_completed"] == 48
    assert work["native_update_counts"]["solve_rows"] == 224
    assert work["native_event_row_weights"]["solve_rows"] == 384
    assert work["normalization_calls"] == 12 and work["normalization_rows"] == 20
    json.dumps(report, allow_nan=False)


def test_should_keep_charged_replay_failure_after_two_successful_wakes_and_stop(completed):
    _, _, _, initial, runs = completed
    result, captured = runs["failed_replay"]
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert "first replay proposal" in report["error"]
    assert captured["proposals"] == 3 and len(captured["trains"]) == 3
    assert [len(row["features"]) for row in captured["trains"]] == [2, 2, 1]
    assert "after" not in captured["trains"][-1]
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 2
    assert len(report["applied_consolidations"]) == 0
    assert captured["wake"].updates_completed == 2
    assert captured["replay"].replay_examples_completed == 1
    assert report["replay"]["charged_exposures"] == 1
    assert report["replay"]["attempted_updates"] == 1
    assert report["replay"]["returned_updates"] == report["replay"]["committed_updates"] == 0
    assert report["replay"]["stopped"] is True
    failed = report["replay"]["events"][0]["native_failure_work"]
    assert failed["normalization_completed"] == 1 and failed["normalization_rows"] == 1
    assert failed["published_updates"] == 0
    assert_state(initial, captured["runtime"].actor.snapshot_state().state)


def test_should_refuse_final_timer_corruption_without_erasing_twelve_paid_updates(completed):
    _, _, _, initial, runs = completed
    result, captured = runs["late"]
    report = result.report
    assert captured["armed"] is True and captured["mutations"] == 1
    assert report["complete"] is False and result.candidate is None
    assert len(report["native_updates"]) == 12 and len(report["applied_updates"]) == 8
    assert len(report["applied_consolidations"]) == 4
    assert report["replay"]["charged_exposures"] == 4
    assert (
        captured["wake"].updates_completed == 8
        and captured["replay"].replay_examples_completed == 4
    )
    before = captured["before_corruption"]
    after = captured["mutated_owner"].snapshot_state()
    assert after.completed_updates == before.completed_updates == 13
    assert after.parameters[3][0] == before.parameters[3][0] + 0.125
    assert_state(initial, captured["runtime"].actor.snapshot_state().state)


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "list",
        "final_test",
        "probe_train",
        "probe_test",
        "spec",
        "duplicate",
        "incomplete",
        "too_many",
        "too_large",
        "binary",
        "diagnostics",
    ],
)
def test_should_refuse_invalid_bank_or_selected_mode_before_any_private_fork(
    completed, monkeypatch, fault
):
    owner, corrections, bank, _, _ = completed
    source = owner.fork()
    references: Any = (bank,)
    options: dict[str, Any] = dict(
        head_gauss_newton=True, now=3, replay_plan=ContinuousReplayPlan(2, 4)
    )
    if fault == "empty":
        references = ()
    elif fault == "list":
        references = [bank]
    elif fault in ("final_test", "probe_train", "probe_test"):
        references = (replace(bank, role=fault),)
    elif fault == "spec":
        references = (
            replace(
                bank, spec=replace(bank.spec, action_units=("other", *bank.spec.action_units[1:]))
            ),
        )
    elif fault == "duplicate":
        references = (bank, bank)
    elif fault == "incomplete":
        damaged = deepcopy(bank)
        object.__setattr__(damaged.transitions[0], "terminated", False)
        references = (damaged,)
    elif fault == "too_many":
        references = tuple(replace(bank, episode_id=f"reference-{i}") for i in range(5))
    elif fault == "too_large":
        row = bank.transitions[0]
        references = (
            replace(
                bank,
                transitions=tuple(
                    replace(
                        row,
                        step=i,
                        observation=(-1.0, 0.0, 0.0),
                        next_observation=(-1.0, 0.0, 0.0),
                        terminated=i == 128,
                    )
                    for i in range(129)
                ),
            ),
        )
    elif fault == "binary":
        options["binary_gripper_axis"] = 3
    else:
        options["diagnostic_stage_bindings"] = object()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid replay admission reached fork/runtime/native training")

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(demo, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            source, corrections, reference_episodes=references, **options
        )


def test_should_pass_the_selected_reference_only_to_the_private_driver_update(
    completed, monkeypatch, tmp_path
):
    owner, corrections, bank, initial, runs = completed
    cached, _ = runs["success"]
    calls = []

    def rollout(output, **options):
        calls.append(("rollout", options))
        seed = options["train_seed"]
        record = (
            corrections.episode
            if seed == 19
            else replace(corrections.episode, episode_id="pick-cube-smoke-17", context="pick-cube")
        )
        report = dict(
            complete=True,
            policy="frozen_continuous_bc",
            episode=asdict(record),
            learner_policy=asdict(options["learner"].policy),
        )
        if options.get("corrective_queries"):
            report["corrections"] = dict(
                provenance=PROVENANCE,
                teacher_schedule_steps=2,
                labels=[
                    dict(
                        asdict(label),
                        observed_at=label.observed_at,
                        labels_arrived_at=label.labels_arrived_at,
                    )
                    for label in corrections.labels
                ],
            )
        return smoke.FrozenRolloutResult(
            report, corrections if options.get("corrective_queries") else None
        )

    def update(source, arrived, **options):
        calls.append(("update", options))
        assert source is owner and arrived == corrections
        return replace(cached, report=deepcopy(cached.report))

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", update)
    plan = ContinuousReplayPlan(2, 4)
    result = smoke.run_corrective_adaptation(
        owner,
        output_directory=tmp_path / "driver",
        visual_observation=False,
        head_gauss_newton=True,
        replay_plan=plan,
        reference_episodes=(bank,),
    )
    assert result["complete"] is True
    assert [kind for kind, _ in calls] == ["rollout", "update", "rollout", "rollout"]
    assert calls[1][1] == dict(
        now=3, head_gauss_newton=True, replay_plan=plan, reference_episodes=(bank,)
    )
    for kind, options in calls:
        if kind == "rollout":
            assert not {"head_gauss_newton", "replay_plan", "reference_episodes"} & options.keys()
    assert (
        result["update"]["protocol_id"] == "continuous_head_gauss_newton_reference_correction_8_v1"
    )
    assert_state(initial, owner.snapshot_state())


@pytest.mark.parametrize("field", ["replay", "consolidations"])
def test_should_refuse_a_late_rewrite_of_cached_paid_reference_evidence(completed, field):
    from src.adapters.continuous_head_corrections import require_replay_evidence

    report = deepcopy(completed[-1]["success"][0].report)
    expected_replay = json.dumps(report["replay"], sort_keys=True, allow_nan=False)
    expected_receipts = deepcopy(report["applied_consolidations"])
    require_replay_evidence(report, expected_replay, expected_receipts)
    if field == "replay":
        report["replay"]["charged_exposures"] += 1
    else:
        report["applied_consolidations"][0]["attempt_number"] += 1
    with pytest.raises(RuntimeError, match="reference report or receipts changed"):
        require_replay_evidence(report, expected_replay, expected_receipts)
