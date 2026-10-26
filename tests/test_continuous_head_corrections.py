"""Explicit head updates on arrived labels, preserving frozen continuation.

One D3/H2/A4 initializer14 draws, one warm SGD plus eight default SGD updates.
Eighteen GN attempts/36 rows,17 returned proposals/68 solves/340 augmented rows.
Four RNG-free continuation runs; all driver cases are cached-result stubs.
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
from src.core import continuous_bc as task
from src.core import continuous_head_gauss_newton as head
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_corrections import (
    PROVENANCE,
    PickCubeCorrectionBatch,
    PickCubeCorrectionLabel,
)


def batch():
    declared = ContinuousActionSpec(
        3,
        ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper"),
        (-1.0,) * 4,
        (1.0,) * 4,
    )
    rows = tuple(
        ContinuousTransition(
            i,
            (float(i), 0.0, 0.0),
            (0.125, -0.25, 0.5, -0.5),
            float(i == 1),
            (float(i + 1), 0.0, 0.0),
            i == 1,
            False,
        )
        for i in range(2)
    )
    episode = ContinuousEpisode("pick-cube-smoke-19", "pick-cube-train-19", "train", declared, rows)
    return PickCubeCorrectionBatch(
        19,
        episode,
        tuple(
            PickCubeCorrectionLabel(i, row.observation, (1.0,) * 4, "align")
            for i, row in enumerate(rows)
        ),
    )


def assert_state(first, second):
    assert first.policy == second.policy
    assert (first.completed_updates, first.training_examples) == (
        second.completed_updates,
        second.training_examples,
    )
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def completed():
    corrections = batch()
    owner = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(
            corrections.episode.spec,
            hidden_dim=2,
            seed=17,
            observation_standardization=ObservationStandardization((0.0,) * 3, (1.0,) * 3, 2),
        )
    )
    x = np.asarray([row.observation for row in corrections.episode.transitions], dtype=np.float64)
    owner.train_batch(x, np.full((2, 4), 0.25))
    initial = owner.snapshot_state()
    assert any(np.any(v != 0.0) for v in initial.velocity)
    original = asdict(corrections)
    result_by_mode = {}
    factory, proposal = demo.ActorShadowRuntime, head.propose_head_gauss_newton
    native_train = bp.NumpyContinuousBCLearner.train_head_gauss_newton_batch
    record_work, clock = demo._record_head_work, demo.monotonic
    for mode in ("head", "default", "failed", "late"):
        source = owner.fork()
        capture: dict[str, Any] = dict(trains=[], proposals=0, armed=False, mutations=0)

        def runtime(model, **options):
            made = factory(model, **options)
            capture.update(source_facade=model, runtime=made, budget=options["budget"])
            return made

        def propose(*args):
            capture["proposals"] += 1
            if mode == "failed" and capture["proposals"] == 2:
                raise RuntimeError("second correction proposal refused before core")
            return proposal(*args)

        def train(model, features, targets):
            before = model.snapshot_state()
            returned = native_train(model, features, targets)
            capture["trains"].append(
                dict(
                    features=features.copy(),
                    targets=targets.copy(),
                    before=before,
                    after=model.snapshot_state(),
                )
            )
            return returned

        def work(*args):
            returned = record_work(*args)
            if mode == "late":
                capture["armed"] = True
            return returned

        def timer():
            returned = clock()
            if capture["armed"] and capture["mutations"] == 0:
                candidate = capture["runtime"]._candidate._learner
                capture["mutated_owner"] = candidate
                capture["before_corruption"] = candidate.snapshot_state()
                candidate._state.parameters[3][0] += 0.125
                capture["mutations"] += 1
            return returned

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(demo, "ActorShadowRuntime", runtime)
            patch.setattr(head, "propose_head_gauss_newton", propose)
            patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", train)
            patch.setattr(demo, "_record_head_work", work)
            patch.setattr(demo, "monotonic", timer)
            options: dict[str, Any] = {} if mode == "default" else {"head_gauss_newton": True}
            result = demo.run_continuous_correction_update(source, corrections, now=3, **options)
        assert_state(source.snapshot_state(), initial)
        assert asdict(corrections) == original
        result_by_mode[mode] = result, capture
    return owner, corrections, initial, result_by_mode


def test_should_continue_the_full_warmed_source_with_only_head_parameters_changed(completed):
    owner, _, initial, runs = completed
    result, capture = runs["head"]
    assert (
        result.report["complete"] is True and type(result.candidate) is bp.NumpyContinuousBCLearner
    )
    final = result.candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (9, 18)
    assert final.policy == initial.policy
    for index in (0, 1):
        np.testing.assert_array_equal(final.parameters[index], initial.parameters[index])
        np.testing.assert_array_equal(final.velocity[index], initial.velocity[index])
    assert any(not np.array_equal(final.parameters[i], initial.parameters[i]) for i in (2, 3))
    for index in (2, 3):
        np.testing.assert_array_equal(final.velocity[index], np.zeros_like(final.velocity[index]))
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, capture["runtime"].actor.snapshot_state().state)
    assert_state(final, capture["runtime"].candidate_snapshot().state)
    assert result.candidate is not owner and result.candidate is not capture["runtime"]._candidate


def test_should_train_on_arrived_targets_eight_times_without_relabeling_executed_actions(completed):
    _, corrections, _, runs = completed
    result, capture = runs["head"]
    observations, targets = corrections.training_pairs(now=3)
    assert len(capture["trains"]) == 8
    for index, row in enumerate(capture["trains"]):
        np.testing.assert_array_equal(row["features"], observations)
        np.testing.assert_array_equal(row["targets"], targets)
        assert not np.array_equal(
            row["targets"], [v.action for v in corrections.episode.transitions]
        )
        assert row["before"].completed_updates == index + 1
        assert row["after"].completed_updates == index + 2
    report = result.report
    assert report["provenance"] == PROVENANCE
    assert report["source_updates"] == 1 and report["source_training_rows"] == 2
    for index, (event, receipt) in enumerate(
        zip(report["events"], report["applied_updates"], strict=True)
    ):
        assert event["success_validated"] is True and event["applied"] == [receipt]
        assert receipt["sample_id"] == f"corrective-batch-{index}"
        assert receipt["event_id"] == f"corrective-label-{index}"
        assert receipt["episode_id"] == corrections.episode.episode_id
        assert (receipt["observed_at"], receipt["arrived_at"]) == (2, 3)
        assert receipt["applied_at"] >= 3 and receipt["update_number"] == index + 1
        assert receipt["diagnostic"]["definition"] == task.OBJECTIVE
    assert capture["budget"].updates_completed == 8
    assert capture["budget"].replay_examples_completed == 0


def test_should_report_actual_head_solver_work_without_sgd_attribution(completed):
    _, _, _, runs = completed
    result, _ = runs["head"]
    report = result.report
    assert report["protocol_id"] == "continuous_head_gauss_newton_correction_8_v1"
    assert report["head_gauss_newton"]["definition"] == "continuous_bp_head_gauss_newton_v1"
    assert len(report["native_updates"]) == 8
    for native, receipt in zip(report["native_updates"], report["applied_updates"], strict=True):
        assert native["diagnostic"] == receipt["diagnostic"] and native["training_rows"] == 2
        assert native["loss_after"] < native["diagnostic"]["value"]
        assert native["work"]["solve_completed"] == 4
        assert native["work"]["solve_rows"] == 20
        assert native["work"]["published_updates"] == 1
    work = report["work"]
    assert work["optimizer_updates"] == work["head_gauss_newton_updates"] == 8
    assert work["sgd_updates"] == work["training_gradient_evaluations"] == 0
    assert work["training_rows"] == 16 and work["reference_replay_rows"] == work["fit_calls"] == 0
    assert work["native_update_counts"]["solve_completed"] == 32
    assert work["native_update_counts"]["solve_rows"] == 160
    assert work["native_event_row_weights"]["solve_rows"] == 320
    json.dumps(report, allow_nan=False)


def test_should_keep_real_implicit_default_sgd_and_its_original_report_schema(completed):
    _, _, initial, runs = completed
    result, capture = runs["default"]
    assert result.report["complete"] is True and result.candidate is not None
    assert result.report["protocol_id"] == "continuous_correction_8_v1"
    assert "head_gauss_newton" not in result.report and "native_updates" not in result.report
    assert capture["proposals"] == 0 and capture["trains"] == []
    final = result.candidate.snapshot_state()
    assert final.completed_updates == initial.completed_updates + 8
    assert final.training_examples == initial.training_examples + 16
    assert result.report["work"]["forward_passes"] == result.report["work"]["backward_passes"] == 8


def test_should_keep_the_first_native_receipt_when_the_second_proposal_fails(completed):
    _, _, _, runs = completed
    result, capture = runs["failed"]
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert "second correction proposal" in report["error"]
    assert capture["proposals"] == 2 and capture["budget"].updates_completed == 1
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert report["events"][0]["success_validated"] is True
    assert len(report["events"]) == 2
    assert report["failed_native_work"]["normalization_completed"] == 1
    assert report["failed_native_work"]["published_updates"] == 0
    assert report["native_updates"] == report["sealed_native_updates"]


def test_should_refuse_late_candidate_corruption_after_all_eight_paid_updates(completed):
    _, _, _, runs = completed
    result, capture = runs["late"]
    report = result.report
    assert capture["armed"] is True and capture["mutations"] == 1
    assert report["complete"] is False and result.candidate is None
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 8
    assert capture["budget"].updates_completed == 8
    assert all(event["success_validated"] is True for event in report["events"])
    after = capture["mutated_owner"].snapshot_state()
    before = capture["before_corruption"]
    assert after.parameters[3][0] == before.parameters[3][0] + 0.125
    assert after.completed_updates == before.completed_updates == 9


def test_should_refuse_prior_native_work_or_identity_changes_before_another_train_call(
    completed, monkeypatch
):
    _, _, _, runs = completed
    result, capture = runs["head"]
    facade = capture["source_facade"]
    sink = result.report["native_updates"]
    original = sink[0]

    def forbidden(*args, **kwargs):
        pytest.fail("corrupted native prefix reached another head method")

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
    try:
        original["work"]["solve_rows"] += 1
        with pytest.raises(RuntimeError, match="returned prefix changed"):
            facade.train_batch(np.zeros((2, 3)), np.ones((2, 4)))
    finally:
        original["work"]["solve_rows"] -= 1
    try:
        sink[0] = deepcopy(original)
        with pytest.raises(RuntimeError, match="returned prefix changed"):
            facade.require_reports(sink)
    finally:
        sink[0] = original
    with pytest.raises(RuntimeError, match="list was replaced"):
        facade.require_reports(list(sink))
    facade.require_reports(sink)


@pytest.mark.parametrize(
    "fault",
    ["unarrived", "bad_flag", "role", "mixed_policy", "binary", "replay", "diagnostics", "foreign"],
)
def test_should_refuse_invalid_selected_admission_before_fork_or_update(
    completed, monkeypatch, fault
):
    owner, corrections, _, _ = completed
    source: Any = owner.fork()
    options: dict[str, Any] = dict(head_gauss_newton=True, now=3)
    if fault == "unarrived":
        options["now"] = 2
    elif fault == "bad_flag":
        options["head_gauss_newton"] = 1
    elif fault in ("role", "mixed_policy"):
        state = source.snapshot_state()
        policy = replace(
            state.policy,
            **({"training_role": "probe_train"} if fault == "role" else {"binary_gripper_axis": 3}),
        )
        source._policy = policy
        source._state = bp._snapshot(replace(state, policy=policy), policy)
    elif fault == "binary":
        options["binary_gripper_axis"] = 3
    elif fault == "replay":
        options["replay_plan"] = object()
    elif fault == "diagnostics":
        options["diagnostic_stage_bindings"] = object()
    else:
        source = object()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid head correction forked or updated")

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(source, corrections, **options)


@pytest.mark.parametrize("selection", ["omitted", "false", "true"])
def test_should_forward_head_selection_only_to_the_private_update_in_the_existing_three_scene_driver(
    completed, monkeypatch, tmp_path, selection
):
    owner, corrections, initial, runs = completed
    selected = selection == "true"
    cached, _ = runs["head" if selected else "default"]
    calls = []

    def rollout(output, **options):
        calls.append(("rollout", options))
        seed = options["train_seed"]
        actual_episode = (
            corrections.episode
            if seed == 19
            else replace(corrections.episode, episode_id="pick-cube-smoke-17", context="pick-cube")
        )
        report = dict(
            complete=True,
            episode=asdict(actual_episode),
            policy="frozen_continuous_bc",
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

    def update(source, actual, **options):
        calls.append(("update", options))
        assert source is owner and actual == corrections and options["now"] == 3
        return replace(cached, report=deepcopy(cached.report))

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", update)
    options = {} if selection == "omitted" else {"head_gauss_newton": selected}
    result = smoke.run_corrective_adaptation(
        owner, output_directory=tmp_path / selection, visual_observation=False, **options
    )
    assert result["complete"] is True
    assert [kind for kind, _ in calls] == ["rollout", "update", "rollout", "rollout"]
    assert [o["train_seed"] for kind, o in calls if kind == "rollout"] == [19, 17, 19]
    assert all("head_gauss_newton" not in o for kind, o in calls if kind == "rollout")
    assert calls[1][1] == (dict(now=3, head_gauss_newton=True) if selected else dict(now=3))
    assert_state(owner.snapshot_state(), initial)


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_should_refuse_nonboolean_driver_selection_before_collection(
    completed, monkeypatch, tmp_path, flag
):
    owner, _, _, _ = completed
    monkeypatch.setattr(
        smoke, "run_frozen_rollout", lambda *a, **k: pytest.fail("invalid flag collected")
    )
    with pytest.raises(ValueError):
        smoke.run_corrective_adaptation(
            owner,
            output_directory=tmp_path / "never",
            visual_observation=False,
            head_gauss_newton=flag,
        )
    assert not (tmp_path / "never").exists()
