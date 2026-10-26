"""Explicit head updates in the existing delayed TRAIN stream, no new runner.

Prospective four H64 streams:4 initializers512 normal draws/4 A fits4 rows;
8 GN attempts8 rows (7 real solves462 augmented rows),3 default SGD rows.
Twenty-eight conditioning/residual/actor readouts each; no probe/native work.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_learnability_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.core import continuous_bc as task
from src.core import continuous_head_gauss_newton as head
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


def episode(name="A", *, rows=1, dimensions=1, role="train"):
    observation = (0.0,) * dimensions
    return ContinuousEpisode(
        "source-" + name,
        "context-" + name,
        role,
        ContinuousActionSpec(dimensions, ("normalized_gripper",), (-1.0,), (1.0,)),
        tuple(
            ContinuousTransition(
                i,
                observation,
                (0.25 if name == "A" else -0.25,),
                float(i == rows - 1),
                observation,
                i == rows - 1,
                False,
            )
            for i in range(rows)
        ),
    )


def assert_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def streams():
    results = {}
    checkpoint, runtime_factory = demo._checkpoint, demo.ActorShadowRuntime
    fit, propose = demo.fit_training_standardization, head.propose_head_gauss_newton
    head_work, monotonic = demo.head_work, demo.monotonic
    for mode in ("head", "sgd", "failed", "late"):
        warm, new = episode(), episode("B")
        original = asdict(warm), asdict(new)
        captured: dict[str, Any] = dict(
            checkpoints=[], fits=[], proposals=0, armed=False, mutations=0
        )

        def fitting(value, **options):
            captured["fits"].append((value.episode_id, options))
            return fit(value, **options)

        def runtime(source, **options):
            created = runtime_factory(source, **options)
            captured.update(
                source=source,
                runtime=created,
                budget=options["budget"],
                clock=options["clock"],
                initial=source.snapshot_state(),
            )
            return created

        def observing(report, source, runtime, readout, *args, **options):
            checkpoint(report, source, runtime, readout, *args, **options)
            captured["checkpoints"].append(runtime.candidate_snapshot().state)

        def proposing(*args):
            captured["proposals"] += 1
            if mode == "failed" and captured["proposals"] == 2:
                raise RuntimeError("second head call stopped before core work")
            return propose(*args)

        def reporting(*args):
            returned = head_work(*args)
            if mode == "late":
                captured["armed"] = True
            return returned

        def timer():
            returned = monotonic()
            if mode == "late" and captured["armed"] and captured["mutations"] == 0:
                latest = captured["source"]._latest
                assert len(latest) == 1
                owner, _, pinned = latest[0]
                captured["mutated_owner"] = owner
                captured["pinned_final"] = deepcopy(pinned)
                owner._state.parameters[3][0] += 0.125
                captured["mutations"] += 1
            return returned

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(demo, "fit_training_standardization", fitting)
            patch.setattr(demo, "ActorShadowRuntime", runtime)
            patch.setattr(demo, "_checkpoint", observing)
            patch.setattr(head, "propose_head_gauss_newton", proposing)
            patch.setattr(demo, "head_work", reporting)
            patch.setattr(demo, "monotonic", timer)
            options: dict[str, Any] = {} if mode == "sgd" else {"head_gauss_newton": True}
            result = demo.run_continuous_learnability(warm, new, phase_updates=1, **options)
        assert (asdict(warm), asdict(new)) == original
        results[mode] = result, captured
    return results


def test_should_apply_three_existing_delayed_packets_with_original_receipt_identity_and_row_charges(
    streams,
):
    result, captured = streams["head"]
    report = result.report
    assert report["complete"] is True and type(result.candidate) is bp.NumpyContinuousBCLearner
    assert report["protocol_id"] == "continuous_head_gauss_newton_train_aba_v1"
    assert [event["phase"] for event in report["events"]] == ["warm", "new", "warm_return"]
    assert [state.completed_updates for state in captured["checkpoints"]] == [0, 1, 2, 3]
    assert [state.training_examples for state in captured["checkpoints"]] == [0, 1, 2, 3]
    for index, event in enumerate(report["events"]):
        assert event["deferred_exact"] is event["success_validated"] is True
        (receipt,) = event["applied"]
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (
            2 * index,
            2 * index + 1,
            2 * index + 1,
        )
        assert receipt["episode_id"] == ("source-B" if index == 1 else "source-A")
        assert receipt["update_number"] == index + 1
        assert receipt["diagnostic"]["definition"] == task.OBJECTIVE
        assert receipt["actor_version"] == "actor-0" and receipt["learner_version"] == "candidate-0"
    assert (
        captured["budget"].updates_completed == 3
        and captured["budget"].replay_examples_completed == 0
    )
    assert report["progress"] == dict(
        update_attempts=3, updates_completed=3, replay_examples_completed=0, next_tick=6
    )


def test_should_keep_source_actor_and_frozen_statistics_unchanged_while_returning_the_actual_private_candidate(
    streams,
):
    result, captured = streams["head"]
    assert captured["fits"] == [("source-A", {})]
    initial = captured["initial"]
    assert initial.policy.hidden_dim == 64 and initial.policy.seed == 17
    assert initial.policy.observation_standardization.means == (0.0,)
    assert initial.policy.observation_standardization.scales == (1.0,)
    assert_state(initial, captured["source"].snapshot_state())
    assert_state(initial, captured["runtime"].actor.snapshot_state().state)
    assert_state(captured["checkpoints"][-1], result.candidate.snapshot_state())
    final = result.candidate.snapshot_state()
    for index in (0, 1):
        np.testing.assert_array_equal(final.parameters[index], initial.parameters[index])
        np.testing.assert_array_equal(final.velocity[index], initial.velocity[index])
    assert all(
        row["state_exact"] is row["owners_exact"] is True for row in result.report["checkpoints"]
    )


def test_should_report_native_head_work_separately_from_readonly_conditioning_and_default_sgd(
    streams,
):
    result, _ = streams["head"]
    report = result.report
    assert report["head_gauss_newton"]["definition"] == "continuous_bp_head_gauss_newton_v1"
    assert len(report["native_updates"]) == 3
    for native, event in zip(report["native_updates"], report["events"], strict=True):
        assert native["diagnostic"] == event["applied"][0]["diagnostic"]
        assert native["training_rows"] == 1 and native["work"]["published_updates"] == 1
        assert native["loss_after"] < native["diagnostic"]["value"]
        assert native["work"]["solve_attempts"] == native["work"]["solve_completed"] == 1
        assert native["work"]["solve_rows"] == 66
    work = report["work"]
    assert (
        work["optimizer_updates"] == work["head_gauss_newton_updates"] == work["training_rows"] == 3
    )
    assert work["sgd_updates"] == work["training_gradient_evaluations"] == 0
    assert work["conditioning_calls"] == work["conditioning_rows"] == 8
    assert work["native_update_counts"]["solve_attempts"] == 3
    assert work["native_update_counts"]["solve_rows"] == 198
    assert work["native_work_flags"]["solver_work_known"] is False
    assert work["conditioning_work"]["forward_passes"] == 16
    assert work["conditioning_work"]["backward_passes"] == 8
    assert work["gradient_values"] == 8 * 193
    assert work["native_update_counts"]["known_dense_multiply_accumulates"] == 576
    assert work["known_dense_multiply_accumulates"] == 6208


def test_should_keep_the_implicit_default_sgd_stream_schema_and_math_path_unchanged(streams):
    result, captured = streams["sgd"]
    assert result.report["complete"] is True and result.candidate is not None
    assert result.report["protocol_id"] == "continuous_mlp_train_aba_v1"
    assert "head_gauss_newton" not in result.report and "native_updates" not in result.report
    assert captured["proposals"] == 0
    assert [state.completed_updates for state in captured["checkpoints"]] == [0, 1, 2, 3]
    assert_state(captured["initial"], captured["source"].snapshot_state())
    assert_state(captured["initial"], captured["runtime"].actor.snapshot_state().state)
    assert result.report["work"]["optimizer_updates"] == 3


def test_should_stop_after_the_second_native_failure_and_keep_the_first_receipt_and_failed_paid_work(
    streams,
):
    result, captured = streams["failed"]
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert "second head call" in report["error"]
    assert captured["proposals"] == 2
    assert len(report["events"]) == 2 and len(report["native_updates"]) == 1
    assert report["events"][0]["success_validated"] is True
    assert report["events"][1]["success_validated"] is False
    assert len(report["applied_updates"]) == 1 and report["progress"]["updates_completed"] == 1
    assert report["failed_native_work"]["normalization_completed"] == 1
    assert report["failed_native_work"]["published_updates"] == 0
    assert report["native_updates"][0]["work"]["published_updates"] == 1
    assert report["native_updates"][0]["diagnostic"] == report["applied_updates"][0]["diagnostic"]
    assert [state.completed_updates for state in captured["checkpoints"]] == [0, 1]
    assert "work" not in report
    assert_state(captured["initial"], captured["source"].snapshot_state())
    assert_state(captured["initial"], captured["runtime"].actor.snapshot_state().state)


def test_should_refuse_changed_or_replaced_native_receipts_before_capacity_or_another_training_call(
    streams, monkeypatch
):
    result, captured = streams["head"]
    facade = captured["source"]
    sink = result.report["native_updates"]
    facade.require_reports(sink)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("rewritten native receipt reached another training method")

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
    original = sink[0]
    old_count = original["work"]["solve_rows"]
    try:
        original["work"]["solve_rows"] = old_count + 1
        with pytest.raises(RuntimeError, match="returned prefix changed"):
            facade.require_reports(sink)
        with pytest.raises(RuntimeError, match="returned prefix changed"):
            facade.train_batch(np.zeros((1, 1)), np.zeros((1, 1)))
    finally:
        original["work"]["solve_rows"] = old_count
    try:
        sink[0] = deepcopy(original)
        with pytest.raises(RuntimeError, match="returned prefix changed"):
            facade.require_reports(sink)
        with pytest.raises(RuntimeError, match="returned prefix changed"):
            facade.train_batch(np.zeros((1, 1)), np.zeros((1, 1)))
    finally:
        sink[0] = original
    with pytest.raises(RuntimeError, match="list was replaced"):
        facade.require_reports(list(sink))
    facade.require_reports(sink)
    assert calls == []


def test_should_withhold_the_candidate_after_a_final_timer_changes_native_state_and_keep_all_paid_work(
    streams,
):
    result, captured = streams["late"]
    report = result.report
    assert captured["armed"] is True and captured["mutations"] == 1
    assert result.candidate is None and report["complete"] is False and report["error"]
    assert captured["proposals"] == 3
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 3
    assert report["progress"]["updates_completed"] == captured["budget"].updates_completed == 3
    assert report["work"]["head_gauss_newton_updates"] == 3
    assert report["work"]["native_update_counts"]["solve_rows"] == 198
    assert all(event["success_validated"] is True for event in report["events"])
    after = captured["mutated_owner"].snapshot_state()
    before = captured["pinned_final"]
    assert after.completed_updates == before.completed_updates == 3
    assert after.parameters[3][0] == before.parameters[3][0] + 0.125
    assert_state(captured["initial"], captured["source"].snapshot_state())
    assert_state(captured["initial"], captured["runtime"].actor.snapshot_state().state)


@pytest.mark.parametrize(
    "fault",
    [
        "family",
        "binary",
        "replay",
        "probe",
        "rgb",
        "balanced",
        "a_only",
        "role",
        "dimension",
        "rows",
    ],
)
def test_should_refuse_unsupported_modes_and_inputs_before_fit_or_model_construction(
    monkeypatch, fault
):
    warm, new = episode(), episode("B")
    options: dict[str, Any] = dict(head_gauss_newton=True, phase_updates=1)
    if fault == "family":
        options["learner_family"] = "pc"
    elif fault == "binary":
        options["binary_gripper_axis"] = 0
    elif fault in ("replay", "probe"):
        options[fault + "_plan"] = object()
    elif fault == "rgb":
        options["rgb_range_recipe"] = object()
    elif fault == "balanced":
        options["stage_balanced_packets"] = True
    elif fault == "a_only":
        options["a_only_control"] = True
    elif fault == "role":
        new = replace(new, role="probe_test")
    elif fault == "dimension":
        warm, new = episode(dimensions=65), episode("B", dimensions=65)
    else:
        warm = episode(rows=129)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported selected mode reached fit/model")

    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(warm, new, **options)


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_should_refuse_nonboolean_head_selection_before_any_runner_or_fitter(monkeypatch, flag):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid flag reached fitter")

    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(episode(), episode("B"), head_gauss_newton=flag)
    with pytest.raises(ValueError):
        demo.run_continuous_learnability_demo(episode(), episode("B"), head_gauss_newton=flag)


@pytest.mark.parametrize("selection", ["omitted", "false", "true"])
def test_should_forward_only_an_explicit_true_selection_from_the_json_wrapper(
    monkeypatch, selection
):
    calls = []
    report = {"owned_json_report": True}

    def run(*args, **options):
        calls.append((args, options))
        return SimpleNamespace(report=report)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    warm, new = episode(), episode("B")
    options = {} if selection == "omitted" else {"head_gauss_newton": selection == "true"}
    assert demo.run_continuous_learnability_demo(warm, new, **options) is report
    expected: dict[str, Any] = {"expected_standardization": None}
    if selection == "true":
        expected["head_gauss_newton"] = True
    assert calls == [((warm, new), expected)]
