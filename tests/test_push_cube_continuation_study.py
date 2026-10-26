"""Frozen Push helper/receipt boundaries with controlled owners and fake IO.

Literal zero parameter arrays are controlled fixtures, not initialized learners.
No model forward, training, RNG, teacher query, SDK or physics is admitted.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import push_cube_continuation_study as study
from src.adapters import numpy_continuous_bc as bp
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import (
    CubeTaskSpaceState,
    cube_taskspace_observation,
    cube_taskspace_spec,
)
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history
from src.core.push_cube_teacher import PushCubeTeacher
from src.infra import maniskill_push_cube as native
from test_push_cube_continuous import FakePushEnvironment


def forbidden(*args, **kwargs):
    pytest.fail("fixture admitted actual learner initialization, training, teacher or SDK work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        factory_attempts=0,
        factory_returns=0,
        reset_attempts=0,
        reset_returns=0,
        control_attempts=0,
        control_returns=0,
        grasp_attempts=0,
        grasp_returns=0,
        close_attempts=0,
        close_returns=0,
        prediction_attempts=0,
        prediction_returns=0,
        routing_source_calls=0,
        routing_acquisition_calls=0,
        routing_continuation_calls=0,
        routing_push_calls=0,
        routing_pick_calls=0,
        routing_control_reads=0,
        routing_mixed_reads=0,
        routing_fit_calls=0,
        routing_probe_calls=0,
        routing_longer_calls=0,
        routing_longer_reads=0,
        routing_retention_reads=0,
        routing_correction_input_reads=0,
        routing_correction_fit_calls=0,
        model_initializations=0,
        model_forwards=0,
        optimizer_updates=0,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        patch.setattr(PushCubeTeacher, "__init__", forbidden)
        patch.setattr(native, "_make_environment", forbidden)
        yield counts
    if os.environ.get("PLC_FIXTURE_LEDGER"):
        with Path(os.environ["PLC_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["prediction_attempts"] <= 64 and counts["control_attempts"] <= 64
    assert counts["factory_attempts"] <= 32 and counts["close_attempts"] <= 32
    assert (
        counts["model_initializations"]
        == counts["model_forwards"]
        == counts["optimizer_updates"]
        == 0
    )


@pytest.fixture
def source():
    # Why: exercise exact native owner/state guards while admitting zero initializer
    # and zero model arithmetic; prediction is a counted boundary double below.
    owner = object.__new__(bp.NumpyContinuousBCLearner)
    owner._policy = bp.ContinuousBCPolicy(cube_taskspace_spec(), hidden_dim=2, seed=17)
    parameters = (
        np.zeros((12, 2), np.float64),
        np.zeros(2, np.float64),
        np.zeros((2, 4), np.float64),
        np.zeros(4, np.float64),
    )
    velocity = (
        parameters[0].copy(),
        parameters[1].copy(),
        parameters[2].copy(),
        parameters[3].copy(),
    )
    owner._state = bp.ContinuousBCSnapshot(owner._policy, parameters, velocity, 40, 2776)
    return owner


@pytest.fixture
def boundary(monkeypatch, ledger):
    environment = FakePushEnvironment(ledger, success=False)
    mode: dict[str, Any] = {}

    def factory():
        ledger["factory_attempts"] += 1
        if mode.get("factory_error"):
            raise RuntimeError("factory unavailable")
        ledger["factory_returns"] += 1
        return environment

    def predict(owner, features):
        ledger["prediction_attempts"] += 1
        assert features.shape == (1, 12) and features.dtype == np.float64
        if mode.get("prediction_error"):
            raise RuntimeError("prediction return unavailable")
        returned = np.array([[0.123456789, 0.0, 0.0, 1.0]], np.float64)
        if mode.get("shape"):
            returned = np.zeros((2, 4), np.float64)
        if mode.get("nonfinite"):
            returned[0, 0] = np.nan
        ledger["prediction_returns"] += 1
        if mode.get("mutate_source"):
            owner._state.parameters[3][0] += 0.125
        return returned

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", predict)
    return environment, mode


def test_should_record_complete_negative_frozen_rollout_and_actual_native_commands(
    source, boundary
):
    environment, _ = boundary
    initial = bp._snapshot(source._state, source._policy)
    report = study.roll_push(source, label="synthetic-negative")
    assert report["complete"] is True and report["error"] is None
    assert report["source_unchanged"] is True
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert report["unreturned_prediction_possible"] is False
    assert (
        report["final_success"] is report["any_success"] is False and report["sparse_return"] == 0.0
    )
    assert (
        report["work"]["completed_steps"] == 2
        and report["work"]["completed_control_physics_steps"] == 10
    )
    assert (
        report["episode"]["context"] == "push-cube-train-17"
        and report["episode"]["role"] == "train"
    )
    assert tuple(report["transitions"]) == report["episode"]["transitions"]
    for row, command in zip(report["transitions"], environment.commands, strict=True):
        assert row["action"] == tuple(float(value) for value in command[0])
    assert report["transitions"][0]["action"][0] != 0.123456789
    assert environment.reset_seeds == [17] and environment.close_calls == 1
    assert study.same_continuous_state(initial, bp._snapshot(source._state, source._policy))
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("fault", ["shape", "nonfinite", "prediction_error", "mutate_source"])
def test_should_refuse_untrusted_prediction_before_any_native_control(source, boundary, fault):
    environment, mode = boundary
    mode[fault] = True
    report = study.roll_push(source, label=fault)
    assert report["complete"] is False
    assert report["prediction_attempts"] == 1
    assert report["prediction_returns"] == (0 if fault == "prediction_error" else 1)
    assert report["unreturned_prediction_possible"] is (fault == "prediction_error")
    assert report["work"]["step_attempts"] == 0 and environment.commands == []
    assert environment.close_calls == 1
    assert report["source_unchanged"] is (fault != "mutate_source")
    if fault == "mutate_source":
        assert source._state.parameters[3][0] == 0.125


def test_should_recheck_source_after_native_observation_getter_before_prediction(
    source, boundary, monkeypatch
):
    environment, _ = boundary
    original = vars(native.ManiSkillPushCubeEpisode)["observation"].fget

    def observation(owner):
        value = original(owner)
        source._state.parameters[3][0] = 0.25
        return value

    monkeypatch.setattr(native.ManiSkillPushCubeEpisode, "observation", property(observation))
    report = study.roll_push(source, label="observation-callback")
    assert report["complete"] is False and report["source_unchanged"] is False
    assert report["prediction_attempts"] == report["prediction_returns"] == 0
    assert environment.commands == [] and environment.close_calls == 1
    assert source._state.parameters[3][0] == 0.25


@pytest.mark.parametrize("fault", ["factory", "native", "late_source", "cleanup"])
def test_should_preserve_failed_paid_prefix_and_close_original_port(
    source, boundary, monkeypatch, fault
):
    environment, mode = boundary
    if fault == "factory":
        mode["factory_error"] = True
    elif fault == "native":
        environment.step_error = RuntimeError("native control unavailable")
    elif fault == "late_source":
        environment.after_step = lambda: source._state.parameters[3].fill(0.5)
    else:
        close = environment.close

        def failed_close():
            close()
            environment.counts["close_returns"] -= 1
            raise RuntimeError("cleanup unavailable")

        monkeypatch.setattr(environment, "close", failed_close)
    report = study.roll_push(source, label=fault)
    assert report["complete"] is False
    expected = 0 if fault == "factory" else 2 if fault == "cleanup" else 1
    assert report["prediction_attempts"] == report["prediction_returns"] == expected
    assert len(environment.commands) == expected and environment.close_calls == (
        0 if fault == "factory" else 1
    )
    if fault == "factory":
        assert report["unreturned_native_work_unknown"] is True
        assert report["work"]["construction_attempts"] == 1
        assert report["work"]["construction_returns"] == 0
        assert report["work"]["step_attempts"] == report["work"]["completed_steps"] == 0
    elif fault == "native":
        assert report["work"]["step_attempts"] == 1 and report["work"]["completed_steps"] == 0
        assert report["work"]["possible_failed_physics_steps"] == 5
    elif fault == "late_source":
        assert len(report["transitions"]) == report["work"]["completed_steps"] == 1
        assert report["source_unchanged"] is False and source._state.parameters[3][0] == 0.5
    else:
        assert (
            report["work"]["completed_steps"] == 2
            and "cleanup unavailable" in report["close_error"]
        )


def synthetic_receipt():
    """Eighteen serialized held commands; no teacher or native calls generate it."""
    state = CubeTaskSpaceState(False, (-0.06, 0.0, 0.12), (0.0, 0.0, 0.02), (0.0, 0.0, 0.02))
    history = ActionHistoryState()
    rows = []
    action = (0.0, 0.0, 0.0, -1.0)
    for index in range(18):
        before = cube_taskspace_observation(state, history)
        history = advance_action_history(history, action)
        rows.append(
            ContinuousTransition(
                index,
                before,
                action,
                float(index == 17),
                cube_taskspace_observation(state, history),
                index == 17,
                False,
            )
        )
    episode = ContinuousEpisode(
        "push-cube-teacher-17", "push-cube-train-17", "train", cube_taskspace_spec(), tuple(rows)
    )
    collection = dict(
        episode=asdict(episode),
        definition="push_cube_closed_fist_xy_v1",
        seed=17,
        role="train",
        states=[asdict(state)] * 18,
        decisions=[dict(stage="hold", action=action)] * 18,
        transitions=[asdict(row) for row in rows],
        attempted_actions=[action] * 18,
        returned_actions=[action] * 18,
        decision_attempts=18,
        decision_returns=18,
        native_work=asdict(
            native.PushCubeWork(1, 1, 1, 1, 18, 18, 90, 0, 19, 19, 19, 19, 38, 38, 1, 1)
        ),
        metadata=dict(task_id="PushCube-v1", seed=17, role="train"),
        error=None,
        unreturned_work_possible=False,
    )
    return dict(
        complete=True,
        training_release=True,
        collection=collection,
        training_episode=asdict(episode),
    )


def write_receipt(tmp_path, monkeypatch, payload):
    path = tmp_path / "synthetic-push.json"
    encoded = json.dumps(payload, allow_nan=False).encode("utf8")
    path.write_bytes(encoded)
    monkeypatch.setattr(study, "PUSH_SHA256", hashlib.sha256(encoded).hexdigest())
    return path


def test_should_rehydrate_only_the_bound_successful_eighteen_row_training_release(
    tmp_path, monkeypatch
):
    raw = synthetic_receipt()
    path = write_receipt(tmp_path, monkeypatch, raw)
    actual = study.read_push_training(path)
    assert asdict(actual) == asdict(study._episode(raw["training_episode"]))
    assert len(actual.transitions) == 18 and actual.role == "train"
    assert actual.transitions[0].observation[7:] == (0.0,) * 5
    assert actual.transitions[17].next_observation[-1] == 1.0


@pytest.mark.parametrize("fault", ["hash", "release", "receipt_cost", "exported_action"])
def test_should_refuse_replaced_receipt_or_export_without_requerying_any_labels(
    tmp_path, monkeypatch, fault
):
    raw = deepcopy(synthetic_receipt())
    if fault == "release":
        raw["training_release"] = False
    elif fault == "receipt_cost":
        raw["collection"]["native_work"]["completed_steps"] = 17
    elif fault == "exported_action":
        raw["training_episode"]["transitions"][0]["action"] = (0.5, 0.0, 0.0, -1.0)
    path = write_receipt(tmp_path, monkeypatch, raw)
    if fault == "hash":
        monkeypatch.setattr(study, "PUSH_SHA256", "0" * 64)
        monkeypatch.setattr(study, "_episode", forbidden)
    with pytest.raises(ValueError):
        study.read_push_training(path)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "raw_prefix",
        "saved_prefix",
        "replaced_prefix",
        "deleted_prefix",
        "replaced_raw",
        "last_clock",
        "reused_control",
    ],
)
@pytest.mark.parametrize("profile", ["head_gauss_newton", "bp_mixed", "bp_mixed_retention"])
def test_should_route_fixed_study_and_refuse_prior_evidence_or_final_clock_corruption(
    tmp_path, monkeypatch, source, ledger, fault, profile
):
    run_routing_case(tmp_path, monkeypatch, source, ledger, fault, profile)


def synthetic_longer_continuation(
    fault=None, *, correction_fit=False, row_weights=None, aggregated=False
):
    updates, rows = (8, 198) if aggregated else (8, 148) if correction_fit else (64, 98)
    native: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for index in range(updates):
        diagnostic = dict(definition=study.MIXED_OBJECTIVE, value=0.25)
        native.append(dict(training_rows=rows, diagnostic=diagnostic))
        applied.append(
            dict(
                diagnostic=diagnostic,
                update_number=index + 1,
                observed_at=1000 + 2 * index,
                arrived_at=1001 + 2 * index,
                applied_at=1001 + 2 * index,
                episode_id="continuous-TRAIN-continuation",
                sample_id=f"continuation-{index}",
                event_id=f"continuation-label-{index}",
                actor_version="actor-0",
                learner_version="candidate-0",
            )
        )
    if fault == "receipt":
        applied[3 if correction_fit else 17]["arrived_at"] += 1
    if fault == "clock_type":
        applied[17]["arrived_at"] = float(applied[17]["arrived_at"])
    value = dict(
        elapsed_seconds=2.0,
        complete=fault not in ("longer_incomplete", "incomplete"),
        source_exact=True,
        actor_exact=True,
        rows=rows,
        protocol="continuous_train_mixed_gripper_continuation_v1",
        budget=dict(max_training_updates=updates),
        initial_counts=dict(updates=120, training_rows=11016)
        if aggregated
        else dict(updates=112, training_rows=9832)
        if correction_fit
        else dict(updates=48, training_rows=3560),
        unreturned_training_work_unknown=fault == "unknown",
        final_counts=dict(updates=128, training_rows=12600)
        if aggregated
        else dict(updates=120, training_rows=11016)
        if correction_fit
        else dict(updates=112, training_rows=9832),
        progress=dict(
            updates_completed=updates,
            replay_examples_completed=0,
            logical_tick=1000 + 2 * updates - 1,
        ),
        native_updates=native,
        sealed_native_updates=deepcopy(native),
        validated_native_updates=deepcopy(native),
        applied_updates=applied,
        sealed_applied_updates=deepcopy(applied),
        events=[dict(deferred_exact=True, success_validated=True) for _ in range(updates)],
    )
    if row_weights is not None:
        value["protocol"] = "continuous_train_row_weighted_mixed_gripper_continuation_v1"
        raw_weights = np.asarray(row_weights, np.float64)
        scaled = raw_weights / raw_weights.max()
        value["row_weighting"] = dict(
            definition=study.WEIGHTED_MIXED_OBJECTIVE,
            values=list(row_weights),
            normalized_masses=(scaled / scaled.sum()).tolist(),
        )
        value["policy_transition"] = dict(candidate_objective=study.WEIGHTED_MIXED_OBJECTIVE)
        for native_row, receipt in zip(native, applied, strict=True):
            diagnostic = dict(definition=study.WEIGHTED_MIXED_OBJECTIVE, value=0.25)
            native_row.update(
                diagnostic=diagnostic,
                objective_row_weights=list(row_weights),
                work=dict(
                    objective_weight_values=rows,
                    objective_weight_numeric_bytes=8 * rows,
                    native_row_weight_mass_admissions=2,
                ),
            )
            receipt["diagnostic"] = diagnostic
        value["sealed_native_updates"] = deepcopy(native)
        value["validated_native_updates"] = deepcopy(native)
        value["sealed_applied_updates"] = deepcopy(applied)
    if aggregated:
        value["row_scope"] = dict(
            maximum_rows=198,
            default_maximum_rows=148,
            scope="explicit expanded released TRAIN inputs; no authority or update-budget renewal",
        )
    return value


def scalar_corrective_readout(updates, *, aggregated=False):
    value = (
        (0.125 if updates == 120 else 0.0625) if aggregated else (0.25 if updates == 112 else 0.125)
    )
    names: tuple[str, ...] = ("corrections", "push", "a", "b", "train20")
    sizes: tuple[int, ...] = (50, 18, 15, 50, 15)
    if aggregated:
        names, sizes = ("current_corrections",) + names, (50,) + sizes
    return dict(
        complete=True,
        label=f"aggregated{updates}" if aggregated else f"corrective{updates}",
        source_updates=updates,
        source_training_examples=12600
        if aggregated and updates == 128
        else 9832
        if updates == 112
        else 11016,
        group_order=list(names),
        native_objective=study.MIXED_OBJECTIVE,
        physical_objective="normalized_active_axis_half_mse",
        work=dict(unknown_unreturned_work=False),
        phase_readouts=dict(forks=1, validated=1),
        groups={
            name: dict(
                rows=rows,
                conditioning=dict(conditioning=dict(objective_loss=value)),
                residual=dict(
                    residuals=dict(
                        overall=dict(
                            normalized_half_mse=value,
                            axes=[
                                dict(
                                    index=i,
                                    unit="normalized",
                                    normalized_bias=value,
                                    normalized_mae=value,
                                    normalized_half_mse=value,
                                )
                                for i in range(4)
                            ],
                        )
                    )
                ),
            )
            for name, rows in zip(names, sizes, strict=True)
        },
        operations=[
            dict(group=name, kind=kind, rows=rows, attempted=True, returned=True, validated=True)
            for name, rows in zip(names, sizes, strict=True)
            for kind in ("conditioning", "residual")
        ],
    )


def run_routing_case(
    tmp_path,
    monkeypatch,
    source,
    ledger,
    fault,
    profile,
    *,
    training_diagnostics=False,
    diagnostics_only=False,
    readout_fault=None,
    reserved_probes=False,
    probe_fault=None,
    longer_fit=False,
    longer_fault=None,
    native_endpoint=False,
    endpoint_fault=None,
    push_corrections=False,
    correction_fault=None,
    correction_fit=False,
    fit_fault=None,
    correction_readouts=False,
    diagnostic_fault=None,
    balanced_correction_fit=False,
    balanced_fault=None,
    zero_velocity_fit=False,
    zero_fault=None,
    balanced_push_corrections=False,
    collection_fault=None,
    aggregated_correction_fit=False,
    aggregated_fault=None,
):
    from test_joint_acquisition_robot_learning import workflow_inputs

    inputs = workflow_inputs()
    training = study._episode(synthetic_receipt()["training_episode"])
    reads: list[str] = []

    def read_training(path):
        reads.append("training")
        return training

    def read_inputs(*paths):
        reads.append("inputs")
        return inputs, {}

    monkeypatch.setattr(study, "read_push_training", read_training)
    monkeypatch.setattr(study, "read_joint_acquisition_inputs", read_inputs)
    candidate = object.__new__(bp.NumpyContinuousBCLearner)
    candidate._policy = (
        replace(source._policy, binary_gripper_axis=3)
        if profile != "head_gauss_newton"
        else source._policy
    )
    candidate._state = replace(
        source._state,
        policy=candidate._policy,
        parameters=tuple(a.copy() for a in source._state.parameters),
        velocity=tuple(a.copy() for a in source._state.velocity),
        completed_updates=48,
        training_examples=3560,
    )
    source_raw = dict(complete=True, marker="original", stream=dict(elapsed_seconds=1.0))
    residuals = {
        when: {
            group: dict(result=dict(overall=dict(normalized_half_mse=value)))
            for group in ("a", "b", "curated_a", "retained_train20")
        }
        for when, value in (("before", 0.0), ("after", 1.0))
    }

    def push_report():
        return dict(
            complete=True,
            final_success=False,
            any_success=False,
            sparse_return=0,
            transitions=[dict(observation=(0.0,) * 12)],
            metadata=dict(
                task_id="PushCube-v1",
                seed=17,
                role="train",
                context="push-cube-train-17",
                recipe={},
                controller={},
                horizon=50,
            ),
        )

    control: dict[str, Any] = {}
    if profile != "head_gauss_newton":
        recipe = study.PickCubeTaskSpaceHistoryRecipe(mode="executed")
        view = study.action_history_correction_view(inputs[2], recipe)
        _, targets = view.model.training_pairs(now=1100)
        corrected = replace(
            view.model.episode,
            episode_id="derived-B19-supervision",
            transitions=tuple(
                replace(row, action=target)
                for row, target in zip(view.model.episode.transitions, targets, strict=True)
            ),
        )
        references = (
            study.action_history_training_episode(inputs[0].episode, recipe),
            corrected,
            study.action_history_training_episode(inputs[4].episode, recipe),
        )
        control.update(
            protocol="push_cube_continuation_train_v1",
            complete=True,
            source_unchanged=True,
            push_initial_exact=True,
            before_push=push_report(),
            source=deepcopy(source_raw),
            acquisition=dict(complete=True, residuals=deepcopy(residuals)),
            input_receipts={},
            training=asdict(training),
            references=[asdict(value) for value in references],
            b_original=asdict(inputs[2]),
            b_projected_actual=asdict(view.model),
            reused_push_sha256=study.PUSH_SHA256,
        )
        # The accepted control is serialized, while a live asdict row retains
        # tuples. Exercise that actual boundary for all mixed routing cases.
        control = json.loads(json.dumps(control, allow_nan=False))

    continued_raw = dict(
        complete=True,
        elapsed_seconds=2.0,
        source_exact=True,
        actor_exact=True,
        unreturned_training_work_unknown=False,
    )
    mixed: dict[str, Any] = {}
    if profile == "bp_mixed_retention":
        mixed = deepcopy(control)
        mixed.update(
            protocol="push_cube_mixed_continuation_train_v1",
            complete=False,
            error="RuntimeError: fixed Push before/after reset binding differs",
            pick=[],
            unreturned_stage_work_unknown=False,
            continuation=dict(continued_raw, elapsed_seconds=1.0),
            after_push=dict(
                push_report(),
                source_unchanged=True,
                unreturned_prediction_possible=False,
                unreturned_native_work_unknown=False,
            ),
        )
        mixed = json.loads(json.dumps(mixed, allow_nan=False))

    longer_cache: dict[str, Any] = {}
    retention_cache: dict[str, Any] = {}
    if (
        native_endpoint
        or push_corrections
        or correction_readouts
        or balanced_push_corrections
        or aggregated_correction_fit
    ):
        longer_cache = deepcopy(control)
        longer_cache.update(
            complete=True,
            source_unchanged=True,
            unreturned_stage_work_unknown=False,
            continuation=deepcopy(continued_raw),
            pick=[],
            longer_continuation=synthetic_longer_continuation(),
            longer_readout=dict(
                complete=True, label="mixed112", work={"prediction_calls": 0}, operations=[]
            ),
        )
        retention_cache = deepcopy(control)
        retention_cache.update(
            protocol="push_cube_mixed_retention_train_v1",
            complete=True,
            unreturned_stage_work_unknown=False,
            continuation=deepcopy(continued_raw),
            after_push=deepcopy(mixed["after_push"]),
            pick=[
                dict(
                    seed=seed,
                    complete=True,
                    final_success=True,
                    any_success=True,
                    sparse_return=1.0,
                    role="train",
                    task="PickCube-v1",
                    context=f"pick-cube-train-{seed}",
                    controller="pd_ee_delta_pos",
                    observation_recipe="state42",
                    action_history=dict(recipe=asdict(recipe)),
                    episode=dict(transitions=[dict(observation=(0.0,) * 42)] * count),
                )
                for seed, count in zip((17, 19, 20), (16, 50, 14), strict=True)
            ],
        )
        if endpoint_fault == "longer_binding":
            longer_cache["longer_continuation"]["events"][0]["extra"] = "changed"
        elif endpoint_fault == "readout_binding":
            longer_cache["longer_readout"]["work"]["prediction_calls"] = 1

    def read_control(path):
        ledger["routing_control_reads"] += 1
        reads.append("control")
        assert profile != "head_gauss_newton"
        return control

    monkeypatch.setattr(study, "read_reused_control", read_control)

    def read_mixed(path):
        ledger["routing_mixed_reads"] += 1
        reads.append("mixed")
        assert profile == "bp_mixed_retention"
        return mixed

    monkeypatch.setattr(study, "read_reused_mixed", read_mixed)

    def read_longer(path):
        assert (
            native_endpoint
            or push_corrections
            or correction_readouts
            or balanced_push_corrections
            or aggregated_correction_fit
        ) and path == study.REUSED_LONGER
        ledger["routing_longer_reads"] += 1
        reads.append("longer")
        return longer_cache

    def read_retention(path):
        assert (
            native_endpoint
            or push_corrections
            or correction_readouts
            or balanced_push_corrections
            or aggregated_correction_fit
        ) and path == study.REUSED_RETENTION
        ledger["routing_retention_reads"] += 1
        reads.append("retention")
        return retention_cache

    monkeypatch.setattr(
        study,
        "read_reused_longer",
        read_longer
        if native_endpoint
        or push_corrections
        or correction_readouts
        or balanced_push_corrections
        or aggregated_correction_fit
        else forbidden,
    )
    monkeypatch.setattr(
        study,
        "read_reused_retention",
        read_retention
        if native_endpoint
        or push_corrections
        or correction_readouts
        or balanced_push_corrections
        or aggregated_correction_fit
        else forbidden,
    )
    correction_batch = None
    native_cache = deepcopy(longer_cache)
    if push_corrections:
        from src.core.push_cube_corrections import PushCubeCorrectionBatch, PushCubeCorrectionLabel

        original = synthetic_receipt()
        state = CubeTaskSpaceState(**original["collection"]["states"][0])
        correction_batch = PushCubeCorrectionBatch(
            17,
            replace(training, episode_id="push-cube-student-17"),
            tuple(
                PushCubeCorrectionLabel(row.step, state, row.observation, row.action, "hold")
                for row in training.transitions
            ),
        )
        native_cache["native_push"] = dict(
            push_report(), transitions=[asdict(row) for row in training.transitions]
        )
    correction_packet = None
    correction_receipt = {"kind": "synthetic50 correction release", "rows": 50}
    if correction_fit:
        state = CubeTaskSpaceState(**synthetic_receipt()["collection"]["states"][0])
        history = ActionHistoryState()
        packet_rows = []
        for index in range(50):
            observation = cube_taskspace_observation(state, history)
            action = (0.0, 0.0, 0.0, -1.0)
            history = advance_action_history(history, action)
            packet_rows.append(
                ContinuousTransition(
                    index,
                    observation,
                    action,
                    0.0,
                    cube_taskspace_observation(state, history),
                    index == 49,
                    False,
                )
            )
        correction_packet = replace(
            training, episode_id="push-cube-correction-targets-17", transitions=tuple(packet_rows)
        )
        native_cache["native_push"] = dict(
            push_report(), transitions=push_report()["transitions"] + [{}] * 49
        )
        native_cache["pick"] = deepcopy(retention_cache["pick"])
        for old, count, success in zip(
            native_cache["pick"], (41, 50, 50), (True, False, False), strict=True
        ):
            old.update(final_success=success, any_success=success, sparse_return=float(success))
            old["episode"]["transitions"] = [dict(observation=(0.0,) * 42)] * count

    corrective_cache = deepcopy(native_cache)
    if (
        correction_readouts
        or balanced_correction_fit
        or balanced_push_corrections
        or aggregated_correction_fit
    ):
        corrective_cache["correction_continuation"] = synthetic_longer_continuation(
            correction_fit=True
        )
        corrective_cache["correction_fit_protocol"] = dict(
            updates=8,
            rows=148,
            initial_counts=[112, 9832],
            final_counts=[120, 11016],
            input_receipt=deepcopy(correction_receipt),
            scope="all50corrections+18expert+80oldTRAIN; no row selection",
        )
        if diagnostic_fault == "binding":
            corrective_cache["correction_continuation"]["events"][0]["extra"] = "changed"

    def read_corrective(path):
        reads.append("corrective")
        assert path == study.REUSED_CORRECTIVE
        return corrective_cache

    monkeypatch.setattr(
        study,
        "read_reused_corrective",
        read_corrective
        if correction_readouts
        or balanced_correction_fit
        or balanced_push_corrections
        or aggregated_correction_fit
        else forbidden,
    )

    uniform_readouts = deepcopy(corrective_cache)
    if balanced_correction_fit or balanced_push_corrections or aggregated_correction_fit:
        uniform_readouts.update(
            corrective_readout112=scalar_corrective_readout(112),
            corrective_readout120=scalar_corrective_readout(120),
        )
        corrective_cache["native_push"] = dict(
            push_report(), transitions=push_report()["transitions"] + [{}] * 49
        )
        corrective_cache["pick"] = deepcopy(retention_cache["pick"])
        for old in corrective_cache["pick"]:
            old.update(final_success=False, any_success=False, sparse_return=0.0)
            old["episode"]["transitions"] = [dict(observation=(0.0,) * 42)] * 50

    def read_uniform_readouts(path):
        assert (
            balanced_correction_fit or balanced_push_corrections or aggregated_correction_fit
        ) and path == study.REUSED_CORRECTIVE_READOUTS
        reads.append("uniform readouts")
        return uniform_readouts

    if balanced_correction_fit or balanced_push_corrections or aggregated_correction_fit:
        monkeypatch.setattr(study, "read_reused_corrective_readouts", read_uniform_readouts)

    inherited_balanced: dict[str, Any] = {}
    if zero_velocity_fit or balanced_push_corrections or aggregated_correction_fit:
        fixed_weights, fixed_recipe = study.build_balanced_correction_weights(
            tuple(
                zip(
                    ("corrections", "push", "a", "b", "train20"),
                    (correction_packet, training, *references),
                    strict=True,
                )
            )
        )
        inherited_balanced = deepcopy(corrective_cache)
        inherited_balanced.update(
            protocol="push_cube_prefix_group_balanced_correction_train_v1",
            balanced_correction_protocol=fixed_recipe,
            balanced_readout120=dict(scalar_corrective_readout(120), label="balanced120"),
            corrective_readout112=scalar_corrective_readout(112),
            correction_continuation=synthetic_longer_continuation(
                correction_fit=True, row_weights=fixed_weights
            ),
        )

    def read_inherited(path):
        assert (
            zero_velocity_fit or balanced_push_corrections or aggregated_correction_fit
        ) and path == study.REUSED_BALANCED
        reads.append("balanced")
        return inherited_balanced

    if zero_velocity_fit or balanced_push_corrections or aggregated_correction_fit:
        monkeypatch.setattr(study, "_read_push_study", read_inherited)

    if balanced_push_corrections:
        from src.core.push_cube_corrections import PushCubeCorrectionBatch, PushCubeCorrectionLabel

        state = CubeTaskSpaceState(**synthetic_receipt()["collection"]["states"][0])
        history = ActionHistoryState()
        current_rows = []
        labels = []
        for index in range(50):
            observed = cube_taskspace_observation(state, history)
            command = (0.0, 0.0, 0.0, 1.0)
            history = advance_action_history(history, command)
            current_rows.append(
                ContinuousTransition(
                    index,
                    observed,
                    command,
                    0.0,
                    cube_taskspace_observation(state, history),
                    False,
                    index == 49,
                )
            )
            labels.append(
                PushCubeCorrectionLabel(index, state, observed, (0.0, 0.0, 0.0, -1.0), "hold")
            )
        current_episode = ContinuousEpisode(
            "push-cube-student-17",
            "push-cube-train-17",
            "train",
            training.spec,
            tuple(current_rows),
        )
        correction_batch = PushCubeCorrectionBatch(17, current_episode, tuple(labels))
        inherited_balanced["native_push"] = dict(
            push_report(), transitions=[asdict(row) for row in current_rows]
        )
        inherited_balanced["balanced_correction_protocol"].update(
            reused_uniform120=dict(
                path=str(study.REUSED_CORRECTIVE), sha256=study.REUSED_CORRECTIVE_SHA256
            ),
            reused_readouts=dict(
                path=str(study.REUSED_CORRECTIVE_READOUTS),
                sha256=study.REUSED_CORRECTIVE_READOUTS_SHA256,
            ),
        )

    current_packet = None
    current_receipt = dict(rows=50, source_updates=120, source_training_rows=11016, release_at=1100)
    if aggregated_correction_fit:
        assert correction_packet is not None
        current_packet = replace(
            correction_packet, episode_id="push-cube-correction-targets-17-source120"
        )
        inherited_balanced["balanced_correction_protocol"].update(
            reused_uniform120=dict(
                path=str(study.REUSED_CORRECTIVE), sha256=study.REUSED_CORRECTIVE_SHA256
            ),
            reused_readouts=dict(
                path=str(study.REUSED_CORRECTIVE_READOUTS),
                sha256=study.REUSED_CORRECTIVE_READOUTS_SHA256,
            ),
        )
        inherited_balanced["native_push"] = dict(
            push_report(), transitions=push_report()["transitions"] + [{}] * 49
        )
        inherited_balanced["pick"] = deepcopy(corrective_cache["pick"])

    def load_current():
        reads.append("current inputs")
        return current_packet, current_receipt

    if aggregated_correction_fit:
        monkeypatch.setattr(study, "load_balanced_push_correction_inputs", load_current)

    original_correction_input = None
    if balanced_push_corrections:
        assert correction_packet is not None
        original_correction_input = asdict(correction_packet)

    def load_correction_inputs():
        assert correction_fit
        ledger["routing_correction_input_reads"] += 1
        reads.append("correction inputs")
        return correction_packet, correction_receipt

    monkeypatch.setattr(
        study,
        "load_push_correction_inputs",
        load_correction_inputs if correction_fit else forbidden,
    )

    def read_native(path):
        reads.append("native")
        return native_cache

    monkeypatch.setattr(
        study,
        "read_reused_native",
        read_native if push_corrections or correction_fit else forbidden,
    )

    def collect_corrections(owner, *, seed, role, check_boundary):
        ledger["routing_collection_calls"] = ledger.get("routing_collection_calls", 0) + 1
        assert owner is captured[
            "corrected" if balanced_push_corrections else "extended"
        ].candidate and (seed, role) == (17, "train")
        check_boundary()
        assert correction_batch is not None
        raw = dict(
            (inherited_balanced if balanced_push_corrections else native_cache)["native_push"],
            complete=correction_fault != "incomplete",
            source_unchanged=True,
            training_release=True,
            unreturned_query_possible=False,
            unreturned_prediction_possible=False,
            unreturned_native_work_unknown=False,
            episode=asdict(correction_batch.episode),
            labels=[asdict(label) for label in correction_batch.labels],
        )
        if balanced_push_corrections:
            raw.update(source_updates=120, source_training_rows=11016)
            if collection_fault == "incomplete":
                raw["complete"] = False
            elif collection_fault == "unknown":
                raw["unreturned_query_possible"] = True
            elif collection_fault == "collection_counter":
                raw["source_training_rows"] = 9832
            elif collection_fault == "labels":
                raw["labels"][0]["target"] = (0.125, 0.0, 0.0, -1.0)
            elif collection_fault == "trajectory":
                raw["transitions"] = deepcopy(raw["transitions"])
                raw["transitions"][0]["reward"] = 1.0
            elif collection_fault == "prior":
                fit_reports[-1]["groups"]["push"]["rows"] = 1
            captured["collected_raw"] = raw
        if correction_fault == "trajectory":
            raw["transitions"] = deepcopy(raw["transitions"])
            raw["transitions"][0]["reward"] += 0.1
        if correction_fault == "labels":
            raw["labels"] = []
        if correction_fault == "prior":
            captured["extended"].report["complete"] = False
        return SimpleNamespace(report=raw, corrections=correction_batch)

    monkeypatch.setattr(
        study,
        "collect_push_cube_corrections",
        collect_corrections
        if push_corrections or balanced_push_corrections or aggregated_correction_fit
        else forbidden,
    )
    if balanced_push_corrections and collection_fault == "release":
        release = PushCubeCorrectionBatch.training_episode

        def released(batch, *, now):
            packet = release(batch, now=now)
            if batch is correction_batch:
                captured["collected_raw"]["labels"][0]["target"] = (0.125, 0.0, 0.0, -1.0)
            return packet

        monkeypatch.setattr(PushCubeCorrectionBatch, "training_episode", released)

    captured: dict[str, Any] = {}
    push_labels: list[str] = []
    fit_labels: list[str] = []
    fit_reports: list[dict[str, Any]] = []
    write = study._write

    def capture(output, report, **options):
        captured["saved"] = report
        write(output, report, **options)

    def built(*args, **kwargs):
        ledger["routing_source_calls"] += 1
        assert kwargs == dict(corrections_now=99, curated_now=199)
        if aggregated_correction_fit:
            assert reads == [
                "control",
                "mixed",
                "longer",
                "retention",
                "native",
                "corrective",
                "uniform readouts",
                "balanced",
                "correction inputs",
                "current inputs",
                "training",
                "inputs",
            ]
        assert aggregated_correction_fit or reads == (
            [
                "control",
                "mixed",
                "longer",
                "retention",
                "native",
                "corrective",
                "uniform readouts",
                "balanced",
                "correction inputs",
                "training",
                "inputs",
            ]
            if balanced_push_corrections
            else (
                [
                    "control",
                    "mixed",
                    "longer",
                    "retention",
                    "native",
                    "corrective",
                    "balanced",
                    "correction inputs",
                    "training",
                    "inputs",
                ]
                if zero_velocity_fit
                else (
                    [
                        "control",
                        "mixed",
                        "longer",
                        "retention",
                        "native",
                        "corrective",
                        "uniform readouts",
                        "correction inputs",
                        "training",
                        "inputs",
                    ]
                    if balanced_correction_fit
                    else (
                        [
                            "control",
                            "mixed",
                            "longer",
                            "retention",
                            "native",
                            "corrective",
                            "correction inputs",
                            "training",
                            "inputs",
                        ]
                        if correction_readouts
                        else [
                            "control",
                            "mixed",
                            "longer",
                            "retention",
                            "native",
                            "correction inputs",
                            "training",
                            "inputs",
                        ]
                        if correction_fit
                        else [
                            "control",
                            "mixed",
                            "longer",
                            "retention",
                            "native",
                            "training",
                            "inputs",
                        ]
                        if push_corrections
                        else ["control", "mixed", "longer", "retention", "training", "inputs"]
                        if native_endpoint
                        else ["control", "mixed", "training", "inputs"]
                        if profile == "bp_mixed_retention"
                        else ["control", "training", "inputs"]
                        if profile == "bp_mixed"
                        else ["training", "inputs"]
                    )
                )
            )
        )
        if collection_fault == "cache":
            inherited_balanced["complete"] = False
        elif collection_fault == "recipe":
            inherited_balanced["balanced_correction_protocol"]["values"][0] += 1.0
        if zero_fault == "baseline":
            inherited_balanced["balanced_readout120"]["label"] = "changed baseline"
        if balanced_fault == "uniform_cache":
            corrective_cache["complete"] = False
        if diagnostic_fault == "cache":
            corrective_cache["complete"] = False
        if fault == "reused_control":
            if profile == "bp_mixed_retention":
                mixed["complete"] = True
            else:
                control["complete"] = False
        if endpoint_fault in ("longer_cache", "retention_cache"):
            (longer_cache if endpoint_fault == "longer_cache" else retention_cache)["complete"] = (
                False
            )
        return SimpleNamespace(report=source_raw, candidate=source)

    def acquired(*args, **kwargs):
        ledger["routing_acquisition_calls"] += 1
        return SimpleNamespace(report=dict(complete=True, residuals=residuals), candidate=source)

    def continued(owner, train, *, references, updates, **options):
        ledger["routing_continuation_calls"] += 1
        if (
            aggregated_correction_fit
            and owner is captured.get("corrected", SimpleNamespace(candidate=None)).candidate
        ):
            assert train is current_packet and updates == 8
            assert tuple(len(value.transitions) for value in references) == (50, 18, 15, 50, 15)
            assert references[0] is correction_packet and references[1] is training
            expected = study.build_aggregated_correction_weights(
                tuple(
                    zip(
                        ("current_corrections", "corrections", "push", "a", "b", "train20"),
                        (train, *references),
                        strict=True,
                    )
                )
            )[0]
            assert set(options) == {"update_rule", "row_weights", "maximum_rows"}
            assert options["update_rule"] == "bp_mixed" and options["maximum_rows"] == 198
            np.testing.assert_array_equal(options["row_weights"], expected)
            expanded = object.__new__(bp.NumpyContinuousBCLearner)
            expanded._policy = owner._policy
            expanded._state = replace(
                owner._state,
                parameters=tuple(x.copy() for x in owner._state.parameters),
                velocity=tuple(x.copy() for x in owner._state.velocity),
                completed_updates=127 if aggregated_fault == "counter" else 128,
                training_examples=12600,
            )
            value = synthetic_longer_continuation(
                "incomplete"
                if aggregated_fault == "incomplete"
                else "receipt"
                if aggregated_fault == "receipt"
                else None,
                correction_fit=True,
                row_weights=expected,
                aggregated=True,
            )
            if aggregated_fault == "scope":
                value["row_scope"]["maximum_rows"] = 148
            if aggregated_fault == "prior":
                captured["corrected"].report["complete"] = False
            if aggregated_fault == "packet":
                object.__setattr__(current_packet.transitions[0], "action", (0.125, 0.0, 0.0, -1.0))
            holder = SimpleNamespace(report=value, candidate=expanded)
            captured["aggregated"] = holder
            return holder
        if correction_fit and (
            owner is captured.get("extended", SimpleNamespace(candidate=None)).candidate
            or zero_velocity_fit
            and owner is captured.get("zero", SimpleNamespace(candidate=None)).candidate
        ):
            ledger["routing_correction_fit_calls"] += 1
            assert train is correction_packet and updates == 8
            assert train is not None
            expected_weights = None
            if (
                balanced_correction_fit
                or zero_velocity_fit
                or balanced_push_corrections
                or aggregated_correction_fit
            ):
                expected_weights = (
                    (49.0,)
                    + (1.0,) * 49
                    + tuple(
                        weight for count in (18, 15, 50, 15) for weight in (98.0 / count,) * count
                    )
                )
                assert set(options) == {"update_rule", "row_weights"}
                assert options["update_rule"] == "bp_mixed"
                np.testing.assert_array_equal(options["row_weights"], expected_weights)
            else:
                assert options == dict(update_rule="bp_mixed")
            assert tuple(len(value.transitions) for value in references) == (18, 15, 50, 15)
            assert references[0] is training and len(train.transitions) == 50
            corrected = object.__new__(bp.NumpyContinuousBCLearner)
            corrected._policy = owner._policy
            corrected._state = replace(
                owner._state,
                parameters=tuple(value.copy() for value in owner._state.parameters),
                velocity=tuple(value.copy() for value in owner._state.velocity),
                completed_updates=119
                if fit_fault == "counter" or collection_fault == "counter"
                else 120,
                training_examples=11016,
            )
            selected_fault = (
                balanced_fault if balanced_fault in ("incomplete", "receipt") else fit_fault
            )
            value = synthetic_longer_continuation(
                selected_fault, correction_fit=True, row_weights=expected_weights
            )
            holder = SimpleNamespace(report=value, candidate=corrected)
            captured["corrected"] = holder
            if balanced_push_corrections:
                value["elapsed_seconds"] = 7.0
                if collection_fault == "data":
                    object.__setattr__(train.transitions[0], "action", (0.125, 0.0, 0.0, -1.0))
                elif collection_fault == "reconstruction":
                    value["native_updates"][0]["diagnostic"]["value"] += 0.125
                    value["applied_updates"][0]["diagnostic"]["value"] += 0.0
                    value["sealed_native_updates"] = deepcopy(value["native_updates"])
                    value["validated_native_updates"] = deepcopy(value["native_updates"])
                    value["sealed_applied_updates"] = deepcopy(value["applied_updates"])
                elif collection_fault == "weights":
                    value["native_updates"][0]["objective_row_weights"][0] += 1.0
            if zero_fault == "private":
                captured["zero"].candidate._state.parameters[3][0] += 0.125
            elif zero_fault == "prior":
                captured["zero"].report["work"]["velocity_numeric_bytes_zeroed"] += 8
            if fit_fault == "prior" or balanced_fault == "prior":
                captured["extended"].report["complete"] = False
            elif fit_fault == "packet":
                object.__setattr__(train.transitions[0], "action", (0.125, 0.0, 0.0, -1.0))
            return holder
        if updates == 64:
            ledger["routing_longer_calls"] += 1
            assert longer_fit and owner is candidate and train is training
            assert options == dict(update_rule="bp_mixed")
            assert tuple(len(item.transitions) for item in references) == (15, 50, 15)
            extended = object.__new__(bp.NumpyContinuousBCLearner)
            extended._policy = candidate._policy
            extended._state = replace(
                candidate._state,
                parameters=tuple(a.copy() for a in candidate._state.parameters),
                velocity=tuple(a.copy() for a in candidate._state.velocity),
                completed_updates=111 if longer_fault == "counter" else 112,
                training_examples=9832,
            )
            value = synthetic_longer_continuation(longer_fault)
            holder = SimpleNamespace(report=value, candidate=extended)
            captured["extended"] = holder
            return holder
        assert owner is source and train is training and updates == 8
        assert options == (dict(update_rule="bp_mixed") if profile != "head_gauss_newton" else {})
        assert tuple(len(e.transitions) for e in references) == (15, 50, 15)
        assert references[1].episode_id == "derived-B19-supervision"
        assert tuple(row.action for row in references[1].transitions) == tuple(
            label.target for label in inputs[2].labels
        )
        # Original student command history differs from the correction target;
        # both original and projected-actual batches stay separate in the report.
        assert (
            references[1].transitions[1].observation[7:11]
            == inputs[2].episode.transitions[0].action
        )
        assert asdict(inputs[2]) == captured["saved"]["b_original"]
        holder = SimpleNamespace(report=deepcopy(continued_raw), candidate=candidate)
        captured["continued"] = holder
        if probe_fault == "prior_probe":
            probe_reports[0]["metrics"]["fresh_acquisition_gain"] = 99.0
        return holder

    def prepared_zero(owner, *, check_boundary):
        ledger["routing_zero_preparation_calls"] = (
            ledger.get("routing_zero_preparation_calls", 0) + 1
        )
        assert zero_velocity_fit and owner is captured["extended"].candidate
        check_boundary()
        zero = object.__new__(bp.NumpyContinuousBCLearner)
        zero._policy = owner._policy
        zero._state = replace(
            owner._state,
            parameters=tuple(value.copy() for value in owner._state.parameters),
            velocity=tuple(np.zeros_like(value) for value in owner._state.velocity),
            completed_updates=111 if zero_fault == "candidate" else 112,
        )
        # This is an opaque H64 preparation receipt double, not executed H64 work.
        report = dict(
            definition="continuous_private_zero_velocity_bc_v1",
            complete=zero_fault != "preparation",
            source_unchanged=True,
            initial_counts=dict(updates=112, training_rows=9832),
            work=dict(
                source_fork_attempts=1,
                source_fork_returns=1,
                additional_source_owners=1,
                velocity_zero_array_attempts=4,
                velocity_zero_array_returns=4,
                velocity_values_zeroed=1092,
                velocity_numeric_bytes_zeroed=8736,
                restore_attempts=1,
                restore_returns=1,
                fits=0,
                parameter_initializations=0,
                seeded_normal_values=0,
                parameter_momentum_numeric_bytes_per_source=17472,
            ),
        )
        holder = SimpleNamespace(
            report=report, candidate=None if zero_fault == "preparation" else zero
        )
        captured["zero"] = holder
        return holder

    if zero_velocity_fit:
        monkeypatch.setattr(study, "prepare_zero_velocity_bc", prepared_zero)

    def corrupt_preceding():
        if fault == "raw_prefix":
            source_raw["marker"] = "corrupted"
        if fault == "saved_prefix":
            captured["saved"]["continuation"]["complete"] = False
        if fault == "replaced_prefix":
            captured["saved"]["continuation"] = deepcopy(captured["saved"]["continuation"])
        if fault == "deleted_prefix":
            del captured["saved"]["continuation"]
        if fault == "replaced_raw":
            captured["continued"].report = dict(complete=False, replaced=True)

    def pushed(owner, *, label):
        ledger["routing_push_calls"] += 1
        push_labels.append(label)
        assert owner is (
            captured["aggregated"].candidate
            if label == "mixed128"
            else captured["corrected"].candidate
            if label == "mixed120"
            else captured["extended"].candidate
            if label == "mixed112"
            else source
            if label == "before"
            else candidate
        )
        if label == "after":
            corrupt_preceding()
        value = push_report()
        if label in ("mixed112", "mixed120", "mixed128"):
            value.update(
                source_unchanged=True,
                unreturned_prediction_possible=False,
                unreturned_native_work_unknown=False,
                transitions=value["transitions"] + [{}, {}],
            )
            if endpoint_fault == "positive":
                value.update(final_success=True, any_success=True, sparse_return=1.0)
            elif endpoint_fault == "push_incomplete":
                value["complete"] = False
            elif endpoint_fault == "push_unknown":
                value["unreturned_prediction_possible"] = True
            elif endpoint_fault == "push_reset":
                value["metadata"]["horizon"] = 49
            captured["native_push_raw"] = value
        return value

    def copied(value):
        result = deepcopy(value)
        if profile == "bp_mixed_retention" and value is mixed["after_push"]:
            corrupt_preceding()
        return result

    def picked(path, *, learner, train_seed, action_history_recipe):
        ledger["routing_pick_calls"] += 1
        assert learner is (
            captured["aggregated"].candidate
            if aggregated_correction_fit
            else captured["corrected"].candidate
            if correction_fit
            else captured["extended"].candidate
            if native_endpoint
            else candidate
        ) and train_seed in (17, 19, 20)
        value: dict[str, Any] = dict(
            complete=True,
            final_success=False,
            any_success=False,
            sparse_return=0,
            episode=dict(transitions=[{}, {}]),
        )
        if native_endpoint:
            value.update(
                seed=train_seed,
                source_updates=128 if aggregated_correction_fit else 120 if correction_fit else 112,
                source_training_rows=12600
                if aggregated_correction_fit
                else 11016
                if correction_fit
                else 9832,
                frozen_state_exact=True,
                original_owner_exact=True,
                action_history_exact=True,
                optimizer_updates=0,
                prediction_attempts=2,
                prediction_returns=2,
                role="train",
                task="PickCube-v1",
                context=f"pick-cube-train-{train_seed}",
                controller="pd_ee_delta_pos",
                observation_recipe="state42",
                action_history=dict(recipe=asdict(action_history_recipe)),
            )
            value["episode"]["transitions"][0] = dict(observation=(0.0,) * 42)
            if endpoint_fault == "positive":
                value.update(final_success=True, any_success=True, sparse_return=1.0)
            elif endpoint_fault == "pick_incomplete":
                value["complete"] = False
            elif endpoint_fault == "pick_unknown":
                value["prediction_returns"] = 1
            elif endpoint_fault == "pick_binding":
                value["source_updates"] = 48
            elif endpoint_fault == "pick_reset":
                value["episode"]["transitions"][0]["observation"] = (1e-15,) + (0.0,) * 41
            elif endpoint_fault == "prior_native":
                captured["native_push_raw"]["sparse_return"] = 7.0
        return SimpleNamespace(report=value, corrections=None)

    def diagnosed(owner, diagnostic_groups, *, label, guard, corrective=False, aggregated=False):
        ledger["routing_fit_calls"] += 1
        guard()
        if aggregated:
            assert aggregated_correction_fit and corrective
            assert tuple(name for name, _ in diagnostic_groups) == (
                "current_corrections",
                "corrections",
                "push",
                "a",
                "b",
                "train20",
            )
            assert tuple(len(item.transitions) for _, item in diagnostic_groups) == (
                50,
                50,
                18,
                15,
                50,
                15,
            )
            checkpoint = 120 if label == "aggregated120" else 128
            assert owner is captured["corrected" if checkpoint == 120 else "aggregated"].candidate
            result = scalar_corrective_readout(checkpoint, aggregated=True)
            if aggregated_fault == label:
                result["complete"] = False
            elif aggregated_fault == "paid_readout" and checkpoint == 128:
                captured["aggregated"].report["native_updates"][0]["diagnostic"]["value"] = 999.0
            fit_labels.append(label)
            fit_reports.append(result)
            return result
        if corrective:
            assert (
                correction_readouts
                or balanced_correction_fit
                or balanced_push_corrections
                or aggregated_correction_fit
            )
            assert tuple(name for name, _ in diagnostic_groups) == (
                "corrections",
                "push",
                "a",
                "b",
                "train20",
            )
            assert tuple(len(value.transitions) for _, value in diagnostic_groups) == (
                50,
                18,
                15,
                50,
                15,
            )
            assert (
                owner is captured["extended" if label == "corrective112" else "corrected"].candidate
            )
            result = scalar_corrective_readout(112 if label == "corrective112" else 120)
            if balanced_correction_fit or balanced_push_corrections or aggregated_correction_fit:
                assert label == "balanced120"
                result["label"] = label
                if balanced_fault == "readout":
                    result["complete"] = False
                elif balanced_fault == "unknown":
                    result["work"]["unknown_unreturned_work"] = True
                elif balanced_fault == "previous_readout":
                    uniform_readouts["corrective_readout120"]["label"] = "changed prior"
            if balanced_push_corrections:
                if collection_fault == "readout_incomplete":
                    result["complete"] = False
                elif collection_fault == "readout_mismatch":
                    result["groups"]["push"]["conditioning"]["conditioning"]["objective_loss"] += (
                        0.125
                    )
                elif collection_fault == "paid_readout":
                    captured["corrected"].report["events"][0]["extra"] = (
                        "changed preceding paid update"
                    )
            if (
                aggregated_correction_fit
                and aggregated_fault == "balanced_readout"
                and label == "balanced120"
            ):
                result["groups"]["push"]["rows"] = 17
            fit_labels.append(label)
            if zero_velocity_fit and label == "corrective120":
                if zero_fault == "readout":
                    result["complete"] = False
                elif zero_fault == "raw_preparation":
                    captured["zero"].report["work"]["velocity_zero_array_returns"] = 3
            if diagnostic_fault == label:
                result["complete"] = False
            elif diagnostic_fault == "unknown" and label == "corrective112":
                result["work"]["unknown_unreturned_work"] = True
            elif diagnostic_fault == "source_count" and label == "corrective112":
                result["source_updates"] = 111
            elif diagnostic_fault == "prior" and label == "corrective120":
                fit_reports[1]["label"] = "changed prior112"
            fit_reports.append(result)
            return result
        assert tuple(name for name, _ in diagnostic_groups) == ("push", "a", "b", "train20")
        assert tuple(len(value.transitions) for _, value in diagnostic_groups) == (18, 15, 50, 15)
        assert owner is (
            captured["extended"].candidate
            if label == "mixed112"
            else source
            if label == "source40"
            else candidate
        )
        fit_labels.append(label)
        result = dict(
            complete=(
                longer_fault != "longer_readout"
                if label == "mixed112"
                else readout_fault != ("fit_before" if label == "source40" else "fit_after")
            ),
            label=label,
            work={"prediction_calls": 0},
            operations=[],
        )
        if readout_fault == "prior_readout" and label != "source40":
            fit_reports[0]["label"] = "corrupted"
        if longer_fault == "prior_longer":
            captured["extended"].report["native_updates"][0]["diagnostic"]["value"] = 7.0
        fit_reports.append(result)
        return result

    times = iter((10.0, 11.0))

    probe_reports: list[dict[str, Any]] = []
    probe_plan = None
    if reserved_probes:
        from src.adapters.continuous_probe import ContinuousProbePlan

        reserved = tuple(
            replace(training, role=role, context="reserved", episode_id=role)
            for role in ("probe_train", "probe_test")
        )
        probe_plan = ContinuousProbePlan(reserved[0], reserved[1], updates=1, diagnostics=True)

    def load_probes():
        assert reads == ["control", "mixed", "training", "inputs"]
        return probe_plan, [{"fixture": "reserved input boundary"}]

    def probed(owner, plan, *, training_readouts, check_boundary):
        ledger["routing_probe_calls"] += 1
        check_boundary()
        assert plan is not probe_plan and plan == probe_plan
        assert tuple(len(item.transitions) for item in training_readouts) == (15, 22, 15)
        assert tuple(item.episode_id for item in training_readouts) == tuple(
            item.episode.episode_id for item in (inputs[0], inputs[1], inputs[4])
        )
        assert all(item.role == "train" for item in training_readouts)
        assert owner is (source if not probe_reports else candidate)
        before = not probe_reports
        diagnostic = dict(
            definition=study.OBJECTIVE if before else study.MIXED_OBJECTIVE, value=0.25
        )
        receipt = dict(
            diagnostic=diagnostic,
            update_number=1,
            observed_at=0,
            arrived_at=1,
            applied_at=1,
            episode_id=plan.training.episode_id,
            event_id="probe-label-0",
            sample_id="probe-batch-0",
            actor_version="probe-actor",
            learner_version="probe-candidate",
        )
        value = dict(
            complete=probe_fault != ("probe_before" if before else "probe_after"),
            receipt_observation_complete=True,
            plan=dict(
                updates=1,
                training_id=plan.training.episode_id,
                evaluation_id=plan.evaluation.episode_id,
                context=plan.training.context,
                diagnostics=True,
            ),
            source_updates=owner._state.completed_updates,
            source_training_examples=owner._state.training_examples,
            progress=dict(updates_completed=1, replay_examples_completed=0, next_tick=2),
            native_updates=[dict(diagnostic=diagnostic, training_rows=18)],
            applied_updates=[receipt],
            events=[dict(success_validated=True, deferred_exact=True, returned_receipts=[receipt])],
            checkpoints=[
                dict(complete=True, updates=owner._state.completed_updates + i) for i in (0, 1)
            ],
            metrics=dict(
                fresh_acquisition_gain=0.1,
                fresh_training_gain=0.2,
                old_retention_loss=dict(old_0=-0.1, old_1=0.0, old_2=0.1),
            ),
        )
        if before and probe_fault == "receipt":
            receipt["arrived_at"] = 2
        if before and probe_fault == "objective":
            diagnostic["definition"] = study.MIXED_OBJECTIVE
        if before and probe_fault == "source":
            source._state.parameters[3][0] = 0.125
        if before and probe_fault == "plan":
            value["plan"]["evaluation_id"] = "replaced-probe-test"
        if before and probe_fault == "plan_boolean":
            value["plan"]["updates"] = True
        probe_reports.append(value)
        return value

    def clock():
        value = next(times)
        if value == 11.0 and fault == "last_clock":
            source._state.parameters[3][0] = 0.125
        return value

    monkeypatch.setattr(study, "_write", capture)
    monkeypatch.setattr(study, "run_joint_readiness_stream", built)
    monkeypatch.setattr(study, "run_joint_readiness_corrections", acquired)
    monkeypatch.setattr(study, "run_continuous_train_continuation", continued)
    monkeypatch.setattr(study, "roll_push", pushed)
    monkeypatch.setattr(study, "deepcopy", copied)
    monkeypatch.setattr(study, "run_frozen_rollout", picked)
    monkeypatch.setattr(study, "run_training_readout", diagnosed)
    monkeypatch.setattr(
        study, "load_reserved_probe_inputs", load_probes if reserved_probes else forbidden
    )
    monkeypatch.setattr(study, "run_continuous_probe", probed if reserved_probes else forbidden)
    monkeypatch.setattr(study, "monotonic", clock)
    output = tmp_path / "run.json"
    before_picks = ledger["routing_pick_calls"]
    options: dict[str, Any] = dict(profile=profile) if profile != "head_gauss_newton" else {}
    if training_diagnostics:
        options["training_diagnostics"] = True
    if diagnostics_only:
        options["diagnostics_only"] = True
    if reserved_probes:
        options["reserved_probes"] = True
    if longer_fit:
        options["longer_fit"] = True
    if native_endpoint:
        options["native_endpoint"] = True
    if push_corrections:
        options["push_corrections"] = True
    if correction_fit:
        options["correction_fit"] = True
    if correction_readouts:
        options["correction_readouts"] = True
    if balanced_correction_fit:
        options["balanced_correction_fit"] = True
    if zero_velocity_fit:
        options["zero_velocity_fit"] = True
    if balanced_push_corrections:
        options["balanced_push_corrections"] = True
    if aggregated_correction_fit:
        options["aggregated_correction_fit"] = True
    result = study.run_study(output, **options)
    if aggregated_correction_fit:
        assert json.loads(output.read_text()) == json.loads(json.dumps(result))
        assert result["protocol"] == "push_cube_aggregated_correction_train_v1"
        expected_success = aggregated_fault is None and endpoint_fault is None
        assert result["complete"] is expected_success, result.get("error")
        if expected_success:
            assert fit_labels == ["mixed112", "balanced120", "aggregated120", "aggregated128"]
            assert push_labels == ["mixed128"] and ledger["routing_pick_calls"] - before_picks == 3
            assert result["aggregated_correction_protocol"]["initial_counts"] == [120, 11016]
            assert result["aggregated_correction_protocol"]["final_counts"] == [128, 12600]
            assert result["aggregated_continuation"]["rows"] == 198
            assert current_packet is not None
            assert result["current_correction_packet"]["episode_id"] == current_packet.episode_id
            comparison = result["comparisons"]["aggregated_native_vs_balanced120"]
            assert comparison["checkpoint"] == 128 and comparison["baseline_checkpoint"] == 120
            assert len(comparison["pick"]) == 3
            assert tuple(result["comparisons"]["aggregated_fit_vs120"]) == (
                "current_corrections",
                "corrections",
                "push",
                "a",
                "b",
                "train20",
            )
        else:
            assert result.get("error")
            if aggregated_fault is not None:
                assert not push_labels and ledger["routing_pick_calls"] == before_picks
        return result
    if balanced_push_corrections:
        assert json.loads(output.read_text()) == json.loads(json.dumps(result))
        assert result["protocol"] == "push_cube_balanced_student_corrections_train_v1"
        assert result["complete"] is (collection_fault is None), result.get("error")
        assert (
            not push_labels and not result["pick"] and ledger["routing_pick_calls"] == before_picks
        )
        assert "native_push" not in result["attempted"]
        if collection_fault is None:
            assert fit_labels == ["mixed112", "balanced120"]
            assert result["attempted"][-3:] == [
                "correction_continuation",
                "balanced_readout120",
                "push_corrections",
            ]
            assert result["push_correction_protocol"]["checkpoint"] == 120
            assert result["push_correction_protocol"]["source_training_rows"] == 11016
            assert (
                result["balanced_correction_protocol"]
                == inherited_balanced["balanced_correction_protocol"]
            )
            assert result["push_corrections"]["complete"] is True
            assert result["push_corrections"].get("final_success", False) is False
            packet = result["push_correction_target_packet"]
            assert len(packet["transitions"]) == 50
            assert result["push_correction_release_at"] == 1100
            assert tuple(packet["transitions"][1]["observation"][7:11]) == (0.0, 0.0, 0.0, 1.0)
            assert tuple(packet["transitions"][1]["action"]) == (0.0, 0.0, 0.0, -1.0)
            assert correction_packet is not None
            assert asdict(correction_packet) == original_correction_input
            assert asdict(correction_packet)["transitions"][1]["observation"][10] == -1.0
        else:
            assert "push_correction_target_packet" not in result
            if collection_fault in ("cache", "recipe"):
                assert result["attempted"] == ["source"]
            elif collection_fault in (
                "data",
                "reconstruction",
                "weights",
                "counter",
                "readout_incomplete",
                "readout_mismatch",
                "paid_readout",
            ):
                assert "push_corrections" not in result["attempted"]
                if collection_fault in ("readout_incomplete", "readout_mismatch"):
                    assert result["balanced_readout120"] is not None
                elif collection_fault == "paid_readout":
                    assert result["refused_return"]["stage"] == "balanced_readout120"
            else:
                assert result["attempted"][-1] == "push_corrections"
                assert (
                    result.get("push_corrections") is not None
                    or result.get("refused_return") is not None
                )
        return result
    if zero_velocity_fit:
        assert json.loads(output.read_text()) == json.loads(json.dumps(result))
        assert result["protocol"] == "push_cube_private_zero_velocity_fit_train_v1"
        assert result["complete"] is (zero_fault is None), result.get("error")
        assert (
            not push_labels and not result["pick"] and ledger["routing_pick_calls"] == before_picks
        )
        if zero_fault is None:
            assert fit_labels == ["mixed112", "corrective112", "corrective120"]
            assert result["zero_velocity_preparation"]["complete"] is True
            assert (
                result["correction_continuation"]["protocol"]
                == "continuous_train_row_weighted_mixed_gripper_continuation_v1"
            )
            assert result["correction_continuation"]["final_counts"] == dict(
                updates=120, training_rows=11016
            )
            assert tuple(result["comparisons"]["zero_velocity_fit_vs_inherited"]) == (
                "corrections",
                "push",
                "a",
                "b",
                "train20",
            )
            assert tuple(result["comparisons"]["zero_velocity_fit_vs_112"]) == (
                "corrections",
                "push",
                "a",
                "b",
                "train20",
            )
            assert result["zero_velocity_protocol"]["values"] == list(fixed_weights)
        elif zero_fault == "baseline":
            assert result["attempted"] == ["source"]
        else:
            assert result["zero_velocity_preparation"] is not None
            if zero_fault in ("preparation", "candidate"):
                assert "correction_continuation" not in result["attempted"]
            elif zero_fault in ("private", "prior"):
                assert result["refused_return"]["stage"] == "correction_continuation"
            elif zero_fault == "raw_preparation":
                assert result["refused_return"]["stage"] == "corrective_readout120"
            else:
                assert result["corrective_readout120"]["complete"] is False
        return result
    if balanced_correction_fit:
        assert json.loads(output.read_text()) == json.loads(json.dumps(result))
        assert result["protocol"] == "push_cube_prefix_group_balanced_correction_train_v1"
        assert result["complete"] is (balanced_fault is None), result.get("error")
        if balanced_fault is None:
            assert push_labels == ["mixed120"] and ledger["routing_pick_calls"] - before_picks == 3
            assert fit_labels == ["mixed112", "balanced120"]
            recipe = result["balanced_correction_protocol"]
            assert recipe["definition"] == "positive_train_prefix_group_weights_v1"
            assert recipe["values"][0] == 49.0 and len(recipe["values"]) == 148
            assert result["balanced_readout120"]["label"] == "balanced120"
            assert result["reused_uniform120_readout"] == uniform_readouts["corrective_readout120"]
            assert tuple(result["comparisons"]["balanced_fit_vs_uniform120"]) == (
                "corrections",
                "push",
                "a",
                "b",
                "train20",
            )
            assert tuple(result["comparisons"]["balanced_fit_vs_112"]) == (
                "corrections",
                "push",
                "a",
                "b",
                "train20",
            )
            native = result["comparisons"]["balanced_native_vs_uniform120"]
            assert native["push"]["controls"] == 3 and native["push"]["control_delta"] == -47
            assert len(native["pick"]) == 3
            assert all(row["control_delta"] == -48 for row in native["pick"])
        else:
            assert push_labels == [] and ledger["routing_pick_calls"] == before_picks
            assert "native_push" not in result["attempted"]
            if balanced_fault in ("readout", "unknown"):
                assert "balanced_readout120" in result
                assert result["correction_continuation"]["complete"] is True
            elif balanced_fault == "previous_readout":
                assert result["refused_return"]["stage"] == "balanced_readout120"
            elif balanced_fault == "uniform_cache":
                assert "correction_continuation" not in result["attempted"]
        return result
    if correction_readouts:
        assert json.loads(output.read_text()) == json.loads(json.dumps(result))
        assert (
            not push_labels and not result["pick"] and ledger["routing_pick_calls"] == before_picks
        )
        assert "native_push" not in result["attempted"] and "native_endpoint_protocol" not in result
        assert result["complete"] is (diagnostic_fault is None), result.get("error")
        if diagnostic_fault is None:
            assert fit_labels == ["mixed112", "corrective112", "corrective120"]
            assert result["attempted"][-3:] == [
                "corrective_readout112",
                "correction_continuation",
                "corrective_readout120",
            ]
            assert tuple(result["comparisons"]["corrective_fit_deltas"]) == (
                "corrections",
                "push",
                "a",
                "b",
                "train20",
            )
            for value in result["comparisons"]["corrective_fit_deltas"].values():
                assert value["native_objective"] == pytest.approx(-0.125)
                assert value["physical_half_mse"] == pytest.approx(-0.125)
                assert len(value["axes"]) == 4
                assert all(a["normalized_mae"] == pytest.approx(-0.125) for a in value["axes"])
            assert "all native scores historical" in result["comparisons"]["scope"]
        elif diagnostic_fault in ("corrective112", "unknown", "source_count", "cache"):
            assert "correction_continuation" not in result["attempted"]
        else:
            assert result["correction_continuation"]["complete"] is True
            if diagnostic_fault == "binding":
                assert "corrective_readout120" not in result["attempted"]
            elif diagnostic_fault == "prior":
                assert result["refused_return"]["stage"] == "corrective_readout120"
                assert result["corrective_readout112"]["label"] == "corrective112"
            else:
                assert result["corrective_readout120"]["complete"] is False
        return result
    if correction_fit:
        assert json.loads(output.read_text()) == json.loads(json.dumps(result))
        assert fit_labels == ["mixed112"] and "fit_after" not in result
        if fit_fault not in ("prior", "packet"):
            assert result["correction_continuation"] is not None
        assert result["correction_fit_protocol"]["rows"] == 148
        if fit_fault is None:
            assert result["complete"] is True, result.get("error")
            assert push_labels == ["mixed120"] and ledger["routing_pick_calls"] - before_picks == 3
            assert result["native_endpoint_protocol"]["checkpoint"] == 120
            assert result["attempted"][-5:] == [
                "correction_continuation",
                "native_push",
                "pick-17",
                "pick-19",
                "pick-20",
            ]
            assert result["comparisons"]["native_endpoint_vs_mixed112"] == dict(
                checkpoint=120,
                baseline_checkpoint=112,
                push=dict(
                    final_success=0.0,
                    any_success=0.0,
                    sparse_return=0.0,
                    controls=3,
                    control_delta=-47,
                ),
                pick=[
                    dict(
                        seed=seed,
                        final_success=-1.0 if seed == 17 else 0.0,
                        any_success=-1.0 if seed == 17 else 0.0,
                        sparse_return=-1.0 if seed == 17 else 0.0,
                        controls=2,
                        control_delta=2 - old,
                    )
                    for seed, old in zip((17, 19, 20), (41, 50, 50), strict=True)
                ],
            )
        else:
            assert result["complete"] is False and push_labels == []
            assert (
                ledger["routing_pick_calls"] == before_picks
                and "native_push" not in result["attempted"]
            )
            if fit_fault in ("prior", "packet"):
                assert result["refused_return"]["stage"] == "correction_continuation"
                assert result["refused_return"]["report"]["complete"] is True
            else:
                assert len(result["correction_continuation"]["native_updates"]) == 8
        return result
    if push_corrections:
        assert (
            not result["pick"] and not push_labels and ledger["routing_pick_calls"] == before_picks
        )
        assert result["complete"] is (correction_fault is None), result.get("error")
        if correction_fault is None:
            assert result["attempted"][-1] == "push_corrections" and fit_labels == ["mixed112"]
            assert (
                result["push_correction_target_packet"]["episode_id"]
                == "push-cube-correction-targets-17"
            )
            assert "supervision only" in result["comparisons"]["scope"]
        else:
            assert "push_correction_target_packet" not in result
        return result
    if native_endpoint:
        assert json.loads(output.read_text()) == json.loads(json.dumps(result))
        if endpoint_fault not in ("longer_cache", "retention_cache"):
            assert result["continuation"] == continued_raw
        if endpoint_fault in (None, "positive"):
            assert result["complete"] is True, result.get("error")
            assert result["source_unchanged"] is result["native_push_initial_exact"] is True
            assert push_labels == ["mixed112"] and fit_labels == ["mixed112"]
            assert ledger["routing_pick_calls"] - before_picks == 3
            assert result["attempted"][-4:] == ["native_push", "pick-17", "pick-19", "pick-20"]
            assert result["after_push"] == mixed["after_push"]
            assert result["native_endpoint_protocol"]["checkpoint"] == 112
            assert result["comparisons"]["native_endpoint_vs_mixed48"] == dict(
                checkpoint=112,
                baseline_checkpoint=48,
                push=dict(
                    final_success=float(endpoint_fault == "positive"),
                    any_success=float(endpoint_fault == "positive"),
                    sparse_return=float(endpoint_fault == "positive"),
                    controls=3,
                    control_delta=2,
                ),
                pick=[
                    dict(
                        seed=seed,
                        final_success=0.0 if endpoint_fault == "positive" else -1.0,
                        any_success=0.0 if endpoint_fault == "positive" else -1.0,
                        sparse_return=0.0 if endpoint_fault == "positive" else -1.0,
                        controls=2,
                        control_delta=2 - old,
                    )
                    for seed, old in zip((17, 19, 20), (16, 50, 14), strict=True)
                ],
            )
            assert [item["control_delta"] for item in result["comparisons"]["pick"]] == [
                -15,
                -18,
                -18,
            ]
            assert "source40" in result["comparisons"]["scope"]
        else:
            assert result["complete"] is False
            if endpoint_fault in (
                "longer_cache",
                "retention_cache",
                "longer_binding",
                "readout_binding",
            ):
                assert push_labels == [] and ledger["routing_pick_calls"] == before_picks
                assert "native_push" not in result["attempted"]
            elif endpoint_fault in ("push_incomplete", "push_unknown", "push_reset"):
                assert push_labels == ["mixed112"] and ledger["routing_pick_calls"] == before_picks
                assert "native_push" in result and result["native_push"] is not None
            else:
                assert ledger["routing_pick_calls"] - before_picks == 1
                assert "pick-19" not in result["attempted"]
                if endpoint_fault == "prior_native":
                    assert result["refused_return"]["stage"] == "pick-17"
                    assert result["native_push"]["sparse_return"] == 0
                else:
                    assert len(result["pick"]) == 1
        return result
    if longer_fit:
        assert (
            not result["pick"] and ledger["routing_pick_calls"] == before_picks and not push_labels
        )
        assert json.loads(output.read_text()) == json.loads(json.dumps(result))
        assert result["longer_fit_protocol"]["historical_after_push_checkpoint"] == 48
        assert result["continuation"] == continued_raw
        assert "probe_before" not in result and "fit_before" not in result
        if longer_fault is None:
            assert result["complete"] is True and result["source_unchanged"] is True
            assert result["longer_continuation"]["final_counts"] == dict(
                updates=112, training_rows=9832
            )
            assert result["longer_readout"]["label"] == "mixed112" and fit_labels == ["mixed112"]
            assert "historical Push scores belong to mixed48" in result["comparisons"]["scope"]
            assert result["after_push"] == mixed["after_push"]
        else:
            assert result["complete"] is False and "after_push" not in result["attempted"]
            if longer_fault in ("longer_incomplete", "counter", "receipt", "clock_type", "unknown"):
                assert "longer_readout" not in result["attempted"] and not fit_labels
            elif longer_fault == "longer_readout":
                assert result["longer_readout"]["complete"] is False
            else:
                assert result["refused_return"]["stage"] == "longer_readout"
                assert (
                    result["longer_continuation"]["native_updates"][0]["diagnostic"]["value"]
                    == 0.25
                )
            assert len(result["longer_continuation"]["native_updates"]) == 64
        return result
    if reserved_probes:
        assert (
            not result["pick"] and ledger["routing_pick_calls"] == before_picks and not push_labels
        )
        assert json.loads(output.read_text()) == json.loads(json.dumps(result))
        if probe_fault is None:
            assert result["complete"] is True and result["error"] is None
            assert result["source_unchanged"] is True
            assert result["attempted"].index("probe_before") < result["attempted"].index(
                "continuation"
            )
            assert result["attempted"].index("continuation") < result["attempted"].index(
                "probe_after"
            )
            assert result["comparisons"]["reserved_probe_metrics"] == {
                key: result[key]["metrics"] for key in ("probe_before", "probe_after")
            }
            assert (
                "different native objectives" in result["comparisons"]["reserved_probe_directions"]
            )
            assert "fit_before" not in result and "fit_after" not in result
        else:
            assert result["complete"] is False
            assert "after_push" not in result["attempted"]
            if probe_fault in (
                "probe_before",
                "receipt",
                "objective",
                "source",
                "plan",
                "plan_boolean",
            ):
                assert "continuation" not in result["attempted"]
                assert len(probe_reports) == 1
            if probe_fault == "prior_probe":
                assert result["refused_return"]["stage"] == "continuation"
                assert result["probe_before"]["metrics"]["fresh_acquisition_gain"] == 0.1
                assert "probe_after" not in result["attempted"]
            elif probe_fault == "source":
                assert result["refused_return"]["stage"] == "probe_before"
            else:
                assert result["probe_before"]["native_updates"][0]["training_rows"] == 18
        return result
    if training_diagnostics:
        assert fit_labels == (
            ["source40"]
            if readout_fault == "fit_before"
            else ["source40", "head48" if profile == "head_gauss_newton" else "mixed48"]
        )
        if readout_fault is not None:
            assert result["complete"] is False
            assert "after_push" not in result["attempted"]
            assert ledger["routing_pick_calls"] == before_picks and push_labels == []
            if readout_fault == "fit_before":
                assert result["fit_before"]["complete"] is False and result["continuation"] is None
            elif readout_fault == "fit_after":
                assert (
                    result["fit_after"]["complete"] is False
                    and result["continuation"]["complete"] is True
                )
            else:
                assert result["refused_return"]["stage"] == "fit_after"
                assert result["refused_return"]["report"]["complete"] is True
                assert result["fit_before"]["label"] == "source40"
            assert json.loads(output.read_text()) == json.loads(json.dumps(result))
            return result
        assert result["fit_before"]["label"] == "source40"
        assert result["fit_after"]["label"] == fit_labels[1]
        assert (
            result["attempted"].index("fit_before")
            < result["attempted"].index("continuation")
            < result["attempted"].index("fit_after")
        )
        if diagnostics_only:
            assert result["pick"] == result["comparisons"]["pick"] == []
            assert ledger["routing_pick_calls"] == before_picks and push_labels == []
            assert (
                result["comparisons"]["scope"]
                == "TRAIN readouts only; Push scores historical; no new native results"
            )
    else:
        assert fit_labels == [] and "fit_before" not in result and "fit_after" not in result
    assert push_labels == (
        []
        if profile == "bp_mixed_retention" or (fault == "reused_control" and profile == "bp_mixed")
        else ["after"]
        if profile == "bp_mixed"
        else ["before", "after"]
    )
    assert json.loads(output.read_text()) == json.loads(json.dumps(result))
    if fault is None or (fault == "reused_control" and profile == "head_gauss_newton"):
        assert result["complete"] is True and result["error"] is None
        assert result["comparisons"]["push"] == dict(
            final_success=0.0, any_success=0.0, sparse_return=0.0
        )
        assert all(row["final_success"] == -1.0 for row in result["comparisons"]["pick"])
        assert all(
            value == 1.0
            for value in result["comparisons"]["source_acquisition_fit_deltas"].values()
        )
        assert result["protocol"] == (
            "push_cube_mixed_retention_train_v1"
            if profile == "bp_mixed_retention"
            else "push_cube_mixed_continuation_train_v1"
            if profile == "bp_mixed"
            else "push_cube_continuation_train_v1"
        )
        if profile != "head_gauss_newton":
            assert result["reused_control"] == dict(
                path=str(study.REUSED_CONTROL),
                sha256=study.REUSED_CONTROL_SHA256,
                scope="accepted nonconcurrent common-MSE source40/2776",
            )
            assert (
                result["before_push"] == control["before_push"]
                and result["before_push"] is not control["before_push"]
            )
        else:
            assert "reused_control" not in result
        if profile == "bp_mixed_retention":
            assert result["reused_after_push"] == dict(
                path=str(study.REUSED_MIXED),
                sha256=study.REUSED_MIXED_SHA256,
                scope="accepted complete negative endpoint from failed mixed study",
            )
            assert (
                result["after_push"] == mixed["after_push"]
                and result["after_push"] is not mixed["after_push"]
            )
        else:
            assert "reused_after_push" not in result
    else:
        assert result["complete"] is False and result.get("evidence_error")
        if fault == "reused_control":
            expected_error = (
                "accepted reused mixed endpoint changed"
                if profile == "bp_mixed_retention"
                else "accepted reused control changed"
            )
            assert expected_error in result["evidence_error"]
            assert result["attempted"] == ["source"]
            assert result["refused_return"]["stage"] == "source"
        if fault in (
            "raw_prefix",
            "saved_prefix",
            "replaced_prefix",
            "deleted_prefix",
            "replaced_raw",
        ):
            assert ledger["routing_pick_calls"] == before_picks
            assert result["refused_return"]["stage"] == "after_push"
            assert result["refused_return"]["report"]["complete"] is True

            assert result["continuation"]["complete"] is True
            assert result["source"]["marker"] == "original"
            assert result["rejected_evidence"]
    return result


@pytest.mark.parametrize(
    "probe_fault",
    [
        None,
        "probe_before",
        "probe_after",
        "receipt",
        "objective",
        "source",
        "prior_probe",
        "plan",
        "plan_boolean",
    ],
)
def test_should_route_disposable_reserved_probes_and_refuse_incomplete_or_changed_paid_evidence(
    tmp_path, monkeypatch, source, ledger, probe_fault
):
    run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        reserved_probes=True,
        diagnostics_only=True,
        probe_fault=probe_fault,
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(reserved_probes=1),
        dict(reserved_probes=None),
        dict(reserved_probes=True),
        dict(profile="bp_mixed_retention", reserved_probes=True),
        dict(profile="bp_mixed", reserved_probes=True, diagnostics_only=True),
    ],
)
def test_should_refuse_invalid_reserved_probe_flags_before_io(tmp_path, monkeypatch, options):
    monkeypatch.setattr(study, "load_reserved_probe_inputs", forbidden)
    monkeypatch.setattr(study, "read_push_training", forbidden)
    monkeypatch.setattr(study, "read_reused_control", forbidden)
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        study.run_study(output, **options)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["native", "only", "fit_before", "fit_after", "prior_readout"])
def test_should_route_optional_train_diagnostics_and_stop_after_incomplete_or_corrupted_readout(
    tmp_path, monkeypatch, source, ledger, mode
):
    run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "head_gauss_newton" if mode == "native" else "bp_mixed_retention",
        training_diagnostics=True,
        diagnostics_only=mode != "native",
        readout_fault=mode if mode in ("fit_before", "fit_after", "prior_readout") else None,
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(training_diagnostics=1),
        dict(training_diagnostics="true"),
        dict(diagnostics_only=1),
        dict(diagnostics_only=None),
        dict(profile="bp_mixed_retention", diagnostics_only=True),
        dict(training_diagnostics=True, diagnostics_only=True),
        dict(profile="bp_mixed", training_diagnostics=True, diagnostics_only=True),
    ],
)
def test_should_refuse_invalid_train_diagnostic_flags_before_any_receipt_read_or_output(
    tmp_path, monkeypatch, options
):
    for name in (
        "read_reused_control",
        "read_reused_mixed",
        "read_push_training",
        "read_joint_acquisition_inputs",
    ):
        monkeypatch.setattr(study, name, forbidden)
    output = tmp_path / "absent.json"
    with pytest.raises(ValueError):
        study.run_study(output, **options)
    assert not output.exists()


def cached_control():
    """Synthetic producer evidence; no scientific payloads or helpers are read."""
    return dict(
        protocol="push_cube_continuation_train_v1",
        complete=True,
        source_unchanged=True,
        push_initial_exact=True,
        before_push=dict(complete=True),
        source=dict(stream=dict(elapsed_seconds=1.0, model_updates=24), complete=True),
        acquisition=dict(complete=True, elapsed_seconds=2.0),
        input_receipts={"kind": "synthetic"},
        training={"rows": 18},
        references=[{"rows": 15}, {"rows": 50}, {"rows": 15}],
        b_original={"source": "actual"},
        b_projected_actual={"history": "executed"},
        reused_push_sha256="synthetic-push-hash",
    )


def write_control(tmp_path, monkeypatch, raw):
    path = tmp_path / "synthetic-control.json"
    payload = json.dumps(raw, allow_nan=False).encode("utf8")
    path.write_bytes(payload)
    monkeypatch.setattr(study, "REUSED_CONTROL_SHA256", hashlib.sha256(payload).hexdigest())
    return path


def test_should_read_exact_accepted_control_without_aliasing_serialized_input(
    tmp_path, monkeypatch
):
    raw = cached_control()
    actual = study.read_reused_control(write_control(tmp_path, monkeypatch, raw))
    assert actual == raw and actual is not raw
    actual["source"]["stream"]["model_updates"] = 999
    assert raw["source"]["stream"]["model_updates"] == 24


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "oversize",
        "complete",
        "protocol",
        "source_unchanged",
        "push_initial_exact",
        "before_complete",
    ],
)
def test_should_refuse_replaced_oversized_or_incomplete_reused_control(
    tmp_path, monkeypatch, fault
):
    raw = cached_control()
    if fault in ("complete", "source_unchanged", "push_initial_exact"):
        raw[fault] = False
    elif fault == "protocol":
        raw["protocol"] = "other"
    elif fault == "before_complete":
        raw["before_push"]["complete"] = False
    path = write_control(tmp_path, monkeypatch, raw)
    if fault == "hash":
        monkeypatch.setattr(study, "REUSED_CONTROL_SHA256", "0" * 64)
    elif fault == "oversize":
        payload = b" " * (2 * 1024**2 + 1)
        path.write_bytes(payload)
        monkeypatch.setattr(study, "REUSED_CONTROL_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError):
        study.read_reused_control(path)


def test_should_ignore_only_stream_elapsed_value_without_mutating_either_evidence():
    control = cached_control()
    report = deepcopy(control)
    report["source"]["stream"]["elapsed_seconds"] = 99.0
    original = deepcopy(control), deepcopy(report)
    study._require_reused_source(control, report)
    assert (control, report) == original


@pytest.mark.parametrize(
    "key",
    [
        "source",
        "acquisition",
        "input_receipts",
        "training",
        "references",
        "b_original",
        "b_projected_actual",
        "reused_push_sha256",
    ],
)
def test_should_refuse_changed_source_or_training_evidence_beyond_the_single_clock_value(key):
    control = cached_control()
    report = deepcopy(control)
    if key == "source":
        report[key]["stream"]["model_updates"] = 25
    elif key == "acquisition":
        report[key]["elapsed_seconds"] = 3.0
    else:
        report[key] = {"changed": True}
    with pytest.raises(RuntimeError, match="reconstructed source differs"):
        study._require_reused_source(control, report)


@pytest.mark.parametrize("missing", ["control", "report"])
def test_should_require_original_elapsed_field_presence_in_both_producers(missing):
    control = cached_control()
    report = deepcopy(control)
    del (control if missing == "control" else report)["source"]["stream"]["elapsed_seconds"]
    with pytest.raises((KeyError, RuntimeError)):
        study._require_reused_source(control, report)


class ForeignProfile(str):
    pass


@pytest.mark.parametrize("fault", [None, "observation", "metadata"])
def test_should_compare_every_reset_value_across_live_and_cached_json_representations(fault):
    live: dict[str, Any] = dict(
        transitions=[dict(observation=(0.0,) * 12)],
        metadata=dict(
            task_id="PushCube-v1",
            seed=17,
            role="train",
            context="push-cube-train-17",
            recipe={"definition": "fixed"},
            controller={"frame": "root_translation"},
            horizon=50,
        ),
    )
    cached = json.loads(json.dumps(live, allow_nan=False))
    assert type(cached["transitions"][0]["observation"]) is list
    assert type(live["transitions"][0]["observation"]) is tuple
    if fault == "observation":
        live["transitions"][0]["observation"] = (0.0,) * 11 + (1e-15,)
    elif fault == "metadata":
        live["metadata"]["horizon"] = 49
    assert study._same_push_reset(cached, live) is (fault is None)


@pytest.mark.parametrize("profile", [None, True, "mixed", ForeignProfile("bp_mixed")])
def test_should_refuse_undeclared_profile_before_any_input_or_output(
    tmp_path, monkeypatch, profile
):
    monkeypatch.setattr(study, "read_reused_control", forbidden)
    monkeypatch.setattr(study, "read_push_training", forbidden)
    monkeypatch.setattr(study, "read_joint_acquisition_inputs", forbidden)
    output = tmp_path / "absent.json"
    with pytest.raises(ValueError):
        study.run_study(output, profile=profile)
    assert not output.exists()


def cached_mixed():
    """Synthetic failed study with a complete negative endpoint, never retried."""
    raw = cached_control()
    reset = dict(
        complete=True,
        source_unchanged=True,
        any_success=False,
        final_success=False,
        sparse_return=0.0,
        unreturned_prediction_possible=False,
        unreturned_native_work_unknown=False,
        transitions=[dict(observation=(0.0,) * 12)],
        metadata=dict(
            task_id="PushCube-v1",
            seed=17,
            role="train",
            context="push-cube-train-17",
            recipe={},
            controller={},
            horizon=50,
        ),
    )
    raw.update(
        protocol="push_cube_mixed_continuation_train_v1",
        complete=False,
        error="RuntimeError: fixed Push before/after reset binding differs",
        pick=[],
        unreturned_stage_work_unknown=False,
        before_push=reset,
        after_push=deepcopy(reset),
        continuation=dict(
            complete=True,
            source_exact=True,
            actor_exact=True,
            elapsed_seconds=1.0,
            unreturned_training_work_unknown=False,
            native_updates=[
                dict(
                    diagnostic=dict(definition="synthetic-mixed", value=1.0),
                    work={"published_updates": 8},
                )
            ],
            applied_updates=[dict(applied_at=1001)],
            progress={"updates_completed": 8},
            work={"training_rows": 784},
            policy_transition={"candidate_objective": "mixed"},
        ),
    )
    return json.loads(json.dumps(raw, allow_nan=False))


def write_mixed(tmp_path, monkeypatch, raw):
    path = tmp_path / "synthetic-mixed.json"
    payload = json.dumps(raw, allow_nan=False).encode("utf8")
    path.write_bytes(payload)
    monkeypatch.setattr(study, "REUSED_MIXED_SHA256", hashlib.sha256(payload).hexdigest())
    return path


def test_should_admit_only_bound_failed_mixed_study_without_upgrading_its_negative(
    tmp_path, monkeypatch
):
    raw = cached_mixed()
    actual = study.read_reused_mixed(write_mixed(tmp_path, monkeypatch, raw))
    assert actual == raw and actual is not raw
    assert actual["complete"] is False and actual["pick"] == []
    assert (
        actual["after_push"]["complete"] is True and actual["after_push"]["final_success"] is False
    )


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "oversize",
        "protocol",
        "complete",
        "error",
        "pick",
        "continuation",
        "source_exact",
        "actor_exact",
        "after_complete",
        "after_source",
        "any_success",
        "final_success",
        "sparse_return",
        "reset",
        "stage_unknown",
        "training_unknown",
        "prediction_unknown",
        "native_unknown",
    ],
)
def test_should_refuse_replaced_unfinished_unknown_or_different_mixed_endpoint(
    tmp_path, monkeypatch, fault
):
    raw = cached_mixed()
    if fault == "protocol":
        raw["protocol"] = "other"
    elif fault == "complete":
        raw["complete"] = True
    elif fault == "error":
        raw["error"] = "different failure"
    elif fault == "pick":
        raw["pick"] = [{"already_attempted": True}]
    elif fault in ("continuation", "source_exact", "actor_exact"):
        raw["continuation"]["complete" if fault == "continuation" else fault] = False
    elif fault in ("after_complete", "after_source"):
        raw["after_push"]["complete" if fault == "after_complete" else "source_unchanged"] = False
    elif fault in ("any_success", "final_success"):
        raw["after_push"][fault] = True
    elif fault == "sparse_return":
        raw["after_push"]["sparse_return"] = 1.0
    elif fault == "reset":
        raw["after_push"]["transitions"][0]["observation"][11] = 1e-15
    elif fault == "stage_unknown":
        raw["unreturned_stage_work_unknown"] = True
    elif fault == "training_unknown":
        raw["continuation"]["unreturned_training_work_unknown"] = True
    elif fault == "prediction_unknown":
        raw["after_push"]["unreturned_prediction_possible"] = True
    elif fault == "native_unknown":
        raw["after_push"]["unreturned_native_work_unknown"] = True
    path = write_mixed(tmp_path, monkeypatch, raw)
    if fault == "hash":
        monkeypatch.setattr(study, "REUSED_MIXED_SHA256", "0" * 64)
    elif fault == "oversize":
        payload = b" " * (2 * 1024**2 + 1)
        path.write_bytes(payload)
        monkeypatch.setattr(study, "REUSED_MIXED_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError):
        study.read_reused_mixed(path)


def test_should_ignore_only_present_continuation_root_elapsed_value():
    control = cached_mixed()
    report = deepcopy(control)
    report["continuation"]["elapsed_seconds"] = 99.0
    original = deepcopy(control), deepcopy(report)
    study._require_reused_continuation(control, report)
    assert (control, report) == original


@pytest.mark.parametrize(
    "key", ["native_updates", "applied_updates", "progress", "work", "policy_transition"]
)
def test_should_refuse_any_changed_continuation_work_receipt_objective_or_diagnostic(key):
    control = cached_mixed()
    report = deepcopy(control)
    report["continuation"][key] = {"changed": True}
    with pytest.raises(RuntimeError, match="reconstructed mixed continuation differs"):
        study._require_reused_continuation(control, report)


@pytest.mark.parametrize("missing", ["control", "report"])
def test_should_require_both_original_continuation_elapsed_fields(missing):
    control = cached_mixed()
    report = deepcopy(control)
    del (control if missing == "control" else report)["continuation"]["elapsed_seconds"]
    with pytest.raises(KeyError):
        study._require_reused_continuation(control, report)


@pytest.mark.parametrize(
    "longer_fault",
    [
        None,
        "longer_incomplete",
        "counter",
        "receipt",
        "clock_type",
        "unknown",
        "longer_readout",
        "prior_longer",
    ],
)
def test_should_route_fixed_longer_train_fit_and_keep_historical_push_at_eight_updates(
    tmp_path, monkeypatch, source, ledger, longer_fault
):
    run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        longer_fit=True,
        diagnostics_only=True,
        longer_fault=longer_fault,
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(longer_fit=1),
        dict(longer_fit=None),
        dict(longer_fit=True),
        dict(longer_fit=True, diagnostics_only=True, profile="bp_mixed"),
        dict(
            longer_fit=True,
            diagnostics_only=True,
            profile="bp_mixed_retention",
            training_diagnostics=True,
        ),
        dict(
            longer_fit=True,
            diagnostics_only=True,
            profile="bp_mixed_retention",
            reserved_probes=True,
        ),
    ],
)
def test_should_refuse_invalid_or_combined_longer_fit_flags_before_io(
    tmp_path, monkeypatch, options
):
    monkeypatch.setattr(study, "read_push_training", forbidden)
    monkeypatch.setattr(study, "read_reused_control", forbidden)
    output = tmp_path / "invalid-longer.json"
    with pytest.raises(ValueError):
        study.run_study(output, **options)
    assert not output.exists()


@pytest.mark.parametrize(
    "options",
    [
        dict(native_endpoint=1),
        dict(native_endpoint=None),
        dict(native_endpoint=True),
        dict(native_endpoint=True, longer_fit=True),
        dict(native_endpoint=True, longer_fit=True, profile="bp_mixed"),
        dict(
            native_endpoint=True,
            longer_fit=True,
            profile="bp_mixed_retention",
            diagnostics_only=True,
        ),
        dict(
            native_endpoint=True,
            longer_fit=True,
            profile="bp_mixed_retention",
            training_diagnostics=True,
        ),
        dict(
            native_endpoint=True,
            longer_fit=True,
            profile="bp_mixed_retention",
            reserved_probes=True,
        ),
    ],
)
def test_should_refuse_invalid_native_endpoint_selection_before_any_read_or_output(
    tmp_path, monkeypatch, options
):
    for name in (
        "read_reused_control",
        "read_reused_mixed",
        "read_reused_longer",
        "read_reused_retention",
        "read_push_training",
        "read_joint_acquisition_inputs",
    ):
        monkeypatch.setattr(study, name, forbidden)
    output = tmp_path / "invalid-native-endpoint.json"
    with pytest.raises(ValueError):
        study.run_study(output, **options)
    assert not output.exists()


@pytest.mark.parametrize(
    "endpoint_fault",
    [
        None,
        "positive",
        "push_incomplete",
        "push_unknown",
        "push_reset",
        "pick_incomplete",
        "pick_unknown",
        "pick_binding",
        "pick_reset",
        "prior_native",
        "longer_cache",
        "retention_cache",
        "longer_binding",
        "readout_binding",
    ],
)
def test_should_route_only_reproduced_112_native_endpoint_and_preserve_ordered_negative_or_partial_returns(
    tmp_path, monkeypatch, source, ledger, endpoint_fault
):
    run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        longer_fit=True,
        native_endpoint=True,
        endpoint_fault=endpoint_fault,
    )


def cached_native_endpoint(kind):
    raw = cached_control()
    raw.update(
        complete=True,
        source_unchanged=True,
        unreturned_stage_work_unknown=False,
        continuation=dict(complete=True, elapsed_seconds=1.0),
    )
    if kind == "longer":
        raw.update(
            pick=[],
            longer_continuation=synthetic_longer_continuation(),
            longer_readout=dict(complete=True, label="mixed112", work={"rows": 98}),
        )
    else:
        raw.update(
            protocol="push_cube_mixed_retention_train_v1",
            after_push=dict(complete=True),
            pick=[dict(seed=seed, complete=True) for seed in (17, 19, 20)],
        )
    return raw


def write_native_endpoint(tmp_path, monkeypatch, kind, raw):
    path = tmp_path / ("synthetic-" + kind + ".json")
    payload = json.dumps(raw, allow_nan=False).encode("utf8")
    path.write_bytes(payload)
    monkeypatch.setattr(
        study, "REUSED_" + kind.upper() + "_SHA256", hashlib.sha256(payload).hexdigest()
    )
    return path


@pytest.mark.parametrize("kind", ["longer", "retention"])
def test_should_read_exact_owned_accepted_native_endpoint_cache_without_model_or_label_work(
    tmp_path, monkeypatch, kind
):
    raw = cached_native_endpoint(kind)
    reader = study.read_reused_longer if kind == "longer" else study.read_reused_retention
    actual = reader(write_native_endpoint(tmp_path, monkeypatch, kind, raw))
    assert actual == raw and actual is not raw
    actual["source"]["complete"] = False
    assert raw["source"]["complete"] is True


@pytest.mark.parametrize("kind", ["longer", "retention"])
@pytest.mark.parametrize("fault", ["hash", "oversize", "complete", "unknown", "endpoint"])
def test_should_refuse_replaced_or_incomplete_native_endpoint_cache_before_reconstruction(
    tmp_path, monkeypatch, kind, fault
):
    raw = cached_native_endpoint(kind)
    if fault == "complete":
        raw["complete"] = False
    elif fault == "unknown":
        raw["unreturned_stage_work_unknown"] = True
    elif fault == "endpoint":
        if kind == "longer":
            raw["longer_readout"]["complete"] = False
        else:
            raw["pick"][1]["seed"] = 17
    path = write_native_endpoint(tmp_path, monkeypatch, kind, raw)
    if fault == "hash":
        monkeypatch.setattr(study, "REUSED_" + kind.upper() + "_SHA256", "0" * 64)
    elif fault == "oversize":
        payload = b" " * (2 * 1024**2 + 1)
        path.write_bytes(payload)
        monkeypatch.setattr(
            study, "REUSED_" + kind.upper() + "_SHA256", hashlib.sha256(payload).hexdigest()
        )
    reader = study.read_reused_longer if kind == "longer" else study.read_reused_retention
    with pytest.raises(ValueError):
        reader(path)


@pytest.mark.parametrize(
    "fault",
    [None, "native", "readout", "nested_elapsed", "missing_cached_elapsed", "missing_live_elapsed"],
)
def test_should_bind_whole_112_continuation_and_readout_exempting_only_present_root_elapsed_value(
    fault,
):
    control = cached_native_endpoint("longer")
    report = deepcopy(control)
    report["longer_continuation"]["elapsed_seconds"] = 99.0
    if fault == "native":
        report["longer_continuation"]["native_updates"][0]["diagnostic"]["value"] += 0.125
    elif fault == "readout":
        report["longer_readout"]["work"]["rows"] = 97
    elif fault == "nested_elapsed":
        report["longer_readout"]["elapsed_seconds"] = 3.0
    elif fault == "missing_cached_elapsed":
        del control["longer_continuation"]["elapsed_seconds"]
    elif fault == "missing_live_elapsed":
        del report["longer_continuation"]["elapsed_seconds"]
    originals = deepcopy(control), deepcopy(report)
    if fault is None:
        study._require_reused_longer(control, report)
    else:
        with pytest.raises((RuntimeError, KeyError)):
            study._require_reused_longer(control, report)
    assert (control, report) == originals


@pytest.mark.parametrize("correction_fault", [None, "incomplete", "trajectory", "labels", "prior"])
def test_should_collect_only_frozen_112_corrections_and_refuse_changed_prior_evidence(
    tmp_path, monkeypatch, source, ledger, correction_fault
):
    run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        longer_fit=True,
        push_corrections=True,
        correction_fault=correction_fault,
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(push_corrections=1),
        dict(push_corrections=True),
        dict(push_corrections=True, longer_fit=True),
        dict(
            push_corrections=True,
            longer_fit=True,
            profile="bp_mixed_retention",
            native_endpoint=True,
        ),
        dict(
            push_corrections=True,
            longer_fit=True,
            profile="bp_mixed_retention",
            diagnostics_only=True,
        ),
    ],
)
def test_should_refuse_correction_modes_before_cache_or_label_access(
    tmp_path, monkeypatch, options
):
    monkeypatch.setattr(study, "read_reused_control", forbidden)
    with pytest.raises(ValueError):
        study.run_study(tmp_path / "invalid-corrections.json", **options)


@pytest.mark.parametrize(
    "options",
    [
        dict(correction_fit=1),
        dict(correction_fit=True),
        dict(correction_fit=True, longer_fit=True, native_endpoint=True, profile="bp_mixed"),
        dict(correction_fit=True, native_endpoint=True, profile="bp_mixed_retention"),
        dict(correction_fit=True, longer_fit=True, profile="bp_mixed_retention"),
        dict(
            correction_fit=True,
            longer_fit=True,
            native_endpoint=True,
            profile="bp_mixed_retention",
            push_corrections=True,
        ),
    ],
)
def test_should_refuse_invalid_corrective_fit_flags_before_any_source_or_correction_input(
    tmp_path, monkeypatch, options
):
    for name in (
        "read_reused_control",
        "read_reused_mixed",
        "read_reused_longer",
        "read_reused_retention",
        "read_reused_native",
        "load_push_correction_inputs",
        "read_push_training",
        "read_joint_acquisition_inputs",
    ):
        monkeypatch.setattr(study, name, forbidden)
    output = tmp_path / "invalid-corrective-fit.json"
    with pytest.raises(ValueError):
        study.run_study(output, **options)
    assert not output.exists()


@pytest.mark.parametrize("fit_fault", [None, "incomplete", "counter", "receipt", "prior", "packet"])
def test_should_route_all148_corrective_fit_from112_to120_and_preserve_failed_paid_prefix_before_native(
    tmp_path, monkeypatch, source, ledger, fit_fault
):
    run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        longer_fit=True,
        native_endpoint=True,
        correction_fit=True,
        fit_fault=fit_fault,
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(correction_readouts=1),
        dict(correction_readouts=True),
        dict(
            correction_readouts=True,
            correction_fit=True,
            longer_fit=True,
            profile="bp_mixed_retention",
        ),
        dict(
            correction_readouts=True,
            correction_fit=True,
            longer_fit=True,
            profile="bp_mixed_retention",
            native_endpoint=True,
        ),
        dict(
            correction_readouts=True,
            longer_fit=True,
            diagnostics_only=True,
            profile="bp_mixed_retention",
        ),
        dict(
            correction_readouts=True,
            correction_fit=True,
            longer_fit=True,
            diagnostics_only=True,
            profile="bp_mixed_retention",
            training_diagnostics=True,
        ),
    ],
)
def test_should_refuse_invalid_corrective_readout_mode_before_cache_or_model(
    tmp_path, monkeypatch, options
):
    for name in (
        "read_reused_control",
        "read_reused_mixed",
        "read_reused_longer",
        "read_reused_retention",
        "read_reused_native",
        "read_reused_corrective",
        "load_push_correction_inputs",
    ):
        monkeypatch.setattr(study, name, forbidden)
    with pytest.raises(ValueError):
        study.run_study(tmp_path / "refused-readout.json", **options)


@pytest.mark.parametrize(
    "diagnostic_fault",
    [
        None,
        "corrective112",
        "corrective120",
        "unknown",
        "source_count",
        "prior",
        "binding",
        "cache",
    ],
)
def test_should_measure_five_fixed_groups_without_sdk_and_refuse_unvalidated_or_changed_prior_evidence(
    tmp_path, monkeypatch, source, ledger, diagnostic_fault
):
    run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        longer_fit=True,
        diagnostics_only=True,
        correction_fit=True,
        correction_readouts=True,
        diagnostic_fault=diagnostic_fault,
    )


@pytest.mark.parametrize("fault", [None, "hash", "oversize", "complete", "unknown", "receipt"])
def test_should_bind_whole_accepted120_corrective_scope_before_diagnostics(
    tmp_path, monkeypatch, fault
):
    raw = dict(
        complete=True,
        source_unchanged=True,
        unreturned_stage_work_unknown=False,
        correction_fit_protocol=dict(rows=148),
        correction_continuation=synthetic_longer_continuation(correction_fit=True),
    )
    if fault == "complete":
        raw["complete"] = False
    if fault == "unknown":
        raw["unreturned_stage_work_unknown"] = True
    if fault == "receipt":
        raw["correction_continuation"]["applied_updates"][0]["arrived_at"] += 1
    path = write_native_endpoint(tmp_path, monkeypatch, "corrective", raw)
    if fault == "hash":
        monkeypatch.setattr(study, "REUSED_CORRECTIVE_SHA256", "0" * 64)
    if fault == "oversize":
        payload = b" " * (4 * 1024**2 + 1)
        path.write_bytes(payload)
        monkeypatch.setattr(study, "REUSED_CORRECTIVE_SHA256", hashlib.sha256(payload).hexdigest())
    if fault is None:
        assert study.read_reused_corrective(path) == raw
    else:
        with pytest.raises((ValueError, RuntimeError)):
            study.read_reused_corrective(path)


@pytest.mark.parametrize(
    "options",
    [
        dict(balanced_correction_fit=1),
        dict(balanced_correction_fit=True),
        dict(
            profile="bp_mixed",
            balanced_correction_fit=True,
            longer_fit=True,
            native_endpoint=True,
            correction_fit=True,
        ),
        dict(
            profile="bp_mixed_retention",
            balanced_correction_fit=True,
            native_endpoint=True,
            correction_fit=True,
        ),
        dict(
            profile="bp_mixed_retention",
            balanced_correction_fit=True,
            longer_fit=True,
            correction_fit=True,
        ),
        dict(
            profile="bp_mixed_retention",
            balanced_correction_fit=True,
            longer_fit=True,
            native_endpoint=True,
        ),
        dict(
            profile="bp_mixed_retention",
            balanced_correction_fit=True,
            longer_fit=True,
            native_endpoint=True,
            correction_fit=True,
            correction_readouts=True,
        ),
        dict(
            profile="bp_mixed_retention",
            balanced_correction_fit=True,
            longer_fit=True,
            native_endpoint=True,
            correction_fit=True,
            training_diagnostics=True,
        ),
    ],
)
def test_should_refuse_invalid_balanced_selection_before_any_control_or_training_input(
    tmp_path,
    monkeypatch,
    options,
):
    for name in (
        "read_reused_control",
        "read_reused_mixed",
        "read_reused_longer",
        "read_reused_retention",
        "read_reused_native",
        "read_reused_corrective",
        "read_reused_corrective_readouts",
        "load_push_correction_inputs",
        "read_push_training",
        "read_joint_acquisition_inputs",
    ):
        monkeypatch.setattr(study, name, forbidden)
    output = tmp_path / "invalid-balanced.json"
    with pytest.raises(ValueError):
        study.run_study(output, **options)
    assert not output.exists()


@pytest.mark.parametrize(
    "balanced_fault",
    [
        None,
        "incomplete",
        "receipt",
        "prior",
        "readout",
        "unknown",
        "uniform_cache",
        "previous_readout",
    ],
)
def test_should_route_one_fixed_balanced_fit_and_stop_after_untrusted_or_partial_paid_evidence(
    tmp_path,
    monkeypatch,
    source,
    ledger,
    balanced_fault,
):
    run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        longer_fit=True,
        native_endpoint=True,
        correction_fit=True,
        balanced_correction_fit=True,
        balanced_fault=balanced_fault,
    )


@pytest.mark.parametrize(
    "fault", [None, "hash", "oversize", "incomplete", "readout", "receipt", "order"]
)
def test_should_read_only_bound_complete_uniform_readouts_without_recomputing_models(
    tmp_path,
    monkeypatch,
    fault,
):
    raw = dict(
        complete=True,
        source_unchanged=True,
        unreturned_stage_work_unknown=False,
        correction_continuation=synthetic_longer_continuation(correction_fit=True),
        corrective_readout112=scalar_corrective_readout(112),
        corrective_readout120=scalar_corrective_readout(120),
    )
    if fault == "incomplete":
        raw["complete"] = False
    elif fault == "readout":
        raw["corrective_readout120"]["source_updates"] = 112
    elif fault == "receipt":
        raw["correction_continuation"]["applied_updates"][0]["arrived_at"] += 1
    elif fault == "order":
        raw["corrective_readout120"]["group_order"].reverse()
    path = write_native_endpoint(tmp_path, monkeypatch, "corrective_readouts", raw)
    # Match the actual producer's canonical JSON; map order is not group order.
    payload = json.dumps(raw, sort_keys=True).encode("utf-8")
    path.write_bytes(payload)
    monkeypatch.setattr(
        study, "REUSED_CORRECTIVE_READOUTS_SHA256", hashlib.sha256(payload).hexdigest()
    )
    if fault == "hash":
        monkeypatch.setattr(study, "REUSED_CORRECTIVE_READOUTS_SHA256", "0" * 64)
    elif fault == "oversize":
        payload = b" " * (4 * 1024**2 + 1)
        path.write_bytes(payload)
        monkeypatch.setattr(
            study, "REUSED_CORRECTIVE_READOUTS_SHA256", hashlib.sha256(payload).hexdigest()
        )
    if fault is None:
        assert study.read_reused_corrective_readouts(path) == raw
    else:
        with pytest.raises((ValueError, RuntimeError)):
            study.read_reused_corrective_readouts(path)
