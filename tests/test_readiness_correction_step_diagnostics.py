"""First paid correction-step evidence, without a second gradient oracle.

Per platform: one D12/H2/A4 initializer, 32 normal values/P38; one warm,
eight diagnostic, eight default and one failed-diagnostic update =18 SGD36rows.
Six read-only physical gradients cover8 overlapping reference rows:24 total
forward/backward calls44rows,912 gradient values7296B,3168 dense MAC.
24 returned normalizations44rows528coordinates plus one failed attempt.
Zero fits, prediction calls, replay, PC inference, scenes or native physics.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_nearest_continuous_bc as nearest
from src.adapters import readiness_step_diagnostics as leaf
from src.core import continuous_bc as task
from src.core import continuous_step_diagnostics as direction
from src.core import pick_cube_action_history as history
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.pick_cube_action_history_corrections import action_history_correction_view
from src.core.pick_cube_public_vectors import public_vector_projection
from src.core.readiness_stage_diagnostics import ReadinessStageBinding
from test_goal_consistent_observed_grasp_corrections import RECIPE, batch, raw
from test_observed_grasp_corrections import assert_state


def forbidden(*args, **kwargs):
    pytest.fail("readiness reference diagnostics admitted extra private work")


def references():
    """Actual recorded commands, not correction targets or inferred phases."""
    result = []
    actions = ((0.25, -0.5, 0.125, -1.0), (0.0, 0.0, 0.0, -1.0))
    for seed, offset in ((17, 0), (19, 2)):
        episode = ContinuousEpisode(
            f"pick-cube-readiness-teacher-{seed}",
            "pick-cube" if seed == 17 else "pick-cube-train-19",
            "train",
            batch().episode.spec,
            tuple(
                ContinuousTransition(
                    index,
                    raw(offset + index),
                    action,
                    float(index == 1),
                    raw(offset + index + 1),
                    index == 1,
                    False,
                )
                for index, action in enumerate(actions)
            ),
        )
        result.append(
            ReadinessStageBinding(
                history.action_history_training_episode(episode, RECIPE), ("lift", "settle")
            )
        )
    return result[0], result[1]


@pytest.fixture(scope="module")
def source():
    view = action_history_correction_view(batch(), RECIPE)
    owner = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(view.model.episode.spec, hidden_dim=2, seed=17, binary_gripper_axis=3)
    )
    features, targets = view.model.training_pairs(now=3)
    owner.train_batch(np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64))
    assert any(np.any(value != 0.0) for value in owner.snapshot_state().velocity)
    return owner


@pytest.fixture(scope="module")
def completed(source):
    captured: dict[str, Any] = dict(trains=[], diagnostics=[])
    owner, initial, bindings = source.fork(), source.snapshot_state(), references()
    runtime_factory = demo.ActorShadowRuntime
    train = bp.NumpyContinuousBCLearner.train_batch
    diagnose = leaf.diagnose_step_direction

    def runtime(*args, **options):
        value = runtime_factory(*args, **options)
        captured.update(runtime=value, budget=options["budget"])
        return value

    def record_train(learner, features, targets):
        before = learner.snapshot_state()
        diagnostic = train(learner, features, targets)
        captured["trains"].append(
            dict(before=before, after=learner.snapshot_state(), diagnostic=diagnostic)
        )
        return diagnostic

    def record_direction(features, targets, before, velocity, after, spec, **options):
        record = dict(
            features=features.copy(),
            targets=targets.copy(),
            before=deepcopy(before),
            velocity=deepcopy(velocity),
            after=deepcopy(after),
            spec=spec,
            learning_rate=options["learning_rate"],
            momentum=options["momentum"],
        )
        gradient = direction.task.continuous_bc_loss_and_gradients
        calls = []

        def paid_gradient(*args, **keywords):
            assert keywords["binary_gripper_axis"] is None
            value = gradient(*args, **keywords)
            calls.append(deepcopy(value.gradients))
            return value

        # Capture the gradient already requested by this diagnostic. No oracle
        # forward/backward is added, and native mixed training is outside scope.
        with pytest.MonkeyPatch.context() as nested:
            nested.setattr(direction.task, "continuous_bc_loss_and_gradients", paid_gradient)
            value = diagnose(features, targets, before, velocity, after, spec, **options)
        assert len(calls) == 1
        record.update(gradients=calls[0], result=asdict(value))
        captured["diagnostics"].append(record)
        return value

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", record_train)
        patch.setattr(leaf, "diagnose_step_direction", record_direction)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        result = demo.run_continuous_correction_update(
            owner,
            batch(),
            now=3,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
            diagnostic_stage_bindings=bindings,
        )
    assert result.report["complete"] is True and result.candidate is not None
    return dict(owner=owner, initial=initial, bindings=bindings, result=result, **captured)


@pytest.fixture(scope="module")
def default_run(source):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(leaf, "diagnose_step_direction", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        result = demo.run_continuous_correction_update(
            source.fork(),
            batch(),
            now=3,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
        )
    assert result.report["complete"] is True and result.candidate is not None
    return result


def test_should_diagnose_only_the_first_paid_step_with_distinct_native_and_physical_objectives(
    completed,
):
    report = completed["result"].report
    diagnostic = report["step_diagnostics"]
    assert diagnostic["complete"] is True
    assert diagnostic["definition"] == "readiness_reference_realized_step_direction_v1"
    assert diagnostic["native_objective_definition"] == task.MIXED_OBJECTIVE
    assert diagnostic["physical_objective_definition"] == task.OBJECTIVE
    assert (diagnostic["source_updates"], diagnostic["after_updates"]) == (1, 2)
    assert len(completed["trains"]) == len(report["applied_updates"]) == 8
    assert len(completed["diagnostics"]) == 6
    final = completed["result"].candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (9, 18)
    assert_state(completed["initial"], completed["owner"].snapshot_state())
    assert_state(completed["initial"], completed["runtime"].actor.snapshot_state().state)
    assert_state(final, completed["runtime"].candidate_snapshot().state)
    assert completed["budget"].updates_completed == 8
    assert completed["budget"].replay_examples_completed == 0
    assert report["progress"]["next_tick"] == 4
    assert all(
        row["diagnostic"]["definition"] == task.MIXED_OBJECTIVE for row in report["applied_updates"]
    )


@pytest.mark.parametrize("context", ["warm", "new"])
def test_should_keep_actual_reference_phases_and_empty_groups_without_extra_gradients(
    completed, context
):
    diagnostic = completed["result"].report["step_diagnostics"]
    position = 0 if context == "warm" else 1
    binding = completed["bindings"][position]
    metadata = diagnostic["metadata"][context]
    assert metadata["episode_id"] == binding.episode.episode_id
    assert metadata["stages"] == ("lift", "settle") and metadata["rows"] == 2
    groups = diagnostic["contexts"][context]["groups"]
    assert tuple(groups) == ("overall", "align", "descend", "grasp", "lift", "settle")
    for name in ("align", "descend", "grasp"):
        assert groups[name] == dict(
            rows=0,
            complete=True,
            result=None,
            work=asdict(direction._StepDirectionWork()),
            input_numeric_bytes=0,
        )
    for name, rows in (("overall", 2), ("lift", 1), ("settle", 1)):
        group = groups[name]
        assert group["rows"] == rows and group["complete"] is True
        assert group["result"]["objective_definition"] == task.OBJECTIVE
        assert all(value == 1 for value in group["work"].values())
    captures = completed["diagnostics"][3 * position : 3 * position + 3]
    expected_indices = ((0, 1), (0,), (1,))
    for captured, indices in zip(captures, expected_indices, strict=True):
        assert captured["features"].tolist() == [
            list(binding.episode.transitions[i].observation) for i in indices
        ]
        assert captured["targets"].tolist() == [
            list(binding.episode.transitions[i].action) for i in indices
        ]
    assert captures[0]["targets"].tolist() != [list(label.target) for label in batch().labels]


def test_should_measure_realized_step_against_already_paid_physical_gradients_and_old_momentum(
    completed,
):
    first = completed["trains"][0]
    for capture in completed["diagnostics"]:
        for actual, expected in zip(capture["before"], first["before"].parameters, strict=True):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(capture["velocity"], first["before"].velocity, strict=True):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(capture["after"], first["after"].parameters, strict=True):
            np.testing.assert_array_equal(actual, expected)
        dots, momenta, sgd_dots = [], [], []
        for name, g, before, after, velocity, group in zip(
            ("w1", "b1", "w2", "b2"),
            capture["gradients"],
            capture["before"],
            capture["after"],
            capture["velocity"],
            capture["result"]["groups"],
            strict=True,
        ):
            dot = math.fsum(
                float(a) * (float(c) - float(b))
                for a, b, c in zip(g.flat, before.flat, after.flat, strict=True)
            )
            inherited = math.fsum(
                float(a) * (-capture["learning_rate"] * capture["momentum"] * float(v))
                for a, v in zip(g.flat, velocity.flat, strict=True)
            )
            nominal = -capture["learning_rate"] * math.fsum(float(a) ** 2 for a in g.flat)
            assert group["name"] == name
            summary = group["summary"]
            assert summary["gradient_dot_step"] == pytest.approx(dot, rel=1e-12, abs=1e-14)
            assert summary["gradient_dot_momentum_step"] == pytest.approx(
                inherited, rel=1e-12, abs=1e-14
            )
            assert summary["gradient_dot_sgd_step"] == pytest.approx(nominal, rel=1e-12, abs=1e-14)
            dots.append(dot)
            momenta.append(inherited)
            sgd_dots.append(nominal)
        for name, values in (
            ("gradient_dot_step", dots),
            ("gradient_dot_momentum_step", momenta),
            ("gradient_dot_sgd_step", sgd_dots),
        ):
            assert capture["result"]["overall"][name] == pytest.approx(
                math.fsum(values), rel=1e-12, abs=1e-14
            )


def test_should_charge_overlapping_groups_as_work_not_additional_training_or_unique_data(completed):
    diagnostic = completed["result"].report["step_diagnostics"]
    work = diagnostic["work"]
    assert diagnostic["reference_rows"] == work["reference_rows"] == 4
    assert work["groups_recorded"] == 12 and work["nonempty_groups_attempted"] == 6
    for key in asdict(direction._StepDirectionWork()):
        assert work[key] == 6
    assert work["normalization_rows_returned"] == work["task_gradient_rows_returned"] == 8
    assert work["normalization_values_returned"] == 96
    assert work["gradient_values_returned"] == 228
    assert work["gradient_numeric_bytes_returned"] == 1824
    assert work["derived_task_dense_multiply_accumulates"] == 576
    assert work["input_numeric_bytes"] == 1024
    assert all(
        work[key] == 0
        for key in (
            "model_creation_calls",
            "fit_calls",
            "optimizer_updates",
            "extra_prediction_calls",
        )
    )
    assert completed["result"].report["work"]["training_rows"] == 16


def test_should_leave_default_full_state_receipts_costs_and_schema_exact(completed, default_run):
    selected = completed["result"]
    assert "step_diagnostics" not in default_run.report
    assert_state(selected.candidate.snapshot_state(), default_run.candidate.snapshot_state())
    first, second = deepcopy(selected.report), deepcopy(default_run.report)
    first.pop("step_diagnostics")
    first.pop("elapsed_seconds")
    second.pop("elapsed_seconds")
    assert first == second


def test_should_retain_one_paid_update_and_attempted_diagnostic_cost_when_readout_fails(
    source, monkeypatch
):
    owner, initial = source.fork(), source.snapshot_state()
    captured: dict[str, Any] = {}
    runtime_factory = demo.ActorShadowRuntime
    calls = []

    def runtime(*args, **options):
        value = runtime_factory(*args, **options)
        captured.update(runtime=value, budget=options["budget"])
        return value

    def fail(*args, **options):
        calls.append(1)
        options["_work"].normalization_attempts += 1
        raise RuntimeError("diagnostic normalization failed")

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(leaf, "diagnose_step_direction", fail)
    result = demo.run_continuous_correction_update(
        owner,
        batch(),
        now=3,
        action_history_recipe=RECIPE,
        binary_gripper_axis=3,
        diagnostic_stage_bindings=references(),
    )
    report, diagnostic = result.report, result.report["step_diagnostics"]
    assert report["complete"] is False and result.candidate is None
    assert "diagnostic normalization failed" in report["error"]
    assert calls == [1] and len(report["applied_updates"]) == 1
    assert report["events"][0]["success_validated"] is True
    assert report["progress"]["updates_completed"] == captured["budget"].updates_completed == 1
    assert diagnostic["complete"] is False
    work = diagnostic["work"]
    assert work["normalization_attempts"] == 1 and work["normalization_returns"] == 0
    assert (
        work["normalization_rows_attempted"] == 2 and work["normalization_values_attempted"] == 24
    )
    assert work["task_gradient_attempts"] == work["task_gradient_returns"] == 0
    assert work["gradient_values_returned"] == work["optimizer_updates"] == 0
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, captured["runtime"].actor.snapshot_state().state)
    paid = captured["runtime"].candidate_snapshot().state
    assert (paid.completed_updates, paid.training_examples) == (2, 4)


@pytest.mark.parametrize(
    "fault", ["list", "reversed", "role", "spec", "history", "unsupported_version"]
)
def test_should_refuse_bad_reference_binding_before_forks_or_private_updates(
    source, monkeypatch, fault
):
    bindings: Any = references()
    if fault == "list":
        bindings = list(bindings)
    elif fault == "reversed":
        bindings = bindings[::-1]
    elif fault == "role":
        object.__setattr__(bindings[1].episode, "role", "probe_test")
    elif fault == "unsupported_version":
        object.__setattr__(
            bindings[0].episode, "episode_id", "pick-cube-goal-consistent-observed-grasp-teacher-17"
        )
    elif fault == "history":
        values = list(bindings[0].episode.transitions[0].observation)
        values[7] = 0.5
        object.__setattr__(bindings[0].episode.transitions[0], "observation", tuple(values))
    else:
        altered = []
        for binding in bindings:
            episode = binding.episode
            episode = replace(
                episode,
                spec=replace(episode.spec, observation_dim=13),
                transitions=tuple(
                    replace(
                        row,
                        observation=row.observation + (0.0,),
                        next_observation=row.next_observation + (0.0,),
                    )
                    for row in episode.transitions
                ),
            )
            altered.append(ReadinessStageBinding(episode, binding.stages))
        bindings = tuple(altered)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(demo, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            source,
            batch(),
            now=3,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
            diagnostic_stage_bindings=bindings,
        )


@pytest.mark.parametrize("fault", ["nearest", "public_projection"])
def test_should_refuse_unsupported_diagnostic_collection_modes_before_learner_observation(
    source, monkeypatch, fault
):
    owner: Any = source
    options: dict[str, Any] = {}
    if fault == "nearest":
        owner = object.__new__(nearest.NumpyNearestContinuousBCLearner)
    else:
        options["observation_projection"] = public_vector_projection()
    monkeypatch.setattr(demo, "ActorShadowRuntime", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner, batch(), now=3, diagnostic_stage_bindings=references(), **options
        )


@pytest.mark.parametrize("fault", ["direction", "work", "empty_stage", "paid_receipt", "validated"])
def test_should_refuse_changed_retained_diagnostic_or_first_paid_event_without_more_model_work(
    completed, fault
):
    report = deepcopy(completed["result"].report)
    diagnostic_seal = json.dumps(report["step_diagnostics"], sort_keys=True, allow_nan=False)
    paid_seal = json.dumps(report["events"][0], sort_keys=True, allow_nan=False)
    demo._require_readiness_step_evidence(report, diagnostic_seal, paid_seal)
    if fault == "direction":
        report["step_diagnostics"]["contexts"]["warm"]["groups"]["overall"]["result"]["overall"][
            "gradient_dot_step"
        ] += 0.125
    elif fault == "work":
        report["step_diagnostics"]["work"]["task_gradient_returns"] = True
    elif fault == "empty_stage":
        report["step_diagnostics"]["contexts"]["new"]["groups"].pop("align")
    elif fault == "paid_receipt":
        report["events"][0]["applied"][0]["diagnostic"]["value"] += 0.125
    else:
        report["events"][0]["success_validated"] = False
    with pytest.raises(RuntimeError, match="preceding readiness"):
        demo._require_readiness_step_evidence(report, diagnostic_seal, paid_seal)
