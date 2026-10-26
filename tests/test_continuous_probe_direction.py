"""Opt-in TRAIN fit and realized-step readouts, independent of native objectives.

Fixed synthetic arrays and handwritten gradient/reduction oracles. No favorable
gain assertion, optimizer replacement, scientific data or native simulator.
"""

from dataclasses import asdict, replace
import math
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_learnability_demo as demo
from src.adapters import continuous_probe as probe
from src.core import continuous_step_diagnostics as core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")
CPC_WORK = {
    "chemistry_attempts",
    "chemistry_returns",
    "gate_applications_attempted",
    "gate_applications_returned",
}


def episode(name="A"):
    if name == "A":
        positions, actions, context, role = (
            [0.0, 2.0, 4.0],
            [(-2.0, -1.0), (6.0, 1.0)],
            "context-A",
            "train",
        )
    elif name == "B":
        positions, actions, context, role = (
            [10.0, 11.0, 12.0, 13.0],
            [(5.0, 1.0), (4.0, 0.5), (3.0, 0.0)],
            "context-B",
            "train",
        )
    elif name == "Ftrain":
        positions, actions, context, role = (
            [-1.0, 1.0, 2.0],
            [(2.0, 0.5), (1.0, -0.5)],
            "fresh-F",
            "probe_train",
        )
    else:
        positions, actions, context, role = (
            [-0.5, 0.5, 1.5],
            [(1.5, 0.25), (1.25, -0.25)],
            "fresh-F",
            "probe_test",
        )
    observations = [(value, 1.0 if name == "B" else 0.0) for value in positions]
    declared = ContinuousActionSpec(2, ("m", "m"), (-2.0, -1.0), (6.0, 1.0))
    return ContinuousEpisode(
        "synthetic-" + name,
        context,
        role,
        declared,
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                action,
                0.0,
                observations[index + 1],
                index == len(actions) - 1,
                False,
            )
            for index, action in enumerate(actions)
        ),
    )


def plan(**changes):
    return probe.ContinuousProbePlan(
        episode("Ftrain"), episode("Ftest"), diagnostics=True, **changes
    )


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_boolean_diagnostic_flags_before_source_fit(monkeypatch, flag):
    with pytest.raises(ValueError):
        probe.ContinuousProbePlan(episode("Ftrain"), episode("Ftest"), diagnostics=flag)
    value = plan()
    object.__setattr__(value, "diagnostics", flag)
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("corrupt diagnostic flag fitted data"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(episode(), episode("B"), phase_updates=1, probe_plan=value)


def test_should_copy_explicit_diagnostic_flag_and_keep_default_plan_false():
    default = probe.ContinuousProbePlan(episode("Ftrain"), episode("Ftest"))
    explicit = probe.ContinuousProbePlan(episode("Ftrain"), episode("Ftest"), diagnostics=False)
    assert asdict(default) == asdict(explicit) and default.diagnostics is False
    value = plan()
    owned = probe.copy_probe_plan(value, (episode(), episode("B")))
    assert owned is not value and owned.diagnostics is True
    object.__setattr__(value, "diagnostics", False)
    assert owned.diagnostics is True


@pytest.fixture(scope="module")
def four_arms():
    warm, new, value = episode(), episode("B"), plan()
    original = asdict(warm), asdict(new), asdict(value)
    results: dict[str, Any] = {}
    captured: dict[str, Any] = {}
    states: dict[str, Any] = {}
    original_runtime, original_direction = probe.ActorShadowRuntime, probe.diagnose_step_direction
    with pytest.MonkeyPatch.context() as patch:

        def runtime(*args, **options):
            owner = original_runtime(*args, **options)
            ready = owner.train_ready

            def record(*args, **kwargs):
                before = owner.candidate_snapshot().state
                receipts = ready(*args, **kwargs)
                if receipts:
                    states[family].append((before, owner.candidate_snapshot().state))
                return receipts

            patch.setattr(owner, "train_ready", record)
            return owner

        def direction(features, targets, before, velocity, after, declared, **options):
            previous, current = states[family][-1]
            for actual, expected in zip(
                before + velocity + after,
                previous.parameters + previous.velocity + current.parameters,
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(features, [[-1.0, 0.0], [1.0, 0.0]])
            np.testing.assert_array_equal(targets, [[2.0, 0.5], [1.0, -0.5]])
            assert options["standardization"] == ObservationStandardization(
                (1.0, 0.0), (1.0, 1.0), 2
            )
            assert options["learning_rate"] == 0.01 and options["momentum"] == 0.9
            result = original_direction(
                features, targets, before, velocity, after, declared, **options
            )
            captured[family].append(result)
            return result

        patch.setattr(probe, "ActorShadowRuntime", runtime)
        patch.setattr(probe, "diagnose_step_direction", direction)
        for family in FAMILIES:
            captured[family], states[family] = [], []
            results[family] = demo.run_continuous_learnability(
                warm, new, phase_updates=1, learner_family=family, probe_plan=value
            )
            assert (asdict(warm), asdict(new), asdict(value)) == original
    results["_directions"] = captured
    return results


@pytest.mark.parametrize("family", FAMILIES)
def test_should_score_fresh_training_and_heldout_separately_without_changing_roles_or_budgets(
    four_arms, family
):
    parent = four_arms[family].report
    assert parent["complete"] is True and four_arms[family].candidate is not None
    assert parent["progress"]["updates_completed"] == 3
    assert [entry["benchmark_updates"] for entry in parent["probes"]] == [0, 1, 2, 3]
    for entry in parent["probes"]:
        report = entry["report"]
        assert report["complete"] is True
        assert report["progress"]["updates_completed"] == 1
        assert len(report["applied_updates"]) == len(report["native_updates"]) == 1
        assert report["applied_updates"][0]["episode_id"] == "synthetic-Ftrain"
        first, final = [point["scores"] for point in report["checkpoints"]]
        assert first["fresh"]["role"] == final["fresh"]["role"] == "probe_test"
        assert first["fresh_training"]["role"] == final["fresh_training"]["role"] == "probe_train"
        assert first["fresh_training"]["episode_id"] == "synthetic-Ftrain"
        loss = lambda point, name: point[name]["residual"]["overall"]["normalized_half_mse"]
        assert report["metrics"]["fresh_training_gain"] == loss(first, "fresh_training") - loss(
            final, "fresh_training"
        )
        assert report["metrics"]["fresh_acquisition_gain"] == loss(first, "fresh") - loss(
            final, "fresh"
        )
        assert report["metrics"]["old_retention_loss"] == {
            key: loss(final, key) - loss(first, key) for key in ("old_0", "old_1")
        }
        assert report["work"]["prediction_attempts"] == report["work"]["prediction_returns"] == 8
        assert (
            report["work"]["prediction_rows_attempted"]
            == report["work"]["prediction_rows_returned"]
            == 18
        )
        assert report["work"]["score_rows_returned_by_role"] == {
            "probe_test": 4,
            "probe_train": 4,
            "train": 10,
        }


def test_should_keep_pc_neutral_common_physical_scores_native_work_and_direction_exact(four_arms):
    first, second = four_arms["pc"].report, four_arms["neutral-cpc"].report
    assert first["metrics"] == second["metrics"] and first["checkpoints"] == second["checkpoints"]
    for a, b in zip(first["probes"], second["probes"], strict=True):
        left, right = a["report"], b["report"]
        assert left["checkpoints"] == right["checkpoints"]
        assert left["metrics"] == right["metrics"]
        assert left["applied_updates"] == right["applied_updates"]
        native, neutral = left["native_updates"][0], right["native_updates"][0]
        assert native["diagnostic"] == neutral["diagnostic"]
        assert native["inference"] == neutral["inference"]
        assert native["work"] == {
            key: value for key, value in neutral["work"].items() if key not in CPC_WORK
        }
        assert left["update_diagnostics"] == right["update_diagnostics"]


def small_problem():
    features, targets = np.array([[0.4], [0.8]]), np.array([[0.2], [-0.3]])
    before = (np.array([[0.2]]), np.array([-0.1]), np.array([[0.3]]), np.array([0.05]))
    velocity = (np.array([[0.1]]), np.array([-0.2]), np.array([[0.3]]), np.array([-0.4]))
    delta = (np.array([[0.02]]), np.array([-0.03]), np.array([[0.04]]), np.array([0.01]))
    after = tuple(value + step for value, step in zip(before, delta, strict=True))
    declared = ContinuousActionSpec(1, ("normalized",), (-1.0,), (1.0,))
    transform = ObservationStandardization((0.1,), (0.5,), 2)
    return features, targets, before, velocity, after, declared, transform


def hand_gradient(features, targets, before, transform):
    """Two-layer chain rule directly; no implementation derivative helper."""
    values = (features - np.array(transform.means)) / np.array(transform.scales)
    hidden = np.tanh(values @ before[0] + before[1])
    prediction = np.tanh(hidden @ before[2] + before[3])
    error = prediction - targets
    head = error * (1.0 - prediction**2) / len(values)
    hidden_delta = (head @ before[2].T) * (1.0 - hidden**2)
    return float(np.sum(error**2) / (2 * len(values))), (
        values.T @ hidden_delta,
        hidden_delta.sum(axis=0),
        hidden.T @ head,
        head.sum(axis=0),
    )


def dot(first, second):
    return sum(float(a) * float(b) for a, b in zip(first, second, strict=True))


def norm(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def flatten(values):
    return [float(value) for array in values for value in array.flat]


def diagnose(problem=None, **options):
    features, targets, before, velocity, after, declared, transform = (
        small_problem() if problem is None else problem
    )
    keywords: dict[str, Any] = dict(standardization=transform, learning_rate=0.05, momentum=0.8)
    keywords.update(options)
    return core.diagnose_step_direction(
        features, targets, before, velocity, after, declared, **keywords
    )


def assert_summary(summary, gradient, realized, inherited, rate):
    gradient_norm, step_norm, momentum_norm = norm(gradient), norm(realized), norm(inherited)
    assert summary.gradient_l2 == pytest.approx(gradient_norm, rel=1e-14, abs=1e-15)
    assert summary.step_l2 == pytest.approx(step_norm, rel=1e-14, abs=1e-15)
    assert summary.momentum_step_l2 == pytest.approx(momentum_norm, rel=1e-14, abs=1e-15)
    assert summary.gradient_dot_step == pytest.approx(dot(gradient, realized), rel=1e-14, abs=1e-15)
    assert summary.gradient_dot_momentum_step == pytest.approx(
        dot(gradient, inherited), rel=1e-14, abs=1e-15
    )
    assert summary.gradient_dot_sgd_step == pytest.approx(
        -rate * dot(gradient, gradient), rel=1e-14, abs=1e-15
    )
    for actual, other, other_norm in (
        (summary.gradient_step_cosine, realized, step_norm),
        (summary.gradient_momentum_cosine, inherited, momentum_norm),
    ):
        if gradient_norm == 0 or other_norm == 0:
            assert actual is None
        else:
            assert actual == pytest.approx(
                dot(gradient, other) / gradient_norm / other_norm, abs=1e-14
            )


def test_should_match_handwritten_prestate_gradient_realized_step_and_nominal_momentum():
    problem = small_problem()
    features, targets, before, velocity, after, _, transform = problem
    originals = tuple(value.copy() for value in (features, targets, *before, *velocity, *after))
    loss, gradients = hand_gradient(features, targets, before, transform)
    work = core._StepDirectionWork()
    result = diagnose(problem, _work=work)
    assert result.definition == "continuous_task_bp_realized_step_direction_v1"
    assert (
        result.objective_definition
        == "continuous_bc_half_mean_active_axis_normalized_squared_error_v1"
    )
    assert result.task_loss_before == pytest.approx(loss, rel=0, abs=1e-15)
    realized = tuple(new - old for old, new in zip(before, after, strict=True))
    inherited = tuple(-0.05 * 0.8 * old for old in velocity)
    assert [group.name for group in result.groups] == ["w1", "b1", "w2", "b2"]
    for group, gradient, step, momentum_step in zip(
        result.groups, gradients, realized, inherited, strict=True
    ):
        assert_summary(
            group.summary, flatten((gradient,)), flatten((step,)), flatten((momentum_step,)), 0.05
        )
    assert_summary(result.overall, flatten(gradients), flatten(realized), flatten(inherited), 0.05)
    assert result.work == dict(
        normalization_attempts=1,
        normalization_returns=1,
        task_gradient_attempts=1,
        task_gradient_returns=1,
        summary_attempts=1,
        summary_returns=1,
        gradient_values_returned=4,
        gradient_numeric_bytes_returned=32,
        gradient_rows_returned=2,
        derived_task_dense_multiply_accumulates=10,
    )
    saved = asdict(result)
    for value, original in zip(
        (features, targets, *before, *velocity, *after), originals, strict=True
    ):
        np.testing.assert_array_equal(value, original)
        value[:] = 99
    assert asdict(result) == saved
    result.work["task_gradient_returns"] = 999
    assert work.task_gradient_returns == 1


def test_should_report_none_cosines_for_zero_gradients_and_steps():
    features = np.zeros((1, 1))
    parameters = (np.zeros((1, 1)), np.zeros(1), np.zeros((1, 1)), np.zeros(1))
    declared = ContinuousActionSpec(1, ("normalized",), (-1.0,), (1.0,))
    result = diagnose(
        (
            features,
            features.copy(),
            parameters,
            tuple(value.copy() for value in parameters),
            tuple(value.copy() for value in parameters),
            declared,
            None,
        )
    )
    assert result.task_loss_before == 0
    for summary in (result.overall, *(group.summary for group in result.groups)):
        assert summary.gradient_l2 == summary.step_l2 == summary.momentum_step_l2 == 0
        assert (
            summary.gradient_dot_step
            == summary.gradient_dot_momentum_step
            == summary.gradient_dot_sgd_step
            == 0
        )
        assert summary.gradient_step_cosine is summary.gradient_momentum_cosine is None


def extreme_problem():
    before = (np.zeros((1, 1)), np.zeros(1), np.ones((1, 1)), np.zeros(1))
    velocity = tuple(np.zeros_like(value) for value in before)
    after = tuple(value.copy() for value in before)
    after[0][0, 0] = 1e-200
    return (
        np.array([[1e200]]),
        np.array([[-1.0]]),
        before,
        velocity,
        after,
        ContinuousActionSpec(1, ("normalized",), (-1.0,), (1.0,)),
        None,
    )


def test_should_keep_representable_large_norms_tiny_steps_and_cosines_scale_safe():
    result = diagnose(extreme_problem(), learning_rate=1e-200, momentum=0.0)
    assert result.task_loss_before == 0.5
    assert result.overall.gradient_l2 == pytest.approx(1e200)
    assert result.overall.step_l2 == pytest.approx(1e-200, rel=1e-14, abs=0)
    assert result.overall.gradient_dot_step == pytest.approx(1.0)
    assert result.overall.gradient_dot_sgd_step == pytest.approx(-1e200)
    assert result.overall.gradient_step_cosine == pytest.approx(1.0)
    assert result.overall.gradient_momentum_cosine is None


def test_should_refuse_unrepresentable_reduction_and_preserve_actual_work():
    work = core._StepDirectionWork()
    with pytest.raises(FloatingPointError):
        diagnose(extreme_problem(), learning_rate=0.1, momentum=0.0, _work=work)
    assert asdict(work) == dict(
        normalization_attempts=1,
        normalization_returns=1,
        task_gradient_attempts=1,
        task_gradient_returns=1,
        summary_attempts=1,
        summary_returns=0,
    )


@pytest.mark.parametrize(
    "damage",
    [
        "features_dtype",
        "targets_nonfinite",
        "before_shape",
        "velocity_nonfinite",
        "after_dtype",
        "after_shape",
        "rate_bool",
        "rate_zero",
        "momentum_negative",
        "momentum_one",
    ],
)
def test_should_refuse_malformed_diagnostic_inputs_without_mutating_any_borrowed_array(damage):
    features, targets, before, velocity, after, declared, transform = small_problem()
    options: dict[str, Any] = {}
    if damage == "features_dtype":
        features = features.astype(np.float32)
    elif damage == "targets_nonfinite":
        targets[0, 0] = np.nan
    elif damage == "before_shape":
        before = (np.ones((2, 1)), *before[1:])
    elif damage == "velocity_nonfinite":
        velocity[0][0, 0] = np.inf
    elif damage == "after_dtype":
        after = (after[0].astype(np.float32), *after[1:])
    elif damage == "after_shape":
        after = (*after[:3], np.zeros(2))
    elif damage.startswith("rate"):
        options["learning_rate"] = True if damage == "rate_bool" else 0
    else:
        options["momentum"] = -0.1 if damage == "momentum_negative" else 1.0
    originals = tuple(value.copy() for value in (features, targets, *before, *velocity, *after))
    with pytest.raises(ValueError):
        diagnose((features, targets, before, velocity, after, declared, transform), **options)
    for value, original in zip(
        (features, targets, *before, *velocity, *after), originals, strict=True
    ):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("family", FAMILIES)
def test_should_record_preupdate_task_loss_and_actual_diagnostic_work_after_native_success(
    four_arms, family
):
    parent = four_arms[family].report
    assert len(four_arms["_directions"][family]) == 4
    for entry in parent["probes"]:
        report = entry["report"]
        (row,) = report["update_diagnostics"]
        assert row["update_number"] == 1 and row["complete"] is True
        assert row["work"] == dict(
            normalization_attempts=1,
            normalization_returns=1,
            task_gradient_attempts=1,
            task_gradient_returns=1,
            summary_attempts=1,
            summary_returns=1,
        )
        diagnostic = row["result"]
        assert diagnostic["task_loss_before"] == pytest.approx(
            report["checkpoints"][0]["scores"]["fresh_training"]["residual"]["overall"][
                "normalized_half_mse"
            ],
            abs=1e-14,
        )
        assert [group["name"] for group in diagnostic["groups"]] == ["w1", "b1", "w2", "b2"]
        assert report["work"]["step_diagnostics"] == diagnostic["work"]
        assert diagnostic["work"]["gradient_values_returned"] == 322
        assert diagnostic["work"]["gradient_numeric_bytes_returned"] == 2576
        assert diagnostic["work"]["gradient_rows_returned"] == 2
        assert diagnostic["work"]["derived_task_dense_multiply_accumulates"] == 1280
        if family in ("neutral-cpc", "active-cpc"):
            assert four_arms[family].candidate.snapshot_state().wake_state.wake_updates == 3


def test_should_preserve_default_false_reports_and_skip_diagnostic_gradient_calls(
    four_arms, monkeypatch
):
    source = four_arms["bp"].candidate
    before = source.snapshot_state()
    monkeypatch.setattr(
        probe,
        "diagnose_step_direction",
        lambda *_a, **_k: pytest.fail("False diagnostics computed a direction"),
    )
    default = probe.run_continuous_probe(
        source,
        probe.ContinuousProbePlan(episode("Ftrain"), episode("Ftest")),
        training_readouts=(episode(), episode("B")),
    )
    explicit = probe.run_continuous_probe(
        source,
        probe.ContinuousProbePlan(episode("Ftrain"), episode("Ftest"), diagnostics=False),
        training_readouts=(episode(), episode("B")),
    )
    assert default["complete"] is explicit["complete"] is True
    assert {key: value for key, value in default.items() if key != "elapsed_seconds"} == {
        key: value for key, value in explicit.items() if key != "elapsed_seconds"
    }
    assert "update_diagnostics" not in default and "step_diagnostics" not in default["work"]
    assert "fresh_training_gain" not in default["metrics"]
    assert "fresh_training" not in default["checkpoints"][0]["scores"]
    after = source.snapshot_state()
    assert before.policy == after.policy and before.completed_updates == after.completed_updates
    assert before.training_examples == after.training_examples
    for first, second in zip(
        before.parameters + before.velocity, after.parameters + after.velocity, strict=True
    ):
        np.testing.assert_array_equal(first, second)


def test_should_keep_paid_native_and_receipt_when_task_gradient_diagnostic_fails(monkeypatch):
    original = probe.diagnose_step_direction

    def fail_diagnostic(*args, **kwargs):
        with pytest.MonkeyPatch.context() as patch:

            def fail_gradient(*_args, **_kwargs):
                raise FloatingPointError("diagnostic task gradient failed")

            patch.setattr(core.task, "continuous_bc_loss_and_gradients", fail_gradient)
            return original(*args, **kwargs)

    monkeypatch.setattr(probe, "diagnose_step_direction", fail_diagnostic)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="pc", probe_plan=plan()
    )
    parent = result.report
    assert result.candidate is None and parent["complete"] is False
    assert parent["progress"]["updates_completed"] == 0 and parent["applied_updates"] == []
    report = parent["probes"][0]["report"]
    assert report["complete"] is False and "diagnostic task gradient" in report["error"]
    assert report["progress"]["updates_completed"] == 1
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert report["native_updates"][0]["work"]["published_updates"] == 1
    (row,) = report["update_diagnostics"]
    assert row["complete"] is False and "result" not in row
    assert row["work"] == dict(
        normalization_attempts=1,
        normalization_returns=1,
        task_gradient_attempts=1,
        task_gradient_returns=0,
        summary_attempts=0,
        summary_returns=0,
    )
    assert report["work"]["step_diagnostics"]["task_gradient_attempts"] == 1
    assert report["work"]["step_diagnostics"]["gradient_rows_returned"] == 0
    assert "metrics" not in report and "metrics" not in parent


def test_should_retain_returned_direction_before_refusing_coherent_candidate_wake_replacement(
    monkeypatch,
):
    original_runtime, original_direction = probe.ActorShadowRuntime, probe.diagnose_step_direction
    scopes = []

    def runtime(*args, **options):
        owner = original_runtime(*args, **options)
        scopes.append(owner)
        return owner

    def mutate(*args, **kwargs):
        report = original_direction(*args, **kwargs)
        owner = scopes[-1]._candidate
        state = owner.snapshot_state()
        wake, config = state.wake_state, state.policy.circadian
        chemistry = wake.chemical_state + 0.01
        gates = np.clip(
            np.exp(-config.plasticity_sensitivity * chemistry), config.min_plasticity, 1.0
        )
        owner.restore_state(
            replace(
                state, wake_state=replace(wake, chemical_state=chemistry, plasticity_state=gates)
            )
        )
        return report

    monkeypatch.setattr(probe, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(probe, "diagnose_step_direction", mutate)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="active-cpc", probe_plan=plan()
    )
    parent = result.report
    assert result.candidate is None and parent["complete"] is False
    assert parent["progress"]["updates_completed"] == 0
    report = parent["probes"][0]["report"]
    assert report["complete"] is False and "actor/candidate changed" in report["error"]
    assert report["progress"]["updates_completed"] == 1
    assert len(report["applied_updates"]) == len(report["native_updates"]) == 1
    (row,) = report["update_diagnostics"]
    assert row["complete"] is True and row["result"]["work"]["task_gradient_returns"] == 1
    assert "metrics" not in report and "metrics" not in parent
