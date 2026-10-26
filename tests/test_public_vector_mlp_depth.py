"""Synthetic optimization-depth receipts/comparisons; no fit, SGD or forward.

One initialized/restored model supplies archive payloads only. Manual A2/B3
reports exercise admission and accounting without reconstructing any experiment.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.continuous_bc import OBJECTIVE
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.infra.maniskill_continuous import ACTION_UNITS
from test_continuous_learnability_demo import same_state


case = importlib.import_module("artifacts.runs.public-vector-mlp-depth-20261019.run_case")
FIELDS = (
    "normalized_bias",
    "normalized_mae",
    "normalized_rmse",
    "normalized_half_mse",
    "maximum_absolute_normalized_error",
)


def forbidden(*_args, **_kwargs):
    pytest.fail("failed prerequisite admitted new work")


def training_episode(seed, rows):
    spec = ContinuousActionSpec(21, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    context = "pick-cube" if seed == 17 else "pick-cube-train-19"
    offset = 0.0 if seed == 17 else 10.0
    observations = tuple((offset + index,) + (0.0,) * 20 for index in range(rows + 1))
    return ContinuousEpisode(
        f"pick-cube-task-teacher-{seed}",
        context,
        "train",
        spec,
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                (-0.2, 0.0, 0.3, 1.0),
                float(index == rows - 1),
                observations[index + 1],
                index == rows - 1,
                False,
            )
            for index in range(rows)
        ),
    )


def residual(rows, value):
    axes = []
    for index, unit in enumerate(ACTION_UNITS):
        axes.append(
            dict(
                index=index,
                unit=unit,
                active=True,
                normalized_bias=value / 10,
                normalized_mae=value,
                normalized_rmse=math.sqrt(2 * value),
                normalized_half_mse=value,
                maximum_absolute_normalized_error=2 * value,
            )
        )
    overall = dict(rows=rows, active_axes=4, normalized_half_mse=value, axes=axes)
    return dict(
        residuals=dict(
            overall=overall,
            stages=[dict(label="all", row_indices=list(range(rows)), summary=deepcopy(overall))],
        ),
        prediction_calls=1,
        prediction_rows=rows,
        prediction_values=4 * rows,
        prediction_bytes=32 * rows,
    )


def work(count):
    """Tiny fixed21x64x4 arithmetic; A2/B3 supplies seven rows per phase unit."""
    updates, rows, backward = 3 * count, 7 * count, 3 * count + 8
    return dict(
        fit_calls=1,
        fit_rows=2,
        fit_feature_values=42,
        optimizer_updates=updates,
        training_rows=rows,
        unique_source_rows=5,
        repeated_training_rows=rows - 5,
        replay_budget_rows=0,
        conditioning_calls=8,
        conditioning_rows=20,
        residual_calls=8,
        residual_rows=20,
        actor_prediction_calls=8,
        actor_prediction_rows=20,
        normalization_calls=updates + 24,
        normalization_rows=rows + 60,
        normalization_values=(rows + 60) * 21,
        forward_passes=updates + 32,
        forward_rows=rows + 80,
        backward_passes=backward,
        backward_rows=rows + 20,
        gradient_values=backward * 1668,
        gradient_numeric_bytes=backward * 13344,
        dense_multiply_accumulates=rows * 3456 + 165120,
        parameter_values_per_owner=1668,
        model_numeric_bytes_per_owner=26688,
        statistics_numeric_bytes_per_owner=336,
        named_model_owners=4,
        inbox_feature_target_numeric_bytes=rows * 200,
        residual_prediction_bytes=640,
        actor_prediction_bytes=640,
        scope="successful fixed run; dense math/numeric payloads exclude copies/validation/elementwise/process overhead",
    )


def metrics(points):
    losses = [
        {
            name: point["contexts"][name]["residual"]["residuals"]["overall"]["normalized_half_mse"]
            for name in ("warm", "new")
        }
        for point in points
    ]
    return dict(
        warm_acquisition_gain=losses[0]["warm"] - losses[1]["warm"],
        new_acquisition_gain=losses[1]["new"] - losses[2]["new"],
        warm_retention_loss=losses[2]["warm"] - losses[1]["warm"],
        warm_return_recovery=losses[2]["warm"] - losses[3]["warm"],
        warm_return_deficit=losses[3]["warm"] - losses[1]["warm"],
        new_retention_loss=losses[3]["new"] - losses[2]["new"],
    )


def sequence(warm, new, policy, count):
    points, events, applied = [], [], []
    for phase, rows in enumerate((0, 2 * count, 5 * count, 7 * count)):
        contexts = {}
        for name, episode, offset in (("warm", warm, 0.0), ("new", new, 0.2)):
            value = 1.0 + offset + 0.1 * phase
            diagnostic = residual(len(episode.transitions), value)
            contexts[name] = dict(
                episode_id=episode.episode_id,
                residual=diagnostic,
                conditioning=dict(
                    conditioning=dict(objective_definition=OBJECTIVE, objective_loss=value),
                    standardization_calls=1,
                    standardization_rows=len(episode.transitions),
                    standardization_values=21 * len(episode.transitions),
                    standardization_applied=True,
                ),
                actor_predictions=[[0.0, 0.0, 0.0, 0.0] for _ in episode.transitions],
            )
        points.append(
            dict(
                updates=phase * count,
                training_examples=rows,
                state_exact=True,
                owners_exact=True,
                contexts=contexts,
            )
        )
    total_rows = 0
    for index in range(3 * count):
        episode = new if count <= index < 2 * count else warm
        phase = "warm" if index < count else "new" if index < 2 * count else "warm_return"
        total_rows += len(episode.transitions)
        receipt = dict(
            sample_id=f"mlp-batch-{index}",
            episode_id=episode.episode_id,
            event_id=f"mlp-label-{index}",
            actor_version="actor-0",
            learner_version="candidate-0",
            observed_at=2 * index,
            arrived_at=2 * index + 1,
            applied_at=2 * index + 1,
            update_number=index + 1,
            diagnostic=dict(definition=OBJECTIVE, value=0.5),
        )
        applied.append(receipt)
        events.append(
            dict(
                index=index,
                phase=phase,
                context="new" if episode is new else "warm",
                observed_at=2 * index,
                arrived_at=2 * index + 1,
                deferred_exact=True,
                update_attempted=True,
                applied=[deepcopy(receipt)],
                success_validated=True,
                candidate_updates=index + 1,
                candidate_training_examples=total_rows,
            )
        )
    return dict(
        complete=True,
        policy=asdict(policy),
        expected_standardization_checked=count == 32,
        settings=dict(
            phase_updates=count,
            phases=["warm", "new", "warm_return"],
            max_training_updates=3 * count,
            promotion_calls=0,
            simulator_calls=0,
            probe_reads=0,
            final_reads=0,
            guard_reads=0,
        ),
        source_episodes={
            name: dict(episode_id=episode.episode_id, context=episode.context, role="train")
            for name, episode in (("warm", warm), ("new", new))
        },
        events=events,
        applied_updates=applied,
        checkpoints=points,
        progress=dict(
            update_attempts=3 * count,
            updates_completed=3 * count,
            replay_examples_completed=0,
            next_tick=6 * count,
        ),
        work=work(count),
        metrics=metrics(points),
    )


@pytest.fixture(scope="module")
def data():
    warm, new = training_episode(17, 2), training_episode(19, 3)
    statistics = ObservationStandardization((0.5,) + (0.0,) * 20, (0.5,) + (1.0,) * 20, 2)
    policy = ContinuousBCPolicy(warm.spec, observation_standardization=statistics)
    baseline, current = sequence(warm, new, policy, 8), sequence(warm, new, policy, 32)
    return dict(
        warm=warm, new=new, statistics=statistics, policy=policy, baseline=baseline, current=current
    )


@pytest.fixture(scope="module")
def candidate(data):
    """ONE constructor+ONE restore; zero fit, forward and optimizer calls."""
    owner = NumpyContinuousBCLearner(data["policy"])
    initial = owner.snapshot_state()
    velocity = tuple(
        np.full_like(value, (index + 1) * 0.125) for index, value in enumerate(initial.velocity)
    )
    owner.restore_state(
        replace(initial, velocity=velocity, completed_updates=96, training_examples=224)
    )
    return owner


@pytest.mark.parametrize("count", [8, 32])
def test_should_validate_complete_manual_phase_receipts_without_executing_training(data, count):
    value = data["baseline"] if count == 8 else data["current"]
    original = deepcopy(value)
    case.check_progress(value, data["warm"], data["new"], count)
    assert value == original
    assert len(value["applied_updates"]) == 3 * count
    assert value["progress"]["next_tick"] == 6 * count
    assert value["checkpoints"][-1]["training_examples"] == 7 * count


@pytest.mark.parametrize(
    "damage",
    [
        "clock",
        "budget_type",
        "phase",
        "checkpoint_count",
        "checkpoint_type",
        "row_count",
        "source",
        "receipt_clock",
        "receipt_bool",
        "receipt_negative",
        "receipt_objective",
        "receipt_identity",
        "event_phase",
        "event_return",
        "event_rows",
        "event_extra",
        "flag",
    ],
)
def test_should_refuse_changed_delayed_receipts_progress_or_checkpoint_contract(data, damage):
    value = deepcopy(data["current"])
    if damage == "clock":
        value["progress"]["next_tick"] = 191
    elif damage == "budget_type":
        value["settings"]["max_training_updates"] = 96.0
    elif damage == "phase":
        value["settings"]["phases"] = ["warm", "warm", "new"]
    elif damage == "checkpoint_count":
        value["checkpoints"].pop()
    elif damage == "checkpoint_type":
        value["checkpoints"][0]["updates"] = False
    elif damage == "row_count":
        value["checkpoints"][2]["training_examples"] = 159
    elif damage == "source":
        value["source_episodes"]["new"]["role"] = "probe_train"
    elif damage == "receipt_clock":
        value["applied_updates"][0]["arrived_at"] = 0
    elif damage == "receipt_bool":
        value["applied_updates"][0]["diagnostic"]["value"] = True
    elif damage == "receipt_negative":
        value["applied_updates"][0]["diagnostic"]["value"] = -0.01
        value["events"][0]["applied"] = [deepcopy(value["applied_updates"][0])]
    elif damage == "receipt_objective":
        value["applied_updates"][0]["diagnostic"]["definition"] = "selected_loss"
    elif damage == "receipt_identity":
        value["applied_updates"][32]["episode_id"] = data["warm"].episode_id
    elif damage == "event_phase":
        value["events"][32]["phase"] = "warm"
    elif damage == "event_return":
        value["events"][0]["applied"] = []
    elif damage == "event_rows":
        value["events"][95]["candidate_training_examples"] = 223
    elif damage == "event_extra":
        value["events"][0]["unobserved"] = True
    else:
        value["checkpoints"][3]["owners_exact"] = 1
    with pytest.raises((AssertionError, ValueError)):
        case.check_progress(value, data["warm"], data["new"], 32)


def test_should_match_one_frozen_policy_initial_point_and_independent_tiny_work(data):
    current, baseline = deepcopy(data["current"]), deepcopy(data["baseline"])
    original = deepcopy(current), deepcopy(baseline)
    case.verify_pair(current, baseline, data["warm"], data["new"])
    assert (current, baseline) == original
    assert current["work"]["optimizer_updates"] == 96
    assert current["work"]["training_rows"] == 224
    assert current["work"]["gradient_values"] == 173472
    assert current["work"]["gradient_numeric_bytes"] == 1387776
    assert current["work"]["dense_multiply_accumulates"] == 939264
    assert baseline["work"]["dense_multiply_accumulates"] == 358656


@pytest.mark.parametrize(
    "damage",
    [
        "policy",
        "extra_policy",
        "stats",
        "expected_fit",
        "initial",
        "diagnostic_cost",
        "training_cost",
        "work_type",
        "scope",
        "axis_count",
        "axis_type",
        "axis_nonfinite",
        "negative_error",
        "overall_negative",
        "overall_rows",
        "metric_keys",
        "metric_value",
    ],
)
def test_should_refuse_changed_policy_initial_measurements_or_accounting_without_training(
    data, damage
):
    current, baseline = deepcopy(data["current"]), deepcopy(data["baseline"])
    if damage == "policy":
        current["policy"]["seed"] = 19
    elif damage == "extra_policy":
        current["policy"]["selection"] = None
    elif damage == "stats":
        current["policy"]["observation_standardization"]["means"] = (0.6,) + (0.0,) * 20
    elif damage == "expected_fit":
        current["expected_standardization_checked"] = False
    elif damage == "initial":
        current["checkpoints"][0]["contexts"]["warm"]["actor_predictions"][0][0] = 0.01
    elif damage == "diagnostic_cost":
        current["work"]["conditioning_rows"] = 19
    elif damage == "training_cost":
        current["work"]["training_rows"] = 223
    elif damage == "work_type":
        current["work"]["optimizer_updates"] = 96.0
    elif damage == "scope":
        current["work"]["scope"] = "unmeasured scope"
    elif damage == "axis_count":
        current["checkpoints"][3]["contexts"]["new"]["residual"]["residuals"]["overall"][
            "axes"
        ].pop()
    elif damage == "axis_type":
        current["checkpoints"][3]["contexts"]["new"]["residual"]["residuals"]["overall"]["axes"][0][
            "index"
        ] = False
    elif damage == "axis_nonfinite":
        current["checkpoints"][3]["contexts"]["new"]["residual"]["residuals"]["overall"]["axes"][3][
            "normalized_mae"
        ] = float("nan")
    elif damage == "negative_error":
        current["checkpoints"][3]["contexts"]["new"]["residual"]["residuals"]["overall"]["axes"][3][
            "normalized_mae"
        ] = -0.01
    elif damage == "overall_negative":
        current["checkpoints"][3]["contexts"]["new"]["residual"]["residuals"]["overall"][
            "normalized_half_mse"
        ] = -0.01
        current["metrics"] = metrics(current["checkpoints"])
    elif damage == "overall_rows":
        current["checkpoints"][3]["contexts"]["new"]["residual"]["residuals"]["overall"]["rows"] = (
            3.0
        )
    elif damage == "metric_keys":
        current["metrics"]["selected_axis"] = 0.0
    else:
        current["metrics"]["warm_retention_loss"] += 0.01
    with pytest.raises((AssertionError, ValueError)):
        case.verify_pair(current, baseline, data["warm"], data["new"])


def test_should_report_every_phase_and_axis_delta_with_mixed_favorable_and_negative_results(data):
    current, baseline = deepcopy(data["current"]), deepcopy(data["baseline"])
    shifts = [(0.0, 0.0), (-0.1, 0.05), (0.2, -0.1), (-0.3, 0.15)]
    for point, delta in zip(current["checkpoints"], shifts, strict=True):
        for name, difference in zip(("warm", "new"), delta, strict=True):
            result = point["contexts"][name]
            summary = result["residual"]["residuals"]["overall"]
            summary["normalized_half_mse"] += difference
            for axis in summary["axes"]:
                for field in FIELDS:
                    axis[field] += difference
            result["residual"]["residuals"]["stages"][0]["summary"] = deepcopy(summary)
    current["metrics"] = metrics(current["checkpoints"])
    case.verify_pair(current, baseline, data["warm"], data["new"])
    compared = case.comparison(current, baseline)
    assert compared["primary_endpoint"] == 96
    assert compared["primary"] == pytest.approx(dict(warm=-0.3, new=0.15))
    assert "negative better" in compared["primary_direction"]
    assert [row["phase"] for row in compared["endpoints"]] == [
        "initial",
        "warm",
        "new",
        "warm_return",
    ]
    assert [row["updates"] for row in compared["endpoints"]] == [0, 32, 64, 96]
    assert [row["baseline_updates"] for row in compared["endpoints"]] == [0, 8, 16, 24]
    for point, delta in zip(compared["endpoints"], shifts, strict=True):
        for name, difference in zip(("warm", "new"), delta, strict=True):
            result = point["contexts"][name]
            assert result["p32_minus_p8"] == pytest.approx(difference)
            assert len(result["axes"]) == 4
            for index, axis in enumerate(result["axes"]):
                assert axis["index"] == index and axis["unit"] == ACTION_UNITS[index]
                assert set(axis["p32_minus_p8"]) == set(FIELDS)
                assert axis["p32_minus_p8"] == {
                    field: pytest.approx(difference) for field in FIELDS
                }
    assert compared["signed_metric_differences"] == {
        name: current["metrics"][name] - baseline["metrics"][name] for name in baseline["metrics"]
    }
    assert compared["metric_directions"] == dict(
        warm_acquisition_gain="positive better",
        new_acquisition_gain="positive better",
        warm_retention_loss="negative better",
        warm_return_recovery="positive better",
        warm_return_deficit="negative better",
        new_retention_loss="negative better",
    )
    assert compared["work_differences"]["training_rows"] == 168
    assert compared["work_differences"]["optimizer_updates"] == 72
    assert compared["work_differences"]["dense_multiply_accumulates"] == 580608
    assert "unequal compute" in compared["exposure_comparison"]
    assert compared["metric_directions"] is not case.METRIC_DIRECTIONS
    compared["metric_directions"]["warm_acquisition_gain"] = "changed local metadata"
    assert case.METRIC_DIRECTIONS["warm_acquisition_gain"] == "positive better"


@pytest.fixture
def source_admission(data, tmp_path, monkeypatch):
    baseline = deepcopy(data["baseline"])
    rgb = deepcopy(baseline)
    rgb["policy"]["spec"]["observation_dim"] = 789
    statistics = rgb["policy"]["observation_standardization"]
    statistics["means"] = tuple(statistics["means"]) + (0.25,) * 768
    statistics["scales"] = tuple(statistics["scales"]) + (1.0,) * 768
    projection = dict(fixture="synthetic_projection_no_new_data")
    accepted = dict(
        complete=True,
        python=sys.version,
        numpy=np.__version__,
        input_projection=projection,
        sequence=baseline,
    )
    path = tmp_path / "prior-case.json"
    path.write_text(json.dumps(accepted), encoding="utf8")

    def load(report):
        report["input_projection"] = deepcopy(projection)
        report["source_paths"] = dict(baseline="synthetic-rgb-reference.json")
        return data["warm"], data["new"], rgb

    monkeypatch.setattr(case.old, "load_inputs", load)
    monkeypatch.setattr(case, "BASELINE_PATH", path)
    monkeypatch.setattr(case, "run_continuous_learnability", forbidden)
    return path, accepted


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "complete",
        "python",
        "numpy",
        "projection",
        "nonfinite",
        "policy",
        "stats",
        "work",
        "work_type",
        "receipt",
        "metric",
        "axis",
    ],
)
def test_should_admit_valid_baseline_and_refuse_changes_before_new_fit(
    data, source_admission, damage
):
    path, accepted = source_admission
    prior = accepted["sequence"]
    if damage == "complete":
        accepted["complete"] = 1
    elif damage in ("python", "numpy"):
        accepted[damage] = "changed-runtime"
    elif damage == "projection":
        accepted["input_projection"] = dict(other_projection=True)
    elif damage == "nonfinite":
        prior["metrics"]["warm_return_deficit"] = float("inf")
    elif damage == "policy":
        prior["policy"]["training_role"] = "probe_train"
    elif damage == "stats":
        prior["policy"]["observation_standardization"]["fitted_rows"] = 3
    elif damage == "work":
        prior["work"]["dense_multiply_accumulates"] += 1
    elif damage == "work_type":
        prior["work"]["optimizer_updates"] = 24.0
    elif damage == "receipt":
        prior["applied_updates"][0]["arrived_at"] = 0
    elif damage == "metric":
        prior["metrics"]["warm_retention_loss"] += 0.01
    elif damage == "axis":
        prior["checkpoints"][3]["contexts"]["new"]["residual"]["residuals"]["overall"]["axes"][0][
            "active"
        ] = 1
    path.write_text(json.dumps(accepted), encoding="utf8")
    report = {}
    if damage is not None:
        with pytest.raises((AssertionError, ValueError)):
            case.load_inputs(report)
    else:
        warm, new, baseline, stats = case.load_inputs(report)
        assert warm is data["warm"] and new is data["new"]
        assert case.exact_json(baseline, accepted["sequence"])
        assert stats == data["statistics"]
        assert report["source_paths"]["rgb_reference"] == "synthetic-rgb-reference.json"
        assert report["source_paths"]["baseline"] == str(path)


@pytest.mark.parametrize("damage", ["duplicate", "nonfinite", "large"])
def test_should_bound_and_refuse_ambiguous_or_nonfinite_prior_json(tmp_path, damage):
    path = tmp_path / "prior.json"
    payload = (
        '{"complete":true,"complete":false}'
        if damage == "duplicate"
        else '{"value":NaN}'
        if damage == "nonfinite"
        else " " * (512 * 1024 + 1)
    )
    path.write_text(payload, encoding="utf8")
    with pytest.raises((AssertionError, ValueError)):
        case.read_case(path)


def test_should_save_all_eight_parameter_and_nonzero_momentum_arrays_exclusively(
    data, candidate, tmp_path, monkeypatch
):
    monkeypatch.setattr(case, "stage", tmp_path)
    before = candidate.snapshot_state()
    report = dict(sequence=deepcopy(data["current"]))
    case.save_state(candidate, report)
    saved = report["state_archive"]
    names = [f"{group}_{index}" for group in ("parameters", "velocity") for index in range(4)]
    assert saved["array_names"] == names
    assert saved["updates"] == 96 and saved["training_examples"] == 224
    assert saved["numeric_bytes"] == 26688 and saved["file_bytes"] <= 128 * 1024
    assert saved["save_attempts"] == saved["save_returns"] == 1
    with np.load(tmp_path / "final-state.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(names)
        for group in ("parameters", "velocity"):
            for index, value in enumerate(getattr(before, group)):
                assert archive[f"{group}_{index}"].dtype == np.float64
                np.testing.assert_array_equal(archive[f"{group}_{index}"], value)
    same_state(before, candidate.snapshot_state())
    original = (tmp_path / "final-state.npz").read_bytes()
    with pytest.raises(FileExistsError):
        case.save_state(candidate, dict(sequence=deepcopy(data["current"])))
    assert (tmp_path / "final-state.npz").read_bytes() == original


@pytest.mark.parametrize(
    "damage",
    ["updates", "update_type", "rows", "row_type", "policy", "dtype", "momentum", "numeric_size"],
)
def test_should_refuse_invalid_complete_snapshot_before_opening_archive(
    data, candidate, tmp_path, monkeypatch, damage
):
    monkeypatch.setattr(case, "stage", tmp_path)
    state = candidate.snapshot_state()
    report = dict(sequence=deepcopy(data["current"]))
    if damage == "updates":
        state = replace(state, completed_updates=24)
    elif damage == "update_type":
        state = replace(state, completed_updates=96.0)
    elif damage == "rows":
        state = replace(state, training_examples=223)
    elif damage == "row_type":
        state = replace(state, training_examples=True)
    elif damage == "policy":
        report["sequence"]["policy"]["seed"] = 19
    elif damage == "dtype":
        state = replace(
            state, parameters=(state.parameters[0].astype(np.float32),) + state.parameters[1:]
        )
    elif damage == "momentum":
        state.velocity[2][0, 0] = np.nan
    else:
        small_policy = replace(state.policy, hidden_dim=2)
        state = replace(
            state,
            policy=small_policy,
            parameters=(np.zeros((21, 2)), np.zeros(2), np.zeros((2, 4)), np.zeros(4)),
            velocity=(np.zeros((21, 2)), np.zeros(2), np.zeros((2, 4)), np.zeros(4)),
        )
        report["sequence"]["policy"] = asdict(small_policy)
    owner = SimpleNamespace(policy=state.policy, snapshot_state=lambda: state)
    with pytest.raises((AssertionError, ValueError, TypeError)):
        case.save_state(owner, report)
    assert list(tmp_path.iterdir()) == []


def test_should_preserve_archive_attempt_when_write_fails_after_open(
    data, candidate, tmp_path, monkeypatch
):
    monkeypatch.setattr(case, "stage", tmp_path)

    def fail(*_args, **_kwargs):
        raise OSError("bounded archive write failed")

    monkeypatch.setattr(case.np, "savez", fail)
    report = dict(sequence=deepcopy(data["current"]))
    with pytest.raises(OSError, match="archive write failed"):
        case.save_state(candidate, report)
    assert report["state_archive"]["save_attempts"] == 1
    assert report["state_archive"]["save_returns"] == 0
    assert report["state_archive"]["updates"] == 96
    assert (tmp_path / "final-state.npz").exists()
    assert report["sequence"]["progress"]["updates_completed"] == 96


def test_should_retain_returned_archive_size_when_the_fixed_storage_bound_is_exceeded(
    data, candidate, tmp_path, monkeypatch
):
    monkeypatch.setattr(case, "stage", tmp_path)

    def oversized(handle, **_arrays):
        handle.write(b"x" * (128 * 1024 + 1))

    monkeypatch.setattr(case.np, "savez", oversized)
    report = dict(sequence=deepcopy(data["current"]))
    with pytest.raises(AssertionError):
        case.save_state(candidate, report)
    assert report["state_archive"]["save_attempts"] == report["state_archive"]["save_returns"] == 1
    assert report["state_archive"]["file_bytes"] == 128 * 1024 + 1
    assert report["sequence"]["progress"]["updates_completed"] == 96


def test_should_compose_one_fixed32_call_with_accepted_statistics_and_save_returned_state(
    data, candidate, tmp_path, monkeypatch
):
    monkeypatch.setattr(case, "stage", tmp_path)
    calls = []
    monkeypatch.setattr(
        case,
        "load_inputs",
        lambda _report: (data["warm"], data["new"], data["baseline"], data["statistics"]),
    )
    result = SimpleNamespace(report=deepcopy(data["current"]), candidate=candidate)

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(case, "run_continuous_learnability", run)
    report = dict(complete=False)
    case.run_case(report)
    assert report["complete"] is True and report["sequence_attempted"] is True
    assert calls == [
        (
            (data["warm"], data["new"]),
            dict(expected_standardization=data["statistics"], phase_updates=32),
        )
    ]
    assert report["sequence"] is result.report and report["baseline"] is data["baseline"]
    assert report["state_archive"]["updates"] == 96 and report["state_archive"]["save_returns"] == 1
    assert report["comparison"]["primary"] == dict(warm=0.0, new=0.0)
    assert report["comparison"]["work_differences"]["training_rows"] == 168


@pytest.mark.parametrize(
    "failure", ["incomplete", "missing_candidate", "initial_measurement", "archive"]
)
def test_should_retain_all_returned_paid_receipts_and_progress_before_later_failure(
    data, candidate, tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(
        case,
        "load_inputs",
        lambda _report: (data["warm"], data["new"], data["baseline"], data["statistics"]),
    )
    payload = deepcopy(data["current"])
    if failure == "incomplete":
        payload.update(complete=False, error="paid update return failed")
    if failure == "initial_measurement":
        payload["checkpoints"][0]["contexts"]["warm"]["actor_predictions"][0][0] = 0.125
    result = SimpleNamespace(
        report=payload,
        candidate=None if failure in ("incomplete", "missing_candidate") else candidate,
    )
    monkeypatch.setattr(case, "run_continuous_learnability", lambda *_a, **_k: result)
    if failure == "archive":

        def fail(*_args, **_kwargs):
            raise OSError("archive disk failure")

        monkeypatch.setattr(case, "save_state", fail)
    assert case.main() == 1
    report = json.loads((tmp_path / "case.json").read_text(encoding="utf8"))
    assert report["complete"] is False and "error" in report
    assert report["sequence"]["progress"]["updates_completed"] == 96
    assert report["sequence"]["progress"]["next_tick"] == 192
    assert len(report["sequence"]["applied_updates"]) == 96
    assert report["sequence"]["work"]["training_rows"] == 224
    assert (
        report["baseline_new_optimizer_updates"]
        == report["simulator_calls"]
        == report["guard_reads"]
        == report["probe_reads"]
        == report["final_reads"]
        == report["promotion_calls"]
        == 0
    )
    assert not (tmp_path / "final-state.npz").exists()


@pytest.mark.parametrize("name", ["case.json", "final-state.npz"])
def test_should_preserve_existing_outputs_before_any_source_read_or_recipe(
    data, tmp_path, monkeypatch, name
):
    monkeypatch.setattr(case, "stage", tmp_path)
    path = tmp_path / name
    path.write_bytes(b"original evidence")
    monkeypatch.setattr(case, "load_inputs", forbidden)
    monkeypatch.setattr(case, "run_continuous_learnability", forbidden)
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"original evidence"
    assert len(list(tmp_path.iterdir())) == 1
