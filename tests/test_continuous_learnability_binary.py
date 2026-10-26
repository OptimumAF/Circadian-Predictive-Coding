"""One paired48-SGD fixture plus one1-SGD refusal per invocation.

mixed_direct_reference adds24 standalone core SGD calls and must run once only.
All input/CLI/wrapper refusal tests execute zero SGD. No real-data inputs.
"""

from dataclasses import asdict, replace
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_native_stream_demo as cli
from src.adapters import continuous_learnability_demo as demo
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core import continuous_bc as core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.observation_standardization import ObservationStandardization


def episode(name="A", *, role="train", unit="normalized_gripper", grip_bounds=(-1.0, 1.0)):
    observations = (
        [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        if name == "A"
        else [(10.0, 1.0), (11.0, 1.0), (12.0, 1.0), (13.0, 1.0)]
    )
    actions = [(-2.0, -1.0), (6.0, 1.0)] if name == "A" else [(5.0, 1.0), (4.0, -1.0), (3.0, 1.0)]
    spec = ContinuousActionSpec(2, ("m", unit), (-2.0, grip_bounds[0]), (6.0, grip_bounds[1]))
    return ContinuousEpisode(
        "binary-synthetic-" + name,
        "context-" + name,
        role,
        spec,
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


def arrays(record):
    return (
        np.asarray([row.observation for row in record.transitions]),
        np.asarray([row.action for row in record.transitions]),
    )


def assert_arrays_and_counters(first, second):
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for group in ("parameters", "velocity"):
        for left, right in zip(getattr(first, group), getattr(second, group), strict=True):
            assert left.dtype == right.dtype == np.float64
            np.testing.assert_array_equal(left, right)


def initial_objective(record, state, *, binary):
    """Independent scalar-loop oracle on this tiny fixed initial network."""
    w1, b1, w2, b2 = state.parameters
    total = 0.0
    for row in record.transitions:
        x = ((row.observation[0] - 1.0), row.observation[1])
        hidden = [math.tanh(sum(x[d] * w1[d, h] for d in range(2)) + b1[h]) for h in range(64)]
        logits = [sum(hidden[h] * w2[h, axis] for h in range(64)) + b2[axis] for axis in range(2)]
        position = math.tanh(logits[0]) - (row.action[0] - 2.0) / 4.0
        gripper = (
            math.log1p(math.exp(-2 * row.action[1] * logits[1]))
            if binary
            else 0.5 * (math.tanh(logits[1]) - row.action[1]) ** 2
        )
        total += 0.5 * position**2 + gripper
    return total / (2 * len(record.transitions))


@pytest.fixture(scope="module")
def paired():
    """Exactly two24-update recipes shared by all complete-arm assertions."""
    warm, new = episode(), episode("B")
    inputs = asdict(warm), asdict(new)
    current = [None]
    observed = {
        name: dict(fits=[], trains=[], kernels=[], checkpoints={}) for name in ("mse", "mixed")
    }
    original_runtime, original_fit = demo.ActorShadowRuntime, demo.fit_training_standardization
    original_train, original_kernel = (
        NumpyContinuousBCLearner.train_batch,
        core.continuous_bc_loss_and_gradients,
    )
    original_checkpoint = demo._checkpoint
    with pytest.MonkeyPatch.context() as patch:

        def fit(record):
            observed[current[0]]["fits"].append(record)
            return original_fit(record)

        def runtime(source, **options):
            value = original_runtime(source, **options)
            observed[current[0]].update(
                source=source, runtime=value, options=options, initial=source.snapshot_state()
            )
            return value

        def train(owner, features, targets):
            observed[current[0]]["trains"].append((owner, len(features)))
            return original_train(owner, features, targets)

        def kernel(features, *args, **kwargs):
            observed[current[0]]["kernels"].append(
                (len(features), kwargs.get("binary_gripper_axis"))
            )
            return original_kernel(features, *args, **kwargs)

        def checkpoint(*args, **kwargs):
            state = args[2].candidate_snapshot().state
            observed[current[0]]["checkpoints"][state.completed_updates] = state
            return original_checkpoint(*args, **kwargs)

        patch.setattr(demo, "fit_training_standardization", fit)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(NumpyContinuousBCLearner, "train_batch", train)
        patch.setattr(core, "continuous_bc_loss_and_gradients", kernel)
        patch.setattr(demo, "_checkpoint", checkpoint)
        results = {}
        for name, axis in (("mse", None), ("mixed", 1)):
            current[0] = name
            options = {} if axis is None else {"binary_gripper_axis": axis}
            results[name] = demo.run_continuous_learnability(
                warm,
                new,
                expected_standardization=ObservationStandardization((1.0, 0.0), (1.0, 1.0), 2),
                **options,
            )
            assert results[name].report["complete"] is True
    assert (asdict(warm), asdict(new)) == inputs
    return results, observed, warm, new


def test_should_share_all_eight_initial_arrays_and_defaults_with_only_explicit_objective_axis_changed(
    paired,
):
    results, observed, _, _ = paired
    first, second = observed["mse"]["initial"], observed["mixed"]["initial"]
    assert_arrays_and_counters(first, second)
    assert first.completed_updates == first.training_examples == 0
    assert second.policy == replace(first.policy, binary_gripper_axis=1)
    assert first.policy.binary_gripper_axis is None
    assert first.policy.seed == 17 and first.policy.hidden_dim == 64
    assert first.policy.learning_rate == 0.01 and first.policy.momentum == 0.9
    assert first.policy.observation_standardization == ObservationStandardization(
        (1.0, 0.0), (1.0, 1.0), 2
    )
    for arm in ("mse", "mixed"):
        data = observed[arm]
        report = results[arm].report
        assert len(data["fits"]) == 1 and data["fits"][0].episode_id == "binary-synthetic-A"
        assert len(data["trains"]) == 24
        assert all(owner is data["runtime"]._candidate for owner, _ in data["trains"])
        assert [rows for _, rows in data["trains"]] == [2] * 8 + [3] * 8 + [2] * 8
        assert data["options"]["budget"].updates_completed == 24
        assert data["options"]["budget"].replay_examples_completed == 0
        assert data["source"].policy == data["initial"].policy
        assert data["runtime"].actor.snapshot_state().state.policy == data["initial"].policy
        assert results[arm].candidate.policy == data["initial"].policy
        assert results[arm].candidate.snapshot_state().completed_updates == 24
        assert results[arm].candidate.snapshot_state().training_examples == 56
        assert_arrays_and_counters(data["initial"], data["source"].snapshot_state())
        assert_arrays_and_counters(data["initial"], data["runtime"].actor.snapshot_state().state)
        assert_arrays_and_counters(
            results[arm].candidate.snapshot_state(), data["runtime"].candidate_snapshot().state
        )
        assert report["progress"] == dict(
            update_attempts=24, updates_completed=24, replay_examples_completed=0, next_tick=48
        )


def test_should_keep_initial_common_physical_residuals_and_all_frozen_actor_predictions_identical(
    paired,
):
    results, observed, warm, new = paired
    mse, mixed = results["mse"].report, results["mixed"].report
    for name, record in (("warm", warm), ("new", new)):
        initial_mse = mse["checkpoints"][0]["contexts"][name]
        initial_mixed = mixed["checkpoints"][0]["contexts"][name]
        assert initial_mse["residual"] == initial_mixed["residual"]
        assert initial_mse["actor_predictions"] == initial_mixed["actor_predictions"]
        for report in (mse, mixed):
            for point in report["checkpoints"]:
                assert (
                    point["contexts"][name]["actor_predictions"] == initial_mse["actor_predictions"]
                )
        for arm, binary in (("mse", False), ("mixed", True)):
            loss = results[arm].report["checkpoints"][0]["contexts"][name]["conditioning"][
                "conditioning"
            ]["objective_loss"]
            assert loss == pytest.approx(
                initial_objective(record, observed[arm]["initial"], binary=binary), abs=1e-12
            )


def test_should_use_actual_policy_objective_in_every_training_and_conditioning_receipt(paired):
    results, observed, warm, new = paired
    for arm, objective, axis in (("mse", core.OBJECTIVE, None), ("mixed", core.MIXED_OBJECTIVE, 1)):
        report, data = results[arm].report, observed[arm]
        assert len(report["applied_updates"]) == len(report["events"]) == 24
        assert len(data["kernels"]) == 32 and sum(rows for rows, _ in data["kernels"]) == 76
        assert all(observed_axis == axis for _, observed_axis in data["kernels"])
        assert [point["updates"] for point in report["checkpoints"]] == [0, 8, 16, 24]
        assert [point["training_examples"] for point in report["checkpoints"]] == [0, 16, 40, 56]
        for index, (receipt, event) in enumerate(
            zip(report["applied_updates"], report["events"], strict=True)
        ):
            assert receipt["diagnostic"]["definition"] == objective
            assert math.isfinite(receipt["diagnostic"]["value"])
            assert event["applied"] == [receipt] and event["success_validated"] is True
            assert receipt["observed_at"] == index * 2
            assert receipt["arrived_at"] == receipt["applied_at"] == index * 2 + 1
        for point in report["checkpoints"]:
            assert point["state_exact"] is point["owners_exact"] is True
            for context, record in (("warm", warm), ("new", new)):
                row = point["contexts"][context]
                conditioning, residual = (
                    row["conditioning"]["conditioning"],
                    row["residual"]["residuals"],
                )
                assert conditioning["objective_definition"] == objective
                assert (
                    conditioning["rows"] == residual["overall"]["rows"] == len(record.transitions)
                )
                assert [
                    (entry["index"], entry["unit"]) for entry in residual["overall"]["axes"]
                ] == [(0, "m"), (1, "normalized_gripper")]
                assert len(residual["stages"]) == 1 and residual["stages"][0]["label"] == "all"
                if arm == "mse":
                    assert conditioning["objective_loss"] == pytest.approx(
                        residual["overall"]["normalized_half_mse"], abs=1e-14
                    )
        json.dumps(report, allow_nan=False)


def test_should_keep_common_costs_and_physical_metric_directions_with_explicit_ce_row_work(paired):
    results, _, _, _ = paired
    mse, mixed = results["mse"].report, results["mixed"].report
    assert "binary_gripper_loss_rows" not in mse["work"]
    common = {
        key: value for key, value in mixed["work"].items() if key != "binary_gripper_loss_rows"
    }
    assert common == mse["work"]
    assert mixed["work"]["binary_gripper_loss_rows"] == 76
    assert common["optimizer_updates"] == 24 and common["training_rows"] == 56
    assert common["forward_passes"] == 56 and common["backward_passes"] == 32
    assert common["gradient_values"] == 10304 and common["dense_multiply_accumulates"] == 64000
    for report in (mse, mixed):
        values = [
            {
                name: row["residual"]["residuals"]["overall"]["normalized_half_mse"]
                for name, row in point["contexts"].items()
            }
            for point in report["checkpoints"]
        ]
        assert report["metrics"] == dict(
            warm_acquisition_gain=values[0]["warm"] - values[1]["warm"],
            new_acquisition_gain=values[1]["new"] - values[2]["new"],
            warm_retention_loss=values[2]["warm"] - values[1]["warm"],
            warm_return_recovery=values[2]["warm"] - values[3]["warm"],
            warm_return_deficit=values[3]["warm"] - values[1]["warm"],
            new_retention_loss=values[3]["new"] - values[2]["new"],
        )


def test_should_match_one_mixed_direct_reference_every_full_endpoint_and_loss(paired):
    """EXACTLY24 extra core SGD calls; admitted once globally, excluded later."""
    results, observed, warm, new = paired
    initial = observed["mixed"]["initial"]
    parameters, velocity = (
        tuple(value.copy() for value in initial.parameters),
        tuple(value.copy() for value in initial.velocity),
    )
    losses, rows = [], 0
    for index in range(24):
        record = new if 8 <= index < 16 else warm
        features, targets = arrays(record)
        features = features - np.asarray([1.0, 0.0])
        result = core.continuous_bc_loss_and_gradients(
            features, targets, parameters, initial.policy.spec, binary_gripper_axis=1
        )
        losses.append(result.loss)
        velocity = tuple(
            0.9 * old + gradient for old, gradient in zip(velocity, result.gradients, strict=True)
        )
        parameters = tuple(
            old - 0.01 * step for old, step in zip(parameters, velocity, strict=True)
        )
        rows += len(record.transitions)
        if index + 1 in (8, 16, 24):
            expected = observed["mixed"]["checkpoints"][index + 1]
            assert expected.completed_updates == index + 1 and expected.training_examples == rows
            for group, values in (("parameters", parameters), ("velocity", velocity)):
                for actual, target in zip(values, getattr(expected, group), strict=True):
                    np.testing.assert_array_equal(actual, target)
    assert [
        row["diagnostic"]["value"] for row in results["mixed"].report["applied_updates"]
    ] == losses


def test_should_refuse_wrong_mse_receipt_after_one_real_mixed_update_and_preserve_charge(
    monkeypatch,
):
    """Exactly1 mixed SGD; the original mixed receipt remains independently stored."""
    captured = []
    original = demo.ActorShadowRuntime

    def runtime(source, **options):
        owner = original(source, **options)
        captured.append((owner, options))
        ready = owner.train_ready

        def wrong_objective(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if applied:
                assert options["budget"].updates_completed == 1
                return (
                    replace(
                        applied[0],
                        diagnostic=TrainingDiagnostic(core.OBJECTIVE, applied[0].diagnostic.value),
                    ),
                )
            return applied

        monkeypatch.setattr(owner, "train_ready", wrong_objective)
        return owner

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_learnability(episode(), episode("B"), binary_gripper_axis=1)
    owner, options = captured[0]
    assert result.candidate is None and result.report["complete"] is False
    assert len(result.report["events"]) == len(result.report["applied_updates"]) == 1
    assert result.report["events"][0]["success_validated"] is False
    assert result.report["events"][0]["applied"][0]["diagnostic"]["definition"] == core.OBJECTIVE
    assert result.report["applied_updates"][0]["diagnostic"]["definition"] == core.MIXED_OBJECTIVE
    assert (
        result.report["progress"]["updates_completed"] == options["budget"].updates_completed == 1
    )
    assert owner.candidate_snapshot().state.completed_updates == 1


@pytest.mark.parametrize("entry", ["result", "json"])
@pytest.mark.parametrize("axis", [-1, 2, 0, True, 1.0, "1", np.int64(1)])
def test_should_refuse_invalid_axis_type_index_and_unit_before_any_fit_or_model(
    monkeypatch, entry, axis
):
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("invalid objective reached fit"),
    )
    monkeypatch.setattr(
        demo,
        "NumpyContinuousBCLearner",
        lambda *_a, **_k: pytest.fail("invalid objective constructed model"),
    )
    execute = (
        demo.run_continuous_learnability
        if entry == "result"
        else demo.run_continuous_learnability_demo
    )
    with pytest.raises(ValueError):
        execute(episode(), episode("B"), binary_gripper_axis=axis)


@pytest.mark.parametrize("position", ["warm", "new"])
@pytest.mark.parametrize("label", [0.0, 0.25])
def test_should_check_both_episode_endpoint_labels_before_fit_or_model(
    monkeypatch, position, label
):
    warm, new = episode(), episode("B")
    record = warm if position == "warm" else new
    rows = (
        replace(record.transitions[0], action=(record.transitions[0].action[0], label)),
    ) + record.transitions[1:]
    changed = replace(record, transitions=rows)
    if position == "warm":
        warm = changed
    else:
        new = changed
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("nonbinary label reached fitting"),
    )
    monkeypatch.setattr(
        demo,
        "NumpyContinuousBCLearner",
        lambda *_a, **_k: pytest.fail("nonbinary label constructed model"),
    )
    with pytest.raises(ValueError, match="endpoint"):
        demo.run_continuous_learnability(warm, new, binary_gripper_axis=1)


@pytest.mark.parametrize("entry", ["result", "json"])
def test_should_preserve_mse_acceptance_of_interior_gripper_labels_without_training(
    monkeypatch, entry
):
    warm, new = episode(), episode("B")
    new = replace(
        new, transitions=(replace(new.transitions[0], action=(5.0, 0.5)),) + new.transitions[1:]
    )
    calls = []

    class ReachedFit(RuntimeError):
        pass

    def fit(record):
        calls.append(record)
        raise ReachedFit("default MSE passed objective admission")

    monkeypatch.setattr(demo, "fit_training_standardization", fit)
    execute = (
        demo.run_continuous_learnability
        if entry == "result"
        else demo.run_continuous_learnability_demo
    )
    with pytest.raises(ReachedFit):
        execute(warm, new)
    assert calls == [warm]


@pytest.mark.parametrize("bad", ["bounds", "constant", "unit"])
def test_should_refuse_wrong_gripper_geometry_before_fit_even_for_valid_complete_records(
    monkeypatch, bad
):
    if bad == "unit":
        warm, new = episode(unit="m"), episode("B", unit="m")
    elif bad == "bounds":
        warm, new = episode(grip_bounds=(-2.0, 2.0)), episode("B", grip_bounds=(-2.0, 2.0))
    else:
        values = []
        for record in (episode(), episode("B")):
            rows = tuple(replace(row, action=(row.action[0], 1.0)) for row in record.transitions)
            spec = ContinuousActionSpec(2, record.spec.action_units, (-2.0, 1.0), (6.0, 1.0))
            values.append(replace(record, spec=spec, transitions=rows))
        warm, new = values
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("invalid binary geometry reached fitting"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(warm, new, binary_gripper_axis=1)


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
def test_should_refuse_held_out_roles_before_binary_label_access_or_fit(monkeypatch, role):
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("held-out objective reached fit"),
    )
    monkeypatch.setattr(
        core,
        "_binary_gripper_targets",
        lambda *_a, **_k: pytest.fail("held-out labels reached binary validation"),
    )
    with pytest.raises(ValueError, match="TRAIN"):
        demo.run_continuous_learnability(episode(), episode("B", role=role), binary_gripper_axis=1)


@pytest.mark.parametrize("axis", [None, 1])
def test_should_preserve_default_json_delegate_kwargs_and_forward_only_explicit_binary_axis(
    monkeypatch, axis
):
    payload, calls = {"complete": True}, []

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(report=payload, candidate=object())

    monkeypatch.setattr(demo, "run_continuous_learnability", execute)
    warm, new = episode(), episode("B")
    assert demo.run_continuous_learnability_demo(warm, new, binary_gripper_axis=axis) is payload
    expected = {"expected_standardization": None}
    if axis is not None:
        expected["binary_gripper_axis"] = axis
    assert calls == [((warm, new), expected)]


@pytest.mark.parametrize("axis", [None, 3])
@pytest.mark.parametrize("visual", [False, True])
def test_should_forward_explicit_cli_axis_only_to_the_learning_recipe(
    monkeypatch, capsys, axis, visual
):
    reads, calls = [], []
    records = [object(), object()]

    def read(path, seed, **kwargs):
        reads.append((str(path), seed, kwargs))
        return records[len(reads) - 1]

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(complete=True)

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", execute)
    arguments = [
        "demo",
        "--scenario",
        "continuous-mlp-learnability",
        "--warm-teacher",
        "A.json",
        "--new-teacher",
        "B.json",
    ]
    if axis is not None:
        arguments += ["--binary-gripper-axis", str(axis)]
    if visual:
        arguments.append("--visual-observation")
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    options = {"visual_observation": True} if visual else {}
    assert reads == [("A.json", 17, options), ("B.json", 19, options)]
    assert calls == [(tuple(records), {} if axis is None else {"binary_gripper_axis": axis})]
    result = json.loads(capsys.readouterr().out)
    assert result["complete"] is True
    assert ("input_observation" in result) is visual


@pytest.mark.parametrize(
    "scenario",
    [
        None,
        "stream",
        "continuous-nearest",
        "promotion-rejection",
        "coupled-neutral",
        "matched-baselines",
    ],
)
def test_should_refuse_binary_flag_for_every_other_cli_scenario_before_teacher_reads(
    monkeypatch, scenario
):
    arguments = ["demo", "--binary-gripper-axis", "1"]
    if scenario is not None:
        arguments += ["--scenario", scenario]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        cli, "_read_teacher", lambda *_a, **_k: pytest.fail("illegal objective mode read a teacher")
    )
    monkeypatch.setattr(
        cli, "run_demo", lambda *_a, **_k: pytest.fail("illegal objective mode ran legacy models")
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("axis", ["1.0", "True", "gripper"])
def test_should_refuse_non_integer_cli_axis_before_file_reads(monkeypatch, axis):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--scenario",
            "continuous-mlp-learnability",
            "--binary-gripper-axis",
            axis,
            "--warm-teacher",
            "A.json",
            "--new-teacher",
            "B.json",
        ],
    )
    monkeypatch.setattr(
        cli,
        "_read_teacher",
        lambda *_a, **_k: pytest.fail("noninteger objective flag read a teacher"),
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
