"""Fixed semantic RGB scaling; one shared24-SGD integration per invocation.

All other checks are pure math, admission, or routing without model updates.
No direct-reference training is needed because prediction/SGD math is unchanged.
"""

from dataclasses import asdict, replace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_native_stream_demo as cli
from src.adapters import continuous_learnability_demo as demo
from src.adapters import numpy_continuous_bc as adapter
from src.core.continuous_bc import OBJECTIVE
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import (
    fit_observation_standardization,
    standardize_observations,
)
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.core import pick_cube_rgb_standardization as rgb
from src.infra.maniskill_continuous import ACTION_UNITS
from test_continuous_learnability_demo import same_state


def features():
    values = np.empty((2, 789), dtype=np.float64)
    values[0, :21], values[1, :21] = 0.0, 2.0
    values[:, 1] = 7.0
    values[0, 21:], values[1, 21:] = 0.1, 0.9
    return values


def episode(name="A", *, role="train"):
    rows, offset = (2, 0) if name == "A" else (3, 10)
    observations = []
    for index in range(rows + 1):
        prefix = (float(offset + 2 * index), 7.0) + (float(offset + 2 * index),) * 19
        pixels = tuple(float((index + column) % 3) / 2 for column in range(768))
        observations.append(prefix + pixels)
    return ContinuousEpisode(
        f"synthetic-rgb-range-{name}",
        f"context-{name}",
        role,
        ContinuousActionSpec(789, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4),
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                (0.25, -0.5, 0.75, 1.0 if index % 2 == 0 else -1.0),
                0.0,
                observations[index + 1],
                index == rows - 1,
                False,
            )
            for index in range(rows)
        ),
    )


def test_should_fit_only_twenty_one_public_vectors_and_use_semantic_rgb_range():
    values = features()
    result = rgb.fit_rgb_range_standardization(values, pick_cube_rgb_recipe())
    assert result.fitted_rows == 2
    assert result.means[:21] == (1.0, 7.0) + (1.0,) * 19
    assert result.scales[:21] == (1.0,) * 21
    assert result.means[21:] == result.scales[21:] == (0.5,) * 768
    prefix = fit_observation_standardization(values[:, :21].copy())
    assert result.means[:21] == prefix.means and result.scales[:21] == prefix.scales
    empirical = fit_observation_standardization(values)
    assert empirical.scales[21] != result.scales[21]
    values[:] = 0.0
    assert result.means[1] == 7.0 and result.scales[-1] == 0.5


def test_should_map_rgb_endpoints_exactly_and_preserve_new_constant_vector_deviations():
    transform = rgb.fit_rgb_range_standardization(features(), pick_cube_rgb_recipe())
    values = features()
    values[0, 21:], values[1, 21:] = 0.0, 1.0
    values[:, 1] = 9.0
    spec = episode().spec
    result = standardize_observations(values, transform, spec)
    np.testing.assert_array_equal(result[0, 21:], -np.ones(768))
    np.testing.assert_array_equal(result[1, 21:], np.ones(768))
    np.testing.assert_array_equal(result[:, 1], np.array([2.0, 2.0]))
    assert result[0, 0] == -1.0 and result[1, 0] == 1.0
    assert not np.shares_memory(result, values)


def test_should_validate_owned_float64_copy_without_clipping_prefix_or_changing_rng():
    values = features()
    values[:, 0] = (-1e100, 1e100)
    values.setflags(write=False)
    before = np.random.get_state()
    validated = rgb.validate_rgb_range_features(values, pick_cube_rgb_recipe())
    rgb.fit_rgb_range_standardization(values, pick_cube_rgb_recipe())
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(validated, values)
    assert type(validated) is np.ndarray and validated.dtype == np.float64
    assert validated.flags.writeable and not np.shares_memory(validated, values)
    validated[0, 0] = 0.0
    assert values[0, 0] == -1e100


@pytest.mark.parametrize("operation", ["validate", "fit"])
@pytest.mark.parametrize(
    "damage",
    [
        "list",
        "subclass",
        "float32",
        "integer",
        "one_dimensional",
        "zero_rows",
        "too_many_rows",
        "wrong_dimension",
        "nan",
        "infinity",
        "rgb_low",
        "rgb_high",
    ],
)
def test_should_refuse_non_native_or_out_of_contract_features(operation, damage):
    values = features()
    if damage == "list":
        values = values.tolist()
    elif damage == "subclass":

        class Derived(np.ndarray):
            pass

        values = values.view(Derived)
    elif damage == "float32":
        values = values.astype(np.float32)
    elif damage == "integer":
        values = values.astype(np.int64)
    elif damage == "one_dimensional":
        values = values[0]
    elif damage == "zero_rows":
        values = values[:0]
    elif damage == "too_many_rows":
        values = np.zeros((257, 789), dtype=np.float64)
    elif damage == "wrong_dimension":
        values = values[:, :788]
    else:
        coordinate = 0 if damage in ("nan", "infinity") else 788
        values[0, coordinate] = {
            "nan": float("nan"),
            "infinity": float("inf"),
            "rgb_low": -1e-12,
            "rgb_high": 1.0 + 1e-12,
        }[damage]
    execute = (
        rgb.validate_rgb_range_features
        if operation == "validate"
        else rgb.fit_rgb_range_standardization
    )
    with pytest.raises(ValueError):
        execute(values, pick_cube_rgb_recipe())


@pytest.mark.parametrize(
    "damage", ["none", "mapping", "subclass", "equal_wrong_type", "changed", "tuple_item_type"]
)
def test_should_require_full_exact_frozen_recipe(damage):
    recipe = pick_cube_rgb_recipe()
    if damage == "none":
        recipe = None
    elif damage == "mapping":
        recipe = asdict(recipe)
    elif damage == "subclass":

        class Derived(type(recipe)):
            pass

        recipe = Derived()
    elif damage == "equal_wrong_type":
        object.__setattr__(recipe, "feature_dim", 789.0)
    elif damage == "changed":
        object.__setattr__(recipe, "camera", "other_camera")
    else:
        object.__setattr__(recipe, "pooled_shape", (16.0, 16, 3))
    with pytest.raises(ValueError):
        rgb.fit_rgb_range_standardization(features(), recipe)


@pytest.mark.parametrize("rows", [1, 256])
def test_should_accept_bounded_row_extremes_and_constant_prefix(rows):
    values = np.zeros((rows, 789), dtype=np.float64)
    result = rgb.fit_rgb_range_standardization(values, pick_cube_rgb_recipe())
    assert result.fitted_rows == rows
    assert result.means[:21] == (0.0,) * 21 and result.scales[:21] == (1.0,) * 21
    assert result.means[21:] == result.scales[21:] == (0.5,) * 768


def test_should_pass_only_owned_prefix_to_population_fitter(monkeypatch):
    values, calls = features(), []
    original = rgb.fit_observation_standardization

    def fit(prefix):
        calls.append(prefix.copy())
        assert prefix.shape == (2, 21)
        assert not np.shares_memory(prefix, values)
        return original(prefix)

    monkeypatch.setattr(rgb, "fit_observation_standardization", fit)
    rgb.fit_rgb_range_standardization(values, pick_cube_rgb_recipe())
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], values[:, :21])


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
def test_should_check_train_authority_before_range_validation_or_fit(monkeypatch, role):
    monkeypatch.setattr(
        adapter,
        "fit_rgb_range_standardization",
        lambda *_: pytest.fail("heldout reached range fit"),
    )
    with pytest.raises(ValueError):
        adapter.fit_training_standardization(
            episode(role=role), rgb_range_recipe=pick_cube_rgb_recipe()
        )


def test_should_preserve_default_fit_exactly_and_forward_only_explicit_recipe(monkeypatch):
    original = episode()
    values = np.asarray([row.observation for row in original.transitions], dtype=np.float64)
    assert adapter.fit_training_standardization(original) == fit_observation_standardization(values)
    calls = []
    expected = rgb.fit_rgb_range_standardization(values, pick_cube_rgb_recipe())

    def fit(received, recipe):
        calls.append((received.copy(), recipe))
        return expected

    monkeypatch.setattr(adapter, "fit_rgb_range_standardization", fit)
    assert (
        adapter.fit_training_standardization(original, rgb_range_recipe=pick_cube_rgb_recipe())
        == expected
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], values)
    assert calls[0][1] == pick_cube_rgb_recipe()


@pytest.fixture(scope="module")
def completed():
    """Exactly24 runtime SGD per pytest process; all integration assertions share it."""
    warm, new = episode(), episode("B")
    originals = asdict(warm), asdict(new)
    observed = dict(fits=[], runtimes=[], trains=[], checkpoints=[])
    original_fit = demo.fit_training_standardization
    original_runtime = demo.ActorShadowRuntime
    original_train = adapter.NumpyContinuousBCLearner.train_batch
    original_checkpoint = demo._checkpoint
    with pytest.MonkeyPatch.context() as patch:

        def fit(record, **options):
            observed["fits"].append((record, options))
            return original_fit(record, **options)

        def runtime(source, **options):
            created = original_runtime(source, **options)
            observed["runtimes"].append((source, created, options, source.snapshot_state()))
            return created

        def train(owner, values, targets):
            observed["trains"].append((owner, len(values)))
            return original_train(owner, values, targets)

        def checkpoint(*args, **kwargs):
            observed["checkpoints"].append(args[3])
            return original_checkpoint(*args, **kwargs)

        patch.setattr(demo, "fit_training_standardization", fit)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(adapter.NumpyContinuousBCLearner, "train_batch", train)
        patch.setattr(demo, "_checkpoint", checkpoint)
        result = demo.run_continuous_learnability(
            warm, new, rgb_range_recipe=pick_cube_rgb_recipe()
        )
    assert (asdict(warm), asdict(new)) == originals
    assert result.report["complete"] is True
    return result, observed, warm, new


def test_should_fit_a_once_keep_b_unseen_and_preserve_fixed_runtime_and_owners(completed):
    result, observed, warm, _ = completed
    source, runtime, options, initial = observed["runtimes"][0]
    assert len(observed["runtimes"]) == 1
    assert observed["fits"] == [(warm, {"rgb_range_recipe": pick_cube_rgb_recipe()})]
    assert len(observed["trains"]) == 24
    assert [rows for _, rows in observed["trains"]] == [2] * 8 + [3] * 8 + [2] * 8
    assert all(owner is runtime._candidate for owner, _ in observed["trains"])
    assert len(observed["checkpoints"]) == 4
    assert all(owner is result.candidate for owner in observed["checkpoints"])
    same_state(initial, source.snapshot_state())
    same_state(initial, runtime.actor.snapshot_state().state)
    same_state(result.candidate.snapshot_state(), runtime.candidate_snapshot().state)
    assert options["budget"].updates_completed == 24
    state = result.candidate.snapshot_state()
    assert state.completed_updates == 24 and state.training_examples == 56
    assert state.policy.observation_standardization.means[21:] == (0.5,) * 768
    assert state.policy.observation_standardization.scales[21:] == (0.5,) * 768
    assert state.policy.observation_standardization.fitted_rows == 2
    assert state.policy.seed == 17 and state.policy.hidden_dim == 64
    assert state.policy.learning_rate == 0.01 and state.policy.momentum == 0.9
    assert state.policy.binary_gripper_axis is None


def test_should_preserve_all_delayed_receipts_endpoints_and_training_objective(completed):
    result, _, _, _ = completed
    report = result.report
    assert [row["updates"] for row in report["checkpoints"]] == [0, 8, 16, 24]
    assert [row["training_examples"] for row in report["checkpoints"]] == [0, 16, 40, 56]
    assert all(row["state_exact"] is row["owners_exact"] is True for row in report["checkpoints"])
    for index, row in enumerate(report["events"]):
        assert row["observed_at"] == 2 * index and row["arrived_at"] == 2 * index + 1
        assert row["success_validated"] is row["deferred_exact"] is True
        assert row["applied"][0]["diagnostic"]["definition"] == OBJECTIVE
        assert row["applied"][0] == report["applied_updates"][index]
    assert report["progress"]["updates_completed"] == 24
    assert report["progress"]["next_tick"] == 48
    json.dumps(report, allow_nan=False)


def test_should_change_only_declared_input_statistics_in_numeric_initialization(completed):
    _, observed, warm, _ = completed
    _, _, _, initial = observed["runtimes"][0]
    empirical = adapter.fit_training_standardization(warm)
    policy = replace(initial.policy, observation_standardization=empirical)
    template = adapter.NumpyContinuousBCLearner(policy).snapshot_state()
    assert (
        template.policy.observation_standardization.means[:21]
        == initial.policy.observation_standardization.means[:21]
    )
    assert (
        template.policy.observation_standardization.scales[:21]
        == initial.policy.observation_standardization.scales[:21]
    )
    assert (
        template.policy.observation_standardization.scales[21:]
        != initial.policy.observation_standardization.scales[21:]
    )
    assert (
        replace(
            template.policy, observation_standardization=initial.policy.observation_standardization
        )
        == initial.policy
    )
    assert template.completed_updates == initial.completed_updates == 0
    assert template.training_examples == initial.training_examples == 0
    for group in ("parameters", "velocity"):
        for left, right in zip(getattr(template, group), getattr(initial, group), strict=True):
            np.testing.assert_array_equal(left, right)


def test_should_account_for_reduced_fit_only_while_preserving_all_apply_and_model_work(completed):
    result, _, warm, new = completed
    report = result.report
    assert report["input_normalization"] == dict(
        definition="pick_cube_train_prefix21_fixed_rgb_range_v1",
        recipe=asdict(pick_cube_rgb_recipe()),
        fitted_prefix_dimensions=21,
        fixed_range_dimensions=768,
        rgb_mean=0.5,
        rgb_scale=0.5,
        range_validation_rows=7,
        terminal_range_validation_rows=3,
    )
    # The pre-existing work oracle uses only shape/row counts, independent of transform.
    ordinary = demo._work(dict(warm=warm, new=new), result.candidate.policy)
    observed = dict(report["work"])
    assert observed.pop("fixed_range_dimensions") == 768
    assert observed.pop("fit_feature_values") == 42
    ordinary.pop("fit_feature_values")
    assert observed == ordinary
    assert report["work"]["fit_rows"] == 2
    assert report["work"]["training_rows"] == 56
    assert report["work"]["normalization_values"] == 116 * 789
    assert report["work"]["forward_passes"] == 56
    assert report["work"]["backward_passes"] == 32


@pytest.mark.parametrize("position", ["warm", "new"])
@pytest.mark.parametrize(
    "damage", ["low", "high", "role", "dimension", "terminal_low", "terminal_high"]
)
def test_should_validate_both_train_inputs_before_fit_or_model(monkeypatch, position, damage):
    warm, new = episode(), episode("B")
    selected = warm if position == "warm" else new
    if damage == "role":
        selected = replace(selected, role="probe_test")
    elif damage.startswith("terminal_"):
        changed = list(selected.transitions[-1].next_observation)
        changed[788] = -0.01 if damage == "terminal_low" else 1.01
        selected = replace(
            selected,
            transitions=selected.transitions[:-1]
            + (replace(selected.transitions[-1], next_observation=tuple(changed)),),
        )
    elif damage == "dimension":
        selected = replace(
            selected,
            spec=replace(selected.spec, observation_dim=788),
            transitions=tuple(
                replace(
                    row,
                    observation=row.observation[:788],
                    next_observation=row.next_observation[:788],
                )
                for row in selected.transitions
            ),
        )
    else:
        changed = list(selected.transitions[0].observation)
        changed[788] = -0.01 if damage == "low" else 1.01
        selected = replace(
            selected,
            transitions=(replace(selected.transitions[0], observation=tuple(changed)),)
            + selected.transitions[1:],
        )
    if position == "warm":
        warm = selected
    else:
        new = selected
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_a, **_k: pytest.fail("invalid range reached fitting"),
    )
    monkeypatch.setattr(
        demo,
        "NumpyContinuousBCLearner",
        lambda *_a, **_k: pytest.fail("invalid range created model"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(warm, new, rgb_range_recipe=pick_cube_rgb_recipe())


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_should_refuse_invalid_terminal_rgb_before_direct_training_fit(monkeypatch, value):
    source = episode()
    changed = list(source.transitions[-1].next_observation)
    changed[21] = value
    source = replace(
        source,
        transitions=source.transitions[:-1]
        + (replace(source.transitions[-1], next_observation=tuple(changed)),),
    )
    monkeypatch.setattr(
        adapter,
        "fit_rgb_range_standardization",
        lambda *_: pytest.fail("terminal violation reached fitting"),
    )
    with pytest.raises(ValueError, match="RGB"):
        adapter.fit_training_standardization(source, rgb_range_recipe=pick_cube_rgb_recipe())


@pytest.mark.parametrize("entry", ["result", "json"])
def test_should_compare_expected_statistics_after_single_a_fit_before_constructing_model(
    monkeypatch, entry
):
    warm, new = episode(), episode("B")
    empirical = adapter.fit_training_standardization(warm)
    calls, fit = [], demo.fit_training_standardization

    def counted(record, **options):
        calls.append((record, options))
        return fit(record, **options)

    monkeypatch.setattr(demo, "fit_training_standardization", counted)
    monkeypatch.setattr(
        demo,
        "NumpyContinuousBCLearner",
        lambda *_a, **_k: pytest.fail("mismatched expected stats constructed model"),
    )
    execute = (
        demo.run_continuous_learnability
        if entry == "result"
        else demo.run_continuous_learnability_demo
    )
    with pytest.raises(ValueError, match="expected frozen"):
        execute(
            warm, new, rgb_range_recipe=pick_cube_rgb_recipe(), expected_standardization=empirical
        )
    assert calls == [(warm, dict(rgb_range_recipe=pick_cube_rgb_recipe()))]


@pytest.mark.parametrize("axis", [None, 3])
@pytest.mark.parametrize("explicit", [False, True])
def test_should_preserve_default_json_calls_and_forward_only_opted_in_recipe(
    monkeypatch, axis, explicit
):
    calls, payload = [], {"complete": True}

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(report=payload, candidate=None)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    warm, new = episode(), episode("B")
    options = dict(binary_gripper_axis=axis)
    if explicit:
        options["rgb_range_recipe"] = pick_cube_rgb_recipe()
    assert demo.run_continuous_learnability_demo(warm, new, **options) is payload
    expected = dict(expected_standardization=None)
    if explicit or axis is not None:
        expected["binary_gripper_axis"] = axis
    if explicit:
        expected["rgb_range_recipe"] = pick_cube_rgb_recipe()
    assert calls == [((warm, new), expected)]


@pytest.mark.parametrize("axis", [None, 3])
def test_should_forward_visual_range_flag_and_independent_objective_before_recipe(
    monkeypatch, capsys, axis
):
    records, reads, runs = [object(), object()], [], []

    def read(path, seed, **options):
        reads.append((str(path), seed, options))
        return records[len(reads) - 1]

    def run(*args, **options):
        runs.append((args, options))
        return dict(complete=True)

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
    arguments = [
        "demo",
        "--scenario",
        "continuous-mlp-learnability",
        "--warm-teacher",
        "A.json",
        "--new-teacher",
        "B.json",
        "--visual-observation",
        "--rgb-range-normalization",
    ]
    if axis is not None:
        arguments += ["--binary-gripper-axis", str(axis)]
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    assert reads == [
        ("A.json", 17, dict(visual_observation=True)),
        ("B.json", 19, dict(visual_observation=True)),
    ]
    expected = dict(rgb_range_recipe=pick_cube_rgb_recipe())
    if axis is not None:
        expected["binary_gripper_axis"] = axis
    assert runs == [(tuple(records), expected)]
    assert json.loads(capsys.readouterr().out)["input_observation"]["mode"] == "public_rgb"


@pytest.mark.parametrize(
    "scenario,visual",
    [
        (None, False),
        ("stream", True),
        ("continuous-nearest", True),
        ("promotion-rejection", True),
        ("coupled-neutral", False),
        ("matched-baselines", True),
        ("continuous-mlp-learnability", False),
    ],
)
def test_should_refuse_range_flag_without_visual_mlp_mode_before_any_io(
    monkeypatch, scenario, visual
):
    arguments = [
        "demo",
        "--rgb-range-normalization",
        "--warm-teacher",
        "A.json",
        "--new-teacher",
        "B.json",
    ]
    if scenario is not None:
        arguments += ["--scenario", scenario]
    if visual:
        arguments.append("--visual-observation")
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        cli, "_read_teacher", lambda *_a, **_k: pytest.fail("illegal range mode read teacher")
    )
    monkeypatch.setattr(
        cli, "run_demo", lambda *_a, **_k: pytest.fail("illegal range mode ran model")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
