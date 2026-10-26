"""One shared tiny public21 learning run; routing and stage fakes add no SGD."""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
import math
import sys

import numpy as np
import pytest

from scripts import run_native_stream_demo as cli
from src.adapters import continuous_learnability_demo as demo
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core.continuous_bc import OBJECTIVE
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_public_vectors import public_vector_projection
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from test_continuous_learnability_demo import same_state
from test_visual_native_context_case import teacher_receipt


case = importlib.import_module("artifacts.runs.public-vector-mlp-learnability-20261019.run_case")


def episode(seed=17, *, role="train", rows=2):
    value = teacher_receipt(seed, rows)["candidate_episode"]
    spec = value["spec"]
    return ContinuousEpisode(
        value["episode_id"],
        value["context"],
        role,
        ContinuousActionSpec(
            spec["observation_dim"],
            tuple(spec["action_units"]),
            tuple(spec["lower_bounds"]),
            tuple(spec["upper_bounds"]),
        ),
        tuple(
            ContinuousTransition(
                row["step"],
                tuple(row["observation"]),
                tuple(row["action"]),
                row["reward"],
                tuple(row["next_observation"]),
                row["terminated"],
                row["truncated"],
            )
            for row in value["transitions"]
        ),
    )


def test_should_project_owned_train_pair_and_count_every_observation_and_terminal_read():
    warm, new = episode(), episode(19)
    originals = asdict(warm), asdict(new)
    a, b, metadata = cli._project_public_inputs(warm, new)
    assert a.spec.observation_dim == b.spec.observation_dim == 21
    assert a.context == "pick-cube" and b.context == "pick-cube-train-19"
    for original, projected in ((warm, a), (new, b)):
        assert projected.episode_id == original.episode_id and projected.role == "train"
        assert projected.spec.action_units == original.spec.action_units
        for source, result in zip(original.transitions, projected.transitions, strict=True):
            assert result.observation == source.observation[:21]
            assert result.next_observation == source.next_observation[:21]
            assert (result.action, result.reward, result.terminated, result.truncated) == (
                source.action,
                source.reward,
                source.terminated,
                source.truncated,
            )
    assert (asdict(warm), asdict(new)) == originals
    assert metadata["projection"] == asdict(public_vector_projection())
    assert metadata["source_recipe"] == asdict(pick_cube_rgb_recipe())
    assert metadata["observation_projections"] == 8
    assert metadata["source_values_validated"] == 8 * 789
    assert metadata["projected_values"] == 8 * 21
    assert metadata["projected_numeric_bytes"] == 8 * 168
    object.__setattr__(warm.transitions[0], "action", (0.0,) * 4)
    assert a.transitions[0].action != (0.0,) * 4


@pytest.mark.parametrize("position", ["warm", "new"])
@pytest.mark.parametrize(
    "damage", ["role", "context", "rgb_observation", "rgb_terminal", "dimension"]
)
def test_should_refuse_invalid_raw_pair_before_any_recipe_or_fit(monkeypatch, position, damage):
    warm, new = episode(), episode(19)
    target = warm if position == "warm" else new
    if damage == "role":
        target = replace(target, role="final_test")
    elif damage == "context":
        target = replace(target, context="pick-cube-probe-23")
    elif damage == "rgb_observation":
        first = target.transitions[0]
        target = replace(
            target,
            transitions=(replace(first, observation=first.observation[:-1] + (1.01,)),)
            + target.transitions[1:],
        )
    elif damage == "rgb_terminal":
        last = target.transitions[-1]
        target = replace(
            target,
            transitions=target.transitions[:-1]
            + (replace(last, next_observation=last.next_observation[:-1] + (-0.01,)),),
        )
    else:
        target = replace(
            target,
            spec=replace(target.spec, observation_dim=788),
            transitions=tuple(
                replace(
                    row,
                    observation=row.observation[:-1],
                    next_observation=row.next_observation[:-1],
                )
                for row in target.transitions
            ),
        )
    if position == "warm":
        warm = target
    else:
        new = target
    monkeypatch.setattr(
        cli,
        "run_continuous_learnability_demo",
        lambda *_a, **_k: pytest.fail("invalid raw source reached recipe"),
    )
    with pytest.raises(ValueError):
        cli._project_public_inputs(warm, new)


@pytest.fixture(scope="module")
def completed():
    """Exactly ONE24-SGD21D run per pytest process, shared by all model assertions."""
    raw_a, raw_b = episode(), episode(19)
    warm, new, projection = cli._project_public_inputs(raw_a, raw_b)
    observed = dict(fits=[], trains=[], runtimes=[], restores=[])
    fit, train, runtime_factory = (
        demo.fit_training_standardization,
        NumpyContinuousBCLearner.train_batch,
        demo.ActorShadowRuntime,
    )
    restore = NumpyContinuousBCLearner.restore_state
    with pytest.MonkeyPatch.context() as patch:

        def fitted(record, **options):
            observed["fits"].append((record, options))
            return fit(record, **options)

        def trained(owner, features, targets):
            observed["trains"].append((owner, len(features)))
            return train(owner, features, targets)

        def runtime(source, **options):
            created = runtime_factory(source, **options)
            observed["runtimes"].append((source, created, options, source.snapshot_state()))
            return created

        def restored(owner, state):
            observed["restores"].append((state.completed_updates, state.training_examples))
            return restore(owner, state)

        patch.setattr(demo, "fit_training_standardization", fitted)
        patch.setattr(NumpyContinuousBCLearner, "train_batch", trained)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(NumpyContinuousBCLearner, "restore_state", restored)
        result = demo.run_continuous_learnability(warm, new)
    assert result.report["complete"] is True
    return result, observed, warm, new, projection


def test_should_fit_public_a_once_and_keep_full_source_actor_and_optimizer_continuation(completed):
    result, observed, warm, _, _ = completed
    assert observed["fits"] == [(warm, {})]
    assert len(observed["runtimes"]) == 1
    source, runtime, options, initial = observed["runtimes"][0]
    assert [rows for _, rows in observed["trains"]] == [2] * 24
    assert all(owner is runtime._candidate for owner, _ in observed["trains"])
    same_state(initial, source.snapshot_state())
    same_state(initial, runtime.actor.snapshot_state().state)
    state = result.candidate.snapshot_state()
    same_state(state, runtime.candidate_snapshot().state)
    assert state.completed_updates == options["budget"].updates_completed == 24
    assert state.training_examples == 48 and state.policy.spec.observation_dim == 21
    assert state.policy.hidden_dim == 64 and state.policy.seed == 17
    assert state.policy.learning_rate == 0.01 and state.policy.momentum == 0.9
    assert state.policy.training_role == "train" and state.policy.binary_gripper_axis is None
    assert state.policy.observation_standardization.means == (0.5,) + (0.0,) * 20
    assert state.policy.observation_standardization.scales == (0.5,) + (1.0,) * 20
    assert state.policy.observation_standardization.fitted_rows == 2
    assert [value.shape for value in state.parameters] == [(21, 64), (64,), (64, 4), (4,)]
    assert any(np.any(value != 0) for value in state.velocity)
    assert observed["restores"] == [(0, 0), (8, 16), (16, 32), (24, 48)]


def test_should_retain_all_delayed_losses_endpoints_and_dimension_specific_work(completed):
    result, _, _, _, _ = completed
    report = result.report
    assert [row["updates"] for row in report["checkpoints"]] == [0, 8, 16, 24]
    assert [row["training_examples"] for row in report["checkpoints"]] == [0, 16, 32, 48]
    for index, row in enumerate(report["events"]):
        assert row["observed_at"] == 2 * index and row["arrived_at"] == 2 * index + 1
        assert row["deferred_exact"] is row["success_validated"] is True
        assert row["applied"][0] == report["applied_updates"][index]
        assert row["applied"][0]["diagnostic"]["definition"] == OBJECTIVE
    assert set(report["metrics"]) == {
        "warm_acquisition_gain",
        "new_acquisition_gain",
        "warm_retention_loss",
        "warm_return_recovery",
        "warm_return_deficit",
        "new_retention_loss",
    }
    expected = dict(
        fit_rows=2,
        fit_feature_values=42,
        optimizer_updates=24,
        training_rows=48,
        unique_source_rows=4,
        repeated_training_rows=44,
        normalization_rows=96,
        normalization_values=2016,
        forward_passes=56,
        forward_rows=112,
        backward_passes=32,
        backward_rows=64,
        parameter_values_per_owner=1668,
        model_numeric_bytes_per_owner=26688,
        statistics_numeric_bytes_per_owner=336,
        gradient_values=53376,
        gradient_numeric_bytes=427008,
        inbox_feature_target_numeric_bytes=9600,
    )
    for key, value in expected.items():
        assert report["work"][key] == value, key
    assert "input_normalization" not in report
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("axis", [None, 3])
def test_should_route_explicit_cli_projection_before_recipe_and_keep_binary_independent(
    monkeypatch, capsys, axis
):
    originals, reads, calls = [episode(), episode(19)], [], []

    def read(path, seed, **options):
        reads.append((str(path), seed, options))
        return originals[len(reads) - 1]

    def run(warm, new, **options):
        calls.append((warm, new, options))
        assert warm.spec.observation_dim == new.spec.observation_dim == 21
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
        "--public-vector-only",
    ]
    if axis is not None:
        arguments += ["--binary-gripper-axis", str(axis)]
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    assert reads == [
        ("A.json", 17, dict(visual_observation=True)),
        ("B.json", 19, dict(visual_observation=True)),
    ]
    assert len(calls) == 1 and calls[0][2] == (
        {} if axis is None else dict(binary_gripper_axis=axis)
    )
    for original, projected in zip(originals, calls[0][:2], strict=True):
        assert original.episode_id == projected.episode_id
        assert projected.transitions[0].action == original.transitions[0].action
        assert projected.transitions[0].observation == original.transitions[0].observation[:21]
    report = json.loads(capsys.readouterr().out)
    assert report["input_observation"]["mode"] == "public_vector21"
    assert report["input_projection"]["observation_projections"] == 8
    assert report["input_projection"]["projection"]["feature_dim"] == 21
    assert report["input_projection"]["source_values_validated"] == 6312


@pytest.mark.parametrize(
    "scenario,visual,range_flag",
    [
        (None, False, False),
        ("stream", True, False),
        ("continuous-nearest", True, False),
        ("promotion-rejection", True, False),
        ("coupled-neutral", True, False),
        ("matched-baselines", True, False),
        ("continuous-mlp-learnability", False, False),
        ("continuous-mlp-learnability", True, True),
    ],
)
def test_should_reject_invalid_projection_cli_combinations_before_io_or_fit(
    monkeypatch, scenario, visual, range_flag
):
    arguments = [
        "demo",
        "--public-vector-only",
        "--warm-teacher",
        "A.json",
        "--new-teacher",
        "B.json",
    ]
    if scenario is not None:
        arguments += ["--scenario", scenario]
    if visual:
        arguments.append("--visual-observation")
    if range_flag:
        arguments.append("--rgb-range-normalization")
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        cli, "_read_teacher", lambda *_a, **_k: pytest.fail("invalid projection mode read source")
    )
    for name in (
        "run_demo",
        "run_coupled_demo",
        "run_matched_demo",
        "run_rejection_demo",
        "run_continuous_demo",
        "run_continuous_learnability_demo",
    ):
        monkeypatch.setattr(
            cli, name, lambda *_a, **_k: pytest.fail("invalid projection mode executed recipe")
        )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("visual", [False, True])
def test_should_preserve_default_recipe_calls_and_report_schema_without_projection(
    monkeypatch, capsys, visual
):
    sources, reads, calls = [episode(), episode(19)], [], []

    def read(path, seed, **options):
        reads.append((seed, options))
        return sources[len(reads) - 1]

    def run(*args, **options):
        calls.append((args, options))
        return dict(complete=True)

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(
        cli, "_project_public_inputs", lambda *_: pytest.fail("default projected inputs")
    )
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
    arguments = [
        "demo",
        "--scenario",
        "continuous-mlp-learnability",
        "--warm-teacher",
        "A.json",
        "--new-teacher",
        "B.json",
    ]
    if visual:
        arguments.append("--visual-observation")
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    assert calls == [(tuple(sources), {})]
    assert reads == [
        (17, dict(visual_observation=True) if visual else {}),
        (19, dict(visual_observation=True) if visual else {}),
    ]
    report = json.loads(capsys.readouterr().out)
    assert "input_projection" not in report
    assert ("input_observation" in report) is visual
    if visual:
        assert report["input_observation"]["mode"] == "public_rgb"


def test_should_reject_corrupt_terminal_rgb_through_cli_before_recipe(monkeypatch):
    warm, new = episode(), episode(19)
    last = new.transitions[-1]
    new = replace(
        new,
        transitions=new.transitions[:-1]
        + (replace(last, next_observation=last.next_observation[:-1] + (1.01,)),),
    )
    monkeypatch.setattr(
        cli, "_read_teacher", lambda path, seed, **options: warm if seed == 17 else new
    )
    monkeypatch.setattr(
        cli,
        "run_continuous_learnability_demo",
        lambda *_a, **_k: pytest.fail("invalid dropped terminal reached recipe"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--scenario",
            "continuous-mlp-learnability",
            "--warm-teacher",
            "A.json",
            "--new-teacher",
            "B.json",
            "--visual-observation",
            "--public-vector-only",
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def synthetic_rgb_baseline(result):
    """Metadata/comparison double only: no synthetic RGB training is performed."""
    baseline = deepcopy(result.report)
    policy = baseline["policy"]
    policy["spec"]["observation_dim"] = 789
    stats = policy["observation_standardization"]
    stats["means"] = tuple(stats["means"]) + (0.25,) * 768
    stats["scales"] = tuple(stats["scales"]) + (1.0,) * 768
    # Different layout/fit/cost is allowed; shared exposure protocol stays exact.
    baseline["work"]["fit_feature_values"] = 2 * 789
    baseline["work"]["dense_multiply_accumulates"] += 1000
    return baseline


def test_should_compare_same_exposures_while_allowing_different_layout_stats_and_compute(completed):
    result, _, warm, new, _ = completed
    baseline = synthetic_rgb_baseline(result)
    original = deepcopy(baseline)
    case.verify_pair(result.report, baseline, warm, new)
    assert baseline == original
    assert result.report["policy"]["spec"]["observation_dim"] == 21
    assert baseline["policy"]["spec"]["observation_dim"] == 789


@pytest.mark.parametrize(
    "damage",
    [
        "receipt_clock",
        "receipt_bool",
        "endpoint_bool",
        "endpoint_float_rows",
        "source",
        "role",
        "units",
        "work_rows",
        "budget",
        "expected_fit",
    ],
)
def test_should_refuse_changed_comparative_protocol_without_running_models(completed, damage):
    result, _, warm, new, _ = completed
    current, baseline = deepcopy(result.report), synthetic_rgb_baseline(result)
    if damage == "receipt_clock":
        current["applied_updates"][0]["arrived_at"] += 1
    elif damage == "receipt_bool":
        current["applied_updates"][0]["diagnostic"]["value"] = True
    elif damage == "endpoint_bool":
        current["checkpoints"][0]["updates"] = False
    elif damage == "endpoint_float_rows":
        current["checkpoints"][1]["training_examples"] = 16.0
    elif damage == "source":
        current["source_episodes"]["new"]["episode_id"] = "selected-new-task"
    elif damage == "role":
        current["policy"]["training_role"] = "probe_train"
    elif damage == "units":
        current["policy"]["spec"]["action_units"] = ("m",) * 4
    elif damage == "work_rows":
        current["work"]["training_rows"] += 1
    elif damage == "budget":
        current["progress"]["updates_completed"] = 23
    else:
        current["expected_standardization_checked"] = True
    with pytest.raises((AssertionError, ValueError)):
        case.verify_pair(current, baseline, warm, new)


def test_should_keep_all_four_endpoints_and_signed_axis_metric_differences(completed):
    result, _, _, _, _ = completed
    baseline = synthetic_rgb_baseline(result)
    metric_names = set(baseline["metrics"])
    for name in metric_names:
        baseline["metrics"][name] += 0.25
    for endpoint in baseline["checkpoints"]:
        for context in ("warm", "new"):
            overall = endpoint["contexts"][context]["residual"]["residuals"]["overall"]
            overall["normalized_half_mse"] += 0.5
            for axis in overall["axes"]:
                axis["normalized_half_mse"] += 0.125
    compared = case.comparison(result.report, baseline)
    assert compared["primary_endpoint"] == 24
    assert compared["primary"] == pytest.approx(dict(warm=-0.5, new=-0.5))
    assert [row["updates"] for row in compared["endpoints"]] == [0, 8, 16, 24]
    assert compared["signed_metric_differences"] == {
        name: pytest.approx(-0.25) for name in metric_names
    }
    for endpoint in compared["endpoints"]:
        for context in ("warm", "new"):
            assert endpoint["contexts"][context]["public21_minus_rgb"] == pytest.approx(-0.5)
            assert len(endpoint["contexts"][context]["axes"]) == 4
            assert all(
                axis["public21_minus_rgb"]["normalized_half_mse"] == pytest.approx(-0.125)
                for axis in endpoint["contexts"][context]["axes"]
            )


def test_should_save_all_eight_final_parameter_momentum_arrays_and_owned_counters(
    completed, monkeypatch, tmp_path
):
    result, _, _, _, _ = completed
    monkeypatch.setattr(case, "stage", tmp_path)
    before = result.candidate.snapshot_state()
    report = dict(sequence=deepcopy(result.report))
    case.save_state(result.candidate, report)
    saved = report["state_archive"]
    expected = [f"{group}_{index}" for group in ("parameters", "velocity") for index in range(4)]
    assert saved["array_names"] == expected
    assert saved["updates"] == 24 and saved["training_examples"] == 48
    assert saved["save_attempts"] == saved["save_returns"] == 1
    assert saved["numeric_bytes"] == 26688 and saved["file_bytes"] <= 128 * 1024
    with np.load(tmp_path / "final-state.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(expected)
        for group in ("parameters", "velocity"):
            for index, value in enumerate(getattr(before, group)):
                assert archive[f"{group}_{index}"].dtype == np.float64
                np.testing.assert_array_equal(archive[f"{group}_{index}"], value)
    same_state(before, result.candidate.snapshot_state())
    original = (tmp_path / "final-state.npz").read_bytes()
    with pytest.raises(FileExistsError):
        case.save_state(result.candidate, dict(sequence=deepcopy(result.report)))
    assert (tmp_path / "final-state.npz").read_bytes() == original


def test_should_refuse_archive_policy_mismatch_before_writing_any_arrays(
    completed, monkeypatch, tmp_path
):
    result, _, _, _, _ = completed
    monkeypatch.setattr(case, "stage", tmp_path)
    report = dict(sequence=deepcopy(result.report))
    report["sequence"]["policy"]["momentum"] = 0.8
    with pytest.raises(AssertionError):
        case.save_state(result.candidate, report)
    assert not (tmp_path / "final-state.npz").exists()


def test_should_compose_existing_result_and_describe_new_fit_difference_without_extra_sgd(
    completed, monkeypatch, tmp_path
):
    result, _, warm, new, projection = completed
    baseline = synthetic_rgb_baseline(result)
    original = deepcopy(baseline)
    stats = baseline["policy"]["observation_standardization"]
    stats["means"] = (stats["means"][0] + 1e-7,) + tuple(stats["means"][1:])
    calls = []
    monkeypatch.setattr(case, "stage", tmp_path)

    def load(report):
        report["input_projection"] = projection
        report["baseline_new_optimizer_updates"] = 0
        return warm, new, baseline

    def sequence(*args, **options):
        calls.append((args, options))
        return result

    monkeypatch.setattr(case, "load_inputs", load)
    monkeypatch.setattr(case, "run_continuous_learnability", sequence)
    report = dict(complete=False)
    case.run_case(report)
    assert report["complete"] is True and calls == [((warm, new), {})]
    assert report["sequence"] is result.report and report["baseline"] is baseline
    assert report["normalization_prefix_difference"]["means"] == pytest.approx(1e-7)
    assert report["normalization_prefix_difference"]["scales"] == 0.0
    assert report["baseline_new_optimizer_updates"] == 0
    assert report["state_archive"]["save_returns"] == 1
    original["policy"]["observation_standardization"]["means"] = stats["means"]
    assert baseline == original


@pytest.mark.parametrize("failure", ["incomplete", "archive_write"])
def test_should_preserve_returned_paid_sequence_before_later_stage_failure(
    completed, monkeypatch, tmp_path, failure
):
    result, _, warm, new, _ = completed
    baseline = synthetic_rgb_baseline(result)
    payload = deepcopy(result.report)
    if failure == "incomplete":
        payload.update(complete=False, error="synthetic update return failure")
        returned = demo.ContinuousLearnabilityResult(payload, None)
    else:
        returned = demo.ContinuousLearnabilityResult(payload, result.candidate)
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(case, "load_inputs", lambda report: (warm, new, baseline))
    monkeypatch.setattr(case, "run_continuous_learnability", lambda *_a, **_k: returned)
    if failure == "archive_write":
        monkeypatch.setattr(
            case,
            "save_state",
            lambda *_: (_ for _ in ()).throw(OSError("synthetic archive write failure")),
        )
    assert case.main() == 1
    saved = json.loads((tmp_path / "case.json").read_text(encoding="utf8"))
    assert saved["complete"] is False and saved["sequence"]["progress"]["updates_completed"] == 24
    assert len(saved["sequence"]["applied_updates"]) == 24
    assert not (tmp_path / "final-state.npz").exists()


@pytest.mark.parametrize("name", ["case.json", "final-state.npz"])
def test_should_keep_existing_stage_outputs_without_loading_or_running_recipe(
    monkeypatch, tmp_path, name
):
    monkeypatch.setattr(case, "stage", tmp_path)
    path = tmp_path / name
    path.write_bytes(b"preserved prior attempt")
    monkeypatch.setattr(
        case, "load_inputs", lambda *_: pytest.fail("existing evidence loaded source")
    )
    monkeypatch.setattr(
        case,
        "run_continuous_learnability",
        lambda *_a, **_k: pytest.fail("existing evidence trained"),
    )
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"preserved prior attempt"


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "version",
        "nonfinite_prior",
        "work_type",
        "receipt_clock",
        "policy_role",
        "source_identity",
    ],
)
def test_should_refuse_bad_accepted_baseline_before_projection_or_new_recipe(
    completed, monkeypatch, tmp_path, damage
):
    """Synthetic29/30-row metadata gate only; no RGB or public21 training here."""
    result, _, _, _, _ = completed
    baseline = synthetic_rgb_baseline(result)
    warm, new = episode(rows=29), episode(19, rows=30)
    baseline["policy"]["observation_standardization"] = dict(
        means=(14.0,) + (0.0,) * 20 + (0.25,) * 768,
        scales=(math.sqrt(70.0),) + (1.0,) * 788,
        fitted_rows=29,
    )
    for row, updates, rows in zip(
        baseline["checkpoints"], (0, 8, 16, 24), (0, 232, 472, 704), strict=True
    ):
        row["updates"], row["training_examples"] = updates, rows
    baseline["work"].update(
        optimizer_updates=24,
        training_rows=704,
        unique_source_rows=59,
        repeated_training_rows=645,
        replay_budget_rows=0,
        conditioning_calls=8,
        conditioning_rows=236,
        residual_calls=8,
        residual_rows=236,
        actor_prediction_calls=8,
        actor_prediction_rows=236,
    )
    for index, event in enumerate(baseline["events"]):
        event["candidate_training_examples"] = (
            29 * min(index + 1, 8) + 30 * min(max(index - 7, 0), 8) + 29 * max(index - 15, 0)
        )
    accepted = dict(complete=True, python=sys.version, numpy=np.__version__, sequence=baseline)
    if damage == "version":
        accepted["numpy"] = "different-frozen-runtime"
    elif damage == "nonfinite_prior":
        baseline["metrics"]["warm_retention_loss"] = float("inf")
    elif damage == "work_type":
        baseline["work"]["training_rows"] = 704.0
    elif damage == "receipt_clock":
        baseline["applied_updates"][0]["arrived_at"] += 1
    elif damage == "policy_role":
        baseline["policy"]["training_role"] = "probe_train"
    elif damage == "source_identity":
        baseline["source_episodes"]["new"]["episode_id"] = "selected-new-identity"
    path = tmp_path / "synthetic-baseline.json"
    path.write_text(json.dumps(accepted), encoding="utf8")
    monkeypatch.setattr(case, "BASELINE_PATH", path)
    monkeypatch.setattr(case, "WARM_PATH", tmp_path / "synthetic-A.json")
    monkeypatch.setattr(case, "NEW_PATH", tmp_path / "synthetic-B.json")
    monkeypatch.setattr(
        case, "_read_teacher", lambda path, seed, **options: warm if seed == 17 else new
    )
    monkeypatch.setattr(
        case, "run_continuous_learnability", lambda *_a, **_k: pytest.fail("bad baseline trained")
    )
    if damage is None:
        report = {}
        projected_a, projected_b, accepted_baseline = case.load_inputs(report)
        assert projected_a.spec.observation_dim == projected_b.spec.observation_dim == 21
        assert len(projected_a.transitions) == 29 and len(projected_b.transitions) == 30
        assert case.exact_json(accepted_baseline, baseline)
        assert report["baseline_new_optimizer_updates"] == 0
        assert report["input_projection"]["observation_projections"] == 118
        assert report["input_projection"]["source_values_validated"] == 93102
    else:
        monkeypatch.setattr(
            case,
            "_project_public_inputs",
            lambda *_: pytest.fail("bad baseline reached projection"),
        )
        with pytest.raises((AssertionError, ValueError)):
            case.load_inputs({})
