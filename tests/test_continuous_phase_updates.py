"""Bounded phase counts:64 SGD/149 training rows per full invocation.

The imported omitted-default fixture uses24 SGD; shared explicit8/p2/mixedp1
runs use33, and paid-failure/final-empty cases use1+6. No direct reference,
real teacher, native collection or additional model updates occur here.
"""

from copy import deepcopy
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
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from test_continuous_learnability_binary import episode as binary_episode
from test_continuous_learnability_demo import completed as completed
from test_continuous_learnability_demo import episode, same_state


def forbidden(*_args, **_kwargs):
    pytest.fail("invalid settings admitted downstream work")


@pytest.fixture(scope="module")
def runs(completed):
    """Exactly33 new SGD; the requested old fixture adds24, shared once."""
    current = [None]
    observed = {}
    original_fit, original_train = (
        demo.fit_training_standardization,
        NumpyContinuousBCLearner.train_batch,
    )
    original_runtime, original_checkpoint = demo.ActorShadowRuntime, demo._checkpoint
    with pytest.MonkeyPatch.context() as patch:

        def fit(record, **options):
            observed[current[0]]["fits"].append((record, options))
            return original_fit(record, **options)

        def train(owner, features, targets):
            observed[current[0]]["trains"].append((owner, len(features)))
            return original_train(owner, features, targets)

        def runtime(source, **options):
            owner = original_runtime(source, **options)
            capture = observed[current[0]]
            capture.update(
                source=source, runtime=owner, options=options, initial=source.snapshot_state()
            )
            ready = owner.train_ready

            def poll(*args, **kwargs):
                before, paid_before = (
                    owner.candidate_snapshot().state,
                    options["budget"].updates_completed,
                )
                result = ready(*args, **kwargs)
                capture["polls"].append(
                    (
                        options["clock"].now(),
                        paid_before,
                        options["budget"].updates_completed,
                        result,
                        before,
                        owner.candidate_snapshot().state,
                    )
                )
                return result

            patch.setattr(owner, "train_ready", poll)
            return owner

        def checkpoint(*args, **kwargs):
            observed[current[0]]["states"].append(args[2].candidate_snapshot().state)
            return original_checkpoint(*args, **kwargs)

        patch.setattr(demo, "fit_training_standardization", fit)
        patch.setattr(NumpyContinuousBCLearner, "train_batch", train)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(demo, "_checkpoint", checkpoint)
        for name, count, binary in (("default", 8, False), ("short", 2, False), ("mixed", 1, True)):
            current[0] = name
            warm, new = (
                (binary_episode(), binary_episode("B")) if binary else (episode(), episode("B"))
            )
            capture = dict(fits=[], trains=[], states=[], polls=[], warm=warm, new=new)
            observed[name] = capture
            original_inputs = asdict(warm), asdict(new)
            options = dict(
                phase_updates=count,
                expected_standardization=ObservationStandardization((1.0, 0.0), (1.0, 1.0), 2),
            )
            if binary:
                options["binary_gripper_axis"] = 1
            capture["result"] = demo.run_continuous_learnability(warm, new, **options)
            assert capture["result"].report["complete"] is True
            assert (asdict(warm), asdict(new)) == original_inputs
    return observed, completed


def test_should_preserve_omitted_and_explicit_eight_reports_and_all_eight_arrays(runs):
    observed, (omitted, original, _, _) = runs
    explicit = observed["default"]
    first, second = deepcopy(omitted), deepcopy(explicit["result"].report)
    first.pop("elapsed_seconds")
    second.pop("elapsed_seconds")
    assert first == second
    same_state(original["endpoints"][0], explicit["initial"])
    same_state(original["endpoints"][24], explicit["result"].candidate.snapshot_state())
    assert explicit["options"]["max_experiences"] == 24
    assert explicit["options"]["budget"].budget.max_training_updates == 24


@pytest.mark.parametrize(
    "name,count,expected_rows", [("short", 2, [0, 4, 10, 14]), ("mixed", 1, [0, 2, 5, 7])]
)
def test_should_derive_runtime_capacity_ticks_receipts_and_full_state_from_phase_count(
    runs, name, count, expected_rows
):
    observed, _ = runs
    data = observed[name]
    report, source, runtime = data["result"].report, data["source"], data["runtime"]
    assert report["settings"]["phase_updates"] == count
    assert report["settings"]["max_training_updates"] == 3 * count
    assert report["settings"]["phases"] == ["warm", "new", "warm_return"]
    assert data["options"]["max_experiences"] == 3 * count
    assert data["options"]["max_consolidations"] == 0
    assert data["options"]["budget"].budget.max_training_updates == 3 * count
    assert report["progress"] == dict(
        update_attempts=3 * count,
        updates_completed=3 * count,
        replay_examples_completed=0,
        next_tick=6 * count,
    )
    assert data["options"]["budget"].replay_examples_completed == 0
    assert len(data["fits"]) == 1 and data["fits"][0] == (data["warm"], {})
    assert [rows for _, rows in data["trains"]] == [2] * count + [3] * count + [2] * count
    assert all(owner is runtime._candidate for owner, _ in data["trains"])
    assert [point["updates"] for point in report["checkpoints"]] == [0, count, 2 * count, 3 * count]
    assert [point["training_examples"] for point in report["checkpoints"]] == expected_rows
    assert [state.training_examples for state in data["states"]] == expected_rows
    same_state(data["initial"], source.snapshot_state())
    same_state(data["initial"], runtime.actor.snapshot_state().state)
    same_state(data["states"][-1], data["result"].candidate.snapshot_state())
    same_state(data["states"][-1], runtime.candidate_snapshot().state)
    assert (
        data["result"].candidate is not source
        and data["result"].candidate is not runtime._candidate
    )
    assert len(data["polls"]) == 6 * count + 1
    objective = core.MIXED_OBJECTIVE if name == "mixed" else core.OBJECTIVE
    rows = 0
    for index, (event, receipt) in enumerate(
        zip(report["events"], report["applied_updates"], strict=True)
    ):
        selected = data["new"] if count <= index < 2 * count else data["warm"]
        phase = "warm" if index < count else "new" if index < 2 * count else "warm_return"
        rows += len(selected.transitions)
        assert event["index"] == index and event["phase"] == phase
        assert event["context"] == ("new" if selected is data["new"] else "warm")
        assert (
            event["deferred_exact"]
            is event["update_attempted"]
            is event["success_validated"]
            is True
        )
        assert event["applied"] == [receipt]
        assert (
            event["candidate_updates"] == index + 1 and event["candidate_training_examples"] == rows
        )
        assert (
            receipt["sample_id"] == f"mlp-batch-{index}"
            and receipt["event_id"] == f"mlp-label-{index}"
        )
        assert (
            receipt["episode_id"] == selected.episode_id and receipt["update_number"] == index + 1
        )
        assert receipt["observed_at"] == 2 * index
        assert receipt["arrived_at"] == receipt["applied_at"] == 2 * index + 1
        assert receipt["actor_version"] == "actor-0" and receipt["learner_version"] == "candidate-0"
        assert receipt["diagnostic"]["definition"] == objective and math.isfinite(
            receipt["diagnostic"]["value"]
        )
        deferred, applied = data["polls"][2 * index : 2 * index + 2]
        assert deferred[:4] == (2 * index, index, index, ())
        same_state(deferred[4], deferred[5])
        assert applied[:3] == (2 * index + 1, index, index + 1) and len(applied[3]) == 1
    final = data["polls"][-1]
    assert final[:4] == (6 * count - 1, 3 * count, 3 * count, ())
    same_state(final[4], final[5])


@pytest.mark.parametrize("name", ["short", "mixed"])
def test_should_keep_all_four_readouts_actor_predictions_and_signed_metrics(runs, name):
    data = runs[0][name]
    report = data["result"].report
    objective = core.MIXED_OBJECTIVE if name == "mixed" else core.OBJECTIVE
    losses = []
    for point in report["checkpoints"]:
        assert point["state_exact"] is point["owners_exact"] is True
        endpoint = {}
        for context in ("warm", "new"):
            record, result = data[context], point["contexts"][context]
            rows = len(record.transitions)
            assert result["episode_id"] == record.episode_id
            assert (
                result["actor_predictions"]
                == report["checkpoints"][0]["contexts"][context]["actor_predictions"]
            )
            assert result["conditioning"]["standardization_rows"] == rows
            assert result["conditioning"]["standardization_values"] == rows * 2
            assert result["conditioning"]["conditioning"]["objective_definition"] == objective
            residual = result["residual"]
            assert residual["prediction_calls"] == 1 and residual["prediction_rows"] == rows
            assert (
                residual["prediction_values"] == 2 * rows
                and residual["prediction_bytes"] == 16 * rows
            )
            overall = residual["residuals"]["overall"]
            assert overall["rows"] == rows and overall["active_axes"] == 2
            assert [axis["index"] for axis in overall["axes"]] == [0, 1]
            (stage,) = residual["residuals"]["stages"]
            assert stage["label"] == "all" and stage["summary"] == overall
            assert tuple(stage["row_indices"]) == tuple(range(rows))
            if name == "short":
                assert result["conditioning"]["conditioning"]["objective_loss"] == pytest.approx(
                    overall["normalized_half_mse"], abs=1e-14
                )
            endpoint[context] = overall["normalized_half_mse"]
        losses.append(endpoint)
    assert report["metrics"] == dict(
        warm_acquisition_gain=losses[0]["warm"] - losses[1]["warm"],
        new_acquisition_gain=losses[1]["new"] - losses[2]["new"],
        warm_retention_loss=losses[2]["warm"] - losses[1]["warm"],
        warm_return_recovery=losses[2]["warm"] - losses[3]["warm"],
        warm_return_deficit=losses[3]["warm"] - losses[1]["warm"],
        new_retention_loss=losses[3]["new"] - losses[2]["new"],
    )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "name,updates,rows,normcalls,forwards,backwards,gradient,dense",
    [("short", 6, 14, 30, 38, 14, 4508, 37120), ("mixed", 3, 7, 27, 35, 11, 3542, 32640)],
)
def test_should_count_training_exposure_diagnostics_normalization_and_dense_work(
    runs, name, updates, rows, normcalls, forwards, backwards, gradient, dense
):
    work = runs[0][name]["result"].report["work"]
    assert work["optimizer_updates"] == updates and work["training_rows"] == rows
    assert work["fit_calls"] == 1 and work["fit_rows"] == 2 and work["fit_feature_values"] == 4
    assert work["unique_source_rows"] == 5 and work["repeated_training_rows"] == rows - 5
    assert work["replay_budget_rows"] == 0
    for kind in ("conditioning", "residual", "actor_prediction"):
        assert work[kind + "_calls"] == 8 and work[kind + "_rows"] == 20
    assert work["normalization_calls"] == normcalls
    assert work["normalization_rows"] == rows + 60 and work["normalization_values"] == 2 * (
        rows + 60
    )
    assert work["forward_passes"] == forwards and work["forward_rows"] == rows + 80
    assert work["backward_passes"] == backwards and work["backward_rows"] == rows + 20
    assert work["gradient_values"] == gradient and work["gradient_numeric_bytes"] == 8 * gradient
    assert work["dense_multiply_accumulates"] == dense
    assert (
        work["parameter_values_per_owner"] == 322 and work["model_numeric_bytes_per_owner"] == 5152
    )
    assert work["statistics_numeric_bytes_per_owner"] == 32 and work["named_model_owners"] == 4
    assert work["inbox_feature_target_numeric_bytes"] == rows * 32
    assert work["residual_prediction_bytes"] == work["actor_prediction_bytes"] == 320
    if name == "mixed":
        assert work["binary_gripper_loss_rows"] == 27
    else:
        assert "binary_gripper_loss_rows" not in work


@pytest.mark.parametrize("value", [1, 8, 128])
def test_should_accept_exact_native_phase_counts_without_any_model_work(value):
    assert demo.validate_phase_updates(value) == value
    assert type(demo.validate_phase_updates(value)) is int


class IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.int64(2),
        np.int32(8),
        np.bool_(True),
        2.0,
        "2",
        None,
        0,
        -1,
        129,
        IntSubclass(2),
    ],
)
@pytest.mark.parametrize("api", ["validator", "result", "json"])
def test_should_refuse_invalid_counts_before_episode_admission_fitting_or_model(
    monkeypatch, value, api
):
    monkeypatch.setattr(demo, "_training_episodes", forbidden)
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    with pytest.raises(ValueError, match="phase_updates"):
        if api == "validator":
            demo.validate_phase_updates(value)
        else:
            function = (
                demo.run_continuous_learnability
                if api == "result"
                else demo.run_continuous_learnability_demo
            )
            function(object(), object(), phase_updates=value)


def test_should_allocate_bounded_upper_count_and_stop_without_sgd_on_failed_first_checkpoint(
    monkeypatch,
):
    captured = []
    original = demo.ActorShadowRuntime

    def runtime(source, **options):
        owner = original(source, **options)
        captured.append((source, owner, options))
        return owner

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(
        demo,
        "_checkpoint",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("initial readout failed")),
    )
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    result = demo.run_continuous_learnability(episode(), episode("B"), phase_updates=128)
    source, runtime, options = captured[0]
    assert result.candidate is None and result.report["complete"] is False
    assert "initial readout failed" in result.report["error"]
    assert options["budget"].budget.max_training_updates == options["max_experiences"] == 384
    assert options["budget"].updates_completed == 0
    assert result.report["settings"]["phase_updates"] == 128
    assert result.report["applied_updates"] == [] and result.report["events"] == []
    same_state(source.snapshot_state(), runtime.actor.snapshot_state().state)


def test_should_retain_one_permanent_charge_and_original_receipt_when_update_return_fails(
    monkeypatch,
):
    """ExactlyONE SGD/two rows; preceding-success failure admits no second packet."""
    captured = []
    original = demo.ActorShadowRuntime

    def runtime(source, **options):
        owner = original(source, **options)
        initial = source.snapshot_state()
        captured.append((source, owner, options, initial))
        ready = owner.train_ready

        def fail_after_commit(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if applied:
                raise RuntimeError("postcommit return failed")
            return applied

        monkeypatch.setattr(owner, "train_ready", fail_after_commit)
        return owner

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_learnability(episode(), episode("B"), phase_updates=2)
    source, runtime, options, initial = captured[0]
    report = result.report
    assert result.candidate is None and report["complete"] is False
    assert "postcommit return failed" in report["error"]
    assert report["settings"]["max_training_updates"] == 6
    assert report["progress"] == dict(
        update_attempts=1, updates_completed=1, replay_examples_completed=0, next_tick=2
    )
    assert options["budget"].updates_completed == 1 and len(report["events"]) == 1
    assert (
        report["events"][0]["applied"] == [] and report["events"][0]["success_validated"] is False
    )
    (receipt,) = report["applied_updates"]
    assert receipt["observed_at"] == 0 and receipt["arrived_at"] == receipt["applied_at"] == 1
    assert receipt["diagnostic"]["definition"] == core.OBJECTIVE and math.isfinite(
        receipt["diagnostic"]["value"]
    )
    assert (
        len(runtime.applied_updates) == 1
        and runtime.candidate_snapshot().state.training_examples == 2
    )
    same_state(initial, source.snapshot_state())
    same_state(initial, runtime.actor.snapshot_state().state)
    assert "work" not in report and "metrics" not in report


def test_should_revalidate_final_empty_poll_at_the_nondefault_budget(monkeypatch):
    """ExactlySIX SGD/14rows; a paid sequence cannot hide late state corruption."""
    captured = []
    original = demo.ActorShadowRuntime

    def runtime(source, **options):
        owner = original(source, **options)
        captured.append((source, owner, options, source.snapshot_state()))
        ready = owner.train_ready

        def change_empty(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if not applied and options["budget"].updates_completed == 6:
                state = owner._candidate.snapshot_state()
                state.velocity[0][0, 0] += 0.25
                owner._candidate.restore_state(state)
            return applied

        monkeypatch.setattr(owner, "train_ready", change_empty)
        return owner

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_learnability(episode(), episode("B"), phase_updates=2)
    source, runtime, options, initial = captured[0]
    assert result.candidate is None and result.report["complete"] is False
    assert "final empty poll" in result.report["error"]
    assert result.report["progress"] == dict(
        update_attempts=6, updates_completed=6, replay_examples_completed=0, next_tick=12
    )
    assert len(result.report["applied_updates"]) == len(result.report["events"]) == 6
    assert all(event["success_validated"] is True for event in result.report["events"])
    assert len(result.report["checkpoints"]) == 4
    assert (
        options["budget"].updates_completed == 6
        and runtime.candidate_snapshot().state.training_examples == 14
    )
    same_state(initial, source.snapshot_state())
    same_state(initial, runtime.actor.snapshot_state().state)
    assert "work" not in result.report and "metrics" not in result.report


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
@pytest.mark.parametrize("position", ["warm", "new"])
def test_should_keep_nondefault_training_role_gates_before_fitting(monkeypatch, role, position):
    warm = episode(role=role) if position == "warm" else episode()
    new = episode("B", role=role) if position == "new" else episode("B")
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    with pytest.raises(ValueError, match="TRAIN"):
        demo.run_continuous_learnability(warm, new, phase_updates=2)


def test_should_refuse_nonendpoint_binary_labels_before_fit_with_nondefault_count(monkeypatch):
    warm, new = binary_episode(), binary_episode("B")
    invalid = replace(new.transitions[0], action=(new.transitions[0].action[0], 0.5))
    new = replace(new, transitions=(invalid,) + new.transitions[1:])
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(warm, new, phase_updates=1, binary_gripper_axis=1)


@pytest.mark.parametrize("count", [None, 8, 2])
@pytest.mark.parametrize("composition", ["plain", "binary", "rgb", "both"])
def test_should_preserve_default_wrapper_call_shape_and_forward_nondefault_compositions(
    monkeypatch, count, composition
):
    calls, report = [], dict(complete=True)

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(report=report)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    warm, new = object(), object()
    options, expected = {}, dict(expected_standardization=None)
    if count is not None:
        options["phase_updates"] = count
    if count == 2:
        expected["phase_updates"] = 2
    if composition in ("rgb", "both"):
        options["rgb_range_recipe"] = pick_cube_rgb_recipe()
        expected.update(rgb_range_recipe=options["rgb_range_recipe"], binary_gripper_axis=None)
    if composition in ("binary", "both"):
        options["binary_gripper_axis"] = 1
        expected["binary_gripper_axis"] = 1
    assert demo.run_continuous_learnability_demo(warm, new, **options) is report
    assert calls == [((warm, new), expected)]


@pytest.mark.parametrize(
    "count,composition",
    [
        (None, "plain"),
        (8, "plain"),
        (2, "plain"),
        (128, "plain"),
        (2, "binary"),
        (2, "public"),
        (2, "public_binary"),
        (2, "rgb_binary"),
    ],
)
def test_should_route_explicit_phase_counts_and_existing_options_without_model_work(
    monkeypatch, capsys, count, composition
):
    records, projected = (object(), object()), (object(), object())
    calls, reads, projections = [], [], []
    visual = composition in ("public", "public_binary", "rgb_binary")
    public = composition in ("public", "public_binary")
    binary = composition in ("binary", "public_binary", "rgb_binary")

    def read(path, seed, **options):
        reads.append((str(path), seed, options))
        return records[len(reads) - 1]

    def project(warm, new):
        projections.append((warm, new))
        return *projected, dict(fixture_projection=True)

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(complete=True)

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "_project_public_inputs", project)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", execute)
    arguments = [
        "stream",
        "--scenario",
        "continuous-mlp-learnability",
        "--warm-teacher",
        "A.json",
        "--new-teacher",
        "B.json",
    ]
    expected = {}
    if count is not None:
        arguments.extend(["--phase-updates", str(count)])
        expected["phase_updates"] = count
    if binary:
        arguments.extend(["--binary-gripper-axis", "1"])
        expected["binary_gripper_axis"] = 1
    if visual:
        arguments.append("--visual-observation")
    if public:
        arguments.append("--public-vector-only")
    if composition == "rgb_binary":
        arguments.append("--rgb-range-normalization")
        expected["rgb_range_recipe"] = pick_cube_rgb_recipe()
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    assert reads == [
        ("A.json", 17, {"visual_observation": True} if visual else {}),
        ("B.json", 19, {"visual_observation": True} if visual else {}),
    ]
    assert calls == [(projected if public else records, expected)]
    assert projections == ([records] if public else [])
    result = json.loads(capsys.readouterr().out)
    if public:
        assert result["input_observation"]["mode"] == "public_vector21"
        assert result["input_projection"] == dict(fixture_projection=True)
    elif visual:
        assert result["input_observation"]["mode"] == "public_rgb"
    else:
        assert result == dict(complete=True)


@pytest.mark.parametrize(
    "scenario,count",
    [
        ("stream", "2"),
        ("continuous-nearest", "2"),
        ("matched-baselines", "8"),
        ("continuous-mlp-learnability", "0"),
        ("continuous-mlp-learnability", "129"),
        ("continuous-mlp-learnability", "2.0"),
        ("continuous-mlp-learnability", "-1"),
    ],
)
def test_should_refuse_invalid_cli_count_or_scenario_before_teacher_or_legacy_work(
    monkeypatch, scenario, count
):
    monkeypatch.setattr(cli, "_read_teacher", forbidden)
    monkeypatch.setattr(cli, "run_demo", forbidden)
    monkeypatch.setattr(cli, "run_matched_demo", forbidden)
    monkeypatch.setattr(cli, "run_continuous_demo", forbidden)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", forbidden)
    args = ["stream", "--scenario", scenario, "--phase-updates", count]
    if scenario.startswith("continuous-"):
        args.extend(["--warm-teacher", "A.json", "--new-teacher", "B.json"])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_preserve_partial_failure_receipts_and_status_in_nondefault_cli(monkeypatch, capsys):
    records, calls = [], []

    def read(*_args, **_kwargs):
        value = object()
        records.append(value)
        return value

    partial = dict(
        complete=False,
        error="paid receipt refused",
        applied_updates=[dict(update_number=1)],
        progress=dict(updates_completed=1),
    )

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return partial

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-mlp-learnability",
            "--warm-teacher",
            "A.json",
            "--new-teacher",
            "B.json",
            "--phase-updates",
            "2",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 1
    assert calls == [(tuple(records), dict(phase_updates=2))]
    assert json.loads(capsys.readouterr().out) == partial
