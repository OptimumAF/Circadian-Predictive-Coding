"""Bounded A-only information-use control:16 SGD/34 training rows per invocation.

Four shared phase1 runs use12 SGD/26rows, the direct A path3/6, and one
postcommit failure1/2. No real data, simulator, seed selection or model tuning.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import run_native_stream_demo as cli
from src.adapters import continuous_learnability_demo as demo
from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
    continuous_training_pairs,
)
from src.core.observation_standardization import ObservationStandardization


# Why: local tiny helpers keep this bounded module independent of legacy fixtures.
def episode(name="A", *, role="train"):
    observations = (
        [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        if name == "A"
        else [(10.0, 1.0), (11.0, 1.0), (12.0, 1.0), (13.0, 1.0)]
    )
    actions = [(-2.0, -1.0), (6.0, 1.0)] if name == "A" else [(5.0, 1.0), (4.0, 0.5), (3.0, 0.0)]
    declared = ContinuousActionSpec(2, ("m", "m"), (-2.0, -1.0), (6.0, 1.0))
    return ContinuousEpisode(
        "synthetic-" + name,
        "context-" + name,
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


def same_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for group in ("parameters", "velocity"):
        for left, right in zip(getattr(first, group), getattr(second, group), strict=True):
            np.testing.assert_array_equal(left, right)


def forbidden(*_args, **_kwargs):
    pytest.fail("invalid control admitted model, fitting or input work")


@pytest.fixture(scope="module")
def runs():
    observed: dict[str, Any] = {}
    current = [""]
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
            observed[current[0]]["trains"].append((owner, features.copy(), targets.copy()))
            return original_train(owner, features, targets)

        def checkpoint(*args, **kwargs):
            observed[current[0]]["states"].append(args[2].candidate_snapshot().state)
            return original_checkpoint(*args, **kwargs)

        def runtime(source, **options):
            owner = original_runtime(source, **options)
            capture = observed[current[0]]
            capture.update(
                source=source, runtime=owner, options=options, initial=source.snapshot_state()
            )
            original_ready = owner.train_ready

            def poll(*args, **kwargs):
                before = owner.candidate_snapshot().state
                paid = options["budget"].updates_completed
                result = original_ready(*args, **kwargs)
                capture["polls"].append(
                    (options["clock"].now(), paid, options["budget"].updates_completed, result)
                )
                if not result:
                    same_state(before, owner.candidate_snapshot().state)
                return result

            patch.setattr(owner, "train_ready", poll)
            for name in ("record_experience", "record_label"):
                original = getattr(owner, name)

                def record(value, _name=name, _original=original):
                    capture["inbox"].append((_name, value))
                    return _original(value)

                patch.setattr(owner, name, record)
            return owner

        patch.setattr(demo, "fit_training_standardization", fit)
        patch.setattr(NumpyContinuousBCLearner, "train_batch", train)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(demo, "_checkpoint", checkpoint)
        for name, selected in (
            ("omitted", None),
            ("false", False),
            ("control", True),
            ("changed_b", True),
        ):
            current[0] = name
            warm, new = episode(), episode("B")
            if name == "changed_b":
                new = replace(
                    new,
                    transitions=tuple(replace(row, action=(-2.0, -1.0)) for row in new.transitions),
                )
            capture: dict[str, Any] = dict(
                warm=warm, new=new, fits=[], trains=[], states=[], polls=[], inbox=[]
            )
            observed[name] = capture
            inputs = asdict(warm), asdict(new)
            options: dict[str, Any] = dict(
                phase_updates=1,
                expected_standardization=ObservationStandardization((1.0, 0.0), (1.0, 1.0), 2),
            )
            if selected is not None:
                options["a_only_control"] = selected
            capture["result"] = demo.run_continuous_learnability(warm, new, **options)
            assert capture["result"].report["complete"] is True
            assert (asdict(warm), asdict(new)) == inputs
    return observed


def test_should_keep_explicit_false_and_omitted_reports_policies_and_all_arrays_identical(runs):
    omitted, explicit = runs["omitted"], runs["false"]
    first, second = deepcopy(omitted["result"].report), deepcopy(explicit["result"].report)
    first.pop("elapsed_seconds")
    second.pop("elapsed_seconds")
    assert first == second
    assert [len(features) for _, features, _ in omitted["trains"]] == [2, 3, 2]
    assert (
        not {"a_only_control", "training_contexts", "diagnostic_contexts"}
        & first["settings"].keys()
    )
    assert (
        not {"unique_training_rows", "training_rows_by_context", "diagnostic_rows_by_context"}
        & first["work"].keys()
    )
    for left, right in zip(omitted["states"], explicit["states"], strict=True):
        same_state(left, right)
    same_state(
        omitted["result"].candidate.snapshot_state(), explicit["result"].candidate.snapshot_state()
    )


def test_should_train_only_a_with_original_delays_identities_and_one_charge_per_packet(runs):
    data = runs["control"]
    report, runtime, options = data["result"].report, data["runtime"], data["options"]
    assert report["settings"]["a_only_control"] is True
    assert report["settings"]["training_contexts"] == ["warm"] * 3
    assert report["settings"]["diagnostic_contexts"] == ["warm", "new"]
    assert report["settings"]["phases"] == ["warm", "new", "warm_return"]
    assert options["max_experiences"] == options["budget"].budget.max_training_updates == 3
    assert options["max_consolidations"] == options["budget"].replay_examples_completed == 0
    assert data["fits"] == [(data["warm"], {})]
    features, targets = continuous_training_pairs(data["warm"])
    assert len(data["trains"]) == 3
    for owner, actual_features, actual_targets in data["trains"]:
        assert owner is runtime._candidate
        np.testing.assert_array_equal(actual_features, features)
        np.testing.assert_array_equal(actual_targets, targets)
    assert len(data["inbox"]) == 6
    for name, packet in data["inbox"]:
        assert packet.episode_id == data["warm"].episode_id
        values = packet.features if name == "record_experience" else packet.targets
        np.testing.assert_array_equal(values, features if name == "record_experience" else targets)
    assert [event["phase"] for event in report["events"]] == ["warm", "new", "warm_return"]
    assert [event["context"] for event in report["events"]] == ["warm"] * 3
    for index, (event, receipt) in enumerate(
        zip(report["events"], report["applied_updates"], strict=True)
    ):
        assert event["deferred_exact"] is event["success_validated"] is True
        assert event["applied"] == [receipt]
        assert event["candidate_updates"] == receipt["update_number"] == index + 1
        assert event["candidate_training_examples"] == 2 * (index + 1)
        assert receipt["episode_id"] == data["warm"].episode_id
        assert (
            receipt["sample_id"] == f"mlp-batch-{index}"
            and receipt["event_id"] == f"mlp-label-{index}"
        )
        assert receipt["observed_at"] == 2 * index
        assert receipt["arrived_at"] == receipt["applied_at"] == 2 * index + 1
        assert receipt["actor_version"] == "actor-0" and receipt["learner_version"] == "candidate-0"
        assert np.isfinite(receipt["diagnostic"]["value"])
    assert [row[:3] for row in data["polls"]] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 1, 1),
        (3, 1, 2),
        (4, 2, 2),
        (5, 2, 3),
        (5, 3, 3),
    ]
    assert all(data["polls"][index][3] == () for index in (0, 2, 4, 6))
    assert report["progress"] == dict(
        update_attempts=3, updates_completed=3, replay_examples_completed=0, next_tick=6
    )
    same_state(data["initial"], data["source"].snapshot_state())
    same_state(data["initial"], runtime.actor.snapshot_state().state)


def test_should_still_diagnose_both_actual_contexts_and_preserve_untrained_actor(runs):
    data = runs["control"]
    report = data["result"].report
    assert [row["updates"] for row in report["checkpoints"]] == [0, 1, 2, 3]
    assert [row["training_examples"] for row in report["checkpoints"]] == [0, 2, 4, 6]
    actor_predictions = {}
    for point in report["checkpoints"]:
        assert point["state_exact"] is point["owners_exact"] is True
        assert list(point["contexts"]) == ["warm", "new"]
        for name in ("warm", "new"):
            result, source = point["contexts"][name], data[name]
            assert result["episode_id"] == source.episode_id
            assert result["conditioning"]["standardization_rows"] == len(source.transitions)
            assert result["residual"]["prediction_rows"] == len(source.transitions)
            assert len(result["actor_predictions"]) == len(source.transitions)
            if name not in actor_predictions:
                actor_predictions[name] = result["actor_predictions"]
            assert result["actor_predictions"] == actor_predictions[name]
    assert report["source_episodes"]["new"]["episode_id"] == data["new"].episode_id
    for update_number, state in zip((0, 1, 2, 3), data["states"], strict=True):
        assert state.completed_updates == update_number
    same_state(data["states"][-1], data["result"].candidate.snapshot_state())
    json.dumps(report, allow_nan=False)


def test_should_keep_b_targets_out_of_optimizer_while_retaining_their_diagnostics(runs):
    first, changed = runs["control"], runs["changed_b"]
    assert first["result"].report["policy"] == changed["result"].report["policy"]
    for left, right in zip(first["states"], changed["states"], strict=True):
        same_state(left, right)
    same_state(
        first["result"].candidate.snapshot_state(), changed["result"].candidate.snapshot_state()
    )
    for original_point, changed_point in zip(
        first["result"].report["checkpoints"], changed["result"].report["checkpoints"], strict=True
    ):
        assert original_point["contexts"]["warm"] == changed_point["contexts"]["warm"]
        assert (
            original_point["contexts"]["new"]["actor_predictions"]
            == changed_point["contexts"]["new"]["actor_predictions"]
        )
        assert (
            original_point["contexts"]["new"]["residual"]["residuals"]["overall"][
                "normalized_half_mse"
            ]
            != changed_point["contexts"]["new"]["residual"]["residuals"]["overall"][
                "normalized_half_mse"
            ]
        )


def test_should_account_for_unequal_training_and_diagnostic_sources_independently(runs):
    report = runs["control"]["result"].report
    work = report["work"]
    expected = dict(
        fit_calls=1,
        fit_rows=2,
        fit_feature_values=4,
        optimizer_updates=3,
        training_rows=6,
        unique_source_rows=5,
        unique_training_rows=2,
        repeated_training_rows=4,
        replay_budget_rows=0,
        training_rows_by_context=dict(warm=6, new=0),
        diagnostic_rows_by_context=dict(warm=8, new=12),
        conditioning_calls=8,
        conditioning_rows=20,
        residual_calls=8,
        residual_rows=20,
        actor_prediction_calls=8,
        actor_prediction_rows=20,
        normalization_calls=27,
        normalization_rows=66,
        normalization_values=132,
        forward_passes=35,
        forward_rows=86,
        backward_passes=11,
        backward_rows=26,
        parameter_values_per_owner=322,
        gradient_values=3542,
        gradient_numeric_bytes=28336,
        dense_multiply_accumulates=32000,
        inbox_feature_target_numeric_bytes=192,
        residual_prediction_bytes=320,
        actor_prediction_bytes=320,
    )
    assert {key: work[key] for key in expected} == expected
    ordinary = runs["omitted"]["result"].report["work"]
    assert ordinary["training_rows"] == 7 and ordinary["repeated_training_rows"] == 2
    assert work["conditioning_rows"] == ordinary["conditioning_rows"]
    assert work["residual_rows"] == ordinary["residual_rows"]
    assert work["actor_prediction_rows"] == ordinary["actor_prediction_rows"]


def test_should_report_all_six_original_metrics_from_both_actual_diagnostic_contexts(runs):
    report = runs["control"]["result"].report
    errors = [
        {
            name: point["contexts"][name]["residual"]["residuals"]["overall"]["normalized_half_mse"]
            for name in ("warm", "new")
        }
        for point in report["checkpoints"]
    ]
    expected = dict(
        warm_acquisition_gain=errors[0]["warm"] - errors[1]["warm"],
        new_acquisition_gain=errors[1]["new"] - errors[2]["new"],
        warm_retention_loss=errors[2]["warm"] - errors[1]["warm"],
        warm_return_recovery=errors[2]["warm"] - errors[3]["warm"],
        warm_return_deficit=errors[3]["warm"] - errors[1]["warm"],
        new_retention_loss=errors[3]["new"] - errors[2]["new"],
    )
    assert report["metrics"] == expected and len(expected) == 6
    assert runs["control"]["result"].candidate is not runs["control"]["runtime"]._candidate
    assert runs["control"]["result"].candidate is not runs["control"]["source"]


def test_should_match_three_direct_a_updates_including_all_momentum_arrays(runs):
    data = runs["control"]
    reference = data["source"].fork()
    reference.restore_state(data["initial"])
    for index in range(3):
        diagnostic = reference.train_episode(data["warm"])
        receipt = data["result"].report["applied_updates"][index]
        assert diagnostic.definition == receipt["diagnostic"]["definition"]
        assert diagnostic.value == receipt["diagnostic"]["value"]
        same_state(reference.snapshot_state(), data["states"][index + 1])


@pytest.mark.parametrize("api", ["run_continuous_learnability", "run_continuous_learnability_demo"])
@pytest.mark.parametrize("bad", [None, 0, 1, "false", [], np.bool_(True), np.int64(1)])
def test_should_refuse_non_boolean_control_before_episode_admission(api, bad, monkeypatch):
    monkeypatch.setattr(demo, "_training_episodes", forbidden)
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    with pytest.raises(ValueError, match="a_only_control"):
        getattr(demo, api)(object(), object(), phase_updates=1, a_only_control=bad)


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
@pytest.mark.parametrize("position", ["warm", "new"])
def test_should_validate_both_roles_before_fit_even_when_b_never_trains(
    monkeypatch, role, position
):
    warm = episode(role=role) if position == "warm" else episode()
    new = episode("B", role=role) if position == "new" else episode("B")
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    with pytest.raises(ValueError, match="TRAIN"):
        demo.run_continuous_learnability(warm, new, phase_updates=1, a_only_control=True)


@pytest.mark.parametrize("damage", ["spec", "identity", "label", "observation", "missing"])
def test_should_refuse_invalid_diagnostic_b_source_before_fit_or_model(monkeypatch, damage):
    warm, new = episode(), episode("B")
    if damage == "spec":
        new = replace(
            new, spec=ContinuousActionSpec(2, ("different", "m"), (-2.0, -1.0), (6.0, 1.0))
        )
    elif damage == "identity":
        new = replace(new, episode_id=warm.episode_id)
    elif damage == "label":
        object.__setattr__(new.transitions[0], "action", (float("nan"), 0.0))
    elif damage == "observation":
        object.__setattr__(new.transitions[-1], "next_observation", (float("inf"), 1.0))
    else:
        new = None
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(episode(), new, phase_updates=1, a_only_control=True)


def test_should_validate_binary_b_labels_even_when_they_do_not_train(monkeypatch):
    warm, new = episode(), episode("B")
    spec = replace(warm.spec, action_units=("m", "normalized_gripper"))
    warm, new = replace(warm, spec=spec), replace(new, spec=spec)
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(
            warm, new, phase_updates=1, a_only_control=True, binary_gripper_axis=1
        )


def test_should_count_binary_conditioning_rows_on_both_contexts_but_only_a_training():
    warm, new = episode(), episode("B")
    new = replace(
        new, transitions=tuple(replace(row, action=(row.action[0], 1.0)) for row in new.transitions)
    )
    spec = replace(warm.spec, action_units=("m", "normalized_gripper"))
    policy = ContinuousBCPolicy(
        spec,
        observation_standardization=ObservationStandardization((1.0, 0.0), (1.0, 1.0), 2),
        binary_gripper_axis=1,
    )
    work = demo._work(
        dict(warm=replace(warm, spec=spec), new=replace(new, spec=spec)),
        policy,
        1,
        a_only_control=True,
    )
    assert work["training_rows"] == 6 and work["conditioning_rows"] == 20
    assert work["binary_gripper_loss_rows"] == 26
    assert work["training_rows_by_context"] == dict(warm=6, new=0)


def test_should_preserve_first_paid_a_receipt_and_source_when_delivery_throws(monkeypatch):
    original = demo.ActorShadowRuntime
    captured = []

    def runtime(source, **options):
        owner = original(source, **options)
        captured.append((source, owner, options, source.snapshot_state()))
        ready = owner.train_ready

        def fail_after_commit(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if applied:
                raise RuntimeError("synthetic postcommit delivery failure")
            return applied

        monkeypatch.setattr(owner, "train_ready", fail_after_commit)
        return owner

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, a_only_control=True
    )
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert "postcommit delivery" in report["error"]
    assert report["progress"] == dict(
        update_attempts=1, updates_completed=1, replay_examples_completed=0, next_tick=2
    )
    assert report["settings"]["training_contexts"] == ["warm"] * 3
    assert len(report["events"]) == len(report["applied_updates"]) == 1
    assert (
        report["events"][0]["applied"] == [] and report["events"][0]["success_validated"] is False
    )
    receipt = report["applied_updates"][0]
    assert receipt["episode_id"] == episode().episode_id
    assert receipt["observed_at"] == 0 and receipt["arrived_at"] == receipt["applied_at"] == 1
    assert "work" not in report and "metrics" not in report
    source, owner, options, initial = captured[0]
    assert (
        options["budget"].updates_completed == 1
        and options["budget"].replay_examples_completed == 0
    )
    assert owner.candidate_snapshot().state.training_examples == 2
    same_state(initial, source.snapshot_state())
    same_state(initial, owner.actor.snapshot_state().state)


@pytest.mark.parametrize("flag", [False, True])
def test_should_preserve_json_wrapper_call_shape_and_forward_only_opt_in(monkeypatch, flag):
    calls = []
    report = dict(complete=True, stub=True)

    def run(*args, **options):
        calls.append((args, options))
        return SimpleNamespace(report=report)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    warm, new = object(), object()
    assert demo.run_continuous_learnability_demo(warm, new, a_only_control=flag) is report
    expected: dict[str, Any] = dict(expected_standardization=None)
    if flag:
        expected["a_only_control"] = True
    assert calls == [((warm, new), expected)]


@pytest.mark.parametrize("flag", [False, True])
def test_should_route_cli_control_without_reading_relabelled_or_additional_sources(
    monkeypatch, capsys, flag
):
    records, reads, calls = [object(), object()], [], []

    def read(path, seed, **options):
        reads.append((str(path), seed, options))
        return records[len(reads) - 1]

    def run(*args, **options):
        calls.append((args, options))
        return dict(complete=True)

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
    flags = ["--a-only-control"] if flag else []
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
            *flags,
        ],
    )
    cli.main()
    assert reads == [("A.json", 17, {}), ("B.json", 19, {})]
    assert calls == [(tuple(records), {"a_only_control": True} if flag else {})]
    assert json.loads(capsys.readouterr().out)["complete"] is True


def test_should_compose_cli_a_only_with_existing_mass_phase_and_binary_options(monkeypatch, capsys):
    warm, new, reads, calls = object(), object(), [], []

    def read(path, seed, **options):
        reads.append((str(path), seed, options))
        return warm

    def mass(path, **options):
        reads.append((str(path), "mass17", options))
        return new, dict(
            new_seed=17, acquisition_scope="retained_teacher_work_not_new_simulator_work"
        )

    def run(*args, **options):
        calls.append((args, options))
        return dict(complete=True)

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "_read_mass_teacher", mass)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
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
            "mass.json",
            "--mass-drift",
            "--phase-updates",
            "1",
            "--binary-gripper-axis",
            "3",
            "--a-only-control",
        ],
    )
    cli.main()
    assert reads == [("A.json", 17, {}), ("mass.json", "mass17", {})]
    assert calls == [
        ((warm, new), dict(phase_updates=1, binary_gripper_axis=3, a_only_control=True))
    ]
    assert json.loads(capsys.readouterr().out)["input_context"]["new_seed"] == 17


def test_should_keep_incomplete_cli_control_report_and_paid_progress(monkeypatch, capsys):
    reads = []

    def read(_path, seed, **_options):
        reads.append(seed)
        return object()

    def run(_warm, _new, **options):
        assert options == dict(a_only_control=True)
        return dict(
            complete=False,
            error="retained paid control failure",
            progress=dict(updates_completed=1),
        )

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
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
            "--a-only-control",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    report = json.loads(capsys.readouterr().out)
    assert caught.value.code == 1 and reads == [17, 19]
    assert report["complete"] is False and report["progress"] == dict(updates_completed=1)
    assert report["error"] == "retained paid control failure"


@pytest.mark.parametrize(
    "scenario",
    ["stream", "promotion-rejection", "coupled-neutral", "matched-baselines", "continuous-nearest"],
)
def test_should_refuse_cli_control_outside_mlp_before_input_or_runner(monkeypatch, scenario):
    monkeypatch.setattr(cli, "_read_teacher", forbidden)
    monkeypatch.setattr(cli, "_read_mass_teacher", forbidden)
    for name in (
        "run_demo",
        "run_coupled_demo",
        "run_matched_demo",
        "run_rejection_demo",
        "run_continuous_demo",
        "run_continuous_learnability_demo",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["demo", "--scenario", scenario, "--a-only-control"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
