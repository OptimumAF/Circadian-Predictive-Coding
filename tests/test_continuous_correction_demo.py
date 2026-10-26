"""Bounded private correction updates and fake-IO collection-to-rollout integration.

One shared success executes8 SGD per invocation; failure cases use1+1+1+8,
the complete fake pipeline uses8, and direct_reference adds8 exactly once.
No other tests train. The executable runner owns the aggregate budget.
"""

from dataclasses import asdict, replace
import json
import sys

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core.continuous_bc import OBJECTIVE
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_corrections import PickCubeCorrectionBatch
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_maniskill_bc_smoke import FakeEnvironment, assert_same_state, learner
from test_pick_cube_corrections import episode, labels_for


def source(**options):
    values = dict(
        observation_standardization=ObservationStandardization((0.5, 0.0, 0.0), (0.5, 1.0, 1.0), 2)
    )
    values.update(options)
    return learner(**values)


def batch():
    record = episode(19)
    return PickCubeCorrectionBatch(19, record, labels_for(record))


@pytest.fixture(scope="module")
def completed():
    owner, corrections = source(), batch()
    initial, original_batch = owner.snapshot_state(), asdict(corrections)
    captured = dict(trains=[], runtimes=[], forks=[])
    runtime_factory = demo.ActorShadowRuntime
    original_train = NumpyContinuousBCLearner.train_batch
    original_fork = NumpyContinuousBCLearner.fork
    with pytest.MonkeyPatch.context() as patch:

        def runtime(model, **options):
            created = runtime_factory(model, **options)
            captured["runtimes"].append((created, options))
            return created

        def train(model, features, targets):
            captured["trains"].append((model, features.copy(), targets.copy()))
            return original_train(model, features, targets)

        def fork(model):
            child = original_fork(model)
            captured["forks"].append((model, child))
            return child

        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(NumpyContinuousBCLearner, "train_batch", train)
        patch.setattr(NumpyContinuousBCLearner, "fork", fork)
        result = demo.run_continuous_correction_update(owner, corrections, now=3)
    assert result.report["complete"] is True
    assert asdict(corrections) == original_batch
    return owner, corrections, initial, result, captured


def test_should_continue_full_pretrained_policy_momentum_and_counters_on_private_candidate(
    completed,
):
    owner, corrections, initial, result, captured = completed
    assert type(result) is ContinuousLearnabilityResult
    assert type(result.candidate) is NumpyContinuousBCLearner
    runtime, options = captured["runtimes"][0]
    assert len(captured["runtimes"]) == 1
    assert len(captured["trains"]) == 8
    assert all(model is runtime._candidate for model, _, _ in captured["trains"])
    assert result.candidate is not owner and result.candidate is not runtime._candidate
    assert_same_state(initial, owner.snapshot_state())
    assert_same_state(initial, runtime.actor.snapshot_state().state)
    state = result.candidate.snapshot_state()
    assert state.policy == initial.policy
    assert state.completed_updates == initial.completed_updates + 8 == 40
    assert state.training_examples == initial.training_examples + 16 == 944
    assert_same_state(state, runtime.candidate_snapshot().state)
    assert any(
        not np.array_equal(a, b) for a, b in zip(initial.velocity, state.velocity, strict=True)
    )
    assert options["budget"].updates_completed == 8
    assert options["budget"].replay_examples_completed == 0
    assert options["budget"].budget.max_training_updates == 8
    observations, targets = corrections.training_pairs(now=3)
    for _, features, labels in captured["trains"]:
        np.testing.assert_array_equal(features, np.asarray(observations))
        np.testing.assert_array_equal(labels, np.asarray(targets))
        assert not np.array_equal(
            labels, np.asarray([row.action for row in corrections.episode.transitions])
        )


def test_should_preserve_original_label_ticks_unique_exposure_ids_and_all_charges(completed):
    _, corrections, initial, result, _ = completed
    report = result.report
    assert report["source_updates"] == initial.completed_updates
    assert report["source_training_rows"] == initial.training_examples
    assert len(report["events"]) == len(report["applied_updates"]) == 8
    for index, (event, receipt) in enumerate(
        zip(report["events"], report["applied_updates"], strict=True)
    ):
        assert event["success_validated"] is event["update_attempted"] is True
        assert event["applied"] == [receipt]
        assert receipt["sample_id"] == f"corrective-batch-{index}"
        assert receipt["event_id"] == f"corrective-label-{index}"
        assert receipt["episode_id"] == corrections.episode.episode_id
        assert receipt["observed_at"] == 2 and receipt["arrived_at"] == 3
        assert receipt["applied_at"] >= receipt["arrived_at"]
        assert receipt["update_number"] == index + 1
        assert receipt["actor_version"] == "actor-0" and receipt["learner_version"] == "candidate-0"
        assert receipt["diagnostic"]["definition"] == OBJECTIVE
        assert np.isfinite(receipt["diagnostic"]["value"])
    assert report["progress"]["updates_completed"] == 8
    assert report["progress"]["replay_examples_completed"] == 0
    json.dumps(report, allow_nan=False)


def test_should_match_direct_reference_from_full_pretrained_momentum(completed):
    """Exactly8 additional SGD; executable runner selects this once globally."""
    owner, corrections, initial, result, _ = completed
    reference = owner.fork()
    assert_same_state(initial, reference.snapshot_state())
    observations, targets = corrections.training_pairs(now=3)
    features = np.asarray(observations, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    losses = [reference.train_batch(features, labels).value for _ in range(8)]
    assert losses == [row["diagnostic"]["value"] for row in result.report["applied_updates"]]
    assert_same_state(reference.snapshot_state(), result.candidate.snapshot_state())


def test_should_return_owned_candidate_snapshots_without_payload_aliases(completed):
    owner, corrections, initial, result, captured = completed
    state = result.candidate.snapshot_state()
    state.parameters[0].flat[0] += 100.0
    state.velocity[0].flat[0] -= 100.0
    runtime, _ = captured["runtimes"][0]
    assert_same_state(result.candidate.snapshot_state(), runtime.candidate_snapshot().state)
    assert_same_state(initial, owner.snapshot_state())
    assert corrections.episode.transitions[0].action != corrections.labels[0].target


def test_should_count_only_eight_correction_exposures_without_fit_replay_or_diagnostics(completed):
    _, _, _, result, _ = completed
    expected = dict(
        optimizer_updates=8,
        training_rows=16,
        unique_correction_rows=2,
        repeated_training_rows=14,
        reference_replay_rows=0,
        fit_calls=0,
        forward_passes=8,
        forward_rows=16,
        backward_passes=8,
        backward_rows=16,
        normalization_calls=8,
        normalization_rows=16,
        normalization_values=48,
        gradient_values=160,
        gradient_numeric_bytes=1280,
        dense_multiply_accumulates=576,
        model_numeric_bytes_per_owner=320,
        statistics_numeric_bytes_per_owner=48,
        named_model_owners=4,
        inbox_feature_target_numeric_bytes=896,
    )
    for key, value in expected.items():
        assert result.report["work"][key] == value, key
    assert result.report["settings"] == dict(
        max_training_updates=8,
        reference_replay_rows=0,
        fit_calls=0,
        promotion_calls=0,
        simulator_calls=0,
        guard_reads=0,
        probe_reads=0,
        final_reads=0,
    )


@pytest.mark.parametrize("now", [True, -1, 2, 3.0, None])
def test_should_refuse_unarrived_or_invalid_time_before_fork(monkeypatch, now):
    owner, corrections = source(), batch()
    initial = owner.snapshot_state()
    monkeypatch.setattr(
        NumpyContinuousBCLearner, "fork", lambda *_: pytest.fail("invalid time forked")
    )
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(owner, corrections, now=now)
    assert_same_state(initial, owner.snapshot_state())


@pytest.mark.parametrize(
    "damage",
    [
        "source_type",
        "source_role",
        "binary",
        "statistics",
        "spec",
        "batch_type",
        "batch_role",
        "batch_label",
    ],
)
def test_should_refuse_invalid_source_or_corrections_before_fork(monkeypatch, damage):
    options = {}
    if damage == "source_role":
        options["training_role"] = "probe_train"
    elif damage == "binary":
        options["binary_gripper_axis"] = 3
    elif damage == "statistics":
        options["observation_standardization"] = None
    owner, corrections = source(**options), batch()
    if damage == "source_type":
        owner = object()
    elif damage == "spec":
        object.__setattr__(corrections.episode.spec, "action_units", ("other",) * 4)
    elif damage == "batch_type":
        corrections = asdict(corrections)
    elif damage == "batch_role":
        object.__setattr__(corrections.episode, "role", "final_test")
    elif damage == "batch_label":
        object.__setattr__(corrections.labels[0], "observation", (99.0, 0.0, 0.0))
    monkeypatch.setattr(
        NumpyContinuousBCLearner, "fork", lambda *_: pytest.fail("invalid input forked")
    )
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(owner, corrections, now=3)


@pytest.mark.parametrize("failure", ["corrupt_receipt", "postcommit_throw"])
def test_should_stop_after_one_real_update_and_retain_permanent_receipt(monkeypatch, failure):
    """Exactly1 SGD per case; no later attempt may be admitted."""
    owner = source()
    initial = owner.snapshot_state()
    captured = []
    original = demo.ActorShadowRuntime

    def runtime(model, **options):
        created = original(model, **options)
        captured.append((created, options))
        ready = created.train_ready

        def changed(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if applied:
                assert options["budget"].updates_completed == 1
                if failure == "postcommit_throw":
                    raise RuntimeError("synthetic correction postcommit failure")
                return (replace(applied[0], arrived_at=applied[0].arrived_at + 1),)
            return applied

        monkeypatch.setattr(created, "train_ready", changed)
        return created

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_correction_update(owner, batch(), now=3)
    runtime_owner, options = captured[0]
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["progress"]["updates_completed"] == 1
    assert len(result.report["events"]) == len(result.report["applied_updates"]) == 1
    assert result.report["events"][0]["success_validated"] is False
    receipt = result.report["applied_updates"][0]
    assert receipt["arrived_at"] == 3 and receipt["observed_at"] == 2
    assert receipt["diagnostic"]["definition"] == OBJECTIVE
    assert options["budget"].updates_completed == 1
    assert "work" not in result.report
    assert_same_state(initial, owner.snapshot_state())
    assert_same_state(initial, runtime_owner.actor.snapshot_state().state)


def test_should_refuse_replaced_preceding_receipt_before_admitting_another_update(monkeypatch):
    """Exactly1 SGD; a formerly validated receipt cannot silently change."""
    owner = source()
    initial = owner.snapshot_state()
    captured = []
    factory, check_owners = demo.ActorShadowRuntime, demo._require_owners
    altered = []

    def runtime(model, **options):
        created = factory(model, **options)
        captured.append((created, options))
        return created

    def change_after_validation(model, runtime_owner, expected):
        check_owners(model, runtime_owner, expected)
        if runtime_owner.applied_updates and not altered:
            key, receipt = next(iter(runtime_owner._inbox._applied.items()))
            runtime_owner._inbox._applied[key] = replace(receipt, arrived_at=receipt.arrived_at + 1)
            altered.append(True)

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(demo, "_require_owners", change_after_validation)
    result = demo.run_continuous_correction_update(owner, batch(), now=3)
    assert result.candidate is None and result.report["complete"] is False
    assert "preceding" in result.report["error"]
    assert len(result.report["events"]) == 1
    assert result.report["events"][0]["success_validated"] is True
    assert result.report["events"][0]["applied"][0]["arrived_at"] == 3
    assert result.report["applied_updates"][0]["arrived_at"] == 4
    assert result.report["progress"]["updates_completed"] == 1
    assert captured[0][1]["budget"].updates_completed == 1
    assert_same_state(initial, owner.snapshot_state())
    assert_same_state(initial, captured[0][0].actor.snapshot_state().state)


def test_should_refuse_final_empty_poll_mutation_after_eight_charged_updates(monkeypatch):
    """Exactly8 SGD for the final-drain corruption regression."""
    owner = source()
    initial = owner.snapshot_state()
    captured = []
    original = demo.ActorShadowRuntime

    def runtime(model, **options):
        created = original(model, **options)
        captured.append((created, options))
        ready = created.train_ready

        def changed(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if not applied and options["budget"].updates_completed == 8:
                state = created._candidate.snapshot_state()
                state.velocity[0].flat[0] += 0.5
                created._candidate.restore_state(state)
            return applied

        monkeypatch.setattr(created, "train_ready", changed)
        return created

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_correction_update(owner, batch(), now=3)
    assert result.candidate is None and result.report["complete"] is False
    assert len(result.report["applied_updates"]) == 8
    assert all(event["success_validated"] is True for event in result.report["events"])
    assert result.report["progress"]["updates_completed"] == 8
    assert "work" not in result.report
    assert captured[0][1]["budget"].updates_completed == 8
    assert_same_state(initial, owner.snapshot_state())
    assert_same_state(initial, captured[0][0].actor.snapshot_state().state)


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environments = []

    def factory():
        environment = FakeEnvironment()
        environments.append(environment)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-correction-pipeline")
    monkeypatch.setattr(
        queries,
        "_geometry",
        lambda _: PickCubeGeometry((0.0, 0.0, 0.0), (0.05, 0.0, 0.02), (0.1, 0.0, 0.2)),
    )
    return environments, tmp_path


def test_should_release_programmatic_batch_only_after_closed_frozen_rollout(boundary):
    environments, directory = boundary
    owner = source()
    before = owner.snapshot_state()
    output = directory / "collection.json"
    result = smoke.run_frozen_rollout(output, learner=owner, train_seed=19, corrective_queries=True)
    assert type(result) is smoke.FrozenRolloutResult and result.report["complete"] is True
    assert type(result.corrections) is PickCubeCorrectionBatch
    assert environments[0].close_calls == 1
    assert_same_state(before, owner.snapshot_state())
    assert result.corrections.train_seed == 19
    assert result.corrections.episode.episode_id == "pick-cube-smoke-19"
    observations, targets = result.corrections.training_pairs(now=3)
    assert observations == tuple(
        tuple(row["observation"]) for row in result.report["episode"]["transitions"]
    )
    assert targets == ((0.5, 0.0, 1.0, 1.0),) * 2
    result.report["episode"]["transitions"][0]["observation"] = (100.0, 0.0, 0.0)
    assert result.corrections.training_pairs(now=3) == (observations, targets)
    with pytest.raises(FileExistsError):
        smoke.run_frozen_rollout(output, learner=owner, corrective_queries=True)
    assert len(environments) == 1


def test_should_release_no_batch_when_cleanup_fails_after_collection(boundary, monkeypatch):
    environments, directory = boundary
    factory = native._make_environment

    def failing_factory():
        environment = factory()
        environment.close_error = RuntimeError("synthetic late close failure")
        return environment

    monkeypatch.setattr(native, "_make_environment", failing_factory)
    result = smoke.run_frozen_rollout(
        directory / "failed.json", learner=source(), corrective_queries=True
    )
    assert result.corrections is None and result.report["complete"] is False
    assert result.report["cleanup_error"] == "synthetic late close failure"
    assert result.report["corrections"]["batch"] is None
    assert len(result.report["corrections"]["labels"]) == 2
    assert environments[0].close_calls == 1 and len(environments[0].commands) == 2


def test_should_keep_default_programmatic_and_cli_reports_equivalent(boundary, monkeypatch):
    environments, directory = boundary
    programmatic = smoke.run_frozen_rollout(directory / "api.json")
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(directory / "cli.json")])
    assert smoke.main() == 0
    api_report = json.loads((directory / "api.json").read_text(encoding="utf8"))
    cli_report = json.loads((directory / "cli.json").read_text(encoding="utf8"))
    api_report.pop("elapsed_seconds")
    cli_report.pop("elapsed_seconds")
    assert api_report == cli_report
    assert programmatic.corrections is None and "corrections" not in api_report
    assert len(environments) == 2
    assert all(environment.reset_calls == [17] for environment in environments)


@pytest.mark.parametrize(
    "options", [{"corrective_queries": True}, {"train_seed": 23}, {"corrective_queries": 1}]
)
def test_should_keep_programmatic_refusals_before_scene(boundary, options):
    environments, directory = boundary
    result = smoke.run_frozen_rollout(directory / "refused.json", **options)
    assert result.report["complete"] is False and result.corrections is None
    assert environments == []


def test_should_collect_update_and_roll_out_both_train_contexts_without_changing_source(
    boundary, monkeypatch
):
    """Exactly8 private SGD, one real collection and two frozen fake-IO rollouts."""
    environments, directory = boundary
    owner = source()
    initial = owner.snapshot_state()
    captured = []
    actual_update = smoke.run_continuous_correction_update

    def update(model, corrections, *, now):
        result = actual_update(model, corrections, now=now)
        captured.append((corrections, now, result, result.candidate.snapshot_state()))
        return result

    monkeypatch.setattr(smoke, "run_continuous_correction_update", update)
    output = directory / "pipeline"
    summary = smoke.run_corrective_adaptation(
        owner, output_directory=output, visual_observation=False
    )
    assert summary["complete"] is True and summary["source_state_exact"] is True
    assert [environment.reset_calls for environment in environments] == [[19], [17], [19]]
    assert all(
        environment.close_calls == 1 and len(environment.commands) == 2
        for environment in environments
    )
    assert_same_state(initial, owner.snapshot_state())
    assert {path.name for path in output.iterdir()} == {
        "collection.json",
        "update.json",
        "updated-17.json",
        "updated-19.json",
        "summary.json",
    }
    reports = {
        name: json.loads((output / name).read_text(encoding="utf8"))
        for name in (
            "collection.json",
            "update.json",
            "updated-17.json",
            "updated-19.json",
            "summary.json",
        )
    }
    assert reports["summary.json"] == json.loads(json.dumps(summary))
    assert (
        summary["guard_reads"]
        == summary["probe_reads"]
        == summary["final_reads"]
        == summary["promotion_calls"]
        == 0
    )
    assert [row["seed"] for row in summary["rollouts"]] == [17, 19]
    assert summary["collection"]["queries"]["geometry_attempts"] == 2
    assert len(captured) == 1
    corrections, now, updated, candidate_state = captured[0]
    assert now == 3 and corrections.train_seed == 19
    assert updated.report["progress"]["updates_completed"] == 8
    assert_same_state(candidate_state, updated.candidate.snapshot_state())
    assert candidate_state.completed_updates == 40 and candidate_state.training_examples == 944
    for index, name in enumerate(("collection.json", "updated-17.json", "updated-19.json")):
        report = reports[name]
        assert report["optimizer_updates"] == 0
        assert report["frozen_state_exact"] is report["original_owner_exact"] is True
        assert report["source_updates"] == (32 if index == 0 else 40)
        assert report["source_training_rows"] == (928 if index == 0 else 944)
        assert ("corrections" in report) is (index == 0)
        predictor = owner if index == 0 else updated.candidate
        for row, command in zip(
            report["episode"]["transitions"], environments[index].commands, strict=True
        ):
            expected = predictor.predict(np.asarray([row["observation"]], dtype=np.float64)).astype(
                np.float32
            )
            np.testing.assert_array_equal(command, expected)
            assert row["action"] == command[0].tolist()
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(environments[0].commands, environments[1].commands, strict=True)
    )


@pytest.mark.parametrize("failure", ["collection", "update", "update_write", "first_rollout"])
def test_should_stop_pipeline_at_failed_stage_and_preserve_returned_reports(
    boundary, monkeypatch, failure
):
    """Zero SGD: fake update returns a full owned continuation only when needed."""
    environments, directory = boundary
    owner = source()
    initial = owner.snapshot_state()
    collected = []
    real_rollout = smoke.run_frozen_rollout

    def rollout(output, **options):
        if failure == "collection":
            result = smoke.FrozenRolloutResult(
                dict(complete=False, error="collection refused"), None
            )
        elif failure == "first_rollout" and collected:
            result = smoke.FrozenRolloutResult(
                dict(complete=False, error="first rollout refused"), None
            )
        else:
            result = real_rollout(output, **options)
        collected.append((options["train_seed"], result))
        return result

    def update(model, corrections, *, now):
        if failure in ("update", "update_write"):
            return ContinuousLearnabilityResult(
                dict(
                    complete=False,
                    error="paid update failed",
                    applied_updates=[dict(update_number=1)],
                    progress=dict(updates_completed=1),
                ),
                None,
            )
        candidate = model.fork()
        candidate.restore_state(replace(initial, completed_updates=40, training_examples=944))
        return ContinuousLearnabilityResult(dict(complete=True, synthetic=True), candidate)

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", update)
    if failure == "update_write":
        original_open = type(directory).open

        def open_without_update(path, *args, **kwargs):
            if path.name == "update.json":
                raise OSError("synthetic update receipt write failure")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(type(directory), "open", open_without_update)
    output = directory / "failed-pipeline"
    summary = smoke.run_corrective_adaptation(
        owner, output_directory=output, visual_observation=False
    )
    assert summary["complete"] is False and summary["source_state_exact"] is True
    assert [seed for seed, _ in collected] == ([19, 17] if failure == "first_rollout" else [19])
    assert_same_state(initial, owner.snapshot_state())
    assert (output / "summary.json").exists()
    if failure in ("update", "update_write"):
        assert summary["update"]["applied_updates"] == [dict(update_number=1)]
        if failure == "update":
            assert (
                json.loads((output / "update.json").read_text())["progress"]["updates_completed"]
                == 1
            )
        else:
            assert "synthetic update receipt write failure" in summary["error"]
            assert not (output / "update.json").exists()
    if failure == "first_rollout":
        assert len(summary["rollouts"]) == 1 and summary["rollouts"][0]["complete"] is False
    assert len(environments) == (0 if failure == "collection" else 1)


@pytest.mark.parametrize(
    "damage",
    ["role", "binary", "statistics", "visual_flag", "visual_dimension", "existing_directory"],
)
def test_should_refuse_invalid_pipeline_admission_without_creating_scene(boundary, damage):
    environments, directory = boundary
    options = {}
    if damage == "role":
        options["training_role"] = "probe_train"
    elif damage == "binary":
        options["binary_gripper_axis"] = 3
    elif damage == "statistics":
        options["observation_standardization"] = None
    output = directory / "refused-pipeline"
    if damage == "existing_directory":
        output.mkdir()
        (output / "preserved.txt").write_text("prior attempt", encoding="utf8")
    with pytest.raises((ValueError, FileExistsError)):
        smoke.run_corrective_adaptation(
            source(**options),
            output_directory=output,
            visual_observation=1 if damage == "visual_flag" else damage == "visual_dimension",
        )
    assert environments == []
    if damage == "existing_directory":
        assert (output / "preserved.txt").read_text(encoding="utf8") == "prior attempt"
