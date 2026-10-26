"""Synthetic receipts around the existing nearest actor/runtime, with no SDK/SGD."""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from src.app.actor_shadow import StableActor
from src.core.pick_cube_camera_context import camera_context_recipe, pick_cube_camera_context
from test_visual_native_context_case import (
    read_audit,
    rollout_receipt,
    teacher_receipt,
    write_new,
)


case = importlib.import_module("artifacts.runs.camera-context-return-20261019.native_case")


def camera_teacher(rows=2):
    raw = teacher_receipt(17, rows)
    camera = pick_cube_camera_context()
    raw.update(
        context=camera.context,
        camera_context=asdict(camera),
        observation_recipe=json.loads(json.dumps(asdict(camera_context_recipe(camera)))),
        observed_success=True,
        creation_attempts=1,
        geometry_attempts=rows,
        geometry_returns=rows,
        teacher_actions=rows,
        stages=["align"] * rows,
        traceback=None,
        guard_seed=None,
        labels_arrived_at=None,
        optimizer_updates=0,
    )
    episode = raw["candidate_episode"]
    episode["context"] = camera.context
    episode["episode_id"] = f"{camera.context}-teacher"
    for row in episode["transitions"]:
        row["observation"][21:] = [0.75] * 768
        row["next_observation"][21:] = [0.75] * 768
        row["action"][0] = 1.0
    return raw


def camera_rollout(teacher, *, success=False):
    raw = rollout_receipt(teacher, success=success)
    if "camera_context" in teacher:
        raw["camera_context"] = deepcopy(teacher["camera_context"])
        raw["observation_recipe"] = deepcopy(teacher["observation_recipe"])
    return raw


@pytest.fixture
def synthetic_case(monkeypatch, tmp_path):
    warm, new = teacher_receipt(), camera_teacher()
    warm_path, reference_path = tmp_path / "accepted-a.json", tmp_path / "accepted-reference.json"
    write_new(warm_path, warm)
    write_new(reference_path, camera_rollout(warm, success=True))
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(case, "WARM_PATH", warm_path)
    monkeypatch.setattr(case, "REFERENCE_PATH", reference_path)
    observed = dict(
        warm=warm,
        new=new,
        teacher_calls=[],
        smoke_calls=[],
        fits=[],
        trains=[],
        runtime=[],
        stream=[],
        mutate=None,
        fail_at=None,
        teacher_code=0,
        successes={},
    )
    original_fit = case.fit_training_standardization
    original_train = case.NumpyNearestContinuousBCLearner.train_batch
    original_runtime = case.ActorShadowRuntime

    def fit(episode):
        observed["fits"].append(episode)
        return original_fit(episode)

    def train(owner, features, targets):
        observed["trains"].append((owner, len(features)))
        return original_train(owner, features, targets)

    def runtime(source, **options):
        owner = original_runtime(source, **options)
        observed["runtime"].append((source, owner, options, source.snapshot_state()))
        for method_name in ("record_experience", "record_label", "train_ready"):
            original = getattr(owner, method_name)

            def tracked(*args, _name=method_name, _original=original, **kwargs):
                before = options["budget"].updates_completed
                result = _original(*args, **kwargs)
                observed["stream"].append(
                    (
                        _name,
                        options["clock"].now(),
                        before,
                        options["budget"].updates_completed,
                        args,
                    )
                )
                return result

            monkeypatch.setattr(owner, method_name, tracked)
        return owner

    def teacher_main():
        observed["teacher_calls"].append(tuple(sys.argv))
        assert "--visual-observation" in sys.argv and "--camera-drift" in sys.argv
        assert sys.argv[sys.argv.index("--train-seed") + 1] == "17"
        assert not observed["fits"] and not observed["runtime"]
        write_new(Path(sys.argv[sys.argv.index("--output") + 1]), observed["new"])
        return observed["teacher_code"]

    def smoke_main(*, train_seed, actor=None, learner=None, camera_context=None):
        destination = Path(sys.argv[sys.argv.index("--output") + 1])
        name = destination.name
        observed["smoke_calls"].append((name, train_seed, actor, learner, camera_context))
        assert train_seed == 17 and "--visual-observation" in sys.argv
        assert ("--camera-drift" in sys.argv) is (camera_context is not None)
        if observed["fail_at"] == name:
            write_new(destination, dict(complete=False, error="synthetic renderer failure"))
            return 1
        source = observed["new"] if camera_context is not None else observed["warm"]
        raw = camera_rollout(source, success=observed["successes"].get(name, False))
        owner = actor if actor is not None else learner
        state = owner.snapshot_state().state if actor is not None else owner.snapshot_state()
        rows, prototypes = len(raw["episode"]["transitions"]), state.training_examples
        indices = []
        for row in raw["episode"]["transitions"]:
            query = np.asarray([row["observation"]], dtype=np.float64)
            if actor is not None:
                result = actor.predict(query)
                assert result.actor_version == "serving-0"
                action = result.prediction[0]
            else:
                result = learner.predict_with_work(query)
                indices.extend(result.selected_indices)
                action = result.actions[0]
            row["action"] = action.tolist()
        raw.update(
            policy="frozen_stable_nearest_continuous_bc"
            if actor is not None
            else "frozen_nearest_continuous_bc",
            frozen_state_exact=True,
            original_owner_exact=True,
            source_updates=state.completed_updates,
            source_training_rows=prototypes,
            nearest_prototype_rows=prototypes,
            nearest_prototype_numeric_bytes=prototypes * 793 * 8,
            source_update_definition=case.OBJECTIVE,
            prediction_attempts=rows,
            prediction_returns=rows,
            validated_prediction_rows=rows,
            validated_prediction_bytes=32 * rows,
            optimizer_updates=0,
            nearest_prediction_attempts=0 if actor is not None else rows,
            nearest_prediction_returns=0 if actor is not None else rows,
            nearest_selected_indices=indices,
            state_guard_scope="actual_serving_actor_no_private_fork"
            if actor is not None
            else "private_frozen_fork",
            actor_version="serving-0" if actor is not None else None,
            actor_version_exact=True,
            actor_backend_exact=True,
            actor_prediction_version_exact=True,
            actor_prediction_versions=["serving-0"] * rows if actor is not None else [],
            actor_prediction_attempts=rows if actor is not None else 0,
            actor_prediction_returns=rows if actor is not None else 0,
            actor_distance_count_source="derived_from_exact_unchanged_typed_actor_and_query_shapes",
        )
        for prefix, active in (("actor", actor is not None), ("nearest", actor is None)):
            count = rows * prototypes if active else 0
            raw[prefix + "_distance_candidate_upper_bound"] = count
            raw[prefix + "_distance_candidates_completed"] = count
            raw[prefix + "_distance_feature_upper_bound"] = count * 789
            raw[prefix + "_distance_feature_values_completed"] = count * 789
        if observed["mutate"] is not None:
            observed["mutate"](name, raw)
        write_new(destination, raw)
        return 0

    monkeypatch.setattr(case, "fit_training_standardization", fit)
    monkeypatch.setattr(case.NumpyNearestContinuousBCLearner, "train_batch", train)
    monkeypatch.setattr(case, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(case.teacher_cli, "main", teacher_main)
    monkeypatch.setattr(case.smoke, "main", smoke_main)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])
    return observed


def test_should_run_same_reset_camera_acquisition_then_fresh_original_return(
    synthetic_case, tmp_path
):
    observed = synthetic_case
    inputs = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert case.main() == 0
    audit = read_audit(tmp_path)
    assert audit["complete"] is True and "error" not in audit
    assert len(observed["teacher_calls"]) == len(observed["fits"]) == len(observed["runtime"]) == 1
    assert len(observed["fits"][0].transitions) == 29
    assert [
        (name, seed, camera is not None) for name, seed, _, _, camera in observed["smoke_calls"]
    ] == [
        ("b-before.json", 17, True),
        ("b-after.json", 17, True),
        ("policy-rollout.json", 17, False),
    ]
    source, runtime, options, initial = observed["runtime"][0]
    assert type(runtime.actor) is StableActor and observed["smoke_calls"][0][2] is runtime.actor
    assert options["max_experiences"] == 1 and options["max_consolidations"] == 0
    assert options["budget"].updates_completed == 1 and options["clock"].now() == 2
    assert [rows for _, rows in observed["trains"]] == [29, 2]
    assert observed["trains"][0][0] is source and observed["trains"][1][0] is runtime._candidate
    case.same_state(initial, source.snapshot_state())
    case.same_state(initial, runtime.actor.snapshot_state().state)
    final = runtime.candidate_snapshot().state
    assert (
        final.policy == initial.policy
        and final.training_examples == 31
        and final.completed_updates == 2
    )
    np.testing.assert_array_equal(final.prototype_features[:29], initial.prototype_features)
    np.testing.assert_array_equal(final.prototype_actions[:29], initial.prototype_actions)
    assert [(name, tick, old, new) for name, tick, old, new, _ in observed["stream"]] == [
        ("record_experience", 0, 0, 0),
        ("train_ready", 0, 0, 0),
        ("train_ready", 1, 0, 0),
        ("record_label", 2, 0, 0),
        ("train_ready", 2, 0, 1),
        ("train_ready", 2, 1, 1),
    ]
    assert audit["events"] == [
        dict(tick=0, candidate_rows=29, budget_updates=0, applied=0),
        dict(tick=1, candidate_rows=29, budget_updates=0, applied=0),
        dict(tick=2, candidate_rows=31, budget_updates=1, applied=1),
    ]
    assert audit["applied"]["observed_at"] == 0
    assert audit["applied"]["arrived_at"] == audit["applied"]["applied_at"] == 2
    assert audit["applied"]["diagnostic"] == dict(definition=case.OBJECTIVE, value=2.0)
    assert (
        audit["permanent_update_charge"] == 1
        and audit["a_prefix_exact"] is audit["frozen_a_statistics_exact"] is True
    )
    assert audit["initial_camera_effect"] == dict(
        physical_prefix_exact=True,
        differing_rgb_coordinates=768,
        rgb_mean_absolute_change=0.5,
        rgb_max_absolute_change=0.5,
    )
    for name, payload in inputs.items():
        assert (tmp_path / name).read_bytes() == payload


def test_should_continue_after_successful_baseline_and_keep_negative_endpoints(
    synthetic_case, tmp_path
):
    synthetic_case["successes"]["b-before.json"] = True
    assert case.main() == 0
    audit = read_audit(tmp_path)
    assert len(synthetic_case["trains"]) == 2
    assert audit["b_after_minus_before"] == dict(
        sparse_return=-1.0, any_success=-1.0, final_success=-1.0
    )
    assert audit["a_reference_minus_return_retention_loss"] == dict(
        sparse_return=1.0, any_success=1.0, final_success=1.0
    )
    assert audit["rollouts"]["b-before.json"]["final_success"] is True
    assert audit["rollouts"]["b-after.json"]["final_success"] is False


def test_should_count_one_bounded_archive_and_all_controls_exposures_and_distances(
    synthetic_case, tmp_path
):
    assert case.main() == 0
    audit = read_audit(tmp_path)
    assert audit["prediction_rows"] == 33 and audit["prediction_bytes"] == 1056
    assert audit["distance_candidates"] == 2 * 29 + 31 * 31
    assert audit["distance_feature_values"] == (2 * 29 + 31 * 31) * 789
    assert audit["control_attempts"] == 35 and audit["known_control_physics_steps"] == 175
    assert audit["visual_projections"] == 39 and audit["raw_frame_input_bytes"] == 39 * 49152
    assert audit["normalization_calls"] == 35 and audit["normalization_rows"] == 64
    assert audit["fit_rows"] == 29 and audit["fit_numeric_bytes"] == 12624
    assert audit["prototype_growth_numeric_bytes"] == audit["inbox_numeric_bytes"] == 2 * 793 * 8
    assert audit["source_rows"] == 29 and audit["candidate_rows"] == 31
    archives = list(tmp_path.glob("*.npz"))
    assert len(archives) == 1 and archives[0].stat().st_size <= 1024**2
    with np.load(archives[0], allow_pickle=False) as saved:
        assert set(saved.files) == {
            "warm_features",
            "warm_actions",
            "candidate_features",
            "candidate_actions",
        }
        assert saved["candidate_features"].shape == (31, 789)
        np.testing.assert_array_equal(saved["warm_features"], saved["candidate_features"][:29])
        np.testing.assert_array_equal(saved["warm_actions"], saved["candidate_actions"][:29])


def test_should_restore_camera_teacher_through_existing_qualification_without_model(tmp_path):
    raw = camera_teacher()
    path = tmp_path / "teacher.json"
    write_new(path, raw)
    result = case.read_camera_teacher(path)
    assert result.episode_id == f"{pick_cube_camera_context().context}-teacher"
    assert result.context == pick_cube_camera_context().context and result.role == "train"
    assert result.spec.observation_dim == 789
    assert result.transitions[0].observation[:21] == (0.0,) * 21
    assert result.transitions[0].observation[21:] == (0.75,) * 768
    assert result.transitions[-1].reward == 1.0
    raw["candidate_episode"]["transitions"][0]["observation"][0] = 100.0
    assert result.transitions[0].observation[0] == 0.0


@pytest.mark.parametrize(
    "damage",
    [
        "unqualified",
        "wrong_role",
        "wrong_seed",
        "missing_camera",
        "camera_type",
        "old_recipe",
        "recipe_type",
        "old_identity",
        "wrong_context",
        "wrong_spec",
        "observed_type",
        "final_type",
        "guard_metadata",
        "stages",
        "geometry_count",
        "work_type",
        "visual_work",
        "sparse_reward",
        "terminal_rgb",
        "optimizer",
    ],
)
def test_should_refuse_unqualified_camera_teacher_before_releasing_pairs(tmp_path, damage):
    raw = camera_teacher()
    if damage == "unqualified":
        raw["qualified"] = False
    elif damage == "wrong_role":
        raw["role"] = "probe_train"
    elif damage == "wrong_seed":
        raw["seed"] = 19
    elif damage == "missing_camera":
        del raw["camera_context"]
    elif damage == "camera_type":
        raw["camera_context"]["seed"] = 17.0
    elif damage == "old_recipe":
        raw["observation_recipe"] = teacher_receipt()["observation_recipe"]
    elif damage == "recipe_type":
        raw["observation_recipe"]["feature_dim"] = 789.0
    elif damage == "old_identity":
        raw["candidate_episode"]["episode_id"] = "pick-cube-task-teacher-17"
    elif damage == "wrong_context":
        raw["candidate_episode"]["context"] = "pick-cube"
    elif damage == "wrong_spec":
        raw["candidate_episode"]["spec"]["observation_dim"] = 789.0
    elif damage == "observed_type":
        raw["observed_success"] = 1
    elif damage == "final_type":
        raw["final_success"] = 1
    elif damage == "guard_metadata":
        raw["guard_seed"] = 31
    elif damage == "stages":
        raw["stages"][0] = "descend"
    elif damage == "geometry_count":
        raw["geometry_returns"] = 2.0
    elif damage == "work_type":
        raw["work"]["possible_failed_physics_steps"] = False
    elif damage == "visual_work":
        raw["visual_work"]["returned_frame_bytes"] -= 1
    elif damage == "sparse_reward":
        raw["candidate_episode"]["transitions"][0]["reward"] = 0.5
    elif damage == "terminal_rgb":
        raw["candidate_episode"]["transitions"][-1]["next_observation"][-1] = 1.01
    else:
        raw["optimizer_updates"] = False
    path = tmp_path / "bad-teacher.json"
    write_new(path, raw)
    with pytest.raises((AssertionError, ValueError, KeyError)):
        case.read_camera_teacher(path)


def test_should_reject_duplicate_camera_receipt_keys_before_qualification(tmp_path):
    path = tmp_path / "ambiguous.json"
    text = json.dumps(camera_teacher())
    text = text.replace('"qualified": true', '"qualified": false, "qualified": true', 1)
    path.write_text(text, encoding="utf8")
    with pytest.raises(ValueError, match="duplicate"):
        case.read_camera_teacher(path)


@pytest.mark.parametrize(
    "damage",
    [
        "camera_missing",
        "camera_extra",
        "camera_seed",
        "recipe",
        "context",
        "seed",
        "initial_observation",
    ],
)
def test_should_bind_camera_rollout_metadata_and_initial_observation_before_admission(
    tmp_path, damage
):
    teacher = camera_teacher()
    path = tmp_path / "teacher.json"
    write_new(path, teacher)
    episode = case.read_camera_teacher(path)
    raw = camera_rollout(teacher)
    if damage == "camera_missing":
        raw.pop("camera_context")
    elif damage == "camera_extra":
        raw["camera_context"]["tuned"] = False
    elif damage == "camera_seed":
        raw["camera_context"]["seed"] = 19
    elif damage == "recipe":
        raw["observation_recipe"] = teacher_receipt()["observation_recipe"]
    elif damage == "context":
        raw["context"] = "pick-cube"
    elif damage == "seed":
        raw["seed"] = True
    else:
        raw["episode"]["transitions"][0]["observation"][0] = 0.25
    with pytest.raises((AssertionError, ValueError, KeyError)):
        case.episode_from_trace(raw, episode, 17, pick_cube_camera_context())


@pytest.mark.parametrize(
    "damage", ["prefix", "no_rgb_change", "same_seed_wrong_context", "teacher_return_code"]
)
def test_should_gate_camera_teacher_and_same_reset_change_before_fit_or_baseline(
    synthetic_case, tmp_path, damage
):
    raw = synthetic_case["new"]
    if damage == "prefix":
        raw["candidate_episode"]["transitions"][0]["observation"][0] = 0.25
    elif damage == "no_rgb_change":
        for row in raw["candidate_episode"]["transitions"]:
            row["observation"][21:] = row["next_observation"][21:] = [0.25] * 768
    elif damage == "same_seed_wrong_context":
        raw["context"] = "pick-cube"
    else:
        synthetic_case["teacher_code"] = 1
    assert case.main() == 1
    audit = read_audit(tmp_path)
    assert audit["teacher_attempts"] == audit["teacher_returns"] == 1
    assert audit["fit_attempts"] == audit["warm_append_attempts"] == audit["runtime_attempts"] == 0
    assert (
        not synthetic_case["fits"]
        and not synthetic_case["trains"]
        and not synthetic_case["smoke_calls"]
    )


@pytest.mark.parametrize(
    "stage_name,attempts,charge",
    [("b-before.json", 1, 0), ("b-after.json", 2, 1), ("policy-rollout.json", 3, 1)],
)
def test_should_stop_after_failed_completed_stage_preserving_charge_and_receipt(
    synthetic_case, tmp_path, stage_name, attempts, charge
):
    original_argv = sys.argv
    synthetic_case["fail_at"] = stage_name
    assert case.main() == 1
    audit = read_audit(tmp_path)
    assert audit["complete"] is False
    assert audit["runtime_attempts"] == audit["runtime_returns"] == attempts
    assert len(synthetic_case["smoke_calls"]) == attempts
    assert synthetic_case["runtime"][0][2]["budget"].updates_completed == charge
    if charge:
        assert audit["paid_updates_after_return"] == 1
        assert audit["returned_applied"][0]["arrived_at"] == 2
    assert json.loads((tmp_path / stage_name).read_text())["error"] == "synthetic renderer failure"
    assert sys.argv is original_argv


@pytest.mark.parametrize(
    "field",
    [
        "visual_recipe_exact",
        "actor_version_exact",
        "actor_backend_exact",
        "actor_prediction_version_exact",
        "actor_distance_candidates_completed",
    ],
)
def test_should_validate_preceding_actor_trace_before_admitting_any_camera_labels(
    synthetic_case, tmp_path, field
):
    def corrupt(name, raw):
        if name == "b-before.json":
            raw[field] = False if field.endswith("exact") else raw[field] - 1

    synthetic_case["mutate"] = corrupt
    assert case.main() == 1
    assert len(synthetic_case["smoke_calls"]) == 1 and not synthetic_case["stream"]
    assert len(synthetic_case["trains"]) == 1
    assert read_audit(tmp_path)["runtime_append_attempts"] == 0


def test_should_refuse_changed_actual_serving_owner_before_admitting_camera_labels(
    synthetic_case, monkeypatch, tmp_path
):
    original = case.smoke.main

    def changed_owner(**options):
        code = original(**options)
        actor = options.get("actor")
        if actor is not None:
            state = actor._learner.snapshot_state()
            state.prototype_actions[0, 0] = 0.5
            actor._learner.restore_state(state)
        return code

    monkeypatch.setattr(case.smoke, "main", changed_owner)
    assert case.main() == 1
    assert len(synthetic_case["trains"]) == 1 and not synthetic_case["stream"]
    audit = read_audit(tmp_path)
    assert audit["runtime_append_attempts"] == 0 and "full learner state changed" in audit["error"]


@pytest.mark.parametrize("failure", ["postcommit_throw", "returned_receipt"])
def test_should_preserve_permanent_charge_and_original_receipt_after_update_boundary_failure(
    synthetic_case, monkeypatch, tmp_path, failure
):
    factory = case.ActorShadowRuntime

    def runtime(source, **options):
        created = factory(source, **options)
        ready = created.train_ready

        def changed(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if applied:
                if failure == "postcommit_throw":
                    raise RuntimeError("synthetic postcommit delivery failure")
                return (replace(applied[0], arrived_at=3),)
            return applied

        monkeypatch.setattr(created, "train_ready", changed)
        return created

    monkeypatch.setattr(case, "ActorShadowRuntime", runtime)
    assert case.main() == 1
    audit = read_audit(tmp_path)
    assert audit["complete"] is False and audit["paid_updates_after_return"] == 1
    assert audit["paid_replay_rows_after_return"] == 0
    assert len(audit["retained_applied"]) == 1
    assert audit["retained_applied"][0]["observed_at"] == 0
    assert (
        audit["retained_applied"][0]["arrived_at"]
        == audit["retained_applied"][0]["applied_at"]
        == 2
    )
    assert audit["retained_applied"][0]["diagnostic"] == dict(definition=case.OBJECTIVE, value=2.0)
    if failure == "postcommit_throw":
        assert audit["returned_applied"] == [] and "postcommit delivery failure" in audit["error"]
    else:
        assert audit["returned_applied"][0]["arrived_at"] == 3
    assert len(synthetic_case["smoke_calls"]) == 1 and len(synthetic_case["trains"]) == 2
    assert not list(tmp_path.glob("*.npz"))


def test_should_validate_full_preceding_success_before_final_empty_poll(
    synthetic_case, monkeypatch, tmp_path
):
    guard = case.guard_owners
    phases = []

    def changed_before_empty(source, runtime, initial, candidate, audit, phase):
        phases.append(phase)
        if phase == "before_empty_poll":
            state = runtime._candidate.snapshot_state()
            state.prototype_actions[0, 0] = 0.5
            runtime._candidate.restore_state(state)
        return guard(source, runtime, initial, candidate, audit, phase)

    monkeypatch.setattr(case, "guard_owners", changed_before_empty)
    assert case.main() == 1 and "before_empty_poll" in phases
    audit = read_audit(tmp_path)
    assert audit["paid_updates_after_return"] == 1
    assert len(audit["retained_applied"]) == 1
    assert audit["guards"][-1]["phase"] == "before_empty_poll"
    assert len(synthetic_case["smoke_calls"]) == 1
    assert [name for name, _, _, _, _ in synthetic_case["stream"]].count("train_ready") == 3
    assert not list(tmp_path.glob("*.npz"))


def test_should_refuse_receipt_replacement_during_empty_poll_before_archive_or_native(
    synthetic_case, monkeypatch, tmp_path
):
    factory = case.ActorShadowRuntime

    def runtime(source, **options):
        created = factory(source, **options)
        ready = created.train_ready

        def changed(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if not applied and options["budget"].updates_completed == 1:
                key, receipt = next(iter(created._inbox._applied.items()))
                created._inbox._applied[key] = replace(receipt, arrived_at=3)
            return applied

        monkeypatch.setattr(created, "train_ready", changed)
        return created

    monkeypatch.setattr(case, "ActorShadowRuntime", runtime)
    assert case.main() == 1
    audit = read_audit(tmp_path)
    assert audit["paid_updates_after_return"] == 1
    assert len(audit["retained_applied"]) == 1
    before = next(row for row in audit["guards"] if row["phase"] == "before_empty_poll")
    assert before["receipt_exact"] is True
    after = audit["guards"][-1]
    assert after["phase"] == "after_empty_poll" and after["receipt_exact"] is False
    assert after["observed_applied"][0]["arrived_at"] == 3
    assert audit["retained_applied"][0]["arrived_at"] == audit["applied"]["arrived_at"] == 2
    assert "preceding append receipt changed" in audit["error"]
    assert len(synthetic_case["smoke_calls"]) == 1
    assert [name for name, _, _, _, _ in synthetic_case["stream"]].count("train_ready") == 4
    assert not list(tmp_path.glob("*.npz"))


@pytest.mark.parametrize(
    "name",
    [
        "policy-rollout.json",
        "teacher-b.json",
        "b-before.json",
        "b-after.json",
        "prototype-states.npz",
    ],
)
def test_should_preserve_existing_artifacts_before_case_admission(synthetic_case, tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"preserved prior evidence")
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"preserved prior evidence"
    assert not synthetic_case["teacher_calls"] and not synthetic_case["trains"]
