"""Synthetic receipt boundaries around the real bounded visual stream; no SDK."""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from src.core.actor_ports import VersionedState
from src.app.actor_shadow import StableActor
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.infra.maniskill_continuous import ACTION_UNITS


case = importlib.import_module("artifacts.runs.visual-context-return-20261019.native_case")


def json_recipe():
    return json.loads(json.dumps(asdict(pick_cube_rgb_recipe())))


def visual_work(rows):
    return dict(
        observation_attempts=rows + 1,
        observation_returns=rows + 1,
        input_frame_bytes_upper_bound=(rows + 1) * 49152,
        feature_values_upper_bound=(rows + 1) * 789,
        returned_frame_bytes=(rows + 1) * 49152,
        returned_feature_values=(rows + 1) * 789,
        returned_feature_numeric_bytes=(rows + 1) * 6312,
        current_owned_frame_bytes=49152,
        frame_export_copies=0,
        frame_export_bytes=0,
    )


def work(rows):
    return dict(
        step_attempts=rows,
        completed_steps=rows,
        completed_control_physics_steps=5 * rows,
        possible_failed_physics_steps=0,
    )


def teacher_receipt(seed=17, rows=29):
    """Declared synthetic feature packets, deliberately not simulator evidence."""
    context = "pick-cube" if seed == 17 else "pick-cube-train-19"
    offset = 0.0 if seed == 17 else 100.0
    observations = [[offset + index] + [0.0] * 20 + [0.25] * 768 for index in range(rows + 1)]
    transitions = [
        dict(
            step=index,
            observation=observations[index].copy(),
            action=[(-1.0 if seed == 17 else 1.0), (index % 3) * 0.25, 0.0, 1.0],
            reward=float(index == rows - 1),
            next_observation=observations[index + 1].copy(),
            terminated=index == rows - 1,
            truncated=False,
        )
        for index in range(rows)
    ]
    return dict(
        fixture_kind="synthetic_not_native_evidence",
        qualified=True,
        final_success=True,
        error=None,
        seed=seed,
        context=context,
        role="train",
        task="PickCube-v1",
        controller="pd_ee_delta_pos",
        observation_recipe=json_recipe(),
        visual_work=visual_work(rows),
        work=work(rows),
        candidate_episode=dict(
            episode_id=f"pick-cube-task-teacher-{seed}",
            role="train",
            context=context,
            spec=dict(
                observation_dim=789,
                action_units=list(ACTION_UNITS),
                lower_bounds=[-1.0] * 4,
                upper_bounds=[1.0] * 4,
            ),
            transitions=transitions,
        ),
    )


def rollout_receipt(teacher, *, success=False):
    seed = teacher["seed"]
    episode = deepcopy(teacher["candidate_episode"])
    episode["episode_id"] = f"pick-cube-smoke-{seed}"
    rows = len(episode["transitions"])
    for index, row in enumerate(episode["transitions"]):
        row["reward"] = float(success and index == rows - 1)
    return dict(
        complete=True,
        error=None,
        role="train",
        seed=seed,
        context=teacher["context"],
        observation_recipe=json_recipe(),
        visual_recipe_exact=True,
        episode=episode,
        sparse_return=int(success),
        any_success=success,
        final_success=success,
        visual_work=visual_work(rows),
        work=work(rows),
        elapsed_seconds=0.125,
    )


def write_new(path, value):
    with path.open("x", encoding="utf8") as handle:
        json.dump(value, handle, allow_nan=False)


@pytest.fixture
def synthetic_case(monkeypatch, tmp_path):
    warm, new = teacher_receipt(), teacher_receipt(19, 2)
    warm_path, reference_path = tmp_path / "accepted-a.json", tmp_path / "accepted-reference.json"
    write_new(warm_path, warm)
    write_new(reference_path, rollout_receipt(warm, success=True))
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
        observed["runtime"].append((source, owner, options))
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
        assert not observed["fits"] and not observed["runtime"]
        assert sys.argv[1:4] == ["--visual-observation", "--train-seed", "19"]
        write_new(Path(sys.argv[-1]), observed["new"])
        return observed["teacher_code"]

    def smoke_main(*, train_seed, actor=None, learner=None):
        name = Path(sys.argv[-1]).name
        owner = actor if actor is not None else learner
        observed["smoke_calls"].append((name, train_seed, actor, learner))
        assert "--visual-observation" in sys.argv
        if observed["fail_at"] == name:
            write_new(Path(sys.argv[-1]), dict(complete=False, error="synthetic renderer failure"))
            return 1
        source = observed["warm"] if train_seed == 17 else observed["new"]
        raw = rollout_receipt(source, success=observed["successes"].get(name, False))
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
            if actor
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
            nearest_prediction_attempts=0 if actor else rows,
            nearest_prediction_returns=0 if actor else rows,
            nearest_selected_indices=indices,
            state_guard_scope="actual_serving_actor_no_private_fork"
            if actor
            else "private_frozen_fork",
            actor_version="serving-0" if actor else None,
            actor_version_exact=True,
            actor_backend_exact=True,
            actor_prediction_version_exact=True,
            actor_prediction_versions=["serving-0"] * rows if actor else [],
            actor_prediction_attempts=rows if actor else 0,
            actor_prediction_returns=rows if actor else 0,
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
        write_new(Path(sys.argv[-1]), raw)
        return 0

    monkeypatch.setattr(case, "fit_training_standardization", fit)
    monkeypatch.setattr(case.NumpyNearestContinuousBCLearner, "train_batch", train)
    monkeypatch.setattr(case, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(case.teacher_cli, "main", teacher_main)
    monkeypatch.setattr(case.smoke, "main", smoke_main)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])
    return observed


def read_audit(path):
    return json.loads((path / "rollout-audit.json").read_text(encoding="utf8"))


def test_should_execute_fixed_actual_actor_delayed_append_and_fresh_return_without_refit(
    synthetic_case, tmp_path
):
    observed = synthetic_case
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert case.main() == 0
    audit = read_audit(tmp_path)
    assert audit["complete"] is True and "error" not in audit
    assert len(observed["teacher_calls"]) == len(observed["fits"]) == 1
    assert len(observed["fits"][0].transitions) == 29
    assert [(name, seed) for name, seed, _, _ in observed["smoke_calls"]] == [
        ("b-before.json", 19),
        ("b-after.json", 19),
        ("policy-rollout.json", 17),
    ]
    source, runtime, options = observed["runtime"][0]
    assert observed["smoke_calls"][0][2] is runtime.actor
    assert type(runtime.actor) is StableActor
    assert all(row[2] is None for row in observed["smoke_calls"][1:])
    assert options["max_experiences"] == 1 and options["max_consolidations"] == 0
    assert options["budget"].updates_completed == 1 and options["clock"].now() == 2
    assert len(observed["trains"]) == 2
    assert [rows for _, rows in observed["trains"]] == [29, 2]
    assert observed["trains"][0][0] is source
    assert observed["trains"][1][0] is not source
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
    assert all(all(row[key] for key in ("source", "actor", "candidate")) for row in audit["guards"])
    assert audit["source_publications"] == 1 and audit["candidate_publications"] == 2
    assert audit["source_rows_ingested"] == audit["candidate_rows"] == 31
    assert audit["source_rows"] == 29 and audit["permanent_update_charge"] == 1
    assert audit["a_prefix_exact"] and audit["frozen_a_statistics_exact"]
    for key in (
        "optimizer_updates",
        "replayed_training_rows",
        "probe_file_reads",
        "final_file_reads",
        "promotion_calls",
    ):
        assert audit[key] == 0
    for name, payload in before.items():
        assert (tmp_path / name).read_bytes() == payload


def test_should_preserve_negative_completed_endpoints_and_frozen_metric_directions(
    synthetic_case, tmp_path
):
    synthetic_case["successes"]["b-before.json"] = True
    assert case.main() == 0
    audit = read_audit(tmp_path)
    assert audit["b_after_minus_before"] == dict(
        sparse_return=-1.0, any_success=-1.0, final_success=-1.0
    )
    assert audit["a_reference_minus_return_retention_loss"] == dict(
        sparse_return=1.0, any_success=1.0, final_success=1.0
    )
    assert audit["rollouts"]["b-after.json"]["final_success"] is False
    assert audit["rollouts"]["policy-rollout.json"]["final_success"] is False


def test_should_account_for_789d_distances_current_frames_exposures_and_one_bounded_state_archive(
    synthetic_case, tmp_path
):
    assert case.main() == 0
    audit = read_audit(tmp_path)
    assert audit["prediction_rows"] == 33 and audit["prediction_bytes"] == 1056
    assert audit["distance_candidates"] == 2 * 29 + (2 + 29) * 31
    assert audit["distance_feature_values"] == (2 * 29 + 31 * 31) * 789
    assert audit["control_attempts"] == 35 and audit["known_control_physics_steps"] == 175
    assert audit["visual_projections"] == 39
    assert audit["raw_frame_input_bytes"] == 39 * 49152
    assert audit["returned_feature_values"] == 39 * 789
    assert audit["returned_feature_numeric_bytes"] == 39 * 6312
    assert audit["normalization_calls"] == 35 and audit["normalization_rows"] == 64
    assert audit["warm_prototype_numeric_bytes"] == 29 * 793 * 8
    assert audit["candidate_prototype_numeric_bytes"] == 31 * 793 * 8
    assert audit["prototype_growth_numeric_bytes"] == audit["inbox_numeric_bytes"] == 2 * 793 * 8
    assert audit["fit_rows"] == 29 and audit["fit_numeric_bytes"] == 12624
    archives = list(tmp_path.glob("*.npz"))
    assert len(archives) == 1 and archives[0].stat().st_size <= 1024**2
    with np.load(archives[0], allow_pickle=False) as saved:
        assert set(saved.files) == {
            "warm_features",
            "warm_actions",
            "candidate_features",
            "candidate_actions",
        }
        assert saved["warm_features"].shape == (29, 789)
        assert saved["candidate_features"].shape == (31, 789)
        np.testing.assert_array_equal(saved["warm_features"], saved["candidate_features"][:29])
        np.testing.assert_array_equal(saved["warm_actions"], saved["candidate_actions"][:29])


@pytest.mark.parametrize(
    "failure", ["unqualified", "return_code", "role", "recipe", "context", "sparse_reward"]
)
def test_should_stop_unqualified_or_corrupt_b_before_any_fit_or_baseline(
    synthetic_case, tmp_path, failure
):
    raw = synthetic_case["new"]
    if failure == "unqualified":
        raw.update(qualified=False, error="synthetic renderer unavailable")
    elif failure == "return_code":
        synthetic_case["teacher_code"] = 1
    elif failure == "role":
        raw["role"] = "final_test"
    elif failure == "recipe":
        raw["observation_recipe"]["pixel_divisor"] = 255
    elif failure == "context":
        raw["context"] = "pick-cube"
    else:
        raw["candidate_episode"]["transitions"][0]["reward"] = 0.5
    assert case.main() == 1
    audit = read_audit(tmp_path)
    assert audit["complete"] is False and audit["teacher_attempts"] == audit["teacher_returns"] == 1
    assert audit["fit_attempts"] == audit["warm_append_attempts"] == audit["runtime_attempts"] == 0
    assert (
        not synthetic_case["fits"]
        and not synthetic_case["runtime"]
        and not synthetic_case["smoke_calls"]
    )
    assert (tmp_path / "teacher-b.json").exists()
    assert not list(tmp_path.glob("*.npz"))


def test_should_preserve_unreturned_teacher_attempt_without_fitting_or_retry(
    synthetic_case, monkeypatch, tmp_path
):
    calls = []

    def failed_teacher():
        calls.append(tuple(sys.argv))
        raise RuntimeError("synthetic teacher constructor failure")

    monkeypatch.setattr(case.teacher_cli, "main", failed_teacher)
    assert case.main() == 1
    audit = read_audit(tmp_path)
    assert audit["teacher_attempts"] == 1 and audit["teacher_returns"] == 0
    assert "synthetic teacher constructor failure" in audit["error"]
    assert len(calls) == 1 and audit["fit_attempts"] == audit["runtime_attempts"] == 0
    assert not synthetic_case["fits"] and not synthetic_case["smoke_calls"]


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_warm_rows",
        "warm_recipe",
        "reference_context",
        "reference_not_complete",
        "reference_negative",
    ],
)
def test_should_validate_reused_sources_before_collecting_new_data(
    synthetic_case, tmp_path, failure
):
    if failure.startswith("warm") or failure == "wrong_warm_rows":
        payload = teacher_receipt(rows=28) if failure == "wrong_warm_rows" else teacher_receipt()
        if failure == "warm_recipe":
            payload["observation_recipe"] = None
        path = case.WARM_PATH
    else:
        payload = rollout_receipt(synthetic_case["warm"], success=True)
        if failure == "reference_context":
            payload["context"] = "pick-cube-train-19"
        elif failure == "reference_not_complete":
            payload["complete"] = False
        else:
            payload = rollout_receipt(synthetic_case["warm"], success=False)
        path = case.REFERENCE_PATH
    path.write_text(json.dumps(payload), encoding="utf8")
    assert case.main() == 1
    audit = read_audit(tmp_path)
    assert audit["teacher_attempts"] == audit["fit_attempts"] == 0
    assert not synthetic_case["teacher_calls"] and not synthetic_case["fits"]


@pytest.mark.parametrize(
    "name,attempts,charged",
    [("b-before.json", 1, 0), ("b-after.json", 2, 1), ("policy-rollout.json", 3, 1)],
)
def test_should_retain_failed_partial_receipt_and_stop_dependent_native_calls(
    synthetic_case, tmp_path, name, attempts, charged
):
    original_argv = sys.argv
    synthetic_case["fail_at"] = name
    assert case.main() == 1
    audit = read_audit(tmp_path)
    assert audit["complete"] is False and "native runtime did not complete" in audit["error"]
    assert audit["runtime_attempts"] == audit["runtime_returns"] == attempts
    assert len(synthetic_case["smoke_calls"]) == attempts
    assert synthetic_case["runtime"][0][2]["budget"].updates_completed == charged
    assert json.loads((tmp_path / name).read_text())["error"] == "synthetic renderer failure"
    assert sys.argv is original_argv


@pytest.mark.parametrize(
    "field",
    [
        "visual_recipe_exact",
        "frozen_state_exact",
        "actor_version_exact",
        "actor_prediction_version_exact",
        "actor_backend_exact",
    ],
)
def test_should_refuse_baseline_guard_claim_failures_before_admitting_labels(
    synthetic_case, tmp_path, field
):
    synthetic_case["mutate"] = lambda name, raw: raw.update({field: False})
    assert case.main() == 1
    assert len(synthetic_case["smoke_calls"]) == 1
    assert not synthetic_case["stream"]
    assert synthetic_case["runtime"][0][2]["budget"].updates_completed == 0


@pytest.mark.parametrize(
    "field",
    [
        "prediction_returns",
        "validated_prediction_bytes",
        "actor_distance_candidates_completed",
        "actor_distance_feature_values_completed",
    ],
)
def test_should_refuse_incomplete_baseline_prediction_accounting_before_append(
    synthetic_case, tmp_path, field
):
    def corrupt(name, raw):
        if name == "b-before.json":
            raw[field] -= 1

    synthetic_case["mutate"] = corrupt
    assert case.main() == 1
    assert not synthetic_case["stream"]
    assert read_audit(tmp_path)["runtime_append_attempts"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "nearest_distance_candidate_upper_bound",
        "nearest_distance_feature_upper_bound",
        "nearest_distance_candidates_completed",
        "nearest_distance_feature_values_completed",
    ],
)
def test_should_refuse_unreturned_nearest_report_work_on_actual_actor_route(
    synthetic_case, tmp_path, field
):
    def corrupt(name, raw):
        if name == "b-before.json":
            raw[field] = 1

    synthetic_case["mutate"] = corrupt
    assert case.main() == 1
    assert not synthetic_case["stream"]
    assert read_audit(tmp_path)["runtime_append_attempts"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "returned_feature_values",
        "returned_frame_bytes",
        "current_owned_frame_bytes",
        "frame_export_copies",
    ],
)
def test_should_refuse_corrupt_visual_costs_before_more_native_work(
    synthetic_case, tmp_path, field
):
    def corrupt(name, raw):
        if name == "b-before.json":
            raw["visual_work"][field] += 1

    synthetic_case["mutate"] = corrupt
    assert case.main() == 1
    assert not synthetic_case["stream"] and len(synthetic_case["smoke_calls"]) == 1


def test_should_reject_changed_actual_actor_state_after_baseline_before_new_labels(
    synthetic_case, monkeypatch, tmp_path
):
    original = case.smoke.main

    def mutate_actor(**kwargs):
        code = original(**kwargs)
        if kwargs.get("actor") is not None:
            actor = kwargs["actor"]
            state = actor._learner.snapshot_state()
            state.prototype_actions[0, 0] = 0.5
            actor._learner.restore_state(state)
        return code

    monkeypatch.setattr(case.smoke, "main", mutate_actor)
    assert case.main() == 1
    assert "full learner state changed" in read_audit(tmp_path)["error"]
    assert not synthetic_case["stream"]


def test_should_require_typed_full_snapshot_even_for_equal_valued_corruption(
    synthetic_case, monkeypatch, tmp_path
):
    original = case.guard_owners

    def corrupt_counter(source, runtime, initial, candidate, audit, phase):
        if phase == "after_baseline_before_labels":
            snapshot = runtime.actor.snapshot_state

            def invalid_snapshot():
                tagged = snapshot()
                return VersionedState(
                    tagged.actor_version, replace(tagged.state, completed_updates=True)
                )

            monkeypatch.setattr(runtime.actor, "snapshot_state", invalid_snapshot)
        return original(source, runtime, initial, candidate, audit, phase)

    monkeypatch.setattr(case, "guard_owners", corrupt_counter)
    assert case.main() == 1
    assert not synthetic_case["stream"]
    assert read_audit(tmp_path)["guards"][-1]["actor"] is False


@pytest.mark.parametrize("bad_index", [-1, 31, True, 0.0])
def test_should_refuse_non_native_or_out_of_range_candidate_indices(
    synthetic_case, tmp_path, bad_index
):
    def corrupt(name, raw):
        if name == "b-after.json":
            raw["nearest_selected_indices"][0] = bad_index
            if bad_index == -1:
                # Match the final action deliberately: Python negative indexing
                # must not turn a corrupt work receipt into accepted evidence.
                raw["episode"]["transitions"][0]["action"] = synthetic_case["new"][
                    "candidate_episode"
                ]["transitions"][-1]["action"].copy()

    synthetic_case["mutate"] = corrupt
    assert case.main() == 1
    assert len(synthetic_case["smoke_calls"]) == 2
    assert synthetic_case["runtime"][0][2]["budget"].updates_completed == 1


def test_should_reject_mismatched_candidate_action_before_old_context_return(
    synthetic_case, tmp_path
):
    def corrupt(name, raw):
        if name == "b-after.json":
            raw["episode"]["transitions"][0]["action"][0] = -0.5

    synthetic_case["mutate"] = corrupt
    assert case.main() == 1
    assert len(synthetic_case["smoke_calls"]) == 2
    assert read_audit(tmp_path)["runtime_append_returns"] == 1


def test_should_keep_all_owner_and_readout_snapshots_independent_after_completion(
    synthetic_case, tmp_path
):
    assert case.main() == 0
    source, runtime, _ = synthetic_case["runtime"][0]
    actor = runtime.actor.snapshot_state().state
    candidate = runtime.candidate_snapshot().state
    detached = synthetic_case["smoke_calls"][1][3]
    for owner, rows in ((source, 29), (detached, 31)):
        snapshot = owner.snapshot_state()
        snapshot.prototype_features[:] = 999.0
        snapshot.prototype_actions[:] = 0.0
        assert owner.snapshot_state().training_examples == rows
        assert not np.any(owner.snapshot_state().prototype_features == 999.0)
    detached_state = detached.snapshot_state()
    detached_state.prototype_actions[0] = 0.0
    detached.restore_state(detached_state)
    case.same_state(actor, runtime.actor.snapshot_state().state)
    case.same_state(candidate, runtime.candidate_snapshot().state)
    assert not np.shares_memory(actor.prototype_features, candidate.prototype_features)
    assert not np.shares_memory(actor.prototype_actions, candidate.prototype_actions)


def test_should_reserve_audit_exclusively_before_any_new_work(synthetic_case, tmp_path):
    path = tmp_path / "rollout-audit.json"
    path.write_bytes(b"preserved prior attempt\n")
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"preserved prior attempt\n"
    assert not synthetic_case["teacher_calls"] and not synthetic_case["fits"]


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
def test_should_preserve_existing_evidence_and_refuse_before_new_native_work(
    synthetic_case, tmp_path, name
):
    path = tmp_path / name
    path.write_bytes(b"preserved failed or accepted evidence\n")
    with pytest.raises(FileExistsError, match="existing native evidence"):
        case.main()
    assert path.read_bytes() == b"preserved failed or accepted evidence\n"
    assert not synthetic_case["teacher_calls"] and not synthetic_case["fits"]
    assert not (tmp_path / "rollout-audit.json").exists()


@pytest.mark.parametrize(
    "name",
    ["teacher-b.json", "b-before.json", "b-after.json", "prototype-states.npz", "arbitrary.json"],
)
def test_should_refuse_primary_output_aliases_before_new_work(
    synthetic_case, monkeypatch, tmp_path, name
):
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / name)])
    with pytest.raises(AssertionError, match="fixed primary receipt"):
        case.main()
    assert not synthetic_case["teacher_calls"] and not synthetic_case["fits"]
    assert not (tmp_path / "rollout-audit.json").exists()


def test_should_keep_output_inside_owned_stage_before_any_new_work(
    synthetic_case, monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path.parent / "outside.json")])
    with pytest.raises(AssertionError, match="output must remain"):
        case.main()
    assert not synthetic_case["teacher_calls"] and not (tmp_path / "rollout-audit.json").exists()
