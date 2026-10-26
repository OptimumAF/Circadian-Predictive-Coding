"""Fixed synthetic guard outcomes through the actual promotion controller, no SDK."""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from test_visual_native_context_case import teacher_receipt, rollout_receipt, visual_work, work
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
)
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.core import observation_standardization as standardization
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.infra.maniskill_continuous import ACTION_UNITS, ManiSkillVisualWork, ManiSkillWork
from src.infra.maniskill_task_teacher import TaskTeacherResult


case = importlib.import_module("artifacts.runs.native-guard-promotion-20261019.native_case")


def write(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf8")
    return path


def native_spec():
    return ContinuousActionSpec(789, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)


def frozen_statistics():
    # Independent population moments of the A observations0..28; pixel planes
    # and other public coordinates are constant with the prescribed scale1.
    return ObservationStandardization(
        (14.0,) + (0.0,) * 20 + (0.25,) * 768, (math.sqrt(70.0),) + (1.0,) * 788, 29
    )


def state_arrays(warm, new):
    means, scales = np.asarray(frozen_statistics().means), np.asarray(frozen_statistics().scales)
    a = warm["candidate_episode"]["transitions"]
    b = new["candidate_episode"]["transitions"]
    warm_features = (np.asarray([row["observation"] for row in a]) - means) / scales
    warm_actions = np.asarray([row["action"] for row in a])
    candidate_features = np.concatenate(
        [warm_features, (np.asarray([row["observation"] for row in b]) - means) / scales]
    )
    candidate_actions = np.concatenate([warm_actions, np.asarray([row["action"] for row in b])])
    return dict(
        warm_features=warm_features,
        warm_actions=warm_actions,
        candidate_features=candidate_features,
        candidate_actions=candidate_actions,
    )


def guard_result(seed, *, accepted=True):
    old = seed == 31
    rows = []
    for index in range(2):
        prefix = float(index + (0 if old else 100))
        before = (prefix,) + (0.0,) * 20 + (0.25,) * 768
        after = (prefix + 1,) + (0.0,) * 20 + (0.25,) * 768
        action = (
            (-1.0, index * 0.25, 0.0, 1.0)
            if old
            else ((1.0, index * 0.25, 0.0, 1.0) if accepted else (-1.0, 0.25, 0.0, 1.0))
        )
        rows.append(
            ContinuousTransition(index, before, action, float(index == 1), after, index == 1, False)
        )
    episode = ContinuousEpisode(
        f"pick-cube-guard-teacher-{seed}",
        f"pick-cube-inner-guard-{seed}",
        "inner_guard",
        native_spec(),
        tuple(rows),
    )
    return TaskTeacherResult(
        episode,
        True,
        True,
        1,
        2,
        2,
        2,
        ("align", "align"),
        ManiSkillWork(2, 2, 10, 0),
        None,
        None,
        pick_cube_rgb_recipe(),
        ManiSkillVisualWork(**visual_work(2)),
        seed,
        2,
    )


def prior_inputs(directory):
    """Make source files solely from the declared synthetic fixture, never archives."""
    directory.mkdir()
    warm, new = teacher_receipt(), teacher_receipt(19, 30)
    policy = NearestContinuousBCPolicy(native_spec(), frozen_statistics())
    metadata = dict(
        complete=True,
        policy=asdict(policy),
        source_publications=1,
        source_rows=29,
        candidate_publications=2,
        candidate_rows=59,
        permanent_update_charge=1,
        a_prefix_exact=True,
        frozen_a_statistics_exact=True,
        source_rows_ingested=59,
        fit_rows=29,
        fit_returns=1,
        fit_numeric_bytes=12624,
        warm_append_returns=1,
        warm_append_rows=29,
        runtime_append_returns=1,
        runtime_append_row_upper_bound=30,
        events=[
            dict(tick=0, candidate_rows=29, budget_updates=0, applied=0),
            dict(tick=1, candidate_rows=29, budget_updates=0, applied=0),
            dict(tick=2, candidate_rows=59, budget_updates=1, applied=1),
        ],
        applied=dict(
            sample_id="visual-B-pairs",
            episode_id="pick-cube-task-teacher-19",
            event_id="visual-B-label",
            actor_version="serving-0",
            learner_version="candidate-0",
            observed_at=0,
            arrived_at=2,
            applied_at=2,
            update_number=1,
            diagnostic=dict(definition="continuous_nearest_bc_added_pairs_v1", value=30),
        ),
        rollouts={"b-before.json": dict(sparse_return=0, any_success=False, final_success=False)},
    )
    write(directory / "rollout-audit.json", metadata)
    write(directory / "teacher-b.json", new)
    write(directory / "teacher-visual.json", warm)
    write(directory / "b-before.json", rollout_receipt(new, success=False))
    write(directory / "policy-rollout.json", rollout_receipt(warm, success=True))
    np.savez(directory / "prototype-states.npz", **state_arrays(warm, new))
    return warm, new, metadata


@pytest.fixture
def fixed_case(monkeypatch, tmp_path):
    source_dir = tmp_path / "synthetic-prior"
    warm, new, metadata = prior_inputs(source_dir)
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(case, "PRIOR_STAGE", source_dir)
    monkeypatch.setattr(case, "WARM_PATH", source_dir / "teacher-visual.json")
    observed = dict(
        warm=warm,
        new=new,
        metadata=metadata,
        source_dir=source_dir,
        accepted=True,
        teachers=[],
        smoke=[],
        getters=[],
        models=[],
        runtimes=[],
        controllers=[],
        bad_guard=None,
        fail_runtime=None,
        mutate_rollout=None,
        throw_guard=None,
        forks=[],
    )
    original_load, original_runtime = case.load_models, case.ActorShadowRuntime
    original_controller = case.ServingPromotionController
    original_getter = TaskTeacherResult.guard_episode
    original_fork = NumpyNearestContinuousBCLearner.fork

    def load(audit):
        values = original_load(audit)
        observed["models"].append(values)
        return values

    def runtime(source, **options):
        owner = original_runtime(source, **options)
        observed["runtimes"].append((owner, options))
        return owner

    def controller(*args, **options):
        owner = original_controller(*args, **options)
        observed["controllers"].append(owner)
        return owner

    def getter(result, *, now):
        observed["getters"].append((result.guard_seed, now))
        return original_getter(result, now=now)

    def fork(owner):
        child = original_fork(owner)
        observed["forks"].append((owner, child))
        return child

    def teacher_main():
        seed = int(sys.argv[sys.argv.index("--guard-seed") + 1])
        observed["teachers"].append(seed)
        assert sys.argv[sys.argv.index("--labels-arrived-at") + 1] == "2"
        assert "--visual-observation" in sys.argv
        result = guard_result(seed, accepted=observed["accepted"])
        raw = asdict(result)
        raw.update(
            qualified=result.qualified,
            role="inner_guard",
            seed=seed,
            context=result.candidate_episode.context,
            task="PickCube-v1",
            controller="pd_ee_delta_pos",
        )
        if observed["bad_guard"] is not None:
            observed["bad_guard"](seed, raw)
        path = Path(sys.argv[-1])
        with path.open("x", encoding="utf8") as handle:
            json.dump(raw, handle)
        if observed["throw_guard"] == seed:
            raise RuntimeError("synthetic unreturned guard collection")
        return 0 if raw["qualified"] else 1

    def smoke_main(*, actor, train_seed):
        name = Path(sys.argv[-1]).name
        observed["smoke"].append((actor, train_seed, name))
        assert type(actor) is PromotableActor and "--visual-observation" in sys.argv
        if observed["fail_runtime"] == train_seed:
            with Path(sys.argv[-1]).open("x", encoding="utf8") as handle:
                json.dump(
                    dict(complete=False, error="synthetic failed serving", work=work(0)), handle
                )
            return 1
        teacher = warm if train_seed == 17 else new
        raw = rollout_receipt(teacher, success=False)
        state = actor.serving_snapshot()
        rows = len(raw["episode"]["transitions"])
        for row in raw["episode"]["transitions"]:
            result = actor.predict(np.asarray([row["observation"]], dtype=np.float64))
            assert result.actor_version == "candidate-0"
            row["action"] = result.prediction[0].tolist()
        raw.update(
            policy="frozen_promotable_nearest_continuous_bc",
            actor_version=state.actor_version,
            actor_generation=state.generation,
            state_guard_scope="actual_promotable_actor_no_private_fork",
            actor_prediction_versions=[state.actor_version] * rows,
            actor_prediction_attempts=rows,
            actor_prediction_returns=rows,
            actor_distance_candidate_upper_bound=rows * 59,
            actor_distance_candidates_completed=rows * 59,
            actor_distance_feature_upper_bound=rows * 59 * 789,
            actor_distance_feature_values_completed=rows * 59 * 789,
            nearest_prediction_attempts=0,
            nearest_prediction_returns=0,
            nearest_distance_candidate_upper_bound=0,
            nearest_distance_candidates_completed=0,
            nearest_distance_feature_upper_bound=0,
            nearest_distance_feature_values_completed=0,
            nearest_selected_indices=[],
            prediction_attempts=rows,
            prediction_returns=rows,
            validated_prediction_rows=rows,
            validated_prediction_bytes=rows * 32,
            optimizer_updates=0,
            source_updates=2,
            source_training_rows=59,
            nearest_prototype_rows=59,
            nearest_prototype_numeric_bytes=59 * 793 * 8,
            source_update_definition="continuous_nearest_bc_added_pairs_v1",
        )
        for field in (
            "frozen_state_exact",
            "original_owner_exact",
            "actor_backend_exact",
            "actor_version_exact",
            "actor_generation_exact",
            "actor_configuration_exact",
            "actor_cache_exact",
            "actor_metadata_exact",
            "actor_last_cache_tick_exact",
            "actor_serving_bundle_exact",
            "actor_prediction_version_exact",
        ):
            raw[field] = True
        if observed["mutate_rollout"] is not None:
            observed["mutate_rollout"](train_seed, actor, raw)
        with Path(sys.argv[-1]).open("x", encoding="utf8") as handle:
            json.dump(raw, handle)
        return 0

    monkeypatch.setattr(case, "load_models", load)
    monkeypatch.setattr(case, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(case, "ServingPromotionController", controller)
    monkeypatch.setattr(TaskTeacherResult, "guard_episode", getter)
    monkeypatch.setattr(case.teacher_cli, "main", teacher_main)
    monkeypatch.setattr(case.smoke, "main", smoke_main)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "fork", fork)
    monkeypatch.setattr(
        standardization,
        "fit_observation_standardization",
        lambda *_a, **_k: pytest.fail("new protocol refitted observations"),
    )
    monkeypatch.setattr(
        NumpyNearestContinuousBCLearner,
        "train_batch",
        lambda *_a, **_k: pytest.fail("new protocol called a training update"),
    )
    monkeypatch.setattr(
        NumpyNearestContinuousBCLearner,
        "train_episode",
        lambda *_a, **_k: pytest.fail("new protocol called episode training"),
    )
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])
    return observed


def audit_at(directory):
    return json.loads((directory / "rollout-audit.json").read_text(encoding="utf8"))


def test_should_restore_exact_owned_models_and_publish_only_after_actual_fixed_guard_acceptance(
    fixed_case, tmp_path
):
    source_bytes = {path.name: path.read_bytes() for path in fixed_case["source_dir"].iterdir()}
    assert case.main() == 0
    audit = audit_at(tmp_path)
    report = json.loads((tmp_path / "guard-report.json").read_text())
    summary = json.loads((tmp_path / "policy-rollout.json").read_text())
    assert (
        audit["complete"] is True and audit["outcome"] == summary["outcome"] == "accepted_published"
    )
    assert audit["decision"]["accepted"] is report["decision"]["accepted"] is True
    assert fixed_case["teachers"] == [31, 37]
    assert fixed_case["getters"] == [(31, 1), (31, 2), (37, 1), (37, 2)]
    assert audit["prepare_attempts"] == audit["prepare_returns"] == 1
    assert audit["commit_attempts"] == audit["commit_returns"] == 1
    assert audit["runtime_attempts"] == audit["runtime_returns"] == 2
    assert [(seed, name) for _, seed, name in fixed_case["smoke"]] == [
        (19, "b-serving.json"),
        (17, "a-serving.json"),
    ]
    runtime, options = fixed_case["runtimes"][0]
    assert options["budget"].budget.max_training_updates == options["budget"].updates_completed == 0
    assert options["clock"].now() == 2 and runtime.applied_updates == ()
    assert all(actor is runtime.actor for actor, _, _ in fixed_case["smoke"])
    assert type(fixed_case["controllers"][0]) is ServingPromotionController
    assert len(fixed_case["forks"]) == 2
    assert audit["final_actor_version"] == "candidate-0" and audit["final_actor_generation"] == 1
    assert audit["original_permanent_update_charge"] == 1 and audit["current_training_updates"] == 0
    assert audit["restored_model_rows"] == [29, 59] and audit["restored_model_publications"] == [
        1,
        2,
    ]
    assert audit["retained_candidate_rows"] == 59
    for key in (
        "fit_calls",
        "append_calls",
        "optimizer_updates",
        "training_replay_rows",
        "probe_file_reads",
        "final_file_reads",
    ):
        assert audit[key] == 0
    for name, content in source_bytes.items():
        assert (fixed_case["source_dir"] / name).read_bytes() == content
    assert not list(tmp_path.glob("*.npz"))
    # Completed negative native task scores are preserved after a valid guard.
    assert all(
        row["final_success"] is False and row["sparse_return"] == 0 for row in audit["rollouts"]
    )


def test_should_complete_honest_rejection_with_full_measurements_and_no_serving_rollouts(
    fixed_case, tmp_path
):
    fixed_case["accepted"] = False
    assert case.main() == 0
    audit = audit_at(tmp_path)
    report = json.loads((tmp_path / "guard-report.json").read_text())
    assert audit["complete"] is True and audit["outcome"] == "rejected_unchanged"
    assert report["decision"]["accepted"] is False and report["decision"]["reasons"]
    assert report["evidence"]["candidate_new_utility"] < -0.25
    assert report["evidence"]["candidate_new_utility"] < report["evidence"]["actor_new_utility"]
    assert audit["prepare_attempts"] == audit["prepare_returns"] == 1
    assert audit["commit_attempts"] == audit["commit_returns"] == 0
    assert audit["runtime_attempts"] == audit["runtime_returns"] == 0
    assert not fixed_case["smoke"] and audit["rollouts"] == []
    assert audit["final_actor_version"] == "serving-0" and audit["final_actor_generation"] == 0
    assert audit["original_permanent_update_charge"] == 1 and audit["current_training_updates"] == 0
    runtime, _ = fixed_case["runtimes"][0]
    source, candidate, _, _ = fixed_case["models"][0]
    case.prior.same_state(source.snapshot_state(), runtime.actor.serving_snapshot().model_state)
    case.prior.same_state(candidate.snapshot_state(), runtime.candidate_snapshot().state)
    assert not (tmp_path / "b-serving.json").exists() and not (tmp_path / "a-serving.json").exists()


def test_should_preserve_frozen_metric_thresholds_disjoint_ids_and_hand_counted_guard_costs(
    fixed_case, tmp_path
):
    assert case.main() == 0
    audit = audit_at(tmp_path)
    report = json.loads((tmp_path / "guard-report.json").read_text())
    assert report["policy"] == dict(
        metric_id="pick_cube_guard_negative_mse_v1",
        resource_metric_id="nearest_numeric_payload_v1",
        min_new_utility=-0.25,
        min_new_gain=0.01,
        max_old_drop=0.02,
        max_prediction_seconds=0.1,
        max_resource_bytes=524288,
        allowed_actions=["bounded"],
    )
    assert (
        report["evidence"]["candidate_new_utility"]
        == report["evidence"]["candidate_old_utility"]
        == 0.0
    )
    assert report["evidence"]["actor_old_utility"] == 0.0
    assert report["evidence"]["actor_new_utility"] == pytest.approx(-1.0078125)
    assert report["evidence"]["candidate_resource_bytes"] == 59 * 793 * 8 + 789 * 16
    assert audit["guard_prediction_batches_returned"] == 4
    assert audit["guard_prediction_rows_derived"] == 8
    assert audit["guard_distance_candidates_derived"] == 352
    assert audit["guard_distance_feature_values_derived"] == 352 * 789
    assert audit["guard_prediction_bytes_derived"] == 256
    assert audit["source_validation_normalization_rows"] == 59
    assert audit["source_validation_normalization_values"] == 59 * 789
    training_ids = {tuple(value) for value in audit["training_ids"]}
    assert len(training_ids) == 60
    assert ("pick-cube-task-teacher-19", "visual-B-pairs") in training_ids
    assert {("pick-cube-task-teacher-17", str(index)) for index in range(29)} <= training_ids
    assert {("pick-cube-task-teacher-19", str(index)) for index in range(30)} <= training_ids
    old_ids, new_ids = map(
        lambda key: {tuple(value) for value in report[key]}, ("old_guard_keys", "new_guard_keys")
    )
    assert not old_ids & new_ids and not training_ids & (old_ids | new_ids)
    assert old_ids == {("pick-cube-guard-teacher-31", "0"), ("pick-cube-guard-teacher-31", "1")}
    assert new_ids == {("pick-cube-guard-teacher-37", "0"), ("pick-cube-guard-teacher-37", "1")}


@pytest.mark.parametrize("seed", [31, 37])
@pytest.mark.parametrize(
    "bad", ["unqualified", "role", "arrival", "context", "recipe", "work", "task", "controller"]
)
def test_should_stop_on_first_unqualified_or_malformed_guard_before_next_dependency(
    fixed_case, tmp_path, seed, bad
):
    def corrupt(observed_seed, raw):
        if observed_seed != seed:
            return
        if bad == "unqualified":
            raw.update(qualified=False, error="synthetic qualification failed")
        elif bad == "role":
            raw["role"] = "train"
        elif bad == "arrival":
            raw["labels_arrived_at"] = True
        elif bad == "context":
            raw["context"] = "pick-cube"
        elif bad == "recipe":
            raw["observation_recipe"]["pixel_divisor"] = 255
        elif bad in ("task", "controller"):
            raw[bad] = "wrong-native-source"
        else:
            raw["work"]["completed_steps"] -= 1

    fixed_case["bad_guard"] = corrupt
    assert case.main() == 1
    audit = audit_at(tmp_path)
    expected = [31] if seed == 31 else [31, 37]
    assert fixed_case["teachers"] == expected
    assert audit["teacher_attempts"] == audit["teacher_returns"] == len(expected)
    assert audit["prepare_attempts"] == audit["commit_attempts"] == audit["runtime_attempts"] == 0
    assert (tmp_path / f"teacher{seed}.json").exists()
    assert not (tmp_path / "guard-report.json").exists()


def test_should_retain_unreturned_guard_attempt_and_partial_receipt_without_retry(
    fixed_case, tmp_path
):
    fixed_case["throw_guard"] = 31
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["teacher_attempts"] == 1 and audit["teacher_returns"] == 0
    assert "synthetic unreturned guard collection" in audit["error"]
    assert fixed_case["teachers"] == [31] and not fixed_case["smoke"]
    assert (tmp_path / "teacher31.json").exists()


def test_should_stop_if_arrival_getter_releases_guard_early(fixed_case, monkeypatch, tmp_path):
    monkeypatch.setattr(
        TaskTeacherResult, "guard_episode", lambda result, **_: result.candidate_episode
    )
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert "guard labels released early" in audit["error"]
    assert fixed_case["teachers"] == [31]
    assert audit["prepare_attempts"] == audit["runtime_attempts"] == 0


@pytest.mark.parametrize("bad", ["final", "probe", "overlap", "future"])
def test_should_preserve_real_evaluator_isolation_before_any_extra_prediction(
    fixed_case, tmp_path, bad
):
    fixed_case["accepted"] = False
    assert case.main() == 0
    controller = fixed_case["controllers"][0]
    runtime = fixed_case["runtimes"][0][0]
    new = case.guard_batch(guard_result(37).guard_episode(now=2))
    old = case.guard_batch(guard_result(31).guard_episode(now=2))
    if bad in ("final", "probe"):
        object.__setattr__(new, "role", "final_test" if bad == "final" else "probe_test")
    elif bad == "overlap":
        new = replace(new, sample_keys=(("pick-cube-task-teacher-17", "0"), new.sample_keys[1]))
    else:
        new = replace(new, labels_arrived_at=3)
    audit = audit_at(tmp_path)
    training_ids = frozenset(tuple(row) for row in audit["training_ids"])
    before = case.pickle.dumps(runtime.actor.serving_snapshot())
    with pytest.raises(ValueError):
        controller.prepare(runtime, new, old, now=2, training_ids=training_ids)
    assert case.pickle.dumps(runtime.actor.serving_snapshot()) == before
    assert runtime.applied_updates == ()
    assert not fixed_case["smoke"]


@pytest.mark.parametrize("seed,returns", [(19, 1), (17, 2)])
def test_should_retain_commit_and_stop_after_failed_native_serving_receipt(
    fixed_case, tmp_path, seed, returns
):
    fixed_case["fail_runtime"] = seed
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["commit_attempts"] == audit["commit_returns"] == 1
    assert audit["runtime_attempts"] == audit["runtime_returns"] == returns
    assert len(fixed_case["smoke"]) == returns
    assert (
        audit["complete"] is False and "actual serving rollout did not complete" in audit["error"]
    )
    actor = fixed_case["runtimes"][0][0].actor
    assert actor.version == "candidate-0" and actor.serving_snapshot().generation == 1
    assert fixed_case["runtimes"][0][1]["budget"].updates_completed == 0


@pytest.mark.parametrize(
    "field",
    [
        "actor_generation_exact",
        "actor_serving_bundle_exact",
        "actor_prediction_version_exact",
        "actor_distance_candidates_completed",
    ],
)
def test_should_refuse_inconsistent_serving_receipts_before_second_rollout(
    fixed_case, tmp_path, field
):
    def corrupt(seed, actor, raw):
        if seed == 19:
            raw[field] = False if field.endswith("exact") else raw[field] - 1

    fixed_case["mutate_rollout"] = corrupt
    assert case.main() == 1
    assert len(fixed_case["smoke"]) == 1
    assert audit_at(tmp_path)["commit_returns"] == 1


@pytest.mark.parametrize("kind", ["generation", "metadata"])
def test_should_refuse_unobserved_actor_bundle_change_after_returned_smoke(
    fixed_case, tmp_path, kind
):
    def corrupt(seed, actor, raw):
        if seed != 19:
            return
        slot = actor._slot
        if kind == "generation":
            actor._slot = replace(slot, generation=2)
        else:
            actor._slot = replace(slot, bundle=replace(slot.bundle, metadata={"unobserved": True}))

    fixed_case["mutate_rollout"] = corrupt
    assert case.main() == 1
    assert len(fixed_case["smoke"]) == 1
    assert audit_at(tmp_path)["commit_returns"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        "complete",
        "charge",
        "charge_type",
        "publication_type",
        "rows",
        "policy_role",
        "stats_rows",
        "early_charge",
        "arrival_history",
        "historical_diagnostic",
        "prefix_witness",
        "prefix",
        "nonfinite",
        "array_dtype",
        "new_labels",
        "new_feature_binding",
    ],
)
def test_should_refuse_bad_prior_evidence_before_any_new_teacher_or_prediction(
    fixed_case, tmp_path, bad
):
    metadata = deepcopy(fixed_case["metadata"])
    path = fixed_case["source_dir"]
    if bad == "complete":
        metadata["complete"] = False
    elif bad == "charge":
        metadata["permanent_update_charge"] = 0
    elif bad == "charge_type":
        metadata["permanent_update_charge"] = True
    elif bad == "publication_type":
        metadata["source_publications"] = True
    elif bad == "rows":
        metadata["candidate_rows"] = 58
    elif bad == "policy_role":
        metadata["policy"]["training_role"] = "probe_train"
    elif bad == "stats_rows":
        metadata["policy"]["observation_standardization"]["fitted_rows"] = 30
    elif bad == "early_charge":
        metadata["events"][1]["budget_updates"] = 1
    elif bad == "arrival_history":
        metadata["applied"]["arrived_at"] = 1
    elif bad == "historical_diagnostic":
        metadata["applied"]["diagnostic"]["value"] = 29
    elif bad == "prefix_witness":
        metadata["a_prefix_exact"] = False
    else:
        arrays = state_arrays(fixed_case["warm"], fixed_case["new"])
        if bad == "prefix":
            arrays["candidate_features"][0, 0] += 1
        elif bad == "nonfinite":
            arrays["candidate_features"][-1, 0] = np.inf
        elif bad == "new_labels":
            arrays["candidate_actions"][-1, 0] = 0.5
        elif bad == "new_feature_binding":
            arrays["candidate_features"][-1, 0] += 1
        else:
            arrays["candidate_actions"] = arrays["candidate_actions"].astype(np.float32)
        np.savez(path / "prototype-states.npz", **arrays)
    write(path / "rollout-audit.json", metadata)
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["teacher_attempts"] == audit["prepare_attempts"] == audit["runtime_attempts"] == 0
    assert not fixed_case["teachers"] and not fixed_case["smoke"]


@pytest.mark.parametrize(
    "name",
    [
        "policy-rollout.json",
        "teacher31.json",
        "teacher37.json",
        "guard-report.json",
        "b-serving.json",
        "a-serving.json",
        "rollout-audit.json",
    ],
)
def test_should_refuse_existing_evidence_before_loading_or_new_work(fixed_case, tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"preserved previous attempt\n")
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"preserved previous attempt\n"
    assert not fixed_case["models"] and not fixed_case["teachers"]
