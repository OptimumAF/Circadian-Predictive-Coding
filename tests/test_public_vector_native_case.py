"""Synthetic A29 restoration and public21 actor boundaries; no fit/append/SGD.

One shared A29/camera2 dataset supplies31 unique fixture rows per invocation.
Copies/restorations reuse those identities; candidate archive arrays are never read.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
import math
import sys

import numpy as np
import pytest

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_camera_context import pick_cube_camera_context
from src.core.pick_cube_public_vectors import public_vector_projection
from test_camera_context_return_case import camera_teacher, camera_rollout
from test_visual_native_context_case import teacher_receipt, write_new


case = importlib.import_module("artifacts.runs.public-vector-native-20261019.native_case")


def episode_from_teacher(raw):
    value = raw["candidate_episode"]
    spec = value["spec"]
    return ContinuousEpisode(
        value["episode_id"],
        value["context"],
        value["role"],
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


@pytest.fixture(scope="module")
def source_data():
    warm_raw, camera_raw = teacher_receipt(), camera_teacher()
    warm = episode_from_teacher(warm_raw)
    # For the29 equally spaced coordinates0..28, mean14/variance70 are analytic.
    means = [14.0] + [0.0] * 20 + [0.25] * 768
    scales = [math.sqrt(70.0)] + [1.0] * 788
    normalized = np.zeros((29, 789), dtype=np.float64)
    normalized[:, 0] = [(index - 14.0) / math.sqrt(70.0) for index in range(29)]
    actions = np.asarray([row.action for row in warm.transitions], dtype=np.float64)
    policy = dict(
        spec=asdict(warm.spec),
        observation_standardization=dict(means=means, scales=scales, fitted_rows=29),
        max_prototypes=256,
        training_role="train",
    )
    camera_context = pick_cube_camera_context().context
    accepted = dict(
        complete=True,
        source_publications=1,
        source_rows=29,
        candidate_publications=2,
        candidate_rows=31,
        permanent_update_charge=1,
        policy=policy,
        rollouts={
            "b-before.json": dict(
                seed=17,
                context=camera_context,
                sparse_return=0.0,
                any_success=False,
                final_success=False,
            ),
            "b-after.json": dict(
                seed=17,
                context=camera_context,
                sparse_return=1.0,
                any_success=True,
                final_success=True,
            ),
            "policy-rollout.json": dict(
                seed=17,
                context="pick-cube",
                sparse_return=1.0,
                any_success=True,
                final_success=True,
            ),
        },
    )
    return dict(
        warm_raw=warm_raw,
        camera_raw=camera_raw,
        warm=warm,
        accepted=json.loads(json.dumps(accepted)),
        normalized=normalized,
        actions=actions,
    )


@pytest.fixture
def archived_source(source_data, tmp_path):
    path = tmp_path / "prototype-states.npz"
    np.savez(
        path,
        warm_features=source_data["normalized"],
        warm_actions=source_data["actions"],
        candidate_features=np.empty((0, 789), dtype=np.float64),
        candidate_actions=np.empty((0, 4), dtype=np.float64),
    )
    return path


def test_should_restore_exact_a_prefix_statistics_prototypes_and_owned_state_without_learning(
    source_data, archived_source, monkeypatch
):
    accepted = deepcopy(source_data["accepted"])
    original = deepcopy(accepted)
    monkeypatch.setattr(
        case.NumpyNearestContinuousBCLearner,
        "train_batch",
        lambda *_a, **_k: pytest.fail("restore appended prototypes"),
    )
    audit = {}
    model = case.restore_source(source_data["warm"], accepted, archived_source, audit)
    state = model.snapshot_state()
    assert accepted == original
    assert state.policy.spec.observation_dim == 21 and state.policy.max_prototypes == 256
    assert state.policy.training_role == "train"
    assert state.policy.observation_standardization.fitted_rows == 29
    assert state.policy.observation_standardization.means == tuple(
        original["policy"]["observation_standardization"]["means"][:21]
    )
    assert state.policy.observation_standardization.scales == tuple(
        original["policy"]["observation_standardization"]["scales"][:21]
    )
    assert state.completed_updates == 1 and state.training_examples == 29
    np.testing.assert_array_equal(state.prototype_features, source_data["normalized"][:, :21])
    np.testing.assert_array_equal(state.prototype_actions, source_data["actions"])
    state.prototype_features[0, 0] += 1.0
    np.testing.assert_array_equal(
        model.snapshot_state().prototype_features, source_data["normalized"][:, :21]
    )


def test_should_access_only_two_warm_arrays_and_never_candidate_arrays(
    source_data, archived_source, monkeypatch
):
    accessed = []

    class WarmOnlyArchive:
        files = ["warm_features", "warm_actions", "candidate_features", "candidate_actions"]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __getitem__(self, key):
            accessed.append(key)
            assert key in ("warm_features", "warm_actions"), (
                "candidate labels/prototypes were accessed"
            )
            return source_data["normalized"] if key == "warm_features" else source_data["actions"]

    monkeypatch.setattr(case.np, "load", lambda path, *, allow_pickle: WarmOnlyArchive())
    case.restore_source(source_data["warm"], deepcopy(source_data["accepted"]), archived_source, {})
    assert accessed == ["warm_features", "warm_actions"]


@pytest.mark.parametrize(
    "damage",
    [
        "complete",
        "publications",
        "publication_type",
        "rows",
        "role",
        "capacity",
        "fit_rows",
        "fit_type",
        "means_length",
        "scales_zero",
        "means_nonfinite",
        "spec",
        "extra_policy",
    ],
)
def test_should_refuse_bad_accepted_source_before_reading_archive(
    source_data, archived_source, monkeypatch, damage
):
    accepted = deepcopy(source_data["accepted"])
    policy = accepted["policy"]
    if damage == "complete":
        accepted["complete"] = False
    elif damage == "publications":
        accepted["source_publications"] = 2
    elif damage == "publication_type":
        accepted["source_publications"] = True
    elif damage == "rows":
        accepted["source_rows"] = 30
    elif damage == "role":
        policy["training_role"] = "probe_train"
    elif damage == "capacity":
        policy["max_prototypes"] = 128
    elif damage == "fit_rows":
        policy["observation_standardization"]["fitted_rows"] = 30
    elif damage == "fit_type":
        policy["observation_standardization"]["fitted_rows"] = 29.0
    elif damage == "means_length":
        policy["observation_standardization"]["means"].pop()
    elif damage == "scales_zero":
        policy["observation_standardization"]["scales"][0] = 0.0
    elif damage == "means_nonfinite":
        policy["observation_standardization"]["means"][0] = float("inf")
    elif damage == "spec":
        policy["spec"]["observation_dim"] = 789.0
    else:
        policy["selected_coordinates"] = list(range(21))
    monkeypatch.setattr(case.np, "load", lambda *_a, **_k: pytest.fail("bad source opened archive"))
    with pytest.raises((AssertionError, ValueError, KeyError)):
        case.restore_source(source_data["warm"], accepted, archived_source, {})


@pytest.mark.parametrize("damage", ["final_role", "probe_role", "unknown_context", "terminal_rgb"])
def test_should_validate_full_warm_train_episode_before_archive_access(
    source_data, archived_source, monkeypatch, damage
):
    warm = source_data["warm"]
    if damage == "final_role":
        warm = replace(warm, role="final_test")
    elif damage == "probe_role":
        warm = replace(warm, role="probe_train")
    elif damage == "unknown_context":
        warm = replace(warm, context="pick-cube-probe-23")
    else:
        last = warm.transitions[-1]
        warm = replace(
            warm,
            transitions=warm.transitions[:-1]
            + (replace(last, next_observation=last.next_observation[:-1] + (1.01,)),),
        )
    monkeypatch.setattr(
        case.np, "load", lambda *_a, **_k: pytest.fail("invalid raw source opened archive")
    )
    with pytest.raises((AssertionError, ValueError)):
        case.restore_source(warm, deepcopy(source_data["accepted"]), archived_source, {})


@pytest.mark.parametrize(
    "damage",
    ["features", "actions", "feature_dtype", "action_dtype", "shape", "nonfinite", "missing_key"],
)
def test_should_bind_entire_789d_warm_features_and_actions_before_restoring21(
    source_data, tmp_path, damage
):
    features, actions = source_data["normalized"].copy(), source_data["actions"].copy()
    if damage == "features":
        features[0, 788] = 0.01
    elif damage == "actions":
        actions[0, 0] = 0.5
    elif damage == "feature_dtype":
        features = features.astype(np.float32)
    elif damage == "action_dtype":
        actions = actions.astype(np.float32)
    elif damage == "shape":
        features = features[:28]
    elif damage == "nonfinite":
        features[0, 788] = float("nan")
    fields = dict(
        warm_features=features,
        warm_actions=actions,
        candidate_features=np.empty((0, 789)),
        candidate_actions=np.empty((0, 4)),
    )
    if damage == "missing_key":
        fields.pop("warm_actions")
    path = tmp_path / "corrupt.npz"
    np.savez(path, **fields)
    with pytest.raises((AssertionError, ValueError, KeyError)):
        case.restore_source(source_data["warm"], deepcopy(source_data["accepted"]), path, {})


@pytest.fixture
def synthetic_case(source_data, archived_source, monkeypatch, tmp_path):
    stage, accepted_stage = tmp_path / "current", tmp_path / "accepted-camera"
    stage.mkdir()
    accepted_stage.mkdir()
    warm_raw, camera_raw = deepcopy(source_data["warm_raw"]), deepcopy(source_data["camera_raw"])
    warm_path, reference_path = tmp_path / "warm.json", tmp_path / "original-reference.json"
    write_new(warm_path, warm_raw)
    write_new(reference_path, camera_rollout(warm_raw, success=True))
    write_new(accepted_stage / "rollout-audit.json", source_data["accepted"])
    write_new(accepted_stage / "teacher-b.json", camera_raw)
    (accepted_stage / "prototype-states.npz").write_bytes(archived_source.read_bytes())
    for name, teacher, success in (
        ("b-before.json", camera_raw, False),
        ("b-after.json", camera_raw, True),
        ("policy-rollout.json", warm_raw, True),
    ):
        write_new(accepted_stage / name, camera_rollout(teacher, success=success))
    monkeypatch.setattr(case, "stage", stage)
    monkeypatch.setattr(case, "WARM_PATH", warm_path)
    monkeypatch.setattr(case, "REFERENCE_PATH", reference_path)
    monkeypatch.setattr(case, "ACCEPTED_STAGE", accepted_stage)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(stage / "policy-rollout.json")])
    observed = dict(calls=[], restored=[], mutate=None, successes={}, fail=None)
    actual_restore = case.restore_source

    def restored(*args, **kwargs):
        owner = actual_restore(*args, **kwargs)
        observed["restored"].append((owner, owner.snapshot_state()))
        return owner

    def frozen(
        output, *, actor, train_seed, visual_observation, camera_context, observation_projection
    ):
        assert len(observed["restored"]) == 1
        assert train_seed == 17 and visual_observation is True
        assert observation_projection == public_vector_projection()
        observed["calls"].append((output.name, actor, camera_context))
        teacher = camera_raw if camera_context is not None else warm_raw
        report = camera_rollout(teacher, success=observed["successes"].get(output.name, False))
        rows = len(report["episode"]["transitions"])
        state = actor.snapshot_state().state
        for row in report["episode"]["transitions"]:
            returned = actor.predict(np.asarray([row["observation"][:21]], dtype=np.float64))
            assert returned.actor_version == "serving-0"
            row["action"] = returned.prediction[0].tolist()
        report.update(
            policy="frozen_stable_nearest_continuous_bc",
            state_guard_scope="actual_serving_actor_no_private_fork",
            frozen_state_exact=True,
            original_owner_exact=True,
            observation_projection=asdict(public_vector_projection()),
            projection_exact=True,
            projection_work=dict(
                observation_attempts=rows,
                observation_returns=rows,
                source_values_upper_bound=rows * 789,
                source_values_validated=rows * 789,
                projected_values=rows * 21,
                projected_numeric_bytes=rows * 168,
            ),
            actor_version="serving-0",
            actor_version_exact=True,
            actor_backend_exact=True,
            actor_prediction_version_exact=True,
            actor_prediction_versions=["serving-0"] * rows,
            learner_policy=asdict(state.policy),
            source_updates=state.completed_updates,
            source_training_rows=state.training_examples,
            optimizer_updates=0,
            prediction_attempts=rows,
            prediction_returns=rows,
            validated_prediction_rows=rows,
            validated_prediction_bytes=rows * 32,
            actor_prediction_attempts=rows,
            actor_prediction_returns=rows,
            actor_distance_count_source="derived_from_exact_unchanged_typed_actor_and_query_shapes",
            actor_distance_candidate_upper_bound=rows * 29,
            actor_distance_candidates_completed=rows * 29,
            actor_distance_feature_upper_bound=rows * 29 * 21,
            actor_distance_feature_values_completed=rows * 29 * 21,
            nearest_prediction_attempts=0,
            nearest_prediction_returns=0,
            nearest_distance_candidate_upper_bound=0,
            nearest_distance_candidates_completed=0,
            nearest_distance_feature_upper_bound=0,
            nearest_distance_feature_values_completed=0,
            nearest_selected_indices=[],
        )
        if observed["fail"] == output.name:
            report.update(complete=False, error="synthetic renderer did not complete")
        if observed["mutate"] is not None:
            observed["mutate"](output.name, report, actor)
        write_new(output, report)
        return case.smoke.FrozenRolloutResult(report, None)

    monkeypatch.setattr(case, "restore_source", restored)
    monkeypatch.setattr(case.smoke, "run_frozen_rollout", frozen)
    monkeypatch.setattr(
        case.NumpyNearestContinuousBCLearner,
        "train_batch",
        lambda *_a, **_k: pytest.fail("case appended prototypes"),
    )
    observed.update(
        stage=stage,
        accepted_stage=accepted_stage,
        warm_path=warm_path,
        reference_path=reference_path,
    )
    return observed


def audit_for(observed):
    return json.loads((observed["stage"] / "rollout-audit.json").read_text(encoding="utf8"))


def test_should_run_actual_a_only_actor_b_then_a_and_accept_complete_negative_outcomes(
    synthetic_case,
):
    observed = synthetic_case
    original_inputs = {path: path.read_bytes() for path in observed["accepted_stage"].iterdir()}
    assert case.main() == 0
    audit = audit_for(observed)
    assert audit["complete"] is True
    assert [(name, context is not None) for name, _, context in observed["calls"]] == [
        ("b-before.json", True),
        ("policy-rollout.json", False),
    ]
    assert observed["calls"][0][1] is observed["calls"][1][1]
    source, initial = observed["restored"][0]
    actor = observed["calls"][0][1]
    case.prior.same_state(initial, source.snapshot_state())
    case.prior.same_state(initial, actor.snapshot_state().state)
    assert actor.version == "serving-0" and initial.training_examples == 29
    assert all(row["source"] is row["actor"] is True for row in audit["guards"])
    assert audit["comparisons"] == dict(
        public_B_minus_rgb_warm_B=dict(sparse_return=0.0, any_success=0.0, final_success=0.0),
        public_B_minus_rgb_candidate_B=dict(
            sparse_return=-1.0, any_success=-1.0, final_success=-1.0
        ),
        public_A_minus_rgb_candidate_A=dict(
            sparse_return=-1.0, any_success=-1.0, final_success=-1.0
        ),
        A_reference_minus_public_A_retention_loss=dict(
            sparse_return=1.0, any_success=1.0, final_success=1.0
        ),
    )
    assert audit["rollouts"]["b-before.json"]["teacher_transition_reconstruction_exact"] is False
    for path, payload in original_inputs.items():
        assert path.read_bytes() == payload
    for name in (
        "fit_attempts",
        "append_attempts",
        "optimizer_updates",
        "new_teacher_calls",
        "new_label_rows",
        "replayed_training_rows",
        "probe_file_reads",
        "final_file_reads",
        "promotion_calls",
    ):
        assert audit[name] == 0


def test_should_account_for_full_raw_validation_and_rendering_while_distance_uses21(synthetic_case):
    assert case.main() == 0
    audit = audit_for(synthetic_case)
    assert audit["source_validation_values"] == 29 * 789
    assert audit["archive_arrays_read"] == ["warm_features", "warm_actions"]
    assert audit["restored_numeric_bytes"] == 5800 and audit["statistics_numeric_bytes"] == 336
    assert audit["prediction_rows"] == audit["control_attempts"] == 31
    assert audit["known_control_physics_steps"] == 155
    assert audit["visual_projections"] == 33 and audit["raw_frame_input_bytes"] == 33 * 49152
    assert audit["returned_feature_values"] == 33 * 789
    assert audit["distance_candidates"] == 31 * 29
    assert audit["distance_feature_values"] == 31 * 29 * 21
    assert (
        audit["normalization_rows"] == 60
        and audit["normalization_feature_values"] == 29 * 789 + 31 * 21
    )
    assert audit["source_and_actor_prototype_numeric_bytes"] == 2 * 5800
    assert audit["source_and_actor_statistics_numeric_bytes"] == 2 * 336
    with np.load(synthetic_case["stage"] / "prototype-states.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"warm_features", "warm_actions"}
        assert archive["warm_features"].shape == (29, 21)
    for name in ("b-before.json", "policy-rollout.json"):
        raw = json.loads((synthetic_case["stage"] / name).read_text(encoding="utf8"))
        assert raw["episode"]["spec"]["observation_dim"] == 789
        assert (
            raw["projection_work"]["source_values_validated"]
            == len(raw["episode"]["transitions"]) * 789
        )


@pytest.mark.parametrize("damage", ["source_policy", "source_prototype", "historical_camera_trace"])
def test_should_refuse_invalid_reused_evidence_before_any_new_rollout(
    synthetic_case, monkeypatch, damage
):
    observed = synthetic_case
    if damage == "source_policy":
        path = observed["accepted_stage"] / "rollout-audit.json"
        raw = json.loads(path.read_text())
        raw["policy"]["training_role"] = "probe_train"
        path.write_text(json.dumps(raw), encoding="utf8")
    elif damage == "source_prototype":
        path = observed["accepted_stage"] / "prototype-states.npz"
        with np.load(path, allow_pickle=False) as archive:
            fields = {name: archive[name] for name in archive.files}
        fields["warm_features"][0, -1] += 0.01
        np.savez(path, **fields)
    else:
        path = observed["accepted_stage"] / "b-before.json"
        raw = json.loads(path.read_text())
        raw["observation_recipe"]["camera_eye"][1] = 0.0
        path.write_text(json.dumps(raw), encoding="utf8")
    if damage != "historical_camera_trace":
        monkeypatch.setattr(
            case.prior,
            "read_camera_teacher",
            lambda *_: pytest.fail("bad source reached camera labels"),
        )
    assert case.main() == 1
    assert not observed["calls"]
    assert audit_for(observed)["runtime_attempts"] == 0


@pytest.mark.parametrize(
    "damage",
    [
        "projection_metadata",
        "projection_work",
        "actor_count",
        "render_cost",
        "learner_statistics",
        "negative_incomplete",
    ],
)
def test_should_refuse_corrupt_first_report_before_second_rollout_and_keep_paid_file(
    synthetic_case, damage
):
    def corrupt(name, report, _actor):
        if name != "b-before.json":
            return
        if damage == "projection_metadata":
            report["observation_projection"]["feature_dim"] = 21.0
        elif damage == "projection_work":
            report["projection_work"]["source_values_validated"] -= 1
        elif damage == "actor_count":
            report["actor_distance_feature_values_completed"] -= 1
        elif damage == "render_cost":
            report["visual_work"]["returned_frame_bytes"] -= 1
        elif damage == "learner_statistics":
            scales = list(report["learner_policy"]["observation_standardization"]["scales"])
            scales[0] += 0.01
            report["learner_policy"]["observation_standardization"]["scales"] = scales
        else:
            report.update(complete=False, error="synthetic incomplete native output")

    synthetic_case["mutate"] = corrupt
    assert case.main() == 1
    audit = audit_for(synthetic_case)
    assert audit["complete"] is False and audit["runtime_attempts"] == audit["runtime_returns"] == 1
    assert len(synthetic_case["calls"]) == 1
    raw = json.loads((synthetic_case["stage"] / "b-before.json").read_text())
    assert raw["work"]["completed_steps"] == 2
    assert not (synthetic_case["stage"] / "policy-rollout.json").exists()


def test_should_refuse_changed_actual_actor_before_admitting_original_camera_return(synthetic_case):
    def corrupt(name, _report, actor):
        if name == "b-before.json":
            state = actor._learner.snapshot_state()
            state.prototype_actions[0, 0] = 0.5
            actor._learner.restore_state(state)

    synthetic_case["mutate"] = corrupt
    assert case.main() == 1
    assert len(synthetic_case["calls"]) == 1
    audit = audit_for(synthetic_case)
    assert "full learner state changed" in audit["error"]
    source, initial = synthetic_case["restored"][0]
    case.prior.same_state(initial, source.snapshot_state())


def test_should_refuse_replaced_preceding_retained_trace_before_second_native_call(
    synthetic_case, monkeypatch
):
    guard = case.guard_owners

    def replaced(source, actor, initial, audit, phase):
        guard(source, actor, initial, audit, phase)
        if phase == "before_A_return":
            path = synthetic_case["stage"] / "b-before.json"
            raw = json.loads(path.read_text())
            raw["projection_work"]["projected_values"] += 1
            path.write_text(json.dumps(raw), encoding="utf8")

    monkeypatch.setattr(case, "guard_owners", replaced)
    assert case.main() == 1
    assert len(synthetic_case["calls"]) == 1
    assert "preceding B trace was replaced" in audit_for(synthetic_case)["error"]


def test_should_bind_returned_and_retained_receipt_before_validating_first_rollout(
    synthetic_case, monkeypatch
):
    frozen = case.smoke.run_frozen_rollout

    def changed(*args, **options):
        outcome = frozen(*args, **options)
        returned = deepcopy(outcome.report)
        returned["projection_exact"] = False
        return case.smoke.FrozenRolloutResult(returned, None)

    monkeypatch.setattr(case.smoke, "run_frozen_rollout", changed)
    assert case.main() == 1
    assert len(synthetic_case["calls"]) == 1
    assert "returned/retained rollout evidence differs" in audit_for(synthetic_case)["error"]


def test_should_keep_unreturned_native_attempt_and_retained_file_without_retry(
    synthetic_case, monkeypatch
):
    frozen = case.smoke.run_frozen_rollout

    def failed_after_write(*args, **options):
        frozen(*args, **options)
        raise RuntimeError("synthetic frozen API return failed")

    monkeypatch.setattr(case.smoke, "run_frozen_rollout", failed_after_write)
    assert case.main() == 1
    audit = audit_for(synthetic_case)
    assert audit["runtime_attempts"] == 1 and audit["runtime_returns"] == 0
    assert "synthetic frozen API return failed" in audit["error"]
    assert (synthetic_case["stage"] / "b-before.json").exists()
    assert len(synthetic_case["calls"]) == 1


@pytest.mark.parametrize(
    "name", ["policy-rollout.json", "b-before.json", "prototype-states.npz", "rollout-audit.json"]
)
def test_should_preserve_existing_outputs_before_restoration_or_native(synthetic_case, name):
    path = synthetic_case["stage"] / name
    path.write_bytes(b"preserved failed or accepted attempt")
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"preserved failed or accepted attempt"
    assert not synthetic_case["restored"] and not synthetic_case["calls"]
