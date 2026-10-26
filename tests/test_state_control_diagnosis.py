"""Explicit saved state42 diagnosis, not policy execution or relabeling.

Per platform: three successful3-teacher/4-student diagnoses plus one finite
distance-square overflow normalize28rows/1176coordinates and attempt2016
state-distance coordinates. Successful calls compare1512 exact coordinates.
One reset-refusal stops before normalization; other admission tests forbid it.
Each successful selected payload is9132B +7224B temporaries =16356B, not RSS.
Two CLI calls stub diagnosis. Zero models/fits/RNG/forwards/gradients/native IO.
"""

from copy import deepcopy
import json
import sys

import pytest

from scripts import diagnose_learned_robot_control as diagnostic


UNITS = ["normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper"]
MISSING = object()


def observation(position, *, extra=0.0):
    value = [0.0] * 42
    value[30], value[31] = float(position), float(extra)
    return value


def episode(identity, observations, actions):
    return dict(
        episode_id=identity,
        context="pick-cube",
        role="train",
        spec=dict(
            observation_dim=42,
            action_units=UNITS.copy(),
            lower_bounds=[-1.0] * 4,
            upper_bounds=[1.0] * 4,
        ),
        transitions=[
            dict(
                step=index,
                observation=row.copy(),
                action=action.copy(),
                reward=0.0,
                next_observation=(
                    observations[index + 1].copy()
                    if index + 1 < len(observations)
                    else observation(5)
                ),
                terminated=False,
                truncated=index == len(observations) - 1,
            )
            for index, (row, action) in enumerate(zip(observations, actions, strict=True))
        ],
    )


def inputs():
    warm = episode(
        "pick-cube-task-teacher-17",
        [observation(x) for x in (0, 2, 4)],
        [[0.5, -0.5, 0.0, -1.0], [-0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0, -1.0]],
    )
    student = episode(
        "pick-cube-smoke-17",
        [observation(0), observation(1), observation(4), observation(4, extra=2)],
        [
            [0.6, -0.4, 0.25, 1.0],
            [1.0, -1.0, -1.0, -1.0],
            [0.0, 0.5, 0.75, -1.0],
            [-1.0, -1.0, -1.0, 1.0],
        ],
    )
    scales = [1.0] * 42
    scales[30], scales[31] = 2.0, 4.0
    policy = dict(
        spec=deepcopy(warm["spec"]),
        hidden_dim=64,
        seed=17,
        learning_rate=0.01,
        momentum=0.9,
        training_role="train",
        binary_gripper_axis=None,
        observation_standardization=dict(means=[0.0] * 42, scales=scales, fitted_rows=3),
    )
    teacher = dict(
        qualified=True,
        role="train",
        seed=17,
        task="PickCube-v1",
        controller="pd_ee_delta_pos",
        context="pick-cube",
        candidate_episode=warm,
        observation_recipe=None,
        visual_work=None,
    )
    learned = dict(
        complete=True,
        candidate_unchanged=True,
        training=dict(
            complete=True,
            policy=policy,
            work=dict(optimizer_updates=24, training_rows=704),
            source_episodes=dict(
                warm=dict(episode_id=warm["episode_id"], context="pick-cube", role="train")
            ),
        ),
        rollout=dict(
            task="PickCube-v1",
            complete=True,
            role="train",
            seed=17,
            context="pick-cube",
            controller="pd_ee_delta_pos",
            optimizer_updates=0,
            frozen_state_exact=True,
            original_owner_exact=True,
            learner_policy=deepcopy(policy),
            source_updates=24,
            source_training_rows=704,
            episode=student,
            observation_recipe=None,
            visual_work=None,
            sparse_return=0.0,
            any_success=False,
            final_success=False,
        ),
    )
    return teacher, learned


@pytest.fixture(scope="module")
def completed():
    teacher, learned = inputs()
    before = deepcopy((teacher, learned))
    result = diagnostic.diagnose_control(teacher, learned, state_observation=True)
    assert (teacher, learned) == before
    return result


def test_should_use_all42_frozen_coordinates_in_support_and_exact_label_binding(completed):
    assert completed["definition"] == "saved_train_state42_exact_input_fit_and_support_v1"
    assert completed["nearest_state42_teacher_steps"] == [0, 0, 2, 2]
    assert completed["nearest_state42_normalized_rms"] == pytest.approx(
        [0.0, 0.5 / 42**0.5, 0.0, 0.5 / 42**0.5]
    )
    assert completed["first_nonzero_state42_support_distance"] == 1
    assert completed["observation_blocks"] == dict(
        state42=dict(
            outside_training_range_coordinates_by_step=[0, 0, 0, 1],
            max_abs_frozen_normalized_by_step=[0.0, 0.5, 2.0, 2.0],
        )
    )
    matches = completed["exact_observation_matches"]
    assert [row["student_step"] for row in matches] == [0, 2]
    assert matches[0]["teacher_steps"] == [0] and matches[1]["teacher_steps"] == [2]
    assert matches[0]["action_error"] == pytest.approx([0.1, 0.1, 0.25, 2.0])
    assert matches[0]["desired_translation_delta_error_m"] == pytest.approx([0.01, 0.01, 0.025])
    assert matches[1]["action_error"] == pytest.approx([0.0, 0.5, -0.25, 0.0])
    assert "nearest_public21_teacher_steps" not in completed
    assert "nearest_public21_normalized_rms" not in completed
    assert "first_nonzero_prefix_support_distance" not in completed


def test_should_report_only_state_work_and_finite_selected_payload_without_model_costs(completed):
    assert completed["rows"] == dict(teacher=3, student=4)
    assert completed["work"] == dict(
        normalization_calls=2,
        normalization_rows=7,
        normalization_coordinates=294,
        state42_distance_coordinates=504,
        exact_match_coordinate_comparisons=504,
        selected_numeric_boolean_bytes=9132,
        selected_temporary_allowance_bytes=7224,
        selected_working_payload_bound_bytes=16356,
        model_initializations=0,
        fits=0,
        optimizer_updates=0,
        forwards=0,
        gradients=0,
        rng_draws=0,
        simulator_controls=0,
    )
    assert completed["recorded_outcome"] == dict(
        sparse_return=0.0, any_success=False, final_success=False
    )
    assert len(json.dumps(completed, allow_nan=False)) < 65536


def test_should_not_treat_nearest_or_same_time_commands_as_counterfactual_labels(completed):
    teacher, learned = inputs()
    teacher["candidate_episode"]["transitions"][1]["action"] = [1.0, -1.0, -1.0, -1.0]
    assert diagnostic.diagnose_control(teacher, learned, state_observation=True) == completed


def test_should_leave_identical_inputs_with_conflicting_teacher_labels_unscored():
    teacher, learned = inputs()
    rows = teacher["candidate_episode"]["transitions"]
    rows[1]["observation"] = rows[0]["observation"].copy()
    rows[0]["next_observation"] = rows[1]["observation"].copy()
    report = diagnostic.diagnose_control(teacher, learned, state_observation=True)
    assert report["exact_observation_matches"][0] == dict(
        student_step=0, teacher_steps=[0, 1], labels_consistent=False
    )
    assert report["exact_observation_matches"][1]["student_step"] == 2


def forbidden(*args, **kwargs):
    pytest.fail("invalid saved authority reached frozen normalization")


def test_should_refuse_changed_reset_before_normalization(monkeypatch):
    teacher, learned = inputs()
    learned["rollout"]["episode"]["transitions"][0]["observation"][30] = 0.01
    monkeypatch.setattr(diagnostic, "standardize_observations", forbidden)
    with pytest.raises(ValueError, match="reset"):
        diagnostic.diagnose_control(teacher, learned, state_observation=True)


def test_should_refuse_finite_state_distance_square_overflow_without_changing_inputs():
    teacher, learned = inputs()
    rows = learned["rollout"]["episode"]["transitions"]
    rows[1]["observation"][30] = 1e155
    rows[0]["next_observation"] = rows[1]["observation"].copy()
    before = deepcopy((teacher, learned))
    with pytest.raises(ValueError, match="finite"):
        diagnostic.diagnose_control(teacher, learned, state_observation=True)
    assert (teacher, learned) == before


@pytest.mark.parametrize("mode", [None, 1, 0, "true"])
def test_should_refuse_non_boolean_mode_before_observing_inputs(mode):
    with pytest.raises(ValueError):
        diagnostic.diagnose_control({}, {}, state_observation=mode)


@pytest.mark.parametrize(
    "path,value",
    [
        (("teacher", "role"), "final_test"),
        (("teacher", "candidate_episode", "role"), "probe_train"),
        (("learned", "rollout", "role"), "inner_guard"),
        (("learned", "rollout", "episode", "role"), "probe_test"),
        (("teacher", "seed"), 17.0),
        (("learned", "candidate_unchanged"), False),
        (("learned", "training", "policy", "training_role"), "final_test"),
        (("learned", "training", "policy", "seed"), True),
        (("learned", "training", "policy", "observation_standardization", "means"), [0.0] * 41),
        (("learned", "training", "policy", "observation_standardization", "scales"), [0.0] * 42),
        (("teacher", "candidate_episode", "context"), "other"),
        (("learned", "rollout", "episode", "spec", "action_units"), ["m"] * 4),
        (("learned", "rollout", "episode", "spec", "observation_dim"), 21),
        (("learned", "rollout", "controller"), "pd_joint_pos"),
        (("learned", "rollout", "task"), "Other-v1"),
        (("learned", "rollout", "task"), MISSING),
        (("learned", "observation_projection"), {"definition": "projected"}),
        (("learned", "rollout", "observation_projection"), {"definition": "projected"}),
        (("learned", "canonical_state_quaternions"), True),
        (("learned", "canonical_state_quaternions"), None),
        (("learned", "canonical_state_quaternions"), 0),
        (("learned", "input_projection_work"), {}),
        (("learned", "input_projection_work"), None),
        (("learned", "training", "work", "optimizer_updates"), 24.0),
        (("learned", "rollout", "source_training_rows"), True),
        (("teacher", "observation_recipe"), {}),
        (("learned", "rollout", "observation_recipe"), {}),
        (("teacher", "visual_work"), {}),
        (("learned", "rollout", "visual_work"), {}),
        (("teacher", "observation_recipe"), MISSING),
        (("learned", "rollout", "visual_work"), MISSING),
        (("teacher", "candidate_episode", "transitions", 0, "observation", 30), float("nan")),
        (
            ("learned", "training", "policy", "observation_standardization", "means"),
            [float("inf")] * 42,
        ),
    ],
)
def test_should_refuse_wrong_roles_policy_state_metadata_and_costs_before_normalization(
    monkeypatch, path, value
):
    teacher, learned = inputs()
    roots = dict(teacher=teacher, learned=learned)
    owner = roots[path[0]]
    for key in path[1:-1]:
        owner = owner[key]
    if value is MISSING:
        del owner[path[-1]]
    else:
        owner[path[-1]] = value
    learned["rollout"]["learner_policy"] = deepcopy(learned["training"]["policy"])
    monkeypatch.setattr(diagnostic, "standardize_observations", forbidden)
    with pytest.raises(ValueError):
        diagnostic.diagnose_control(teacher, learned, state_observation=True)


@pytest.mark.parametrize("fault", ["default_state", "explicit_rgb"])
def test_should_require_explicit_correct_dimension_selection_before_normalization(
    monkeypatch, fault
):
    teacher, learned = inputs()
    if fault == "explicit_rgb":
        for value in (teacher["candidate_episode"], learned["rollout"]["episode"]):
            value["spec"]["observation_dim"] = 789
            for row in value["transitions"]:
                for field in ("observation", "next_observation"):
                    row[field] += [0.25] * 747
        policy = learned["training"]["policy"]
        policy["spec"] = deepcopy(teacher["candidate_episode"]["spec"])
        policy["observation_standardization"]["means"] = [0.0] * 789
        policy["observation_standardization"]["scales"] = [1.0] * 789
        learned["rollout"]["learner_policy"] = deepcopy(policy)
    monkeypatch.setattr(diagnostic, "standardize_observations", forbidden)
    with pytest.raises(ValueError):
        diagnostic.diagnose_control(
            teacher, learned, **(dict(state_observation=True) if fault == "explicit_rgb" else {})
        )


@pytest.mark.parametrize("selected", [False, True])
def test_should_forward_state_cli_only_when_selected_without_extra_diagnosis(
    tmp_path, monkeypatch, capsys, selected
):
    teacher, learned = inputs()
    first, second = tmp_path / "teacher.json", tmp_path / "learned.json"
    first.write_text(json.dumps(teacher), encoding="utf8")
    second.write_text(json.dumps(learned), encoding="utf8")
    calls = []

    def stub(raw_teacher, raw_learned, **options):
        calls.append((raw_teacher, raw_learned, options))
        return dict(complete=True, recorded_outcome=dict(final_success=False))

    monkeypatch.setattr(diagnostic, "diagnose_control", stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose",
            "--teacher",
            str(first),
            "--learned",
            str(second),
            *(["--state-observation"] if selected else []),
        ],
    )
    diagnostic.main()
    assert calls == [(teacher, learned, dict(state_observation=True) if selected else {})]
    assert json.loads(capsys.readouterr().out) == dict(
        complete=True, recorded_outcome=dict(final_success=False)
    )
