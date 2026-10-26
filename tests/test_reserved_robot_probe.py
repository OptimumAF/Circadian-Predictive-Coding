"""Reserved probe collection and role-preserving features, without a learner.

Per platform: four fake scenes/reset/close, seven returned controls, eight
geometry reads, seven returned decisions; zero actual physics or ML work.
Three successful collections and four model-space projections are module-cached.
All other cases copy metadata, validate literal tuples, or refuse before IO.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, replace
import json
import sys
from typing import Any

import pytest

from scripts import maniskill_task_teacher_smoke as cli
from src.adapters.continuous_probe import ContinuousProbePlan
from src.core.continuous_demonstration import continuous_training_pairs
from src.core.pick_cube_action_history import (
    PickCubeActionHistoryRecipe,
    action_history_probe_episode,
    action_history_training_episode,
    validate_action_history_probe_episode,
)
from src.core.pick_cube_probe_context import PickCubeProbeBinding, pick_cube_probe_binding
from src.core.pick_cube_readiness_teacher import (
    DEFINITION,
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_task_teacher as collector
from test_observed_grasp_teacher import ObservedEnvironment, install


ROLES = ("probe_train", "probe_test")
CONTEXT = "pick-cube-reserved-probe-23-24"


def geometry(port):
    goal = (0.01, 0.0, 0.0) if port._records else (0.018, 0.018, 0.0)
    return PickCubeGeometry((-0.3, 0.0, 0.25), (0.0, 0.0, 0.0), goal)


@pytest.fixture(scope="module")
def collected():
    records: dict[str, Any] = {}
    for role in (*ROLES, "legacy"):
        environment = ObservedEnvironment((1.0, 1.0, 1.0))
        geometry_calls = []

        def current(port):
            geometry_calls.append(len(port._records))
            return geometry(port)

        with pytest.MonkeyPatch.context() as patch:
            calls = install(patch, environment, current)
            result = (
                collector.collect_readiness_teacher(observed_grasp=True, goal_consistent=True)
                if role == "legacy"
                else collector.collect_probe_teacher(role=role)
            )
        assert result.qualified is True, result
        records[role] = result, environment, calls, geometry_calls
    return records


@pytest.fixture(scope="module")
def encoded(collected):
    records = {}
    for role in ROLES:
        raw = collected[role][0].probe_episode()
        for recipe in (PickCubeTaskSpaceHistoryRecipe(), PickCubeActionHistoryRecipe()):
            records[role, recipe.feature_dim] = (
                raw,
                recipe,
                action_history_probe_episode(raw, recipe),
            )
    return records


@pytest.mark.parametrize(
    "role,seed,part", [("probe_train", 23, "train"), ("probe_test", 24, "test")]
)
def test_should_bind_fixed_resets_and_original_identities_before_observing_data(role, seed, part):
    binding = pick_cube_probe_binding(role)
    assert type(binding) is PickCubeProbeBinding
    assert asdict(binding) == dict(
        role=role, seed=seed, context=CONTEXT, episode_id=f"pick-cube-probe-{part}-teacher-{seed}"
    )
    assert pick_cube_probe_binding(role) is not binding
    with pytest.raises(FrozenInstanceError):
        setattr(binding, "seed", 17)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("seed", 24),
        ("context", "pick-cube"),
        ("episode_id", "relabeled"),
        ("role", "train"),
    ],
)
def test_should_refuse_forged_fixed_bindings(field, value):
    options: dict[str, Any] = asdict(pick_cube_probe_binding("probe_train"))
    options[field] = value
    with pytest.raises(ValueError):
        PickCubeProbeBinding(**options)


@pytest.mark.parametrize("role", ROLES)
def test_should_collect_qualified_reserved_data_with_original_flags_commands_and_paid_work(
    collected, role
):
    result, environment, calls, geometry_calls = collected[role]
    binding = pick_cube_probe_binding(role)
    assert type(result) is collector.ReservedProbeTaskTeacherResult
    assert result.probe_seed == binding.seed
    assert result.teacher_definition == GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
    assert result.observed_grasp == (True, True)
    assert result.stages == ("lift", "settle")
    assert result.work == native.ManiSkillWork(2, 2, 10, 0)
    assert result.creation_attempts == 1
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 2
    assert result.error is result.traceback is None
    raw = result.probe_episode()
    assert (raw.role, raw.context, raw.episode_id) == (role, CONTEXT, binding.episode_id)
    assert raw.spec.observation_dim == 42
    assert raw.transitions[-1].terminated and raw.transitions[-1].reward == 1.0
    assert all(row.observation[18] == row.next_observation[18] == 1.0 for row in raw.transitions)
    assert all(command.shape == (1, 4) for command in environment.commands)
    assert [row.action for row in raw.transitions] == [
        tuple(command[0].tolist()) for command in environment.commands
    ]
    assert calls == [{}] and environment.reset_calls == [binding.seed]
    assert environment.close_calls == 1 and geometry_calls == [0, 1]
    owned = result.probe_episode()
    assert owned == raw and owned is not raw
    object.__setattr__(owned.transitions[0], "reward", 0.5)
    assert result.probe_episode() == raw
    with pytest.raises(ValueError):
        result.training_episode()
    with pytest.raises(ValueError):
        result.guard_episode(now=100)


def test_should_preserve_the_existing_train17_v2_family_without_probe_metadata(collected):
    result, environment, _, _ = collected["legacy"]
    assert type(result) is collector.GoalConsistentObservedGraspTaskTeacherResult
    assert "probe_seed" not in {field.name for field in fields(result)}
    episode = result.training_episode()
    assert episode.role == "train" and episode.context == "pick-cube"
    assert episode.episode_id == "pick-cube-goal-consistent-observed-grasp-teacher-17"
    assert environment.reset_calls == [17]


@pytest.mark.parametrize(
    "damage",
    [
        "seed_bool",
        "seed_other",
        "id",
        "context",
        "role",
        "definition",
        "feedback",
        "stage",
        "work",
        "flag_bool",
        "family",
        "base_erasure",
        "base_marker_train",
        "coherent_train",
    ],
)
def test_should_refuse_reserved_qualification_corruption_without_releasing_a_train_alias(
    collected, damage
):
    result = deepcopy(collected["probe_train"][0])
    episode = result.candidate_episode
    assert episode is not None
    if damage in ("seed_bool", "seed_other"):
        object.__setattr__(result, "probe_seed", True if damage == "seed_bool" else 24)
    elif damage in ("id", "context", "role"):
        field = {"id": "episode_id", "context": "context", "role": "role"}[damage]
        object.__setattr__(
            episode,
            field,
            {"id": "pick-cube-probe-test-teacher-24", "context": "pick-cube", "role": "train"}[
                damage
            ],
        )
    elif damage == "definition":
        object.__setattr__(result, "teacher_definition", "pick_cube_geometry_readiness_v1")
    elif damage == "feedback":
        object.__setattr__(result, "observed_grasp", (True, False))
    elif damage == "stage":
        object.__setattr__(result, "stages", ("grasp", "settle"))
    elif damage == "work":
        assert result.work is not None
        object.__setattr__(result.work, "completed_steps", True)
    elif damage == "flag_bool":
        row = episode.transitions[-1]
        values = list(row.next_observation)
        values[18] = True
        object.__setattr__(row, "next_observation", tuple(values))
    elif damage == "family":
        object.__setattr__(
            result, "__class__", collector.GoalConsistentObservedGraspTaskTeacherResult
        )
    elif damage in ("base_erasure", "base_marker_train"):
        base_fields = {
            field.name: getattr(result, field.name) for field in fields(collector.TaskTeacherResult)
        }
        base_fields["teacher_definition"] = None
        result = collector.TaskTeacherResult(**base_fields)
        if damage == "base_marker_train":
            object.__setattr__(result, "probe_seed", 23)
            object.__setattr__(episode, "role", "train")
            object.__setattr__(episode, "context", "pick-cube")
            object.__setattr__(episode, "episode_id", "pick-cube-readiness-teacher-17")
            object.__setattr__(result, "teacher_definition", DEFINITION)
    else:
        object.__setattr__(episode, "role", "train")
        object.__setattr__(episode, "context", "pick-cube")
        object.__setattr__(
            episode, "episode_id", "pick-cube-goal-consistent-observed-grasp-teacher-17"
        )
        object.__setattr__(result, "probe_seed", 17)
    assert result.qualified is False
    if type(result) is collector.ReservedProbeTaskTeacherResult:
        with pytest.raises(ValueError):
            result.probe_episode()
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("dimension", [12, 47])
def test_should_encode_only_preceding_native_commands_and_preserve_the_probe_role(
    encoded, role, dimension
):
    raw, recipe, model = encoded[role, dimension]
    original = deepcopy(raw)
    assert (model.episode_id, model.context, model.role) == (raw.episode_id, raw.context, role)
    assert model.spec.observation_dim == dimension
    first, second = model.transitions
    assert first.observation[-5:] == (0.0,) * 5
    assert first.next_observation[-5:] == (*first.action, 1.0 / 6.0)
    assert second.observation[-5:] == (*first.action, 1.0 / 6.0)
    assert second.next_observation[-5:] == (*second.action, 2.0 / 6.0)
    # The teacher's float64 .18 proposal is not the executed float32 command.
    assert first.action[0] != 0.18 and first.action[0] == pytest.approx(0.18)
    assert [
        (row.action, row.reward, row.terminated, row.truncated) for row in model.transitions
    ] == [(row.action, row.reward, row.terminated, row.truncated) for row in raw.transitions]
    owned = validate_action_history_probe_episode(model, recipe)
    assert owned == model and owned is not model
    assert raw == original


@pytest.mark.parametrize("role", ROLES)
def test_should_keep_train_feature_extraction_closed_to_reserved_roles(collected, role):
    with pytest.raises(ValueError):
        action_history_training_episode(
            collected[role][0].probe_episode(), PickCubeTaskSpaceHistoryRecipe()
        )


@pytest.mark.parametrize("damage", ["context", "id", "role", "current_bool", "next_integer"])
def test_should_refuse_relabeling_and_original_flag_aliases_before_raw_projection(
    collected, damage
):
    episode = collected["probe_train"][0].probe_episode()
    if damage in ("context", "id", "role"):
        object.__setattr__(
            episode, {"context": "context", "id": "episode_id", "role": "role"}[damage], "train"
        )
    else:
        row = episode.transitions[0]
        field = "observation" if damage == "current_bool" else "next_observation"
        values = list(getattr(row, field))
        values[18] = True if damage == "current_bool" else 1
        object.__setattr__(row, field, tuple(values))
    with pytest.raises(ValueError):
        action_history_probe_episode(episode, PickCubeTaskSpaceHistoryRecipe())


@pytest.mark.parametrize("dimension", [12, 47])
def test_should_refuse_changed_executed_history_without_reconstructing_new_teacher_labels(
    encoded, dimension
):
    _, recipe, original = encoded["probe_train", dimension]
    model = deepcopy(original)
    row = model.transitions[0]
    values = list(row.next_observation)
    values[-2] = 1.0
    object.__setattr__(row, "next_observation", tuple(values))
    with pytest.raises(ValueError):
        validate_action_history_probe_episode(model, recipe)
    assert original.transitions[0].next_observation[-2] == -1.0


@pytest.mark.parametrize("dimension", [42, 12])
def test_should_compose_existing_probe_plan_without_changing_roles_or_training_on_test_data(
    collected, encoded, dimension
):
    train = (
        collected["probe_train"][0].probe_episode()
        if dimension == 42
        else encoded["probe_train", 12][2]
    )
    test = (
        collected["probe_test"][0].probe_episode()
        if dimension == 42
        else encoded["probe_test", 12][2]
    )
    plan = ContinuousProbePlan(train, test)
    assert plan.training == train and plan.evaluation == test
    assert plan.training is not train and plan.evaluation is not test
    features, targets = continuous_training_pairs(plan.training, role="probe_train")
    assert features == tuple(row.observation for row in train.transitions)
    assert targets == tuple(row.action for row in train.transitions)
    with pytest.raises(ValueError):
        continuous_training_pairs(plan.evaluation, role="probe_train")
    with pytest.raises(ValueError):
        continuous_training_pairs(plan.training)
    with pytest.raises(ValueError):
        ContinuousProbePlan(test, train)


@pytest.mark.parametrize("role", ["train", "final_test", True])
def test_should_refuse_nonreserved_producer_roles_before_creating_a_port(monkeypatch, role):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("invalid producer role created a native port")

    monkeypatch.setattr(collector, "ManiSkillPickCubeEpisode", forbidden)
    with pytest.raises(ValueError):
        collector.collect_probe_teacher(role=role)
    assert calls == []


@pytest.mark.parametrize("role", ["train", "probe_train", "inner_guard", "final_test"])
def test_should_reserve_reset24_for_scoring_before_any_environment_exists(monkeypatch, role):
    calls = []

    def forbidden(**options):
        calls.append(options)
        pytest.fail("score-only reset24 created an environment under another role")

    monkeypatch.setattr(native, "_make_environment", forbidden)
    with pytest.raises(ValueError):
        native.ManiSkillPickCubeEpisode("forbidden24", CONTEXT, role=role, seed=24)
    assert calls == []


def test_should_stop_after_paid_prior_record_mutation_without_more_decisions_or_controls():
    environment = ObservedEnvironment((1.0, 1.0, 1.0))
    geometry_calls = []

    def corrupt_prior(port):
        geometry_calls.append(len(port._records))
        if port._records:
            object.__setattr__(port._records[0], "action", (0.0, 0.0, 0.0, -1.0))
        return geometry(port)

    with pytest.MonkeyPatch.context() as patch:
        calls = install(patch, environment, corrupt_prior)
        result = collector.collect_probe_teacher(role="probe_train")
    assert type(result) is collector.ReservedProbeTaskTeacherResult
    assert result.qualified is False and result.error is not None
    assert result.probe_seed == 23 and result.observed_grasp == (True,)
    assert result.geometry_attempts == result.geometry_returns == 2
    assert result.teacher_actions == 1
    assert result.work == native.ManiSkillWork(1, 1, 5, 0)
    assert len(environment.commands) == 1 and environment.close_calls == 1
    assert calls == [{}] and geometry_calls == [0, 1]
    with pytest.raises(ValueError):
        result.probe_episode()


@pytest.mark.parametrize(
    "role,returned_role",
    [("probe_train", "probe_train"), ("probe_test", "probe_test"), ("probe_train", "probe_test")],
)
def test_should_bind_cli_output_to_the_requested_reserved_allocation_without_recollection(
    collected, monkeypatch, tmp_path, role, returned_role
):
    output = tmp_path / "probe.json"
    calls = []
    returned = collected[returned_role][0]

    def collect(**options):
        calls.append(options)
        return returned

    monkeypatch.setattr(cli, "collect_probe_teacher", collect)
    monkeypatch.setattr(cli, "version", lambda _: "test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teacher",
            "--output",
            str(output),
            "--probe-role",
            role,
            "--geometry-readiness",
            "--observed-grasp",
            "--goal-consistent",
        ],
    )
    assert cli.main() == (0 if role == returned_role else 1)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["qualified"] is (role == returned_role)
    assert payload["probe_seed"] == returned.probe_seed
    assert payload["teacher_actions"] == 2 and payload["work"]["completed_steps"] == 2
    assert payload["candidate_episode"] == json.loads(
        json.dumps(asdict(returned.candidate_episode))
    )
    assert ("probe_request_error" in payload) is (role != returned_role)
    assert calls == [{"role": role}]


def test_should_retain_an_ordinary_paid_probe_failure_without_claiming_a_request_mismatch(
    collected, monkeypatch, tmp_path
):
    returned = replace(
        collected["probe_train"][0], final_success=False, error="paid collection failed"
    )
    output = tmp_path / "failed.json"
    calls = []

    def collect(**options):
        calls.append(options)
        return returned

    monkeypatch.setattr(cli, "collect_probe_teacher", collect)
    monkeypatch.setattr(cli, "version", lambda _: "test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teacher",
            "--output",
            str(output),
            "--probe-role",
            "probe_train",
            "--geometry-readiness",
            "--observed-grasp",
            "--goal-consistent",
        ],
    )
    assert cli.main() == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["qualified"] is False and payload["error"] == "paid collection failed"
    assert payload["work"]["completed_steps"] == payload["teacher_actions"] == 2
    assert "probe_request_error" not in payload
    assert calls == [{"role": "probe_train"}]


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--geometry-readiness", "--observed-grasp"],
        ["--geometry-readiness", "--observed-grasp", "--goal-consistent", "--visual-observation"],
    ],
)
def test_should_refuse_incomplete_or_visual_probe_cli_selection_before_output_or_collection(
    monkeypatch, tmp_path, options
):
    output = tmp_path / "forbidden.json"

    def forbidden(**kwargs):
        pytest.fail("invalid probe CLI selection collected data")

    monkeypatch.setattr(cli, "collect_probe_teacher", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["teacher", "--output", str(output), "--probe-role", "probe_train", *options]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not output.exists()
