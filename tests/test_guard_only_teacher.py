"""Guard-only qualification and isolation through existing fake native IO."""

from dataclasses import asdict, replace
import inspect
import json
import sys

import numpy as np
import pytest

from test_maniskill_task_teacher import FakeEnvironment
from test_maniskill_rgb_observations import RGBEnvironment
from scripts import maniskill_task_teacher_smoke as cli
from scripts import run_native_stream_demo as stream_cli
from src.adapters import continuous_stream_demo as stream
from src.adapters import numpy_continuous_bc as mlp
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_demonstration import continuous_training_pairs, copy_continuous_episode
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_guard_context import pick_cube_guard_context
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.infra import maniskill_continuous as port
from src.infra import maniskill_task_teacher as teacher


@pytest.fixture
def native_io(monkeypatch):
    def create(*, visual=False):
        environment = RGBEnvironment() if visual else FakeEnvironment()
        environment.done_at = 3
        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            return environment

        monkeypatch.setattr(port, "_make_environment", factory)
        return environment, calls

    return create


def collect(native_io, *, seed=31, visual=False, arrival=2):
    environment, calls = native_io(visual=visual)
    result = teacher.collect_guard_teacher(
        guard_seed=seed, labels_arrived_at=arrival, visual_observation=visual
    )
    return result, environment, calls


@pytest.mark.parametrize(
    "seed,context", [(31, "pick-cube-inner-guard-31"), (37, "pick-cube-inner-guard-37")]
)
def test_should_use_only_the_two_prospectively_allocated_guard_contexts(seed, context):
    assert pick_cube_guard_context(seed) == context


@pytest.mark.parametrize(
    "seed", [-1, 0, 17, 19, 23, 29, 32, 38, True, 31.0, "31", None, np.int64(31)]
)
def test_should_refuse_training_reserved_or_non_native_guard_seeds(seed):
    with pytest.raises(ValueError):
        pick_cube_guard_context(seed)


@pytest.mark.parametrize("seed", [31, 37])
@pytest.mark.parametrize("visual", [False, True])
def test_should_collect_actual_guard_port_with_fixed_identity_and_owned_arrived_data(
    native_io, seed, visual
):
    result, environment, calls = collect(native_io, seed=seed, visual=visual)
    assert result.qualified is True and result.error is None
    assert result.guard_seed == seed and result.labels_arrived_at == 2
    assert environment.reset_calls == [seed] and environment.close_calls == 1
    assert calls == ([{"visual_observation": True}] if visual else [{}])
    assert result.creation_attempts == 1
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 3
    assert result.work == port.ManiSkillWork(3, 3, 15, 0)
    episode = result.guard_episode(now=2)
    assert episode.role == "inner_guard"
    assert episode.context == f"pick-cube-inner-guard-{seed}"
    assert episode.episode_id == f"pick-cube-guard-teacher-{seed}"
    assert episode is not result.candidate_episode
    assert episode.transitions[0] is not result.candidate_episode.transitions[0]
    assert episode.spec is not result.candidate_episode.spec
    assert episode.spec.observation_dim == (789 if visual else 3)
    assert episode.transitions[-1].reward == 1 and episode.transitions[-1].truncated
    for row, command in zip(episode.transitions, environment.commands, strict=True):
        assert row.action == tuple(float(value) for value in command[0])
    if visual:
        assert result.observation_recipe == pick_cube_rgb_recipe()
        assert (
            result.visual_work.observation_attempts == result.visual_work.observation_returns == 4
        )
        assert result.visual_work.returned_feature_values == 4 * 789
        assert result.visual_work.returned_frame_bytes == 4 * 49152
        assert result.visual_work.current_owned_frame_bytes == 49152
        assert result.visual_work.frame_export_copies == 0
    else:
        assert result.observation_recipe is result.visual_work is None
    expected = result.candidate_episode.transitions[0].observation
    object.__setattr__(
        episode.transitions[0], "observation", (99.0,) * episode.spec.observation_dim
    )
    assert result.guard_episode(now=3).transitions[0].observation == expected
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize("now", [0, 1, -1, True, 2.0, "2", None, np.int64(2)])
def test_should_refuse_unarrived_or_non_native_time_before_copying_guard_payload(
    native_io, monkeypatch, now
):
    result, _, _ = collect(native_io)
    monkeypatch.setattr(
        teacher,
        "copy_continuous_episode",
        lambda *_: pytest.fail("invalid arrival accessed guard payload"),
    )
    with pytest.raises(ValueError):
        result.guard_episode(now=now)


@pytest.mark.parametrize("bad", [-1, True, 2.0, "2", None, np.int64(2)])
def test_should_refuse_invalid_declared_arrival_before_port_creation(monkeypatch, bad):
    monkeypatch.setattr(
        teacher,
        "ManiSkillPickCubeEpisode",
        lambda *_a, **_k: pytest.fail("invalid arrival created a port"),
    )
    with pytest.raises(ValueError):
        teacher.collect_guard_teacher(guard_seed=31, labels_arrived_at=bad)


@pytest.mark.parametrize("bad", [0, 1, None, "rgb", np.bool_(True)])
def test_should_refuse_non_boolean_visual_guard_mode_before_port(monkeypatch, bad):
    monkeypatch.setattr(
        teacher,
        "ManiSkillPickCubeEpisode",
        lambda *_a, **_k: pytest.fail("invalid visual flag created a port"),
    )
    with pytest.raises(ValueError):
        teacher.collect_guard_teacher(guard_seed=31, labels_arrived_at=0, visual_observation=bad)


@pytest.mark.parametrize(
    "role,seed,context",
    [
        ("inner_guard", 17, "pick-cube-inner-guard-31"),
        ("inner_guard", 23, "pick-cube-inner-guard-31"),
        ("inner_guard", 29, "pick-cube-inner-guard-31"),
        ("inner_guard", 31, "pick-cube"),
        ("inner_guard", 37, "pick-cube-inner-guard-31"),
        ("train", 31, "pick-cube-inner-guard-31"),
        ("probe_train", 31, "pick-cube-inner-guard-31"),
        ("probe_test", 37, "pick-cube-inner-guard-37"),
    ],
)
def test_should_enforce_guard_seed_role_and_context_before_environment_factory(
    monkeypatch, role, seed, context
):
    monkeypatch.setattr(
        port,
        "_make_environment",
        lambda **_: pytest.fail("invalid guard allocation created a scene"),
    )
    with pytest.raises(ValueError):
        port.ManiSkillPickCubeEpisode("guard-port", context, role=role, seed=seed)


@pytest.mark.parametrize(
    "change",
    [
        "role",
        "context",
        "episode_id",
        "seed",
        "arrival",
        "final_flag",
        "observed_flag",
        "final_reward",
        "sparse_reward",
        "bounds",
        "seam",
        "incomplete",
    ],
)
def test_should_revalidate_guard_metadata_and_complete_core_record_on_extraction(native_io, change):
    result, _, _ = collect(native_io)
    episode = result.candidate_episode
    if change == "role":
        result = replace(result, candidate_episode=replace(episode, role="train"))
    elif change == "context":
        result = replace(result, candidate_episode=replace(episode, context="pick-cube"))
    elif change == "episode_id":
        result = replace(
            result, candidate_episode=replace(episode, episode_id="pick-cube-task-teacher-17")
        )
    elif change == "seed":
        result = replace(result, guard_seed=37)
    elif change == "arrival":
        result = replace(result, labels_arrived_at=True)
    elif change == "final_flag":
        result = replace(result, final_success=1)
    elif change == "observed_flag":
        result = replace(result, observed_success=1)
    elif change == "final_reward":
        rows = episode.transitions[:-1] + (replace(episode.transitions[-1], reward=0.0),)
        result = replace(result, candidate_episode=replace(episode, transitions=rows))
    elif change == "sparse_reward":
        rows = (replace(episode.transitions[0], reward=0.5),) + episode.transitions[1:]
        result = replace(result, candidate_episode=replace(episode, transitions=rows))
    elif change == "bounds":
        object.__setattr__(episode.transitions[0], "action", (2.0, 0.0, 0.0, 1.0))
    elif change == "seam":
        object.__setattr__(episode.transitions[1], "observation", (99.0, 0.0, 0.0))
    else:
        object.__setattr__(episode.transitions[-1], "truncated", False)
    with pytest.raises(ValueError):
        result.guard_episode(now=2)


def test_should_preserve_train_api_and_refuse_guard_metadata_even_on_train_record(native_io):
    native_io()
    result = teacher.collect_task_teacher()
    parameters = inspect.signature(teacher.collect_task_teacher).parameters
    assert tuple(parameters) == (
        "train_seed",
        "visual_observation",
        "camera_context",
        "mass_context",
        "lighting_context",
    )
    assert (
        parameters["train_seed"].default == 17 and parameters["visual_observation"].default is False
    )
    assert parameters["camera_context"].default is None
    assert parameters["lighting_context"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["lighting_context"].default is None
    assert all(value.kind is inspect.Parameter.KEYWORD_ONLY for value in parameters.values())
    assert result.guard_seed is result.labels_arrived_at is None
    assert result.training_episode().role == "train"
    with pytest.raises(ValueError):
        result.guard_episode(now=2)
    for corrupted in (replace(result, guard_seed=31), replace(result, labels_arrived_at=2)):
        with pytest.raises(ValueError):
            corrupted.training_episode()


@pytest.mark.parametrize("failure", ["physics", "geometry", "cleanup", "final_negative", "sparse"])
def test_should_retain_failed_guard_work_and_refuse_extraction(native_io, monkeypatch, failure):
    environment, _ = native_io()
    if failure == "physics":
        environment.step_failure_at = 2
    elif failure == "geometry":
        original = teacher._geometry
        calls = []

        def geometry(session):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("guard geometry unavailable")
            return original(session)

        monkeypatch.setattr(teacher, "_geometry", geometry)
    elif failure == "cleanup":
        environment.close_error = RuntimeError("guard cleanup failed")
    elif failure == "final_negative":
        environment.rewards[3] = 0.0
    else:
        environment.rewards[2] = 0.5
    result = teacher.collect_guard_teacher(guard_seed=37, labels_arrived_at=2)
    assert result.qualified is False and environment.close_calls == 1
    assert result.guard_seed == 37 and result.labels_arrived_at == 2
    assert environment.reset_calls == [37]
    if failure == "physics":
        assert result.work == port.ManiSkillWork(2, 1, 5, 5)
        assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 2
    elif failure == "geometry":
        assert result.work == port.ManiSkillWork(1, 1, 5, 0)
        assert (
            result.geometry_attempts == 2 and result.geometry_returns == result.teacher_actions == 1
        )
    elif failure in ("cleanup", "final_negative"):
        assert result.work == port.ManiSkillWork(3, 3, 15, 0)
        assert result.candidate_episode is not None
    else:
        assert result.work == port.ManiSkillWork(2, 2, 10, 0)
    with pytest.raises(ValueError):
        result.guard_episode(now=2)
    with pytest.raises(ValueError):
        result.training_episode()


def test_should_retain_malformed_visual_step_costs_without_guard_data(native_io):
    environment, _ = native_io(visual=True)
    environment.next_packet = {"agent": {}, "extra": {}, "sensor_data": {}}
    result = teacher.collect_guard_teacher(
        guard_seed=31, labels_arrived_at=2, visual_observation=True
    )
    assert not result.qualified and result.candidate_episode is None and result.error
    # Native step returned before RGB encoding failed: its physics completed.
    assert result.work == port.ManiSkillWork(1, 1, 5, 0)
    assert result.visual_work.observation_attempts == 2
    assert result.visual_work.observation_returns == 1
    assert result.visual_work.current_owned_frame_bytes == 49152
    assert environment.close_calls == 1


def test_should_refuse_visual_recipe_corruption_after_cleanup(native_io, monkeypatch):
    environment, _ = native_io(visual=True)
    original_recipe = port.ManiSkillPickCubeEpisode.observation_recipe
    original_close = environment.close
    closed = []

    def close():
        original_close()
        closed.append(True)

    def recipe(session):
        value = original_recipe.fget(session)
        return replace(value, pixel_divisor=255) if closed else value

    monkeypatch.setattr(environment, "close", close)
    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "observation_recipe", property(recipe))
    result = teacher.collect_guard_teacher(
        guard_seed=37, labels_arrived_at=2, visual_observation=True
    )
    assert result.candidate_episode is not None and not result.qualified and result.error
    assert result.work.completed_steps == 3 and result.visual_work.observation_returns == 4
    with pytest.raises(ValueError):
        result.guard_episode(now=2)


@pytest.mark.parametrize("bad", ["lost_recipe", "recipe_type", "dimension", "terminal_rgb"])
def test_should_revalidate_visual_recipe_and_terminal_pixel_range_when_extracting(native_io, bad):
    result, _, _ = collect(native_io, visual=bad != "dimension")
    if bad == "lost_recipe":
        result = replace(result, observation_recipe=None)
    elif bad == "recipe_type":
        result = replace(
            result, observation_recipe=replace(result.observation_recipe, pixel_divisor=255)
        )
    elif bad == "dimension":
        result = replace(result, observation_recipe=pick_cube_rgb_recipe())
    else:
        last = result.candidate_episode.transitions[-1]
        changed = last.next_observation[:21] + (1.1,) + last.next_observation[22:]
        object.__setattr__(last, "next_observation", changed)
    with pytest.raises(ValueError):
        result.guard_episode(now=2)


@pytest.mark.parametrize("role", ["train", "probe_train", "inner_guard"])
def test_should_refuse_guard_pairs_even_with_explicit_role_override(native_io, role):
    result, _, _ = collect(native_io)
    episode = result.guard_episode(now=2)
    copied = copy_continuous_episode(episode)
    assert copied.role == "inner_guard" and copied is not episode
    with pytest.raises(ValueError):
        continuous_training_pairs(episode, role=role)


@pytest.mark.parametrize(
    "family", ["fit", "mlp_train", "mlp_probe", "nearest_train", "nearest_probe"]
)
def test_should_refuse_guard_before_statistics_fit_or_learner_update(
    native_io, monkeypatch, family
):
    result, _, _ = collect(native_io)
    episode = result.guard_episode(now=2)
    if family == "fit":
        monkeypatch.setattr(
            mlp.observations,
            "fit_observation_standardization",
            lambda *_: pytest.fail("guard data reached statistics fit"),
        )
        with pytest.raises(ValueError):
            mlp.fit_training_standardization(episode)
        return
    role = "probe_train" if family.endswith("probe") else "train"
    if family.startswith("mlp"):
        owner = mlp.NumpyContinuousBCLearner(
            mlp.ContinuousBCPolicy(episode.spec, hidden_dim=2, training_role=role)
        )
    else:
        stats = ObservationStandardization((0.0,) * 3, (1.0,) * 3, 1)
        owner = NumpyNearestContinuousBCLearner(
            NearestContinuousBCPolicy(episode.spec, stats, training_role=role)
        )
    before = owner.snapshot_state()
    monkeypatch.setattr(
        owner, "train_batch", lambda *_a, **_k: pytest.fail("guard reached native update")
    )
    with pytest.raises(ValueError):
        owner.train_episode(episode)
    after = owner.snapshot_state()
    assert (
        before.policy == after.policy and before.completed_updates == after.completed_updates == 0
    )
    assert before.training_examples == after.training_examples == 0


@pytest.mark.parametrize("guard_first", [False, True])
def test_should_refuse_guard_stream_inputs_before_any_fit(native_io, monkeypatch, guard_first):
    result, _, _ = collect(native_io)
    guard = result.guard_episode(now=2)
    native_io()
    train = teacher.collect_task_teacher().training_episode()
    monkeypatch.setattr(
        stream, "fit_training_standardization", lambda *_: pytest.fail("guard stream reached fit")
    )
    with pytest.raises(ValueError, match="TRAIN"):
        stream.run_continuous_demo(*((guard, train) if guard_first else (train, guard)))


def test_should_refuse_guard_packets_and_labels_in_existing_training_runtime_without_charge(
    native_io,
):
    result, _, _ = collect(native_io)
    episode = result.guard_episode(now=2)
    stats = ObservationStandardization((0.0,) * 3, (1.0,) * 3, 1)
    source = NumpyNearestContinuousBCLearner(NearestContinuousBCPolicy(episode.spec, stats))
    source.train_batch(np.zeros((1, 3)), np.zeros((1, 4)))
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor_version="warm",
        candidate_version="private",
        clock=LogicalClock(2),
        budget=budget,
        max_experiences=1,
        max_consolidations=0,
    )
    before = runtime.candidate_snapshot().state
    packet = Experience(
        "guard",
        episode.episode_id,
        0,
        "warm",
        np.zeros((1, 3)),
        "inner_guard",
        ExperiencePermissions(),
    )
    label = LabelArrival(
        "guard-label", "guard", episode.episode_id, 2, "warm", np.zeros((1, 4)), role="inner_guard"
    )
    with pytest.raises(ValueError):
        runtime.record_experience(packet)
    with pytest.raises(ValueError):
        runtime.record_label(label)
    assert runtime.train_ready() == () and budget.updates_completed == 0
    after = runtime.candidate_snapshot().state
    np.testing.assert_array_equal(before.prototype_features, after.prototype_features)
    np.testing.assert_array_equal(before.prototype_actions, after.prototype_actions)
    assert before.completed_updates == after.completed_updates == 1


@pytest.mark.parametrize("seed", [31, 37])
@pytest.mark.parametrize("visual", [False, True])
def test_should_emit_guard_only_cli_receipt_without_changing_train_default_calls(
    native_io, monkeypatch, tmp_path, seed, visual
):
    result, _, _ = collect(native_io, seed=seed, visual=visual)
    calls = []
    monkeypatch.setattr(
        cli, "collect_guard_teacher", lambda **kwargs: calls.append(kwargs) or result
    )
    monkeypatch.setattr(
        cli, "collect_task_teacher", lambda **_: pytest.fail("guard CLI called TRAIN collector")
    )
    monkeypatch.setattr(cli, "version", lambda _: "fake-version")
    monkeypatch.setattr(cli.faulthandler, "enable", lambda: None)
    output = tmp_path / "guard.json"
    arguments = [
        "teacher",
        "--output",
        str(output),
        "--guard-seed",
        str(seed),
        "--labels-arrived-at",
        "2",
    ]
    if visual:
        arguments.append("--visual-observation")
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 0
    expected = dict(guard_seed=seed, labels_arrived_at=2)
    if visual:
        expected["visual_observation"] = True
    assert calls == [expected]
    payload = json.loads(output.read_text(encoding="utf8"))
    assert payload["qualified"] is True and payload["role"] == "inner_guard"
    assert payload["seed"] == payload["guard_seed"] == seed and payload["labels_arrived_at"] == 2
    assert payload["context"] == f"pick-cube-inner-guard-{seed}"
    assert payload["candidate_episode"]["role"] == "inner_guard"
    assert payload["optimizer_updates"] == 0
    with pytest.raises(ValueError, match="TRAIN"):
        stream_cli._read_teacher(output, 17, visual_observation=visual)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        cli.main()
    assert output.read_bytes() == original and len(calls) == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--guard-seed", "31"],
        ["--guard-seed", "37", "--labels-arrived-at", "-1"],
        ["--train-seed", "17", "--labels-arrived-at", "2"],
        ["--labels-arrived-at", "2"],
        ["--train-seed", "19", "--guard-seed", "31", "--labels-arrived-at", "2"],
        ["--guard-seed", "17", "--labels-arrived-at", "2"],
        ["--guard-seed", "23", "--labels-arrived-at", "2"],
        ["--guard-seed", "29", "--labels-arrived-at", "2"],
        ["--guard-seed", "37", "--labels-arrived-at", "1.0"],
    ],
)
def test_should_refuse_invalid_guard_cli_combinations_before_output_or_collection(
    monkeypatch, tmp_path, extra
):
    output = tmp_path / "not-created.json"
    monkeypatch.setattr(sys, "argv", ["teacher", "--output", str(output), *extra])
    monkeypatch.setattr(
        cli, "collect_guard_teacher", lambda **_: pytest.fail("invalid CLI reached guard collector")
    )
    monkeypatch.setattr(
        cli, "collect_task_teacher", lambda **_: pytest.fail("invalid CLI reached TRAIN collector")
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and not output.exists()


def test_should_preserve_train_cli_no_argument_collector_call(native_io, monkeypatch, tmp_path):
    native_io()
    result = teacher.collect_task_teacher()
    calls = []
    monkeypatch.setattr(
        cli, "collect_task_teacher", lambda **kwargs: calls.append(kwargs) or result
    )
    monkeypatch.setattr(
        cli, "collect_guard_teacher", lambda **_: pytest.fail("default called guard collector")
    )
    monkeypatch.setattr(cli, "version", lambda _: "fake-version")
    monkeypatch.setattr(cli.faulthandler, "enable", lambda: None)
    output = tmp_path / "train.json"
    monkeypatch.setattr(sys, "argv", ["teacher", "--output", str(output)])
    assert cli.main() == 0 and calls == [{}]
    payload = json.loads(output.read_text())
    assert payload["role"] == "train" and payload["seed"] == 17
    assert payload["guard_seed"] is payload["labels_arrived_at"] is None


def test_should_preserve_failed_guard_cli_receipt_and_work(native_io, monkeypatch, tmp_path):
    environment, _ = native_io()
    environment.step_failure_at = 2
    result = teacher.collect_guard_teacher(guard_seed=31, labels_arrived_at=2)
    monkeypatch.setattr(cli, "collect_guard_teacher", lambda **_: result)
    monkeypatch.setattr(cli, "version", lambda _: "fake-version")
    monkeypatch.setattr(cli.faulthandler, "enable", lambda: None)
    output = tmp_path / "failed-guard.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["teacher", "--output", str(output), "--guard-seed", "31", "--labels-arrived-at", "2"],
    )
    assert cli.main() == 1
    raw = json.loads(output.read_text())
    assert raw["qualified"] is False and raw["role"] == "inner_guard"
    assert raw["guard_seed"] == 31 and raw["labels_arrived_at"] == 2
    assert raw["work"] == asdict(port.ManiSkillWork(2, 1, 5, 5))
    assert "physics failed" in raw["error"] and raw["candidate_episode"] is None
