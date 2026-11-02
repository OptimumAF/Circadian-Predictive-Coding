"""New #414 literal query/label/history boundaries, no simulator or model RNG."""

from collections import defaultdict
from dataclasses import replace

import pytest
import numpy as np
from typing import cast

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_probe as probe
from src.adapters import numpy_continuous_bc as bp
from src.core import continuous_bc as math_core
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from test_state_quaternion_robot_learning import QuaternionEnvironment

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from src.core import pick_cube_probe_corrections as data
from src.core.pick_cube_teacher import PickCubeGeometry
from src.core.pick_cube_readiness_teacher import GoalConsistentObservedGraspPickCubeTeacher
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries

WORK: defaultdict[str, int] = defaultdict(int)
CAPS = dict(ports=96, geometry=256, teacher=256, projections=512, controls=96, forbidden=0)
CAPS.update(
    scenes=48,
    sources=32,
    updates=8,
    update_rows=16,
    gradients=32,
    gradient_rows=64,
    forwards=128,
    forward_rows=512,
    snapshots=65536,
    snapshot_bytes=256 * 1024**2,
)
NATIVE_INIT, NATIVE_STEP = (
    native.ManiSkillPickCubeEpisode.__init__,
    native.ManiSkillPickCubeEpisode.step,
)


def charge(key, n=1):
    WORK[key] += n
    assert WORK[key] <= CAPS[key], "permanent414literalcap:" + key


def forbidden(*args, **options):
    charge("forbidden")


def episode(port):
    return ContinuousEpisode(
        port._episode_id, port._context, port._role, port._spec, tuple(port._records)
    )


@pytest.fixture(autouse=True)
def meter(monkeypatch):
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "__init__", forbidden)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", forbidden)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "episode", episode)
    decide = GoalConsistentObservedGraspPickCubeTeacher.decide
    project = data._action_history_episode

    def geometry(port):
        charge("geometry")
        return PickCubeGeometry((0.0, 0.0, 0.2), (0.05, 0.0, 0.02), (0.1, 0.0, 0.2))

    def decided(*args, **options):
        charge("teacher")
        return decide(*args, **options)

    def projected(source, *args, **options):
        charge("projections", 2 * len(source.transitions))
        return project(source, *args, **options)

    monkeypatch.setattr(queries, "_geometry", geometry)
    monkeypatch.setattr(GoalConsistentObservedGraspPickCubeTeacher, "decide", decided)
    monkeypatch.setattr(data, "_action_history_episode", projected)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    for module, name, key, rows in (
        (math_core, "_forward", "forwards", "forward_rows"),
        (math_core, "continuous_bc_loss_and_gradients", "gradients", "gradient_rows"),
        (bp.NumpyContinuousBCLearner, "train_batch", "updates", "update_rows"),
    ):
        original = getattr(module, name)

        def counted(*args, _original=original, _key=key, _rows=rows, **options):
            charge(_key)
            charge(_rows, len(args[1] if _key == "updates" else args[0]))
            return _original(*args, **options)

        monkeypatch.setattr(module, name, counted)
    fork, snapshot = bp.NumpyContinuousBCLearner.fork, bp._snapshot

    def forked(owner):
        charge("sources")
        return fork(owner)

    def copied(*args, **options):
        charge("snapshots")
        result = snapshot(*args, **options)
        charge("snapshot_bytes", sum(x.nbytes for x in result.parameters + result.velocity))
        return result

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forked)
    monkeypatch.setattr(bp, "_snapshot", copied)


def port(*, role="probe_train", seed=23):
    charge("ports")
    value = object.__new__(native.ManiSkillPickCubeEpisode)
    value._seed, value._role = seed, role
    value._context = (
        pick_cube_probe_binding("probe_train").context
        if role != "train"
        else "pick-cube"
        if seed == 17
        else f"pick-cube-train-{seed}"
    )
    value._episode_id = data.STUDENT_ID if role != "train" else f"pick-cube-smoke-{seed}"
    value._closed = value._failed = False
    value._records = []
    value._attempts = value._completed = 0
    value._physics_ratio = 5
    value._spec = ContinuousActionSpec(42, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    value._observation = (0.0,) * 36 + (0.05, 0.0, -0.18, 0.05, 0.0, 0.18)
    return value


def advance(value, *, done=False):
    charge("controls")
    raw = list(value._observation)
    raw[36] -= 0.01
    row = ContinuousTransition(
        len(value._records),
        value._observation,
        (0.125, 0.0, 0.0, -0.25),
        0.0,
        tuple(raw),
        done,
        False,
    )
    value._records.append(row)
    value._observation = row.next_observation
    value._attempts += 1
    value._completed += 1


def completed():
    value = port()
    query = queries.ProbeTrainCorrectionQuery(data.ProbeCorrectionConsent(2))
    for i in range(2):
        query.query(value, i, value.observation)
        advance(value, done=i == 1)
    return value, query


def test_fixed23_queries_keep_student_actions_and_release_arrived_labels():
    value, query = completed()
    raw = episode(value)
    batch = query.finish_probe(raw)
    assert batch.episode == raw
    assert batch.episode.transitions[0].action != batch.labels[0].target
    with pytest.raises(ValueError, match="not all arrived"):
        batch.training_pairs(now=2)
    x, y = batch.training_pairs(now=3)
    assert len(x) == len(y) == 2
    assert x[1][7:11] == raw.transitions[0].action
    assert y == tuple(label.target for label in batch.labels)
    assert batch.model_episode().transitions[0].action == raw.transitions[0].action
    report = query.report()
    assert report["provenance"] == data.PROVENANCE
    assert report["teacher_actions"] == report["geometry_returns"] == 2
    assert report["role"] == "probe_train" and "train_seed" not in report
    with pytest.raises(ValueError, match="releases no usable"):
        query.finish_probe(raw)


@pytest.mark.parametrize(
    "options",
    [
        dict(role="train"),
        dict(role="probe_test"),
        dict(role="final"),
        dict(seed=24),
        dict(seed=17),
        dict(seed=True),
        dict(max_queries=True),
        dict(max_queries=0),
        dict(max_queries=51),
    ],
)
def test_new_query_consent_refuses_wrong_roles_seeds_or_unbounded_work(options):
    args = dict(max_queries=2) | options
    with pytest.raises(ValueError):
        data.ProbeCorrectionConsent(**args)


@pytest.mark.parametrize("changed", ["role", "seed", "identity", "observation", "step"])
def test_new_query_refuses_foreign_current_port_before_teacher_work(changed):
    value = port()
    query = queries.ProbeTrainCorrectionQuery(data.ProbeCorrectionConsent(2))
    raw, step = value.observation, 0
    if changed == "role":
        value._role = "probe_test"
    if changed == "seed":
        value._seed = 24
    if changed == "identity":
        value._episode_id = "foreign"
    if changed == "observation":
        raw = (1.0,) + raw[1:]
    if changed == "step":
        step = 1
    start = WORK["teacher"]
    with pytest.raises(ValueError):
        query.query(value, step, raw)
    assert WORK["teacher"] == start
    assert query.report()["failed"] is True


def test_query_cap_is_permanent_and_no_partial_batch_is_released():
    value = port()
    query = queries.ProbeTrainCorrectionQuery(data.ProbeCorrectionConsent(1))
    query.query(value, 0, value.observation)
    advance(value)
    with pytest.raises(ValueError, match="spent"):
        query.query(value, 1, value.observation)
    advance(value, done=True)
    with pytest.raises(ValueError, match="releases no usable"):
        query.finish_probe(episode(value))
    assert query.report()["teacher_actions"] == 1


def test_lost_geometry_preserves_attempt_and_permanently_refuses_retry(monkeypatch):
    value = port()
    query = queries.ProbeTrainCorrectionQuery(data.ProbeCorrectionConsent(2))

    def lost(*args):
        charge("geometry")
        raise RuntimeError("lost geometry")

    monkeypatch.setattr(queries, "_geometry", lost)
    with pytest.raises(RuntimeError):
        query.query(value, 0, value.observation)
    with pytest.raises(ValueError):
        query.query(value, 0, value.observation)
    report = query.report()
    assert report["geometry_attempts"] == 1 and report["geometry_returns"] == 0
    assert report["teacher_actions"] == 0


def test_late_port_corruption_retains_returned_label_and_paid_teacher_work(monkeypatch):
    value = port()
    query = queries.ProbeTrainCorrectionQuery(data.ProbeCorrectionConsent(2))
    decide = GoalConsistentObservedGraspPickCubeTeacher.decide

    def corrupt(*args, **options):
        result = decide(*args, **options)
        value._role = "probe_test"
        return result

    monkeypatch.setattr(GoalConsistentObservedGraspPickCubeTeacher, "decide", corrupt)
    with pytest.raises(ValueError):
        query.query(value, 0, value.observation)
    report = query.report()
    assert report["teacher_actions"] == 1
    assert len(report["labels"]) == 1
    assert report["failed"] is True


@pytest.mark.parametrize(
    "field,value", [("max_queries", True), ("seed", 24), ("role", "probe_test")]
)
def test_corrupted_consent_refuses_before_query(field, value):
    current = port()
    query = queries.ProbeTrainCorrectionQuery(data.ProbeCorrectionConsent(2))
    object.__setattr__(query._admitted_consent, field, value)
    with pytest.raises(ValueError):
        query.query(current, 0, current.observation)
    assert query.report()["teacher_actions"] == 0


def test_corrupted_raw_grasp_flag_or_label_is_refused_at_release():
    value, query = completed()
    batch = query.finish_probe(episode(value))
    bad = replace(batch.labels[0], target=(0.1, 0.0, 0.0, 1.0))
    object.__setattr__(batch, "labels", (bad, batch.labels[1]))
    with pytest.raises(ValueError, match="paid probe correction evidence changed"):
        batch.training_pairs(now=3)
    query2 = queries.ProbeTrainCorrectionQuery(data.ProbeCorrectionConsent(2))
    value2 = port()
    query2.query(value2, 0, value2.observation)
    object.__setattr__(query2._labels[0], "target", bad.target)
    advance(value2, done=True)
    with pytest.raises(ValueError):
        query2.finish_probe(episode(value2))
    assert query2.report()["teacher_actions"] == 1


@pytest.mark.parametrize("seed", [17, 19, 20])
def test_original_train_query_and_batch_keep_original_authority(seed):
    value = port(role="train", seed=seed)
    query = queries.PickCubeCorrectionQuery(
        seed,
        readiness_corrections=True,
        observed_grasp_corrections=True,
        goal_consistent_corrections=True,
    )
    query.query(value, 0, value.observation)
    advance(value, done=True)
    batch = query.finish(episode(value))
    assert batch.train_seed == seed and batch.episode.role == "train"
    assert query.report()["train_seed"] == seed


def test_original_train_constructor_still_refuses_reserved_seed():
    with pytest.raises(ValueError):
        queries.PickCubeCorrectionQuery(23)


@pytest.mark.parametrize(
    "field,value", [("_role", "probe_test"), ("_seed", 24), ("_closed", True), ("_completed", 0)]
)
def test_late_episode_getter_corruption_refuses_batch(monkeypatch, field, value):
    current, query = completed()
    original = episode(current)

    def changed(port):
        setattr(port, field, value)
        return original

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "episode", changed)
    with pytest.raises(ValueError):
        query.finish_probe(original)
    assert query.report()["teacher_actions"] == 2


def test_work_getter_cannot_change_role_before_privileged_geometry(monkeypatch):
    current = port()
    query = queries.ProbeTrainCorrectionQuery(data.ProbeCorrectionConsent(2))
    work = native.ManiSkillPickCubeEpisode.__dict__["work"].fget

    def changed(owner):
        owner._role = "probe_test"
        return work(owner)

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "work", property(changed))
    before = WORK["geometry"]
    with pytest.raises(ValueError):
        query.query(current, 0, current.observation)
    assert WORK["geometry"] == before


def test_equal_paid_label_replacement_refuses_more_teacher_work():
    current = port()
    query = queries.ProbeTrainCorrectionQuery(data.ProbeCorrectionConsent(2))
    query.query(current, 0, current.observation)
    advance(current)
    query._labels[0] = replace(query._labels[0])
    before = WORK["teacher"]
    with pytest.raises(ValueError):
        query.query(current, 1, current.observation)
    assert WORK["teacher"] == before


def learner():
    charge("sources")
    owner = object.__new__(bp.NumpyContinuousBCLearner)
    policy = bp.ContinuousBCPolicy(cube_taskspace_spec(), hidden_dim=2, binary_gripper_axis=3)
    parameters = cast(
        bp.Parameters,
        (np.zeros((12, 2)), np.zeros(2), np.zeros((2, 4)), np.array([0.0, 0.0, 0.0, -0.4])),
    )
    owner._policy = policy
    owner._state = bp.ContinuousBCSnapshot(
        policy, parameters, cast(bp.Parameters, tuple(np.zeros_like(a) for a in parameters)), 2, 4
    )
    return owner


def run_corrected(owner, batch):
    training = batch.model_episode()
    evaluation = replace(training, episode_id="literal-heldout24", role="probe_test")
    return probe.run_continuous_probe(
        owner,
        probe.ContinuousProbePlan(training, evaluation),
        correction_batch=batch,
        correction_release_tick=3,
    )


def test_arrived_corrections_use_existing_delayed_probe_without_source_update(monkeypatch):
    current, query = completed()
    batch = query.finish_probe(episode(current))
    owner = learner()
    before = owner.snapshot_state()
    expected = batch.training_pairs(now=3)[1]
    train = bp.NumpyContinuousBCLearner.train_batch
    seen = []

    def trained(private, x, y):
        assert private is not owner
        assert tuple(tuple(row) for row in y) == expected
        seen.append(1)
        return train(private, x, y)

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", trained)
    result = run_corrected(owner, batch)
    assert result["complete"], result.get("error")
    assert len(seen) == 1 and result["progress"]["updates_completed"] == 1
    assert result["plan"]["corrections"]["labels_released_at"] == 3
    assert result["events"][0]["deferred_exact"] is True
    assert result["events"][0]["success_validated"] is True
    assert probe.same_continuous_state(owner.snapshot_state(), before)


def test_new_frozen_collection_returns_separate_targets_then_private_probe(monkeypatch, tmp_path):
    environment = QuaternionEnvironment()
    environment.after_step = lambda: setattr(
        environment, "bad_reward", np.array([float(len(environment.commands) == 2)])
    )

    def factory():
        charge("scenes")
        charge("ports")
        return environment

    def stepped(port, action):
        charge("controls")
        return NATIVE_STEP(port, action)

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "__init__", NATIVE_INIT)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", stepped)
    monkeypatch.setattr(smoke, "version", lambda _: "literal-software-only")
    owner = learner()
    before = owner.snapshot_state()
    result = smoke.run_frozen_rollout(
        tmp_path / "new.json",
        learner=owner,
        probe_train_seed=23,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        probe_correction_consent=data.ProbeCorrectionConsent(2),
    )
    assert result.report["complete"], result.report.get("error") or result.report.get(
        "correction_error"
    )
    assert type(result.corrections) is data.ProbeCorrectionBatch
    assert result.report["optimizer_updates"] == 0
    assert result.report["corrections"]["teacher_actions"] == 2
    assert len(environment.commands) == 2
    assert result.corrections.episode.transitions[0].action != result.corrections.labels[0].target
    updated = run_corrected(owner, result.corrections)
    assert updated["complete"], updated.get("error")
    assert probe.same_continuous_state(owner.snapshot_state(), before)


@pytest.mark.parametrize("change", ["clear", "append"])
def test_corrective_sgd_preserves_all_paid_returns_after_history_corruption(monkeypatch, change):
    current, query = completed()
    batch = query.finish_probe(episode(current))
    owner = learner()
    before = owner.snapshot_state()
    training = batch.model_episode()
    evaluation = replace(training, episode_id="literal-readout24", role="probe_test")
    update = probe._RecordedProbeBP.train_batch

    def changed(port, *args, **options):
        result = update(port, *args, **options)
        if len(port.reports) == 2:
            if change == "clear":
                port.reports.clear()
            else:
                port.reports.append(dict(port.reports[-1]))
        return result

    monkeypatch.setattr(probe._RecordedProbeBP, "train_batch", changed)
    result = probe.run_continuous_probe(
        owner,
        probe.ContinuousProbePlan(training, evaluation, updates=2),
        correction_batch=batch,
        correction_release_tick=3,
    )
    assert result["complete"] is False
    assert len(result["native_updates"]) == len(result["applied_updates"]) == 2
    assert result["work"]["training_rows_native_returned"] == 4
    assert result["work"]["training_attempts"] == 2
    assert probe.same_continuous_state(owner.snapshot_state(), before)


@pytest.mark.parametrize(
    "options",
    [
        dict(probe_test_seed=24),
        dict(probe_train_seed=23, corrective_queries=True),
        dict(probe_train_seed=24),
    ],
)
def test_new_query_mode_refuses_nontraining_or_mixed_requests_before_scene(tmp_path, options):
    owner = learner()
    before = WORK["scenes"]
    result = smoke.run_frozen_rollout(
        tmp_path / "refused.json",
        learner=owner,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        probe_correction_consent=data.ProbeCorrectionConsent(2),
        **options,
    )
    assert result.report["complete"] is False
    assert result.corrections is None
    assert WORK["scenes"] == before
