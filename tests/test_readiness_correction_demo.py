"""Distinct readiness labels through the existing arrived-label runtime and fake IO.

No native physics, model selection or gain claim. The prospective tiny fixture
budget is 52 neural updates/104 rows plus two nearest appends/four rows; eight
neural constructor entries draw 73 normal values, one nearest initializer draws
none. There are 12 external prediction rows and at most 175 PC energy calls/140
latent proposals. Fourteen fake scenes perform 33 controls, 27 geometry attempts
(26 returns) and 25 readiness decisions. No fitting or direct replay.
"""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import sys
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
)
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_corrections import (
    PickCubeCorrectionBatch,
    PickCubeCorrectionLabel,
    PROVENANCE as FIXED_PROVENANCE,
)
from src.core.pick_cube_public_corrections import public_vector_correction_view
from src.core.pick_cube_public_vectors import public_vector_projection
from src.core.pick_cube_readiness_corrections import (
    PROVENANCE,
    ReadinessCorrectionBatch,
    ReadinessCorrectionLabel,
)
from src.core.pick_cube_readiness_teacher import DEFINITION, ReadinessPickCubeTeacher
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_maniskill_bc_smoke import FakeEnvironment
from test_maniskill_rgb_observations import RGBEnvironment


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc", "nearest")


def episode(dimension=2):
    spec = ContinuousActionSpec(dimension, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)

    def observation(step):
        return (float(step),) + (0.0,) * (min(dimension, 21) - 1) + (0.25,) * max(0, dimension - 21)

    records = tuple(
        ContinuousTransition(
            i,
            observation(i),
            (0.125, -0.25, 0.5, -0.5),
            float(i == 1),
            observation(i + 1),
            i == 1,
            False,
        )
        for i in range(2)
    )
    return ContinuousEpisode("pick-cube-smoke-19", "pick-cube-train-19", "train", spec, records)


def readiness_batch(dimension=2):
    value = episode(dimension)
    labels = (
        ReadinessCorrectionLabel(
            0, value.transitions[0].observation, (0.0, 0.0, -0.8, 1.0), "descend"
        ),
        ReadinessCorrectionLabel(
            1, value.transitions[1].observation, (0.0, 0.0, 0.0, -1.0), "grasp"
        ),
    )
    return ReadinessCorrectionBatch(19, value, labels)


def fixed_batch():
    value = episode()
    return PickCubeCorrectionBatch(
        19,
        value,
        tuple(
            PickCubeCorrectionLabel(i, row.observation, (0.5, 0.0, 1.0, 1.0), "align")
            for i, row in enumerate(value.transitions)
        ),
    )


def assert_state(first, second):
    assert first.policy == second.policy
    assert (
        first.completed_updates == second.completed_updates
        and first.training_examples == second.training_examples
    )
    if hasattr(first, "prototype_features"):
        np.testing.assert_array_equal(first.prototype_features, second.prototype_features)
        np.testing.assert_array_equal(first.prototype_actions, second.prototype_actions)
        return
    for left, right in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(left, right)
    wake = getattr(first, "wake_state", None)
    other = getattr(second, "wake_state", None)
    if wake is None:
        assert other is None
    else:
        assert other is not None
        assert wake.wake_updates == other.wake_updates
        np.testing.assert_array_equal(wake.chemical_state, other.chemical_state)
        np.testing.assert_array_equal(wake.plasticity_state, other.plasticity_state)


@pytest.mark.parametrize(
    "step,stage,target",
    [
        (0, "settle", (0, 0, 0, -1)),
        (1, "grasp", (0, 0, 0, -1)),
        (49, "align", (0.5, 0, 1, 1)),
        (0, "lift", (1, 0, 0.5, -1)),
        (20, "descend", (0, 0, -0.8, 1)),
    ],
)
def test_should_bind_commands_to_readiness_stage_not_elapsed_schedule(step, stage, target):
    label = ReadinessCorrectionLabel(step, (1, 2), target, stage)
    assert label.stage == stage and label.target == tuple(float(value) for value in target)
    assert label.observed_at == 2 * step and label.labels_arrived_at == 2 * step + 1
    with pytest.raises(FrozenInstanceError):
        setattr(label, "step", 1)
    assert PROVENANCE != FIXED_PROVENANCE


@pytest.mark.parametrize(
    "change",
    [
        dict(step=True),
        dict(step=50),
        dict(stage="unknown"),
        dict(observation=[0, 0]),
        dict(target=(0, 0, 0)),
        dict(target=(2, 0, 0, 1)),
        dict(target=(0, 0, 0, False)),
        dict(stage="grasp", target=(0.1, 0, 0, -1)),
        dict(stage="settle", target=(0, 0, 0.1, -1)),
        dict(stage="lift", target=(0, 0, 0, 1)),
        dict(target=(float("nan"), 0, 0, 1)),
    ],
)
def test_should_refuse_invalid_stage_command_or_native_label_fields(change):
    values: dict[str, Any] = dict(step=0, observation=(0, 0), target=(0, 0, 0, 1), stage="align")
    values.update(change)
    with pytest.raises(ValueError):
        ReadinessCorrectionLabel(**values)


def test_should_release_only_arrived_owned_pairs_without_rewriting_executed_actions():
    source = episode()
    first = ReadinessCorrectionLabel(
        0, source.transitions[0].observation, (0, 0, -0.8, 1), "descend"
    )
    second = ReadinessCorrectionLabel(1, source.transitions[1].observation, (0, 0, 0, -1), "grasp")
    batch = ReadinessCorrectionBatch(19, source, (first, second))
    before = asdict(batch)
    with pytest.raises(ValueError, match="arrived"):
        batch.training_pairs(now=2)
    observations, targets = batch.training_pairs(now=3)
    assert observations == ((0.0, 0.0), (1.0, 0.0))
    assert targets == ((0.0, 0.0, -0.8, 1.0), (0.0, 0.0, 0.0, -1.0))
    assert targets != tuple(row.action for row in batch.episode.transitions)
    object.__setattr__(first, "target", (0.5, 0.0, 0.5, 1.0))
    object.__setattr__(source.transitions[0], "reward", 0.5)
    assert asdict(batch) == before
    object.__setattr__(batch.labels[1], "observation", (99.0, 0.0))
    with pytest.raises(ValueError):
        batch.training_pairs(now=3)


@pytest.mark.parametrize("direction", ["fixed_into_readiness", "readiness_into_fixed"])
def test_should_reject_cross_family_labels_even_if_record_fields_are_structurally_similar(
    direction,
):
    batch: Any
    if direction == "fixed_into_readiness":
        batch = fixed_batch()
        with pytest.raises(ValueError):
            ReadinessCorrectionBatch(19, batch.episode, batch.labels)
    else:
        batch = readiness_batch()
        with pytest.raises(ValueError):
            PickCubeCorrectionBatch(19, batch.episode, batch.labels)


@pytest.mark.parametrize(
    "fault", ["role", "identity", "context", "seed", "label_order", "label_count"]
)
def test_should_refuse_wrong_executed_trace_or_batch_binding(fault):
    batch = readiness_batch()
    value, labels, seed = batch.episode, batch.labels, 19
    if fault == "role":
        value = replace(value, role="final_test")
    elif fault == "identity":
        value = replace(value, episode_id="readiness-expert-teacher-19")
    elif fault == "context":
        value = replace(value, context="pick-cube")
    elif fault == "seed":
        seed = True
    elif fault == "label_order":
        labels = tuple(reversed(labels))
    else:
        labels = labels[:1]
    with pytest.raises(ValueError):
        ReadinessCorrectionBatch(seed, value, labels)


def test_should_project_public21_pairing_in_same_readiness_family_and_keep_original_arrival_gate():
    batch = readiness_batch(789)
    before = asdict(batch)
    view = public_vector_correction_view(
        batch, public_vector_projection(), source_recipe=pick_cube_rgb_recipe()
    )
    assert type(view.source) is type(view.model) is ReadinessCorrectionBatch
    assert (
        view.source.episode.spec.observation_dim == 789
        and view.model.episode.spec.observation_dim == 21
    )
    assert all(type(label) is ReadinessCorrectionLabel for label in view.model.labels)
    assert [(label.observed_at, label.labels_arrived_at) for label in view.model.labels] == [
        (0, 1),
        (2, 3),
    ]
    with pytest.raises(ValueError):
        view.model.training_pairs(now=2)
    assert view.model.training_pairs(now=3)[1] == batch.training_pairs(now=3)[1]
    assert view.work.observation_projections == 4 and view.work.source_values_validated == 3156
    assert asdict(batch) == before


@pytest.fixture(scope="module")
def sources():
    spec = episode().spec
    stats = ObservationStandardization((0.5, 0.0), (0.5, 1.0), 2)
    features = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    targets = np.array([[0.2, -0.2, 0.4, -1.0], [-0.3, 0.5, -0.2, 1.0]], dtype=np.float64)
    output = {}
    for family in FAMILIES:
        base = bp.ContinuousBCPolicy(spec, hidden_dim=2, seed=17, observation_standardization=stats)
        model: Any
        if family == "bp":
            model = bp.NumpyContinuousBCLearner(base)
        elif family == "nearest":
            model = NumpyNearestContinuousBCLearner(
                NearestContinuousBCPolicy(spec, stats, max_prototypes=8)
            )
        else:
            config = (
                None
                if family == "pc"
                else CoupledCircadianConfig.matched_pc_control()
                if family == "neutral-cpc"
                else CoupledCircadianConfig()
            )
            model = pc.NumpyContinuousPCLearner(pc.ContinuousPCPolicy(base, circadian=config))
        model.train_batch(features, targets)
        output[family] = model
    return output


@pytest.fixture(scope="module")
def completed(sources):
    results, runtimes = {}, []
    original = demo.ActorShadowRuntime
    with pytest.MonkeyPatch.context() as patch:

        def runtime(owner, **options):
            created = original(owner, **options)
            runtimes.append(created)
            return created

        patch.setattr(demo, "ActorShadowRuntime", runtime)
        for family, source in sources.items():
            initial = source.snapshot_state()
            batch = readiness_batch()
            before = asdict(batch)
            result = demo.run_continuous_correction_update(source, batch, now=3)
            assert result.report["complete"] is True and asdict(batch) == before
            results[family] = (source, initial, result)
    return results, runtimes


@pytest.mark.parametrize("family", FAMILIES)
def test_should_continue_existing_native_learner_and_preserve_source_actor_and_original_label_ticks(
    completed, family
):
    results, runtimes = completed
    source, initial, result = results[family]
    runtime = runtimes[FAMILIES.index(family)]
    updates = 1 if family == "nearest" else 8
    report = result.report
    assert type(result.candidate) is type(source)
    assert result.candidate is not source
    assert_state(initial, source.snapshot_state())
    assert_state(initial, runtime.actor.snapshot_state().state)
    final = result.candidate.snapshot_state()
    assert_state(final, runtime.candidate_snapshot().state)
    assert final.completed_updates == 1 + updates and final.training_examples == 2 + 2 * updates
    assert report["provenance"] == PROVENANCE and report["teacher_definition"] == DEFINITION
    assert report["progress"]["updates_completed"] == updates
    assert report["settings"]["max_training_updates"] == updates
    assert report["work"]["fit_calls"] == report["work"]["reference_replay_rows"] == 0
    for index, receipt in enumerate(report["applied_updates"]):
        assert receipt["sample_id"] == f"corrective-batch-{index}"
        assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
        assert receipt["episode_id"] == "pick-cube-smoke-19"
        assert receipt["update_number"] == index + 1
    if family in ("pc", "neutral-cpc", "active-cpc"):
        assert len(report["native_updates"]) == 8
        assert report["work"]["native_update_counts"]["published_updates"] == 8
        if family != "pc":
            assert final.wake_state.wake_updates == 9
    if family == "nearest":
        np.testing.assert_array_equal(final.prototype_features[:2], initial.prototype_features)
        np.testing.assert_array_equal(final.prototype_actions[:2], initial.prototype_actions)
        np.testing.assert_array_equal(
            final.prototype_actions[2:], [[0.0, 0.0, -0.8, 1.0], [0.0, 0.0, 0.0, -1.0]]
        )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault", ["future", "heldout", "cross_labels", "source_role", "statistics"]
)
def test_should_refuse_unreleased_or_corrupt_readiness_inputs_before_model_forks(
    sources, monkeypatch, fault
):
    source = sources["bp"].fork()
    batch = readiness_batch()
    now = 3
    if fault == "future":
        now = 2
    elif fault == "heldout":
        object.__setattr__(batch.episode, "role", "probe_train")
    elif fault == "cross_labels":
        object.__setattr__(batch, "labels", fixed_batch().labels)
    else:
        state = source.snapshot_state()
        policy = replace(
            state.policy,
            **(
                {"training_role": "probe_train"}
                if fault == "source_role"
                else {"observation_standardization": None}
            ),
        )
        source._policy, source._state = policy, replace(state, policy=policy)
    monkeypatch.setattr(
        bp.NumpyContinuousBCLearner,
        "fork",
        lambda *_: pytest.fail("unreleased readiness labels forked model"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(source, batch, now=now)


class TwoDimEnvironment(FakeEnvironment):
    def reset(self, *, seed):
        observation, info = super().reset(seed=seed)
        return observation[:, :2], info

    def step(self, command):
        observation, *rest = super().step(command)
        return (observation[:, :2], *rest)


def readiness_geometries():
    return (
        PickCubeGeometry((0.0, 0.0, 0.08), (0.0, 0.0, 0.0), (0.1, 0.0, 0.2)),
        PickCubeGeometry((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.1, 0.0, 0.2)),
    )


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environments, geometry_calls = [], []

    def factory():
        value = TwoDimEnvironment()
        environments.append(value)
        return value

    def geometry(port):
        geometry_calls.append(len(port._records))
        return readiness_geometries()[len(port._records)]

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(queries, "_geometry", geometry)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-readiness-corrections")
    return environments, geometry_calls, tmp_path


def test_should_query_geometry_readiness_at_actual_public_observation_and_keep_student_actions(
    boundary,
):
    environments, _, _ = boundary
    port = native.ManiSkillPickCubeEpisode(
        "pick-cube-smoke-19", "pick-cube-train-19", role="train", seed=19
    )
    query = queries.PickCubeCorrectionQuery(19, readiness_corrections=True)
    action = (0.125, -0.25, 0.5, -0.5)
    try:
        for index in range(2):
            query.query(port, index, port.observation)
            port.step(action)
        batch = query.finish(port.episode())
    finally:
        port.close()
    report = query.report()
    assert type(batch) is ReadinessCorrectionBatch
    assert [label.stage for label in batch.labels] == ["descend", "grasp"]
    assert report["provenance"] == PROVENANCE and report["teacher_definition"] == DEFINITION
    assert report["teacher_decision_steps"] == report["teacher_actions"] == 2
    assert "teacher_schedule_steps" not in report
    assert batch.training_pairs(now=3)[1][1] == (0.0, 0.0, 0.0, -1.0)
    assert all(row.action == action for row in batch.episode.transitions)
    assert len(environments[0].commands) == 2


def test_should_retain_first_query_and_paid_control_after_later_geometry_failure(
    boundary, monkeypatch
):
    environments, _, _ = boundary
    original = queries._geometry

    def failed(port):
        if port._records:
            raise RuntimeError("later geometry failed")
        return original(port)

    monkeypatch.setattr(queries, "_geometry", failed)
    port = native.ManiSkillPickCubeEpisode(
        "pick-cube-smoke-19", "pick-cube-train-19", role="train", seed=19
    )
    query = queries.PickCubeCorrectionQuery(19, readiness_corrections=True)
    try:
        query.query(port, 0, port.observation)
        port.step((0.125, -0.25, 0.5, -0.5))
        with pytest.raises(RuntimeError):
            query.query(port, 1, port.observation)
        with pytest.raises(ValueError):
            query.query(port, 1, port.observation)
        invalid_episode: Any = None
        with pytest.raises(ValueError):
            query.finish(invalid_episode)
    finally:
        port.close()
    report = query.report()
    assert report["geometry_attempts"] == 2 and report["geometry_returns"] == 1
    assert report["teacher_actions"] == len(report["labels"]) == 1 and report["batch"] is None
    assert len(environments[0].commands) == 1 and environments[0].close_calls == 1


@pytest.mark.parametrize("family", ["bp", "active-cpc"])
def test_should_collect19_update_privately_then_roll_candidate17_and19_in_requested_mode(
    sources, boundary, monkeypatch, family
):
    environments, _, directory = boundary
    options: dict[str, Any] = dict(visual_observation=False, readiness_corrections=True)
    source: Any = sources[family].fork()
    if family == "bp":
        spec = ContinuousActionSpec(21, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
        stats = ObservationStandardization((0.0,) * 21, (1.0,) * 21, 2)
        source = bp.NumpyContinuousBCLearner(
            bp.ContinuousBCPolicy(spec, hidden_dim=1, seed=17, observation_standardization=stats)
        )

        def visual_factory(**kwargs):
            assert kwargs == dict(visual_observation=True)
            value = RGBEnvironment()
            environments.append(value)
            return value

        monkeypatch.setattr(native, "_make_environment", visual_factory)
        options.update(visual_observation=True, observation_projection=public_vector_projection())
    initial = source.snapshot_state()
    destination = directory / family
    summary = smoke.run_corrective_adaptation(source, output_directory=destination, **options)
    assert summary["complete"] is summary["source_state_exact"] is True
    assert [value.reset_calls for value in environments] == [[19], [17], [19]]
    assert all(value.close_calls == 1 and len(value.commands) == 2 for value in environments)
    collection = json.loads((destination / "collection.json").read_text(encoding="utf8"))
    update = json.loads((destination / "update.json").read_text(encoding="utf8"))
    assert collection["corrections"]["provenance"] == update["provenance"] == PROVENANCE
    assert (
        collection["corrections"]["teacher_definition"]
        == update["teacher_definition"]
        == DEFINITION
    )
    assert [row["stage"] for row in collection["corrections"]["labels"]] == ["descend", "grasp"]
    assert update["progress"]["updates_completed"] == 8
    assert update["work"]["training_rows"] == 16
    if family == "bp":
        assert collection["episode"]["spec"]["observation_dim"] == 789
        assert collection["projection_exact"] is True
        assert collection["projection_work"]["projected_values"] == 42
        assert update["input_projection"]["source_spec"]["observation_dim"] == 789
        assert update["input_projection"]["model_spec"]["observation_dim"] == 21
        assert update["input_projection"]["work"]["observation_projections"] == 4
    for seed in (17, 19):
        raw = json.loads((destination / f"updated-{seed}.json").read_text(encoding="utf8"))
        assert raw["complete"] is True
        assert raw["source_updates"] == initial.completed_updates + 8
        assert raw["source_training_rows"] == initial.training_examples + 16
        assert "corrections" not in raw and raw["optimizer_updates"] == 0
        if family == "bp":
            assert raw["episode"]["spec"]["observation_dim"] == 789
            assert raw["projection_exact"] is True
            assert raw["projection_work"]["projected_values"] == 42
    assert_state(initial, source.snapshot_state())


@pytest.mark.parametrize("requested", [False, True])
def test_should_refuse_returned_batch_from_other_teacher_family_before_private_update(
    sources, monkeypatch, tmp_path, requested
):
    returned = fixed_batch() if requested else readiness_batch()
    correction_report = (
        {"provenance": FIXED_PROVENANCE, "teacher_schedule_steps": 2}
        if requested
        else {
            "provenance": PROVENANCE,
            "teacher_definition": DEFINITION,
            "teacher_decision_steps": 2,
        }
    )
    outcome = smoke.FrozenRolloutResult(
        {"complete": True, "episode": asdict(returned.episode), "corrections": correction_report},
        returned,
    )
    monkeypatch.setattr(smoke, "run_frozen_rollout", lambda *args, **kwargs: outcome)
    monkeypatch.setattr(
        smoke,
        "run_continuous_correction_update",
        lambda *args, **kwargs: pytest.fail("wrong-family batch reached optimizer"),
    )
    source = sources["bp"].fork()
    before = source.snapshot_state()
    summary = smoke.run_corrective_adaptation(
        source,
        output_directory=tmp_path / "wrong-family",
        visual_observation=False,
        readiness_corrections=requested,
    )
    assert summary["complete"] is False and summary["collection"]["complete"] is True
    assert summary["update"] is None and summary["rollouts"] == []
    assert_state(before, source.snapshot_state())


@pytest.mark.parametrize(
    "options",
    [{"readiness_corrections": True}, {"corrective_queries": True, "readiness_corrections": 1}],
)
def test_should_refuse_incomplete_or_non_boolean_readiness_request_before_scene(
    sources, boundary, options
):
    environments, _, directory = boundary
    outcome = smoke.run_frozen_rollout(
        directory / "refused.json", learner=sources["bp"].fork(), **options
    )
    assert outcome.report["complete"] is False and outcome.corrections is None
    assert outcome.report["creation_attempts"] == 0 and environments == []


def test_should_route_cli_readiness_only_with_explicit_queries_without_new_default_kwargs(
    monkeypatch, tmp_path
):
    calls = []

    def run(args, **options):
        calls.append((args, options))
        return smoke.FrozenRolloutResult({"complete": False}, None)

    monkeypatch.setattr(smoke, "_execute_frozen_rollout", run)
    destination = tmp_path / "cli.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(destination)])
    smoke.main()
    default = calls[-1][1]
    assert "readiness_corrections" not in default
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke", "--output", str(destination), "--corrective-queries", "--readiness-corrections"],
    )
    smoke.main()
    assert calls[-1][0].readiness_corrections is True
    assert calls[-1][0].corrective_queries is True


@pytest.mark.parametrize("command", [1.0, -1.0])
def test_should_count_executed_student_closes_rather_than_unexecuted_teacher_labels(
    boundary, monkeypatch, command
):
    environments, _, _ = boundary
    monkeypatch.setattr(native, "_make_environment", lambda: _long_environment(environments))
    monkeypatch.setattr(
        queries, "_geometry", lambda port: readiness_geometries()[int(bool(port._records))]
    )
    port = native.ManiSkillPickCubeEpisode(
        "pick-cube-smoke-19", "pick-cube-train-19", role="train", seed=19
    )
    query = queries.PickCubeCorrectionQuery(19, readiness_corrections=True)
    try:
        for index in range(8):
            query.query(port, index, port.observation)
            port.step((0.0, 0.0, 0.0, command))
        batch = query.finish(port.episode())
    finally:
        port.close()
    stages = [label.stage for label in batch.labels]
    assert stages[:7] == ["descend"] + ["grasp"] * 6
    assert stages[7] == ("lift" if command < 0 else "grasp")
    assert all(label.target[3] == -1 for label in batch.labels[1:])
    assert all(row.action[3] == command for row in batch.episode.transitions)
    assert len(environments[0].commands) == 8


def _long_environment(environments):
    value = TwoDimEnvironment(steps=8)
    environments.append(value)
    return value


@pytest.mark.parametrize("feedback", [True, float("nan"), float("inf"), 1.01, "-1"])
def test_should_refuse_invalid_executed_feedback_atomically(feedback):
    teacher = ReadinessPickCubeTeacher()
    initial = vars(teacher).copy()
    with pytest.raises(ValueError):
        teacher.decide(readiness_geometries()[0], executed_gripper=feedback)
    assert vars(teacher) == initial


@pytest.mark.parametrize(
    "fault",
    ["mode_during_geometry", "teacher_after_label", "label_after_label", "role_during_spec"],
)
def test_should_poison_coherently_replaced_readiness_authority_before_more_decisions(
    boundary, monkeypatch, fault
):
    environments, geometry_calls, _ = boundary
    port = native.ManiSkillPickCubeEpisode(
        "pick-cube-smoke-19", "pick-cube-train-19", role="train", seed=19
    )
    query = queries.PickCubeCorrectionQuery(19, readiness_corrections=True)
    step = 0
    try:
        if fault in ("teacher_after_label", "label_after_label"):
            query.query(port, 0, port.observation)
            port.step((0.0, 0.0, 0.0, 1.0))
            step = 1
            if fault == "teacher_after_label":
                replacement = ReadinessPickCubeTeacher()
                replacement._steps, replacement._phase = 1, "descend"
                query._teacher = replacement
            else:
                object.__setattr__(query._labels[0], "target", (0.0, 0.0, -0.5, 1.0))
        elif fault == "mode_during_geometry":
            original = queries._geometry

            def changed(value):
                geometry = original(value)
                query._readiness_corrections = False
                query._teacher = queries.PickCubeTeacher()
                return geometry

            monkeypatch.setattr(queries, "_geometry", changed)
        else:
            original_property = vars(native.ManiSkillPickCubeEpisode)["spec"]
            assert isinstance(original_property, property)
            getter = original_property.fget
            assert getter is not None

            def changed_spec(value):
                spec = getter(value)
                value._role = "final_test"
                return spec

            monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "spec", property(changed_spec))
        with pytest.raises(ValueError):
            query.query(port, step, port.observation)
        with pytest.raises(ValueError, match="failed"):
            query.query(port, step, port.observation)
    finally:
        port.close()
    report = query.report()
    expected_labels = int(fault in ("teacher_after_label", "label_after_label"))
    assert report["teacher_actions"] == len(report["labels"]) == expected_labels
    assert report["batch"] is None
    assert len(environments[0].commands) == expected_labels
    assert len(geometry_calls) == (0 if fault == "role_during_spec" else 1)
