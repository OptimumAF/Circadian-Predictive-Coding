"""Explicit fixed mass binding through causal TS12 and frozen fake native IO.

Four cached H2/D12/A4 initializations consume128 seeded normal values. No training,
fitting, head solve or latent inference is permitted. Planned fake work:19 single
row predictions,16 control attempts/15 returns across twelve bounded scenes.
Synthetic restored counters and velocities are tags, not historical training.
"""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.app.action_history_encoding import ActionHistoryEncoding
from src.core import pick_cube_action_history as history
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.pick_cube_mass_context import pick_cube_mass_context
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.infra import maniskill_continuous as native
from test_continuous_probe import assert_state
from test_mass_context_interfaces import InterfaceEnvironment


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


def raw(index):
    values = [0.0] * 42
    values[18] = float(index > 0)
    values[36:42] = [-0.02 + index * 0.005, 0.01, -0.1, 0.03, -0.04, 0.15]
    return tuple(values)


def mass_episode():
    return ContinuousEpisode(
        "pick-cube-smoke-17",
        pick_cube_mass_context().context,
        "train",
        ContinuousActionSpec(42, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4),
        tuple(
            ContinuousTransition(
                i, raw(i), (0.25, 0.0, 0.0, -1.0), float(i), raw(i + 1), i == 1, False
            )
            for i in range(2)
        ),
    )


class MassTaskspaceEnvironment(InterfaceEnvironment):
    """Native mass interface double with valid official-state taskspace inputs."""

    def __init__(self):
        super().__init__()
        self.reset_observation = np.asarray([raw(0)], dtype=np.float32)

    def step(self, command):
        result = super().step(command)
        return np.asarray([raw(len(self.commands))], dtype=np.float32), *result[1:]


def forbidden(*args, **kwargs):
    pytest.fail("request reached forbidden training, fit, inference or native creation")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        prediction_attempts=0,
        prediction_returns=0,
        prediction_rows=0,
        fake_control_attempts=0,
        fake_control_returns=0,
        SGD=0,
        GN=0,
        fits=0,
        PC_energy=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    predicts = (bp.NumpyContinuousBCLearner.predict, pc.NumpyContinuousPCLearner.predict)
    step = MassTaskspaceEnvironment.step

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def predicted(original):
        def predict(owner, features):
            counts["prediction_attempts"] += 1
            counts["prediction_rows"] += len(features)
            returned = original(owner, features)
            counts["prediction_returns"] += 1
            return returned

        return predict

    def stepped(environment, command):
        counts["fake_control_attempts"] += 1
        returned = step(environment, command)
        counts["fake_control_returns"] += 1
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted(predicts[0]))
        patch.setattr(pc.NumpyContinuousPCLearner, "predict", predicted(predicts[1]))
        patch.setattr(MassTaskspaceEnvironment, "step", stepped)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(pc.NumpyContinuousPCLearner, "train_batch_with_work", forbidden)
        patch.setattr(pc.energy, "evaluate_continuous_pc", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("TSM_FIXTURE_LEDGER"):
        with Path(os.environ["TSM_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 4 and counts["normal_draws"] == 128
    assert counts["prediction_attempts"] <= 80 and counts["prediction_rows"] <= 80
    assert counts["fake_control_attempts"] <= 80


@pytest.fixture(scope="module")
def owners(ledger):
    result: dict[str, Any] = {}
    owner: Any
    for family in FAMILIES:
        base = bp.ContinuousBCPolicy(
            ContinuousActionSpec(12, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4),
            hidden_dim=2,
            seed=17,
        )
        chemistry = (
            None
            if family == "pc"
            else CoupledCircadianConfig.matched_pc_control()
            if family == "neutral-cpc"
            else CoupledCircadianConfig()
        )
        owner = (
            bp.NumpyContinuousBCLearner(base)
            if family == "bp"
            else pc.NumpyContinuousPCLearner(pc.ContinuousPCPolicy(base, circadian=chemistry))
        )
        state = owner.snapshot_state()
        parameters = tuple(np.zeros_like(value) for value in state.parameters)
        parameters[3][:] = (0.1, -0.2, 0.3, -0.4)
        wake = getattr(state, "wake_state", None)
        tags = {} if wake is None else dict(wake_state=replace(wake, wake_updates=13))
        owner.restore_state(
            replace(
                state,
                parameters=parameters,
                velocity=tuple(np.full_like(value, 0.125) for value in state.velocity),
                completed_updates=13,
                training_examples=377,
                **tags,
            )
        )
        result[family] = owner
    return result


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment = MassTaskspaceEnvironment()
    creations = []

    def factory(**options):
        creations.append(options)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-taskspace-mass")
    return environment, creations, tmp_path / "mass-taskspace.json"


def test_should_project_explicit_mass_train_history_and_reset_each_episode():
    original, mass, recipe = (
        mass_episode(),
        pick_cube_mass_context(),
        PickCubeTaskSpaceHistoryRecipe(),
    )
    before = asdict(original)
    first = history.action_history_training_episode(original, recipe, mass_context=mass)
    second = history.action_history_training_episode(original, recipe, mass_context=mass)
    assert first == second and first is not second
    assert first.spec.observation_dim == 12
    assert (first.context, first.episode_id, first.role) == (
        mass.context,
        original.episode_id,
        "train",
    )
    assert first.transitions[0].observation[-5:] == (0.0,) * 5
    assert first.transitions[1].observation[-5:] == (0.25, 0.0, 0.0, -1.0, 1 / 6)
    assert first.transitions[1].next_observation[-5:] == (0.25, 0.0, 0.0, -1.0, 2 / 6)
    assert [row.action for row in first.transitions] == [row.action for row in original.transitions]
    assert (
        history.validate_action_history_training_episode(first, recipe, mass_context=mass) == first
    )
    assert asdict(original) == before


@pytest.mark.parametrize("encoded", [False, True])
def test_should_refuse_implicit_mass_context_in_both_training_interfaces(monkeypatch, encoded):
    recipe, episode = PickCubeTaskSpaceHistoryRecipe(), mass_episode()
    if encoded:
        episode = history.action_history_training_episode(
            episode, recipe, mass_context=pick_cube_mass_context()
        )
    operation = (
        history.validate_action_history_training_episode
        if encoded
        else history.action_history_training_episode
    )
    monkeypatch.setattr(history, "action_history_observation", forbidden)
    with pytest.raises(ValueError):
        operation(episode, recipe)


@pytest.mark.parametrize(
    "fault", ["role", "context", "ordinary_context", "seed", "multiplier", "47d", "zero"]
)
def test_should_refuse_unbound_or_unsupported_mass_training_sources(monkeypatch, fault):
    recipe: history.ActionHistoryRecipe
    value, mass, recipe = mass_episode(), pick_cube_mass_context(), PickCubeTaskSpaceHistoryRecipe()
    if fault == "role":
        value = replace(value, role="probe_train")
    elif fault == "context":
        value = replace(value, context="pick-cube-train-19")
    elif fault == "ordinary_context":
        value = replace(value, context="pick-cube")
    elif fault == "seed":
        mass = replace(mass, seed=19)
    elif fault == "multiplier":
        mass = replace(mass, mass_multiplier=3.0)
    elif fault == "47d":
        recipe = history.PickCubeActionHistoryRecipe()
    else:
        recipe = PickCubeTaskSpaceHistoryRecipe(mode="zero")
    monkeypatch.setattr(history, "action_history_observation", forbidden)
    with pytest.raises(ValueError):
        history.action_history_training_episode(value, recipe, mass_context=mass)


def test_should_refuse_coherently_joined_encoded_tail_that_does_not_follow_actual_command():
    recipe, mass = PickCubeTaskSpaceHistoryRecipe(), pick_cube_mass_context()
    encoded = history.action_history_training_episode(mass_episode(), recipe, mass_context=mass)
    tail = list(encoded.transitions[1].observation)
    tail[-5] += 0.125
    corrupted = replace(
        encoded,
        transitions=(
            replace(encoded.transitions[0], next_observation=tuple(tail)),
            replace(encoded.transitions[1], observation=tuple(tail)),
        ),
    )
    with pytest.raises(ValueError):
        history.validate_action_history_training_episode(corrupted, recipe, mass_context=mass)


def test_should_leave_ordinary_train_default_history_unchanged():
    ordinary = replace(mass_episode(), context="pick-cube")
    recipe = PickCubeTaskSpaceHistoryRecipe()
    assert history.action_history_training_episode(
        ordinary, recipe
    ) == history.action_history_training_episode(ordinary, recipe, mass_context=None)
    assert "mass_context" not in ActionHistoryEncoding(recipe, train_seed=17).report()


def test_should_bind_mass_encoding_at_birth_and_verify_original_native_episode():
    original = pick_cube_mass_context()
    encoding = ActionHistoryEncoding(
        PickCubeTaskSpaceHistoryRecipe(), train_seed=17, mass_context=original
    )
    assert encoding.native_binding == (17, original.context, "pick-cube-smoke-17")
    assert encoding.native_role == "train"
    assert encoding.mass_context == original and encoding.mass_context is not original
    for row in mass_episode().transitions:
        encoding.encode(row.observation)
        encoding.commit(row, row.action)
    encoding.verify_episode(mass_episode())
    assert encoding.report()["mass_context"] == asdict(original)
    with pytest.raises(ValueError):
        encoding.verify_episode(replace(mass_episode(), context="pick-cube"))


@pytest.mark.parametrize(
    "fault",
    ["missing_seed", "seed19", "seed20", "bool_seed", "guard", "probe", "47d", "zero", "metadata"],
)
def test_should_refuse_invalid_mass_encoding_before_any_observation(fault):
    recipe: history.ActionHistoryRecipe
    options: dict[str, Any] = dict(train_seed=17, mass_context=pick_cube_mass_context())
    recipe = PickCubeTaskSpaceHistoryRecipe()
    if fault == "missing_seed":
        options.pop("train_seed")
    elif fault in ("seed19", "seed20", "bool_seed"):
        options["train_seed"] = {"seed19": 19, "seed20": 20, "bool_seed": True}[fault]
    elif fault in ("guard", "probe"):
        options["guard_seed" if fault == "guard" else "probe_test_seed"] = (
            31 if fault == "guard" else 24
        )
    elif fault == "47d":
        recipe = history.PickCubeActionHistoryRecipe()
    elif fault == "zero":
        recipe = PickCubeTaskSpaceHistoryRecipe(mode="zero")
    else:
        options["mass_context"] = asdict(options["mass_context"])
    with pytest.raises(ValueError):
        ActionHistoryEncoding(recipe, **options)


@pytest.mark.parametrize("fault", ["original", "owned", "equal_replacement"])
def test_should_refuse_mass_encoding_content_or_identity_replacement(fault):
    mass = pick_cube_mass_context()
    encoding = ActionHistoryEncoding(
        PickCubeTaskSpaceHistoryRecipe(), train_seed=17, mass_context=mass
    )
    if fault == "equal_replacement":
        encoding.mass_context = pick_cube_mass_context()
    else:
        object.__setattr__(mass if fault == "original" else encoding.mass_context, "seed", 19)
    with pytest.raises(ValueError):
        encoding.report()


@pytest.mark.parametrize("family", FAMILIES)
def test_should_freeze_all_four_families_with_mass_readbacks_and_causal_native32_history(
    owners, boundary, monkeypatch, family
):
    environment, creations, output = boundary
    source = owners[family].fork()
    before = source.snapshot_state()
    predict = type(source).predict
    features = []

    def capture(owner, values):
        features.append(values.copy())
        return predict(owner, values)

    monkeypatch.setattr(type(source), "predict", capture)
    result = smoke.run_frozen_rollout(
        output,
        learner=source,
        mass_context=pick_cube_mass_context(),
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    report = result.report
    assert report["complete"] is True, report.get("error", report.get("action_history_error"))
    assert result.corrections is None and report["optimizer_updates"] == 0
    assert report["source_updates"] == 13 and report["source_training_rows"] == 377
    assert report["mass_context_exact"] is report["action_history_exact"] is True
    assert report["frozen_state_exact"] is report["original_owner_exact"] is True
    assert (
        report["mass_context"]
        == report["action_history"]["mass_context"]
        == asdict(pick_cube_mass_context())
    )
    assert report["mass_readback"]["source_mass_kg"] == 0.125
    assert report["mass_readback"]["applied_mass_kg"] == 0.25
    assert report["mass_work"]["write_attempts"] == report["mass_work"]["write_returns"] == 1
    assert report["mass_work"]["read_attempts"] == report["mass_work"]["read_returns"]
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert report["episode"]["context"] == pick_cube_mass_context().context
    assert report["episode"]["episode_id"] == "pick-cube-smoke-17"
    assert (
        report["episode"]["role"] == "train" and report["episode"]["spec"]["observation_dim"] == 42
    )
    assert len(features) == 2 and all(value.shape == (1, 12) for value in features)
    records = report["episode"]["transitions"]
    first_raw = records[0]["observation"]
    assert tuple(features[0][0]) == (
        first_raw[18],
        *(value / 0.1 for value in first_raw[36:42]),
        *(0.0,) * 5,
    )
    sent = tuple(float(value) for value in environment.commands[0][0])
    assert environment.commands[0].dtype == np.dtype("float32")
    assert records[0]["action"] == sent
    assert tuple(features[1][0, -5:]) == (*sent, 1 / 6)
    assert report["action_history"]["work"]["committed_controls"] == 2
    assert creations == [{}] and environment.reset_calls == [17] and environment.close_calls == 1
    assert_state(before, source.snapshot_state())
    assert json.loads(output.read_text(encoding="utf8"))["mass_context_exact"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "seed19",
        "seed20",
        "guard",
        "probe",
        "camera",
        "light",
        "clock",
        "queries",
        "rollin",
        "47d",
        "zero",
        "visual",
        "joint",
        "decoder",
        "unsupported",
    ],
)
def test_should_refuse_mass_history_mixed_modes_before_fork_or_native_work(
    owners, boundary, monkeypatch, fault
):
    _, creations, output = boundary
    options: dict[str, Any] = dict(
        learner=owners["bp"],
        mass_context=pick_cube_mass_context(),
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    changes: dict[str, dict[str, Any]] = {
        "seed19": dict(train_seed=19),
        "seed20": dict(train_seed=20),
        "guard": dict(guard_seed=31),
        "probe": dict(probe_test_seed=24),
        "camera": dict(camera_context=object()),
        "light": dict(lighting_context=object()),
        "clock": dict(observation_clock=object()),
        "queries": dict(corrective_queries=True),
        "rollin": dict(readiness_roll_in=object()),
        "visual": dict(visual_observation=True),
        "joint": dict(monitor_joint_positions=True),
        "decoder": dict(action_decoding_recipe=object()),
        "unsupported": dict(learner=object()),
        "47d": dict(action_history_recipe=history.PickCubeActionHistoryRecipe()),
        "zero": dict(action_history_recipe=PickCubeTaskSpaceHistoryRecipe(mode="zero")),
    }
    options.update(changes[fault])
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    report = smoke.run_frozen_rollout(output, **options).report
    assert report["complete"] is False and report["creation_attempts"] == 0
    assert report["prediction_attempts"] == 0 and creations == []


def test_should_keep_ordinary_taskspace_rollout_default_without_mass_metadata(owners, boundary):
    _, _, output = boundary
    report = smoke.run_frozen_rollout(
        output, learner=owners["bp"], action_history_recipe=PickCubeTaskSpaceHistoryRecipe()
    ).report
    assert report["complete"] is True and report["context"] == "pick-cube"
    assert "mass_context" not in report and "mass_context" not in report["action_history"]


@pytest.mark.parametrize("fault", ["drift", "native_failure", "close_binding"])
def test_should_keep_paid_control_prefix_without_candidate_or_corrections_on_mass_failure(
    owners, boundary, monkeypatch, fault
):
    environment, _, output = boundary
    source = owners["bp"].fork()
    before = source.snapshot_state()
    encodings = []
    original = smoke.ActionHistoryEncoding

    def capture(*args, **options):
        encoding = original(*args, **options)
        encodings.append(encoding)
        return encoding

    monkeypatch.setattr(smoke, "ActionHistoryEncoding", capture)
    if fault == "drift":
        environment.after_step = lambda: environment.cube.value.fill(0.5)
    elif fault == "native_failure":
        environment.step_error = RuntimeError("paid native control failed")
    else:
        environment.close_callback = lambda: setattr(
            encodings[0], "mass_context", pick_cube_mass_context()
        )
    result = smoke.run_frozen_rollout(
        output,
        learner=source,
        mass_context=pick_cube_mass_context(),
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    report = result.report
    assert report["complete"] is False and result.corrections is None
    attempts = 2 if fault == "close_binding" else 1
    assert report["prediction_attempts"] == report["prediction_returns"] == attempts
    assert report["work"]["step_attempts"] == len(environment.commands) == attempts
    assert report["work"]["completed_steps"] == (0 if fault == "native_failure" else attempts)
    assert environment.close_calls == 1
    assert report["mass_work"]["write_returns"] == 1
    assert_state(before, source.snapshot_state())


def test_should_refuse_native_binding_change_from_mass_getter_before_next_control(
    owners, boundary, monkeypatch
):
    environment, _, output = boundary
    source = owners["bp"].fork()
    before = source.snapshot_state()
    predict = bp.NumpyContinuousBCLearner.predict
    readback = cast(Any, native.ManiSkillPickCubeEpisode.mass_readback).fget
    mutations: list[bool]
    armed, mutations = [], []

    def predicted(owner, values):
        returned = predict(owner, values)
        armed.append(True)
        return returned

    def observed(owner):
        returned = readback(owner)
        if armed and not mutations:
            owner._context = "pick-cube"
            mutations.append(True)
        return returned

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "mass_readback", property(observed))
    report = smoke.run_frozen_rollout(
        output,
        learner=source,
        mass_context=pick_cube_mass_context(),
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    ).report
    assert mutations == [True] and report["complete"] is False
    assert report["prediction_returns"] == 1
    assert report["work"]["step_attempts"] == 0 and environment.commands == []
    assert_state(before, source.snapshot_state())


def test_should_refuse_source_mutation_from_final_close_without_restoring_it(owners, boundary):
    environment, _, output = boundary
    source = owners["bp"].fork()
    before = source.snapshot_state()

    def mutate_source():
        source._state.parameters[3][0] += 0.125

    environment.close_callback = mutate_source
    report = smoke.run_frozen_rollout(
        output,
        learner=source,
        mass_context=pick_cube_mass_context(),
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    ).report
    assert report["complete"] is False
    assert report["prediction_returns"] == report["work"]["completed_steps"] == 2
    assert source._state.parameters[3][0] == before.parameters[3][0] + 0.125
    assert report["original_owner_exact"] is not True


@pytest.mark.parametrize("fault", ["readback", "work"])
def test_should_reprove_mass_evidence_after_final_history_callback_before_control(
    owners, boundary, monkeypatch, fault
):
    environment, _, output = boundary
    source = owners["bp"].fork()
    before = source.snapshot_state()
    mutations: list[bool]
    ports, predictions, mutations = [], [], []
    initialize = native.ManiSkillPickCubeEpisode.__init__
    predict = bp.NumpyContinuousBCLearner.predict
    validate = ActionHistoryEncoding.validate_executions
    mass_check = native.ManiSkillPickCubeEpisode.check_mass_metadata
    mass_proofs = []

    def initialized(owner, *args, **options):
        initialize(owner, *args, **options)
        ports.append(owner)

    def predicted(owner, features):
        returned = predict(owner, features)
        predictions.append(True)
        return returned

    def checked(port, *args, **options):
        returned = mass_check(port, *args, **options)
        if predictions:
            mass_proofs.append(True)
        return returned

    def validated(encoding, *args, **options):
        returned = validate(encoding, *args, **options)
        # Inject after both post-prediction mass proofs, at the final history
        # callback before control. An earlier injection is caught by the old path.
        if len(mass_proofs) >= 2 and encoding.pending is not None and not mutations:
            port = ports[0]
            if fault == "readback":
                port._mass_source_mass = 0.25
            else:
                port._mass_reads += 1
                port._mass_read_returns += 1
            mutations.append(True)
        return returned

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "__init__", initialized)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted)
    monkeypatch.setattr(ActionHistoryEncoding, "validate_executions", validated)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "check_mass_metadata", checked)
    report = smoke.run_frozen_rollout(
        output,
        learner=source,
        mass_context=pick_cube_mass_context(),
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    ).report
    assert mutations == [True] and report["complete"] is False
    assert report["prediction_attempts"] == report["prediction_returns"] == 1
    assert report["work"]["step_attempts"] == 0 and environment.commands == []
    assert float(np.asarray(environment.cube.value).reshape(-1)[0]) == 0.25
    assert_state(before, source.snapshot_state())
