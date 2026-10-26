"""Four matched families use arrived corrections with actual command history.

Per platform: four D47/H2 initializations (408 normal draws),64 private updates/
128 rows, no fit; eight fake scenes/16 predictions/controls, no physics. Module
fixtures reuse paid results for refusal checks. Legacy selectors are budgeted
separately in the stage protocol. Only root executes these numerical fixtures.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_action_history import PickCubeActionHistoryRecipe
from src.core.pick_cube_action_history_corrections import action_history_correction_view
from src.core.pick_cube_readiness_roll_in import ReadinessRollInPlan
from src.core.pick_cube_state_quaternions import state_quaternion_projection
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_state_quaternion_corrections import batch
from test_state_quaternion_robot_learning import QuaternionEnvironment


def forbidden(*args, **kwargs):
    pytest.fail("refused request admitted more work")


@pytest.fixture(scope="module", params=["bp", "pc", "neutral", "active"])
def source(request):
    spec = replace(batch().episode.spec, observation_dim=47)
    base = bp.ContinuousBCPolicy(
        spec,
        hidden_dim=2,
        seed=17,
        observation_standardization=ObservationStandardization((0.0,) * 47, (1.0,) * 47, 2),
    )
    model: Any
    if request.param == "bp":
        model = bp.NumpyContinuousBCLearner(base)
    else:
        config = (
            None
            if request.param == "pc"
            else CoupledCircadianConfig.matched_pc_control()
            if request.param == "neutral"
            else CoupledCircadianConfig()
        )
        model = pc.NumpyContinuousPCLearner(pc.ContinuousPCPolicy(base, circadian=config))
    initial = model.snapshot_state()
    parameters: Any = tuple(np.zeros_like(value) for value in initial.parameters)
    parameters[3][3] = -0.75
    model.restore_state(replace(initial, parameters=parameters))
    return model


@pytest.fixture(scope="module")
def completed(source, tmp_path_factory):
    owner, recipe = source.fork(), PickCubeActionHistoryRecipe(canonical_quaternions=True)
    initial = owner.snapshot_state()
    results, updates, trains, environments = [], [], [], []
    rollout, update, episode = (
        smoke.run_frozen_rollout,
        smoke.run_continuous_correction_update,
        native.ManiSkillPickCubeEpisode.episode,
    )
    cls = type(owner)
    method = "train_batch" if cls is bp.NumpyContinuousBCLearner else "train_batch_with_work"
    train = getattr(cls, method)
    with pytest.MonkeyPatch.context() as patch:

        def factory(**options):
            assert options == {}
            env = QuaternionEnvironment()
            environments.append(env)
            return env

        def geometry(port):
            return PickCubeGeometry(
                (0.0, 0.0, 0.08) if not port._records else (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.2),
            )

        def live_episode(port):
            assert not port._closed, "public episode read after SDK close"
            return episode(port)

        def recorded_rollout(*args, **options):
            value = rollout(*args, **options)
            results.append(value)
            return value

        def recorded_update(*args, **options):
            value = update(*args, **options)
            updates.append(value)
            return value

        def recorded_train(model, features, targets):
            trains.append((features.copy(), targets.copy()))
            return train(model, features, targets)

        patch.setattr(native, "_make_environment", factory)
        patch.setattr(queries, "_geometry", geometry)
        patch.setattr(native.ManiSkillPickCubeEpisode, "episode", live_episode)
        patch.setattr(smoke, "version", lambda _: "fake-history-corrections")
        patch.setattr(smoke, "run_frozen_rollout", recorded_rollout)
        patch.setattr(smoke, "run_continuous_correction_update", recorded_update)
        patch.setattr(cls, method, recorded_train)
        summary = smoke.run_corrective_adaptation(
            owner,
            output_directory=tmp_path_factory.mktemp("history-correction") / "case",
            visual_observation=False,
            readiness_corrections=True,
            acquisition_only=True,
            readiness_roll_in=ReadinessRollInPlan(1),
            action_history_recipe=recipe,
        )
    assert summary["complete"] is True, summary
    return owner, initial, recipe, summary, results, updates[0], trains, environments


def test_should_complete_raw_query_mixed_control_private_update_and_pure_return(completed):
    owner, initial, recipe, summary, results, updated, trains, environments = completed
    assert len(results) == 2 and len(environments) == 2
    assert all(value.report["action_history_exact"] is True for value in results)
    raw = results[0].corrections
    assert raw.episode.spec.observation_dim == 42
    expected = action_history_correction_view(raw, recipe).model.training_pairs(now=3)
    assert len(trains) == 8
    for features, targets in trains:
        assert features.tolist() == [list(row) for row in expected[0]]
        assert targets.tolist() == [list(row) for row in expected[1]]
    # The first executed teacher prefix opens; the student's unsent proposal closes.
    assert raw.episode.transitions[0].action[3] == 1.0
    assert trains[0][0][1, 45] == 1.0 and trains[0][0][1, 46] == 0.0
    assert trains[0][0][1, 44] == raw.episode.transitions[0].action[2]
    assert trains[0][0][1, 44] != raw.labels[0].target[2]
    final = updated.candidate.snapshot_state()
    assert final.completed_updates == 8 and final.training_examples == 16
    assert final.policy == initial.policy and demo._same_state(initial, owner.snapshot_state())
    assert updated.report["work"]["fit_calls"] == 0
    assert len(updated.report["applied_updates"]) == 8
    for receipt in updated.report["applied_updates"]:
        assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
    assert results[1].corrections is None
    assert results[1].report["prediction_returns"] == 2
    assert results[1].report["source_updates"] == 8


@pytest.mark.parametrize("fault", ["future", "projection", "recipe"])
def test_should_refuse_bad_request_before_any_fork(source, monkeypatch, fault):
    owner = source.fork()
    monkeypatch.setattr(owner, "fork", forbidden)
    options: Any = dict(now=3, action_history_recipe=PickCubeActionHistoryRecipe())
    if fault == "future":
        options["now"] = 2
    elif fault == "projection":
        options["observation_projection"] = state_quaternion_projection()
    else:
        options["action_history_recipe"] = asdict(options["action_history_recipe"])
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(owner, batch(), **options)


def test_should_preserve_all_eight_paid_updates_when_projection_metadata_is_corrupted(
    source, monkeypatch
):
    owner, initial = source.fork(), source.snapshot_state()
    name = "_record_mlp_work" if type(owner) is bp.NumpyContinuousBCLearner else "_record_pc_work"
    original = getattr(demo, name)

    def changed(report, *args):
        original(report, *args)
        report["input_projection"]["work"]["history_advances"] = 2.0

    monkeypatch.setattr(demo, name, changed)
    result = demo.run_continuous_correction_update(
        owner,
        batch(),
        now=3,
        action_history_recipe=PickCubeActionHistoryRecipe(canonical_quaternions=True),
    )
    assert result.report["complete"] is False and result.candidate is None
    assert len(result.report["applied_updates"]) == 8
    assert result.report["progress"]["updates_completed"] == 8
    assert demo._same_state(initial, owner.snapshot_state())


@pytest.mark.parametrize("fault", ["seed", "source", "recipe"])
def test_should_refuse_replaced_return_binding_without_model_work(completed, fault):
    _, initial, recipe, _, results, _, _, _ = completed
    report = deepcopy(results[0].report)
    if fault == "seed":
        report["seed"] = 19
    elif fault == "source":
        report["source_updates"] += 1
    else:
        report["action_history"]["recipe"]["mode"] = "zero"
    with pytest.raises(ValueError):
        smoke._check_action_history_result(recipe, report, 17, initial)
