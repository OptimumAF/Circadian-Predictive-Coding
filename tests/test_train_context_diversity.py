"""Explicit TRAIN20 allocation, fake collection and ordinary typed admission.

Full module: five fake scenes/resets/closes; nineteen attempted controls,
eighteen completed controls and nineteen returned geometry reads/decisions.
Fourteen capped decoder payload reads and five small verification reads; three
CLI dispatches reuse receipts. No models, RNG, real SDK or physics.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts import maniskill_task_teacher_smoke as cli
from scripts import run_native_stream_demo as stream
from src.core.pick_cube_readiness_teacher import DEFINITION
from src.core.pick_cube_train_context import pick_cube_training_context
from src.infra import maniskill_continuous as native
from src.infra import maniskill_task_teacher as collector
from test_pick_cube_readiness_teacher import (
    install_environment,
    route_geometry,
    successful_environment,
)
from test_observed_grasp_teacher import ObservedEnvironment, install as install_observed
from test_readiness_stream_input import receipt, write


def forbidden(*args, **kwargs):
    pytest.fail("invalid TRAIN allocation admitted native collection or another teacher route")


def train20_receipt():
    # Why: the existing helper deliberately models the original17/19 allocation.
    # Keep its complete payload, changing only the explicitly declared20 context.
    raw = receipt(20)
    raw["context"] = "pick-cube-train-20"
    raw["candidate_episode"]["context"] = "pick-cube-train-20"
    return raw


@pytest.mark.parametrize(
    "seed,context", [(17, "pick-cube"), (19, "pick-cube-train-19"), (20, "pick-cube-train-20")]
)
def test_should_return_only_explicit_native_train_contexts(seed, context):
    assert pick_cube_training_context(seed) == context


@pytest.mark.parametrize("seed", [23, 24, 29, 31, 37, 21, True, False, 20.0, "20"])
def test_should_refuse_reserved_unknown_or_non_native_train_seed_before_io(monkeypatch, seed):
    monkeypatch.setattr(collector, "ManiSkillPickCubeEpisode", forbidden)
    with pytest.raises(ValueError):
        pick_cube_training_context(seed)
    with pytest.raises(ValueError):
        collector.collect_readiness_teacher(train_seed=seed)
    with pytest.raises(ValueError):
        stream._read_readiness_teacher(Path("unread-reserved-teacher.json"), seed)


@pytest.fixture(scope="module")
def collected():
    result = {}
    for seed, steps in ((20, 10), (17, 2)):
        environment = successful_environment(steps)
        with pytest.MonkeyPatch.context() as patch:
            creations = install_environment(patch, environment, route_geometry()[:steps])
            value = (
                collector.collect_readiness_teacher()
                if seed == 17
                else collector.collect_readiness_teacher(train_seed=seed)
            )
        assert value.qualified, value.error
        result[seed] = value, environment, creations
    return result


def test_should_collect_train20_with_original_reset_commands_work_and_owned_episode(collected):
    result, environment, creations = collected[20]
    assert result.teacher_definition == DEFINITION and result.qualified is True
    assert creations == [{}] and environment.reset_calls == [20] and environment.close_calls == 1
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 10
    assert result.work == native.ManiSkillWork(10, 10, 50, 0)
    assert result.stages == ("descend",) + ("grasp",) * 6 + ("lift", "settle", "settle")
    episode = result.training_episode()
    assert episode.role == "train" and episode.context == "pick-cube-train-20"
    assert episode.episode_id == "pick-cube-readiness-teacher-20"
    assert episode.spec.observation_dim == 42
    assert len(episode.transitions) == len(environment.commands) == 10
    for index, (row, command) in enumerate(
        zip(episode.transitions, environment.commands, strict=True)
    ):
        assert command.dtype == np.dtype("float32")
        assert row.action == tuple(float(value) for value in command[0])
        assert row.observation == (float(index),) + (0.0,) * 41
    again = result.training_episode()
    object.__setattr__(episode.transitions[0], "reward", 0.5)
    assert again.transitions[0].reward == result.training_episode().transitions[0].reward == 0.0
    assert set(asdict(result)) == set(asdict(collected[17][0]))


@pytest.mark.parametrize("failure", ["physics", "cleanup"])
def test_should_preserve_failed_train20_collection_cost_without_releasing_training(
    monkeypatch, failure
):
    environment = successful_environment(10 if failure == "physics" else 2)
    if failure == "physics":
        environment.step_failure_at = 3
    else:
        environment.close_error = RuntimeError("TRAIN20 cleanup failed")
    install_environment(monkeypatch, environment, route_geometry())
    result = collector.collect_readiness_teacher(train_seed=20)
    assert result.qualified is False and result.error is not None
    assert environment.reset_calls == [20] and environment.close_calls == 1
    if failure == "physics":
        assert result.work == native.ManiSkillWork(3, 2, 10, 5)
        assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 3
        assert result.candidate_episode is None and len(environment.commands) == 3
    else:
        assert result.work == native.ManiSkillWork(2, 2, 10, 0)
        assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 2
        assert result.candidate_episode is not None
        assert result.candidate_episode.context == "pick-cube-train-20"
    with pytest.raises(ValueError):
        result.training_episode()


def test_should_delegate_cli_train20_and_preserve_original_default_json_schema(
    collected, monkeypatch, tmp_path
):
    calls, payloads = [], []
    monkeypatch.setattr(cli, "collect_task_teacher", forbidden)
    monkeypatch.setattr(cli, "version", lambda _: "fake-train20-boundary")
    for seed in (17, 20):

        def selected(**options):
            calls.append(options)
            return collected[seed][0]

        monkeypatch.setattr(cli, "collect_readiness_teacher", selected)
        output = tmp_path / f"teacher-{seed}.json"
        arguments = ["teacher", "--output", str(output), "--geometry-readiness"]
        if seed == 20:
            arguments += ["--train-seed", "20"]
        monkeypatch.setattr(sys, "argv", arguments)
        assert cli.main() == 0
        raw = json.loads(output.read_text(encoding="utf8"))
        payloads.append(raw)
        assert raw["seed"] == seed and raw["role"] == "train" and raw["qualified"] is True
        assert (
            raw["context"]
            == raw["candidate_episode"]["context"]
            == pick_cube_training_context(seed)
        )
        assert raw["candidate_episode"]["episode_id"] == f"pick-cube-readiness-teacher-{seed}"
        assert raw["work"] == asdict(collected[seed][0].work)
        assert raw["optimizer_updates"] == 0
    assert calls == [{}, dict(train_seed=20)]
    assert set(payloads[0]) == set(payloads[1])


def test_should_refuse_qualified_train19_returned_for_explicit_cli_train20(monkeypatch, tmp_path):
    # Reuse an ordinary synthetic19 receipt; qualification alone does not mean
    # the collector answered the caller's independently fixed20 request.
    declared, _ = stream._read_readiness_teacher(write(tmp_path / "nineteen.json", receipt(19)), 19)
    wrong = collector.TaskTeacherResult(
        declared,
        True,
        True,
        1,
        2,
        2,
        2,
        ("descend", "grasp"),
        native.ManiSkillWork(2, 2, 10, 0),
        None,
        None,
        teacher_definition=DEFINITION,
    )
    assert wrong.qualified is True
    calls = []

    def selected(**options):
        calls.append(options)
        return wrong

    monkeypatch.setattr(cli, "collect_readiness_teacher", selected)
    monkeypatch.setattr(cli, "collect_task_teacher", forbidden)
    monkeypatch.setattr(cli, "version", lambda _: "fake-wrong-declared-allocation")
    output = tmp_path / "wrong-request.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["teacher", "--output", str(output), "--geometry-readiness", "--train-seed", "20"],
    )
    assert cli.main() == 1 and calls == [dict(train_seed=20)]
    raw = json.loads(output.read_text(encoding="utf8"))
    assert raw["qualified"] is False and raw["train_request_error"]
    assert raw["candidate_episode"]["episode_id"] == "pick-cube-readiness-teacher-19"
    assert raw["candidate_episode"]["context"] == "pick-cube-train-19"
    assert raw["teacher_actions"] == raw["work"]["completed_steps"] == 2
    assert wrong.qualified is True


def test_should_collect_and_decode_observed_grasp_train20_without_rewriting_feedback(
    monkeypatch, tmp_path
):
    environment = ObservedEnvironment()
    creations = install_observed(monkeypatch, environment)
    result = collector.collect_readiness_teacher(train_seed=20, observed_grasp=True)
    assert type(result) is collector.ObservedGraspTaskTeacherResult
    assert result.qualified is True and result.observed_grasp == (True, False)
    assert result.stages == ("lift", "align")
    assert creations == [{}] and environment.reset_calls == [20] and environment.close_calls == 1
    assert result.work == native.ManiSkillWork(2, 2, 10, 0)
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 2
    raw = dict(
        asdict(result),
        qualified=True,
        seed=20,
        context="pick-cube-train-20",
        role="train",
        task=native.TASK_ID,
        controller=native.CONTROL_MODE,
        optimizer_updates=0,
    )
    episode, metadata = stream._read_readiness_teacher(
        write(tmp_path / "observed20.json", raw), 20, observed_grasp=True
    )
    assert episode.episode_id == "pick-cube-observed-grasp-teacher-20"
    assert episode.context == metadata["context"] == "pick-cube-train-20"
    assert metadata["observed_grasp"] == [True, False]
    assert metadata["teacher_work"] == asdict(result.work)
    assert tuple(row.observation[18] == 1.0 for row in episode.transitions) == result.observed_grasp
    assert tuple(row.action for row in episode.transitions) == (
        (1.0, 0.0, 1.0, -1.0),
        (1.0, 0.0, -1.0, 1.0),
    )
    for row, command in zip(episode.transitions, environment.commands, strict=True):
        assert row.action == tuple(float(value) for value in command[0])


def test_should_decode_train20_into_owned_typed_pairs_and_original_acquisition_cost(tmp_path):
    raw = train20_receipt()
    original = deepcopy(raw)
    path = write(tmp_path / "twenty.json", raw)
    original_bytes = path.read_bytes()
    episode, metadata = stream._read_readiness_teacher(path, 20)
    assert episode.context == metadata["context"] == "pick-cube-train-20"
    assert metadata["seed"] == 20 and episode.role == "train"
    assert metadata["episode_id"] == episode.episode_id == "pick-cube-readiness-teacher-20"
    assert metadata["teacher_work"] == raw["work"]
    assert metadata["teacher_geometry_returns"] == metadata["teacher_actions"] == 2
    assert metadata["teacher_definition"] == DEFINITION
    assert metadata["source_file_bytes"] == len(original_bytes)
    assert tuple(row.action for row in episode.transitions) == (
        (0.0, 0.0, -0.8, 1.0),
        (0.0, 0.0, 0.0, -1.0),
    )
    metadata["teacher_work"]["completed_steps"] = 99
    object.__setattr__(episode.transitions[0], "reward", 0.5)
    again, retained = stream._read_readiness_teacher(path, 20)
    assert retained["teacher_work"]["completed_steps"] == 2 and again.transitions[0].reward == 0.0
    assert raw == original and path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "fault",
    [
        "seed",
        "context",
        "episode_context",
        "identity",
        "top_role",
        "episode_role",
        "selected17",
        "selected19",
        "float_seed",
        "unknown_context_pair",
    ],
)
def test_should_refuse_seed_context_identity_role_or_selected_binding_replacement(tmp_path, fault):
    raw = train20_receipt()
    selected = 20
    if fault == "seed":
        raw["seed"] = 19
    elif fault == "context":
        raw["context"] = "pick-cube-train-19"
    elif fault == "episode_context":
        raw["candidate_episode"]["context"] = "pick-cube-train-19"
    elif fault == "identity":
        raw["candidate_episode"]["episode_id"] = "pick-cube-readiness-teacher-19"
    elif fault == "top_role":
        raw["role"] = "inner_guard"
    elif fault == "episode_role":
        raw["candidate_episode"]["role"] = "probe_train"
    elif fault.startswith("selected"):
        selected = int(fault[-2:])
    elif fault == "float_seed":
        raw["seed"] = 20.0
    else:
        raw["context"] = raw["candidate_episode"]["context"] = "pick-cube-train-21"
        raw["candidate_episode"]["episode_id"] = "pick-cube-readiness-teacher-21"
    with pytest.raises(ValueError):
        stream._read_readiness_teacher(write(tmp_path / "replaced.json", raw), selected)


@pytest.mark.parametrize("fault", ["mismatched_id", "mismatched_context", "undeclared_context"])
def test_should_require_matching_declared_train_pair_in_result_qualification(collected, fault):
    result = collected[20][0]
    episode = result.candidate_episode
    assert episode is not None
    if fault == "mismatched_id":
        changed = replace(episode, episode_id="pick-cube-readiness-teacher-19")
    elif fault == "mismatched_context":
        changed = replace(episode, context="pick-cube-train-19")
    else:
        changed = replace(
            episode, context="undeclared", episode_id="pick-cube-readiness-teacher-19"
        )
    refused = replace(result, candidate_episode=changed)
    assert refused.qualified is False
    with pytest.raises(ValueError):
        refused.training_episode()
