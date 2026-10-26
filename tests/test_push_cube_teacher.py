"""Push supervision geometry and fake acquisition; no learned or native evidence.

Pure decisions use declared physical states and separately executed history.
Collector doubles preserve paid returns without a simulator, RNG or learner.
"""

from dataclasses import asdict
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.core.cube_taskspace_state import CubeTaskSpaceState
from src.core.pick_cube_action_history import ActionHistoryState
from src.core import push_cube_teacher as policy
from src.infra import maniskill_push_cube as native
from src.infra import maniskill_push_teacher as collector
from test_push_cube_continuous import FakePushEnvironment


ZERO = ActionHistoryState()
CLOSED = ActionHistoryState((0.0, 0.0, 0.0, -1.0), 6)


def state(tcp=(-0.06, 0.0, 0.12), *, cube=(0.0, 0.0, 0.02), goal=(0.4, 0.0, 0.02), grasp=False):
    return CubeTaskSpaceState(grasp, tcp, cube, goal)


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        teacher_attempts=0,
        teacher_returns=0,
        factory_attempts=0,
        factory_returns=0,
        reset_attempts=0,
        reset_returns=0,
        control_attempts=0,
        control_returns=0,
        grasp_attempts=0,
        grasp_returns=0,
        close_attempts=0,
        close_returns=0,
    )
    original = policy.PushCubeTeacher.decide

    def counted(owner, *args, **options):
        counts["teacher_attempts"] += 1
        returned = original(owner, *args, **options)
        counts["teacher_returns"] += 1
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(policy.PushCubeTeacher, "decide", counted)
        yield counts
    if os.environ.get("PST_FIXTURE_LEDGER"):
        with Path(os.environ["PST_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["teacher_attempts"] <= 768
    assert counts["factory_attempts"] <= 64 and counts["reset_attempts"] <= 64
    assert counts["control_attempts"] <= 256 and counts["grasp_attempts"] <= 512


@pytest.mark.parametrize(
    "goal,expected",
    [
        ((0.4, 0.0, 0.02), (-0.5, 0.0, 0.0, -1.0)),
        ((-0.4, 0.0, 0.02), (0.5, 0.0, 0.0, -1.0)),
        ((0.0, 0.4, 0.02), (0.0, -0.5, 0.0, -1.0)),
        ((0.0, -0.4, 0.02), (0.0, 0.5, 0.0, -1.0)),
    ],
)
def test_should_approach_behind_cube_in_the_actual_goal_direction(goal, expected):
    teacher = policy.PushCubeTeacher()
    decision = teacher.decide(state((0.0, 0.0, 0.12), goal=goal), history=ZERO)
    assert decision.stage == teacher.phase == "align"
    assert decision.action == pytest.approx(expected)
    assert teacher.steps == 1 and policy.DEFINITION == "push_cube_closed_fist_xy_v1"


def test_should_use_fixed_physical_scale_and_remain_translation_invariant():
    source = state((-0.04, 0.0, 0.10))
    translated = CubeTaskSpaceState(False, (0.96, 2.0, 3.10), (1.0, 2.0, 3.02), (1.4, 2.0, 3.02))
    first = policy.PushCubeTeacher().decide(source, history=ZERO)
    second = policy.PushCubeTeacher().decide(translated, history=ZERO)
    assert first.action == pytest.approx((-0.2, 0.0, 0.2, -1.0))
    assert second.action == pytest.approx(first.action)


@pytest.mark.parametrize("grasp", [False, True])
def test_should_wait_for_actual_six_closing_controls_instead_of_queried_labels(grasp):
    teacher = policy.PushCubeTeacher()
    at_hover = state(grasp=grasp)
    for _ in range(8):
        queried = teacher.decide(at_hover, history=ZERO)
        assert queried.stage == teacher.phase == "close" and queried.action == (0.0, 0.0, 0.0, -1.0)
    assert teacher.steps == 8 and ZERO.closing_count == 0
    descend = teacher.decide(at_hover, history=CLOSED)
    assert descend.stage == teacher.phase == "descend"
    assert descend.action == (0.0, 0.0, -0.5, -1.0)
    push = teacher.decide(state((-0.06, 0.0, 0.035), grasp=grasp), history=CLOSED)
    assert push.stage == teacher.phase == "push"
    assert push.action == (0.5, 0.0, 0.0, -1.0)


def pushing_teacher():
    teacher = policy.PushCubeTeacher()
    teacher.decide(state(), history=ZERO)
    teacher.decide(state(), history=CLOSED)
    teacher.decide(state((-0.06, 0.0, 0.035)), history=CLOSED)
    assert teacher.phase == "push"
    return teacher


@pytest.mark.parametrize(
    "tcp", [(-0.11, 0.0, 0.035), (0.03, 0.0, 0.035), (-0.06, 0.04, 0.035), (-0.06, 0.0, 0.07)]
)
def test_should_recover_alignment_when_push_contact_geometry_is_lost(tcp):
    teacher = pushing_teacher()
    recovery = teacher.decide(state(tcp), history=CLOSED)
    assert recovery.stage == teacher.phase == "align"
    assert recovery.action[3] == -1.0
    assert all(-0.5 <= value <= 0.5 for value in recovery.action[:3])


def test_should_realign_before_descending_when_lateral_readiness_is_lost():
    teacher = policy.PushCubeTeacher()
    teacher.decide(state(), history=ZERO)
    teacher.decide(state(), history=CLOSED)
    recovery = teacher.decide(state((-0.06, 0.02, 0.08)), history=CLOSED)
    assert recovery.stage == teacher.phase == "align"


def test_should_hold_near_goal_without_claiming_native_success_and_restart_when_goal_moves():
    teacher = policy.PushCubeTeacher()
    elevated = state((0.0, 0.0, 0.7), cube=(0.0, 0.0, 0.5), goal=(0.04, 0.0, 0.02))
    held = teacher.decide(elevated, history=ZERO)
    assert held.stage == teacher.phase == "hold" and held.action == (0.0, 0.0, 0.0, -1.0)
    assert set(asdict(held)) == {"stage", "action"}
    restarted = teacher.decide(state((0.0, 0.0, 0.7), cube=(0.0, 0.0, 0.5)), history=ZERO)
    assert restarted.stage == teacher.phase == "align"


@pytest.mark.parametrize(
    "fault", ["foreign", "flag", "missing", "nan", "overflow", "history", "history_mutation"]
)
def test_should_refuse_unobserved_or_nonfinite_inputs_without_advancing_teacher(fault):
    teacher = policy.PushCubeTeacher()
    source: Any = state()
    history: Any = ActionHistoryState()
    if fault == "foreign":
        source = object()
    elif fault == "flag":
        object.__setattr__(source, "is_grasped", 1)
    elif fault == "missing":
        object.__delattr__(source, "object_position")
    elif fault == "nan":
        object.__setattr__(source, "goal_position", (float("nan"), 0.0, 0.0))
    elif fault == "overflow":
        object.__setattr__(source, "object_position", (-1e308, 0.0, 0.02))
        object.__setattr__(source, "goal_position", (1e308, 0.0, 0.02))
    elif fault == "history":
        history = object()
    else:
        object.__setattr__(history, "closing_count", 6)
    with pytest.raises((ValueError, AttributeError)):
        teacher.decide(source, history=history)
    assert teacher.steps == 0 and teacher.phase == "align"


def test_should_enforce_one_bounded_fifty_decision_lifetime():
    teacher = policy.PushCubeTeacher()
    for _ in range(50):
        teacher.decide(state((1.0, 0.0, 0.1)), history=ZERO)
    assert teacher.steps == 50
    with pytest.raises(ValueError):
        teacher.decide(state(), history=ZERO)
    assert teacher.steps == 50


def install_environment(monkeypatch, ledger, *, success=True):
    environment = FakePushEnvironment(ledger, success=success)
    environment.reconfiguration_freq = 0

    def factory():
        ledger["factory_attempts"] += 1
        ledger["factory_returns"] += 1
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    return environment


@pytest.fixture(scope="module")
def accepted(ledger):
    with pytest.MonkeyPatch.context() as patch:
        environment = install_environment(patch, ledger)
        result = collector.collect_push_cube_teacher()
        assert result.native_work is not None
    return result, environment


@pytest.mark.parametrize("seed", [17, 19, 20])
def test_should_release_owned_successful_train_labels_with_actual_float32_commands(
    seed, accepted, ledger, monkeypatch
):
    if seed == 17:
        result, environment = accepted
    else:
        environment = install_environment(monkeypatch, ledger)
        result = collector.collect_push_cube_teacher(seed=seed)
        assert result.native_work is not None
    assert result.error is None and result.final_success is True
    assert result.definition == policy.DEFINITION and result.seed == seed and result.role == "train"
    assert (
        result.decision_attempts
        == result.decision_returns
        == len(result.states)
        == len(result.decisions)
        == 2
    )
    assert result.unreturned_work_possible is False
    assert result.native_work.completed_steps == 2 and environment.close_calls == 1
    assert result.transitions == result.episode.transitions
    assert (
        result.attempted_actions
        == result.returned_actions
        == tuple(row.action for row in result.transitions)
    )
    owned = result.training_episode()
    assert owned is not result.episode and owned == result.episode
    assert owned.context == f"push-cube-train-{seed}" and owned.role == "train"
    assert owned.spec.observation_dim == 12
    for index, row in enumerate(owned.transitions):
        sent = tuple(float(value) for value in environment.commands[index][0])
        assert row.action == sent
        assert row.observation[7:] == (
            (0.0,) * 5 if index == 0 else (*owned.transitions[index - 1].action, index / 6)
        )
        assert row.next_observation[7:] == (*sent, (index + 1) / 6)
        assert row.action[3] == result.decisions[index].action[3] == -1.0
    queries = ledger["teacher_attempts"]
    assert result.training_episode() == owned
    assert ledger["teacher_attempts"] == queries
    object.__setattr__(owned.transitions[0], "reward", 0.0)
    assert result.episode.transitions[0].reward == 1.0


def test_should_preserve_a_complete_negative_episode_without_training_release(ledger, monkeypatch):
    environment = install_environment(monkeypatch, ledger, success=False)
    result = collector.collect_push_cube_teacher()
    assert result.native_work is not None
    assert result.error is None and result.final_success is False
    assert result.episode is not None and len(result.episode.transitions) == 2
    assert result.episode.transitions[-1].terminated
    assert result.native_work.completed_steps == 2 and environment.close_calls == 1
    assert any(
        row.action != decision.action
        for row, decision in zip(result.transitions, result.decisions, strict=True)
    )
    assert result.returned_actions == tuple(
        tuple(float(value) for value in command[0]) for command in environment.commands
    )
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize(
    "options",
    [
        {"seed": True},
        {"seed": 18},
        {"seed": 23},
        {"seed": 24},
        {"seed": 31},
        {"role": "probe_train"},
        {"role": "probe_test"},
        {"role": "final_test"},
    ],
)
def test_should_refuse_undeclared_train_authority_before_port_construction(monkeypatch, options):
    monkeypatch.setattr(
        collector,
        "ManiSkillPushCubeEpisode",
        lambda *args, **kw: pytest.fail("invalid request constructed port"),
    )
    with pytest.raises(ValueError):
        collector.collect_push_cube_teacher(**options)


@pytest.mark.parametrize("fault", ["teacher", "native", "cleanup", "source"])
def test_should_preserve_paid_teacher_and_native_prefix_without_retry(ledger, monkeypatch, fault):
    environment = install_environment(monkeypatch, ledger)
    if fault == "source":
        environment.grasp_error = RuntimeError("initial physical source unavailable")
    elif fault == "teacher":

        class FailedTeacher(policy.PushCubeTeacher):
            def decide(self, *args, **options):
                if self.steps == 1:
                    ledger["teacher_attempts"] += 1
                    raise RuntimeError("paid teacher label unavailable")
                return super().decide(*args, **options)

        monkeypatch.setattr(collector, "PushCubeTeacher", FailedTeacher)
    elif fault == "native":
        environment.after_step = lambda: setattr(
            environment, "step_error", RuntimeError("native return unavailable")
        )
    else:
        close = environment.close

        def failed_close():
            close()
            ledger["close_returns"] -= 1
            raise RuntimeError("cleanup unavailable")

        monkeypatch.setattr(environment, "close", failed_close)
    result = collector.collect_push_cube_teacher()
    assert result.native_work is not None
    assert result.error is not None
    assert result.final_success is (fault == "cleanup")
    assert result.decision_attempts == (0 if fault == "source" else 2)
    assert result.decision_returns == (0 if fault == "source" else 1 if fault == "teacher" else 2)
    assert len(result.decisions) == result.decision_returns
    assert result.native_work.step_attempts == (
        0 if fault == "source" else 1 if fault == "teacher" else 2
    )
    assert result.native_work.completed_steps == (
        0 if fault == "source" else 2 if fault == "cleanup" else 1
    )
    assert len(environment.commands) == result.native_work.step_attempts
    assert environment.close_calls == 1
    assert len(result.transitions) == result.native_work.completed_steps
    assert len(result.attempted_actions) == result.native_work.step_attempts
    assert len(result.returned_actions) == result.native_work.completed_steps
    if fault == "native":
        assert result.native_work.possible_failed_physics_steps == 5
        assert result.unreturned_work_possible is True
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize(
    "fault", ["role", "context", "decision_count", "label", "cost", "error", "unknown"]
)
def test_should_refuse_replaced_binding_labels_or_costs_at_release(accepted, fault):
    original, _ = accepted
    result = original
    if fault == "role":
        result = replace(original, role="probe_train")
    elif fault == "context":
        result = replace(original, episode=replace(original.episode, context="push-cube-train-19"))
    elif fault == "decision_count":
        result = replace(original, decision_returns=3)
    elif fault == "label":
        changed = replace(original.decisions[0], stage="push", action=(0.5, 0.5, 0.5, -1.0))
        result = replace(original, decisions=(changed, *original.decisions[1:]))
    elif fault == "cost":
        result = replace(original, native_work=replace(original.native_work, completed_steps=1))
    elif fault == "error":
        result = replace(original, error="preserved failure")
    else:
        result = replace(original, unreturned_work_possible=True)
    with pytest.raises(ValueError):
        result.training_episode()


def test_should_refuse_a_corrupted_decision_before_native_control(ledger, monkeypatch):
    environment = install_environment(monkeypatch, ledger)
    original = policy.PushCubeTeacher.decide

    def corrupted(owner, *args, **options):
        decision = original(owner, *args, **options)
        object.__setattr__(decision, "action", (1.0, 0.0, 0.0, -1.0))
        return decision

    monkeypatch.setattr(policy.PushCubeTeacher, "decide", corrupted)
    result = collector.collect_push_cube_teacher()
    assert result.native_work is not None
    assert result.error is not None and result.decision_returns == 1
    assert result.native_work.completed_steps == 0
    assert environment.commands == [] and environment.close_calls == 1


@pytest.mark.parametrize("field", ["action", "next_observation"])
def test_should_refuse_an_altered_return_before_another_teacher_query(ledger, monkeypatch, field):
    environment = install_environment(monkeypatch, ledger)

    class AlteredPort(native.ManiSkillPushCubeEpisode):
        def step(self, action):
            row = super().step(action)
            if field == "action":
                return replace(row, action=(0.5, 0.0, 0.0, -1.0))
            return replace(
                row,
                next_observation=(
                    row.next_observation[0],
                    row.next_observation[1] + 0.25,
                    *row.next_observation[2:],
                ),
            )

    monkeypatch.setattr(collector, "ManiSkillPushCubeEpisode", AlteredPort)
    result = collector.collect_push_cube_teacher()
    assert result.native_work is not None
    assert result.error is not None and result.decision_attempts == 1
    assert len(environment.commands) == result.native_work.completed_steps == 1
    assert environment.close_calls == 1
    if field == "action":
        assert result.returned_actions == ((0.5, 0.0, 0.0, -1.0),)
        assert result.returned_actions != result.attempted_actions
    with pytest.raises(ValueError):
        result.training_episode()


def test_should_refuse_changed_physics_before_teacher_or_control(ledger, monkeypatch):
    environment = install_environment(monkeypatch, ledger)

    class ChangedPhysics(native.ManiSkillPushCubeEpisode):
        @property
        def metadata(self):
            data = super().metadata
            data["control_physics_steps_per_control"] = 6
            return data

    monkeypatch.setattr(collector, "ManiSkillPushCubeEpisode", ChangedPhysics)
    result = collector.collect_push_cube_teacher()
    assert result.native_work is not None
    assert result.error is not None and result.decision_attempts == 0
    assert environment.commands == [] and environment.close_calls == 1
