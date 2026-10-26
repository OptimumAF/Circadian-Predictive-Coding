"""Existing continuous training candidate composed with frozen fake robot IO.

One shared four-family phase-one training fixture; all other training calls are
stubbed. Complete zero-reward runtime outcomes are valid and never ranked.

Per platform: four parameter initializations (seven PC/BC constructor entries),
11,776 normal values, 12 native updates/24 training rows. Readouts each use
32 conditioning, residual, and actor calls/64 rows. Four fake native rollouts
use eight predictions/controls, four resets/closes/name/limit reads, and twelve
joint samples/108 coordinates. Other cases only fork owned state or stub IO;
no additional fitting, updates, initializers, or native simulations. PC kernels
are bounded by 45 energy evaluations and 36 latent proposals.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as entry
from src.adapters import continuous_learnability_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.infra import maniskill_continuous as native
from test_maniskill_joint_positions import StateEnvironment


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


def episode(seed=17, *, role="train", dimension=42):
    warm = seed == 17
    positions = [0.0, 2.0, 4.0] if warm else [10.0, 11.0, 12.0]
    observations = [(value, 0.0 if warm else 1.0) + (0.0,) * (dimension - 2) for value in positions]
    actions = (
        [(-1.0, -1.0, 1.0, -1.0), (1.0, 1.0, -1.0, 1.0)]
        if warm
        else [(0.5, 1.0, -0.5, 1.0), (0.4, 0.5, -0.3, -1.0)]
    )
    declared = ContinuousActionSpec(dimension, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    return ContinuousEpisode(
        f"pick-cube-task-teacher-{seed}",
        "pick-cube" if warm else "pick-cube-train-19",
        role,
        declared,
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                actions[index],
                float(index == 1),
                observations[index + 1],
                index == 1,
                False,
            )
            for index in range(2)
        ),
    )


def assert_state(first, second):
    assert type(first) is type(second) and first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    a, b = getattr(first, "wake_state", None), getattr(second, "wake_state", None)
    if a is None:
        assert b is None
    else:
        assert b is not None
        assert a.wake_updates == b.wake_updates
        np.testing.assert_array_equal(a.chemical_state, b.chemical_state)
        np.testing.assert_array_equal(a.plasticity_state, b.plasticity_state)


def clone_result(result):
    return demo.ContinuousLearnabilityResult(
        deepcopy(result.report), result.candidate.fork() if result.candidate is not None else None
    )


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    folder = tmp_path_factory.mktemp("continuous-learned-robot")
    results: dict[str, Any] = {}
    training: dict[str, Any] = {}
    rollouts: dict[str, Any] = {}
    environments: list[Any] = []
    warm, new = episode(), episode(19)
    original_train, original_rollout = entry.run_continuous_learnability, entry.run_frozen_rollout
    with pytest.MonkeyPatch.context() as patch:

        def train(*args, **kwargs):
            result = original_train(*args, **kwargs)
            training[family] = result
            return result

        def rollout(*args, **kwargs):
            owner = kwargs["learner"]
            before = owner.snapshot_state()
            result = original_rollout(*args, **kwargs)
            assert_state(before, owner.snapshot_state())
            rollouts[family] = result
            return result

        def factory(**options):
            assert options == {}
            environment = StateEnvironment()
            environments.append(environment)
            return environment

        patch.setattr(entry, "run_continuous_learnability", train)
        patch.setattr(entry, "run_frozen_rollout", rollout)
        patch.setattr(native, "_make_environment", factory)
        patch.setattr(smoke, "version", lambda _: "fake-learned-robot")
        for family in FAMILIES:
            output = folder / f"{family}.json"
            results[family] = entry.run_learning_rollout(
                output, warm, new, learner_family=family, phase_updates=1
            )
            assert output.exists() and output.with_name(output.name + ".rollout.json").exists()
            assert json.loads(output.read_text(encoding="utf8")) == json.loads(
                json.dumps(results[family])
            )
    return results, training, rollouts, environments


@pytest.mark.parametrize("family", FAMILIES)
def test_should_pass_actual_detached_trained_candidate_to_frozen_native_boundary(learned, family):
    combined, training, rollouts, environments = learned
    report, trained, native_result = combined[family], training[family], rollouts[family]
    assert report["complete"] is True
    assert trained.report["complete"] is True and native_result.report["complete"] is True
    state = trained.candidate.snapshot_state()
    assert state.completed_updates == 3 and state.training_examples == 6
    assert (
        native_result.report["source_updates"] == 3
        and native_result.report["source_training_rows"] == 6
    )
    assert native_result.report["seed"] == 17 and native_result.report["context"] == "pick-cube"
    assert native_result.report["optimizer_updates"] == 0 and native_result.corrections is None
    assert (
        native_result.report["frozen_state_exact"] is True
        and native_result.report["original_owner_exact"] is True
    )
    assert native_result.report["joint_position_exact"] is True
    if family in ("neutral-cpc", "active-cpc"):
        assert state.wake_state.wake_updates == 3
        assert native_result.report["source_wake_updates"] == 3
    assert all(
        len(environment.commands) == 2
        and environment.reset_calls == [17]
        and environment.close_calls == 1
        for environment in environments
    )


@pytest.fixture
def stubbed(monkeypatch, learned):
    train_calls, native_calls = [], []
    stored = learned[1]["active-cpc"]

    def train(*args, **kwargs):
        train_calls.append((args, kwargs))
        return clone_result(stored)

    def forbidden_native(*args, **kwargs):
        native_calls.append((args, kwargs))
        pytest.fail("invalid training evidence reached native rollout")

    monkeypatch.setattr(entry, "run_continuous_learnability", train)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden_native)
    return train_calls, native_calls, stored


@pytest.mark.parametrize(
    "damage",
    ["warm_role", "new_role", "context", "identity", "dimension", "units", "bounds", "visual_type"],
)
def test_should_refuse_invalid_training_native_binding_before_fit_or_output(
    monkeypatch, tmp_path, damage
):
    warm, new = episode(), episode(19)
    options: dict[str, Any] = {}
    if damage == "warm_role":
        warm = replace(warm, role="final_test")
    elif damage == "new_role":
        new = replace(new, role="probe_train")
    elif damage == "context":
        warm = replace(warm, context="pick-cube-train-19")
    elif damage == "identity":
        new = replace(new, episode_id=warm.episode_id)
    elif damage == "dimension":
        warm, new = episode(dimension=3), episode(19, dimension=3)
    elif damage in ("units", "bounds"):
        declared = replace(
            warm.spec,
            **(
                {"action_units": ("m",) * 4} if damage == "units" else {"lower_bounds": (-2.0,) * 4}
            ),
        )
        warm, new = replace(warm, spec=declared), replace(new, spec=declared)
    else:
        options["visual_observation"] = 1
    monkeypatch.setattr(
        entry,
        "run_continuous_learnability",
        lambda *_a, **_k: pytest.fail("invalid input started training"),
    )
    output = tmp_path / "result.json"
    with pytest.raises(ValueError):
        entry.run_learning_rollout(output, warm, new, **options)
    assert not output.exists() and not output.with_name(output.name + ".rollout.json").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"learner_family": None},
        {"learner_family": "PC"},
        {"learner_family": True},
        {"phase_updates": 0},
        {"phase_updates": True},
        {"phase_updates": 129},
    ],
)
def test_should_refuse_invalid_family_and_phase_limits_before_training(
    monkeypatch, tmp_path, options
):
    monkeypatch.setattr(
        entry,
        "run_continuous_learnability",
        lambda *_a, **_k: pytest.fail("invalid settings started training"),
    )
    output = tmp_path / "result.json"
    with pytest.raises(ValueError):
        entry.run_learning_rollout(output, episode(), episode(19), **options)
    assert not output.exists()


@pytest.mark.parametrize("existing", ["combined", "native"])
def test_should_preserve_either_existing_output_before_fit(monkeypatch, tmp_path, existing):
    output = tmp_path / "result.json"
    target = output if existing == "combined" else output.with_name(output.name + ".rollout.json")
    target.write_bytes(b"preserved evidence")
    monkeypatch.setattr(
        entry,
        "run_continuous_learnability",
        lambda *_a, **_k: pytest.fail("existing output started training"),
    )
    with pytest.raises(FileExistsError):
        entry.run_learning_rollout(output, episode(), episode(19))
    assert target.read_bytes() == b"preserved evidence"
    if existing == "native":
        assert not output.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "missing",
        "foreign",
        "wrong_family",
        "circadian",
        "policy",
        "progress",
        "source_id",
        "checkpoint",
        "wake",
    ],
)
def test_should_keep_returned_training_and_refuse_bad_candidate_before_native(
    stubbed, learned, monkeypatch, tmp_path, damage
):
    _, native_calls, stored = stubbed
    result = clone_result(stored)
    if damage == "incomplete":
        result.report.update(complete=False, error="retained partial learning")
    elif damage in ("missing", "foreign"):
        invalid: Any = None if damage == "missing" else object()
        result = demo.ContinuousLearnabilityResult(result.report, invalid)
    elif damage == "wrong_family":
        result = clone_result(learned[1]["bp"])
    elif damage == "circadian":
        result = clone_result(learned[1]["neutral-cpc"])
        result.report["learner_family"] = "active-cpc"
    elif damage == "policy":
        result.report["policy"]["base"]["learning_rate"] = 0.02
    elif damage == "progress":
        result.report["progress"]["updates_completed"] = 2
    elif damage == "source_id":
        result.report["source_episodes"]["new"]["episode_id"] = "other-teacher"
    elif damage == "checkpoint":
        result.report["checkpoints"][-1]["owners_exact"] = False
    else:
        result.candidate._state = replace(result.candidate.snapshot_state(), wake_state=None)
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *_a, **_k: result)
    output = tmp_path / "result.json"
    report = entry.run_learning_rollout(
        output, episode(), episode(19), learner_family="active-cpc", phase_updates=1
    )
    assert report["complete"] is False and report["rollout_attempted"] is False
    assert report["training"] == result.report and report["rollout"] is None
    assert native_calls == [] and output.exists()
    assert not output.with_name(output.name + ".rollout.json").exists()


def native_stub(learned, *, mutate=None, corrections=None):
    def run(output, **options):
        assert options["train_seed"] == 17 and options["monitor_joint_positions"] is True
        raw = deepcopy(learned[2]["active-cpc"].report)
        if mutate is not None:
            mutate(raw, options["learner"])
        output.write_text(json.dumps(raw), encoding="utf8")
        return smoke.FrozenRolloutResult(raw, corrections)

    return run


@pytest.mark.parametrize(
    "damage",
    ["incomplete", "private", "source", "joint", "float_count", "bool_optimizer", "corrections"],
)
def test_should_retain_native_report_and_refuse_invalid_frozen_return(
    stubbed, learned, monkeypatch, tmp_path, damage
):
    def mutate(raw, _owner):
        if damage == "incomplete":
            raw.update(
                complete=False,
                error="one paid native step",
                work=dict(
                    step_attempts=1,
                    completed_steps=1,
                    completed_control_physics_steps=5,
                    possible_failed_physics_steps=0,
                ),
            )
        elif damage in ("private", "source", "joint"):
            raw[
                {
                    "private": "frozen_state_exact",
                    "source": "original_owner_exact",
                    "joint": "joint_position_exact",
                }[damage]
            ] = False
        elif damage == "float_count":
            raw["source_training_rows"] = 6.0
        elif damage == "bool_optimizer":
            raw["optimizer_updates"] = False

    monkeypatch.setattr(
        entry,
        "run_frozen_rollout",
        native_stub(
            learned, mutate=mutate, corrections=object() if damage == "corrections" else None
        ),
    )
    report = entry.run_learning_rollout(
        tmp_path / "result.json",
        episode(),
        episode(19),
        learner_family="active-cpc",
        phase_updates=1,
    )
    assert report["complete"] is False and report["rollout_attempted"] is True
    assert report["training"]["complete"] is True and report["rollout"] is not None
    assert report["candidate_unchanged"] is True
    if damage == "incomplete":
        assert report["rollout"]["work"]["completed_steps"] == 1
        assert report["rollout"]["error"] == "one paid native step"


def changed_wake(owner):
    state = owner.snapshot_state()
    wake, config = state.wake_state, state.policy.circadian
    chemicals = wake.chemical_state + 0.01
    gates = np.clip(np.exp(-config.plasticity_sensitivity * chemicals), config.min_plasticity, 1.0)
    owner.restore_state(
        replace(state, wake_state=replace(wake, chemical_state=chemicals, plasticity_state=gates))
    )


@pytest.mark.parametrize("damage", ["wake", "velocity"])
def test_should_detect_full_candidate_change_after_retaining_native_result(
    stubbed, learned, monkeypatch, tmp_path, damage
):
    def mutate(_raw, owner):
        if damage == "wake":
            changed_wake(owner)
        else:
            state = owner.snapshot_state()
            state.velocity[0][0, 0] += 0.01
            owner.restore_state(state)

    monkeypatch.setattr(entry, "run_frozen_rollout", native_stub(learned, mutate=mutate))
    report = entry.run_learning_rollout(
        tmp_path / "result.json",
        episode(),
        episode(19),
        learner_family="active-cpc",
        phase_updates=1,
    )
    assert report["complete"] is False and report["candidate_unchanged"] is False
    assert report["rollout"]["complete"] is True
    assert report["rollout"]["work"]["completed_steps"] == 2
    assert "state_guard_error" in report


def test_should_keep_original_native_error_and_separate_state_guard_failure(
    stubbed, monkeypatch, tmp_path
):
    def fail(_output, **options):
        changed_wake(options["learner"])
        raise RuntimeError("original native callback failed")

    monkeypatch.setattr(entry, "run_frozen_rollout", fail)
    report = entry.run_learning_rollout(
        tmp_path / "result.json",
        episode(),
        episode(19),
        learner_family="active-cpc",
        phase_updates=1,
    )
    assert report["complete"] is False and report["rollout_attempted"] is True
    assert "original native callback failed" in report["error"]
    assert "state_guard_error" in report and report["candidate_unchanged"] is False
    assert report["training"]["complete"] is True and report["rollout"] is None


def test_should_allow_complete_zero_reward_frozen_return_without_retraining(
    stubbed, learned, monkeypatch, tmp_path
):
    calls, _, _ = stubbed

    def zero(raw, _owner):
        raw.update(sparse_return=0.0, any_success=False, final_success=False)
        for row in raw["episode"]["transitions"]:
            row["reward"] = 0.0

    monkeypatch.setattr(entry, "run_frozen_rollout", native_stub(learned, mutate=zero))
    report = entry.run_learning_rollout(
        tmp_path / "result.json",
        episode(),
        episode(19),
        learner_family="active-cpc",
        phase_updates=1,
    )
    assert report["complete"] is True and len(calls) == 1
    assert report["rollout"]["sparse_return"] == 0 and report["rollout"]["final_success"] is False


@pytest.mark.parametrize(
    "hook,phase",
    [
        ("snapshot", "pre_native"),
        ("policy", "pre_native"),
        ("snapshot", "post_native"),
        ("policy", "post_native"),
    ],
)
def test_should_detect_native_owned_change_after_clean_public_candidate_callbacks(
    stubbed, learned, monkeypatch, tmp_path, hook, phase
):
    result = clone_result(learned[1]["active-cpc"])
    owner, observed, calls = result.candidate, [], []
    threshold = 2 if phase == "pre_native" else 3
    original_snapshot = owner.snapshot_state
    original_policy = vars(pc.NumpyContinuousPCLearner)["policy"].fget

    def mutate_after_return(value):
        observed.append(True)
        if len(observed) == threshold:
            owner._state.velocity[0][0, 0] += 0.01
        return value

    if hook == "snapshot":
        monkeypatch.setattr(
            owner, "snapshot_state", lambda: mutate_after_return(original_snapshot())
        )
    else:

        def policy(candidate):
            value = original_policy(candidate)
            return mutate_after_return(value) if candidate is owner else value

        monkeypatch.setattr(pc.NumpyContinuousPCLearner, "policy", property(policy))
    native = native_stub(learned)

    def rollout(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *_a, **_k: result)
    monkeypatch.setattr(entry, "run_frozen_rollout", rollout)
    report = entry.run_learning_rollout(
        tmp_path / "result.json",
        episode(),
        episode(19),
        learner_family="active-cpc",
        phase_updates=1,
    )
    assert report["complete"] is False and report["training"]["complete"] is True
    assert len(observed) == threshold
    assert report["rollout_attempted"] is (phase == "post_native")
    assert len(calls) == int(phase == "post_native")
    if phase == "post_native":
        assert report["rollout"]["complete"] is True and report["candidate_unchanged"] is False
        assert "state_guard_error" in report


def test_should_refuse_exact_candidate_class_replacement_during_snapshot_callback(
    stubbed, learned, monkeypatch, tmp_path
):
    result = clone_result(learned[1]["bp"])
    owner = result.candidate
    original = owner.snapshot_state

    class ForeignBP(bp.NumpyContinuousBCLearner):
        pass

    def snapshot():
        returned = original()
        owner.__class__ = ForeignBP
        return returned

    monkeypatch.setattr(owner, "snapshot_state", snapshot)
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *_a, **_k: result)
    report = entry.run_learning_rollout(
        tmp_path / "result.json", episode(), episode(19), learner_family="bp", phase_updates=1
    )
    assert report["complete"] is False and report["rollout_attempted"] is False
    assert report["training"]["complete"] is True and report["rollout"] is None


@pytest.mark.parametrize("terminal", [False, True])
def test_should_refuse_invalid_rgb_even_in_joined_or_terminal_next_observation(
    monkeypatch, tmp_path, terminal
):
    warm, new = episode(dimension=789), episode(19, dimension=789)
    records = list(new.transitions)
    index = 1 if terminal else 0
    changed = list(records[index].next_observation)
    changed[21] = -0.1 if terminal else 1.1
    records[index] = replace(records[index], next_observation=tuple(changed))
    if not terminal:
        records[1] = replace(records[1], observation=tuple(changed))
    new = replace(new, transitions=tuple(records))
    monkeypatch.setattr(
        entry,
        "run_continuous_learnability",
        lambda *_a, **_k: pytest.fail("invalid RGB reached training"),
    )
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="RGB"):
        entry.run_learning_rollout(output, warm, new, visual_observation=True)
    assert not output.exists()


def test_should_allow_finite_unbounded_visual_public_prefix_without_extra_training(
    monkeypatch, tmp_path
):
    warm, new = episode(dimension=789), episode(19, dimension=789)
    records = []
    for transition in new.transitions:
        observation, next_observation = (
            list(transition.observation),
            list(transition.next_observation),
        )
        observation[0], next_observation[0] = 1000.0, 1000.0
        observation[20], next_observation[20] = -1000.0, -1000.0
        records.append(
            replace(
                transition, observation=tuple(observation), next_observation=tuple(next_observation)
            )
        )
    new = replace(new, transitions=tuple(records))
    calls = []
    returned = dict(complete=False, error="stubbed after visual admission")

    def train(*args, **kwargs):
        calls.append((args, kwargs))
        return demo.ContinuousLearnabilityResult(returned, None)

    monkeypatch.setattr(entry, "run_continuous_learnability", train)
    report = entry.run_learning_rollout(
        tmp_path / "result.json", warm, new, visual_observation=True
    )
    assert len(calls) == 1 and report["training"] == returned
    assert report["complete"] is False and report["rollout_attempted"] is False


@pytest.mark.parametrize("route", ["state", "visual", "lighting"])
def test_should_route_cli_owned_teachers_and_explicit_learning_options_without_extra_training(
    monkeypatch, tmp_path, capsys, route
):
    reads, calls = [], []
    warm, new = episode(), episode(19)

    def read(path, seed, **options):
        reads.append((Path(path).name, seed, options))
        return warm if seed == 17 else new

    def lighting(path, **options):
        reads.append((Path(path).name, "lighting", options))
        return new, {"retained": True}

    def run(output, actual_warm, actual_new, **options):
        calls.append((output, actual_warm, actual_new, options))
        return dict(complete=True, training={"paid": 3}, rollout={"sparse_return": 0})

    monkeypatch.setattr(entry, "_read_teacher", read)
    monkeypatch.setattr(entry, "_read_lighting_teacher", lighting)
    monkeypatch.setattr(entry, "run_learning_rollout", run)
    output = tmp_path / "result.json"
    arguments = [
        "--output",
        str(output),
        "--warm-teacher",
        "warm.json",
        "--new-teacher",
        "new.json",
        "--learner-family",
        "neutral-cpc",
        "--phase-updates",
        "1",
    ]
    if route != "state":
        arguments.append("--visual-observation")
    if route == "lighting":
        arguments.append("--lighting-drift")
    assert entry.main(arguments) == 0
    assert reads == [
        ("warm.json", 17, dict(visual_observation=route != "state")),
        (
            "new.json",
            "lighting" if route == "lighting" else 19,
            dict(visual_observation=route != "state"),
        ),
    ]
    assert len(calls) == 1 and calls[0][:3] == (output, warm, new)
    assert calls[0][3] == dict(
        learner_family="neutral-cpc", phase_updates=1, visual_observation=route != "state"
    )
    assert json.loads(capsys.readouterr().out)["rollout"]["sparse_return"] == 0


@pytest.mark.parametrize(
    "extra", [["--lighting-drift"], ["--phase-updates", "0"], ["--learner-family", "foreign"]]
)
def test_should_refuse_invalid_cli_configuration_before_any_teacher_read(
    monkeypatch, tmp_path, extra
):
    monkeypatch.setattr(
        entry, "_read_teacher", lambda *_a, **_k: pytest.fail("invalid CLI read teacher")
    )
    monkeypatch.setattr(
        entry, "_read_lighting_teacher", lambda *_a, **_k: pytest.fail("invalid CLI read lighting")
    )
    monkeypatch.setattr(
        entry, "run_learning_rollout", lambda *_a, **_k: pytest.fail("invalid CLI reached learning")
    )
    with pytest.raises(SystemExit) as caught:
        entry.main(
            [
                "--output",
                str(tmp_path / "result.json"),
                "--warm-teacher",
                "warm.json",
                "--new-teacher",
                "new.json",
            ]
            + extra
        )
    assert caught.value.code == 2


def test_should_print_partial_paid_cli_report_and_return_failure_without_retry(
    monkeypatch, tmp_path, capsys
):
    calls = []
    monkeypatch.setattr(entry, "_read_teacher", lambda _path, seed, **_options: episode(seed))
    returned = dict(
        complete=False,
        error="native stopped",
        training=dict(complete=True, progress=dict(updates_completed=3)),
        rollout=dict(work=dict(completed_steps=1)),
    )

    def run(*args, **options):
        calls.append(options)
        return returned

    monkeypatch.setattr(entry, "run_learning_rollout", run)
    assert (
        entry.main(
            [
                "--output",
                str(tmp_path / "result.json"),
                "--warm-teacher",
                "warm.json",
                "--new-teacher",
                "new.json",
            ]
        )
        == 1
    )
    assert calls == [dict(learner_family="bp", phase_updates=8, visual_observation=False)]
    assert json.loads(capsys.readouterr().out) == returned


def test_should_check_candidate_after_observing_native_correction_result(
    stubbed, learned, monkeypatch, tmp_path
):
    observed = []
    original = smoke.FrozenRolloutResult.__getattribute__

    def rollout(output, **options):
        owner = options["learner"]

        def getattribute(result, name):
            value = original(result, name)
            if name == "corrections":
                observed.append(True)
                owner._state.velocity[0][0, 0] += 0.01
            return value

        monkeypatch.setattr(smoke.FrozenRolloutResult, "__getattribute__", getattribute)
        return native_stub(learned)(output, **options)

    monkeypatch.setattr(entry, "run_frozen_rollout", rollout)
    report = entry.run_learning_rollout(
        tmp_path / "result.json",
        episode(),
        episode(19),
        learner_family="active-cpc",
        phase_updates=1,
    )
    assert observed == [True]
    assert report["complete"] is False and report["candidate_unchanged"] is False
    assert report["rollout"]["complete"] is True and "state_guard_error" in report
