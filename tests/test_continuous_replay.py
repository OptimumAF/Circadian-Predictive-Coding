"""Fixed warm-TRAIN replay in the released continuous runner, never probe data.

Shared four-family phase-one fixture plus one BP/single-probe composition:
34 native updates/73 rows, five fits/ten rows/20 values, five initializers/1,280
normal values. All other tests refuse early or stub/fork these owned models.
No additional optimizer oracle, comparative experiment or native simulation.
"""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_learnability_demo as demo
from src.adapters import continuous_probe as probe
from src.app import continuous_stream_rehearsal as replay
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


def episode(name="A"):
    if name == "A":
        positions, actions, context, role = (
            [0.0, 2.0, 4.0],
            [(-2.0, -1.0), (6.0, 1.0)],
            "context-A",
            "train",
        )
    elif name == "B":
        positions, actions, context, role = (
            [10.0, 11.0, 12.0, 13.0],
            [(5.0, 1.0), (4.0, 0.5), (3.0, 0.0)],
            "context-B",
            "train",
        )
    elif name == "Ftrain":
        positions, actions, context, role = (
            [-1.0, 1.0, 2.0],
            [(2.0, 0.5), (1.0, -0.5)],
            "fresh-F",
            "probe_train",
        )
    else:
        positions, actions, context, role = (
            [-0.5, 0.5, 1.5],
            [(1.5, 0.25), (1.25, -0.25)],
            "fresh-F",
            "probe_test",
        )
    observations = [(value, float(name == "B")) for value in positions]
    declared = ContinuousActionSpec(2, ("m", "m"), (-2.0, -1.0), (6.0, 1.0))
    return ContinuousEpisode(
        "synthetic-" + name,
        context,
        role,
        declared,
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                action,
                0.0,
                observations[index + 1],
                index == len(actions) - 1,
                False,
            )
            for index, action in enumerate(actions)
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


@pytest.fixture(scope="module")
def completed():
    records: dict[str, Any] = {}
    warm, new = episode(), episode("B")
    original = asdict(warm), asdict(new)
    original_runtime, original_helper = demo.ActorShadowRuntime, demo.ContinuousStreamRehearsal
    with pytest.MonkeyPatch.context() as patch:

        def runtime(*args, **options):
            owner = original_runtime(*args, **options)
            records.setdefault(family, {}).update(
                runtime=owner,
                source=args[0],
                initial=args[0].snapshot_state(),
                wake=options["budget"],
            )
            return owner

        def helper(*args, **options):
            value = original_helper(*args, **options)
            records.setdefault(family, {}).update(helper=value, replay=options["replay_budget"])
            return value

        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(demo, "ContinuousStreamRehearsal", helper)
        for family in FAMILIES:
            result = demo.run_continuous_learnability(
                warm,
                new,
                phase_updates=1,
                learner_family=family,
                replay_plan=replay.ContinuousReplayPlan(1, 6),
            )
            records[family]["result"] = result
            assert (asdict(warm), asdict(new)) == original
    return records


@pytest.mark.parametrize("family", FAMILIES)
def test_should_publish_native_wake_and_replay_updates_while_wake_budget_stays_three(
    completed, family
):
    record = completed[family]
    result, runtime = record["result"], record["runtime"]
    assert result.report["complete"] is True and result.candidate is not None
    state = result.candidate.snapshot_state()
    assert state.completed_updates == 6 and state.training_examples == 13
    assert len(runtime.applied_updates) == len(runtime.applied_consolidations) == 3
    assert record["wake"].updates_completed == 3 and record["wake"].replay_examples_completed == 0
    assert (
        record["replay"].updates_completed == 0 and record["replay"].replay_examples_completed == 6
    )
    assert record["wake"] is not record["replay"]
    assert_state(record["initial"], record["source"].snapshot_state())
    assert_state(record["initial"], runtime.actor.snapshot_state().state)
    assert_state(state, runtime.candidate_snapshot().state)
    assert [event.episode_id for event in runtime.applied_updates] == [
        "synthetic-A",
        "synthetic-B",
        "synthetic-A",
    ]
    if family in ("neutral-cpc", "active-cpc"):
        assert state.wake_state.wake_updates == 6
    report = record["helper"].report
    assert (
        report["planned_events"] == report["committed_updates"] == report["returned_updates"] == 3
    )
    assert report["charged_exposures"] == 6 and report["bank_rows"] == 2


def test_should_keep_pc_neutral_parameters_momentum_receipts_and_common_math_exact(completed):
    pc, neutral = (completed[family]["result"] for family in ("pc", "neutral-cpc"))
    first, second = pc.candidate.snapshot_state(), neutral.candidate.snapshot_state()
    assert replace(first.policy, circadian=None) == replace(second.policy, circadian=None)
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    assert pc.report["metrics"] == neutral.report["metrics"]
    assert pc.report["checkpoints"] == neutral.report["checkpoints"]
    assert pc.report["applied_updates"] == neutral.report["applied_updates"]


@pytest.mark.parametrize(
    "damage", ["type", "corrupt_flag", "row_capacity", "byte_capacity", "whole_bank_quota"]
)
def test_should_validate_replay_plan_and_warm_bank_before_fit_or_model(monkeypatch, damage):
    value: Any = replay.ContinuousReplayPlan(1, 6)
    if damage == "type":
        value = object()
    elif damage == "corrupt_flag":
        object.__setattr__(value, "every_updates", True)
    elif damage == "row_capacity":
        value = replace(value, max_retained_rows=1)
    elif damage == "byte_capacity":
        value = replace(value, max_retained_bytes=1)
    else:
        value = replace(value, max_replay_examples=1)
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_a, **_k: pytest.fail("invalid replay reached fitting"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(
            episode(), episode("B"), phase_updates=1, replay_plan=value
        )


@pytest.fixture(scope="module")
def with_probe():
    scopes, helpers = [], []
    original_runtime, original_helper = probe.ActorShadowRuntime, demo.ContinuousStreamRehearsal
    with pytest.MonkeyPatch.context() as patch:

        def runtime(*args, **options):
            owner = original_runtime(*args, **options)
            scopes.append((owner, options["budget"], options["clock"]))
            return owner

        def helper(*args, **options):
            owner = original_helper(*args, **options)
            helpers.append((owner, options["replay_budget"]))
            return owner

        patch.setattr(probe, "ActorShadowRuntime", runtime)
        patch.setattr(demo, "ContinuousStreamRehearsal", helper)
        result = demo.run_continuous_learnability(
            episode(),
            episode("B"),
            phase_updates=1,
            replay_plan=replay.ContinuousReplayPlan(1, 6),
            probe_plan=probe.ContinuousProbePlan(episode("Ftrain"), episode("Ftest")),
        )
    return result, scopes, helpers[0]


def test_should_keep_isolated_probe_allowances_separate_from_warm_bank_replay(with_probe):
    result, scopes, (helper, replay_budget) = with_probe
    assert result.report["complete"] is True and result.candidate is not None
    assert result.candidate.snapshot_state().completed_updates == 6
    assert result.candidate.snapshot_state().training_examples == 13
    points = result.report["probes"]
    assert [point["benchmark_updates"] for point in points] == [0, 2, 4, 6]
    assert [point["benchmark_wake_updates"] for point in points] == [0, 1, 2, 3]
    assert len(scopes) == 4 and len({id(budget) for _owner, budget, _clock in scopes}) == 4
    for point, (runtime, budget, clock) in zip(points, scopes, strict=True):
        nested = point["report"]
        assert nested["complete"] is True
        assert nested["source_updates"] == point["benchmark_updates"]
        assert nested["progress"]["updates_completed"] == 1
        assert nested["work"]["training_rows_completed"] == 2
        assert nested["work"]["prediction_returns"] == 6
        assert nested["work"]["prediction_rows_returned"] == 14
        assert budget is not replay_budget and budget.replay_examples_completed == 0
        assert budget.updates_completed == 1 and budget.budget.max_training_updates == 1
        assert clock.now() == 1 and runtime.applied_consolidations == ()
        assert nested["applied_updates"][0]["episode_id"] == "synthetic-Ftrain"
        assert "replay" not in nested and "update_diagnostics" not in nested
        assert nested["checkpoints"][0]["scores"]["fresh"]["role"] == "probe_test"
    assert helper.report["charged_exposures"] == replay_budget.replay_examples_completed == 6


@pytest.mark.parametrize("family", FAMILIES)
def test_should_label_wake_and_native_progress_and_account_actual_replay_work(completed, family):
    result = completed[family]["result"]
    report = result.report
    assert report["progress"]["updates_completed"] == 3
    assert report["progress"]["native_updates_completed"] == 6
    assert report["progress"]["replay_updates_completed"] == 3
    assert report["progress"]["replay_rows_charged"] == 6
    assert [point["wake_updates"] for point in report["checkpoints"]] == [0, 1, 2, 3]
    assert [point["native_updates"] for point in report["checkpoints"]] == [0, 2, 4, 6]
    assert [point["training_examples"] for point in report["checkpoints"]] == [0, 4, 9, 13]
    work = report["work"]
    assert work["optimizer_updates"] == 6 and work["training_rows"] == 13
    assert work["replay_optimizer_updates"] == work["replay_operation_forks"] == 3
    assert work["replay_budget_rows"] == 6
    assert work["replay_retained_numeric_bytes"] == 2 * (2 * 2 + 2 + 1) * 8
    assert work["replay_input_feature_target_numeric_bytes"] == 6 * (2 + 2) * 8
    assert report["replay"]["bank_episode_ids"] == ["synthetic-A"]
    assert report["replay"]["bank_contexts"] == ["context-A"]
    assert len(report["applied_consolidations"]) == 3
    if family != "bp":
        assert len(report["native_updates"]) == 6
        assert [item["training_rows"] for item in report["native_updates"]] == [2, 2, 3, 2, 2, 2]
        assert work["native_update_counts"]["published_updates"] == 6


@pytest.mark.parametrize("explicit", [False, True])
def test_should_preserve_default_json_wrapper_without_replay_fields_or_extra_calls(
    monkeypatch, explicit
):
    calls, returned = [], dict(complete=True, sentinel="old default schema")

    def run(*args, **options):
        calls.append((args, options))
        return demo.ContinuousLearnabilityResult(returned, None)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    options: dict[str, Any] = {"replay_plan": None} if explicit else {}
    actual = demo.run_continuous_learnability_demo(episode(), episode("B"), **options)
    assert actual is returned and len(calls) == 1
    assert calls[0][1] == dict(expected_standardization=None)
    assert "replay" not in actual and "applied_consolidations" not in actual


def test_should_forward_explicit_plan_without_a_second_or_hidden_runner_call(monkeypatch):
    value, calls = replay.ContinuousReplayPlan(2, 6), []

    def run(*_args, **options):
        calls.append(options)
        return demo.ContinuousLearnabilityResult(
            dict(complete=False, error="stubbed paid replay"), None
        )

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    returned = demo.run_continuous_learnability_demo(episode(), episode("B"), replay_plan=value)
    assert returned["complete"] is False and len(calls) == 1
    assert calls[0]["replay_plan"] is value
