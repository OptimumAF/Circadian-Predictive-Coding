"""Shared physical quotas across fresh policy-generated reward episodes."""

from copy import deepcopy
from dataclasses import asdict
import pickle

import numpy as np
import pytest

import src.app.key_door_training as training
from src.core.key_door_navigation import KeyDoorLayout, KeyDoorNavigation, scripted_actions
from src.core.physical_interaction_quota import PhysicalInteractionQuota
from test_key_door_training import Source, collect


def counts(maximum, attempted=0, completed=0, failed=0):
    return dict(
        maximum_steps=maximum,
        attempted_steps=attempted,
        completed_steps=completed,
        failed_steps=failed,
    )


@pytest.mark.parametrize("mapped", (False, True))
def test_should_fill_one_quota_with_an_early_goal_and_a_fresh_short_final_episode(mapped):
    layout = KeyDoorLayout("B", movement_action_map=(1, 2, 3, 0)) if mapped else KeyDoorLayout("A")
    source = Source(scripted_actions(layout, 1))
    source_before = pickle.dumps(source.state)
    sampler = np.random.Generator(np.random.PCG64(29))
    expected = np.random.Generator(np.random.PCG64(29))
    quota = PhysicalInteractionQuota(12)

    first = collect(
        source, sampler, layout=layout, interaction_quota=quota, episode_id="quota-first"
    )
    assert first["termination"] == "terminal" and first["total_return"] == 1
    assert first["environment_steps"] == first["sampling_draws"] == 8
    assert first["interaction_quota_before"] == counts(12)
    assert first["interaction_quota_after"] == counts(12, 8, 8)
    assert quota.remaining_steps == 4

    second = collect(
        source, sampler, layout=layout, interaction_quota=quota, episode_id="quota-final"
    )
    assert second["status"] == "complete" and second["termination"] == "truncated"
    assert second["total_return"] == 0 and second["environment_steps"] == 4
    assert (
        second["prediction_calls_attempted"]
        == second["prediction_calls_returned"]
        == second["sampling_draws"]
        == 4
    )
    assert second["interaction_quota_before"] == counts(12, 8, 8)
    assert second["interaction_quota_after"] == counts(12, 12, 12)
    assert quota.remaining_steps == 0 and asdict(quota.snapshot()) == counts(12, 12, 12)
    assert first["sampler_after"] == second["sampler_before"]
    assert len(source.sessions) == 2 and source.sessions[0] is not source.sessions[1]
    assert [len(session.inputs) for session in source.sessions] == [8, 4]
    np.testing.assert_array_equal(source.sessions[0].inputs[0], source.sessions[1].inputs[0])
    assert source.sessions[1].inputs[0][0, 6] == 1
    assert all(features[0, 6] == 0 for features in source.sessions[1].inputs[1:])
    assert (
        tuple(slot["action"] for slot in second["transitions"]) == scripted_actions(layout, 1)[:4]
    )
    assert all(not slot["terminal"] for slot in second["transitions"])
    for _ in range(12):
        expected.random()
    assert sampler.bit_generator.state == second["sampler_after"] == expected.bit_generator.state
    assert pickle.dumps(source.state) == source_before
    second["interaction_quota_before"]["attempted_steps"] = 0
    first["interaction_quota_after"]["completed_steps"] = 0
    assert asdict(quota.snapshot()) == counts(12, 12, 12)
    assert second["interaction_quota_after"] == counts(12, 12, 12)


def test_should_honor_smaller_per_episode_cap_without_renewing_shared_quota():
    quota = PhysicalInteractionQuota(5)
    source = Source(scripted_actions(KeyDoorLayout("A"), 1))
    first = collect(source, interaction_quota=quota, max_steps=3)
    assert first["termination"] == "truncated" and first["environment_steps"] == 3
    assert quota.remaining_steps == 2
    second = collect(source, interaction_quota=quota, max_steps=3)
    assert second["termination"] == "truncated" and second["environment_steps"] == 2
    assert asdict(quota.snapshot()) == counts(5, 5, 5)


@pytest.mark.parametrize("unavailable", ("exhausted", "failed", "pending", "corrupt"))
def test_should_refuse_unavailable_quota_before_environment_source_prediction_or_rng(
    monkeypatch, unavailable
):
    quota = PhysicalInteractionQuota(2)
    if unavailable == "exhausted":
        for _ in range(2):
            quota.begin_step()
            quota.complete_step()
    elif unavailable == "failed":
        quota.begin_step()
        quota.fail_step()
    elif unavailable == "pending":
        quota.begin_step()
    else:
        object.__setattr__(quota, "_completed_steps", 3)
    source = Source()
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)
    environment_calls = []

    def forbidden_environment(*args):
        environment_calls.append(args)
        raise AssertionError("unavailable quota constructed an environment")

    monkeypatch.setattr(training, "KeyDoorNavigation", forbidden_environment)
    with pytest.raises(ValueError):
        collect(source, sampler, interaction_quota=quota)
    assert environment_calls == [] and source.snapshots == 0 and source.sessions == []
    assert sampler.bit_generator.state == before


class QuotaSubclass(PhysicalInteractionQuota):
    pass


@pytest.mark.parametrize("invalid", (0, False, True, "12", object(), QuotaSubclass(12)))
def test_should_refuse_non_exact_quota_before_source_work(invalid):
    source = Source()
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)
    with pytest.raises(ValueError):
        collect(source, sampler, interaction_quota=invalid)
    assert source.snapshots == 0 and source.sessions == []
    assert sampler.bit_generator.state == before


@pytest.mark.parametrize(
    "options", ({"max_steps": 0}, {"max_steps": True}, {"role": "final_test"}, {"cue": 0})
)
def test_should_validate_other_arguments_without_spending_an_admitted_quota(options):
    quota = PhysicalInteractionQuota(12)
    source = Source()
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)
    with pytest.raises(ValueError):
        collect(source, sampler, interaction_quota=quota, **options)
    assert source.snapshots == 0 and source.sessions == []
    assert sampler.bit_generator.state == before
    assert asdict(quota.snapshot()) == counts(12)


def test_should_not_charge_physics_for_a_sampled_illegal_action():
    quota = PhysicalInteractionQuota(12)
    sampler = np.random.Generator(np.random.PCG64(29))
    expected = np.random.Generator(np.random.PCG64(29))
    expected.random()
    report = collect(Source((3,)), sampler, interaction_quota=quota)
    assert report["termination"] == "illegal_action" and report["sampling_draws"] == 1
    assert report["environment_steps_attempted"] == report["environment_steps"] == 0
    assert report["interaction_quota_before"] == report["interaction_quota_after"] == counts(12)
    assert asdict(quota.snapshot()) == counts(12)
    assert sampler.bit_generator.state == expected.bit_generator.state


@pytest.mark.parametrize("fail_at", (1, 2))
def test_should_not_charge_a_prediction_failure_as_a_physical_attempt(fail_at):
    quota = PhysicalInteractionQuota(12)
    source = Source(scripted_actions(KeyDoorLayout("A"), 1), fail_at=fail_at)
    sampler = np.random.Generator(np.random.PCG64(29))
    expected = np.random.Generator(np.random.PCG64(29))
    with pytest.raises(training.KeyDoorTrainingFailed) as caught:
        collect(source, sampler, interaction_quota=quota)
    report = caught.value.report
    paid = fail_at - 1
    for _ in range(paid):
        expected.random()
    assert report["prediction_calls_attempted"] == fail_at
    assert report["prediction_calls_returned"] == report["sampling_draws"] == paid
    assert report["environment_steps_attempted"] == report["environment_steps"] == paid
    assert report["interaction_quota_before"] == counts(12)
    assert report["interaction_quota_after"] == asdict(quota.snapshot()) == counts(12, paid, paid)
    assert sampler.bit_generator.state == report["sampler_after"] == expected.bit_generator.state
    assert report["episode"] is None and "frozen prediction fault" in str(caught.value.__cause__)


def test_should_charge_before_failed_physics_and_latch_the_original_quota(monkeypatch):
    quota = PhysicalInteractionQuota(12)
    source = Source((0,))
    sampler = np.random.Generator(np.random.PCG64(29))
    expected = np.random.Generator(np.random.PCG64(29))
    expected.random()
    primary = RuntimeError("paid physical fault")
    observed = []

    def fail(environment, action):
        observed.append(asdict(quota.snapshot()))
        assert action == 0
        raise primary

    monkeypatch.setattr(KeyDoorNavigation, "step", fail)
    with pytest.raises(training.KeyDoorTrainingFailed) as caught:
        collect(source, sampler, interaction_quota=quota)
    report = caught.value.report
    assert caught.value.__cause__ is primary
    assert observed == [counts(12, 1)]
    assert report["sampling_draws"] == report["environment_steps_attempted"] == 1
    assert report["environment_steps"] == 0 and report["transitions"][0]["applied"] is False
    assert report["interaction_quota_after"] == asdict(quota.snapshot()) == counts(12, 1, 0, 1)
    assert quota.remaining_steps == 11
    assert sampler.bit_generator.state == report["sampler_after"] == expected.bit_generator.state
    attempts_before = source.snapshots
    with pytest.raises(ValueError):
        collect(source, sampler, interaction_quota=quota)
    assert source.snapshots == attempts_before and len(source.sessions) == 1
    assert observed == [counts(12, 1)]


@pytest.mark.parametrize(
    "where", ("source_snapshot", "start_episode", "prediction", "before_prediction")
)
def test_should_refuse_unobserved_callback_quota_changes_without_rolling_back_charges(where):
    quota = PhysicalInteractionQuota(12)
    source = Source((0,))
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)

    def mutate_quota():
        quota.begin_step()
        quota.complete_step()

    options = {}
    if where == "source_snapshot":
        original = source.snapshot_state

        def changed_snapshot():
            state = original()
            mutate_quota()
            return state

        setattr(source, "snapshot_state", changed_snapshot)
    elif where in ("start_episode", "prediction"):
        original_start = source.start_episode

        def changed_start():
            session = original_start()
            if where == "start_episode":
                mutate_quota()
            else:
                original_predict = session.predict

                def changed_predict(features):
                    result = original_predict(features)
                    mutate_quota()
                    return result

                session.predict = changed_predict
            return session

        setattr(source, "start_episode", changed_start)
    else:
        options["before_prediction"] = mutate_quota

    with pytest.raises(training.KeyDoorTrainingFailed) as caught:
        collect(source, sampler, interaction_quota=quota, **options)
    report = caught.value.report
    assert report["status"] == "failed" and report["episode"] is None
    assert report["interaction_quota_before"] == counts(12)
    assert report["interaction_quota_after"] == asdict(quota.snapshot()) == counts(12, 1, 1)
    expected_predictions = int(where == "prediction")
    assert (
        report["prediction_calls_attempted"]
        == report["prediction_calls_returned"]
        == expected_predictions
    )
    assert (
        report["sampling_draws"]
        == report["environment_steps_attempted"]
        == report["environment_steps"]
        == 0
    )
    assert sampler.bit_generator.state == before


def test_should_preserve_default_collection_report_rng_and_call_behavior_when_quota_is_disabled():
    actions = scripted_actions(KeyDoorLayout("A"), 1)
    implicit = collect(Source(actions))
    explicit = collect(Source(actions), interaction_quota=None)
    assert pickle.dumps(implicit) == pickle.dumps(explicit)
    assert "interaction_quota_before" not in implicit and "interaction_quota_after" not in implicit


def test_should_leave_corrupted_callback_readback_unknown_without_fabricating_quota_counts():
    quota = PhysicalInteractionQuota(12)
    source = Source((0,))
    sampler = np.random.Generator(np.random.PCG64(29))
    before = deepcopy(sampler.bit_generator.state)

    def corrupt_quota():
        object.__setattr__(quota, "_completed_steps", 99)

    with pytest.raises(training.KeyDoorTrainingFailed) as caught:
        collect(source, sampler, interaction_quota=quota, before_prediction=corrupt_quota)
    report = caught.value.report
    assert report["status"] == "failed" and report["episode"] is None
    assert report["interaction_quota_before"] == counts(12)
    assert report["interaction_quota_after"] is None
    assert report["interaction_quota_readback_error"]
    assert report["prediction_calls_attempted"] == report["sampling_draws"] == 0
    assert report["environment_steps_attempted"] == report["environment_steps"] == 0
    assert sampler.bit_generator.state == report["sampler_after"] == before
    with pytest.raises(ValueError):
        quota.snapshot()


def test_should_keep_independent_policy_quotas_sessions_and_private_rng_progress():
    layout = KeyDoorLayout("A")
    sources = (Source(scripted_actions(layout, 1)), Source((0, 0)))
    quotas = (PhysicalInteractionQuota(12), PhysicalInteractionQuota(12))
    samplers = tuple(np.random.Generator(np.random.PCG64(seed)) for seed in (29, 31))
    untouched = deepcopy(samplers[1].bit_generator.state)
    first = collect(sources[0], samplers[0], interaction_quota=quotas[0])
    assert first["environment_steps"] == 8
    assert asdict(quotas[1].snapshot()) == counts(12)
    assert sources[1].sessions == [] and samplers[1].bit_generator.state == untouched
    first_snapshot, first_rng = quotas[0].snapshot(), deepcopy(samplers[0].bit_generator.state)
    second = collect(sources[1], samplers[1], interaction_quota=quotas[1], max_steps=2)
    assert second["environment_steps"] == 2
    assert quotas[0].snapshot() == first_snapshot and samplers[0].bit_generator.state == first_rng
    assert asdict(quotas[0].snapshot()) == counts(12, 8, 8)
    assert asdict(quotas[1].snapshot()) == counts(12, 2, 2)
    assert sources[0].sessions[0] is not sources[1].sessions[0]
