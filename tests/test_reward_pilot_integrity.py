"""Refuse changed admitted evidence before collecting another on-policy task."""

from copy import deepcopy

import pytest

import src.adapters.key_door_reward_pilot as pilot
from test_key_door_reward_pilot import Learner, environment


@pytest.mark.parametrize("fault", ["receipt", "clock", "sampler"])
def test_should_refuse_replaced_prior_evidence_or_unobserved_progress_before_more_work(
    monkeypatch, fault
):
    environment(monkeypatch)
    slots = pilot._training_slots()
    monkeypatch.setattr(pilot, "_training_slots", lambda: slots)
    budgets, samplers = [], []
    original_session = pilot.shared._session
    original_collect = pilot.collect_key_door_reward_episode
    original_observation = pilot.shared._recurrent_observation

    def session(*args, **kwargs):
        value = original_session(*args, **kwargs)
        budgets.append(value)
        return value

    def collect(*args, **kwargs):
        samplers.append(kwargs["sampler"])
        return original_collect(*args, **kwargs)

    def observation(point, state):
        result = original_observation(point, state)
        if point == 2:
            if fault == "receipt":
                slots[0]["receipt"] = dict(slots[0]["receipt"], update_number=99)
            elif fault == "clock":
                budget = budgets[0]
                previous = budget.last_clock
                budget.last_clock = (budget.started_at + previous) / 2
                assert budget.started_at <= budget.last_clock < previous
            else:
                samplers[0].random()
        return result

    monkeypatch.setattr(pilot.shared, "_session", session)
    monkeypatch.setattr(pilot, "collect_key_door_reward_episode", collect)
    monkeypatch.setattr(pilot.shared, "_recurrent_observation", observation)
    result = pilot.run_pilot()
    row = result["runs"][0]
    assert result["status"] == "incomplete" and row["status"] == "failed"
    assert row["wake_updates"] == 2 and len(Learner.updates) == len(samplers) == 2
    assert row["stream_result"] is None and Learner.classifications == []
    assert all(s["status"] == "not_run" for s in row["training"][2:])
    assert all(e["status"] == "not_run" for e in row["episodes"])
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert row["sampling_state"] == deepcopy(samplers[0].bit_generator.state)
