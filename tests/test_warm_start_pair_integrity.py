"""Paired comparison refuses isolation changes and dependent final work."""

import pytest

from src.adapters import key_door_reward_pilot as pilot
from test_warm_start_control_pilot import Learner, RewardLearner, environment


@pytest.mark.parametrize("changed", ["source", "actor"])
def test_should_refuse_observed_isolation_change_before_admitting_the_other_arm(
    monkeypatch, changed
):
    environment(monkeypatch)
    original = Learner.snapshot_state

    def snapshot(self):
        result = original(self)
        if Learner.commits >= 8 and self.arm_number == 1 and self.state.completed_updates == 0:
            actual_source = self is RewardLearner.sources[0]
            if actual_source == (changed == "source"):
                result.parameters["weight_input"][0, 0] += 0.5
        return result

    monkeypatch.setattr(Learner, "snapshot_state", snapshot)
    report = pilot.run_warm_start_comparison_pilot()
    bc, rl = report["runs"]
    assert report["status"] == "incomplete" and report["comparison"] is None
    assert bc["status"] == "failed" and "isolation changed" in bc["reason"]
    assert bc[f"{changed}_unchanged"] is False
    assert bc["wake_updates"] == Learner.commits == 8
    assert rl["status"] == "not_run" and rl["wake_updates"] == 0
    assert len(RewardLearner.sources) == 1
    assert RewardLearner.classifications == RewardLearner.sessions == []
    assert all(e["status"] == "not_run" for row in report["runs"] for e in row["episodes"])


def test_should_preserve_first_final_failure_without_admitting_other_arm_evaluation(monkeypatch):
    environment(monkeypatch)

    def fail(self, features):
        assert Learner.commits == 24
        raise RuntimeError("first paired final read failed")

    monkeypatch.setattr(Learner, "predict", fail)
    report = pilot.run_warm_start_comparison_pilot()
    bc, rl = report["runs"]
    assert report["status"] == "incomplete" and report["comparison"] is None
    assert bc["status"] == "failed" and "first paired final" in bc["reason"]
    assert rl["status"] == "not_evaluated" and "preceding paired final" in rl["reason"]
    assert bc["wake_updates"] == 8 and rl["wake_updates"] == 16
    assert Learner.commits == 24
    assert rl["stream_result"] is None
    assert all(e["status"] == "not_run" for row in report["runs"] for e in row["episodes"])
