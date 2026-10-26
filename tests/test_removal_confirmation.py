"""Fixed independent-data removal pairs through the real stream, fake learners."""

from copy import deepcopy
import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_stream_rehearsal_pilot as cli
import src.adapters.stream_rehearsal_pilot as rehearsal
import src.adapters.stream_removal_confirmation as confirmation
import src.app.toy_execution_budget as budget_module
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.learner_ports import TrainingDiagnostic
from test_stream_rehearsal_pilot import FakeModel, FakeSampler


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
ORDER = tuple((policy, method) for policy in ("none", "rotating") for method in METHODS)


def forbidden(*args, **kwargs):
    pytest.fail("unselected or native work reached")


class FakeLearner:
    def __init__(self, control, index, state=0.0):
        self.control, self.index, self.state = control, index, state

    def fork(self):
        return FakeLearner(self.control, self.index, self.state)

    def snapshot_state(self):
        return self.state

    def restore_state(self, state):
        self.state = state

    def train_batch(self, features, targets):
        trace = self.control["traces"][self.index]
        trace.append(("train", len(features), features.copy(), targets.copy()))
        if self.index == 0 and self.control.get("fail_wake") and len(trace) == 6:
            raise RuntimeError("declared confirmation wake failure")
        if self.index == 4 and self.control.get("fail_replay") and len(features) == 1:
            raise RuntimeError("declared confirmation replay failure")
        self.state += 1.0 + 0.01 * float(np.mean(features))
        return TrainingDiagnostic("fake_confirmation_native_count", self.state)

    def predict(self, features):
        self.control["traces"][self.index].append(("score", self.state, features.copy()))
        return np.full((len(features), 1), 0.5 + self.state / 1000)


@pytest.fixture
def fake_ports(monkeypatch):
    control: dict[str, Any] = dict(
        traces=[], sources=[], batches=[], bindings=[], models=[], sessions=[]
    )
    now = [0.0]
    clock = lambda: now[0]
    original_data = rehearsal.build_stream
    original_session = ToyBudgetSession.__init__

    def initialize(self, *args, **kwargs):
        original_session(self, *args, **kwargs)
        control["sessions"].append(self)

    def data(stream, **kwargs):
        result = original_data(stream, **kwargs)
        control["batches"].append((stream, kwargs, deepcopy(result)))
        return result

    def build(method, initial):
        index = len(control["sources"])
        control["traces"].append([])
        source = FakeLearner(control, index)
        control["sources"].append(source)
        control["bindings"].append((method, deepcopy(initial)))
        return source

    class Model(FakeModel):
        def __init__(self, *args):
            super().__init__(*args)
            control["models"].append(args)

    monkeypatch.setattr(confirmation, "CoupledPredictiveCodingNetwork", Model)
    monkeypatch.setattr(confirmation, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(rehearsal, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(confirmation, "monotonic", clock)
    monkeypatch.setattr(rehearsal, "monotonic", clock)
    monkeypatch.setattr(rehearsal, "build_stream", data)
    monkeypatch.setattr(rehearsal, "build_learner", build)
    monkeypatch.setattr(ToyBudgetSession, "__init__", initialize)
    monkeypatch.setattr(rehearsal.learnability, "create_sixteen_probe_observer", forbidden)
    return SimpleNamespace(control=control, now=now, original_data=original_data)


def test_should_describe_only_the_prospective_fixed_family_without_native_construction(monkeypatch):
    for name in ("CoupledPredictiveCodingNetwork", "ProcessRssSampler"):
        monkeypatch.setattr(confirmation, name, forbidden)
    monkeypatch.setattr(rehearsal, "_run_one", forbidden)
    descriptor = confirmation.describe_protocol()
    assert descriptor["seeds"] == [23]
    assert descriptor["methods"] == list(METHODS)
    assert descriptor["policies"] == ["none", "rotating"]
    assert descriptor["training_rng"] == 7302 and descriptor["final_rng"] == 9102
    assert descriptor["learning_rate"] == 0.03
    assert descriptor["inference_steps"] == 4 and descriptor["inference_learning_rate"] == 0.2
    assert descriptor["precisions"] == [1.0, 1.0]
    assert descriptor["hidden_dims"] == [8, 8]
    assert descriptor["checkpoint_updates"] == [0, 6, 12, 18]
    assert descriptor["label_delay"] == 2
    assert descriptor["resources"]["training_updates"] == 192
    assert descriptor["resources"]["replay_bytes"] == 384


def test_should_execute_all_eight_rows_once_with_fixed_new_data_and_original_initial_law(
    fake_ports,
):
    report = confirmation.run_pilot()
    assert report["status"] == "complete", report["global_errors"]
    rows = report["runs"]
    assert [(row["policy"], row["method"]) for row in rows] == list(ORDER)
    assert all(row["seed"] == 23 and row["source_learner_unchanged"] for row in rows)
    assert fake_ports.control["models"] == [(2, (8, 8), 23)]
    assert len(fake_ports.control["batches"]) == len(fake_ports.control["bindings"]) == 8
    initial_law = fake_ports.control["bindings"][0][1].weights
    for _, initial in fake_ports.control["bindings"]:
        np.testing.assert_array_equal(initial.weights, initial_law)
    for row, (stream, kwargs, batches), trace in zip(
        rows, fake_ports.control["batches"], fake_ports.control["traces"], strict=True
    ):
        assert stream == "recurring" and kwargs == dict(training_rng=7302, heldout_rng=9102)
        expected = fake_ports.original_data("recurring", training_rng=7302, heldout_rng=9102)
        for actual, oracle in zip(
            (*batches[0], *batches[1]), (*expected[0], *expected[1]), strict=True
        ):
            np.testing.assert_array_equal(actual.features, oracle.features)
            np.testing.assert_array_equal(actual.targets, oracle.targets)
            assert actual.role == oracle.role and actual.row_ids == oracle.row_ids
        training = [event for event in trace if event[0] == "train"]
        expected_calls = 18 if row["policy"] == "none" else 30
        assert len(training) == expected_calls
        assert [event[0] for event in trace] == ["train"] * expected_calls + ["score"] * 8
        assert sum(event[1] for event in training) == (144 if row["policy"] == "none" else 156)
        assert row["wake_updates"] == 18
        assert len(row["stream_result"]["applied_updates"]) == 18
        assert row["stream_result"]["score_forward_calls"] == 8
        if row["policy"] == "none":
            assert row["rehearsal"] is None and row["replay_charged_examples"] == 0
            assert row["planned_event_updates"] == []
        else:
            assert row["replay_charged_examples"] == 12
            assert (
                row["rehearsal"]["returned_updates"] == row["rehearsal"]["committed_updates"] == 12
            )
            assert [event["updates"] for event in row["rehearsal"]["events"]] == [
                3,
                6,
                9,
                12,
                15,
                18,
            ]
    assert all(source.state == 0.0 for source in fake_ports.control["sources"])
    assert report["wake_updates"] == 144
    assert (
        report["replay_charged_examples"]
        == report["replay_returned_updates"]
        == report["replay_committed_updates"]
        == 48
    )
    assert report["known_native_train_calls"] == report["native_train_calls"] == 192
    assert report["row_exposures_charged"] == 1200 and report["final_score_forwards"] == 64
    assert report["scientific_advantage_verified"] is report["full_R28_complete"] is False


def test_should_keep_recorded_levels_all_signed_pair_differences_and_separate_policy_parities(
    fake_ports,
):
    report = confirmation.run_pilot()
    assert report["status"] == "complete", report["global_errors"]
    assert len(report["pairs"]) == 4 and len(report["group_facts"]) == 2
    groups = {group["policy"]: group for group in report["group_facts"]}
    assert groups["none"]["pc_neutral_exact"] is groups["rotating"]["pc_neutral_exact"] is True
    assert groups["none"]["shared_replay_plans_exact"] is None
    assert groups["rotating"]["shared_replay_plans_exact"] is True
    for pair in report["pairs"]:
        assert pair["status"] == "complete" and pair["initial_primary_levels_exact"] is True
        assert pair["native_update_difference"] == pair["training_row_difference"] == 12
        both = {
            policy: next(
                row
                for row in report["runs"]
                if row["policy"] == policy and row["method"] == pair["method"]
            )
            for policy in ("none", "rotating")
        }
        for policy, row in both.items():
            levels = row["stream_result"]["checkpoints"]
            assert pair["primary_levels"][policy] == levels
            assert [point["updates"] for point in levels] == [0, 6, 12, 18]
            points = {point["updates"]: point["scores"] for point in levels}
            expected = dict(
                new_B_gain=points[0]["B"]["bce"] - points[12]["B"]["bce"],
                retention_loss=points[12]["A"]["bce"] - points[6]["A"]["bce"],
                return_gain=points[12]["A"]["bce"] - points[18]["A"]["bce"],
            )
            assert pair["outcomes"][policy] == expected
        assert pair["rotating_minus_none"] == {
            key: pair["outcomes"]["rotating"][key] - pair["outcomes"]["none"][key]
            for key in ("new_B_gain", "retention_loss", "return_gain")
        }
        assert (
            both["none"]["stream_result"]["checkpoints"][0]["scores"]
            == both["rotating"]["stream_result"]["checkpoints"][0]["scores"]
        )
        assert pair["wake_receipt_timing_exact"] is True
        timing = {
            policy: [
                {key: value for key, value in receipt.items() if key != "diagnostic"}
                for receipt in row["stream_result"]["applied_updates"]
            ]
            for policy, row in both.items()
        }
        assert timing["none"] == timing["rotating"]


def test_should_bind_every_arm_to_the_original_clock_deadline_and_sampler(fake_ports):
    report = confirmation.run_pilot()
    assert report["status"] == "complete", report["global_errors"]
    sessions = fake_ports.control["sessions"]
    assert len(sessions) == 12  # Eight wake sessions plus four rotating replay allowances.
    assert len({id(session) for session in sessions}) == 12
    assert all(session.clock is sessions[0].clock for session in sessions)
    assert all(
        session.started_at == 0.0 and session.budget.max_wall_seconds == 60 for session in sessions
    )
    assert all(
        session.process_rss_sampler is sessions[0].process_rss_sampler for session in sessions
    )
    assert sum(session.budget.max_training_updates == 18 for session in sessions) == 8
    assert sum(session.budget.max_replay_examples == 12 for session in sessions) == 4


@pytest.mark.parametrize(
    "fault,updates,exposures", [("fail_wake", 131, 1096), ("fail_replay", 129, 1069)]
)
def test_should_preserve_paid_failures_without_scoring_the_failed_arm_or_dropping_its_pair(
    fake_ports, fault, updates, exposures
):
    fake_ports.control[fault] = True
    report = confirmation.run_pilot()
    assert report["status"] == "incomplete" and len(report["runs"]) == 8
    index = 0 if fault == "fail_wake" else 4
    row = report["runs"][index]
    assert row["status"] == "failed" and "confirmation" in row["reason"]
    assert all(event[0] != "score" for event in fake_ports.control["traces"][index])
    assert report["wake_updates"] == updates and report["row_exposures_charged"] == exposures
    assert report["native_train_calls"] is report["final_score_forwards"] is None
    assert report["known_completed_score_forwards"] == 56
    assert all(source.state == 0 for source in fake_ports.control["sources"])
    pair = next(pair for pair in report["pairs"] if pair["method"] == "bp")
    assert pair["status"] == "incomplete"
    assert pair["primary_levels"] is pair["outcomes"] is pair["rotating_minus_none"] is None
    assert pair["initial_primary_levels_exact"] is None
    assert all(pair["status"] == "complete" for pair in report["pairs"] if pair["method"] != "bp")
    if fault == "fail_replay":
        assert row["wake_updates"] == 3 and row["replay_charged_examples"] == 1
        assert row["rehearsal"]["attempted_updates"] == 1
        assert row["rehearsal"]["returned_updates"] == row["rehearsal"]["committed_updates"] == 0


def test_should_keep_all_slots_and_null_pairs_when_global_setup_fails(monkeypatch):
    def fail():
        raise RuntimeError("declared confirmation setup failure")

    monkeypatch.setattr(confirmation, "ProcessRssSampler", fail)
    monkeypatch.setattr(confirmation, "CoupledPredictiveCodingNetwork", forbidden)
    report = confirmation.run_pilot()
    assert report["status"] == "incomplete" and report["global_errors"]
    assert [(row["policy"], row["method"]) for row in report["runs"]] == list(ORDER)
    assert all(row["status"] == "not_run" for row in report["runs"])
    assert len(report["pairs"]) == 4
    assert all(
        pair["status"] == "incomplete" and pair["rotating_minus_none"] is None
        for pair in report["pairs"]
    )
    assert report["wake_updates"] == report["replay_charged_examples"] == 0
    assert report["native_train_calls"] is None


def test_should_not_start_more_arms_or_renew_global_time_when_setup_uses_the_deadline(
    fake_ports, monkeypatch
):
    class ExpiredModel(FakeModel):
        def __init__(self, *args):
            super().__init__(*args)
            fake_ports.now[0] = 61.0

    monkeypatch.setattr(confirmation, "CoupledPredictiveCodingNetwork", ExpiredModel)
    report = confirmation.run_pilot()
    assert report["status"] == "incomplete"
    assert len(report["runs"]) == 8 and all(row["status"] == "not_run" for row in report["runs"])
    assert fake_ports.control["sources"] == []
    assert report["wake_updates"] == 0
    assert all(pair["rotating_minus_none"] is None for pair in report["pairs"])


@pytest.mark.parametrize("invalid", [None, 0, 1, np.bool_(True)])
def test_should_refuse_non_boolean_confirmation_before_allocating_a_budget(invalid, monkeypatch):
    monkeypatch.setattr(rehearsal, "ToyBudgetSession", forbidden)
    with pytest.raises(ValueError, match="bool"):
        rehearsal._run_one({}, None, 60.0, FakeSampler(), confirmation=invalid)


@pytest.mark.parametrize(
    "extra",
    [
        {"diagnose_learnability": True},
        {"recent_only": True},
        {"track_unit_history": True},
    ],
)
def test_should_refuse_combined_confirmation_presets_before_budget_or_model_work(
    extra, monkeypatch
):
    monkeypatch.setattr(rehearsal, "ToyBudgetSession", forbidden)
    monkeypatch.setattr(rehearsal, "build_learner", forbidden)
    with pytest.raises(ValueError):
        rehearsal._run_one({}, None, 60.0, FakeSampler(), confirmation=True, **extra)


@pytest.mark.parametrize("action", ["describe", "run"])
def test_should_route_only_the_fixed_no_argument_removal_confirmation_entrypoint(
    action, monkeypatch, capsys
):
    calls = []

    def selected():
        calls.append(action)
        return {"status": "complete", "selected": action}

    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(
        cli, "describe_removal_confirmation", selected if action == "describe" else forbidden
    )
    monkeypatch.setattr(cli, "run_removal_confirmation", selected if action == "run" else forbidden)
    monkeypatch.setattr(
        sys, "argv", ["rehearsal", f"--{action}", "--scenario", "removal-confirmation"]
    )
    assert cli.main() == 0 and calls == [action]
    assert json.loads(capsys.readouterr().out)["selected"] == action


@pytest.mark.parametrize("flag", ["--recent-only", "--no-replay", "--diagnose-learnability"])
def test_should_refuse_single_mode_flags_before_the_confirmation_scenario_dispatch(
    flag, monkeypatch
):
    monkeypatch.setattr(cli, "describe_removal_confirmation", forbidden)
    monkeypatch.setattr(cli, "run_removal_confirmation", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["rehearsal", "--describe", "--scenario", "removal-confirmation", flag]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "diagnose,recent,no_replay",
    [(False, False, False), (True, False, False), (False, True, False), (False, False, True)],
)
def test_should_preserve_old_single_scenario_keyword_shapes(
    diagnose, recent, no_replay, monkeypatch, capsys
):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "describe_protocol", selected)
    monkeypatch.setattr(cli, "describe_removal_confirmation", forbidden)
    arguments = ["rehearsal", "--describe"]
    arguments += ["--diagnose-learnability"] if diagnose else []
    arguments += ["--recent-only"] if recent else []
    arguments += ["--no-replay"] if no_replay else []
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 0 and capsys.readouterr().out
    expected = (
        {"no_replay": True}
        if no_replay
        else {"recent_only": True, "diagnose_learnability": diagnose}
        if recent
        else {"diagnose_learnability": True}
        if diagnose
        else {}
    )
    assert calls == [expected]
