"""Real stream/rehearsal orchestration with a small owned fake port; no Torch work."""

from copy import deepcopy
from dataclasses import dataclass
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.torch_feature_rehearsal_pilot as pilot
import src.app.toy_execution_budget as budgets
from src.core.learner_ports import TrainingDiagnostic
from src.shared.process_memory import ProcessRssSegment


def forbidden(*args, **kwargs):
    pytest.fail("unselected native/resource operation reached")


class Sampler:
    start_bytes = 100

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self):
        return 100

    def snapshot(self):
        return ProcessRssSegment(1, 100, 100, 1, 0.01)


class TorchStub:
    @staticmethod
    def set_num_threads(count):
        assert count == 1

    @staticmethod
    def get_num_interop_threads():
        return 1


@dataclass
class State:
    head_state: dict[str, Any]
    completed_updates: int = 0
    training_examples: int = 0
    optimizer_state: Any = None
    gradients: Any = None


class Learner:
    def __init__(self, policy):
        self.policy = policy
        self.state = State({key: np.zeros((1, 1)) for key in pilot.PARAMETERS})

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def fork(self):
        return deepcopy(self)

    def train_batch(self, features, targets):
        self.state.completed_updates += 1
        self.state.training_examples += len(features)
        for tensor in self.state.head_state.values():
            tensor += float(np.mean(features)) * 0.01
        return TrainingDiagnostic("fake_native_energy_v1", 0.5)

    def predict(self, features):
        return np.full((len(features), 1), 0.5, dtype=np.float64)


def environment(monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "TorchFeatureLearner", Learner)
    monkeypatch.setattr(pilot, "_tensor_bytes", lambda state: 0)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))


def test_should_describe_without_torch_models_data_or_resource_work(monkeypatch):
    for name in ("require_torch", "TorchFeatureLearner", "ProcessRssSampler", "build_stream"):
        monkeypatch.setattr(pilot, name, forbidden)
    description = pilot.describe_protocol()
    assert description["seed"] == 17 and description["hidden_dim"] == 3
    assert description["planned_work"]["native_train_calls"] == 60
    assert description["resources"]["process_rss_bytes"] == 768 * 1024**2
    assert description["replay"]["retention"] == dict(max_examples=4, max_bytes=96)


def test_should_build_fixed_disjoint_training_and_final_roles_without_torch(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    training, final = pilot.build_stream()
    assert [row.context for row in training] == ["A", "A", "B", "B", "A", "A"]
    ids = [identity for row in (*training, *final) for identity in row.row_ids]
    assert len(ids) == len(set(ids)) == 28
    for row in (*training, *final):
        axis = 1 if row.context == "B" else 0
        np.testing.assert_array_equal(
            row.targets, (row.features[:, [axis]] >= 0).astype(np.float64)
        )
    for train in training:
        assert train.role == "train" and train.features.shape == (2, 2)
        for heldout in final:
            assert heldout.role == "final_test" and heldout.features.shape == (8, 2)
            assert not np.shares_memory(train.features, heldout.features)


def test_should_reconstruct_complete_stream_payloads_without_repeating_observers_or_training(
    monkeypatch,
):
    environment(monkeypatch)
    report = pilot.run_continuation_pilot()
    assert report["status"] == "complete", report["runs"]
    assert report["completed_resumes"] == 16 and report["complete_owned_reconstruction"] is True
    assert report["native_train_calls"] == 60 and report["final_score_forwards"] == 64
    assert report["protocol"]["pause_updates"] == [2, 4]
    for row in report["runs"]:
        assert row["pause_attempts"] == 2 and len(row["continuation"]) == 2
        assert [point["wake_updates"] for point in row["observations"]] == [0, 2, 4, 6]
        assert [item["pending_sample_ids"] for item in row["continuation"]] == [
            ["batch-2"],
            ["batch-4"],
        ]


@pytest.mark.parametrize("fault", ["counter", "limit"])
def test_should_refuse_complete_result_when_scorer_changes_original_budget(monkeypatch, fault):
    environment(monkeypatch)
    budget = pilot.ToyBudgetSession(pilot.ToyExecutionBudget(max_training_updates=6), lambda: 0.0)
    source = Learner(pilot.TorchFeaturePolicy("pc", 2, 3, 17))
    training, final = pilot.build_stream()
    stream = pilot.AttributionStream(
        source, training, final, checkpoint_updates=pilot.CHECKPOINTS, label_delay=1, budget=budget
    )
    stream.run_until(6)

    def predict(self, features):
        if fault == "counter":
            budget.updates_completed = 0
        else:
            object.__setattr__(budget.budget, "max_training_updates", 7)
        return np.full((len(features), 1), 0.5)

    monkeypatch.setattr(Learner, "predict", predict)
    with pytest.raises(ValueError):
        stream.finish()
    assert stream.completed_updates == len(stream._runtime.applied_updates) == 6
    with pytest.raises(ValueError):
        stream.run_until(6)


def test_should_execute_existing_runner_and_rehearsal_with_separate_clocks_and_owned_actor(
    monkeypatch,
):
    environment(monkeypatch)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["runs"]
    assert report["native_train_calls"] == 60 and report["training_row_exposures"] == 108
    assert report["final_score_forwards"] == 64 and report["wake_updates"] == 48
    assert report["replay_charged_examples"] == report["replay_committed_updates"] == 12
    assert report["pc_neutral_exact"] and report["shared_replay_plans_exact"]
    assert report["native_clocks_exact"] and report["replay_bounds_and_receipts_exact"]
    assert not report["scientific_advantage_verified"] and not report["full_R24_complete"]
    for row in report["runs"]:
        assert row["source_learner_unchanged"] is True
        assert row["outcomes"] == dict(new_B_gain=0.0, retention_loss=0.0, return_gain=0.0)
        if row["replay_enabled"]:
            assert row["deployed_actor_unchanged"] is True
            assert [item["updates"] for item in row["rehearsal"]["events"]] == [2, 4, 6]
        else:
            assert (
                row["rehearsal"]
                is row["final_retention"]
                is row["deployed_actor_unchanged"]
                is None
            )


@pytest.mark.parametrize("failure", ["train", "score"])
def test_should_preserve_failed_replay_charge_and_unknown_native_work(monkeypatch, failure):
    environment(monkeypatch)
    original = Learner.train_batch

    def train(self, features, targets):
        if failure == "train" and len(features) == 1:
            raise RuntimeError("permanent replay fault")
        return original(self, features, targets)

    def score(self, features):
        raise RuntimeError("final score fault")

    monkeypatch.setattr(Learner, "train_batch", train)
    if failure == "score":
        monkeypatch.setattr(Learner, "predict", score)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["native_train_calls"] is None
    on = [row for row in report["runs"] if row["replay_enabled"]]
    assert all(row["status"] == "failed" and row["final_retention"] is None for row in on)
    assert all(row["source_learner_unchanged"] is True for row in on)
    assert all(row["replay_charged_examples"] == (1 if failure == "train" else 3) for row in on)
    if failure == "train":
        assert report["replay_attempted_updates"] == 4 and report["replay_committed_updates"] == 0
        assert all(row["rehearsal"]["events"][0]["receipt"] is None for row in on)


@pytest.mark.parametrize("phase", ["sampler", "enter", "snapshot", "runtime", "data"])
def test_should_preserve_all_unreached_arms_on_setup_failure(monkeypatch, phase):
    environment(monkeypatch)

    def broken(*args):
        raise RuntimeError("preserved setup failure")

    if phase in ("enter", "snapshot"):
        monkeypatch.setattr(Sampler, "__enter__" if phase == "enter" else "snapshot", broken)
        if phase == "snapshot":
            monkeypatch.setattr(pilot, "build_stream", broken)
    else:
        monkeypatch.setattr(
            pilot,
            {"sampler": "ProcessRssSampler", "runtime": "require_torch", "data": "build_stream"}[
                phase
            ],
            broken,
        )
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and len(report["runs"]) == 8
    assert all(row["status"] == "not_run" for row in report["runs"])
    assert report["native_train_calls"] is report["final_score_forwards"] is None
    assert "preserved setup failure" in report["global_errors"][0]


@pytest.mark.parametrize(
    "scenario",
    ["original", "renewal-learnability", "torch-feature-rehearsal", "torch-feature-continuation"],
)
@pytest.mark.parametrize("describe", [False, True])
def test_should_preserve_old_cli_routes_and_select_only_requested_callback(
    monkeypatch, capsys, scenario, describe
):
    prefix = {
        "original": "",
        "renewal-learnability": "renewal_",
        "torch-feature-rehearsal": "torch_",
        "torch-feature-continuation": "continuation_",
    }[scenario]
    selected = "describe_" + prefix + "protocol" if describe else "run_" + prefix + "pilot"
    for name in (
        "describe_protocol",
        "run_pilot",
        "describe_renewal_protocol",
        "run_renewal_pilot",
        "describe_torch_protocol",
        "run_torch_pilot",
        "describe_continuation_protocol",
        "run_continuation_pilot",
    ):
        monkeypatch.setattr(
            cli, name, (lambda: {"status": "complete"}) if name == selected else forbidden
        )
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", scenario] + (["--describe"] if describe else [])
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_should_exit_incomplete_once_and_preserve_failure(monkeypatch, capsys):
    calls = []

    def failed():
        calls.append(1)
        return dict(status="incomplete", reason="honest negative fault")

    monkeypatch.setattr(cli, "run_torch_pilot", failed)
    monkeypatch.setattr(sys, "argv", ["pilot", "--scenario", "torch-feature-rehearsal"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert (
        error.value.code == 2
        and calls == [1]
        and "honest negative fault" in capsys.readouterr().out
    )


@pytest.mark.parametrize("flag", ["--seed", "--recent-only", "--diagnose-learnability"])
def test_should_refuse_unfrozen_cli_knobs_before_work(monkeypatch, flag):
    monkeypatch.setattr(cli, "run_torch_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--scenario", "torch-feature-rehearsal", flag])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
