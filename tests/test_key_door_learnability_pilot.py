"""Actual reserved game packets probe owned checkpoint forks with fake learners."""

from copy import deepcopy
import json
import pickle
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as pilot
from src.core.demonstration_episode import EpisodeReplaySchema, encode_episode
from test_recurrent_key_door_pilot import Learner, fake_environment
from test_torch_feature_rehearsal_pilot import forbidden


def test_should_describe_fixed_probe_work_without_collecting_or_loading_resources(monkeypatch):
    for name in (
        "build_stream",
        "build_probe_stream",
        "KeyDoorNavigation",
        "TorchRecurrentLearner",
        "require_torch",
        "ProcessRssSampler",
        "run_fresh_task_probe",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    protocol = pilot.describe_learnability_protocol()
    assert protocol["methods"] == ["recurrent_bc"]
    assert protocol["planned_work"]["native_train_calls"] == 84
    assert protocol["planned_work"]["training_rows"] == 672
    assert protocol["planned_work"]["live_episodes"] == 80
    assert protocol["planned_work"]["maximum_recurrent_unroll_steps"] == 2528
    assert protocol["seed"] == 17 and protocol["num_classes"] == 5
    assert pilot.describe_protocol()["planned_work"]["native_train_calls"] == 80
    assert pilot.describe_recurrent_protocol()["planned_work"]["native_train_calls"] == 20


def test_should_collect_reserved_complete_eighteen_E_packets_disjoint_from_benchmark_roles(
    monkeypatch,
):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    training, evaluation = pilot.build_probe_stream()
    benchmark_training, benchmark_final = pilot.build_stream()
    assert len(training) == 16 and len(evaluation) == 2
    identities = []
    for index, batch in enumerate((*training, *evaluation)):
        assert batch.episode is not None and batch.num_classes == 5
        assert batch.role == ("probe_train" if index < 16 else "probe_test")
        assert batch.episode.role == batch.role
        assert batch.features.shape == (8, 8) and batch.targets.shape == (8, 1)
        assert np.all(batch.features[:, 7] == 0.75)
        assert batch.features[0, 6] == (1 if index % 2 == 0 else -1)
        assert np.all(batch.features[1:, 6] == 0)
        assert sum(step.reward for step in batch.episode.transitions) == 1
        assert [step.terminal for step in batch.episode.transitions] == [False] * 7 + [True]
        identities.append(batch.episode.episode_id)
        with pytest.raises(ValueError):
            encode_episode(batch.episode, EpisodeReplaySchema(8, 8, 5))
        for benchmark in (*benchmark_training, *benchmark_final):
            assert not set(batch.row_ids) & set(benchmark.row_ids)
            assert not np.shares_memory(batch.features, benchmark.features)
            assert not np.shares_memory(batch.targets, benchmark.targets)
    assert len(set(identities)) == 18
    assert all(not np.any(batch.features[:, 7] == 0.75) for batch in benchmark_training)


def test_should_run_real_disposable_probe_on_four_complete_checkpoints_without_deployed_changes(
    monkeypatch,
):
    fake_environment(monkeypatch)
    original = pilot.run_fresh_task_probe
    observed = []

    def run_probe(source, state, training, evaluation, **options):
        before = pickle.dumps((source.snapshot_state(), state, training, evaluation))
        budget = options["budget"]
        limits, clock = budget.budget, budget.clock
        assert budget.updates_completed == 0 and limits.max_training_updates == 16
        assert options["checkpoint_updates"] == (0, 4, 8, 16)
        assert len(training) == 16 and len(evaluation) == 2
        assert all(batch.role == "probe_train" for batch in training)
        assert all(batch.role == "probe_test" for batch in evaluation)
        assert all(np.all(batch.features[:, 7] == 0.75) for batch in (*training, *evaluation))
        observed.append(deepcopy(state))
        report = original(source, state, training, evaluation, **options)
        assert report["status"] == "complete" and budget.updates_completed == 16
        assert budget.budget is limits and budget.clock is clock
        assert pickle.dumps((source.snapshot_state(), state, training, evaluation)) == before
        return report

    monkeypatch.setattr(pilot, "run_fresh_task_probe", run_probe)
    result = pilot.run_learnability_pilot()
    assert result["status"] == "complete", result["runs"]
    assert [state.completed_updates for state in observed] == [0, 8, 0, 12]
    assert [state.training_examples for state in observed] == [0, 64, 0, 96]
    assert observed[0].optimizer_state == observed[2].optimizer_state == {}
    assert all(value is None for value in observed[0].gradients.values())
    assert observed[1].optimizer_state and observed[3].optimizer_state
    assert len(result["probe_demonstrations"]) == 18
    assert len(Learner.trains) == 84 and sum(len(features) for features, _ in Learner.trains) == 672
    assert len(Learner.classifications) == 112 and len(Learner.sessions) == 80
    assert sum(np.all(features[:, 7] == 0.75) for features, _ in Learner.trains) == 64
    assert sum(np.all(features[:, 7] == 0.75) for features in Learner.classifications) == 32
    for row in result["runs"]:
        assert row["wake_updates"] == 8
        assert row["replay_charged_examples"] == (32 if row["replay_enabled"] else 0)
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        assert [probe["checkpoint_updates"] for probe in row["probes"]] == [0, 8]
        for probe in row["probes"]:
            assert probe["status"] == "complete" and probe["charged_updates"] == 16
            assert (
                probe["source_unchanged"]
                is probe["state_unchanged"]
                is probe["budget_binding_preserved"]
                is True
            )
            report = probe["report"]
            assert (
                report["completed_updates"] == 16 and report["completed_training_examples"] == 128
            )
            assert report["score_forward_calls_completed"] == 8 and report["num_classes"] == 5
            assert [point["updates"] for point in report["checkpoints"]] == [0, 4, 8, 16]
            assert report["episode_scope"]["training_transitions"] == 128
            assert report["episode_scope"]["evaluation_transitions"] == 16
            assert probe["outcomes"]["ce_gain"] == 0
            assert probe["outcomes"]["initial_ce"] == pytest.approx(
                -(np.log(0.6) + 7 * np.log(0.1)) / 8
            )
            assert probe["outcomes"]["final_ce"] == probe["outcomes"]["initial_ce"]
            assert (
                probe["outcomes"]["initial_accuracy"]
                == probe["outcomes"]["final_accuracy"]
                == 1 / 8
            )
        assert all(episode["outcome"] == "illegal_action" for episode in row["episodes"])
    assert result["scientific_advantage_verified"] is result["full_R42_complete"] is False


@pytest.mark.parametrize("failed_checkpoint", [0, 8])
def test_should_preserve_failed_probe_commits_and_unreached_work_without_renewing_budgets(
    monkeypatch, failed_checkpoint
):
    fake_environment(monkeypatch)
    original, train = pilot.run_fresh_task_probe, Learner.train_batch

    def run_probe(source, state, training, evaluation, **options):
        boundary = 0 if state.completed_updates == 0 else 8
        if boundary != failed_checkpoint:
            return original(source, state, training, evaluation, **options)
        calls: list[None] = []

        def fail(owner, features, targets):
            calls.append(None)
            result = train(owner, features, targets)
            if len(calls) == 4:
                raise RuntimeError("fourth disposable episode failed after mutation")
            return result

        with monkeypatch.context() as context:
            context.setattr(Learner, "train_batch", fail)
            return original(source, state, training, evaluation, **options)

    monkeypatch.setattr(pilot, "run_fresh_task_probe", run_probe)
    result = pilot.run_learnability_pilot()
    assert result["status"] == "incomplete"
    for row in result["runs"]:
        assert row["status"] == "failed"
        assert row["wake_updates"] == failed_checkpoint
        assert row["replay_charged_examples"] == (
            32 if failed_checkpoint == 8 and row["replay_enabled"] else 0
        )
        failed = row["probes"][0 if failed_checkpoint == 0 else 1]
        assert failed["status"] == "failed" and failed["charged_updates"] == 3
        assert "fourth disposable episode failed" in failed["report"]["reason"]
        assert failed["report"]["attempted_updates"] == 4
        assert failed["report"]["completed_updates"] == 3
        assert [point["updates"] for point in failed["report"]["checkpoints"]] == [0]
        assert failed["source_unchanged"] is failed["state_unchanged"] is True
        assert failed["budget_binding_preserved"] is True
        assert all(episode["status"] == "not_run" for episode in row["episodes"])
        if failed_checkpoint == 0:
            assert row["probes"][1]["status"] == "not_run"
        else:
            assert row["probes"][0]["status"] == "complete"


def test_should_preserve_primary_probe_error_when_original_source_readback_also_fails(monkeypatch):
    fake_environment(monkeypatch)

    def readback():
        raise RuntimeError("secondary probe source readback")

    def fail(source, state, training, evaluation, **options):
        monkeypatch.setattr(source, "snapshot_state", readback)
        raise RuntimeError("primary reserved probe error")

    monkeypatch.setattr(pilot, "run_fresh_task_probe", fail)
    result = pilot.run_learnability_pilot()
    assert result["status"] == "incomplete"
    for row in result["runs"]:
        assert "primary reserved probe error" in row["reason"]
        assert "secondary probe source readback" in row["readback_error"]
        assert row["wake_updates"] == row["replay_charged_examples"] == 0
        assert row["probes"][0]["status"] == "failed"
        assert row["probes"][1]["status"] == "not_run"
        assert all(episode["status"] == "not_run" for episode in row["episodes"])


@pytest.mark.parametrize("fault", ["state", "source", "budget", "float_updates", "bool_replay"])
def test_should_refuse_changed_original_bindings_after_probe_without_hiding_completed_charges(
    monkeypatch, fault
):
    fake_environment(monkeypatch)
    original = pilot.run_fresh_task_probe

    def corrupt(source, state, training, evaluation, **options):
        result = original(source, state, training, evaluation, **options)
        if fault == "state":
            next(iter(state.parameters.values())).fill(99)
        elif fault == "source":
            source.state.completed_updates += 1
        elif fault == "float_updates":
            options["budget"].updates_completed = 16.0
        elif fault == "bool_replay":
            options["budget"].replay_examples_completed = False
        else:
            object.__setattr__(options["budget"].budget, "max_training_updates", 17)
        return result

    monkeypatch.setattr(pilot, "run_fresh_task_probe", corrupt)
    result = pilot.run_learnability_pilot()
    assert result["status"] == "incomplete"
    key = dict(
        state="state_unchanged",
        source="source_unchanged",
        budget="budget_binding_preserved",
        float_updates="budget_binding_preserved",
        bool_replay="budget_binding_preserved",
    )[fault]
    for row in result["runs"]:
        first = row["probes"][0]
        assert row["status"] == first["status"] == "failed"
        assert first["report"]["status"] == "complete" and first["charged_updates"] == 16
        assert first[key] is False
        assert row["wake_updates"] == row["replay_charged_examples"] == 0
        assert row["probes"][1]["status"] == "not_run"
        assert all(episode["status"] == "not_run" for episode in row["episodes"])


def test_should_preserve_all_probe_and_live_slots_if_resource_setup_fails(monkeypatch):
    def fail():
        raise RuntimeError("probe recipe resource setup")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    result = pilot.run_learnability_pilot()
    assert result["status"] == "incomplete" and result["global_errors"]
    assert len(result["runs"]) == 2
    assert sum(len(row["episodes"]) for row in result["runs"]) == 80
    assert sum(len(row["probes"]) for row in result["runs"]) == 4
    assert all(probe["status"] == "not_run" for row in result["runs"] for probe in row["probes"])


@pytest.mark.parametrize("describe", [False, True])
def test_should_route_only_requested_learnability_cli(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-learnability"]
        + (["--describe"] if describe else []),
    )
    selected = "describe_learnability_protocol" if describe else "run_learnability_pilot"
    for name in ("describe_learnability_protocol", "run_learnability_pilot"):
        monkeypatch.setattr(
            cli,
            name,
            (lambda: dict(status="complete", selected="reserved-probe"))
            if name == selected
            else forbidden,
        )
    for name in (
        "describe_key_door_protocol",
        "run_key_door_pilot",
        "describe_recurrent_protocol",
        "run_recurrent_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    cli.main()
    assert json.loads(capsys.readouterr().out)["selected"] == "reserved-probe"
