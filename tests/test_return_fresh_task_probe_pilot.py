"""Real disposable E probes on the return stream with isolated fake native owners."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.train_acquisition_pilot as pilot
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID
from src.app.coupled_attribution import AttributionStream
from src.app.fresh_task_probe import run_fresh_task_probe
from src.core.learner_ports import TrainingDiagnostic
from test_recurrent_key_door_pilot import Learner as RecurrentLearner, fake_environment
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden
from test_train_acquisition_pilot import Learner


class Session:
    def __init__(self, source):
        self.state = source.snapshot_state()
        self.completed_steps = 0

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, features):
        assert ProbeLearner.final_started and ProbeLearner.completed_probes == 2
        assert features.shape == (1, 8)
        self.completed_steps += 1
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)


class ProbeLearner(Learner):
    active_probe = 0
    completed_probes = 0
    final_started = False
    benchmark_calls: list[Any] = []
    probe_calls: list[Any] = []
    probe_predictions: list[Any] = []
    live_sessions: list[Session] = []
    fail_third_probe = False
    corrupt: Any = None

    def train_batch(self, features, targets):
        assert not ProbeLearner.final_started
        if ProbeLearner.active_probe:
            ProbeLearner.probe_calls.append((ProbeLearner.active_probe, self, features.copy()))
            if ProbeLearner.fail_third_probe and len(ProbeLearner.probe_calls) == 3:
                raise RuntimeError("third disposable probe update failed")
        else:
            ProbeLearner.benchmark_calls.append((self, features.copy()))
        super().train_batch(features, targets)
        if ProbeLearner.active_probe == 2 and ProbeLearner.corrupt is not None:
            ProbeLearner.corrupt()
        return TrainingDiagnostic(TORCH_RECURRENT_CE_ID, 0.5)

    def predict(self, features):
        if ProbeLearner.active_probe:
            ProbeLearner.probe_predictions.append(
                (ProbeLearner.active_probe, self.state.completed_updates, self)
            )
        if ProbeLearner.final_started:
            assert ProbeLearner.completed_probes == 2
        return super().predict(features)

    def start_episode(self):
        assert ProbeLearner.final_started and ProbeLearner.completed_probes == 2
        session = Session(self)
        ProbeLearner.live_sessions.append(session)
        return session


def environment(monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchRecurrentLearner", ProbeLearner)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    for name in ("sources", "training_owners", "prediction_owners"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "fail_third", False)
    monkeypatch.setattr(Learner, "mutate_prediction", False)
    monkeypatch.setattr(Learner, "mutation_update", 0)
    for name in ("benchmark_calls", "probe_calls", "probe_predictions", "live_sessions"):
        monkeypatch.setattr(ProbeLearner, name, [])
    monkeypatch.setattr(ProbeLearner, "active_probe", 0)
    monkeypatch.setattr(ProbeLearner, "completed_probes", 0)
    monkeypatch.setattr(ProbeLearner, "final_started", False)
    monkeypatch.setattr(ProbeLearner, "fail_third_probe", False)
    monkeypatch.setattr(ProbeLearner, "corrupt", None)
    captured: dict[str, Any] = dict(collections=[], probes=[], finished=False)
    collect, finish = shared._collect, AttributionStream.finish

    def collect_packet(identity, layout, cue, role):
        assert role in ("train", "probe_train", "probe_test", "final_test")
        if role == "final_test":
            assert captured["finished"] is True and ProbeLearner.completed_probes == 2
            ProbeLearner.final_started = True
        elif role in ("probe_train", "probe_test"):
            assert layout.layout_id == "E" and not ProbeLearner.final_started
        else:
            assert layout.layout_id in ("A", "B") and not ProbeLearner.final_started
        batch = collect(identity, layout, cue, role)
        captured["collections"].append(deepcopy(batch))
        return batch

    def stream(learner, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True and len(training) == 192
        captured.update(wake=kwargs["budget"], helper=kwargs.get("rehearsal"), isolated=learner)
        instance = AttributionStream(learner, training, heldout, **kwargs)
        captured["stream"] = instance
        return instance

    def finished(instance):
        result = finish(instance)
        assert instance.completed_updates == 192
        captured["finished"] = True
        return result

    def probe(source, state, training, evaluation, **kwargs):
        assert all(batch.role == "probe_train" for batch in training)
        assert all(batch.role == "probe_test" for batch in evaluation)
        assert len(training) == 16 and len(evaluation) == 2
        assert kwargs["decision_diagnostics"] is True
        number = len(captured["probes"]) + 1
        snapshot = deepcopy(source.snapshot_state())
        outer = [(captured["wake"], captured["wake"].last_clock)]
        if captured["helper"] is not None:
            outer.append(
                (captured["helper"]._replay_budget, captured["helper"]._replay_budget.last_clock)
            )
        entry = dict(
            source=source,
            state=deepcopy(state),
            training=deepcopy(training),
            evaluation=deepcopy(evaluation),
            budget=kwargs["budget"],
            source_before=snapshot,
            outer_clocks=outer,
        )
        captured["probes"].append(entry)
        ProbeLearner.active_probe = number
        try:
            report = run_fresh_task_probe(source, state, training, evaluation, **kwargs)
            entry["report"] = report
            entry["source_after"] = deepcopy(source.snapshot_state())
            entry["outer_clocks_after"] = [session.last_clock for session, _ in outer]
            if report["status"] == "complete":
                ProbeLearner.completed_probes += 1
            return report
        finally:
            ProbeLearner.active_probe = 0

    monkeypatch.setattr(shared, "_collect", collect_packet)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(AttributionStream, "finish", finished)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", probe)
    return captured


@pytest.mark.parametrize("replay", [False, True])
def test_should_describe_fixed_probes_without_resources_and_keep_old_protocols(monkeypatch, replay):
    before = deepcopy(pilot.describe_return_evaluation_protocol())
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
        "run_fresh_task_probe",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_return_probe_protocol(episode_replay=replay)
    assert pilot.describe_return_evaluation_protocol() == before
    assert protocol["architecture"] == dict(input_dim=8, hidden_dim=8, num_classes=5, seed=17)
    assert protocol["planned_work"]["native_train_calls"] == (320 if replay else 224)
    assert protocol["planned_work"]["probes"] == 2


@pytest.mark.parametrize("entry", ["describe_return_probe_protocol", "run_return_probe_pilot"])
@pytest.mark.parametrize("invalid", [0, None])
def test_should_reject_non_bool_replay_choice_before_resources(monkeypatch, entry, invalid):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError, match="episode_replay"):
        getattr(pilot, entry)(episode_replay=invalid)


@pytest.mark.parametrize("replay", [False, True])
def test_should_train_two_disposable_E_cohorts_without_changing_benchmark_or_original_budgets(
    monkeypatch, replay
):
    captured = environment(monkeypatch)
    report = pilot.run_return_probe_pilot(episode_replay=replay)
    assert report["status"] == "complete", report["reason"]
    benchmark_native = 288 if replay else 192
    assert len(ProbeLearner.benchmark_calls) == benchmark_native
    assert len(ProbeLearner.probe_calls) == 32
    assert report["charged_updates"] == captured["wake"].updates_completed == 192
    assert len(captured["probes"]) == 2 and captured["finished"] is True
    assert [slot["checkpoint_updates"] for slot in report["probes"]] == [0, 192]
    assert [slot["native_updates_before"] for slot in report["probes"]] == [0, benchmark_native]
    assert report["partial_stream"]["native_candidate_updates"] == benchmark_native
    assert report["partial_stream"]["native_candidate_training_examples"] == benchmark_native * 8
    benchmark_owners = [owner for owner, _ in ProbeLearner.benchmark_calls]
    probe_owners = [owner for _, owner, _ in ProbeLearner.probe_calls]
    assert all(owner not in benchmark_owners for owner in probe_owners)
    assert len({id(owner) for owner in probe_owners}) == 2
    for number, (entry, slot) in enumerate(
        zip(captured["probes"], report["probes"], strict=True), 1
    ):
        assert slot["status"] == "complete" and slot["charged_updates"] == 16
        assert (
            slot["source_unchanged"]
            is slot["state_unchanged"]
            is slot["budget_binding_preserved"]
            is True
        )
        assert entry["budget"] is not captured["wake"]
        assert (
            entry["budget"].budget.max_training_updates == entry["budget"].updates_completed == 16
        )
        assert entry["budget"].replay_examples_completed == 0
        assert pickle.dumps(entry["source_before"]) == pickle.dumps(entry["source_after"])
        assert entry["outer_clocks_after"] == [clock for _, clock in entry["outer_clocks"]]
        curve = slot["report"]
        assert curve["completed_training_examples"] == 128
        assert curve["score_forward_calls_completed"] == 8
        assert [point["updates"] for point in curve["checkpoints"]] == [0, 4, 8, 16]
        source_native = 0 if number == 1 else benchmark_native
        assert [n for cohort, n, _ in ProbeLearner.probe_predictions if cohort == number] == [
            source_native + relative for relative in (0, 4, 8, 16) for _ in range(2)
        ]
        for point in curve["checkpoints"]:
            assert set(point["scores"]) == set(point["decision_diagnostics"]) == {"E-red", "E-blue"}
            for detail in point["decision_diagnostics"].values():
                assert detail["rows"] == len(detail["decisions"]) == 8
                assert detail["predicted_class_counts"] == [8, 0, 0, 0, 0]
        assert slot["outcomes"]["ce_gain"] == 0
        assert slot["outcomes"]["initial_ce"] == slot["outcomes"]["final_ce"]
    assert captured["probes"][0]["budget"] is not captured["probes"][1]["budget"]
    assert all(not np.all(features[:, 7] == 0.75) for _, features in ProbeLearner.benchmark_calls)
    assert all(np.all(features[:, 7] == 0.75) for _, _, features in ProbeLearner.probe_calls)
    assert len(ProbeLearner.live_sessions) == 32
    assert len(RecurrentLearner.classifications) == 100
    assert [batch.role for batch in captured["collections"]].count("probe_train") == 16
    assert [batch.role for batch in captured["collections"]].count("probe_test") == 2
    assert [batch.role for batch in captured["collections"]].count("final_test") == 8
    assert (
        report["source_unchanged"]
        is report["isolated_source_unchanged"]
        is report["actor_unchanged"]
        is True
    )
    assert report["scientific_advantage_verified"] is False


def test_should_preserve_partial_first_probe_charges_and_admit_no_benchmark_or_final_work(
    monkeypatch,
):
    captured = environment(monkeypatch)
    monkeypatch.setattr(ProbeLearner, "fail_third_probe", True)
    report = pilot.run_return_probe_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == captured["wake"].updates_completed == 0
    assert ProbeLearner.benchmark_calls == [] and len(ProbeLearner.probe_calls) == 3
    first, later = report["probes"]
    assert first["status"] == "failed" and first["charged_updates"] == 2
    assert first["report"]["attempted_updates"] == 3
    assert first["report"]["completed_updates"] == 2
    assert first["report"]["completed_training_examples"] == 16
    assert later["status"] == "not_run" and later["charged_updates"] == 0
    assert not any(batch.role == "final_test" for batch in captured["collections"])
    assert report["policy_evaluation"]["status"] == "not_run"


def test_should_refuse_replaced_initial_checkpoint_before_probe_or_benchmark_callbacks(monkeypatch):
    captured = environment(monkeypatch)
    run_checkpoint = pilot.run_return_checkpoint_probe

    def replace_checkpoint(row, source, state, data, **kwargs):
        assert kwargs["checkpoint"] == 0
        altered = deepcopy(state)
        altered.parameters["weight_input"][0, 0] += 0.125
        assert altered.completed_updates == state.completed_updates == 0
        assert altered.training_examples == state.training_examples == 0
        return run_checkpoint(row, source, altered, data, **kwargs)

    monkeypatch.setattr(pilot, "run_return_checkpoint_probe", replace_checkpoint)
    report = pilot.run_return_probe_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == captured["wake"].updates_completed == 0
    assert captured["probes"] == []
    assert ProbeLearner.probe_calls == ProbeLearner.benchmark_calls == []
    assert all(slot["charged_updates"] == 0 for slot in report["probes"])
    assert not any(batch.role == "final_test" for batch in captured["collections"])
    assert report["policy_evaluation"]["status"] == "not_run"


@pytest.mark.parametrize("fault", ["candidate", "source", "receipts", "replay_budget"])
def test_should_refuse_benchmark_mutation_during_second_probe_before_more_updates_or_final_data(
    monkeypatch, fault
):
    captured = environment(monkeypatch)
    changed: list[bool] = []

    def corrupt():
        assert not changed, "corrupted benchmark admitted another probe native update"
        changed.append(True)
        stream = captured["stream"]
        if fault == "candidate":
            stream._runtime._candidate.state.completed_updates += 1
        elif fault == "source":
            Learner.sources[0].state.completed_updates += 1
        elif fault == "receipts":
            inbox = stream._runtime._inbox
            key = next(iter(inbox._applied))
            receipt = inbox._applied[key]
            inbox._applied[key] = replace(
                receipt,
                diagnostic=replace(receipt.diagnostic, value=receipt.diagnostic.value + 0.125),
            )
        else:
            captured["helper"]._replay_budget.replay_examples_completed -= 8

    monkeypatch.setattr(ProbeLearner, "corrupt", staticmethod(corrupt))
    report = pilot.run_return_probe_pilot(episode_replay=True)
    assert report["status"] == "incomplete" and changed == [True]
    assert report["charged_updates"] == captured["wake"].updates_completed == 192
    assert len(ProbeLearner.benchmark_calls) == 288
    assert len(ProbeLearner.probe_calls) == 17
    first, second = report["probes"]
    assert first["status"] == "complete" and first["charged_updates"] == 16
    assert second["status"] == "failed" and second["charged_updates"] == 1
    assert second["report"]["completed_updates"] == 1
    assert second["report"]["diagnostics"] == [dict(definition=TORCH_RECURRENT_CE_ID, value=0.5)]
    assert not any(batch.role == "final_test" for batch in captured["collections"])
    assert ProbeLearner.live_sessions == []
    assert report["policy_evaluation"]["status"] == "not_run"


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_probe_cli_with_replay_option_when_requested(
    monkeypatch, capsys, replay, describe
):
    scenario = "torch-key-door-return-replay-probe" if replay else "torch-key-door-return-probe"
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", scenario] + (["--describe"] if describe else [])
    )
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return dict(status="complete", selected="return-probe")

    monkeypatch.setattr(cli, "describe_return_probe_protocol", selected if describe else forbidden)
    monkeypatch.setattr(cli, "run_return_probe_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_return_replay_pilot", forbidden)
    monkeypatch.setattr(cli, "run_return_policy_pilot", forbidden)
    cli.main()
    assert calls == ([dict(episode_replay=True)] if replay else [{}])
    assert json.loads(capsys.readouterr().out)["selected"] == "return-probe"
