"""Disposable E probes on the existing four-phase benchmark with fake native owners."""

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
from test_four_phase_replay_pilot import environment as ordinary_environment
from test_recurrent_key_door_pilot import Learner as RecurrentLearner, fake_environment
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden
from test_train_acquisition_pilot import Learner


CONTROLS = ("coverage", "recent48", "partitioned48")


class Session:
    def __init__(self, source):
        self.state = source.snapshot_state()
        self.completed_steps = 0

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, features):
        assert ProbeLearner.completed_probes == 2 and ProbeLearner.final_started
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
    fail: str | None = None
    corrupt: Any = None

    def train_batch(self, features, targets):
        assert not ProbeLearner.final_started
        if ProbeLearner.active_probe:
            ProbeLearner.probe_calls.append((ProbeLearner.active_probe, self, features.copy()))
            if ProbeLearner.fail == "probe_first" and len(ProbeLearner.probe_calls) == 3:
                raise RuntimeError("third four-phase probe update failed")
        else:
            ProbeLearner.benchmark_calls.append((self, features.copy()))
            if ProbeLearner.fail == "replay_first" and len(ProbeLearner.benchmark_calls) == 3:
                raise RuntimeError("first four-phase replay update failed")
        super().train_batch(features, targets)
        if ProbeLearner.active_probe == 2 and ProbeLearner.corrupt is not None:
            ProbeLearner.corrupt()
        return TrainingDiagnostic(TORCH_RECURRENT_CE_ID, 0.5)

    def predict(self, features):
        if ProbeLearner.active_probe:
            ProbeLearner.probe_predictions.append(
                (ProbeLearner.active_probe, self.state.completed_updates)
            )
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
    monkeypatch.setattr(ProbeLearner, "fail", None)
    monkeypatch.setattr(ProbeLearner, "corrupt", None)
    captured: dict[str, Any] = dict(collections=[], probes=[], finished=False)
    collect, finish, checkpoint = (
        shared._collect,
        AttributionStream.finish,
        pilot.run_return_checkpoint_probe,
    )

    def collect_packet(identity, layout, cue, role):
        assert role in ("train", "probe_train", "probe_test", "final_test")
        if role == "final_test":
            assert captured["finished"] is True and ProbeLearner.completed_probes == 2
            assert len(ProbeLearner.benchmark_calls) == 384
            ProbeLearner.final_started = True
        elif role in ("probe_train", "probe_test"):
            assert layout.layout_id == "E" and not ProbeLearner.final_started
        else:
            assert layout.layout_id in ("A", "B", "C_train")
        batch = collect(identity, layout, cue, role)
        captured["collections"].append(deepcopy(batch))
        return batch

    def stream(learner, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True and len(training) == 256
        captured.update(wake=kwargs["budget"], helper=kwargs["rehearsal"], isolated=learner)
        instance = AttributionStream(learner, training, heldout, **kwargs)
        captured["stream"] = instance
        return instance

    def finished(instance):
        result = finish(instance)
        assert instance.completed_updates == 256
        captured["finished"] = True
        return result

    def probe(source, state, training, evaluation, **kwargs):
        assert len(training) == 16 and len(evaluation) == 2
        assert all(batch.role == "probe_train" for batch in training)
        assert all(batch.role == "probe_test" for batch in evaluation)
        assert kwargs["decision_diagnostics"] is True
        number = len(captured["probes"]) + 1
        outer = (captured["wake"], captured["helper"]._replay_budget)
        entry = dict(
            state=deepcopy(state),
            budget=kwargs["budget"],
            source_before=deepcopy(source.snapshot_state()),
            outer_before=[
                (b.updates_completed, b.replay_examples_completed, b.last_clock) for b in outer
            ],
        )
        captured["probes"].append(entry)
        ProbeLearner.active_probe = number
        try:
            report = run_fresh_task_probe(source, state, training, evaluation, **kwargs)
            entry.update(
                report=report,
                source_after=deepcopy(source.snapshot_state()),
                outer_after=[
                    (b.updates_completed, b.replay_examples_completed, b.last_clock) for b in outer
                ],
            )
            if report["status"] == "complete":
                ProbeLearner.completed_probes += 1
            return report
        finally:
            ProbeLearner.active_probe = 0

    def capture_row(row, *args, **kwargs):
        captured["row"] = row
        return checkpoint(row, *args, **kwargs)

    monkeypatch.setattr(shared, "_collect", collect_packet)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(AttributionStream, "finish", finished)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", probe)
    monkeypatch.setattr(pilot, "run_return_checkpoint_probe", capture_row)
    return captured


@pytest.mark.parametrize("control", CONTROLS)
def test_should_describe_fixed256_boundary_probes_and_count_all_disposable_work_without_resources(
    monkeypatch, control
):
    before = deepcopy(pilot.describe_four_phase_replay_protocol(control=control))
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    description = pilot.describe_four_phase_probe_protocol(control=control)
    assert pilot.describe_four_phase_replay_protocol(control=control) == before
    assert description["probe"]["source_wake_checkpoints"] == [0, 256]
    assert description["probe"]["source_native_checkpoints"] == [0, 384]
    work = description["planned_work"]
    assert work["native_train_calls"] == 416 and work["training_rows"] == 3328
    assert work["probe_train_calls"] == 32 and work["probe_score_forwards"] == 16
    assert work["probes"] == 2


@pytest.mark.parametrize("invalid", [None, 0, "true"])
def test_should_refuse_non_bool_private_probe_option_before_any_resources(monkeypatch, invalid):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot._run_pilot(four_phase_replay="coverage", four_phase_probes=invalid)


def test_should_preserve_old_direct_mixed_flag_refusal_before_resources(monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot._run_pilot(four_phase_replay="coverage", fresh_task_probes=True)
    with pytest.raises(ValueError):
        pilot._run_pilot(four_phase_probes=True)


def test_should_keep_complete_coverage_benchmark_output_exact_and_own_both_disposable_probe_cohorts(
    monkeypatch,
):
    with monkeypatch.context() as scoped:
        captured = environment(scoped)
        report = pilot.run_four_phase_probe_pilot(control="coverage")
        assert report["status"] == "complete", report["reason"]
        assert len(ProbeLearner.benchmark_calls) == 384 and len(ProbeLearner.probe_calls) == 32
        assert len(RecurrentLearner.classifications) == 158
        assert report["charged_updates"] == captured["wake"].updates_completed == 256
        assert report["replay"]["charged_examples"] == 1024
        assert report["partial_stream"]["native_candidate_updates"] == 384
        assert [slot["checkpoint_updates"] for slot in report["probes"]] == [0, 256]
        assert [slot["native_updates_before"] for slot in report["probes"]] == [0, 384]
        benchmark_owners = [owner for owner, _ in ProbeLearner.benchmark_calls]
        probe_owners = [owner for _, owner, _ in ProbeLearner.probe_calls]
        assert len({id(owner) for owner in probe_owners}) == 2
        assert all(owner not in benchmark_owners for owner in probe_owners)
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
            assert (
                entry["budget"].budget.max_training_updates
                == entry["budget"].updates_completed
                == 16
            )
            assert entry["budget"] is not captured["wake"]
            assert entry["outer_before"] == entry["outer_after"]
            assert pickle.dumps(entry["source_before"]) == pickle.dumps(entry["source_after"])
            curve = slot["report"]
            assert (
                curve["completed_training_examples"] == 128
                and curve["score_forward_calls_completed"] == 8
            )
            assert [point["updates"] for point in curve["checkpoints"]] == [0, 4, 8, 16]
            base_native = 0 if number == 1 else 384
            assert [n for cohort, n in ProbeLearner.probe_predictions if cohort == number] == [
                base_native + relative for relative in (0, 4, 8, 16) for _ in range(2)
            ]
            for point in curve["checkpoints"]:
                assert set(point["decision_diagnostics"]) == {"E-red", "E-blue"}
                assert all(
                    len(detail["decisions"]) == 8
                    for detail in point["decision_diagnostics"].values()
                )
            assert slot["outcomes"]["ce_gain"] == 0
        assert all(np.all(features[:, 7] == 0.75) for _, _, features in ProbeLearner.probe_calls)
        assert all(
            not np.all(features[:, 7] == 0.75) for _, features in ProbeLearner.benchmark_calls
        )
        assert len(ProbeLearner.live_sessions) == 40
        assert (
            report["source_unchanged"]
            is report["isolated_source_unchanged"]
            is report["actor_unchanged"]
            is True
        )
        probe_report = deepcopy(report)
    with monkeypatch.context() as scoped:
        ordinary_environment(scoped)
        ordinary = pilot.run_four_phase_replay_pilot(control="coverage")
        assert ordinary["status"] == "complete", ordinary["reason"]
    excluded = {"protocol", "probes", "total_seconds", "observed_process_rss"}
    assert {k: v for k, v in probe_report.items() if k not in excluded} == {
        k: v for k, v in ordinary.items() if k not in excluded
    }


def test_should_refuse_malformed_initial_complete_probe_before_any_benchmark_update(monkeypatch):
    captured = environment(monkeypatch)
    runner = pilot.run_fresh_task_probe

    def malformed(*args, **kwargs):
        report = runner(*args, **kwargs)
        report["completed_training_examples"] = 0
        return report

    monkeypatch.setattr(pilot, "run_fresh_task_probe", malformed)
    report = pilot.run_four_phase_probe_pilot(control="coverage")
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == captured["wake"].updates_completed == 0
    assert ProbeLearner.benchmark_calls == [] and len(ProbeLearner.probe_calls) == 16
    assert report["probes"][0]["charged_updates"] == 16
    assert report["probes"][1]["status"] == "not_run"
    assert all(batch.role != "final_test" for batch in captured["collections"])


def test_should_refuse_changed_initial_probe_after_first_wake_before_a_second_update(monkeypatch):
    captured = environment(monkeypatch)
    train = ProbeLearner.train_batch
    changed: list[bool] = []

    def change_initial(self, features, targets):
        diagnostic = train(self, features, targets)
        if ProbeLearner.active_probe == 0 and not changed:
            captured["row"]["probes"][0]["report"]["diagnostics"][0]["value"] += 0.125
            changed.append(True)
        return diagnostic

    monkeypatch.setattr(ProbeLearner, "train_batch", change_initial)
    report = pilot.run_four_phase_probe_pilot(control="coverage")
    assert report["status"] == "incomplete" and changed == [True]
    assert report["charged_updates"] == captured["wake"].updates_completed == 1
    assert len(ProbeLearner.benchmark_calls) == 1 and len(ProbeLearner.probe_calls) == 16
    assert report["replay"]["charged_examples"] == 0
    assert report["probes"][0]["charged_updates"] == 16
    assert report["probes"][1]["status"] == "not_run"
    assert all(batch.role != "final_test" for batch in captured["collections"])


@pytest.mark.parametrize("failure", ["probe_first", "replay_first"])
def test_should_keep_independent_probe_and_benchmark_charges_when_initial_work_fails(
    monkeypatch, failure
):
    captured = environment(monkeypatch)
    monkeypatch.setattr(ProbeLearner, "fail", failure)
    report = pilot.run_four_phase_probe_pilot(control="coverage")
    assert report["status"] == "incomplete"
    first, later = report["probes"]
    if failure == "probe_first":
        assert report["charged_updates"] == 0 and first["charged_updates"] == 2
        assert (
            first["report"]["attempted_updates"] == 3 and first["report"]["completed_updates"] == 2
        )
        assert ProbeLearner.benchmark_calls == []
    else:
        assert report["charged_updates"] == 2 and first["charged_updates"] == 16
        assert report["replay"]["charged_examples"] == 8
        assert report["partial_stream"]["native_candidate_updates"] == 2
    assert later["status"] == "not_run" and later["charged_updates"] == 0
    assert all(batch.role != "final_test" for batch in captured["collections"])
    assert report["policy_evaluation"]["status"] == "not_run"


@pytest.mark.parametrize("fault", ["source", "candidate_optimizer", "receipts", "replay_budget"])
def test_should_refuse_benchmark_corruption_during_post256_probe_before_another_update_or_final_data(
    monkeypatch, fault
):
    captured = environment(monkeypatch)
    changed: list[bool] = []

    def corrupt():
        assert not changed
        changed.append(True)
        stream = captured["stream"]
        if fault == "source":
            Learner.sources[0].state.parameters["weight_input"][0, 0] += 0.125
        elif fault == "candidate_optimizer":
            stream._runtime._candidate.state.optimizer_state["weight_input"][0, 0] += 0.125
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
    report = pilot.run_four_phase_probe_pilot(control="coverage")
    assert report["status"] == "incomplete" and changed == [True]
    assert report["charged_updates"] == 256 and len(ProbeLearner.benchmark_calls) == 384
    assert len(ProbeLearner.probe_calls) == 17
    assert [slot["charged_updates"] for slot in report["probes"]] == [16, 1]
    assert report["probes"][1]["report"]["diagnostics"] == [
        dict(definition=TORCH_RECURRENT_CE_ID, value=0.5)
    ]
    assert all(batch.role != "final_test" for batch in captured["collections"])
    assert ProbeLearner.live_sessions == []


def test_should_refuse_changed_completed_probe_evidence_before_more_final_collection(monkeypatch):
    captured = environment(monkeypatch)
    collect = shared._collect

    def change_probe(identity, layout, cue, role):
        batch = collect(identity, layout, cue, role)
        if role == "final_test":
            captured["row"]["probes"][0]["report"]["diagnostics"][0]["value"] += 0.125
        return batch

    monkeypatch.setattr(shared, "_collect", change_probe)
    report = pilot.run_four_phase_probe_pilot(control="coverage")
    assert report["status"] == "incomplete" and report["charged_updates"] == 256
    assert [slot["charged_updates"] for slot in report["probes"]] == [16, 16]
    assert sum(batch.role == "final_test" for batch in captured["collections"]) == 1
    assert report["policy_evaluation"]["checkpoints"] == []
    assert ProbeLearner.live_sessions == []


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_explicit_four_phase_probe_cli_with_fixed_control(
    monkeypatch, capsys, control, describe
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", f"torch-key-door-four-phase-{control}-probe"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected(*, control):
        calls.append(control)
        return dict(status="complete", selected=control)

    monkeypatch.setattr(
        cli, "describe_four_phase_probe_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_four_phase_probe_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_four_phase_replay_pilot", forbidden)
    cli.main()
    assert calls == [control] and json.loads(capsys.readouterr().out)["selected"] == control
