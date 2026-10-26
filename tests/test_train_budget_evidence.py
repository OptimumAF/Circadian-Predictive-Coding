"""Original budget evidence on scripted native models; no real model work."""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from time import monotonic

import pytest

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.minigrid_public_cue_recurrent_learner import METHOD, PublicCueRecurrentLearner
from src.app.coupled_attribution import _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.process_memory import ProcessRssSampler
from test_public_cue_recurrent_learner import configure_scripted_native, ScriptedNative
from test_public_cue_health import sources


@pytest.fixture
def scripted(monkeypatch):
    configure_scripted_native(monkeypatch)
    monkeypatch.setattr(shared, "_same_state", _same_value)


def run(module=shared, row=None, *, flag=True):
    row = shared._row(METHOD) if row is None else row
    with ProcessRssSampler() as memory:
        arm = module._train_arm(
            row,
            shared._packets(sources())[:32],
            memory,
            monotonic() + 30,
            lambda: None,
            learner_factory=lambda m: PublicCueRecurrentLearner(),
            checkpoint_updates=(0, 32),
            **({"record_budget_evidence": flag} if flag is not None else {}),
        )
    return row, arm


def test_should_retain_original_budget_and_bounded_owned_evidence(scripted):
    row, arm = run()
    e = row["budget_evidence"]
    assert e["born"] and all(e["bindings"].values())
    assert e["original"]["limits"]["max_training_updates"] == 32
    assert (
        e["observed"]["updates_completed"] == 32 and e["observed"]["replay_examples_completed"] == 0
    )
    assert e["observed"]["last_clock"] >= e["original"]["started_at"]
    assert row["source_unchanged"] and row["actor_unchanged"]
    assert len(json.dumps(e)) < 4096
    arm["guard"]()


@pytest.mark.parametrize("fault", ["mutate", "replace"])
def test_should_refuse_changed_evidence_without_more_work(scripted, fault):
    row, arm = run()
    calls = len(ScriptedNative.calls)
    if fault == "mutate":
        row["budget_evidence"]["original"]["started_at"] = 0
    else:
        row["budget_evidence"] = deepcopy(row["budget_evidence"])
    with pytest.raises(ValueError):
        arm["guard"]()
    assert len(ScriptedNative.calls) == calls


def test_should_export_birth_and_charges_when_native_readback_is_unavailable(scripted, monkeypatch):
    row = shared._row(METHOD)

    def unavailable(*a, **k):
        raise RuntimeError("candidate snapshot unavailable")

    monkeypatch.setattr(shared, "_readback", unavailable)
    with pytest.raises(RuntimeError):
        run(row=row)
    assert row["status"] == "incomplete"
    assert row["budget_evidence"]["observed"]["updates_completed"] == 32
    assert row["budget_evidence"]["original"]["started_at"] > 0
    assert all(row["budget_evidence"]["bindings"].values())


def test_should_keep_failed_update_prefix_and_real_birth(scripted, monkeypatch):
    original = ScriptedNative.train_batch

    def lost(self, features, targets):
        result = original(self, features, targets)
        if len(ScriptedNative.calls) == 3:
            raise RuntimeError("lost native return")
        return result

    monkeypatch.setattr(ScriptedNative, "train_batch", lost)
    row = shared._row(METHOD)
    with pytest.raises(Exception, match="lost native return"):
        run(row=row)
    e = row["budget_evidence"]
    assert e["observed"]["updates_completed"] == row["charged_updates"] == 2
    assert len(row["partial_stream"]["applied_updates"]) == 2
    assert e["original"]["limits"]["max_training_updates"] == 32
    assert e["observed"]["last_clock"] >= e["original"]["started_at"]


@pytest.mark.parametrize("fault", ["clock", "start", "config", "limits"])
def test_should_keep_original_birth_and_report_changed_bindings(scripted, monkeypatch, fault):
    budgets, born = [], []
    constructor, train = shared.ToyBudgetSession, ScriptedNative.train_batch

    def construct(*args, **kwargs):
        b = constructor(*args, **kwargs)
        budgets.append(b)
        born.append((b.started_at, asdict(b.budget)))
        return b

    def change(self, features, targets):
        result = train(self, features, targets)
        if len(ScriptedNative.calls) == 1:
            b = budgets[0]
            if fault == "clock":
                b.clock = lambda: 0.0
            elif fault == "start":
                b.started_at += 1000
            elif fault == "config":
                b.budget = replace(b.budget, max_training_updates=64)
            else:
                object.__setattr__(b.budget, "max_wall_seconds", 1.0)
        return result

    monkeypatch.setattr(shared, "ToyBudgetSession", construct)
    monkeypatch.setattr(ScriptedNative, "train_batch", change)
    row = shared._row(METHOD)
    with pytest.raises(Exception):
        run(row=row)
    e = row["budget_evidence"]
    assert e["original"] == dict(started_at=born[0][0], limits=born[0][1])
    key = {"clock": "clock", "start": "started_at", "config": "config", "limits": "limits"}[fault]
    assert e["bindings"][key] is False and len(ScriptedNative.calls) == 1


def test_should_keep_off_default_schema_against_prechange_source(scripted, monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/train-budget-evidence-20261025/pilot_before.py"
    )
    spec = importlib.util.spec_from_file_location("before_train_budget", path)
    assert spec is not None and spec.loader is not None
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    monkeypatch.setattr(old, "_same_state", _same_value)
    previous, _ = run(old, flag=None)
    current, _ = run(flag=False)
    assert previous.keys() == current.keys() and "budget_evidence" not in current
    assert previous["stream_result"].keys() == current["stream_result"].keys()
    assert previous["charged_updates"] == current["charged_updates"] == 32
    assert previous["native_training_examples"] == current["native_training_examples"] == 288


@pytest.mark.parametrize("flag", [0, 1, "yes"])
def test_should_refuse_invalid_option_before_native(scripted, flag):
    with pytest.raises(ValueError):
        run(flag=flag)
    assert not ScriptedNative.calls


def test_should_refuse_replacing_prior_budget_evidence(scripted):
    row = shared._row(METHOD)
    row["budget_evidence"] = {"original": "retired"}
    with pytest.raises(ValueError):
        run(row=row)
    assert row["budget_evidence"] == {"original": "retired"} and not ScriptedNative.calls


def test_should_mark_invalid_counter_without_losing_origin():
    b = ToyBudgetSession(ToyExecutionBudget(max_training_updates=32), lambda: 1.0)
    origin = (b, b.budget, asdict(b.budget), b.clock, b.started_at)
    b.updates_completed = True
    result = shared._training_budget_evidence(b, origin)
    assert result["observed"]["updates_completed"] is None
    assert result["invalid_fields"] == ["updates_completed"]
    assert result["original"]["started_at"] == 1.0


def test_should_distinguish_no_budget_birth_when_source_construction_fails(scripted):
    row = shared._row(METHOD)

    def unavailable(method):
        raise RuntimeError("source initialization unavailable")

    with ProcessRssSampler() as memory:
        with pytest.raises(RuntimeError, match="source initialization"):
            shared._train_arm(
                row,
                shared._packets(sources())[:32],
                memory,
                monotonic() + 30,
                lambda: None,
                learner_factory=unavailable,
                checkpoint_updates=(0, 32),
                record_budget_evidence=True,
            )
    assert row["budget_evidence"] == dict(born=False, original=None, observed=None, bindings=None)
    assert not ScriptedNative.calls
