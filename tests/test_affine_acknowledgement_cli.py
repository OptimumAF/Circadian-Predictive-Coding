"""Ordinary CLI/API correct-ACK composition with real stream and literal work."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import random
import sys
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_context_linear_learner as native
from src.adapters.minigrid_context_linear_policy import ContextLinearPolicy
from src.app.coupled_attribution import AttributionStream
from src.core.learner_ports import TrainingDiagnostic
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.reference_controller import ReferencePolicy
from tests.test_arrived_cue_catalogue import sources

WORK = dict(owners=0, snapshots=0, updates=0, rows=0, predicts=0, prediction_rows=0)


class Literal:
    hook = staticmethod(lambda owner, features: None)

    def __init__(self):
        WORK["owners"] += 1
        self.state = native.ContextLinearTrainingSnapshot(ContextLinearPolicy())

    def snapshot_state(self):
        WORK["snapshots"] += 1
        return self.state

    def restore_state(self, state):
        self.state = deepcopy(state)

    def fork(self):
        child = Literal()
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, features, targets):
        WORK["updates"] += 1
        WORK["rows"] += len(features)
        heads = list(self.state.heads)
        marker = tuple(float(i == (5 if np.any(targets[:, 0] == 5) else 2)) for i in range(7))
        heads[int(features[0, 1] > -0.5)] = CategoricalLinearFit(
            296, 7, len(features), ((0.0,) * 7,) * 296 + (marker,), 1, (1.0,) * len(features), 0.0
        )
        self.state = replace(
            self.state,
            completed_updates=self.state.completed_updates + 1,
            training_examples=self.state.training_examples + len(features),
            heads=(heads[0], heads[1]),
        )
        return TrainingDiagnostic("literal", 0.0)

    def predict(self, features):
        WORK["predicts"] += 1
        WORK["prediction_rows"] += len(features)
        self.hook(self, features)
        # Scripted A is correct; each B cue becomes correct after its own fit.
        # This is a literal state machine, not a fitted numeric policy.
        classes = np.where(features[:, 3] < -0.6, 2, 0)
        for cue, head in enumerate(self.state.heads):
            if head is not None and head.coefficients[-1][5] == 1.0:
                classes[(features[:, 3] >= -0.6) & ((features[:, 1] > -0.5) == cue)] = 5
        return np.eye(7, dtype=np.float64)[classes]


@pytest.fixture(autouse=True)
def literal_only(monkeypatch):
    monkeypatch.setattr(native, "ContextLinearLearner", Literal)
    monkeypatch.setattr(Literal, "hook", staticmethod(lambda owner, features: None))

    def forbidden(*a, **kw):
        pytest.fail("genuine native/solver/finish execution")

    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(AttributionStream, "finish", forbidden)


def run(**options):
    return cli.run_context_linear_stream(
        sources(),
        reference_policy=ReferencePolicy("drift"),
        ready_feedback=True,
        acknowledge_correct=True,
        **options,
    )


@pytest.mark.parametrize("bank", [False, True])
def test_should_advance_correct_A_and_fit_actual_B_packets_with_exact_rows(bank):
    packets = sources()
    before = pickle.dumps((packets, random.getstate(), np.random.get_state()))
    r = cli.run_context_linear_stream(
        packets,
        reference_policy=ReferencePolicy("drift"),
        ready_feedback=True,
        acknowledge_correct=True,
        retain_arrived=bank,
    )
    assert r["status"] == "complete", r.get("reason")
    w = r["stream"]
    assert w["applied_packet_indices"] == [2, 3]
    assert w["processed_packets"] == 6 and w["pending_unapplied"] == 0
    assert w["consumption_complete"] and not w["training_complete"]
    assert w["completed_updates"] == r["budget"]["updates_completed"] == 2
    assert w["training_examples"] == (36 if bank else 18)
    assert len(w["acknowledged_experiences"]) == 4 and w["unfitted_packets"] == 4
    assert r["source_unchanged"] and r["actor_unchanged"] and r["endpoints"] == []
    assert (
        r["score_calls_attempted"] == 0 and r["budget"]["initial_limits"] == r["budget"]["limits"]
    )
    assert pickle.dumps((packets, random.getstate(), np.random.get_state())) == before
    if bank:
        assert r["catalogue"]["committed_rows"] == dict(arrivals=18, replay=18, native=36)
        assert r["catalogue"]["attempted_rows_upper_bound"]["native"] >= 36


@pytest.mark.parametrize("mode", ["no_learning", "periodic", "drift", "budget_aware"])
@pytest.mark.parametrize("bank", [False, True])
def test_should_forward_true_through_cli_and_save_separate_proof_costs(
    monkeypatch, tmp_path, capsys, mode, bank
):
    monkeypatch.setattr(cli, "read_training_sources", lambda path: sources())
    output = tmp_path / "ack.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "affine",
            "--context-linear-reference",
            mode,
            "--ready-feedback",
            "--acknowledge-correct",
            "--source",
            "literal.json",
            "--output",
            str(output),
            *(["--reference-retain-arrived"] if bank else []),
        ],
    )
    cli.main()
    result = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert result["protocol"]["acknowledge_correct"] is True
    w = result["stream"]
    assert w["acknowledgement"]["proof"]["prediction_calls_completed"] == len(
        w["acknowledged_experiences"]
    )
    assert len(w["applied_updates"]) == result["budget"]["updates_completed"]
    assert w["processed_packets"] == len(w["applied_updates"]) + len(w["acknowledged_experiences"])


@pytest.mark.parametrize("bank", [False, True])
def test_should_describe_all_ready_and_fresh_proof_upper_bounds_without_work(bank):
    before = dict(WORK)
    r = cli.describe_context_linear_reference(
        ReferencePolicy("drift"), ready_feedback=True, acknowledge_correct=True, retain_arrived=bank
    )
    assert r["diagnostic_prediction_calls_upper_bound"] == r["diagnostic_forks_upper_bound"] == 12
    assert r["diagnostic_prediction_rows_upper_bound"] == (216 if bank else 108)
    assert r["acknowledgement_proof_prediction_calls_upper_bound"] == 6
    assert r["acknowledgement_proof_forks_upper_bound"] == 6
    assert WORK == before


@pytest.mark.parametrize("bad", [None, 1, "true"])
def test_should_refuse_inexact_api_option_before_owners(bad):
    before = dict(WORK)
    with pytest.raises(ValueError, match="acknowledgement"):
        cli.run_context_linear_stream(
            sources(),
            reference_policy=ReferencePolicy("drift"),
            ready_feedback=True,
            acknowledge_correct=bad,
        )
    assert WORK == before


@pytest.mark.parametrize(
    "options",
    [
        dict(),
        dict(reference_policy=ReferencePolicy("periodic")),
        dict(reference_policy=ReferencePolicy("periodic"), prequential_feedback=True),
        dict(
            reference_policy=ReferencePolicy("periodic"),
            ready_feedback=True,
            a4_training=True,
            retain_arrived=True,
        ),
    ],
)
def test_should_refuse_unsupported_api_combinations_before_owners(options):
    before = dict(WORK)
    with pytest.raises(ValueError, match="acknowledgement"):
        cli.run_context_linear_stream(sources(), acknowledge_correct=True, **options)
    assert WORK == before


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--ready-feedback"],
        ["--context-linear-reference", "periodic"],
        ["--context-linear-reference", "periodic", "--prequential-feedback"],
        ["--context-linear-reference", "periodic", "--ready-feedback", "--example-affine-a4"],
        ["--example-affine-published", "--ready-feedback"],
    ],
)
def test_should_refuse_cli_conflicts_before_reader_or_delegation(monkeypatch, arguments):
    before = dict(WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda path: pytest.fail("reader reached"))
    monkeypatch.setattr(sys, "argv", ["affine", "--acknowledge-correct", "--describe", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and WORK == before


def test_should_preserve_false_default_waiting_schema_and_pending():
    r = cli.run_context_linear_stream(
        sources(), reference_policy=ReferencePolicy("drift"), ready_feedback=True
    )
    assert r["status"] == "complete" and r["stream"]["completed_updates"] == 0
    assert r["stream"]["pending_unapplied"] == 6 and "acknowledgement" not in r["stream"]
    assert "acknowledge_correct" not in r["protocol"]


@pytest.mark.parametrize("fault", ["lost", "late"])
def test_should_keep_paid_failed_proof_and_original_budget(monkeypatch, fault):
    now = [0.0]
    monkeypatch.setattr(cli, "monotonic", lambda: now[0])
    calls = []

    def fail(owner, features):
        calls.append(1)
        if len(calls) == 2:
            if fault == "lost":
                raise RuntimeError("lost proof")
            now[0] = 91.0

    monkeypatch.setattr(Literal, "hook", staticmethod(fail))
    r = run()
    assert r["status"] == "failed" and r["budget"]["updates_completed"] == 0
    assert (
        r["budget"]["started_at"] == 0.0 and r["budget"]["initial_limits"] == r["budget"]["limits"]
    )
    proof = r["stream"]["acknowledgement"]["proof"]
    assert proof["prediction_calls_attempted"] == 1
    assert proof["prediction_calls_completed"] == int(fault != "lost")
    assert not r["stream"]["acknowledged_experiences"]


@pytest.mark.parametrize("damage", ["value", "receipt", "ledger"])
def test_should_refuse_changed_prior_ack_at_public_API_and_keep_paid_prefix(monkeypatch, damage):
    original = AttributionStream.run_reference_window
    saved: dict[str, Any] = {}

    def capture(owner, *args, **kwargs):
        saved["stream"] = owner
        return original(owner, *args, **kwargs)

    monkeypatch.setattr(AttributionStream, "run_reference_window", capture)
    damaged: list[bool] = []

    def change(owner, features):
        ledger = getattr(saved["stream"]._runtime._inbox, "_acknowledged", {})
        if ledger and not damaged:
            damaged.append(True)
            key = next(iter(ledger))
            if damage == "value":
                object.__setattr__(ledger[key], "acknowledged_at", 99)
            elif damage == "receipt":
                ledger[key] = deepcopy(ledger[key])
            else:
                saved["stream"]._runtime._inbox._acknowledged = dict(ledger)

    monkeypatch.setattr(Literal, "hook", staticmethod(change))
    r = run()
    assert damaged, r.get("reason")
    assert r["status"] == "failed", r.get("reason")
    assert r["budget"]["updates_completed"] == 0
    assert len(r["stream"]["known_acknowledgements"]) == 1
    assert r["stream"]["known_acknowledgements"][0]["acknowledged_at"] == 1


def test_should_combine_arrived_shift_diagnostics_with_actual_ACK_consumption():
    r = run(shift_diagnostics=True)
    assert r["status"] == "complete", r.get("reason")
    assert r["stream"]["processed_packets"] == 6
    assert r["stream"]["arrived_observation"]["calls_completed"] == 6
    assert r["shift_diagnostics"] is not None


def test_should_describe_actual_CLI_without_reader_or_models(monkeypatch, capsys):
    before = dict(WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda path: pytest.fail("reader reached"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "affine",
            "--context-linear-reference",
            "drift",
            "--ready-feedback",
            "--acknowledge-correct",
            "--reference-retain-arrived",
            "--describe",
        ],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["diagnostic_prediction_rows_upper_bound"] == 216
    assert WORK == before


def test_should_bound_failed_joint_fit_after_ACK_skips_without_fabricating_commits(monkeypatch):
    def fail(owner, features, targets):
        WORK["updates"] += 1
        WORK["rows"] += len(features)
        assert len(features) == 18
        raise RuntimeError("lost joint fit return")

    monkeypatch.setattr(Literal, "train_batch", fail)
    r = run(retain_arrived=True)
    assert r["status"] == "failed"
    assert len(r["stream"]["known_acknowledgements"]) == 2
    assert r["stream"]["attempted_wakes"] == 1
    assert r["catalogue"]["committed_rows"] == dict(arrivals=0, replay=0, native=0)
    assert r["catalogue"]["attempted_rows_upper_bound"] == dict(arrivals=9, replay=9, native=18)
    assert r["catalogue"]["failed_attempt_replay_rows_upper_bound"] == 9
    assert r["catalogue"]["failure_work_may_be_unobserved"] is True


def test_should_save_failed_CLI_proof_cost_before_exit_two(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "read_training_sources", lambda path: sources())
    calls = []

    def fail(owner, features):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("lost CLI proof")

    monkeypatch.setattr(Literal, "hook", staticmethod(fail))
    path = tmp_path / "failed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "affine",
            "--context-linear-reference",
            "drift",
            "--ready-feedback",
            "--acknowledge-correct",
            "--source",
            "literal.json",
            "--output",
            str(path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    saved = json.loads(path.read_text())
    assert saved == json.loads(capsys.readouterr().out)
    assert saved["status"] == "failed"
    proof = saved["stream"]["acknowledgement"]["proof"]
    assert proof["prediction_calls_attempted"] == 1 and proof["prediction_calls_completed"] == 0
