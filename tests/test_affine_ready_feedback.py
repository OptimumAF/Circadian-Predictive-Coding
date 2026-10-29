"""FIFO-aligned arrived-label feedback through real stream and CLI, literal models."""

from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.reference_controller import ReferencePolicy
from tests.test_affine_drift_feedback import make
from tests.test_affine_drift_cli import allow_explicit_diagnostics as allow_explicit_diagnostics
from tests.test_affine_reference_cli import (
    WORK as CLI_WORK,
    invoke_cli,
    scripted_only as scripted_only,
)
from tests.test_arrived_cue_catalogue import sources


def run(owner, mode="drift"):
    return owner.run_reference_window(ReferencePolicy(mode), ready_feedback=True)


def test_should_score_current_oldest_fifo_packet_then_consume_that_exact_packet():
    owner, source, budget, _ = make()
    initial = source.snapshot_state(), owner._runtime.actor.snapshot_state()
    result = run(owner)
    assert source.calls == [
        entry for i in range(4) for entry in [("predict", i, i), ("train", i, i)]
    ]
    rows = result["ready_feedback"]["events"]
    assert [
        (e["batch_index"], e["observed_at"], e["arrived_at"], e["scored_at"], e["wake_updates"])
        for e in rows
    ] == [(i, i, i + 1, i + 1, i) for i in range(4)]
    assert all(e["eligible"] and e["feedback_source"] == "oldest_ready" for e in rows)
    assert [r["sample_id"] for r in result["applied_updates"]] == [f"batch-{i}" for i in range(4)]
    assert (
        result["completed_updates"] == budget.updates_completed == 4
        and result["pending_unapplied"] == 0
    )
    assert result["training_complete"] and result["score_forward_calls"] == 0
    assert (source.snapshot_state(), owner._runtime.actor.snapshot_state()) == initial
    assert "prequential" not in result and result["ready_feedback"]["pending_packets"] == 0
    assert result["ready_feedback"]["peak_pending_rows"] == 1


def test_should_reinspect_correct_oldest_packet_without_skipping_it_to_force_learning():
    owner, source, budget, _ = make(target=0.0)
    result = run(owner)
    assert source.calls == [("predict", 0, 0)] * 4 and budget.updates_completed == 0
    rows = result["ready_feedback"]["events"]
    assert [e["batch_index"] for e in rows] == [0] * 4
    assert [e["arrived_at"] for e in rows] == [1] * 4
    assert [e["scored_at"] for e in rows] == [1, 2, 3, 4]
    assert result["pending_unapplied"] == 4


@pytest.mark.parametrize("delay", [0, 2, 5])
def test_should_wait_for_each_original_label_including_tail_opportunities(delay):
    owner, source, _, _ = make(delay=delay)
    result = run(owner)
    assert [r["applied_at"] for r in result["applied_updates"]] == list(range(delay, delay + 4))
    assert [e["scored_at"] for e in result["ready_feedback"]["events"]] == list(
        range(delay, delay + 4)
    )
    assert [i for kind, i, _ in source.calls if kind == "predict"] == [0, 1, 2, 3]


@pytest.mark.parametrize("mode", ["no_learning", "periodic", "budget_aware", "drift"])
def test_should_record_paid_diagnostic_opportunities_for_each_fixed_control(mode):
    owner, _, _, _ = make()
    result = run(owner, mode)
    d = result["ready_feedback"]
    assert d["prediction_calls_completed"] == d["fork_completed"] == 4
    assert d["prediction_rows_completed"] == 4 and len(d["events"]) == 4
    assert not d["unreturned_prediction_unknown"] and not d["unreturned_fork_unknown"]


def test_should_not_renew_credit_while_diagnosing_ready_backlog():
    owner, _, budget, _ = make(credit=1)
    start = budget.started_at
    result = run(owner)
    assert budget.updates_completed == 1 and budget.started_at == start
    assert result["ready_feedback"]["prediction_calls_completed"] == 4
    assert result["pending_unapplied"] == 3 and result["completed_updates"] == 1


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_should_refuse_nonboolean_ready_option_before_any_work(flag):
    owner, source, _, _ = make()
    with pytest.raises(ValueError):
        owner.run_reference_window(ReferencePolicy("drift"), ready_feedback=flag)
    assert not source.calls and owner.next_tick == 0


def test_should_refuse_mixed_feedback_modes_before_work():
    owner, source, _, _ = make()
    with pytest.raises(ValueError):
        owner.run_reference_window(
            ReferencePolicy("drift"), ready_feedback=True, prequential_feedback=True
        )
    assert not source.calls and owner.reference_report is None


@pytest.mark.parametrize("failure", ["lost", "late", "mutate"])
def test_should_preserve_failed_diagnostic_cost_and_refuse_more_work(failure):
    owner, source, budget, now = make()

    def predict(model, features):
        if failure == "lost":
            raise RuntimeError("lost ready diagnostic")
        if failure == "late":
            now[0] = 61.0
        else:
            model.state += 99
        return np.zeros((1, 1))

    source.hooks["predict"] = predict
    with pytest.raises((RuntimeError, ValueError, ToyExecutionStopped)):
        run(owner)
    result = owner.reference_report
    assert result["status"] == "failed" and budget.updates_completed == 0
    d = result["ready_feedback"]
    assert d["prediction_calls_attempted"] == 1
    assert d["unreturned_prediction_unknown"] is (failure == "lost")
    with pytest.raises(ValueError, match="failed or finished"):
        run(owner)


def test_should_bound_worst_case_repeated_diagnostic_rows_before_prediction():
    # Real stream admission accepts the original<=4096total rows, but ready
    # feedback must budget repeated diagnosis of a large unconsumed first row.
    from tests.test_affine_drift_feedback import Learner

    source = Learner()
    batches = tuple(
        AttributionBatch(
            "train", "A", tuple(f"r-{i}-{j}" for j in range(n)), np.zeros((n, 1)), np.zeros((n, 1))
        )
        for i, n in enumerate([1024, 1, 1, 1, 1])
    )
    owner = AttributionStream(
        source,
        batches,
        (),
        checkpoint_updates=tuple(range(6)),
        label_delay=1,
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=5), lambda: 0.0),
        training_only=True,
    )
    with pytest.raises(ValueError, match="worst-case"):
        run(owner)
    assert not source.calls and owner.reference_report is None


def test_should_refuse_future_or_nonfifo_diagnostic_requests_before_prediction():
    from src.app.reference_prequential import ReferencePrequential

    owner, source, _, _ = make()
    with pytest.raises(ValueError, match="oldest eligible"):
        owner._predict_reference_packet(1, ReferencePrequential(owner._training), scored_at=0)
    assert not source.calls


def test_should_refuse_probe_role_before_ready_diagnostic():
    owner, source, _, _ = make()
    owner._training_role = "probe_train"
    with pytest.raises(ValueError):
        run(owner)
    assert not source.calls and owner.reference_report is None


def test_should_preserve_frozen_prequential_wire_semantics():
    owner, _, _, _ = make()
    result = owner.run_reference_window(ReferencePolicy("drift"), prequential_feedback=True)
    assert "ready_feedback" not in result and result["completed_updates"] == 2
    assert [e["eligible"] for e in result["prequential"]["events"]] == [True, False, True, False]
    assert all("scored_at" not in e for e in result["prequential"]["events"])


@pytest.mark.parametrize("when", ["fork", "predict"])
def test_should_refuse_logical_clock_change_before_ready_vote_or_training(when):
    from tests.test_affine_drift_feedback import Learner

    owner, source, budget, _ = make()
    if when == "fork":

        def fork(model):
            owner._clock.advance_to(99)
            return Learner(model.state, source.hooks, source.calls)

        source.hooks["fork"] = fork
    else:

        def predict(model, features):
            owner._clock.advance_to(99)
            return np.zeros((1, 1))

        source.hooks["predict"] = predict
    with pytest.raises(ValueError, match="logical clock changed"):
        run(owner)
    assert budget.updates_completed == 0
    assert owner.reference_report["ready_feedback"]["prediction_calls_attempted"] == (
        0 if when == "fork" else 1
    )


@pytest.mark.parametrize("mode", ["no_learning", "periodic", "drift"])
def test_should_forward_ready_mode_through_actual_cli_and_keep_episode_receipts(
    monkeypatch, tmp_path, capsys, mode
):
    packets = sources()
    before = deepcopy(packets)
    monkeypatch.setattr(cli, "read_training_sources", lambda path: packets)
    output = tmp_path / "ready.json"
    invoke_cli(
        monkeypatch,
        [
            "--context-linear-reference",
            mode,
            "--ready-feedback",
            "--source",
            "literal.json",
            "--output",
            str(output),
        ],
    )
    result = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == result and packets == before
    assert result["status"] == "complete" and result["protocol"]["ready_feedback"] is True
    d = result["stream"]["ready_feedback"]
    assert d["prediction_calls_completed"] == 6 and d["prediction_rows_completed"] == 54
    assert (
        result["score_calls_completed"] == 0
        and result["endpoints"] == []
        and result["source_unchanged"]
        and result["actor_unchanged"]
    )
    assert len(result["committed_receipts"]) == result["budget"]["updates_completed"]
    if mode == "drift":
        assert result["stream"]["training_complete"] and result["budget"]["updates_completed"] == 6
    assert CLI_WORK["forbidden"] == 0


def test_should_describe_ready_upper_bounds_before_source_or_owner(monkeypatch, capsys):
    before = dict(CLI_WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda path: pytest.fail("source accessed"))
    invoke_cli(
        monkeypatch, ["--context-linear-reference", "drift", "--ready-feedback", "--describe"]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["ready_feedback"] and result["diagnostic_prediction_calls_upper_bound"] == 6
    assert (
        result["diagnostic_prediction_rows_upper_bound"] == 54
        and "prediction_calls_completed" not in result
    )
    assert CLI_WORK == before


@pytest.mark.parametrize(
    "arguments",
    [
        ["--ready-feedback", "--describe"],
        [
            "--context-linear-reference",
            "drift",
            "--ready-feedback",
            "--prequential-feedback",
            "--describe",
        ],
        ["--example-affine-published", "--ready-feedback", "--describe"],
    ],
)
def test_should_refuse_ready_cli_conflicts_before_source_and_delegation(monkeypatch, arguments):
    before = dict(CLI_WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda path: pytest.fail("source accessed"))
    with pytest.raises(SystemExit) as caught:
        invoke_cli(monkeypatch, arguments)
    assert caught.value.code == 2 and CLI_WORK == before
