"""Actual CLI/wrapper/delayed stream with scripted learners; no genuine fitting."""

from dataclasses import dataclass
from copy import deepcopy
import json
import sys
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_context_linear_learner as port
from src.app import coupled_attribution as stream_app
from src.core.learner_ports import TrainingDiagnostic
from src.core.reference_controller import ReferencePolicy
from tests.test_arrived_cue_catalogue import sources

WORK = dict(owners=0, updates=0, rows=0, predictions=0, prediction_rows=0, finishes=0, forbidden=0)
HOOKS: dict[str, Any] = {}
ORIGINAL_FINISH = stream_app.AttributionStream.finish


@dataclass(frozen=True)
class LiteralState:
    completed_updates: int = 0
    training_examples: int = 0


class LiteralAffine:
    def __init__(self, state=None, trace=None):
        WORK["owners"] += 1
        assert WORK["owners"] <= 128
        self.state = LiteralState() if state is None else state
        self.trace = [] if trace is None else trace

    def fork(self):
        return LiteralAffine(self.state, self.trace)

    def snapshot_state(self):
        return self.state

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, x, y):
        WORK["updates"] += 1
        WORK["rows"] += len(x)
        assert WORK["updates"] <= 128 and WORK["rows"] <= 1152
        assert x.shape == (9, 296) and y.shape == (9, 1)
        self.trace.append((x.copy(), y.copy()))
        self.state = LiteralState(
            self.state.completed_updates + 1, self.state.training_examples + len(x)
        )
        if "train" in HOOKS:
            HOOKS["train"](self)
        return TrainingDiagnostic("scripted_affine", float(self.state.completed_updates))

    def predict(self, x):
        WORK["predictions"] += 1
        WORK["prediction_rows"] += len(x)
        assert WORK["predictions"] <= 128 and WORK["prediction_rows"] <= 1152
        if not HOOKS.get("default"):
            WORK["forbidden"] += 1
            raise AssertionError("new reference CLI must not predict")
        return np.full((len(x), 7), 1 / 7, dtype=np.float64)


@pytest.fixture(autouse=True)
def scripted_only(monkeypatch):
    HOOKS.clear()
    monkeypatch.setattr(port, "ContextLinearLearner", LiteralAffine)

    def forbid(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("genuine solver/model/game execution forbidden")

    def finish(owner, *args, **kwargs):
        WORK["finishes"] += 1
        if not HOOKS.get("default"):
            return forbid()
        return ORIGINAL_FINISH(owner, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", forbid)
    monkeypatch.setattr(cli, "diagnose_train_linear_fit", forbid)
    monkeypatch.setattr(cli, "diagnose_context_train_linear_fit", forbid)
    monkeypatch.setattr(cli, "diagnose_train_fit", forbid)
    monkeypatch.setattr(cli.context_control, "run_control", forbid)
    monkeypatch.setattr(stream_app.AttributionStream, "finish", finish)
    yield
    HOOKS.clear()


def invoke_cli(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["affine", *arguments])
    cli.main()


@pytest.mark.parametrize(
    "mode,count,ticks",
    [
        ("no_learning", 0, []),
        ("periodic", 3, [2, 4, 6]),
        ("budget_aware", 2, [5, 6]),
    ],
)
def test_should_execute_actual_cli_stream_and_save_honest_pending_metadata(
    monkeypatch, tmp_path, capsys, mode, count, ticks
):
    packets = sources()
    before = deepcopy(packets)
    monkeypatch.setattr(cli, "read_training_sources", lambda path: packets)
    output = tmp_path / "reference.json"
    invoke_cli(
        monkeypatch,
        ["--context-linear-reference", mode, "--source", "literal.json", "--output", str(output)],
    )
    result = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert packets == before and result["status"] == "complete"
    window = result["stream"]
    assert window["window_complete"] and window["training_complete"] is (count == 6)
    assert window["pending_unapplied"] == 6 - count and window["training_examples"] == count * 9
    assert [r["applied_at"] for r in window["applied_updates"]] == ticks
    assert [r["observed_at"] for r in window["applied_updates"]] == list(range(count))
    assert [r["arrived_at"] for r in window["applied_updates"]] == list(range(1, count + 1))
    assert window["attempted_wakes"] == window["completed_updates"] == count
    assert result["budget"]["updates_completed"] == count
    assert len(result["committed_receipts"]) == count and result["receipt_capture_error"] is None
    assert result["endpoints"] == result["unscored_checkpoints"] == []
    assert result["score_calls_attempted"] == result["score_calls_completed"] == 0
    assert result["source_unchanged"] and result["actor_unchanged"]
    assert result["protocol"] == json.loads(
        json.dumps(cli.describe_context_linear_reference(ReferencePolicy(mode)))
    )
    assert output.stat().st_size < 256 * 1024


@pytest.mark.parametrize("mode", ["no_learning", "periodic", "budget_aware"])
def test_should_describe_real_cli_mode_before_any_source_or_model(monkeypatch, capsys, mode):
    before = dict(WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source reached"))
    invoke_cli(monkeypatch, ["--context-linear-reference", mode, "--describe"])
    result = json.loads(capsys.readouterr().out)
    assert WORK == before and result["reference_policy"] == cli.asdict(ReferencePolicy(mode))
    assert result["window_opportunities"] == 7 and result["arrival_rows"] == 54
    assert result["checkpoint_updates"] == list(range(7))
    assert result["fit_calls_upper_bound"] == 6 and result["fit_rows_upper_bound"] == 54
    assert "fit_calls" not in result and "fit_rows" not in result
    assert result["score_calls"] == result["score_rows"] == result["game_episodes"] == 0


@pytest.mark.parametrize("policy", [False, object(), "periodic", ReferencePolicy("drift")])
def test_should_refuse_unsupported_api_policy_before_payload_or_owner(policy):
    before = dict(WORK)
    with pytest.raises(ValueError, match="exact supported"):
        cli.run_context_linear_stream(object(), reference_policy=policy)
    assert WORK == before


@pytest.mark.parametrize(
    "options",
    [
        {"abc_catalogue": True},
        {"recurring_catalogue": True},
        {"a4_training": True},
        {"a4_current_only": True},
        {"endpoint_observer": lambda state: None},
    ],
)
def test_should_refuse_foreign_recipe_or_endpoint_handoff_before_work(options):
    before = dict(WORK)
    with pytest.raises(ValueError, match="distinct current-only"):
        cli.run_context_linear_stream(
            object(), reference_policy=ReferencePolicy("periodic"), **options
        )
    assert WORK == before


@pytest.mark.parametrize(
    "arguments",
    [
        ["--context-linear-reference", "drift", "--describe"],
        ["--context-linear-reference", "periodic", "--context-linear-stream", "--describe"],
        ["--context-linear-reference", "periodic", "--example-affine-play", "--describe"],
        ["--context-linear-reference", "periodic", "--fit-updates", "2", "--describe"],
        [
            "--context-linear-reference",
            "periodic",
            "--retained-source",
            "foreign.json",
            "--describe",
        ],
        ["--context-linear-reference", "periodic", "--describe", "--source", "foreign.json"],
        ["--context-linear-reference", "periodic", "--describe", "--output", "foreign.json"],
        ["--context-linear-reference", "periodic"],
    ],
)
def test_should_refuse_cli_conflicts_or_missing_source_before_access(monkeypatch, arguments):
    before = dict(WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source reached"))
    with pytest.raises(SystemExit) as raised:
        invoke_cli(monkeypatch, arguments)
    assert raised.value.code == 2 and WORK == before


def test_should_refuse_existing_output_before_source_or_training(monkeypatch, tmp_path):
    output = tmp_path / "keep.json"
    output.write_text("original")
    before = dict(WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source reached"))
    with pytest.raises(FileExistsError):
        invoke_cli(
            monkeypatch,
            [
                "--context-linear-reference",
                "periodic",
                "--source",
                "literal",
                "--output",
                str(output),
            ],
        )
    assert output.read_text() == "original" and WORK == before


@pytest.mark.parametrize("failure", ["lost_return", "late_wall"])
def test_should_save_failed_cli_prefix_and_preserve_original_attempt_costs(
    monkeypatch, tmp_path, capsys, failure
):
    now = [0.0]
    monkeypatch.setattr(cli, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli, "read_training_sources", lambda path: sources())

    def fail(owner):
        if failure == "lost_return":
            raise RuntimeError("lost scripted return")
        now[0] = 91

    HOOKS["train"] = fail
    output = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as raised:
        invoke_cli(
            monkeypatch,
            [
                "--context-linear-reference",
                "periodic",
                "--source",
                "literal",
                "--output",
                str(output),
            ],
        )
    assert raised.value.code == 2
    result = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == result and result["status"] == "failed"
    window = result["stream"]
    assert window["status"] == "failed" and not window["window_complete"]
    assert window["attempted_wakes"] == 1
    expected = 0 if failure == "lost_return" else 1
    assert window["unreturned_work_unknown"] is (failure == "lost_return")
    assert len(window["applied_updates"]) == result["budget"]["updates_completed"] == expected
    if failure == "late_wall":
        # The known inner receipt survives; the strict outer capture refuses
        # the loop counter that did not return from the late-stopped call.
        assert result["committed_receipts"] is None and result["receipt_capture_error"]
    else:
        assert result["committed_receipts"] == [] and result["receipt_capture_error"] is None
    assert result["budget"]["started_at"] == 0 and result["endpoints"] == []


def test_should_refuse_policy_changed_at_initial_outer_callback_before_payload():
    policy = ReferencePolicy("periodic")
    before = dict(WORK)

    def change():
        object.__setattr__(policy, "period", 1)

    with pytest.raises(ValueError, match="policy changed"):
        cli.run_context_linear_stream(object(), reference_policy=policy, check_boundary=change)
    assert WORK == before


def test_should_not_report_success_when_final_receipt_capture_fails(monkeypatch):
    class BrokenCapture(stream_app.AttributionStream):
        @property
        def applied_updates(self):
            raise ValueError("receipt transport failed")

    monkeypatch.setattr(stream_app, "AttributionStream", BrokenCapture)
    result = cli.run_context_linear_stream(sources(), reference_policy=ReferencePolicy("periodic"))
    assert result["stream"]["window_complete"]
    assert result["status"] == "failed" and result["committed_receipts"] is None
    assert result["receipt_capture_error"] and result["budget"]["updates_completed"] == 3


def test_should_preserve_old_none_policy_recipe_scores_and_wire():
    HOOKS["default"] = True
    result = cli.run_context_linear_stream(sources(), reference_policy=None)
    assert result["status"] == "complete", result.get("reason")
    assert result["protocol"] == cli.describe_context_linear_stream()
    assert result["stream"]["completed_updates"] == 6 and "window_complete" not in result["stream"]
    assert [r["updates"] for r in result["endpoints"]] == [0, 2, 4, 6]
    assert result["score_calls_completed"] == 16 and result["score_rows_completed"] == 144
    assert len(result["committed_receipts"]) == result["budget"]["updates_completed"] == 6


def test_should_stop_changed_policy_after_the_original_paid_update():
    policy = ReferencePolicy("periodic")

    def change(owner):
        object.__setattr__(policy, "period", 1)

    HOOKS["train"] = change
    result = cli.run_context_linear_stream(sources(), reference_policy=policy)
    assert result["status"] == "failed" and "policy changed" in result["reason"]
    assert result["stream"]["attempted_wakes"] == 1
    assert len(result["committed_receipts"]) == result["budget"]["updates_completed"] == 1


def test_should_refuse_oversized_cli_metadata_without_writing_output(monkeypatch, tmp_path):
    before = dict(WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: ())
    monkeypatch.setattr(
        cli,
        "run_context_linear_stream",
        lambda *a, **k: dict(status="complete", padding="x" * (256 * 1024)),
    )
    output = tmp_path / "too-large.json"
    with pytest.raises(ValueError, match="reference report exceeds256KiB"):
        invoke_cli(
            monkeypatch,
            [
                "--context-linear-reference",
                "periodic",
                "--source",
                "literal",
                "--output",
                str(output),
            ],
        )
    assert not output.exists() and WORK == before
