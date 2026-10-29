"""Existing joint TRAIN composition and real reference runtime, literal learners."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import pickle
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_context_linear_learner as port
from src.app import coupled_attribution
from src.core.demonstration_episode import episode_arrays
from src.core.learner_ports import TrainingDiagnostic
from src.core.reference_controller import ReferencePolicy
from tests.test_affine_reference_cli import invoke_cli
from tests.test_arrived_cue_catalogue import sources

WORK = dict(owners=0, fits=0, fit_rows=0, predicts=0, predict_rows=0, snapshots=0, forbidden=0)
HOOKS: dict[str, Any] = {}
TRACE: list[Any] = []


@dataclass(frozen=True)
class LiteralState:
    completed_updates: int = 0
    training_examples: int = 0


class LiteralAffine:
    def __init__(self, state=None):
        WORK["owners"] += 1
        self.state = LiteralState() if state is None else state

    def fork(self):
        return LiteralAffine(self.state)

    def snapshot_state(self):
        WORK["snapshots"] += 1
        return self.state

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, x, y):
        WORK["fits"] += 1
        WORK["fit_rows"] += len(x)
        TRACE.append((x.copy(), y.copy()))
        if "train" in HOOKS:
            HOOKS["train"](self, x, y)
        self.state = LiteralState(
            self.state.completed_updates + 1, self.state.training_examples + len(x)
        )
        return TrainingDiagnostic("literal_joint_fit", float(len(x)))

    def predict(self, x):
        WORK["predicts"] += 1
        WORK["predict_rows"] += len(x)
        if "predict" in HOOKS:
            HOOKS["predict"](self, x)
        result = np.zeros((len(x), 7), dtype=np.float64)
        result[:, HOOKS.get("class", 0)] = 1
        return result


@pytest.fixture(autouse=True)
def literal_only(monkeypatch):
    HOOKS.clear()
    TRACE.clear()
    monkeypatch.setattr(port, "ContextLinearLearner", LiteralAffine)

    def forbid(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("real solver/evaluation/finish forbidden")

    monkeypatch.setattr(np.linalg, "lstsq", forbid)
    monkeypatch.setattr(coupled_attribution.AttributionStream, "finish", forbid)
    yield
    HOOKS.clear()
    TRACE.clear()


def run(mode="drift", **options):
    return cli.run_context_linear_stream(
        sources(),
        retain_arrived=True,
        reference_policy=ReferencePolicy(mode),
        ready_feedback=True,
        **options,
    )


@pytest.mark.parametrize(
    "mode,fits,rows,replay,diagnostic_rows",
    [
        ("no_learning", 0, 0, 0, 54),
        ("periodic", 3, 36, 9, 72),
        ("budget_aware", 2, 18, 0, 54),
        ("drift", 6, 90, 36, 90),
    ],
)
def test_should_count_only_consumed_fifo_exposures_separately_from_planned_storage(
    mode, fits, rows, replay, diagnostic_rows
):
    before = dict(WORK)
    result = run(mode)
    assert result["status"] == "complete"
    window, bank = result["stream"], result["catalogue"]
    assert bank["planned"]["fit_row_exposures"] == 90
    assert bank["planned"]["arrival_row_exposures"] == 54
    assert bank["planned"]["replay_row_exposures"] == 36
    assert bank["planned"]["input_array_bytes"] == 213840
    assert bank["planned"]["peak_bank_rows"] == 18
    assert bank["committed_rows"] == dict(arrivals=fits * 9, replay=replay, native=rows)
    assert bank["attempted_rows_upper_bound"] == bank["committed_rows"]
    assert not bank["failure_work_may_be_unobserved"]
    assert window["training_examples"] == rows and window["completed_updates"] == fits
    assert window["pending_unapplied"] == 6 - fits and window["training_complete"] is (fits == 6)
    assert WORK["fits"] - before["fits"] == fits and WORK["fit_rows"] - before["fit_rows"] == rows
    assert window["ready_feedback"]["prediction_rows_completed"] == diagnostic_rows
    assert window["ready_feedback"]["prediction_calls_completed"] == 6
    assert window["score_forward_calls"] == result["score_calls_completed"] == 0
    assert result["endpoints"] == result["unscored_checkpoints"] == []
    assert result["source_unchanged"] and result["actor_unchanged"]
    assert result["budget"]["updates_completed"] == fits
    assert (
        result["budget"]["replay_examples_completed"] == 0
    )  # Within-wake rows, not consolidation.
    assert result["budget"]["initial_limits"] == result["budget"]["limits"]
    assert len(result["committed_receipts"]) == fits and result["receipt_capture_error"] is None


def test_should_fit_only_same_cue_previous_arrived_sources_and_preserve_original_labels():
    packets = sources()
    before = pickle.dumps((packets, np.random.get_state()))
    result = cli.run_context_linear_stream(
        packets, retain_arrived=True, reference_policy=ReferencePolicy("drift"), ready_feedback=True
    )
    expected = (0, 1, 2, 3, 0, 1)
    wakes = result["catalogue"]["planned"]["wakes"]
    assert [len(x) for x, y in TRACE] == [9, 9, 18, 18, 18, 18]
    for i, ((x, y), wake) in enumerate(zip(TRACE, wakes, strict=True)):
        indices = [expected[i]] if i < 2 else [expected[i - 2], expected[i]]
        arrays = [episode_arrays(packets[k]) for k in indices]
        np.testing.assert_array_equal(x, np.concatenate([a for a, b in arrays]))
        np.testing.assert_array_equal(y, np.concatenate([b for a, b in arrays]))
        assert all(t <= i for t in wake["retained_arrival_ticks"])
        assert wake["observed_at"] == i and wake["arrived_at"] == i + 1
        receipt = result["committed_receipts"][i]
        assert (
            receipt["sample_id"],
            receipt["observed_at"],
            receipt["arrived_at"],
            receipt["applied_at"],
        ) == (f"batch-{i}", i, i + 1, i + 1)
    assert pickle.dumps((packets, np.random.get_state())) == before


def test_should_keep_correct_oldest_joint_packet_queued_without_skipping_to_new_context():
    HOOKS["class"] = 2
    result = run()
    assert result["stream"]["completed_updates"] == 0 and result["stream"]["pending_unapplied"] == 6
    assert result["catalogue"]["committed_rows"] == dict(arrivals=0, replay=0, native=0)
    assert [e["batch_index"] for e in result["stream"]["ready_feedback"]["events"]] == [0] * 6
    assert not TRACE


def test_should_preserve_prequential_candidate_retirement_with_joint_batches():
    result = cli.run_context_linear_stream(
        sources(),
        retain_arrived=True,
        reference_policy=ReferencePolicy("drift"),
        prequential_feedback=True,
    )
    window = result["stream"]
    assert result["status"] == "complete" and "ready_feedback" not in window
    assert [e["eligible"] for e in window["prequential"]["events"]] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert window["prequential"]["prediction_rows_completed"] == 90
    assert result["catalogue"]["committed_rows"] == dict(arrivals=27, replay=9, native=36)


@pytest.mark.parametrize("failure", ["lost", "late"])
def test_should_keep_failed_attempt_exposure_bound_and_original_success_charge(
    monkeypatch, failure
):
    clock = [0.0]
    monkeypatch.setattr(cli, "monotonic", lambda: clock[0])

    def fail(model, x, y):
        if len(TRACE) == 3:
            if failure == "lost":
                raise RuntimeError("lost joint fit")
            clock[0] = 91.0

    HOOKS["train"] = fail
    result = run()
    assert result["status"] == "failed"
    assert result["stream"]["attempted_wakes"] == 3
    assert result["catalogue"]["attempted_rows_upper_bound"] == dict(
        arrivals=27, replay=9, native=36
    )
    assert result["catalogue"]["failure_work_may_be_unobserved"]
    # Lost/late calls are not free. Strict outer receipt capture may be unavailable.
    assert result["budget"]["updates_completed"] >= 2
    known = result["catalogue"]["committed_rows"]
    assert known is None or known["native"] >= 18
    assert len(TRACE) == 3


def test_should_refuse_source_corruption_before_admitting_later_joint_work():
    packets = sources()

    def corrupt(model, x, y):
        object.__setattr__(packets[0].transitions[0], "action", 6)

    HOOKS["train"] = corrupt
    result = cli.run_context_linear_stream(
        packets, retain_arrived=True, reference_policy=ReferencePolicy("drift"), ready_feedback=True
    )
    assert result["status"] == "failed" and len(TRACE) == 1


@pytest.mark.parametrize("fault", ["wrong_id", "missing", "truncated", "changed_version"])
def test_should_not_report_committed_rows_from_corrupt_or_missing_receipt_capture(
    monkeypatch, fault
):
    original = vars(coupled_attribution.AttributionStream)["applied_updates"].fget

    def capture(owner):
        receipts = original(owner)
        if fault == "missing":
            raise ValueError("unobserved outer receipt capture")
        if fault == "truncated":
            return receipts[:-1]
        if fault == "changed_version":
            return (replace(receipts[0], learner_version="replaced"), *receipts[1:])
        return (replace(receipts[0], sample_id="replaced"), *receipts[1:])

    monkeypatch.setattr(coupled_attribution.AttributionStream, "applied_updates", property(capture))
    result = run()
    assert result["status"] == "failed" and result["committed_receipts"] is None
    assert result["catalogue"]["committed_rows"] is None
    assert result["catalogue"]["attempted_rows_upper_bound"] == dict(
        arrivals=54, replay=36, native=90
    )
    assert result["catalogue"]["failure_work_may_be_unobserved"]
    assert result["budget"]["updates_completed"] == 6 and len(TRACE) == 6


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_nontrain_sources_before_learner_or_prediction(role):
    packets = sources()
    packets = (replace(packets[0], role=role), *packets[1:])
    before = dict(WORK)
    with pytest.raises(ValueError):
        cli.run_context_linear_stream(
            packets, retain_arrived=True, reference_policy=ReferencePolicy("periodic")
        )
    assert WORK == before


def test_should_forward_optin_through_actual_cli_and_keep_exclusive_output(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "read_training_sources", lambda path: sources())
    output = tmp_path / "joint.json"
    args = [
        "--context-linear-reference",
        "drift",
        "--ready-feedback",
        "--reference-retain-arrived",
        "--source",
        "literal.json",
        "--output",
        str(output),
    ]
    invoke_cli(monkeypatch, args)
    result = json.loads(output.read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["catalogue"]["committed_rows"] == dict(arrivals=54, replay=36, native=90)
    assert result["protocol"]["retain_arrived"] is True
    before = dict(WORK)
    with pytest.raises(FileExistsError):
        invoke_cli(monkeypatch, args)
    assert WORK == before


def test_should_describe_retained_bounds_without_source_or_learner(monkeypatch, capsys):
    monkeypatch.setattr(cli, "read_training_sources", lambda *args: pytest.fail("source accessed"))
    before = dict(WORK)
    invoke_cli(
        monkeypatch,
        [
            "--context-linear-reference",
            "drift",
            "--ready-feedback",
            "--reference-retain-arrived",
            "--describe",
        ],
    )
    protocol = json.loads(capsys.readouterr().out)
    assert protocol["fit_rows_upper_bound"] == 90 and protocol["arrival_rows"] == 54
    assert (
        protocol["replay_rows_upper_bound"] == 36
        and protocol["diagnostic_prediction_rows_upper_bound"] == 108
    )
    assert WORK == before


@pytest.mark.parametrize(
    "args",
    [
        ["--reference-retain-arrived", "--describe"],
        ["--example-affine-a4", "--reference-retain-arrived", "--describe"],
    ],
)
def test_should_refuse_cli_replay_option_without_reference_before_delegation(monkeypatch, args):
    before = dict(WORK)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source accessed"))
    with pytest.raises(SystemExit) as error:
        invoke_cli(monkeypatch, args)
    assert error.value.code == 2 and WORK == before


@pytest.mark.parametrize("value", [1, None, "true"])
def test_should_refuse_nonboolean_reference_retention_before_source_or_owner(value):
    before = dict(WORK)
    with pytest.raises(ValueError):
        cli.run_context_linear_stream(
            object(), retain_arrived=value, reference_policy=ReferencePolicy("periodic")
        )
    assert WORK == before
