"""Cold observed-C baseline, with scalar controls and one tiny native fork.

The accepted warm JSON is read only as bounded reference data by the executor.
No released model, tokenizer, warm rerun or new random initializer is used.
"""

from copy import deepcopy
import hashlib
import json
from math import log
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_open_jev_reserved_probe as runner
from tests.test_native_reserved_text_probe import ProbeChild, native_source


class ColdChild(ProbeChild):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._work = dict.fromkeys(self._work, 0)
        self.last_update = None


@pytest.fixture(scope="module")
def completed():
    child = ColdChild()
    report: dict[str, Any] = {}
    child.report = report
    runner.run_probe_updates(child, report, lambda event: None, source_history="cold")
    return child, report


def test_should_start_from_zero_inherited_work_and_pay_exact_cold_six_update_probe(completed):
    child, report = completed
    assert [(request.query, target) for request, target in child.training] == [
        ("Case marker: tavrix", "billing"),
        ("Intake code tavrix. Route this request.", "billing"),
    ] * 3
    assert len(child.inputs) == len(report["readouts"]) == 12
    assert (
        child.work
        == report["child_native_work"]
        == report["child_incremental_work"]
        == dict(
            training_attempts=6,
            forward_calls=18,
            backward_calls=6,
            optimizer_steps=6,
            committed_updates=6,
        )
    )
    assert report["probe_app_committed_updates"] == 6 and report["probe_complete"] is True
    rows = report["probe_delivery"]
    assert len(rows) == len({row["exposure_id"] for row in rows}) == 6
    assert [row["observed_at"] for row in rows] == list(range(0, 12, 2))
    assert [row["receipt"]["update_number"] for row in rows] == list(range(1, 7))
    assert all(row["role"] == "probe_train" and row["prelabel_updates"] == 0 for row in rows)
    assert all(row["receipt"]["arrived_at"] == row["observed_at"] + 1 for row in rows)


def test_should_label_cold_a_as_transfer_without_claiming_acquired_task_retention(completed):
    report = completed[1]
    summary = report["summary"]
    assert summary["scope"] == "one_fixed_cold_native_observed_context"
    assert summary["criteria"] == dict(
        C_endpoint_success=True,
        C_acquired=True,
        A_retained=None,
        A_transfer_endpoint_success=True,
        anchors_preserved=True,
    )
    assert summary["recovery"] == "not_tested"
    assert all(
        {"target", "correct", "nll", "nll_status"}.issubset(row) for row in report["readouts"]
    )
    assert summary["contrasts"]["C"]["nll_mean"] == pytest.approx(log(0.1) - log(0.8))


def test_should_preserve_default_warm_summary_semantics_for_the_same_scalar_readouts(completed):
    summary = runner.summarize_probe(deepcopy(completed[1]["readouts"]))
    assert summary["scope"] == "one_fixed_warm_native_reserved_context"
    assert summary["criteria"]["A_retained"] is True
    assert "A_transfer_endpoint_success" not in summary["criteria"]


@pytest.mark.parametrize("history", ["warm", True, None])
def test_should_refuse_invalid_history_before_corpus_or_source_capture(monkeypatch, history):
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid history reached admission or copy")

    monkeypatch.setattr(runner, "probe_examples", forbidden)
    monkeypatch.setattr(runner, "_capture_source", forbidden)
    child = ColdChild()
    with pytest.raises(ValueError):
        runner.run_probe_updates(child, {}, forbidden, source_history=history)
    with pytest.raises(ValueError):
        runner.run_fresh_probe(None, {}, forbidden, source_history=history)
    with pytest.raises(ValueError):
        runner.summarize_probe([], source_history=history)
    assert calls == [] and child.training == child.inputs == []


@pytest.mark.parametrize("fault", ["warm_in_cold", "cold_in_warm", "boolean_zero"])
def test_should_refuse_wrong_or_untyped_inherited_counter_baseline_before_inbox(monkeypatch, fault):
    child = ProbeChild() if fault == "warm_in_cold" else ColdChild()
    history = "after_A6" if fault == "cold_in_warm" else "cold"
    if fault == "boolean_zero":
        child._work["training_attempts"] = False
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("wrong baseline reached probe inbox")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    with pytest.raises(ValueError):
        runner.run_probe_updates(child, {}, forbidden, source_history=history)
    assert calls == [] and child.training == child.inputs == []


def test_should_keep_cold_before_readouts_and_three_paid_receipts_when_fourth_update_fails():
    child = ColdChild(fail_train=4)
    report: dict[str, Any] = {}
    child.report = report
    with pytest.raises(RuntimeError):
        runner.run_probe_updates(child, report, lambda event: None, source_history="cold")
    assert len(child.inputs) == 6 and len(child.training) == 4
    assert report["prediction_work"] == dict(attempted=6, returned=6, validated=6)
    assert report["probe_app_committed_updates"] == 3
    assert report["child_incremental_work"]["training_attempts"] == 4
    assert report["child_incremental_work"]["committed_updates"] == 3
    assert sum("receipt" in row for row in report["probe_delivery"]) == 3
    assert report["probe_complete"] is False and "summary" not in report
    assert all("target" not in row for row in report["readouts"])


def metric(accuracy, nll):
    return dict(examples=2, accuracy=accuracy, nll_mean=nll, nll_status="finite")


def comparison_inputs():
    cold = dict(
        by_phase=dict(
            before=dict(C=metric(0.0, 2.0), A=metric(0.5, 1.0), anchor=metric(1.0, 0.2)),
            after=dict(C=metric(1.0, 0.25), A=metric(0.5, 0.8), anchor=metric(1.0, 0.3)),
        )
    )
    warm = dict(
        by_phase=dict(
            before=dict(C=metric(0.5, 1.0), A=metric(1.0, 0.4), anchor=metric(1.0, 0.1)),
            after=dict(C=metric(0.0, 2.0), A=metric(1.0, 0.5), anchor=metric(1.0, 0.4)),
        )
    )
    return cold, warm


def test_should_report_all_signed_cold_minus_warm_endpoints_with_explicit_history_qualification():
    cold, warm = comparison_inputs()
    originals = deepcopy((cold, warm))
    report = runner.compare_probe_summaries(cold, warm)
    assert (cold, warm) == originals
    assert report["direction"] == "cold_minus_warm"
    assert set(report["by_phase"]) == {"before", "after"}
    assert all(set(groups) == {"C", "A", "anchor"} for groups in report["by_phase"].values())
    assert report["by_phase"]["after"]["C"]["accuracy"] == 1.0
    assert report["by_phase"]["after"]["C"]["nll_mean"] == -1.75
    assert report["by_phase"]["after"]["A"]["accuracy"] == -0.5
    assert report["by_phase"]["after"]["A"]["nll_mean"] == pytest.approx(0.3)
    assert report["by_phase"]["before"]["C"]["nll_mean"] == 1.0
    assert (
        report["qualification"]
        == "same_task_dose_and_readout_path_different_history_and_total_compute"
    )
    assert report["A_interpretation"] == "cold_untrained_transfer_vs_warm_retention"
    assert report["causal_scope"] == "history_association_not_weights_vs_optimizer_causality"


def test_should_keep_nonfinite_comparison_unavailable_instead_of_subtracting_infinity():
    cold, warm = comparison_inputs()
    warm["by_phase"]["after"]["C"].update(nll_mean=None, nll_status="infinite")
    result = runner.compare_probe_summaries(cold, warm)
    assert result["by_phase"]["after"]["C"] == dict(
        accuracy=1.0, nll_mean=None, nll_status="nonfinite_comparison_unavailable"
    )
    json.dumps(result, allow_nan=False)


@pytest.fixture(scope="module")
def accepted_warm():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/native-reserved-text-probe-20261022/actual.json"
    )
    with path.open("rb") as stream:
        body = stream.read(1024 * 1024 + 1)
    assert len(body) <= 1024 * 1024
    assert (
        hashlib.sha256(body).hexdigest()
        == "e1a701ed307d32ef4f6b3cfade9e96e357b937af9ab03d585a66243186e0e3ac"
    )
    return path, body, json.loads(body)


def test_should_load_one_bound_accepted_warm_summary_without_reexecuting_its_model(accepted_warm):
    path, body, payload = accepted_warm
    report = runner.load_warm_probe(path)
    assert report == dict(
        source_sha256=hashlib.sha256(body).hexdigest(), summary=payload["summary"]
    )
    report["summary"]["by_phase"]["before"]["C"]["accuracy"] = 123.0
    assert payload["summary"]["by_phase"]["before"]["C"]["accuracy"] != 123.0


@pytest.mark.parametrize("fault", ["changed_bytes", "oversized"])
def test_should_refuse_changed_or_oversized_warm_reference_before_json_use(
    accepted_warm, tmp_path, fault
):
    path = tmp_path / "refused.json"
    path.write_bytes(
        accepted_warm[1] + b" " if fault == "changed_bytes" else b" " * (1024 * 1024 + 1)
    )
    with pytest.raises(ValueError):
        runner.load_warm_probe(path)


@pytest.mark.parametrize("fault", ["definition", "incomplete", "interface", "corpus"])
def test_should_refuse_unqualified_warm_metadata_even_when_fixture_bytes_are_hash_bound(
    accepted_warm, tmp_path, monkeypatch, fault
):
    payload = deepcopy(accepted_warm[2])
    if fault == "definition":
        payload["definition"] = "foreign_study"
    elif fault == "incomplete":
        payload["complete"] = False
    elif fault == "interface":
        payload["prediction_interface"] = "serving_scorer"
    else:
        payload["examples"][0]["request"]["query"] = "foreign marker"
    body = json.dumps(payload, allow_nan=False).encode()
    path = tmp_path / "synthetic-qualified-bytes.json"
    path.write_bytes(body)
    monkeypatch.setattr(runner, "WARM_PROBE_SHA256", hashlib.sha256(body).hexdigest())
    with pytest.raises(ValueError):
        runner.load_warm_probe(path)


@pytest.fixture(scope="module")
def torch_backend():
    return pytest.importorskip("torch", reason="cold native backend unavailable; not a Torch pass")


@pytest.fixture(scope="module")
def native_cold(torch_backend):
    source, model = native_source(torch_backend, source_history="cold")
    before = source.snapshot_state()
    records = deepcopy(model.records)
    report: dict[str, Any] = {}
    runner.run_fresh_probe(source, report, lambda event: None, source_history="cold")
    return source, model, before, records, report


def test_should_run_one_real_cold_native_fork_with_no_source_training_and_preserve_source(
    native_cold, torch_backend
):
    from tests.test_open_jev_training import assert_tensor_tree_equal

    source, model, before, records, report = native_cold
    assert report["source_history"] == "cold" and source.policy.max_updates == 6
    assert report["source_preserved"] is report["probe_complete"] is True
    assert (
        source.work
        == report["source_native_work"]
        == report["child_birth_work"]
        == dict.fromkeys(source.work, 0)
    )
    expected = dict(
        training_attempts=6,
        forward_calls=18,
        backward_calls=6,
        optimizer_steps=6,
        committed_updates=6,
    )
    assert report["child_native_work"] == report["child_incremental_work"] == expected
    assert report["summary"]["criteria"]["A_retained"] is None
    assert report["probe_app_committed_updates"] == 6 and len(report["readouts"]) == 12
    actual = source.snapshot_state()
    for name in ("weights", "gradients", "optimizer", "rng"):
        assert_tensor_tree_equal(torch_backend, getattr(before, name), getattr(actual, name))
    assert model.records == records == [] and model.training is False
    assert source.last_update is None and source._optimizer.state == {}
    assert source.fork_work["fork_attempts"] == source.fork_work["forks_completed"] == 1


def test_should_not_renew_cold_source_fork_allowance_after_a_completed_probe(native_cold):
    source = native_cold[0]
    work = source.fork_work
    with pytest.raises(ValueError):
        runner.run_fresh_probe(source, {}, lambda event: None, source_history="cold")
    assert source.fork_work == work and all(value == 0 for value in source.work.values())


@pytest.mark.parametrize("fault", ["limit", "optimizer", "last_update", "warm_work"])
def test_should_refuse_non_cold_source_state_before_any_snapshot_or_fork(monkeypatch, fault):
    source = SimpleNamespace(
        _guard=lambda: None,
        work=dict(
            training_attempts=0,
            forward_calls=0,
            backward_calls=0,
            optimizer_steps=0,
            committed_updates=0,
        ),
        model=SimpleNamespace(training=False),
        policy=SimpleNamespace(max_updates=6, max_forks=1),
        fork_work=dict(fork_attempts=0),
        optimizer_reset_work=dict(optimizer_reset_attempts=0),
        update_health_work=dict(update_health_attempts=0),
        _optimizer=SimpleNamespace(state={}),
        last_update=None,
    )
    if fault == "limit":
        source.policy.max_updates = 12
    elif fault == "optimizer":
        source._optimizer.state = {0: {"step": "metadata-only nonempty state"}}
    elif fault == "last_update":
        source.last_update = {"loss": 0.5}
    else:
        source.work = dict.fromkeys(source.work, 6)
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid cold source reached capture or fork")

    monkeypatch.setattr(runner, "_capture_source", forbidden)
    with pytest.raises(ValueError):
        runner.run_fresh_probe(source, {}, forbidden, source_history="cold")
    assert calls == []


@pytest.mark.parametrize("flags", [["--cold-control"], ["--warm-reference", "unused.json"]])
def test_should_require_cli_cold_and_warm_reference_pair_before_any_output_or_load(
    tmp_path, monkeypatch, flags
):
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid cold CLI pair reached source or model")

    monkeypatch.setattr(runner, "load_reference", forbidden)
    monkeypatch.setattr(runner, "load_warm_probe", forbidden)
    monkeypatch.setattr(runner, "_load", forbidden)
    output = tmp_path / "refused.json"
    with pytest.raises(SystemExit) as error:
        runner.main(
            [
                "--reference",
                "unused.json",
                "--source",
                "unused",
                "--checkpoint",
                "unused",
                "--output",
                str(output),
                *flags,
            ]
        )
    assert error.value.code == 2 and calls == [] and not output.exists()
