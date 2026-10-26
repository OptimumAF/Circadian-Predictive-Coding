"""CLI byte-boundary fixtures: metadata only, no source/learner/task execution."""

import json

import pytest

from scripts import context_controller_learning as learning
from scripts import run_bp_ticket_serving as cli


@pytest.mark.parametrize(
    "flag,size,accepted",
    [
        ("--context-controller-task-trials", 2 * 1024**2, True),
        ("--context-controller-task-trials", 3 * 1024**2, False),
        ("--context-controller-learning", 2 * 1024**2, False),
    ],
)
def test_should_enforce_explicit_task_report_bound_and_preserve_classifier_limit(
    flag, size, accepted, tmp_path, monkeypatch
):
    def metadata_only(path, **kwargs):
        assert str(path) == "unused"
        assert kwargs == (dict(task_outcomes=True) if "task-trials" in flag else {})
        return dict(complete=True, metadata_fixture="x" * size)

    monkeypatch.setattr(learning, "run_study", metadata_only)
    output = tmp_path / "output.json"
    command = ["--source", "unused", flag, "--output", str(output)]
    if accepted:
        assert cli.main(command) == 0
        assert len(json.loads(output.read_bytes())["metadata_fixture"]) == size
        assert output.stat().st_size < 3 * 1024**2
    else:
        with pytest.raises(ValueError, match="declared byte limit"):
            cli.main(command)
        if "task-trials" in flag:
            saved = json.loads(output.read_bytes())
            assert saved["complete"] is False
            assert saved["task_outcomes_available"] is False
            assert saved["fitted_policy_available"] is False
        else:
            assert output.read_bytes() == b"{}\n"


@pytest.mark.parametrize("trial_count", [0, 1, 4, 5])
def test_should_save_bounded_reported_paid_work_without_implying_saved_trial_evidence(
    trial_count, tmp_path, monkeypatch
):
    work = dict(attempted_updates=1, attempted_rows=3, returned_updates=0)
    reads = dict(attempts=4, returns=3, readouts=2, typed_attempts=3)
    budget = dict(
        updates=0,
        started_at=7,
        last_clock=9,
        limits=dict(max_training_updates=1, max_wall_seconds=30, max_process_rss_bytes=512),
    )
    trial = dict(
        result=dict(
            complete=False,
            training_work=work,
            task_readout_work=dict(before=reads),
            budget=budget,
            elapsed_seconds=1.25,
            memory=dict(start_bytes=100, peak_bytes=200, sample_count=3, interval_seconds=0.02),
            error="lost return " * 1000,
            task_outcomes="PRIVATE OUTCOME",
            native_calls="PRIVATE RECEIPT",
            candidate_state="PRIVATE WEIGHTS",
        )
    )
    report = dict(
        complete=False,
        trials=[trial] * trial_count,
        training_work=work,
        error="producer failure",
        elapsed_seconds=2.5,
        memory=dict(start_bytes=100, peak_bytes=200, sample_count=5, interval_seconds=0.02),
        fitted_policy="PRIVATE FIT",
        metadata_fixture="x" * (3 * 1024**2),
    )
    original = json.dumps(report)
    monkeypatch.setattr(learning, "run_study", lambda *a, **k: report)
    output = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="declared byte limit"):
        cli.main(
            ["--source", "unused", "--context-controller-task-trials", "--output", str(output)]
        )
    data = output.read_bytes()
    saved = json.loads(data)
    assert len(data) < 32 * 1024
    assert saved["complete"] is False
    assert saved["detailed_report_available"] is False
    assert saved["trial_receipts_available"] is False
    assert saved["reported"]["training_work"] == work
    assert saved["reported"]["producer_error"] == "producer failure"
    assert saved["reported"]["trial_count"] == trial_count
    assert len(saved["reported"]["trials"]) == min(trial_count, 4)
    assert saved["reported"]["omitted_trial_count"] == max(0, trial_count - 4)
    assert saved["report_bytes"] == len(
        (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    )
    assert saved["report_byte_limit"] == 3 * 1024**2
    for row in saved["reported"]["trials"]:
        assert row["training_work"] == work
        assert row["task_readout_work"]["before"] == reads
        assert row["task_readout_work"]["after"]["attempts"] is None
        assert row["budget"]["updates"] == 0
        assert row["budget"]["started_at"] == 7
        assert row["budget"]["last_clock"] == 9
        assert row["budget"]["limits"]["max_training_updates"] == 1
        assert row["elapsed_seconds"] == 1.25
        assert row["memory"]["peak_bytes"] == 200
        assert len(row["producer_error"]) == 512
    assert b"PRIVATE" not in data
    assert json.dumps(report) == original
    with pytest.raises(FileExistsError):
        cli.main(
            ["--source", "unused", "--context-controller-task-trials", "--output", str(output)]
        )
    assert output.read_bytes() == data


@pytest.mark.parametrize("bad_count", [True, -1, {"not": "scalar"}, 2**64])
def test_should_mark_unavailable_reported_work_without_inventing_zero(
    bad_count, tmp_path, monkeypatch
):
    report = dict(
        complete=True,
        training_work=dict(attempted_updates=bad_count),
        trials=None,
        metadata_fixture="x" * (3 * 1024**2),
    )
    monkeypatch.setattr(learning, "run_study", lambda *a, **k: report)
    output = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="declared byte limit"):
        cli.main(
            ["--source", "unused", "--context-controller-task-trials", "--output", str(output)]
        )
    saved = json.loads(output.read_bytes())
    assert saved["complete"] is False
    assert saved["reported"]["training_work"]["attempted_updates"] is None
    assert saved["reported"]["training_work"]["returned_updates"] is None
    assert saved["reported"]["trial_count"] is None


@pytest.mark.parametrize("extra,accepted", [(0, True), (1, False)])
def test_should_accept_exact_task_byte_limit_and_refuse_one_byte_more(
    extra, accepted, tmp_path, monkeypatch
):
    report = dict(complete=True, padding="")
    overhead = len((json.dumps(report, indent=2) + "\n").encode())
    report["padding"] = "x" * (3 * 1024**2 - overhead + extra)
    monkeypatch.setattr(learning, "run_study", lambda *a, **k: report)
    output = tmp_path / "boundary.json"
    command = ["--source", "unused", "--context-controller-task-trials", "--output", str(output)]
    if accepted:
        assert cli.main(command) == 0
        assert output.read_bytes() == (json.dumps(report, indent=2) + "\n").encode()
    else:
        with pytest.raises(ValueError, match="declared byte limit"):
            cli.main(command)
        assert json.loads(output.read_bytes())["complete"] is False


def test_should_bound_unicode_error_expansion_before_writing_failure(tmp_path, monkeypatch):
    error = "\U0001f600" * 1000
    report = dict(
        complete=True,
        error=error,
        trials=[dict(result=dict(error=error))] * 4,
        metadata_fixture="x" * (3 * 1024**2),
    )
    monkeypatch.setattr(learning, "run_study", lambda *a, **k: report)
    output = tmp_path / "unicode-failure.json"
    with pytest.raises(ValueError, match="declared byte limit"):
        cli.main(
            ["--source", "unused", "--context-controller-task-trials", "--output", str(output)]
        )
    data = output.read_bytes()
    saved = json.loads(data)
    assert len(data) < 32 * 1024
    assert saved["complete"] is False
    for record in [saved["reported"], *saved["reported"]["trials"]]:
        prefix = record["producer_error"]
        assert prefix and error.startswith(prefix)
        assert len(prefix) <= 512
        assert len(json.dumps(prefix)) <= 1024
