"""Reserved native workflow integration, not a comparative experiment."""

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_native_probe as probe
from scripts import ticket_workflow_native as native
from scripts.context_head_serving import PAIRED_TASKS
from src.adapters.torch_feature_learner import TorchFeatureLearner


@pytest.mark.parametrize("value", [None, 1, "yes", [], {}])
def test_should_refuse_nonboolean_probe_before_source(value, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("source"))
    with pytest.raises(ValueError, match="reserved probe"):
        entry.run_example(reserved_probe=value)


@pytest.mark.parametrize(
    "change",
    [
        dict(method="perceptron"),
        dict(method="linear_fit"),
        dict(context_return=False),
        dict(separate_context_heads=True),
        dict(policy_change=True),
        dict(replay="recent"),
        dict(private_adaptation=True),
        dict(paired_serving="joint"),
        dict(adaptive_serving=True),
        dict(catalog_study=True),
        dict(catalog_action="wait"),
        dict(reference_validation=True),
        dict(incident_stream="stationary"),
        dict(class_balanced_bp=True),
        dict(record_train_fit=True),
        dict(probe_update="minimum_change"),
        dict(train_update="minimum_change"),
    ],
)
def test_should_refuse_conflicting_composition_before_models(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("source"))
    options = dict(method="bp", context_return=True, reserved_probe=True)
    options.update(change)
    with pytest.raises(ValueError, match="reserved probe"):
        entry.run_example(**options)


def test_should_refuse_foreign_source_before_data_or_model(monkeypatch):
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: pytest.fail("data"))
    with pytest.raises(ValueError, match="checkpoint12"):
        probe.run_owned_probe(None, None, {}, None, check_boundary=None, memory=None)


@pytest.mark.parametrize("complete", [True, False])
def test_should_refuse_paid_report_reuse_without_resetting_it(complete, monkeypatch):
    row = dict(complete=complete, training_work=dict(attempted_updates=1, attempted_rows=3))
    original = json.dumps(row)
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: pytest.fail("data"))
    with pytest.raises(ValueError, match="paid work cannot be renewed"):
        probe.run_owned_probe(None, None, {}, None, check_boundary=None, memory=None, row=row)
    assert json.dumps(row) == original


@pytest.mark.parametrize("fault", [None, "native_read", "last_clock", "both_snapshots"])
def test_endpoint_guard_should_refuse_private_changes_before_more_work(fault, monkeypatch):
    @dataclass
    class Snapshot:
        completed_updates: int
        values: list[int]

    @dataclass
    class Work:
        attempts: int = 0

    class Learner:
        def __init__(self):
            self.state = Snapshot(28, [1])

        def snapshot_state(self):
            return deepcopy(self.state)

    learner = Learner()
    reader_calls: list[int] = []

    class Reader:
        work = Work()

        def __call__(self, request):
            self.work.attempts += 1
            reader_calls.append(1)
            if fault == "native_read":
                learner.state.values[0] = 2
            return "literal decision"

    def construct(spec, state):
        if fault == "both_snapshots":
            state.values[0] = learner.state.values[0] = 2
        return Reader()

    def boundary(read=None, *, after_clocks=None):
        if read is not None:
            assert read() is None
        if after_clocks is not None:
            if fault == "last_clock":
                learner.state.values[0] = 2
            assert after_clocks() is None

    def episodes(predictor, phase, require, rows, **kwargs):
        require()
        predictor.serve("literal request")
        rows.append(dict(complete=True))

    monkeypatch.setattr(native, "readout", construct)
    monkeypatch.setattr(native, "require_readout", lambda *a: None)
    monkeypatch.setattr(probe, "_episodes", episodes)
    row: dict[str, Any] = {}
    if fault is None:
        probe._read_tasks(learner, "after_probe", None, lambda: None, row, boundary=boundary)
        assert learner.state.values == [1] and reader_calls == [1]
    else:
        with pytest.raises(ValueError, match="private native endpoint learner changed"):
            probe._read_tasks(learner, "after_probe", None, lambda: None, row, boundary=boundary)
        assert learner.state.values == [2]  # Never restore foreign mutations.
        assert reader_calls == ([1] if fault == "native_read" else [])


@pytest.mark.parametrize("method", native.METHODS)
def test_should_forward_explicit_probe_and_preserve_default(method, tmp_path, monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    path = tmp_path / "probe.json"
    flags = ["--output", str(path), "--context-return", "--method", method]
    assert entry.main([*flags, "--reserved-probe"]) == 0
    assert calls == [dict(context_return=True, method=method, reserved_probe=True)]
    with pytest.raises(FileExistsError):
        entry.main([*flags, "--reserved-probe"])
    assert len(calls) == 1
    calls.clear()
    assert entry.main([*flags, "--output", str(tmp_path / "default.json")]) == 0
    assert calls == [dict(context_return=True, method=method)]


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--method", "linear_fit", "--context-return"],
        ["--method", "bp"],
        ["--method", "bp", "--context-return", "--record-train-fit"],
        ["--method", "pc", "--context-return", "--separate-context-heads"],
        ["--method", "pc", "--context-return", "--private-adaptation"],
    ],
)
def test_should_refuse_bad_cli_before_dispatch(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("models"))
    with pytest.raises(SystemExit):
        entry.main(["--output", str(tmp_path / "bad.json"), "--reserved-probe", *flags])


@pytest.fixture(scope="module", params=native.METHODS)
def completed(request):
    holder: dict[str, Any] = {}
    original = entry.AttributionStream

    def capture(*args, **kwargs):
        stream = original(*args, **kwargs)
        holder.update(
            source=args[0],
            source_pin=args[0].snapshot_state(),
            actor=stream._runtime.actor,
            actor_pin=stream._runtime.actor.snapshot_state(),
        )
        return stream

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(entry, "AttributionStream", capture)
        result = entry.run_example(method=request.param, context_return=True, reserved_probe=True)
    return result, holder


def test_should_run_source_probe_and_return_on_owned_native_descendants(completed, record_property):
    result, holder = completed
    child = result.get("reserved_probe", {})
    record_property(
        "reserved_probe",
        dict(
            method=result["method"],
            complete=result["complete"],
            error=result.get("error"),
            child_complete=child.get("complete"),
            child_error=child.get("error"),
            work=child.get("preserved_training_work"),
            sessions=child.get("sessions"),
            endpoints={
                phase: [e["episode"]["success"] for e in rows]
                for phase, rows in child.get("episodes", {}).items()
            },
        ),
    )
    assert result["complete"], result.get("error")
    assert child["complete"], child.get("error")
    assert result["training_work"] == dict(
        attempted_updates=12,
        attempted_rows=72,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert (
        child["training_work"]
        == child["preserved_training_work"]
        == dict(attempted_updates=32, attempted_rows=144, returned_updates=32)
    )
    assert result["probe_updates"] == 32
    assert [s["updates"] for s in child["sessions"]] == [16, 16]
    assert all(s["limits"] == asdict(probe.LIMITS) for s in child["sessions"])
    assert result["budget"]["updates_completed"] == 12
    assert result["budget"]["limits"]["max_wall_seconds"] == 30
    assert result["source_unchanged"] and result["actor_unchanged"]
    assert result["numpy_rng_unchanged"] and result["torch_rng_unchanged"]
    assert native.same_state(holder["source"].snapshot_state(), holder["source_pin"])
    assert native.same_state(holder["actor"].snapshot_state(), holder["actor_pin"])
    assert result["final_examples"] == child["final_examples"] == child["replay_examples"] == 0
    assert child["probe"]["status"] == child["probe"]["return_phase"]["status"] == "complete"
    assert "original TRAIN source only" in result["costs"]["scope"]
    assert len(result["applied_updates"]) == 12
    for phase, count in (("before_probe", 12), ("after_probe", 28), ("after_return", 44)):
        rows = child["episodes"][phase]
        assert [(r["policy_version"], r["concept"]) for r in rows] == list(PAIRED_TASKS)
        assert all(r["role"] == "controller_train" and r["complete"] for r in rows)
        assert all(read["completed_updates"] == count for r in rows for read in r["reads"])


def test_should_write_complete_new_report_and_preserve_original_output_bound(
    completed, tmp_path, record_property
):
    from scripts.context_catalog_learning import write_catalog_report

    result, _ = completed
    assert result["complete"], result.get("error")
    output = tmp_path / "reserved.json"
    assert write_catalog_report(output, result, max_bytes=4 * 1024**2)
    payload = output.read_bytes()
    assert len(payload) <= 4 * 1024**2
    assert json.loads(payload)["reserved_probe"]["complete"]
    capture = Path("artifacts/runs/ordinary-native-reserved-probe-20261030") / (
        f"{sys.platform}-{result['method']}.json.gz"
    )
    with capture.open("xb") as handle:
        handle.write(gzip.compress(payload, mtime=0))
    record_property(
        "report_bytes", dict(method=result["method"], raw=len(payload), gzip=capture.stat().st_size)
    )


def test_should_keep_legacy_encoder_limit_without_any_model_work():
    with pytest.raises(ValueError, match="512KiB"):
        entry._encode_report(dict(pad="x" * (513 * 1024)))


def test_should_keep_lost_probe_work_and_stop_before_return(monkeypatch, record_property):
    original = TorchFeatureLearner.train_batch
    calls = []

    def lost(owner, features, targets):
        calls.append(len(features))
        diagnostic = original(owner, features, targets)
        if len(calls) == 13:
            raise RuntimeError("lost reserved native return")
        return diagnostic

    monkeypatch.setattr(TorchFeatureLearner, "train_batch", lost)
    result = entry.run_example(method="bp", context_return=True, reserved_probe=True)
    child = result.get("reserved_probe", {})
    record_property(
        "lost_probe",
        dict(
            complete=result["complete"],
            error=result.get("error"),
            work=child.get("preserved_training_work"),
            sessions=child.get("sessions"),
        ),
    )
    assert not result["complete"] and not child["complete"]
    assert calls == [6] * 12 + [3]
    assert child["preserved_training_work"] == dict(
        attempted_updates=1, attempted_rows=3, returned_updates=0
    )
    assert [s["updates"] for s in child["sessions"]] == [0, 0]
    assert result["probe_updates"] is None
    assert result["source_unchanged"] and result["actor_unchanged"]
    assert result["budget"]["updates_completed"] == 12
    assert "after_return" not in child["episodes"]


def test_should_refuse_equal_replacement_of_original_source_budget(monkeypatch, record_property):
    original_run = probe.run_owned_probe
    original_data = probe.build_head_cohorts
    holder = {}

    def run(state, spec, report, budget, **kwargs):
        holder["budget"] = budget
        holder["original"] = budget.budget
        return original_run(state, spec, report, budget, **kwargs)

    def data(*args, **kwargs):
        cohorts = original_data(*args, **kwargs)
        holder["replacement"] = replace(holder["budget"].budget)
        holder["budget"].budget = holder["replacement"]
        return cohorts

    monkeypatch.setattr(probe, "run_owned_probe", run)
    monkeypatch.setattr(probe, "build_head_cohorts", data)
    result = entry.run_example(method="bp", context_return=True, reserved_probe=True)
    child = result["reserved_probe"]
    record_property(
        "replaced_budget",
        dict(
            complete=result["complete"],
            error=result.get("error"),
            work=child.get("preserved_training_work"),
        ),
    )
    assert not result["complete"] and not child["complete"]
    assert "budget origin changed" in result["error"]
    assert child["preserved_training_work"] == dict(
        attempted_updates=0, attempted_rows=0, returned_updates=0
    )
    assert child["sessions"] == [] and child["episodes"] == {}
    assert holder["budget"].budget == holder["original"]
    assert holder["budget"].budget is holder["replacement"] is not holder["original"]
    assert result["source_unchanged"] is result["actor_unchanged"] is None
    assert result["training_work"]["attempted_updates"] == 12
