"""Current-packet LS through existing chronological recipe; no study ranking."""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_workflow_linear as linear
from src.adapters import linear_fit_learner
from src.adapters.linear_fit_decision import LinearFitDecisionPredictor
from src.core.categorical_linear_state import LinearFitPolicy, LinearFitSnapshot
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG
from src.core.typed_decision import DecisionRequest


@pytest.mark.parametrize(
    "options",
    [
        dict(context_return=False),
        dict(replay="recent"),
        dict(policy_change=True),
        dict(private_adaptation=True),
        dict(paired_serving="joint"),
        dict(record_train_fit=True),
        dict(class_balanced_bp=True),
    ],
)
def test_should_refuse_unsupported_linear_modes_before_data(options, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data"))
    kwargs = dict(context_return=True, method="linear_fit")
    kwargs.update(options)
    with pytest.raises(ValueError, match="linear_fit"):
        entry.run_example(**kwargs)


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--policy-change"],
        ["--context-return", "--replay", "coverage"],
        ["--context-return", "--record-train-fit"],
        ["--context-return", "--class-balanced-bp"],
    ],
)
def test_should_refuse_cli_before_output_or_work(flags, tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("ran"))
    with pytest.raises(SystemExit) as error:
        entry.main(["--output", str(path), "--method", "linear_fit", *flags])
    assert error.value.code == 2 and not path.exists()


@pytest.mark.parametrize(
    "method", ["linear_fit", "bp", "pc", "neutral_cpc", "gated_cpc", "perceptron"]
)
def test_should_forward_cli_without_changing_existing_defaults(method, tmp_path, monkeypatch):
    seen = []

    def run(**kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    path = tmp_path / "one.json"
    flags = ["--context-return", "--method", method, "--output", str(path)]
    assert entry.main(flags) == 0
    assert seen == [
        dict(context_return=True, **({} if method == "perceptron" else dict(method=method)))
    ]
    with pytest.raises(FileExistsError):
        entry.main(flags)
    assert len(seen) == 1


def test_should_read_unfitted_uniform_state_without_fit_or_affine_work():
    spec = DecisionFeatureSpec(entry.CONTEXT_FEATURES, CATALOG)
    state = LinearFitSnapshot(LinearFitPolicy(9, 6), None, 0, 0)
    reader = linear.readout(spec, state)
    assert np.array_equal(reader.predict_logits(np.zeros((1, 9))), np.zeros((1, 6)))
    request = DecisionRequest("r", "sealed invoice policy_v1", spec.catalog[:2])
    result = reader(request)
    assert all(score.probability == 0.5 for score in result.scores)
    assert result.selected_id == min(c.candidate_id for c in request.allowed_candidates)
    assert reader.work.unfitted_attempts == reader.work.unfitted_returns == 2
    assert reader.work.affine_attempts == reader.work.nominal_affine_terms == 0
    linear.require_readout(reader, spec, state)
    assert linear.checkpoint(0, state)["coefficient_bytes"] == 0


def test_should_execute_current_packet_d9_stream_and_record_distinct_costs(monkeypatch):
    holder = {}
    observed = []
    original_stream = entry.AttributionStream
    original_fit = linear_fit_learner.fit_categorical_linear

    def capture(*args, **kwargs):
        stream = original_stream(*args, **kwargs)
        holder["stream"] = stream
        return stream

    def fit(x, y, k):
        inbox = holder["stream"]._runtime._inbox
        ready = inbox._ready(inbox._last_time)
        assert len(ready) == 1
        source, label = inbox._experiences[ready[0]], inbox._labels[ready[0]]
        assert source.role == label.role == "train" and source.permissions.training
        assert source.observed_at + 1 == label.arrived_at <= inbox._last_time
        assert np.array_equal(x, source.features) and np.array_equal(y, label.targets)
        observed.append(len(x))
        return original_fit(x, y, k)

    monkeypatch.setattr(entry, "AttributionStream", capture)
    monkeypatch.setattr(linear_fit_learner, "fit_categorical_linear", fit)
    r = entry.run_example(context_return=True, method="linear_fit")
    assert r["complete"], r.get("error")
    assert observed == [6] * 12
    assert r["training_work"] == dict(
        attempted_updates=12,
        attempted_rows=72,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert all(
        r[k]
        for k in (
            "source_unchanged",
            "actor_unchanged",
            "numpy_rng_unchanged",
            "python_rng_unchanged",
        )
    )
    assert (
        r["budget"]["limits"]["max_wall_seconds"] == 30
        and r["budget"]["limits"]["max_process_rss_bytes"] == 512 * 1024**2
    )
    assert r["checkpoints"][0]["state"]["model"] is None
    assert [x["coefficient_bytes"] for x in r["checkpoints"]] == [0, 480, 480, 480]
    assert all(x["state"]["model"]["fit_rows"] == 6 for x in r["checkpoints"][1:])
    assert r["summary"] == entry.summarize_context_return(r["episodes"])
    assert len(r["episodes"]) == 24 and all(x["episode"]["complete"] for x in r["episodes"])
    assert r["stream"]["score_forward_calls"] == 0
    assert all(
        r[k] == 0
        for k in ("final_examples", "probe_updates", "replay_examples", "external_actions")
    )
    costs = r["costs"]
    reads = r["linear_readout_work"]
    assert (
        costs["physical_lstsq_calls"] == 12
        and costs["retained_checkpoint_coefficient_bytes"] == 1440
    )
    assert reads[0]["work"]["unfitted_attempts"] > 0 and reads[0]["work"]["affine_attempts"] == 0
    assert all(x["work"]["unfitted_attempts"] == 0 for x in reads[1:])
    assert (
        costs["linear_readout_totals"]["decision_returns"]
        == r["episode_work"]["prediction_calls_validated"]
    )
    assert (
        costs["linear_readout_totals"]["nominal_affine_terms"]
        == 60 * costs["linear_readout_totals"]["rows_attempted"]
    )
    assert not costs["linear_readout_work_unknown"]
    assert "attempted_training_affine_terms" not in costs and "torch_rng_unchanged" not in r
    assert len(entry._encode_report(r)) <= 512 * 1024


def test_should_preserve_owned_failed_solver_receipt_and_uncertain_physical_cost(monkeypatch):
    original = linear_fit_learner.fit_categorical_linear
    calls = []

    def lost(*args):
        calls.append(True)
        original(*args)
        raise RuntimeError("lost solver return")

    monkeypatch.setattr(linear_fit_learner, "fit_categorical_linear", lost)
    r = entry.run_example(context_return=True, method="linear_fit")
    assert not r["complete"] and calls == [True]
    assert (
        r["training_work"]["attempted_updates"] == 1 and r["training_work"]["returned_updates"] == 0
    )
    assert r["budget"]["updates_completed"] == 0 and len(r["checkpoints"]) == 1
    assert (
        r["native_failure_work"]["fit_attempts"] == 1
        and r["native_failure_work"]["fit_rows_attempted"] == 6
    )
    assert (
        r["native_failure_work"]["unreturned_work_unknown"]
        and r["native_failure_work"]["candidate_commits"] == 0
    )
    assert r["costs"]["physical_lstsq_calls"] is None


@pytest.mark.parametrize("role", ["final_test", "probe_train"])
def test_should_refuse_other_roles_before_solver(role, monkeypatch):
    from dataclasses import replace

    original = entry.prepare_training

    def data(**kw):
        spec, batches = original(**kw)
        return spec, (replace(batches[0], role=role), *batches[1:])

    monkeypatch.setattr(entry, "prepare_training", data)
    monkeypatch.setattr(linear_fit_learner, "fit_categorical_linear", lambda *a: pytest.fail("fit"))
    r = entry.run_example(context_return=True, method="linear_fit")
    assert not r["complete"] and r["training_work"]["attempted_updates"] == 0


@pytest.mark.parametrize("fault", ["reader", "candidate"])
def test_should_stop_before_more_updates_on_frozen_readout_or_source_change(fault, monkeypatch):
    original_stream = entry.AttributionStream
    holder = {}

    def capture(*args, **kw):
        value = original_stream(*args, **kw)
        holder["stream"] = value
        return value

    original = LinearFitDecisionPredictor.__call__

    def corrupt(self, request):
        value = original(self, request)
        if fault == "reader":
            self.spec = deepcopy(self.spec)
        else:
            setattr(holder["stream"]._runtime._candidate, "_policy", LinearFitPolicy(8, 6))
        return value

    monkeypatch.setattr(entry, "AttributionStream", capture)
    monkeypatch.setattr(LinearFitDecisionPredictor, "__call__", corrupt)
    r = entry.run_example(context_return=True, method="linear_fit")
    assert not r["complete"] and r["training_work"]["attempted_updates"] == 0
    assert r["linear_readout_work"][0]["work"]["unfitted_returns"] == 1


@pytest.mark.parametrize("fault", ["point", "work", "row", "list", "empty"])
def test_should_preserve_owned_readout_work_while_refusing_changed_pending_evidence(fault):
    spec = DecisionFeatureSpec(entry.CONTEXT_FEATURES, CATALOG)
    reader = LinearFitDecisionPredictor(spec, None)
    report: dict[str, Any] = {}
    owner = report["linear_readout_work"] = []
    row: dict[str, Any] = dict(source_updates=0, work=asdict(reader.work))
    owner.append(row)
    pins = [(row, deepcopy(row))]
    reader.predict_logits(np.zeros((1, 9)))
    if fault == "point":
        row["source_updates"] = 99
    elif fault == "work":
        row["work"]["unfitted_attempts"] = 99
    elif fault == "row":
        owner[0] = deepcopy(row)
    elif fault == "list":
        report["linear_readout_work"] = deepcopy(owner)
    else:
        owner.clear()
    with pytest.raises(ValueError, match="metadata changed"):
        linear.record_readout_work(report, owner, pins, reader)
    assert report["linear_readout_work"] is owner and owner[0] is row
    assert row["source_updates"] == 0 and row["work"]["unfitted_attempts"] == 1


def test_should_keep_missing_readout_counters_unknown_in_cost_summary():
    report = dict(
        linear_readout_work=[{}],
        training_work=dict(attempted_updates=0, attempted_rows=0, returned_updates=0),
        checkpoints=[],
    )
    costs = linear.costs(report)
    assert (
        costs["linear_readout_work_unknown"]
        and costs["linear_readout_totals"]["affine_attempts"] is None
    )
