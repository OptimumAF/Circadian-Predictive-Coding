"""Original TRAIN diagnostics: math, paid faults, admission and native parity."""

from math import exp, log
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_workflow_native as native


@pytest.fixture
def source():
    spec, _ = entry.prepare_training(context_return=True)
    return spec, native.prepare_train_fit(spec)


def test_should_reuse_original_train_targets_for_finite_phase_margins(source):
    _, data = source
    cases, batch = data
    logits = np.zeros((12, 6), dtype=np.float64)
    logits[np.arange(12), batch.targets[:, 0].astype(int)] = 2
    reader = SimpleNamespace(_learner=SimpleNamespace(predict_logits=lambda x: logits.copy()))
    checks = []
    row: dict[str, Any] = {}
    native.record_train_fit(reader, data, row, lambda: checks.append(True))
    assert row["complete"] and row["role"] == "train"
    assert row["work"] == dict(
        native_forward_attempts=1,
        rows_attempted=12,
        native_forward_returns=1,
        rows_returned=12,
        summary_returns=1,
        unreturned_work_unknown=False,
    )
    fit = row["fit"]
    assert fit["global_mean_ce"] == pytest.approx(log(1 + 5 * exp(-2)))
    assert fit["allowed_correct_rows"] == 12
    assert all(x["allowed_raw_margin"] == 2 and x["global_raw_margin"] == 2 for x in fit["rows"])
    assert [x["phase"] for x in fit["rows"]] == ["sealed", "ready", "staged"] * 4
    assert len(checks) >= 2 and all(c.role == "train" for c in cases)


@pytest.mark.parametrize("fault", ["lost", "shape", "nonfinite"])
def test_should_keep_attempted_diagnostic_work_on_bad_return(source, fault):
    _, data = source

    def forward(x):
        if fault == "lost":
            raise RuntimeError("lost diagnostic return")
        return np.zeros((1, 6)) if fault == "shape" else np.full((12, 6), np.nan)

    reader = SimpleNamespace(_learner=SimpleNamespace(predict_logits=forward))
    row: dict[str, Any] = {}
    with pytest.raises((RuntimeError, ValueError)):
        native.record_train_fit(reader, data, row, lambda: None)
    assert not row["complete"] and row["work"]["native_forward_attempts"] == 1
    assert row["work"]["rows_attempted"] == 12 and row["work"]["summary_returns"] == 0
    assert row["work"]["native_forward_returns"] == int(fault != "lost")


def test_should_refuse_nontrain_teacher_before_diagnostic_forward(source):
    _, data = source
    object.__setattr__(data[0][0], "role", "final_test")
    reader = SimpleNamespace(
        _learner=SimpleNamespace(predict_logits=lambda x: pytest.fail("final reached forward"))
    )
    row: dict[str, Any] = {}
    with pytest.raises(ValueError):
        native.record_train_fit(reader, data, row, lambda: None)
    assert row["work"]["native_forward_attempts"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(record_train_fit=1),
        dict(record_train_fit=True),
        dict(record_train_fit=True, method="bp"),
        dict(record_train_fit=True, context_return=True),
        dict(record_train_fit=True, context_return=True, method="bp", replay="recent"),
    ],
)
def test_should_validate_explicit_mode_before_data(kwargs, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("invalid data work"))
    with pytest.raises(ValueError):
        entry.run_example(**kwargs)


def test_should_forward_cli_and_protect_existing_output(tmp_path, monkeypatch):
    path = tmp_path / "case.json"
    seen = []

    def run(**kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    with pytest.raises(SystemExit):
        entry.main(["--record-train-fit", "--output", str(path)])
    assert not path.exists()
    args = ["--record-train-fit", "--context-return", "--method", "bp", "--output", str(path)]
    assert entry.main(args) == 0 and seen == [
        dict(context_return=True, method="bp", record_train_fit=True)
    ]
    with pytest.raises(FileExistsError):
        entry.main(args)
    assert len(seen) == 1


@pytest.fixture(scope="module")
def paired():
    plain = entry.run_example(context_return=True, method="bp")
    fitted = entry.run_example(context_return=True, method="bp", record_train_fit=True)
    assert plain["complete"], plain.get("error")
    assert fitted["complete"], fitted.get("error")
    return plain, fitted


def test_should_preserve_native_training_and_tasks_with_additional_accounted_reads(paired):
    plain, r = paired
    assert plain["checkpoints"] == r["checkpoints"]
    assert plain["applied_updates"] == r["applied_updates"]
    assert plain["episodes"] == r["episodes"] and plain["summary"] == r["summary"]
    assert plain["training_work"] == r["training_work"]
    assert "train_fit_readouts" not in plain
    assert (
        r["source_unchanged"]
        and r["actor_unchanged"]
        and r["torch_rng_unchanged"]
        and r["numpy_rng_unchanged"]
    )
    rows = r["train_fit_readouts"]
    assert [x["source_updates"] for x in rows] == [0, 4, 8, 12]
    assert all(x["complete"] and x["work"]["rows_returned"] == 12 for x in rows)
    assert r["costs"]["train_fit_native_forward_attempts"] == 4
    assert r["costs"]["train_fit_rows_attempted"] == 48
    assert r["costs"]["attempted_train_fit_nominal_affine_terms"] == 48 * 45
    assert r["budget"]["limits"] == plain["budget"]["limits"]
    assert len(entry._encode_report(r)) <= 512 * 1024


def test_should_support_cpc_readout_without_native_training(source):
    spec, data = source
    model = native.counted_learner(
        "gated_cpc", 9, 6, lambda x: pytest.fail("trained"), lambda: None
    )
    state = model.snapshot_state()
    reader = native.readout(spec, state)
    row: dict[str, Any] = {}
    native.record_train_fit(reader, data, row, lambda: native.require_readout(reader, spec, state))
    assert row["complete"] and native.same_state(model.snapshot_state(), state)
    assert reader.work.native_forward_attempts == 0


@pytest.mark.parametrize("fault", ["lost", "teacher", "paid"])
def test_should_stop_before_training_after_diagnostic_failure(fault, monkeypatch):
    original = native.record_train_fit
    saved = []

    def change(reader, data, row, require):
        if fault == "lost":
            row.update(
                complete=False,
                role="train",
                work=dict(
                    native_forward_attempts=1,
                    rows_attempted=12,
                    native_forward_returns=0,
                    rows_returned=0,
                    summary_returns=0,
                ),
            )
            raise RuntimeError("lost fit report")
        original(reader, data, row, require)
        if fault == "teacher":
            data[1].targets[0, 0] = 99
        saved.append(row)

    monkeypatch.setattr(native, "record_train_fit", change)
    if fault == "paid":
        original_call = native.TorchDecisionPredictor.__call__

        def corrupt(self, request):
            result = original_call(self, request)
            saved[0]["fit"]["rows"][0]["allowed_correct"] = None
            return result

        monkeypatch.setattr(native.TorchDecisionPredictor, "__call__", corrupt)
    r = entry.run_example(context_return=True, method="pc", record_train_fit=True)
    assert not r["complete"] and r["training_work"]["attempted_updates"] == 0
    assert r["train_fit_readouts"][0]["work"]["rows_attempted"] == 12


@pytest.mark.parametrize("field", ["source_updates", "role", "work"])
def test_should_refuse_active_metadata_or_work_changes_before_pinning(field, monkeypatch):
    holder = {}
    original_record = native.record_train_fit
    original_forward = native.TorchFeatureLearner.predict_logits

    def capture(reader, source, row, require):
        holder["row"] = row
        return original_record(reader, source, row, require)

    def corrupt(self, features):
        result = original_forward(self, features)
        row = holder["row"]
        if field == "source_updates":
            row[field] = 99
        elif field == "role":
            row[field] = "final_test"
        else:
            row["work"]["rows_attempted"] = 999
        return result

    monkeypatch.setattr(native, "record_train_fit", capture)
    monkeypatch.setattr(native.TorchFeatureLearner, "predict_logits", corrupt)
    r = entry.run_example(context_return=True, method="pc", record_train_fit=True)
    assert not r["complete"] and "pending TRAIN fit" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0
    assert len(r["train_fit_readouts"]) == 1 and not r["train_fit_readouts"][0]["complete"]
    assert r["train_fit_readouts"][0]["work"]["rows_attempted"] == 12
    assert r["train_fit_readouts"][0]["work"]["native_forward_returns"] == 1


def test_should_complete_final_guarded_diagnostic_path_without_repeating_control():
    r = entry.run_example(context_return=True, method="bp", record_train_fit=True)
    assert r["complete"], r.get("error")
    assert [row["source_updates"] for row in r["train_fit_readouts"]] == [0, 4, 8, 12]
    assert all(row["complete"] and row["role"] == "train" for row in r["train_fit_readouts"])
    assert r["costs"]["train_fit_native_forward_attempts"] == 4
    assert r["costs"]["train_fit_rows_attempted"] == r["costs"]["train_fit_rows_returned"] == 48
    assert r["source_unchanged"] and r["actor_unchanged"] and r["torch_rng_unchanged"]
    assert r["training_work"]["attempted_rows"] == 72


@pytest.mark.parametrize(
    "field,value", [("native_forward_attempts", True), ("rows_attempted", 12.0)]
)
def test_should_reject_counter_type_changes_even_when_values_compare_equal(source, field, value):
    _, data = source
    row: dict[str, Any] = {}

    def corrupt(features):
        row["work"][field] = value
        return np.zeros((12, 6), dtype=np.float64)

    reader = SimpleNamespace(_learner=SimpleNamespace(predict_logits=corrupt))
    with pytest.raises(ValueError, match="pending TRAIN fit work changed"):
        native.record_train_fit(reader, data, row, lambda: None)
    assert not row["complete"] and row["work"]["rows_attempted"] == 12


def test_should_keep_existing_math_entrypoints_bound_to_shared_calculations():
    from scripts import ticket_train_fit as shared
    from scripts import qualify_ticket_bp_fit as qualification
    from scripts import diagnose_ticket_training as diagnosis

    assert qualification._teacher_fit is shared.teacher_fit
    assert diagnosis._summarize_logits is shared.summarize_logits


def test_should_preserve_original_teacher_factory_entrypoint():
    from scripts import qualify_ticket_bp_fit as qualification

    spec = entry.DecisionFeatureSpec(entry.CONTEXT_FEATURES, entry.CATALOG)
    assert entry._same_value(native.prepare_train_fit(spec), qualification._context_teacher(spec))


def test_should_keep_missing_diagnostic_work_unknown_in_costs():
    report: dict[str, Any] = dict(
        native_readout_work=[],
        checkpoints=[],
        training_work=dict(attempted_updates=0, returned_updates=0),
        train_fit_readouts=[{}],
    )
    costs = native.native_costs(report, 0, "bp", feature_dim=9)
    assert costs["train_fit_native_forward_attempts"] is None
    assert costs["train_fit_rows_attempted"] is None and costs["train_fit_rows_returned"] is None
    assert costs["attempted_train_fit_nominal_affine_terms"] is None
    assert costs["train_fit_unreturned_work_unknown"] is True
    report["train_fit_readouts"] = [
        dict(
            work=dict(
                native_forward_attempts=1,
                rows_attempted=12,
                native_forward_returns=0,
                rows_returned=0,
            )
        )
    ]
    costs = native.native_costs(report, 0, "bp", feature_dim=9)
    assert costs["train_fit_rows_attempted"] == 12
    assert (
        costs["train_fit_rows_returned"] == 0 and costs["train_fit_unreturned_work_unknown"] is True
    )
