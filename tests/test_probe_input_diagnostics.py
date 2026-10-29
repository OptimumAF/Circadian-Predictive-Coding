"""Exact TRAIN-input constraints and unchanged disposable probe execution."""

from dataclasses import replace
import pickle

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch
from src.app.fresh_task_probe import run_fresh_task_probe
from test_fresh_task_probe import FakeLearner, session


def diagnose(*args, **kwargs):
    from src.app.probe_input_diagnostics import diagnose_probe_inputs

    return diagnose_probe_inputs(*args, **kwargs)


def packet(role, name, x, y, classes=3):
    features = np.array(x, dtype=np.float64)
    return AttributionBatch(
        role,
        name,
        tuple(f"{name}-{i}" for i in range(len(x))),
        features,
        np.array(y, dtype=np.float64)[:, None],
        num_classes=classes,
    )


def cohorts():
    training = tuple(packet("probe_train", f"F{i}", [[0, 1]], [0]) for i in range(2))
    old = (
        packet("train", "old-invoice", [[0, 1]], [1]),
        packet("train", "old-parcel", [[0.5, 0]], [1]),
    )
    return training, old


def test_should_reuse_exact_bounds_and_identify_cross_cohort_conflict():
    training, old = cohorts()
    before = pickle.dumps((training, old))
    result = diagnose(training, old)
    assert result["status"] == "complete"
    assert result["joint"]["rows"] == 4 and result["joint"]["conflicting_input_keys"] == 1
    assert (
        result["probe_train"]["conflicting_input_keys"]
        == result["old_train"]["conflicting_input_keys"]
        == 0
    )
    assert result["joint"]["row_accuracy_upper_bound"] == 0.75
    assert result["cross_conflicting_input_keys"] == 1
    assert result["witnesses"] == [
        dict(
            probe_train=dict(
                batch_index=0, row_index=0, row_id="F0-0", context="F0", target_class=0
            ),
            old_train=dict(
                batch_index=0,
                row_index=0,
                row_id="old-invoice-0",
                context="old-invoice",
                target_class=1,
            ),
        )
    ]
    assert result["work"] == dict(
        bounds_calls_attempted=3,
        bounds_calls_returned=3,
        bounds_rows=8,
        bounds_coordinates=16,
        array_bytes_attempted=192,
        array_bytes_returned=192,
    )
    assert pickle.dumps((training, old)) == before
    result["witnesses"][0]["probe_train"]["context"] = "edited"
    assert training[0].context == "F0"


def test_should_distinguish_context_without_changing_targets_or_frequencies():
    training, old = cohorts()
    training = tuple(
        replace(b, features=np.column_stack((b.features, np.zeros(len(b.features)))))
        for b in training
    )
    old = tuple(
        replace(b, features=np.column_stack((b.features, np.ones(len(b.features))))) for b in old
    )
    r = diagnose(training, old)
    assert r["cross_conflicting_input_keys"] == r["joint"]["conflicting_input_keys"] == 0
    assert r["joint"]["row_accuracy_upper_bound"] == 1.0


@pytest.mark.parametrize("different", [False, True])
def test_should_merge_signed_zero_but_not_adjacent_float_inputs(different):
    training = (packet("probe_train", "F", [[0.0, -0.0]], [0]),)
    old = (packet("train", "A", [[np.nextafter(0.0, 1.0) if different else -0.0, 0.0]], [1]),)
    assert diagnose(training, old)["cross_conflicting_input_keys"] == int(not different)


def test_should_keep_within_cohort_conflicts_distinct_from_cross_cohort_conflicts():
    training = (packet("probe_train", "F", [[0], [0]], [0, 1]),)
    old = (packet("train", "A", [[1]], [2]),)
    r = diagnose(training, old)
    assert r["probe_train"]["conflicting_input_keys"] == 1
    assert r["cross_conflicting_input_keys"] == 0 and r["witnesses"] == []


def test_should_bound_witnesses_while_retaining_all_conflict_counts():
    x = np.arange(64, dtype=np.float64)[:, None] / 64
    training = tuple(
        packet("probe_train", f"F{i}", x[i * 32 : (i + 1) * 32], [0] * 32) for i in range(2)
    )
    old = tuple(packet("train", f"A{i}", x[i * 32 : (i + 1) * 32], [1] * 32) for i in range(2))
    r = diagnose(training, old)
    assert r["joint"]["rows"] == 128 and r["cross_conflicting_input_keys"] == 64
    assert len(r["witnesses"]) == 8 and r["witnesses_truncated"] is True
    assert r["joint"]["row_accuracy_upper_bound"] == 0.5


@pytest.mark.parametrize("reason", ["binary", "missing_old", "rows", "range"])
def test_should_report_unsupported_inputs_without_false_zero_conflicts(reason):
    training, old = cohorts()
    if reason == "binary":
        training = tuple(replace(b, num_classes=2) for b in training)
        old = tuple(replace(b, num_classes=2) for b in old)
    elif reason == "missing_old":
        old = ()
    elif reason == "rows":
        training = (packet("probe_train", "many", [[0]] * 128, [0] * 128),)
        old = (packet("train", "last", [[0]], [1]),)
    else:
        training = (replace(training[0], features=np.array([[2.0, 1.0]])),)
    r = diagnose(training, old)
    assert r["status"] == "unavailable" and r["reason"]
    assert r["cross_conflicting_input_keys"] is None and r["joint"] is None
    assert r["work"]["bounds_calls_attempted"] == r["work"]["array_bytes_attempted"] == 0


@pytest.mark.parametrize(
    "fault", ["final", "probe_test", "list", "classes", "width", "duplicate", "nan"]
)
def test_should_refuse_malformed_or_unadmitted_sources(fault):
    training, old = cohorts()
    if fault in ("final", "probe_test"):
        old = (replace(old[0], role="final_test" if fault == "final" else "probe_test"),)
    elif fault == "list":
        training = list(training)
    elif fault == "classes":
        old = (replace(old[0], num_classes=4),)
    elif fault == "width":
        old = (replace(old[0], features=np.ones((1, 3))),)
    elif fault == "duplicate":
        old = (replace(old[0], row_ids=training[0].row_ids),)
    else:
        old = (replace(old[0], features=np.array([[np.nan, 0.0]])),)
    with pytest.raises(ValueError):
        diagnose(training, old)


def evaluation_batches(target=2):
    return tuple(
        replace(packet("probe_test", f"eval{i}", [[0, 1]], [target]), context=f"F{i}")
        for i in range(2)
    )


def run_categorical(enabled, *, source=None, evaluation=None):
    training, old = cohorts()
    source = (
        FakeLearner(hooks={"predict": lambda owner, x: np.full((len(x), 3), 1 / 3)})
        if source is None
        else source
    )
    evaluation = evaluation_batches() if evaluation is None else evaluation
    budget = session()
    r = run_fresh_task_probe(
        source,
        np.array([0.0]),
        training,
        evaluation,
        budget=budget,
        checkpoint_updates=(0, 1, 2),
        timer=lambda: 2.5,
        training_readouts=old,
        input_diagnostics=enabled,
    )
    return r, source, budget


def test_should_preserve_learning_trace_metrics_and_budget_and_exclude_test_labels():
    a, sa, ba = run_categorical(False)
    b, sb, bb = run_categorical(True)
    diag = b.pop("input_diagnostics")
    assert a == b and pickle.dumps(sa.trace) == pickle.dumps(sb.trace)
    assert ba.updates_completed == bb.updates_completed == 2
    assert all(not item[3].any() for item in sb.trace if item[0] == "train")
    assert diag["joint"]["rows"] == 4
    changed, _, _ = run_categorical(True, evaluation=evaluation_batches(1))
    assert changed["input_diagnostics"] == diag


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_should_refuse_non_boolean_flag_before_callbacks(flag):
    source = FakeLearner()
    with pytest.raises(ValueError):
        run_categorical(flag, source=source)
    assert source.trace == []


def test_should_use_original_detached_inputs_after_external_source_mutation():
    training, old = cohorts()

    def change():
        training[0].features[:] = 0.75
        old[0].targets[:] = 2

    source = FakeLearner(
        hooks={"fork": change, "predict": lambda owner, x: np.full((len(x), 3), 1 / 3)}
    )
    r = run_fresh_task_probe(
        source,
        np.array([0.0]),
        training,
        evaluation_batches(),
        budget=session(),
        checkpoint_updates=(0, 1, 2),
        timer=lambda: 2.5,
        training_readouts=old,
        input_diagnostics=True,
    )
    assert r["input_diagnostics"]["witnesses"][0]["old_train"]["target_class"] == 1


def test_should_keep_paid_learning_result_when_reporting_fails(monkeypatch):
    from src.app import probe_input_diagnostics as module

    def failed(*args, work, **kwargs):
        work["bounds_calls_attempted"] = 1
        raise RuntimeError("diagnostic unavailable")

    monkeypatch.setattr(module, "diagnose_probe_inputs", failed)
    r, source, budget = run_categorical(True)
    assert r["status"] == "complete" and budget.updates_completed == 2
    assert r["input_diagnostics"]["status"] == "failed"
    assert r["input_diagnostics"]["work"]["bounds_calls_attempted"] == 1


def test_should_preserve_the_supplied_record_frequencies():
    r = diagnose(
        (packet("probe_train", "F", [[0]] * 8, [0] * 8),), (packet("train", "A", [[0]], [1]),)
    )
    assert r["joint"]["rows"] == r["joint"]["conflicting_rows"] == 9
    assert r["joint"]["row_accuracy_upper_bound"] == 8 / 9


def test_should_retain_each_returned_allocation_before_a_later_allocation_failure(monkeypatch):
    original = np.concatenate
    calls = 0
    work: dict[str, int] = {}

    def concatenate(arrays):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise MemoryError("target concatenation failed")
        return original(arrays)

    monkeypatch.setattr(np, "concatenate", concatenate)
    with pytest.raises(MemoryError):
        diagnose(*cohorts(), work=work)
    assert work["array_bytes_attempted"] == 192
    assert work["array_bytes_returned"] == 160
    assert work["bounds_calls_attempted"] == 0


def test_should_report_not_run_if_original_detachment_fails(monkeypatch):
    import src.app.fresh_task_probe as module

    monkeypatch.setattr(
        module, "_copy_batch", lambda *args: (_ for _ in ()).throw(RuntimeError("copy failed"))
    )
    r, source, budget = run_categorical(True)
    assert r["status"] == "failed" and budget.updates_completed == 0 and source.trace == []
    assert r["input_diagnostics"]["status"] == "not_run"


def test_should_still_describe_inputs_after_a_failed_update():
    def fail(*args):
        raise RuntimeError("lost update return")

    source = FakeLearner(
        hooks={"train": fail, "predict": lambda owner, x: np.full((len(x), 3), 1 / 3)}
    )
    r, _, budget = run_categorical(True, source=source)
    assert r["status"] == "failed" and budget.updates_completed == 0
    assert r["input_diagnostics"]["status"] == "complete"


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_should_validate_ticket_forwarding_flag_before_sessions(flag):
    from scripts.probe_ticket_policy_learning import run_arm

    with pytest.raises(ValueError):
        run_arm(None, None, None, None, None, None, "unused", input_diagnostics=flag)


def test_should_enable_diagnostics_in_existing_live_catalog_probe(probing):
    from test_catalog_live_recipes import run
    from scripts.ticket_serving_timeline import LiveServingPlan

    r = run(probing, probe_after_window=True, serving_plan=LiveServingPlan(9, (2, 6)))
    assert r["complete"], r.get("error")
    reports = [arm["probe"]["input_diagnostics"] for arm in r["isolated_probe"]["cases"]]
    assert reports[0] == reports[1]
    assert reports[0]["status"] == "complete" and reports[0]["input_rows"] == 54
    assert r["training_work"]["attempted_updates"] == 5
    assert r["isolated_probe"]["training_work"]["attempted_updates"] == 64


# Reuse literal live-worker/native ports; no genuine native model construction.
from test_catalog_candidate_probe import probing as probing
from test_catalog_live_recipes import recipes as recipes
from test_catalog_live_serving import serving as serving
from test_ticket_catalog_control import driver as driver
