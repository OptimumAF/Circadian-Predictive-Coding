"""Ordinary counted TRAIN updates; fixed engineering fixtures, not a study."""

import numpy as np
import pytest

from scripts import ticket_workflow_linear as counted
from scripts import recurring_ticket_workflow_example as entry
from scripts import context_head_live as live
from src.adapters import linear_fit_learner as linear
from src.adapters.context_linear_fit_learner import (
    CONTEXT_LINEAR_RESIDUAL_ID,
    CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID,
)
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyExecutionBudget, ToyBudgetSession
from src.core.learner_ports import TrainingUpdateFailed
from test_context_catalog_replay import spec
from test_context_linear_transfer_probe import donor


def packet(sign=1.0, label=1.0):
    x = np.zeros((1, 9))
    x[0, 0] = sign
    x[0, 4] = 1.0
    return x, np.array([[label]])


@pytest.mark.parametrize("choice", [True, 1, "cold", "unknown"])
def test_should_refuse_bad_train_selection_before_source(choice, monkeypatch):
    monkeypatch.setattr(entry, "training_cases", lambda **kw: pytest.fail("source"))
    with pytest.raises(ValueError, match="minimum-change TRAIN"):
        entry.run_example(train_update=choice)
    with pytest.raises(ValueError, match="minimum-change TRAIN"):
        counted.counted_learner(None, None, separate=True, spec=spec(), train_update=choice)
    with pytest.raises(ValueError, match="minimum-change TRAIN"):
        live.run_case(None, _train_update=choice)


@pytest.mark.parametrize(
    "change",
    [
        dict(method="bp"),
        dict(context_return=False),
        dict(paired_serving="retained_joint_probe"),
        dict(paired_serving="retained_joint_publish"),
        dict(reference_validation=True),
        dict(adaptive_serving=True),
        dict(probe_update="minimum_change"),
        dict(probe_rehearsal="joint"),
        dict(private_adaptation=True),
        dict(catalog_study=True),
    ],
)
def test_should_refuse_other_modes_before_any_training(change, monkeypatch):
    args = dict(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        train_update="minimum_change",
    )
    args.update(change)
    monkeypatch.setattr(entry, "training_cases", lambda **kw: pytest.fail("source"))
    with pytest.raises(ValueError, match="minimum-change TRAIN"):
        entry.run_example(**args)


def test_should_refuse_probe_only_selection_without_work():
    with pytest.raises(ValueError, match="minimum-change TRAIN"):
        live.run_owned_case(
            None,
            None,
            {},
            None,
            control="retained_joint_probe",
            check_boundary=lambda: None,
            memory=None,
            linear_fit=True,
            train_update="minimum_change",
        )


@pytest.mark.parametrize(
    "choice,value",
    [
        (None, "minimum_change"),
        (None, None),
        ("minimum_change", "cold"),
        ("minimum_change", None),
        ("minimum_change", True),
        ("minimum_change", ["minimum_change"]),
    ],
)
def test_should_pin_explicit_and_omitted_paid_intent_without_model_work(choice, value):
    other = object()
    report = dict(linear_train_update=value, other=other)
    with pytest.raises(ValueError, match="intent changed"):
        counted.require_train_update(report, choice)
    assert report["other"] is other
    if choice is None:
        assert "linear_train_update" not in report
    else:
        assert report["linear_train_update"] == choice
    assert counted.require_train_update(report, choice) is None


@pytest.mark.parametrize("choice", [None, "minimum_change"])
def test_should_keep_counted_fork_choice_callbacks_and_cold_default(choice):
    events: list[tuple[str, int | None]] = []
    kwargs = {} if choice is None else dict(train_update=choice)
    parent = counted.counted_learner(
        lambda x: events.append(("before", len(x))),
        lambda: events.append(("after", None)),
        separate=True,
        spec=spec(),
        **kwargs,
    )
    first = parent.train_batch(*packet())
    before = parent.snapshot_state()
    child = parent.fork()
    child.train_batch(*packet(-1.0, 0.0))
    assert parent.snapshot_state() == before
    assert child.completed_updates == child.training_examples == 2
    assert child.snapshot_state().heads[1] == before.heads[1]
    assert events == [("before", 1), ("after", None)] * 2
    assert first.definition == (
        CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID if choice else CONTEXT_LINEAR_RESIDUAL_ID
    )
    if choice:
        mixed = np.vstack((packet()[0], packet(-1.0, 0.0)[0]))
        assert np.argmax(child.predict(mixed), axis=1).tolist() == [1, 0]
    else:
        np.testing.assert_allclose(child.predict(packet()[0]), np.full((1, 6), 1 / 6), atol=1e-12)
    assert set(before.__dataclass_fields__) == {
        "policy",
        "heads",
        "completed_updates",
        "training_examples",
    }


def test_should_stop_existing_train_stream_on_lost_counted_update(monkeypatch):
    before_calls = []
    after_calls = []
    source = counted.counted_learner(
        lambda x: before_calls.append(len(x)),
        lambda: after_calls.append(1),
        separate=True,
        spec=spec(),
        train_update="minimum_change",
    )
    source.restore_state(donor().snapshot_state())  # literal metadata, no source fits
    before = source.snapshot_state()
    x, y = packet()
    batch = AttributionBatch("train", "c0", ("row0",), x, y, num_classes=6)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    stream = AttributionStream(
        source,
        (batch,),
        (),
        label_delay=1,
        checkpoint_updates=(0, 1),
        budget=budget,
        training_only=True,
    )
    original = linear.update_categorical_linear
    attempts = []

    def lose(*args):
        attempts.append(1)
        original(*args)
        raise RuntimeError("lost ordinary minimum-change fit")

    monkeypatch.setattr(linear, "update_categorical_linear", lose)
    with pytest.raises(TrainingUpdateFailed) as failed:
        stream.run_until(1)
    assert attempts == before_calls == [1] and after_calls == [] and budget.updates_completed == 0
    assert failed.value.definition == CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID
    assert failed.value.work["fit_attempts"] == 1 and failed.value.work["fit_rows_attempted"] == 1
    assert failed.value.work["unreturned_work_unknown"] and stream._runtime._stopped
    assert source.snapshot_state() == stream._runtime.actor.snapshot_state().state == before
    assert stream._runtime._candidate.snapshot_state() == before
    with pytest.raises(ValueError):
        stream.run_until(1)
    assert attempts == [1]


@pytest.mark.parametrize("paired", [False, True])
def test_should_forward_cli_selection_without_source_work(paired, tmp_path, monkeypatch):
    from scripts import context_catalog_learning as writer

    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(
            complete=True,
            training_work=dict(attempted_updates=12, attempted_rows=72),
            paired_serving=dict(training_work=dict(attempted_updates=11, attempted_rows=72)),
        )

    monkeypatch.setattr(entry, "run_example", run)
    monkeypatch.setattr(writer, "write_catalog_report", lambda *a, **kw: True)
    flags = [
        "--method",
        "linear_fit",
        "--context-return",
        "--separate-context-heads",
        "--train-update",
        "minimum_change",
        "--output",
        str(tmp_path / "ordinary.json"),
    ]
    if paired:
        flags += ["--paired-serving", "retained_joint"]
    assert entry.main(flags) == 0
    assert calls[0]["train_update"] == "minimum_change"
    calls.clear()
    with pytest.raises(SystemExit):
        entry.main(["--train-update", "minimum_change"])
    assert calls == []


def test_should_execute_source_and_retained_train_without_probe(monkeypatch, record_property):
    result = entry.run_example(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="retained_joint",
        train_update="minimum_change",
    )
    child = result.get("paired_serving", {})
    record_property(
        "ordinary_minimum_change",
        dict(
            complete=result["complete"],
            error=result.get("error"),
            source_work=result.get("training_work"),
            child_work=child.get("training_work"),
            task_counts=child.get("candidate_return_readout", {}).get("counts"),
            source_unchanged=result.get("source_unchanged"),
            actor_unchanged=result.get("actor_unchanged"),
            child_complete=child.get("complete"),
            child_error=child.get("error"),
            source_budget=result.get("budget"),
            child_budget=child.get("budget"),
        ),
    )
    assert result["complete"], result.get("error")
    assert child["complete"], child.get("error")
    assert result["linear_train_update"] == child["linear_train_update"] == "minimum_change"
    assert result["costs"]["objective"] == CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID
    assert (
        result["training_work"]["attempted_updates"]
        == result["training_work"]["returned_updates"]
        == 12
    )
    assert result["training_work"]["attempted_rows"] == 72
    assert child["training_work"] == dict(
        attempted_updates=11, attempted_rows=72, returned_updates=11
    )
    assert (
        child["candidate_state"]["completed_updates"] == 23
        and child["candidate_state"]["training_examples"] == 144
    )
    assert (
        result["source_unchanged"]
        and result["actor_unchanged"]
        and result["numpy_rng_unchanged"]
        and result["python_rng_unchanged"]
    )
    assert child["source_unchanged"] and child["actor_unchanged"] and child["owned_source_verified"]
    assert (
        result["final_examples"]
        == result["probe_updates"]
        == child["final_examples"]
        == child["probe_updates"]
        == 0
    )
    assert "isolated_transfer_probe" not in child and child["live_serving"]["closed"]
    assert all(
        r["diagnostic"]["definition"] == CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID
        for root in (result, child)
        for r in root["applied_updates"]
    )
    assert child["joint_rehearsal"]["charged_exposures"] == 45
    assert result["budget"]["limits"]["max_wall_seconds"] == 30
    assert child["budget"]["limits"]["max_wall_seconds"] == 60


def test_should_preserve_paid_source_intent_and_refuse_child_after_change(monkeypatch):
    calls = []

    def change(state, schema, report, budget, **kwargs):
        calls.append(1)
        report["linear_train_update"] = "cold"
        kwargs["check_boundary"]()
        pytest.fail("changed source must not admit child")

    monkeypatch.setattr(live, "run_owned_case", change)
    result = entry.run_example(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="retained_joint",
        train_update="minimum_change",
    )
    assert calls == [1] and result["complete"] is False and "intent changed" in result["error"]
    assert (
        result["training_work"]["attempted_updates"]
        == result["training_work"]["returned_updates"]
        == 12
    )
    assert result["training_work"]["attempted_rows"] == 72
    assert result["linear_train_update"] == "minimum_change"
    assert result["costs"]["objective"] == CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID
