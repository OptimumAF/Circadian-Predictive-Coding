"""Literal procedure records only; never import a learner or consume real inputs."""

from copy import deepcopy
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any

import pytest

CASE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/full-network-native-control-20261101/study.py"
    ),
    run_name="literal_procedure_only",
)


def update():
    return dict(
        index=1,
        before=0.5,
        after=0.4,
        step_norm=0.2,
        completed_updates=65,
        training_examples=14300,
        work=dict(
            published_updates=1,
            gradient_requests=2,
            gradient_returns=2,
            solve_attempts=1,
            solve_returns=1,
            accepted_proposals=1,
            unknown_partial_work=False,
        ),
    )


def test_complete_literal_update_record_passes_without_model_execution():
    CASE["require_update"](update(), 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("index", True),
        ("completed_updates", 64),
        ("training_examples", True),
        ("after", float("nan")),
        ("after", 0.6),
        ("before", float("inf")),
        ("step_norm", 1.1),
    ],
)
def test_corrupt_literal_update_refused(field, value):
    record = update()
    record[field] = value
    with pytest.raises((RuntimeError, ValueError)):
        CASE["require_update"](record, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("published_updates", False),
        ("gradient_returns", 1),
        ("solve_returns", False),
        ("accepted_proposals", 0),
        ("unknown_partial_work", True),
    ],
)
def test_lost_or_corrupt_paid_update_ack_refused(field, value):
    record = update()
    record["work"][field] = value
    with pytest.raises(RuntimeError):
        CASE["require_update"](record, 1)


@pytest.mark.parametrize("fault", ["container", "item", "value", "missing", "extra"])
def test_previous_paid_prefix_refuses_replacement_mutation_or_forgotten_work(fault):
    records: list[dict[str, Any]] = []
    seals: list[tuple[dict[str, Any], str]] = []
    CASE["seal_record"](records, seals, update())
    original = records
    CASE["require_prefix"](records, original, seals)
    if fault == "container":
        records = list(records)
    elif fault == "item":
        records[0] = deepcopy(records[0])
    elif fault == "value":
        records[0]["after"] = 0.3
    elif fault == "missing":
        records.clear()
    else:
        records.append(update())
    with pytest.raises(RuntimeError):
        CASE["require_prefix"](records, original, seals)
    assert CASE["restored_prefix"](seals) == [update()]


def native(task):
    result = dict(complete=True, seed=17, role="train", prediction_attempts=2, prediction_returns=2)
    if task == "push":
        result.update(
            source_unchanged=True,
            unreturned_native_work_unknown=False,
            unreturned_prediction_possible=False,
            control_attempts=2,
            control_returns=2,
        )
    else:
        result.update(
            optimizer_updates=0,
            frozen_state_exact=True,
            original_owner_exact=True,
            action_history_exact=True,
        )
    return result


@pytest.mark.parametrize("task", ["push", "pick"])
def test_ordinary_native_contract_accepts_complete_negative_and_actual_field_shapes(task):
    result = native(task)
    result.update(any_success=False, final_success=False, sparse_return=0.0)
    CASE["require_native"](result, task, 17)
    # Neural Push has no optimizer_updates; Pick has no unreturned_prediction_possible.
    assert ("optimizer_updates" in result) == (task == "pick")


@pytest.mark.parametrize("task", ["push", "pick"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("role", "final_test"),
        ("seed", True),
        ("prediction_attempts", False),
        ("prediction_returns", 1),
    ],
)
def test_incomplete_reserved_or_noninteger_native_evidence_refused(task, field, value):
    result = native(task)
    result[field] = value
    with pytest.raises(RuntimeError):
        CASE["require_native"](result, task, 17)


class Failure(Exception):
    def __init__(self, work):
        super().__init__("full-network update rejected: no_strict_improvement")
        self.work = work


def test_only_explicit_complete_unpublished_rejection_is_normal_termination():
    work = update()["work"]
    work.update(published_updates=0, accepted_proposals=0)
    assert CASE["ordinary_rejection"](Failure(work))
    work["solve_returns"] = 0
    assert not CASE["ordinary_rejection"](Failure(work))
    work["solve_returns"] = 1
    work["unknown_partial_work"] = True
    assert not CASE["ordinary_rejection"](Failure(work))


def test_permanent_workstop_cannot_be_cleared_by_erasing_counter():
    clock = CASE["Meter"].__init__.__globals__["time"]
    original = clock.monotonic
    try:
        clock.monotonic = lambda: 0.0
        meter = CASE["Meter"](0.0, SimpleNamespace(peak_bytes=1))
        with pytest.raises(RuntimeError):
            meter.charge("forwards", 256)
        meter.counts["forwards"] = 0
        with pytest.raises(RuntimeError):
            meter.check()
    finally:
        clock.monotonic = original


def test_returned_constructor_is_owned_and_closed_when_next_gate_refuses():
    work = dict(returned=0, closed=0)
    scene = object()

    def acknowledge(value):
        assert value is scene
        work["returned"] += 1

    def deadline():
        raise RuntimeError("deadline after construction")

    owned = None
    with pytest.raises(RuntimeError):
        try:
            owned = CASE["paid_return"](lambda: scene, acknowledge, deadline, defer_check=True)
            deadline()
        finally:
            if owned is not None:
                work["closed"] += 1
    assert owned is scene and work == dict(returned=1, closed=1)


def test_late_returned_update_is_retained_before_next_budget_refusal():
    work = dict(returned=0)
    records: list[dict[str, Any]] = []
    seals: list[tuple[dict[str, Any], str]] = []

    def acknowledge(value):
        work["returned"] += 1

    def deadline():
        raise RuntimeError("deadline after accepted commit")

    with pytest.raises(RuntimeError):
        returned = CASE["paid_return"](update, acknowledge, deadline, defer_check=True)
        CASE["seal_record"](records, seals, returned)
        deadline()
    assert work["returned"] == 1 and CASE["restored_prefix"](seals) == [update()]


@pytest.mark.parametrize(
    "name,admitted",
    [
        ("continue-case.ps1", True),
        ("continuation-pins.json", True),
        ("unrelated-host.ps1", False),
        ("actual.json", False),
    ],
)
def test_declared_host_source_is_admitted_without_opening_other_raw_or_hosts(name, admitted):
    current = runpy.run_path(
        str(Path(CASE["STAGE"]) / "admission-continuation.py"), run_name="literal_admission_only"
    )
    assert current["admitted_artifact_read"](current["STAGE"] / name) is admitted
