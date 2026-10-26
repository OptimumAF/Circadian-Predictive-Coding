"""Known algebra and strict unknown-cost/role chronology boundaries, no models."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import pytest

from src.app.controller_chronology import ControllerChronology
from src.core.controller_objective import (
    ControllerObjective,
    ControllerMeasures,
    ControllerProtocol,
    ControllerStream,
    ControllerTrial,
    ControllerTrialMetadata,
    ControllerFreeze,
    controller_utility,
)
from src.core.experience import LogicalClock
from src.core.reference_controller import ReferencePolicy


def measures(**changes):
    return replace(ControllerMeasures(0, 1, 2, 2, 1, 2, 30.0, 256 * 1024**2, 2, True), **changes)


def protocol(**changes):
    return replace(
        ControllerProtocol(
            "test",
            ControllerObjective(),
            ReferencePolicy("periodic"),
            (
                ControllerStream("train", "controller_train", "source-a"),
                ControllerStream("validation", "controller_validation", "source-b"),
                ControllerStream("final", "sealed_evaluation", "source-c"),
            ),
        ),
        **changes,
    )


def test_outer_utility_has_frozen_normalization_directions_and_all_costs():
    u = controller_utility(ControllerObjective(), measures())
    assert (
        u.normalized_new_gain,
        u.normalized_old_drop,
        u.normalized_compute,
        u.normalized_memory,
        u.normalized_violations,
        u.value,
    ) == (0.5, 0.5, 0.25, 0.5, 2.0, -2.75)
    u = controller_utility(
        ControllerObjective(),
        measures(
            new_before=2,
            new_after=0,
            old_before=0,
            old_after=2,
            execution_seconds=240,
            peak_rss_bytes=1024 * 1024**2,
            violations=0,
        ),
    )
    assert u.normalized_new_gain == -1 and u.normalized_old_drop == 0
    assert u.normalized_compute == u.normalized_memory == 2 and u.value == -5


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_seconds": None},
        {"peak_rss_bytes": None},
        {"violations": None},
        {"execution_seconds": float("nan")},
        {"execution_seconds": -1},
        {"peak_rss_bytes": 0},
        {"violations": True},
        {"work_complete": False},
        {"work_complete": 1},
        {"new_count": 0},
        {"old_after": 3},
        {"new_after": True},
    ],
)
def test_unknown_incomplete_or_invalid_measures_have_no_utility(changes):
    with pytest.raises(ValueError):
        controller_utility(ControllerObjective(), measures(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"new_gain_weight": -1},
        {"memory_weight": float("inf")},
        {"compute_weight": True},
        {"compute_reference_seconds": 0},
        {"memory_reference_bytes": False},
        {"violation_reference_count": 0},
        {
            "new_gain_weight": 0,
            "old_drop_weight": 0,
            "compute_weight": 0,
            "memory_weight": 0,
            "violation_weight": 0,
        },
    ],
)
def test_invalid_objective_is_refused(changes):
    with pytest.raises(ValueError):
        replace(ControllerObjective(), **changes)


def test_nonfinite_weighted_sum_is_refused():
    objective = replace(
        ControllerObjective(), compute_weight=1e308, memory_weight=1e308, violation_weight=1e308
    )
    with pytest.raises(ValueError, match="utility|nonfinite"):
        controller_utility(
            objective, measures(execution_seconds=120, peak_rss_bytes=512 * 1024**2, violations=1)
        )


@pytest.mark.parametrize("change", ["id", "group", "role", "count"])
def test_controller_streams_require_distinct_ids_groups_and_order(change):
    p = protocol()
    streams = list(p.streams)
    if change == "id":
        streams[1] = replace(streams[1], stream_id="train")
    if change == "group":
        streams[1] = replace(streams[1], source_group="source-a")
    if change == "role":
        streams[1] = replace(streams[1], role="controller_train")
    if change == "count":
        streams.pop()
    with pytest.raises(ValueError):
        replace(p, streams=tuple(streams))


def execution(clock, stream, reads, *, metrics=None, metadata=None):
    def run(policy):
        begin = clock.now()
        reads.append(stream.role)
        clock.advance_to(begin + 2)
        m = metadata or ControllerTrialMetadata(stream, policy, begin, begin + 1, begin + 2)
        return ControllerTrial(m, metrics or measures())

    return run


def developed():
    p = protocol()
    clock = LogicalClock()
    c = ControllerChronology(p, clock)
    reads: list[str] = []
    c.run_development("train", execution(clock, p.streams[0], reads))
    assert c.training_rows()[0].metadata.stream.role == "controller_train"
    c.run_development("validation", execution(clock, p.streams[1], reads))
    assert c.validation_rows()[0].metadata.stream.role == "controller_validation"
    return p, clock, c, reads


def test_three_phases_preserve_causal_clock_and_close_training_before_final():
    p, clock, c, reads = developed()
    with pytest.raises(ValueError):
        c.training_rows()
    frozen = c.freeze()
    assert frozen.frozen_at == clock.now() == 4
    row = c.evaluate_final(frozen, execution(clock, p.streams[2], reads))
    assert row.metadata.labels_arrived_at == 5 and row.metadata.completed_at == clock.now() == 6
    assert reads == [s.role for s in p.streams]
    assert c.snapshot()["attempts"] == len(c.snapshot()["rows"]) == 3
    with pytest.raises(ValueError):
        c.evaluate_final(frozen, lambda p: pytest.fail("reread final"))
    with pytest.raises(ValueError):
        c.run_development("train", lambda p: pytest.fail("retrain"))
    with pytest.raises(ValueError):
        c.training_rows()
    with pytest.raises(ValueError):
        c.validation_rows()


def test_early_validation_and_final_are_refused_before_source_callback():
    p = protocol()
    clock = LogicalClock()
    c = ControllerChronology(p, clock)
    with pytest.raises(ValueError):
        c.run_development("validation", lambda p: pytest.fail("read validation"))
    with pytest.raises(ValueError):
        c.freeze()
    with pytest.raises(ValueError):
        c.evaluate_final(ControllerFreeze(p, 0), lambda p: pytest.fail("read final"))
    assert c.snapshot()["attempts"] == 0


@pytest.mark.parametrize("fault", ["copy", "foreign", "mutated_clock", "mutated_objective"])
def test_forged_or_mutated_freeze_never_opens_sealed_source(fault):
    p, clock, c, _ = developed()
    receipt = c.freeze()
    if fault == "copy":
        receipt = deepcopy(receipt)
    if fault == "foreign":
        receipt = ControllerFreeze(p, 4)
    if fault == "mutated_clock":
        object.__setattr__(receipt, "frozen_at", 0)
    if fault == "mutated_objective":
        object.__setattr__(receipt.protocol.objective, "memory_weight", 99.0)
    with pytest.raises(ValueError):
        c.evaluate_final(receipt, lambda p: pytest.fail("opened sealed source"))
    assert c.snapshot()["attempts"] == 2


@pytest.mark.parametrize("kind", ["wrong_role", "future_return", "future_label", "policy"])
def test_invalid_metadata_refuses_metric_payload_and_retains_paid_attempt(kind):
    p = protocol()
    clock = LogicalClock()
    c = ControllerChronology(p, clock)
    reads: list[str] = []
    m = ControllerTrialMetadata(p.streams[0], p.policy, 0, 1, 2)
    if kind == "wrong_role":
        m = replace(m, stream=p.streams[2])
    if kind == "future_return":
        m = replace(m, completed_at=3)
    if kind == "future_label":
        m = replace(m, labels_arrived_at=3)
    if kind == "policy":
        m = replace(m, policy=ReferencePolicy("drift"))

    class Poison:
        def __getattribute__(self, key):
            pytest.fail("metric payload accessed before metadata admission")

    with pytest.raises(ValueError):
        c.run_development(
            "train", execution(clock, p.streams[0], reads, metrics=cast(Any, Poison()), metadata=m)
        )
    assert c.snapshot()["attempts"] == 1 and c.snapshot()["stopped"] and not c.snapshot()["rows"]
    with pytest.raises(ValueError, match="stopped"):
        c.run_development("train", lambda p: pytest.fail("retried paid failed work"))


def test_protocol_is_prospective_detached_and_pinned():
    p = protocol()
    with pytest.raises(ValueError, match="before trial"):
        ControllerChronology(p, LogicalClock(1))
    clock = LogicalClock()
    c = ControllerChronology(p, clock)
    object.__setattr__(p.objective, "compute_weight", 99.0)
    assert c.snapshot()["protocol"].objective.compute_weight == 1.0
    c._protocol = replace(c._protocol, objective=replace(c._protocol.objective, memory_weight=2.0))
    with pytest.raises(ValueError, match="original"):
        c.run_development("train", lambda p: pytest.fail("read after objective change"))


def test_reentrant_source_call_cannot_freeze_or_open_another_stream():
    p = protocol()
    clock = LogicalClock()
    c = ControllerChronology(p, clock)

    def run(policy):
        with pytest.raises(ValueError, match="busy"):
            c.freeze()
        with pytest.raises(ValueError, match="busy"):
            c.run_development("validation", lambda p: pytest.fail("nested source"))
        clock.advance_to(2)
        return ControllerTrial(ControllerTrialMetadata(p.streams[0], policy, 0, 1, 2), measures())

    c.run_development("train", run)
    snapshot = c.snapshot()
    object.__setattr__(snapshot["rows"][0].utility, "value", 999.0)
    assert c.training_rows()[0].utility.value == -2.75
