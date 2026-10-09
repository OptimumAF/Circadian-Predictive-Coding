"""Bounded original reservation and one NEW zero-native managed graph."""

from threading import Lock, Thread

import pytest

from src.app.managed_lifecycle_capture import _lease_lifecycle_sources
from src.app.managed_record_capture import capture_managed_records
from src.app.payload_copy_budget import PayloadCopyBudget
from src.app.payload_ownership import PayloadOwnershipBusy
from src.core.experience import LogicalClock
from src.core.payload_bytes import PayloadCopyLimits
from src.adapters.numpy_learners import BackpropLearner
from test_managed_record_capture import LIMITS, make_owner


def test_should_charge_attempts_inside_original_lease_and_keep_charge_after_failure():
    budget = PayloadCopyBudget(PayloadCopyLimits(10))
    with pytest.raises(RuntimeError, match="copy failed"):
        with budget._lease() as reserve:
            reserve(3)
            reserve(7)
            with pytest.raises(PayloadOwnershipBusy):
                budget.reserve(0)
            with pytest.raises(ValueError, match="exhausted"):
                reserve(1)
            raise RuntimeError("copy failed")
    assert budget.snapshot(0).charged_bytes == 10
    with pytest.raises(ValueError, match="outside"):
        reserve(0)
    with pytest.raises(ValueError, match="exhausted"):
        budget.reserve(1)
    budget.reserve(0)
    assert budget.snapshot(10).charged_bytes == 10


@pytest.mark.parametrize("size", [True, False, -1, 1.0, "1", None])
def test_should_refuse_invalid_size_before_charging(size):
    budget = PayloadCopyBudget(PayloadCopyLimits(8))
    with budget._lease() as reserve:
        with pytest.raises(ValueError):
            reserve(size)
    assert budget.snapshot(0).charged_bytes == 0
    with pytest.raises(ValueError):
        budget.reserve(size)
    assert budget.snapshot(0).charged_bytes == 0


def test_should_refuse_cross_thread_capability_without_charging():
    budget = PayloadCopyBudget(PayloadCopyLimits(8))
    errors: list[str] = []
    with budget._lease() as reserve:

        def attempt():
            try:
                reserve(1)
            except ValueError as error:
                errors.append(str(error))

        thread = Thread(target=attempt)
        thread.start()
        thread.join(1)
        assert not thread.is_alive()
        assert len(errors) == 1 and "thread" in errors[0]
        reserve(2)
    assert budget.snapshot(0).charged_bytes == 2


@pytest.mark.parametrize("field", ["_gate", "_limits"])
def test_should_refuse_changed_original_gate_or_policy(field):
    budget = PayloadCopyBudget(PayloadCopyLimits(8))
    original = getattr(budget, field)
    with budget._lease() as reserve:
        setattr(budget, field, Lock() if field == "_gate" else PayloadCopyLimits(8))
        with pytest.raises(ValueError, match="changed"):
            reserve(1)
        setattr(budget, field, original)
        reserve(2)
    assert budget.snapshot(0).charged_bytes == 2


def test_should_refuse_busy_original_gate_without_yield_or_charge():
    budget = PayloadCopyBudget(PayloadCopyLimits(8))
    with budget._gate:
        with pytest.raises(PayloadOwnershipBusy):
            with budget._lease():
                pytest.fail("busy lease body ran")
    assert budget.snapshot(0).charged_bytes == 0


def test_should_not_revive_expired_capability_during_next_lease():
    budget = PayloadCopyBudget(PayloadCopyLimits(8))
    with budget._lease() as first:
        first(1)
    with budget._lease() as second:
        with pytest.raises(ValueError, match="outside"):
            first(1)
        second(2)
    assert budget.snapshot(0).charged_bytes == 3


def test_should_reserve_under_complete_original_source_gates_before_observation(
    monkeypatch, tmp_path
):
    for name in ("train_batch", "predict", "snapshot_state", "restore_state"):
        monkeypatch.setattr(BackpropLearner, name, lambda *a, **k: pytest.fail("native work"))
    owner, life, runtime, budget, driver, ports = make_owner(0)
    before = [port.calls for port in ports]
    monkeypatch.setattr(LogicalClock, "now", lambda *a: pytest.fail("clock read"))
    copy = life._copy_budget
    assert copy is not None
    with pytest.raises(RuntimeError, match="copy attempt failed"):
        with _lease_lifecycle_sources(owner, limits=LIMITS) as (source, holders, reserve):
            assert source is life and len(holders) == 2 and reserve is not None
            for gate in (
                owner._gate,
                life._registry._gate,
                runtime._write_gate,
                runtime.actor._read_gate,
                life._time_gate,
                copy._gate,
                life._sharing._gate,
            ):
                assert not gate.acquire(blocking=False)
            reserve(56)
            assert copy._charged == 80
            raise RuntimeError("copy attempt failed")
    assert reserve is not None
    with pytest.raises(ValueError, match="outside"):
        reserve(1)
    gates = (
        owner._gate,
        life._registry._gate,
        runtime._write_gate,
        runtime.actor._read_gate,
        life._time_gate,
        copy._gate,
        life._sharing._gate,
    )
    assert driver is not None
    for gate in (*gates, driver._operation_gate, driver._state_gate):
        # Driver state is an RLock: its operation gate is the nonreentrant
        # admission boundary; same-thread state contention cannot be simulated.
        if gate is driver._state_gate:
            continue
        with gate:
            with pytest.raises(PayloadOwnershipBusy):
                with _lease_lifecycle_sources(owner, limits=LIMITS):
                    pytest.fail("busy source body ran")
        assert copy.snapshot(24).charged_bytes == 80
    record = capture_managed_records(owner, limits=LIMITS)
    assert record.metadata.lifecycle.copy is not None
    assert record.metadata.lifecycle.copy.charged_bytes == 80
    assert record.metadata.owner.revision == 5
    assert [port.calls for port in ports] == before
    assert budget.updates_completed == 0 and life._admitted_bytes == 24
    assert driver is not None and driver._thread is None and driver._polls == 0
    (tmp_path / "new-copy-lease-native-work.json").write_text(
        '{"graphs":1,"managed_refusals":1,"snapshots":0,"restores":0,"updates":0,'
        '"predictions":0,"workers":0,"threads":0,"ingress_bytes":24,"charged_bytes":80}',
        encoding="utf8",
    )
