"""Real queued lease exits must not follow qualified history publication.

Eight original numeric graphs, no native model/snapshot/restore work. Wrappers
record only bounded scalar observations; all assertions run outside the scopes
whose entry failures would otherwise be caught by the optional observer.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json

import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.payload_ownership import PayloadOwnershipBusy
from test_automatic_inbox_expiry import (
    AutomaticWork,
    FIRST,
    _assert_authority,
    _assert_successful_raw_cleanup,
    _assert_unpublished,
    _authority,
    _fresh_automatic,
    _observer_refusal,
)


@pytest.fixture(scope="module")
def release_work(tmp_path_factory):
    work = AutomaticWork()
    try:
        yield work
    finally:
        (tmp_path_factory.mktemp("release-work") / "work.json").write_text(
            json.dumps(asdict(work), indent=2), encoding="utf8"
        )
    assert work.graphs == work.updates == 8
    assert work.learners == 24 and work.forks == 16
    assert work.arrays == 48 and work.array_bytes == 576
    assert work.consumed_arrays == work.native_copy_arrays == 16
    assert work.consumed_bytes == work.native_copy_bytes == 192
    assert work.cleanup_calls == 15 and work.post_erase_probes == 7
    assert work.pending_consumed is None


@dataclass
class ExitRecord:
    entered: int = 0
    exits: int = 0
    fired: int = 0
    unpublished: bool = True
    protocol: bool = True


def _record_exit(graph, before, record, *, failing=False):
    record.exits += 1
    cleaned = (
        not graph.runtime._candidate.model.rows
        and not graph.runtime._inbox._experiences
        and not graph.runtime._inbox._labels
    )
    if cleaned or failing:
        record.fired += 1
        history = graph.history
        record.unpublished = record.unpublished and (
            history._storage is before["maps"][0]
            and history._erased_storage is before["maps"][1]
            and history._untrained_storage is before["maps"][2]
            and {row.data.key for row in history._records.values()} == {FIRST}
            and not history._erased_records
            and not history._untrained_records
        )


def _shadow(graph, target, name, before, *, transient=False, failing=False, exit_error=None):
    original = getattr(type(target), name)
    record = ExitRecord()

    @contextmanager
    def wrapper():
        if transient:
            del target.__dict__[name]
        record.entered += 1
        try:
            with original(target) as value:
                if name == "_checkpoint_lease":
                    record.protocol = record.protocol and target._checkpointing is True
                yield value
        finally:
            if name == "_checkpoint_lease":
                record.protocol = record.protocol and target._checkpointing is False
            _record_exit(graph, before, record, failing=failing)
            if exit_error is not None:
                raise exit_error

    target.__dict__[name] = wrapper
    return record


def _assert_released(graph, controller=None):
    assert not graph.owner._gate.locked() and not graph.life._registry._gate.locked()
    assert not graph.runtime.actor._read_gate.locked()
    assert not graph.runtime._write_gate.locked() and not graph.ledger._gate.locked()
    assert graph.gate._checkpointing is False
    if controller is not None:
        assert not controller._gate.locked()


def _assert_fired_refusal(graph, before, record, controller=None):
    _observer_refusal(graph, before)
    assert record.fired == 1 and record.unpublished and record.protocol
    assert record.entered == record.exits
    _assert_released(graph, controller)


def test_should_publish_with_original_promotable_dispatch(release_work):
    graph, ports = _fresh_automatic(release_work, actor=True)
    before = _authority(graph)
    slot = graph.runtime.actor._slot
    graph.clock.advance_to(122)
    report = graph.life.expire()
    _assert_successful_raw_cleanup(graph, report, before)
    assert not graph.history._records and set(graph.history._erased_records) == {FIRST}
    assert not graph.history._untrained_records
    current = graph.runtime.actor._slot
    assert current.generation == slot.generation + 1 and current.previous is None
    assert not current.bundle.cache and not current.bundle.metadata
    assert ports.fired == 0
    _assert_released(graph)


def test_should_refuse_transient_actor_payload_lease_shadow_before_queued_exit(release_work):
    graph, _ = _fresh_automatic(release_work)
    before = _authority(graph)
    record = _shadow(graph, graph.runtime.actor, "_payload_exclusive", before, transient=True)
    graph.clock.advance_to(122)
    _assert_fired_refusal(graph, before, record)


def test_should_refuse_delegated_checkpoint_exclusive_shadow_before_queued_exit(release_work):
    graph, _ = _fresh_automatic(release_work)

    def no_native(*args):
        pytest.fail("release proof must not invoke checkpoint native ports")

    controller = CandidateCheckpointController(
        graph.owner._shared,
        build_learner=no_native,
        state_digest=no_native,
        policy_digest=no_native,
        max_pending=1,
        max_prepare_attempts=1,
    )
    controller._models.append(graph.runtime._candidate)
    before = _authority(graph)
    graph.clock.advance_to(122)
    busy_record = _shadow(
        graph,
        graph.runtime.actor,
        "_payload_exclusive",
        before,
        transient=True,
        failing=True,
        exit_error=RuntimeError("unsafe busy exit"),
    )
    controller._gate.acquire()
    try:
        with pytest.raises(PayloadOwnershipBusy):
            graph.life.expire()
        assert graph.runtime._candidate.model.rows and graph.runtime._inbox._experiences
        assert not graph.owner._revoked_keys and not graph.life._failed
        _assert_unpublished(graph, before)
        assert controller._gate.locked()
        assert not graph.owner._gate.locked() and not graph.life._registry._gate.locked()
        assert (
            not graph.runtime._write_gate.locked() and not graph.runtime.actor._read_gate.locked()
        )
        assert not graph.ledger._gate.locked()
        assert busy_record.entered == busy_record.exits == busy_record.fired == 1
        assert busy_record.unpublished
    finally:
        controller._gate.release()
    record = _shadow(graph, controller, "_exclusive", before, transient=True)
    _assert_fired_refusal(graph, before, record, controller)


def test_should_refuse_owner_operation_exit_shadow_before_publication(release_work):
    graph, _ = _fresh_automatic(release_work)
    before = _authority(graph)
    record = _shadow(graph, graph.owner, "_operation", before)
    graph.clock.advance_to(122)
    _assert_fired_refusal(graph, before, record)
    assert record.exits == 2  # Due exit precedes raw removal and cannot satisfy fired.


def test_should_refuse_registry_lease_exit_shadow_before_publication(release_work):
    graph, _ = _fresh_automatic(release_work)
    before = _authority(graph)
    record = _shadow(graph, graph.life._registry, "_lease", before)
    graph.clock.advance_to(122)
    _assert_fired_refusal(graph, before, record)


def test_should_refuse_sharing_checkpoint_exit_shadow_before_publication(release_work):
    graph, _ = _fresh_automatic(release_work)
    before = _authority(graph)
    record = _shadow(graph, graph.gate, "_checkpoint_lease", before)
    graph.clock.advance_to(122)
    _assert_fired_refusal(graph, before, record)


def test_should_preserve_original_native_error_over_shadow_exit_observation(release_work):
    graph, ports = _fresh_automatic(release_work)
    before = _authority(graph)
    native_error, exit_error = RuntimeError("original native failure"), RuntimeError("unsafe exit")
    ports.native_error = native_error
    record = _shadow(
        graph,
        graph.runtime.actor,
        "_payload_exclusive",
        before,
        failing=True,
        exit_error=exit_error,
    )
    graph.clock.advance_to(122)
    with pytest.raises(RuntimeError) as failure:
        graph.life.expire()
    assert failure.value is native_error
    assert record.entered == record.exits == record.fired == 1 and record.unpublished
    assert graph.life._failed and graph.runtime._stopped
    assert graph.runtime._candidate.model.rows and graph.runtime._inbox._experiences
    assert graph.owner._revoked_keys == {FIRST}
    _assert_authority(graph, before)
    _assert_unpublished(graph, before)
    _assert_released(graph)


def test_should_refuse_transient_class_candidate_lease_shadow_before_queued_exit(
    release_work, monkeypatch
):
    graph, _ = _fresh_automatic(release_work)
    before = _authority(graph)
    original = ActorShadowRuntime._payload_exclusive
    record = ExitRecord()

    @contextmanager
    def wrapper(runtime):
        monkeypatch.setattr(ActorShadowRuntime, "_payload_exclusive", original)
        record.entered += 1
        try:
            with original(runtime) as value:
                yield value
        finally:
            _record_exit(graph, before, record)

    monkeypatch.setattr(ActorShadowRuntime, "_payload_exclusive", wrapper)
    graph.clock.advance_to(122)
    _assert_fired_refusal(graph, before, record)
