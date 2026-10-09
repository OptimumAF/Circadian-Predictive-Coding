"""Owned-copy enrollment and quiescence without native mutation or payload copy."""

from contextlib import contextmanager
import gc
import pickle
from threading import Lock
from typing import Any
import weakref

import pytest

from src.app.payload_ownership import PayloadOwnershipRegistry
from src.core.payload_ownership import PayloadOwnershipLimits, PayloadReferences
from test_resource_sharing import shared
from test_candidate_checkpoint import controller
from test_serving_promotion import setup as promotion_setup, prepare


class Holder:
    def __init__(self):
        self._payload_ready = True
        self.gate = Lock()
        self.calls = 0

    @contextmanager
    def _payload_exclusive(self):
        if not self.gate.acquire(blocking=False):
            raise ValueError("holder busy")
        try:
            yield
        finally:
            self.gate.release()

    def _payload_references(self):
        self.calls += 1
        return PayloadReferences(models=(self,))


@pytest.mark.parametrize("values", [(True, 1), (-1, 1), (1, 1.0), (1, -1)])
def test_should_reject_invalid_ownership_limits(values):
    with pytest.raises(ValueError):
        PayloadOwnershipLimits(*values)


def test_should_keep_weak_membership_but_never_refund_lifetime_enrollment():
    registry = PayloadOwnershipRegistry()
    registry.configure(PayloadOwnershipLimits(1, 2))
    first = Holder()
    registry.enroll("actor", first)
    reference = weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None and registry.snapshot().holders == ()
    second = Holder()
    registry.enroll("candidate", second)
    assert registry.snapshot().total_enrollments == 2
    del second
    gc.collect()
    with pytest.raises(ValueError, match="lifetime"):
        registry.enroll("checkpoint", Holder())
    assert registry.snapshot().total_enrollments == 2


def test_should_bound_live_holders_and_refuse_configuration_renewal():
    registry = PayloadOwnershipRegistry()
    holder = Holder()
    registry.enroll("actor", holder)
    with pytest.raises(ValueError, match="live"):
        registry.configure(PayloadOwnershipLimits(0, 2))
    registry.configure(PayloadOwnershipLimits(1, 3))
    with pytest.raises(ValueError, match="live"):
        registry.enroll("candidate", Holder())
    with pytest.raises(ValueError, match="configured"):
        registry.configure(PayloadOwnershipLimits(10, 10))
    assert registry.snapshot().total_enrollments == 1


@pytest.mark.parametrize("kind", ["role", "holder", "duplicate", "incomplete"])
def test_should_refuse_bad_or_duplicate_enrollment_and_incomplete_quiescence(kind):
    registry = PayloadOwnershipRegistry()
    holder = Holder()
    if kind == "role":
        bad_role: Any = "unknown"
        with pytest.raises(ValueError):
            registry.enroll(bad_role, holder)
    elif kind == "holder":
        bad_holder: Any = object()
        with pytest.raises(ValueError):
            registry.enroll("actor", bad_holder)
    elif kind == "duplicate":
        registry.enroll("actor", holder)
        with pytest.raises(ValueError, match="enrolled"):
            registry.enroll("actor", holder)
    else:
        holder._payload_ready = False
        registry.enroll("actor", holder)
        with pytest.raises(ValueError, match="incomplete"):
            with registry._lease():
                pass
        assert holder.calls == 0


def test_should_acquire_every_owner_before_enumerating_any_payload_reference():
    registry = PayloadOwnershipRegistry()
    first, second = Holder(), Holder()
    registry.enroll("actor", first)
    registry.enroll("candidate", second)
    with second._payload_exclusive(), pytest.raises(ValueError, match="busy"):
        with registry._lease():
            pass
    assert first.calls == second.calls == 0
    with registry._lease() as groups:
        assert [g.holder.kind for g in groups] == ["actor", "candidate"]
        assert groups[0].references.models == (first,) and groups[1].references.models == (second,)
        with pytest.raises(ValueError, match="busy"):
            registry.enroll("checkpoint", Holder())
    assert first.calls == second.calls == 1
    assert first.gate.acquire(blocking=False)
    first.gate.release()
    assert second.gate.acquire(blocking=False)
    second.gate.release()


def test_should_refuse_reentrant_enumeration_and_release_all_leases_after_failure():
    registry = PayloadOwnershipRegistry()
    holder = Holder()
    registry.enroll("actor", holder)

    def references():
        with registry._lease():
            pass

    setattr(holder, "_payload_references", references)
    with pytest.raises(ValueError, match="busy"):
        with registry._lease():
            pass
    setattr(holder, "_payload_references", lambda: PayloadReferences())
    with registry._lease() as groups:
        assert len(groups) == 1


@pytest.mark.parametrize("field", ["models", "inboxes", "snapshots", "auxiliary"])
def test_should_reject_mutable_reference_collections(field):
    invalid: Any = []
    with pytest.raises(ValueError):
        PayloadReferences(**{field: invalid})


def test_should_enroll_actual_actor_candidate_checkpoint_models_and_pending_views():
    wrapper, runtime, _, gate, source, budget = shared()
    control = controller(wrapper)
    gate.pause()
    token = control.capture()
    registry = runtime.actor._payload_registry
    before = pickle.dumps(runtime.candidate_snapshot()), budget.updates_completed, gate.snapshot()
    with registry._lease() as groups:
        by_kind = {g.holder.kind: g.references for g in groups}
        assert by_kind["actor"].models == (runtime.actor._learner,)
        assert by_kind["candidate"].models == (runtime._candidate,) and by_kind[
            "candidate"
        ].inboxes == (runtime._inbox,)
        assert by_kind["checkpoint"].snapshots == (control._pending[token].view,)
        with pytest.raises(ValueError, match="busy"):
            control.capture()
        with pytest.raises(ValueError, match="busy"):
            runtime.train_ready()
    assert (
        pickle.dumps(runtime.candidate_snapshot()),
        budget.updates_completed,
        gate.snapshot(),
    ) == before
    assert source.value == [2, 2]


def test_should_share_original_registry_with_retired_and_replacement_candidate():
    wrapper, old, _, gate, _, budget = shared()
    registry = old.actor._payload_registry
    registry.configure(PayloadOwnershipLimits(8, 8))
    control = controller(wrapper)
    gate.pause()
    new = control.restore(control.capture())
    assert new.actor._payload_registry is registry
    with registry._lease() as groups:
        candidates = [g.references for g in groups if g.holder.kind == "candidate"]
        assert len(candidates) == 2 and any(c.models == (old._candidate,) for c in candidates)
        assert any(c.models == (new._candidate,) for c in candidates)
        checkpoints = [g.references for g in groups if g.holder.kind == "checkpoint"]
        assert checkpoints[0].models == (new._candidate,)
    assert old._retired and new._budget is budget and registry.snapshot().total_enrollments == 4


def test_should_collect_all_prepared_current_and_rollback_promotion_bundles():
    actor, runtime, control = promotion_setup()
    registry = actor._payload_registry
    ticket = prepare(control, runtime)
    prepared = control._pending[ticket].bundle.learner
    with registry._lease() as groups:
        promotion = [g.references for g in groups if g.holder.kind == "promotion"][0]
        assert promotion.models == (prepared,)
        assert promotion.auxiliary == (
            control._pending[ticket].bundle.metadata,
            control._pending[ticket].bundle.cache,
        )
    receipt = control.commit(runtime, ticket)
    with registry._lease() as groups:
        actor_refs = [g.references for g in groups if g.holder.kind == "actor"][0]
        assert actor_refs.models == (actor._slot.bundle.learner, actor._slot.previous.learner)
        with pytest.raises(ValueError, match="busy"):
            control.rollback(receipt)
    assert control.rollback(receipt) == 2


def test_should_refuse_candidate_quota_before_native_fork_and_leave_original_budget():
    from src.app.actor_shadow import ActorShadowRuntime
    from src.core.experience import LogicalClock
    from test_actor_shadow import PausableLearner

    wrapper, runtime, _, _, _, budget = shared()
    registry = runtime.actor._payload_registry
    registry.configure(PayloadOwnershipLimits(2, 2))

    class NoFork(PausableLearner):
        def fork(self):
            raise AssertionError("quota denial touched native source")

    with pytest.raises(ValueError, match="live|lifetime"):
        ActorShadowRuntime(
            NoFork(),
            actor_version="actor-0",
            candidate_version="next",
            clock=LogicalClock(),
            budget=budget,
            actor=runtime.actor,
        )
    assert budget.updates_completed == 0 and registry.snapshot().total_enrollments == 2


def test_should_refuse_new_controller_when_lifetime_capacity_is_exhausted():
    wrapper, runtime, _, _, _, _ = shared()
    registry = runtime.actor._payload_registry
    registry.configure(PayloadOwnershipLimits(4, 3))
    first = controller(wrapper)
    reference = weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None
    with pytest.raises(ValueError, match="lifetime"):
        controller(wrapper)
    assert registry.snapshot().total_enrollments == 3


def test_should_enumerate_failed_checkpoint_model_without_native_snapshot_or_payload_copy():
    from test_actor_shadow import PausableLearner

    wrapper, runtime, _, gate, _, budget = shared()

    def build(state):
        model = PausableLearner(state)
        model.phase = "restore_error"
        return model

    control = controller(wrapper, build_learner=build)
    gate.pause()
    token = control.capture()
    with pytest.raises(RuntimeError):
        control.restore(token)
    model = control._models[0]
    model.snapshot_state = lambda: (_ for _ in ()).throw(
        AssertionError("enumeration called native")
    )
    with runtime.actor._payload_registry._lease() as groups:
        checkpoint = [g.references for g in groups if g.holder.kind == "checkpoint"][0]
        assert checkpoint.models == (model,) and checkpoint.snapshots == (
            control._pending[token].view,
        )
    assert control._attempts == 1 and budget.updates_completed == 0 and wrapper._runtime is runtime


def test_should_preserve_old_owner_and_spent_preparation_when_handoff_enrollment_is_denied():
    wrapper, runtime, _, gate, _, budget = shared()
    registry = runtime.actor._payload_registry
    control = controller(wrapper)
    registry.configure(PayloadOwnershipLimits(3, 3))
    gate.pause()
    token = control.capture()
    before = runtime.candidate_snapshot()
    with pytest.raises(ValueError, match="live|lifetime"):
        control.restore(token)
    assert (
        wrapper._runtime is runtime
        and not runtime._retired
        and runtime.candidate_snapshot() == before
    )
    assert control._attempts == 1 and len(control._models) == 1 and token in control._pending
    assert budget.updates_completed == 0 and registry.snapshot().total_enrollments == 3


def test_should_expose_metadata_snapshot_without_raw_owner_or_payload_serialization():
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("copied payload")

        def __reduce__(self):
            raise AssertionError("serialized payload")

    registry = PayloadOwnershipRegistry()
    holder = Holder()
    raw = CannotCopy()
    setattr(holder, "_payload_references", lambda: PayloadReferences(snapshots=(raw,)))
    registry.enroll("actor", holder)
    assert pickle.dumps(registry.snapshot())
    with registry._lease() as groups:
        assert groups[0].references.snapshots[0] is raw
