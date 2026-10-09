"""Bounded fake native row writes under original managed owner and consent."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import gc
from typing import Any
from weakref import ref

import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_erasure import ErasedExperience
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.replay_origin import ReplayOriginLimits, ReplayOriginPorts
from src.core.replay_write_origin import ReplayWriteLimits, begin_replay_write
from src.core.resource_sharing import SharingLimits
from test_managed_experience import declaration


class Payload:
    def __init__(self, value):
        self.value = value


class Snapshot:
    def __init__(self, features, targets):
        self.features, self.targets = deepcopy(features), deepcopy(targets)


class Model:
    def __init__(self):
        self.rows = []


class Learner:
    def __init__(self, model=None):
        self.model = model or Model()
        self.calls = 0

    def fork(self):
        return Learner(deepcopy(self.model))

    def train_batch(self, features, targets):
        self.calls += 1
        window = begin_replay_write(self.model, features, targets, self.model.rows, 1, "rows")
        if window is not None:
            window.before_copy(0, 1)
        snapshot = Snapshot(features, targets)
        if window is not None:
            window.copied(snapshot)
        # Fake native policy alone deduplicates; the ledger never does so by value.
        self.model.rows = [
            row
            for row in self.model.rows
            if (row.features.value, row.targets.value) != (features.value, targets.value)
        ] + [snapshot]
        self.model.rows = self.model.rows[-2:]
        if window is not None:
            window.finish(self.model.rows)
        return TrainingDiagnostic("fake_row_fixture", 1.0)

    def predict(self, features):
        raise AssertionError("prediction is outside this fake scope")

    def snapshot_state(self):
        raise AssertionError("snapshot is outside this fake scope")

    def restore_state(self, state):
        raise AssertionError("restore is outside this fake scope")


def model_reference(learner):
    return learner.model


def retained(model, maximum):
    assert len(model.rows) <= maximum
    return tuple(model.rows)


def copy_bytes(features, targets, start, count, maximum):
    assert (start, count) == (0, 1) and maximum >= 32
    return 32


def payloads(snapshot):
    return snapshot.features, snapshot.targets


def fingerprint(snapshot, maximum):
    assert maximum >= 32
    return 32, sha256((snapshot.features.value + ":" + snapshot.targets.value).encode()).hexdigest()


PORTS = ReplayOriginPorts(model_reference, retained, copy_bytes, payloads, fingerprint)
LIMITS = ReplayOriginLimits(16, 64, 8, 128 * 1024, 120, 4096)
WINDOW = ReplayWriteLimits(16, 4, 64)


def setup(*, limits=LIMITS, ports=PORTS, budget=None):
    clock = LogicalClock()
    budget = budget or ToyBudgetSession(ToyExecutionBudget(max_training_updates=4), lambda: 0.0)
    runtime = ActorShadowRuntime(
        Learner(),
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=4,
    )
    gate = ServingPriorityGate(SharingLimits(1, 8, 1), resource_available=lambda: True)
    owner = ManagedExperienceOwner(
        ResourceSharedRuntime(runtime, gate), limits=LifecycleLimits(4, 4)
    )
    ledger = ManagedReplayOrigins(owner, ports, limits, WINDOW)
    return owner, ledger, runtime, clock, gate, budget


def queue(owner, clock, sample="s1", value="same", subject="person-1"):
    owner.declare(declaration(sample, subject))
    owner.record_experience(
        Experience(
            sample, "e1", 1, "actor-0", Payload(value), "train", ExperiencePermissions(True, True)
        )
    )
    owner.record_label(LabelArrival("label-" + sample, sample, "e1", 3, "actor-0", Payload("0")))
    clock.advance_to(3)


def test_should_bind_duplicates_to_new_original_subject_and_keep_eviction_charges():
    owner, ledger, runtime, clock, gate, budget = setup()
    queue(owner, clock)
    ledger.train_ready()
    assert ledger.origins()[0].key == ("e1", "s1")
    old = ref(runtime._candidate.model.rows[0])
    first = ledger.accounting()
    queue(owner, clock, "s2", subject="person-2")
    ledger.train_ready()
    gc.collect()
    assert old() is None and ledger.origins()[0].key == ("e1", "s2")
    owner.opt_out("person-1")
    assert ledger.origins()[0].subject_id == "person-2"
    for sample, value in [("s3", "other"), ("s4", "more")]:
        queue(owner, clock, sample, value, subject=sample)
        ledger.train_ready()
    assert [row.key[1] for row in ledger.origins()] == ["s3", "s4"]
    final = ledger.accounting()
    assert (
        final.records_created
        == final.invocations_started
        == budget.updates_completed
        == gate.snapshot().admitted_updates
        == 4
    )
    assert final.live_records == 2 and final.metadata_bytes_charged > first.metadata_bytes_charged


def test_should_release_evicted_payloads_without_refunding_metadata():
    owner, ledger, runtime, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    weak = ref(runtime._candidate.model.rows[0].features)
    before = ledger.accounting()
    runtime._candidate.model.rows.clear()  # Simulated native payload erasure;no erasure method invoked.
    gc.collect()
    assert weak() is None and ledger.origins() == ()
    after = ledger.accounting()
    assert (
        after.records_created == before.records_created
        and after.metadata_bytes_charged == before.metadata_bytes_charged
    )
    assert after.live_records == 0


def test_should_refuse_original_source_age_without_renewal():
    owner, ledger, _, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    before = ledger.accounting()
    clock.advance_to(122)
    with pytest.raises(ValueError, match="age expired"):
        ledger.origins()
    assert ledger.accounting() == before


@pytest.mark.parametrize(
    "change", ["learner", "model", "clock", "budget_origin", "guard", "declaration"]
)
def test_should_refuse_foreign_original_roots_or_declaration(change):
    owner, ledger, runtime, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    if change == "learner":
        runtime._candidate = Learner()
    elif change == "model":
        runtime._candidate.model = Model()
    elif change == "clock":
        runtime._inbox._clock = LogicalClock(3)
    elif change == "budget_origin":
        runtime._budget.started_at = 1.0
    elif change == "guard":
        runtime._inbox._training_guard = lambda *_: True
    else:
        owner._catalog[("e1", "s1")] = replace(owner._catalog[("e1", "s1")])
    with pytest.raises(ValueError):
        ledger.origins()


@pytest.mark.parametrize("change", ["payload_reference", "payload_contents", "metadata"])
def test_should_refuse_already_bound_payload_or_metadata_corruption(change):
    owner, ledger, runtime, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    snapshot = runtime._candidate.model.rows[0]
    if change == "payload_reference":
        snapshot.features = Payload("same")
    elif change == "payload_contents":
        snapshot.features.value = "changed"
    else:
        row = ledger._rows[id(snapshot)]
        row.data = replace(row.data, row_start=1)
    with pytest.raises(ValueError, match="changed"):
        ledger.origins()


@pytest.mark.parametrize("limit", ["max_records_created", "max_invocations"])
def test_should_preserve_spent_limits_and_original_work_after_refused_second_write(limit):
    owner, ledger, runtime, clock, gate, budget = setup(limits=replace(LIMITS, **{limit: 1}))
    queue(owner, clock)
    ledger.train_ready()
    first = ledger.accounting()
    queue(owner, clock, "s2", value="other", subject="person-2")
    with pytest.raises(ValueError, match="limit"):
        ledger.train_ready()
    assert budget.updates_completed == 1 and len(runtime.applied_updates) == 1
    assert gate.snapshot().admitted_updates == 2
    assert ledger.accounting().metadata_bytes_charged >= first.metadata_bytes_charged
    with pytest.raises(ValueError):
        ledger.origins()


def test_should_preserve_primary_observer_fault_and_refuse_uncertain_origin():
    failure = RuntimeError("fingerprint fault")

    def refuse(snapshot, maximum):
        raise failure

    owner, ledger, runtime, clock, gate, budget = setup(ports=replace(PORTS, fingerprint=refuse))
    queue(owner, clock)
    with pytest.raises(RuntimeError) as caught:
        ledger.train_ready()
    assert caught.value is failure and runtime._stopped and not runtime.applied_updates
    assert budget.updates_completed == 0 and gate.snapshot().admitted_updates == 1
    assert ledger.accounting().records_created == ledger.accounting().invocations_started == 1
    with pytest.raises(ValueError, match="uncertain"):
        ledger.origins()


def test_should_keep_committed_receipt_after_post_update_budget_stop():
    seconds = [0.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1), lambda: seconds[0]
    )
    owner, ledger, runtime, clock, gate, _ = setup(budget=budget)
    queue(owner, clock)
    original = runtime._candidate.train_batch

    def train(features, targets):
        value = original(features, targets)
        seconds[0] = 2.0
        return value

    runtime._candidate.train_batch = train
    with pytest.raises(ToyExecutionStopped):
        ledger.train_ready()
    assert budget.updates_completed == gate.snapshot().admitted_updates == 1
    assert runtime.applied_updates[0].update_number == 1 and len(runtime._candidate.model.rows) == 1
    assert budget.started_at == 0.0 and budget.last_clock == 2.0
    with pytest.raises(ValueError, match="uncertain"):
        ledger.origins()


def test_should_keep_pre_update_refusal_without_row_or_invocation_charge():
    owner, ledger, runtime, clock, _, budget = setup(
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    )
    queue(owner, clock)
    with pytest.raises(ToyExecutionStopped):
        ledger.train_ready()
    assert (
        ledger.origins() == ()
        and ledger.accounting().records_created == ledger.accounting().invocations_started == 0
    )
    assert not runtime._stopped and budget.updates_completed == runtime._candidate.calls == 0


def test_should_refuse_untracked_bypass_payload_even_when_content_matches():
    owner, ledger, _, clock, _, _ = setup()
    queue(owner, clock)
    owner.train_ready()
    with pytest.raises(ValueError, match="untracked"):
        ledger.origins()
    assert ledger.accounting().records_created == 0


def test_should_refuse_current_subject_optout_without_refunding_or_erasing_payload():
    owner, ledger, runtime, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    before = ledger.accounting()
    owner.opt_out("person-1")
    with pytest.raises(ValueError, match="opted out"):
        ledger.origins()
    assert len(runtime._candidate.model.rows) == 1 and ledger.accounting() == before


def test_should_release_original_owner_and_payload_graph_when_only_ledger_survives():
    owner, ledger, runtime, clock, gate, budget = setup()
    queue(owner, clock)
    ledger.train_ready()
    owner_ref, payload_ref = ref(owner), ref(runtime._candidate.model.rows[0].features)
    del owner, runtime, clock, gate, budget
    gc.collect()
    assert owner_ref() is None and payload_ref() is None
    with pytest.raises(ValueError, match="expired"):
        ledger.origins()


def test_should_refuse_ledger_and_owner_reentry_during_original_write():
    holder: dict[str, Any] = {}

    def observed(snapshot, maximum):
        with pytest.raises(PayloadOwnershipBusy):
            holder["ledger"]().origins()
        with pytest.raises(PayloadOwnershipBusy):
            holder["owner"]().train_ready()
        return fingerprint(snapshot, maximum)

    owner, ledger, runtime, clock, _, _ = setup(ports=replace(PORTS, fingerprint=observed))
    holder.update(ledger=ref(ledger), owner=ref(owner))
    queue(owner, clock)
    assert ledger.train_ready().updates[0] is runtime.applied_updates[0]
    assert ledger.origins()[0].key == ("e1", "s1")


@pytest.mark.parametrize("reference", ["source", "label", "declaration"])
def test_should_refuse_expired_original_reference_even_when_equal_record_is_installed(reference):
    owner, ledger, runtime, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    row = next(iter(ledger._rows.values()))
    original = getattr(row, reference)
    if reference == "source":
        runtime._inbox._experiences[row.data.key] = replace(original())
    elif reference == "label":
        runtime._inbox._labels[row.data.key] = replace(original())
    else:
        owner._catalog[row.data.key] = replace(original())
    gc.collect()
    assert original() is None
    with pytest.raises(ValueError, match="expired|changed"):
        ledger.origins()


@pytest.mark.parametrize("revocation", ["tombstone", "revoked"])
def test_should_refuse_original_tombstone_or_revoked_key_without_erasing_native_payload(revocation):
    owner, ledger, runtime, clock, _, _ = setup()
    queue(owner, clock)
    ledger.train_ready()
    before = ledger.accounting()
    if revocation == "tombstone":
        runtime._inbox._erased[("e1", "s1")] = ErasedExperience(
            ("e1", "s1"), "actor-0", 1, "label-s1", 3, 3, "deleted"
        )
    else:
        owner._revoked_keys.add(("e1", "s1"))
    with pytest.raises(ValueError, match="tombstone|revoked"):
        ledger.origins()
    assert len(runtime._candidate.model.rows) == 1 and ledger.accounting() == before


def test_should_refuse_metadata_charge_before_native_copy_without_refunding_invocation():
    owner, ledger, runtime, clock, _, budget = setup(
        limits=replace(LIMITS, max_metadata_bytes=1024)
    )
    queue(owner, clock)
    with pytest.raises(ValueError, match="identifier|limit"):
        ledger.train_ready()
    assert runtime._candidate.model.rows == [] and budget.updates_completed == 0
    assert ledger.accounting().records_created == 0
    assert ledger.accounting().invocations_started == 1
    assert ledger.accounting().metadata_bytes_charged == 1024
