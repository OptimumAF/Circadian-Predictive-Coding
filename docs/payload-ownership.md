# Retained payload ownership and quiescence

R3.6b2a enrolls the supported retained-copy owners needed for coordinated deletion.
It does not erase data or complete R3.6b2/R3.6b/R3.6.

## Files

```text
src/core/payload_ownership.py    holder limits, metadata and reference ports
src/app/payload_ownership.py     weak enrollment and nonblocking ownership lease
src/app/actor_shadow.py         actor/current/retired candidate enrollment
src/app/candidate_checkpoint.py pending views and prepared/failed native models
src/app/serving_promotion.py     prepared/current/rollback bundles and auxiliary data
tests/test_payload_ownership.py limits, failure, lease and actual retained-copy tests
```

Each stable/promotable actor owns one registry. Candidates, their handoff successors,
checkpoint controllers and promotion controllers automatically enroll in that same
registry. Caller-retained retired candidates remain visible while they are alive.
Weak references allow otherwise-dead owners to be collected; metadata snapshots
never expose or serialize raw model/inbox/payload references.

## Opt-in limits

Configure `actor._payload_registry.configure(PayloadOwnershipLimits(live, lifetime))`
before using the future managed deletion API. Configuration is single-use and
cannot reset consumed enrollments. Registration attempts that fail validation or
quota checks do not consume an enrollment. An admitted constructor or handoff
attempt consumes its enrollment even if later initialization fails. Garbage
collection can release live capacity, but never refunds lifetime enrollments.
Legacy unconfigured registries preserve existing holder creation behavior.

These limits bound holder registrations, **not native bytes, nested object graphs,
per-controller model/view capacities, or elapsed data retention**. The coordinator
must additionally enforce those policies. Existing checkpoint preparation budgets
and pending capacities retain their own cumulative limits. Handoff enrollment
refusal leaves the original owner alive and preparation charged; the failed
prepared model remains visible through its controller's owned-model ledger.

## Internal cleanup lease

`registry._lease()` takes the registry lock and then tries every enrolled holder's
operation lock in enrollment order. Every lock must succeed before any retained
reference is enumerated. Lock acquisition never waits. Busy, reentrant or incomplete
owners cause explicit refusal and release all acquired locks before cleanup begins.
Membership cannot change while the lease is held. Supported reference methods read
retained fields without native calls, payload copying or snapshot serialization.

Reference groups cover:

- Actor native models, including current and rollback promotion bundles.
- Live/retired candidate native models and inboxes.
- Checkpoint pending views and prepared models, including failed restores.
- Pending promotion models and their metadata/cache dictionaries.
- Promotable actor current/rollback metadata and prediction caches.

References may overlap across holders. A future coordinator must deduplicate
objects by identity before mutation/accounting and acquire the original sharing/
consent authority too. Use raw groups only while the internal lease is held.
Keeping a returned group outside that lease creates an external caller-held copy.

```python
from contextlib import contextmanager
from src.app.payload_ownership import PayloadOwnershipRegistry
from src.core.payload_ownership import PayloadOwnershipLimits, PayloadReferences

class LocalHolder:
    _payload_ready = True
    @contextmanager
    def _payload_exclusive(self):
        yield
    def _payload_references(self):
        return PayloadReferences()

registry = PayloadOwnershipRegistry()
registry.configure(PayloadOwnershipLimits(1, 1))
holder = LocalHolder()
registry.enroll("candidate", holder)
assert registry.snapshot().total_enrollments == 1
with registry._lease() as groups:
    assert groups[0].holder.kind == "candidate"
    assert groups[0].references.models == ()
```

Why this: cleanup of only the current candidate would miss checkpoint, prepared,
retired and rollback copies. Weak enrollment finds retained owners without creating
additional payload retention. Trying all locks before enumeration avoids deadlock
with existing controller-to-candidate-to-actor operations and avoids partial cleanup
on contention. Incomplete constructors are refused until finalized or collected.

## Explicit limits and next action

This is a trusted retained-field port, not a certificate for arbitrary Python
graphs, custom callbacks or custom holder implementations. Caller source models,
inspection snapshots, callback caches and exception tracebacks are outside the
registry's ownership. Temporary guard models are protected by the surrounding
controller operation lease while guard evaluation runs; they are not retained by
the evaluator after it returns. No RAM overwrite or parameter unlearning is implied.

Next R3.6b2b must integrate deletion/expiry/opt-out with the original manager,
declared record/byte/time policies, supported native erasure/measurement ports,
all enrolled holder cleanup and checkpoint/ticket invalidation. Tests must cover
partial failure, retired inbox work accounting, refused resurrection and unchanged
budgets/clocks/gates. Transient/audit-only admission remains disabled until actual
purge semantics pass. Full original parent criteria remain unchecked.

## Validation

```powershell
python -m pytest tests/test_payload_ownership.py -q
python -m ruff check src tests scripts
python -m ruff format --check src/core/payload_ownership.py src/app/payload_ownership.py tests/test_payload_ownership.py src/app/actor_shadow.py src/app/candidate_checkpoint.py src/app/serving_promotion.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

New tests use deterministic fake learners only; no new native experiment or study.
