# ADR-0298: Expire committed serving bundles and rollback authority

## Context

The pending-promotion expiry control covers owned cache, metadata and prepared
tickets. A committed promotion additionally retains the previous serving bundle
and an issued rollback receipt. Those references could otherwise restore data
after expiry.

## Decision

Extend the shared fixed coupled integration with an explicit committed mode.
Keep the original pending interface and schema. Commit C, retain A as previous,
prime and hit C's cache, then expire both generations under the original lifecycle.

Why this: the application already enumerates current/previous bundle references,
clears owned dictionaries and invalidates rollback authority. A shared transaction
with explicit phase differences exercises this behavior without another runtime
or a duplicate cleanup implementation.

At the deadline, rollback must refuse before publication. After cleanup, the
original receipt must be stale and unable to resurrect any bundle. Preserve
complete native A/B/C state, historical receipts, original clocks/resources and
spent quotas. The committed mode declares 656 lifetime raw-copy bytes; the
unchanged pending mode retains its original 640-byte allowance.

## Alternatives

Duplicating the full integration would make controls drift. Changing core
rollback or cleanup without a demonstrated defect would add unnecessary risk.
Performing an actual rollback before expiry would consume the authority that
this control needs to retain and invalidate.

## Consequences

Committed generation1 advances only by legitimate cleanup passes. Both current
and previous dictionaries clear; the actor drops previous and the controller
drops its latest receipt. Caller aliases still permit trusted forensic native
state checks but grant no renewed public serving authority. Raw-reference cleanup
does not overwrite memory, unlearn parameters or prove physical expiry latency,
continuous service or process recovery. Scientific protocols remain unchanged.
