# ADR-0230: Capture lifecycle under original nonblocking leases

## Context

The complete lifecycle contract has 66 source fields and 47 reference slots before
capture synchronization. The original driver operation gate does not cover stop,
wake, fault handling or every worker terminal transition. Its scalar snapshot
cannot supply an atomic complete record. Registry cleanup prunes dead weak entries.

## Decision

Add one original driver state RLock and retain it in the current complete contract
(67 fields, 48 reference slots). Every driver field/event mutation participates.
Keep that lock out of native callbacks, waits and joins. The operation lock still
serializes polling and cleanup and refuses reentrance. Stop can signal while a
callback is running; it joins after releasing the state lock.

`capture_managed_lifecycle` tries the original driver operation/state, manager,
registry, all live holder, retention time, copy budget and sharing leases. All
acquisitions are nonblocking; partial acquisition unwinds on refusal. Enumerate
the retained weak map without pruning and pin its live holders through detachment.
Validate exact source schemas, original runtime/clock/budget/sharing relationships,
native metadata, policy aliases and independent aggregate/string bounds before
deepcopy of metadata. Preserve authority references by identity. No measurement
port, clock or model callback runs during capture.

Why this: a short state lock covers transitions without extending blocking across
arbitrary cleanup work. Separate operation/state leases exclude in-flight cleanup
from capture while preserving responsive stop and original expiry allowances.

## Alternatives

Using only the operation lock leaves stop/wake/terminal races. Holding a state lock
through cleanup or join can delay stop or prevent the worker from terminating.
Using registry cleanup enumeration silently removes retained enrollment history.
Serializing live gates or callbacks invents renewed authority.

## Consequences

Busy, incomplete or inconsistent owners refuse capture without publication or
allowance renewal. Returned metadata is detached; reference values remain live
original objects. Thread liveness is an observation at capture, not a promise that
the thread remains alive afterward. Supported public mutators participate; hostile
direct mutation of private objects is outside the original trusted owner contract.
Complete lifecycle encoding, composite recovery and durable original authority
remain separate unfinished acceptance gates.
