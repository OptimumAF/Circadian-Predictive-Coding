# ADR-0274: Supervised checkpoint writer interruption

## Context

The complete coupled codec and bounded file store are accepted. Existing Windows
adapters retain physical-process identities, and the surviving coordinator keeps
spent counters in SQLite. Those parts have not yet demonstrated an actual native
checkpoint staging child being terminated while its parent actor remains usable.
Native update admission intentionally remains uncertain without a completion
protocol; existing handoff charges another checkpoint attempt.

## Decision

Compose a bounded checkpoint-only process example using these unchanged ports.
Prepare native A/B fixtures before authority creation; children never train.
Reserve a stage attempt and copy ceiling before sending B. Pause at the file
store's replace boundary and finish execute while its child remains alive. Only
then terminate/reap the owned child and prove ended through its retained handle.
Verify A and the exact staged B, then remove that owned orphan. Handoff to a new
read-only restoring child under the same parent, original limits and counters.
Set attempt cap 2 for reservation plus existing handoff charge; refuse request 3
before its callback while the replacement is alive. Actor A remains unchanged.

Use a small standard-library bounded JSON child transport because existing
single-line test fixtures do not bound stdin blocking, aggregate output or thread
cleanup. Launch the existing physical base interpreter with the current venv's
package path to avoid Windows redirector PID confusion. Verify hello identity,
but derive process ownership only from Popen and retained OS registration.

## Alternatives

Resuming training after a native admission would bypass uncertain-work refusal.
Resetting counters in a new coordinator would discard original authority. Killing
inside execute preparation fails its live-worker postchecks and prevents handoff.
Using cap 1 would ignore the existing handoff charge. Unbounded communicate or
closing pipes while blocked readers hold their locks can hang the parent actor.

## Consequences

The supported checkpoint writer crash path becomes runnable and testable with
full model-state recovery, exact actor behavior and retained spent attempts.
Finite IO deadlines and owned cleanup fail loudly; no automatic retry or refund.
Coordinator loss, native completion, resumed learning, full lifecycle authority,
power loss and process-tree termination remain separate unfinished requirements.
The full original R3.7/R3.5b2 acceptance criteria remain unchanged.
