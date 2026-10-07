# ADR-0155: Bind the joint worker and count rolled-back execution

Date: 2026-09-30. Status: accepted; c1/c2/c3 and complete unscored gate verified.

## Context

The independent JSON gate covers all six families, but does not prove that
the current worker produced the facts. Existing adapters check RSS after
training and derive updates from saved facts. Restoring a rejected proposal
also restores model clocks, so those clocks cannot enforce actual work.
The frozen joint budget includes copies, validation and serialization.

## Decision

Split P6.7b2c into exact scope/source/request binding, live resource/artifact
failure gates, then two complete reserved executions and independent readback.
Retain all original criteria and the prospective 16,000-update/600-second/
observed 512-MiB at 5 ms limits. Keep all old scientific sources unchanged.

Bind the exact P6.7a saved bytes and revalidate its complete development
references and usage inventory before a request or reserved source. Record
explicit hashes for the complete local execution dependencies, the adapter,
resolved manifest, command and environment. The child must check the saved
request and current identities before training; readers check them again.

Use a scoped boundary wrapper around baseline `train_epoch` and circadian
`_run_training_step`. The latter is the shared wake and replay optimizer
seam, including inherited parent controls. Count successful executions in an
observer outside model snapshots, stop before an update exceeds the cap,
and compare the total to independently derived complete JSON work. Restore
the original methods on success or failure. This changes no model state,
learning rule, guard, RNG, baseline setting or source identity.

Start the existing RSS sampler before the child binding/training boundary.
Check observed peak RSS and wall time before and after optimizer calls and
at validation/serialization boundaries; the parent also enforces the complete
child timeout. Sampling is observed RSS, with possible missed brief peaks;
it is not an OS allocation limit. Fail when telemetry is unavailable.

Publish an exclusive request before child launch, followed by a verified
result and audit only after all gates. Preserve failure facts and occupied
bytes. A failed or partial bundle cannot pass independent readback.
Use a cooperative claim and preserve noncooperating request collisions
without adding our failure marker. Bind intended request/result encoding
digests before writing; changed published bytes must fail before an audit.

## Alternatives

Editing pinned core methods changes the scientific references. Counting
restored clocks loses rejected replay work. Trusting only saved totals misses
live budget overrun. Raising the budget after a failure changes the planned
experiment. Treating request or test fixtures as confirmation evidence would
close the original execution criterion without its required observations.

## Consequences

Development fixtures and injected failures precede any reserved builder.
Every complete reserved cell and negative/null outcome remains required in
both bounded runs. P6.7c/P6.11 must still freeze independent scoring and
uncertainty before any final value. No successful scientific run is claimed
until the two complete bundles and exact equality are independently checked.

## Correctness evidence before data

All 108 new/444 related tests pass in 76.81 s, zero skipped, after reproducing
and repairing three publication-byte/collision regressions. Scoped counters
match all six development families and keep 12/72 forced rejected schedule/
combined updates. Last-family A parameters, B selector RNG and role labels
are rejected after independent validation. Strict scope/source/request,
resource, child and exclusive-artifact failures are exercised before reserved
builders. Metadata-only IO fixtures are not scientific training evidence.
Ruff/seven-file format/mypy 387 pass; exact twelve historical bundles/twenty
usage files and scope still revalidate. The inspected 79-file local closure
is bound; no original pin, baseline, metric, setting or seed changes.

## Complete reserved execution evidence

Both full 560-cell unscored child processes and public independent readbacks
exit 0. The complete result bytes repeat at SHA
`3d85c60627de63769d0f0fc0bf5ec781c77d50468673dab466d8bbe28089e547`.
Each retains all sixty rows and independently observes/derives 15,210 updates,
including 46 rejected replay calls. Worker time 32.50/32.65 s and observed
RSS peaks 462,479,360/462,348,288 bytes remain within unchanged limits.
All 79 local sources, exact scope/request/result/audit, held live states,
raw work and observed model-kind/resource identities verify in each process
and readback. No failure or remaining claim; no outer/final score or tuning.
`docs/p67-confirmation-training-results.md` records complete seeds, costs,
resources, commands and artifact byte identities. C3/c/b2/b are complete;
P6.7/P6.7c/P6.11 and original matrix/final/analysis criteria remain open.
