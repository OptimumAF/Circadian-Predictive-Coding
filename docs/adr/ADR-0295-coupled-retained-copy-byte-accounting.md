# ADR-0295: Charge coupled raw copies through the existing lifecycle ledger

Date: 2026-10-15. Status: accepted; R3.7y validated with retained harness warning in RESULTS/REVIEW.

## Context

The basic coupled lifecycle factory supports cleanup and ingress limits. The
existing application coordinator also has an optional lifetime copy ledger, but
it needs five coupled-specific measurement callbacks and an inspectable builder.
Silently accepting this policy without measurements would weaken its contract.

## Decision

Add an opt-in copy factory and exact typed measurements outside the app layer.
Count real numeric raw arrays in bounded supported graphs and checkpoint inboxes;
count aliases once per independent graph and refuse cycles, opaque objects and
unsupported arrays before their hooks or copies. Coupled native learned arrays
contain no raw replay batches and contribute zero retained raw payload bytes.
Measure prediction output size from trusted topology without running prediction.

Use an exact inspectable builder that delegates complete-policy reconstruction
to the existing coupled checkpoint callbacks. Preparation sizing admits this
builder without invoking it; the existing coordinator reserves before its call.
Preserve the basic factory and its explicit unsupported-policy refusal.

Extend the shared cleanup example with a separate byte-limited mode. Charge
144 bytes for ingress, 144 for each checkpoint capture, and 144 for successful
inbox restoration. At 576 bytes, deny a second restoration before builder/data
cloning and deny another capture before native snapshot/inbox copying. Cleanup
reduces observed retained raw bytes to zero but leaves lifetime charges spent.
Give ingress separate remaining capacity so post-cleanup denial proves the copy
limit caused refusal. Retain original learned state, resources and receipts.

Why this: the existing coordinator already owns the correct reservation order;
coupled adapters should supply measurement and construction, not duplicate it.

## Alternatives

Actual source inspection found that restore's integrity precheck cloned the live
inbox before reservation. Extract a private validated borrowed cursor and hash it
under the existing exclusive owner/gate leases; public capture remains detached.
Keep full payload/metadata integrity checking. Pickle serialization still creates
temporary bytes before admission; the retained-array ledger does not bound that
temporary work or RAM. Refusal tests explicitly forbid the former inbox clone.

Opaque wrapper builders cannot be measured safely and remain unsupported. Resetting
the ledger after cleanup would refund attempted copies while other owners may
retain them. Counting weights or RSS as raw payload would change the budget's unit.

## Consequences

These logical array bytes exclude model parameters, Python overhead, OS memory
and caller-owned copies. Restore still performs current-state validation snapshots
before byte admission; the guarantee concerns subsequent builder/inbox copying.
The original basic mode needs regression coverage because its private loop changes.
No new app/core algorithm, dependency, scientific comparison or process authority.
