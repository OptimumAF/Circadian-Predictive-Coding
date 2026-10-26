# ADR-0611: Validate each detached FIFO entry once

## Context

The original full recurring FIFO36 case exhausted its 180-second allowance.
Later separately paid saved-state readouts completed known-game evidence; they
did not demonstrate a fresh full run within that allowance. Its original profile
measured substantial snapshot validation cost. FIFO `_owned_entries` rebuilt each
entry with `replace`, then rebuilt and validated a second detached entry.

## Decision

Keep exact record/field checks. Explicitly require exact tuple containers before
iteration, then construct one detached `ActionMemoryEntry` whose constructor
validates every feature/count scalar. Check bound geometry and unique keys on
that copy. Preserve all outer policy/capacity/lifetime/retirement validation.

Why this: the first reconstruction repeats the same scalar validation and adds
an allocation. Removing it retains the complete trust boundary. The explicit
container checks prevent a list or tuple subclass from becoming valid through
coercion. Each call still validates current input; frozen records can be tampered
with through `object.__setattr__` and are never treated as a validation cache.

## Alternatives

Caching by identity or sharing entries would weaken tamper refusal or ownership.
Increasing the native time allowance would leave the demonstrated cost intact.
Changing source checks, budgets, data, metrics or game recipes is unnecessary for
this local optimization. The refusal-control copy path remains unchanged.

## Consequences

Both supported geometries retain full corruption refusal and detached records and
tuples. One fixed synthetic benchmark observed 240 copies in 0.862124 seconds
before and 0.413163 seconds after; entry validations halved. This single local
microbenchmark is not statistical or end-to-end performance evidence. All three
batches and costs are retained in
`artifacts/runs/fifo-copy-cost-20261023/RESULTS.md` and issue95.

The prior failed native cases remain failed/spent. Issue94's exact wheel remains
valid for its tested package contract; it predates this core change. A future
installed native qualification must use a new current-tree wheel and a prospectively
bounded protocol after this change's checks, without renewing the original case.

The subsequent issue96 did build/install that wheel and execute one original-cap
case. It still exhausted180seconds during checkpoint64; later readouts remain
unrun and native delivery unqualified. The microbenchmark result is not an
end-to-end speedup claim. Its immutable failed result and costs are preserved in
`artifacts/runs/installed-fifo-native-20261023/RESULTS.md`; no automatic retry.
