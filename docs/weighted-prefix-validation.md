# Bounded public-head prefix validation

`TorchTrainWeightedLearner` now admits an immutable per-prefix count table along
with its existing complete TRAIN schedule. Each entry stores head0 updates/rows
and head1 updates/rows. Selection uses the actual first public observation,
never context names or labels. State reads use exact counts and still validate
both complete native snapshots, gradients, momentum, policy and failure state.

Why this: the old path converted and decoded every preceding first frame on
every read, even though existing lifetime witnesses already proved the source
unchanged. Those witnesses still check every mutable field, field order/type,
record dictionary and immutable observation identity. The table is an exact
tuple of native integer tuples, pinned to its owner. Replacing it, even with an
equal tuple, refuses. Each fork admits its own table; no success history grows.
At512episodes it contains513 entries and passes the128KiB conservative storage
bound. No TTL/invalidation timer is required for owned immutable data. New data
requires new admission; changed source/policy/owner fails the existing guards.

96 focused scripted checks pass on each cached Windows/Linux runtime, including
every0..128 variable-length prefix, fork/restore,512-episode storage, source/seal
and table mutation, wrong per-head counters, live public cue decoding, old
single-RNN defaults and permanent32-counter failure accounting. Scoped Ruff,
format and fresh Windows mypy (default and Linux target) pass. One finite
independent read-only source/procedure review found no actionable issues; lead
also reviewed the combined source, execution and results. Native Linux Ruff/mypy
remain unavailable and were skipped; no installation or full suite was run.
Unchanged native mathematical parity from issue139 is reused.

Fixed measurements use a literal128-episode,9-row/raw148 schedule and scripted
native heads at **synthetic**128/1152 counters. These are not performed fits.
Each cell below contains all three fixed16-call batch durations in seconds.

| Platform/component | Before | After | Redundant decodes per batch |
| --- | --- | --- | --- |
| Windows `_progress` |2.28553 /2.23989 /2.29213|.00290 /.00309 /.00303|2048 →0|
| Windows full snapshot |2.57366 /2.54805 /2.56083|.33505 /.32957 /.32185|2048 →0|
| Linux `_progress` |1.95460 /1.95470 /1.95801|.00526 /.00497 /.00509|2048 →0|
| Linux full snapshot |2.23664 /2.22740 /2.22711|.30766 /.30156 /.30652|2048 →0|

Admission adds128 cue decodes, paid once per owner. The129-entry table's
conservative shallow-plus-leaf upper account is25,848B Windows/24,808B Linux
(shared integer references counted repeatedly). Profiling, scripted state and
the fixed machine/runtime affect timings: no whole-game speed, learning benefit,
native endpoint recovery, CPC advantage or release claim follows.

Exact commands, receipts, budgets, initial type-check failures, skips and costs:
`artifacts/runs/weighted-prefix-counts-20261025/RESULTS.md`.
The original issue140 readout remains FAILED-SPENT; there is no automatic rerun.
