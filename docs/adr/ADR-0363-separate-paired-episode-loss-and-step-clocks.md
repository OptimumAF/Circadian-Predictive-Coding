# ADR-0363: Separate paired episode loss and optimizer-step clocks

## Context

At the accepted A128 state, original blue/red episode gradients oppose. The
isolated native steps improve their own cue CE while worsening the other cue.
This motivates testing a paired update; it does not establish that batching
improves acquisition. Existing native snapshots assume one episode per optimizer
step, which cannot honestly represent a cold9+9pair as one step and18rows.

## Decision

Provide a separately chosen recurrent episode-batch learner. Independently reset
hidden activation for each complete TRAIN episode, average their two mean CEs,
and take one atomic native SGD step with carried momentum. Unequal episode
lengths retain equal episode weighting. Preserve all legacy math and contracts.

Use a distinct strict snapshot with optimizer-step, episode and row clocks.
Explicit one-way legacy conversion sets prior episodes=steps and preserves full
parameters/gradients/momentum. A private temporary episode-clock view reuses
unchanged native tensor/optimizer validation; it is never a legacy restore,
benchmark or public checkpoint payload. Refuse full owner changes before SGD
and publication; retain paid step/failed publication facts independently.

## Alternatives

- Concatenation would carry hidden state between independent episodes.
- Row-weighted CE would change the declared equal-episode objective for unequal
  lengths. Neither alternative implements this experiment.
- Reinterpreting optimizer steps as episode count, or relaxing the legacy row
  bound globally, would corrupt existing accounting and saved-state contracts.

## Consequences

The existing MiniGrid probe observer now explicitly selects the warm adapter.
Its original16E episodes become8pairs/144rows with row-matched0/2/4/8 readouts,
keeping separate evaluation episodes and legacy16step defaults. The stream's
original source stays cold while candidate checkpoints may be warm; equating
their weights would incorrectly reject legitimate checkpoints. Validate both
independently instead. Only scoped synthetic integration is qualified; a real
reserved-task protocol and useful adaptation remain separate scientific work.

Warm isolated probes use an explicit `probe_train` entrypoint with unchanged
episode roles and IDs. The separate warm array adapter keeps absolute base plus
probe clocks and full optimizer state while the existing probe app counts local
updates under its own budget. Why this: the cold TRAIN-A schedule cannot admit
new reserved tasks, and role relabeling would break evaluation isolation. A small
core typed failure carries owned bounded physical-work facts across the generic
app boundary; only the update call may supply those facts. No new orchestrator
or implicit benchmark conversion is needed. Qualification is separate from any
later fresh-task/retention/recovery protocol.

The primitive is independently qualified. The explicit `recurrent_paired_bc`
stream/scorer/TRAIN-A-live integration now admits its new snapshot and actual
step/episode/row counters. No implicit benchmark switch.
Any subsequent acquisition case must freeze data/seed/metrics/exposure and report
actual steps and rows:1152rows correspond to64paired versus128legacy steps.
No current learning, retention/recovery or circadian benefit follows from the
mathematical tests. Historical negatives and failed experiments remain intact.

The array adapter owns the original complete TRAIN schedule and checks each
18-row transport against its next two records. Why this: the existing stream
accepts arrays, while independent recurrent resets require episode boundaries.
No orchestration framework or bridge-based counter reinterpretation is needed.
Private observed paid reports survive refusal of late callback changes; unknown
model changes remain visible. Prospective acquisition and component correctness
remain separate evidence.

## Measured identity-comparison optimization, 2026-10-17

The first native acquisition case exhausted its original diagnostic20s budget.
An identical fake-route cProfile pair measured schedule/packet leaf lookup as
dominant. Capture strong ordered values and compare with C-level identity
operations, preserving field order/cardinality, full traversal and callback proof
cadence. Signature array bytes and native-state/counter checks remain unchanged.
Why: reduce measured per-field overhead without admitting unobserved mutations.
125focused cases pass; instrumented fake time27.3941→13.1098seconds. No native
retry, deadline extension or scientific result follows; see readout-cost RESULTS.

## Endpoint-first observation, 2026-10-17

Two separately paid cases reached the same20s diagnostic cap before the fixed
endpoint. Add explicit order0,64,16,32,48 under a new protocol identity; preserve
the default and all-five completion criterion. Why: endpoints determine the
prespecified acquisition comparison; middle live/B-C readouts are descriptive.
Do not change training, seed, dose, metrics or clocks to obtain favorable results.

Separate private endpoint evidence can survive a later typed wall-only refusal
only after original45s full-arm proof and pure evidence validation. Resource
callbacks precede final proof; cleanup/readback/corruption faults invalidate
endpoint outcomes, with paid observations retained. Incomplete full outcomes stay
null. This is observed historical evidence, not current serving/release readiness.
