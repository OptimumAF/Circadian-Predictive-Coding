# Saved replay-v8 smoke fixture

P9.5b25 validates and presents an existing two-seed synthetic fixture. This
completes saved presentation, without new scientific or runtime admission.
Both policies, every seed and all 16 sleep events are preserved. No policy,
baseline, seed, metric or winner was selected or changed.

## Source and provenance

Original: `data/continual_replay_policy_v8_smoke.json`, 215150 bytes, SHA256
`d28dbb2830ba2a939c1b7c13a2e2162f05e604fad52e62b88f09e37f9dbb475e`.
The whole original was byte-bound against the complete frozen publication
register before parsing. Five whole metadata bodies, the original body and its
typed values are preserved in `artifacts/runs/p95-replay-v8-20261006/`.
The registered kind remains `corrected_development_or_fixture` with incomplete
original execution environment. Empty missing-field lists do not prove complete
provenance. No current defaults, implementation prose or related runs fill gaps.

Protocol: `continual_replay_policy_comparison_v8`; nested arrived roles
`continual_arrived_roles_v6` and training `continual_global_test_seal_v5`.
Manifest digest `001734383b374414bc66a368aab6f40896d41541c11ca104c3b0cdd4cdd8fb53`
is preserved as an opaque source claim; original encoding and physical state
are not established. The exact invocation, interpreter/dependencies/hardware,
complete attempt provenance, retained arrays and process RSS are absent.

The complete configuration is retained in the view and 3428 typed leaves,
including 195 nulls, 63 booleans and the negative translation value. Saved
settings include hidden dimension 4, two epochs per phase, 40 declared samples
per phase, final fraction .25, inner guard and outer fractions .2, phase B
training fraction .14, noise .8/1.0, rotation 40 and translation (.9,-.7).
Force sleep and phase intervals 1, max splits/prunes 0, replay caps 4 examples/
96 array bytes, one replay step and configured replay_memory_size 1 remain as
recorded. Current code/defaults are not used to infer how configuration layers
mapped into the original retained storage or allocation.

Policies are `recent_fifo` with null policy seed and `seeded_reservoir` with RNG
seed 53. Model/data seeds are 17 and 19. Policy seed 53 is not a third model seed
or an independent replication. Only forward model order BP/PC/CPC is recorded.
This fixture has no matched baseline replay arm, outer candidate selection,
checkpoint continuation or independent model-order confirmation record.

## All saved outcomes

Both policies have the same final balanced scores. Twelve final method records
are retained: two policies × two model seeds × three methods.

| Seed | Method | A pre | A post | B post | Retention | Balanced |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | BP, both policies | 1.0 | 1.0 | .9 | 1.0 | .95 |
| 17 | PC, both policies | .7 | .8 | .7 | 1.142857142857143 | .75 |
| 17 | CPC, both policies | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 19 | BP, both policies | 0.0 | 0.0 | .4 | 0.0 | .2 |
| 19 | PC, both policies | .6 | .6 | .4 | 1.0 | .5 |
| 19 | CPC, FIFO | .4 | .4 | .4 | 1.0 | .4 |
| 19 | CPC, reservoir | .3 | .4 | .4 | 1.3333333333333335 | .4 |

CPC is below PC for both seeds. Its seed 17 balanced score is also below BP;
seed 19 is above BP. No pooled ranking or newly calculated difference was added.
The retention ratios above 1 are preserved on a 0–1.5 axis, with exact source
labels. Zero pre-accuracy retention remains zero as saved; its denominator policy
is not inferred from current code.

Saved balanced mean/std for both policies: BP .575/.375, PC .625/.125,
CPC .2/.2. These are consistent with population standard deviation of the two
saved values, never confidence intervals or independent replications. CPC mean
pre-accuracy differs between policies (.2 vs .15); mean retention is .5 vs
.6666666666666667. All other original aggregate fields remain in whole bodies
and panels. Every CPC case reports hidden start/end 4, sleep 4, splits/prunes 0.
BP/PC metric bodies and baseline state-digest strings agree across policies for
each seed. This is saved consistency, not physical state or freshness proof.

## Replay identities and exposure

Every case retains four sample IDs and 96 array bytes at both phase snapshots,
equal to recorded caps. All retained IDs are unique and belong to the respective
observed-ID collection. The phase A observed list has 18 IDs, with 18 duplicate
IDs/occurrences and two cumulative replay updates. After phase B it has 22 IDs,
22 duplicate IDs/occurrences and four cumulative replay updates. Duplicate lists
and terminal copies agree internally. Full ordered observation streams, arrays
and allocation records are absent; ID accounting does not prove physical bytes,
leak-free execution or memory matching.

Each phase A exposure set has one ID. After B, FIFO has two distinct exposed IDs
per seed; reservoir retains one in the saved cumulative exposure set. Retained
sample selections differ. The complete exposed, observed, duplicate and retained
ID lists are preserved, without treating list cardinality as sample accuracy or
new experiment statistics. These saved policies have equal balanced scores;
no claim of a generally superior replay policy follows.

## All sleep events, roles and telemetry

Each of four cases contains four events. Saved completed_epoch and wake_batches
are 1,2,3,4; separate guard-decision phase epochs are A1,A2,B1,B2. All events
record periodic trigger, accepted outcome/guard_accepted reason, performed and
accepted guards, restored false, width 4 before/proposed/final, no split/prune
proposals or applications, one proposed/applied replay example/update, and null
time limit. Cross-entropy guard values are null, never inferred from accuracy.

All guard deltas/tolerances are zero. Seed 17 guard accuracy is zero throughout;
seed 19 is .6666666666666666 in phase A and .5 in phase B, under both policies.
Each event reports 12 scored guard examples in A or 4 in B, consistent with
saved role sizes and pre/post values. The event and guard-decision hash/accuracy
copies agree. Chemical counts are 4; all 144 before/proposed/final component
summaries satisfy minimum ≤ mean ≤ maximum. Proposed and final summaries agree
for each accepted event. All extrema/counts remain in typed tables; means are
shown for primary/fast/slow at every stage. Chemical units are unspecified;
this is model telemetry in a fixture, not biological measurement.

All 32 saved duration values are displayed in seconds. Each attempt_seconds is
at least core_seconds. This does not establish policy speed, current timings,
budget admission, synchronized instrumentation, host comparability or complete
execution provenance. No new duration aggregate or timing experiment was made.

Each seed's eight role-ID sets are internally disjoint and phase/seed namespaced:
A train18/guard6/outer6/final10, B train4/guard2/outer2/final10. Role hash copies
agree; physical arrays/hash encoding remain unknown. Each 32-entry ledger has
all three models' A training before B arrival, four CPC-only guard entries and
four final input/label releases at global_freeze after development entries.
Declared/arrived phase task information is preserved: A then A/B; BP/PC guard
roles null, CPC inner guard. Source outer_selection roles are released on arrival
but no outer-selection action/candidate selection is saved in this fixture.
Saved ledger ordering is not fresh isolation, final-source sealing or physical
provenance evidence. Parent acceptance requirements remain open.

## Figures, validation and budget

Ten 1800×1350 PNG/SVG pages contain 44 panels and 390 numeric labels. Every full
PNG was visually reviewed. Source bodies and typed tables retain nonnumeric
fields, empty collections, IDs, extrema, units and unknowns outside the panels.

- [FIFO final values](../artifacts/runs/p95-replay-v8-20261006/final-recent_fifo.png)
- [Reservoir final values](../artifacts/runs/p95-replay-v8-20261006/final-seeded_reservoir.png)
- [All saved aggregates](../artifacts/runs/p95-replay-v8-20261006/aggregates.png)
- [Exposure and retained counts](../artifacts/runs/p95-replay-v8-20261006/exposure.png)
- [All guard values](../artifacts/runs/p95-replay-v8-20261006/guards.png)
- [All saved event timings](../artifacts/runs/p95-replay-v8-20261006/timings.png)
- [FIFO seed 17 chemistry](../artifacts/runs/p95-replay-v8-20261006/chemistry-recent_fifo-17.png)
- [FIFO seed 19 chemistry](../artifacts/runs/p95-replay-v8-20261006/chemistry-recent_fifo-19.png)
- [Reservoir seed 17 chemistry](../artifacts/runs/p95-replay-v8-20261006/chemistry-seeded_reservoir-17.png)
- [Reservoir seed 19 chemistry](../artifacts/runs/p95-replay-v8-20261006/chemistry-seeded_reservoir-19.png)

Independent readback validates whole body/typed values, labels, ranges, units,
bar/whisker geometry, decoded PNG pixels including zeros, 104 existing arithmetic
checks, 144 chemical ordering/count checks and ten refusal controls. Controls
cover duplicate key, nonfinite JSON, unknown root, boolean metric/count, missing
event, wrong policy/seed, over-budget retained bytes and changed guard delta.
Original source is rechecked afterward. No failed source claim was hidden or
value repaired. One failed prepare command and whole candidate were retained:
the helper referenced the older b23 terminal rather than the immediately prior
b24 terminal. The corrected comparison accepts the unchanged 1027-file checkout.

Six ignored helpers pass Ruff check, format check, py_compile and full
pre/postformatter AST equality. Complete receipts include scope, entry, frozen
metadata/source, view, typed leaves, visual-review, readback, static-validation,
acceptance, coverage-delta, reversible doc edits, terminal and final accounting.
Exactly six stage helpers; final accounting uses captured inline Python.

Scope: 600 aggregate engineering seconds, hard 60 per captured child, 64 MiB
owned files, fixed 160 manual/discovery/visual/closing reserve plus all captured
commands, including the failed prepare. Final accounting reserves its full
60-second cap. This is not whole-session wall time or process RSS. Previously
spent science 350.7925872/360 and runtime 168.7993043/180 remain unchanged.
Full pytest, mypy, native/Torch/CI/clean-clone and original scientific-reader
gates were skipped for saved-data inspection helpers/additive docs, and remain
open. No model/dataset/device/browser/sweep/dependency/download/publication work.
All unrelated files/task rows/HEAD/packages remain preserved. P9.5/P9.5b,
G0/R0.3/fullR3.1 and missing-source criteria remain open; owning-with j6c remains
human-deferred.

Why this increment: present replay telemetry without selecting a favorable
policy or upgrading saved ID/digest/timing claims into physical execution proof.
No acceptance criterion was weakened or unfinished task removed. Next P9.5b26:
bind complete replay-v9-resume metadata and the whole active_resumed, fresh and
resumed originals before parsing in a fresh small scope; inspect continuation
evidence while retaining physical checkpoint/environment gaps.
