# P6.3 fixed-width replay factor: development result

This is the complete three-seed outer-selection result for the prospectively
fixed [replay factor pilot](p63-replay-factor-pilot.md). It is exploratory
development evidence. No confirmation seed or final role was opened, and
no setting was selected or changed after seeing these outcomes.

## Result integrity and scope

The public `scripts.run_p63_replay_factor_pilot` completed two separate
bounded processes under ignored
`artifacts/runs/p63-replay-factor-pilot/` and
`artifacts/runs/p63-replay-factor-pilot-repeat/`. Each saved a prelaunch
request, a finite 24-cell result, and an audit; neither has a failure
sidecar. The two result JSON files are byte-identical, SHA-256
`ecf3f868c96779507e30328f73070edfbdfce0e255dd76eb12bd3d707c3b1de5`.
An in-process feasibility invocation had printed the same three development
scores after the semantic contract was written but before source and adapter
byte hashes were pinned. Replay rates were subsequently made explicit at
their unchanged core defaults; no numerical setting changed. The public
runs followed the frozen source identities. This early score preview is
part of the exploratory development history, not independent validation.
Request SHA-256 values are
`8a2a9528ea01081756ca33df4e4e5b2775c0ce5c7479606299cf4061c1fff512`
and
`e4a856c685d6ed1d799aba2b4fffeb028635935d008ac5b6c8d3a6c3315d30cc`;
the separately timed audit elapsed values were 0.507 and 0.503 seconds.
Read-only revalidation matched request/source/adapter/manifest hashes,
both file hashes in each audit, every role count/hash, all six selected
ID sets and applied work per seed, exact neutral PC/circadian parameters,
finite metrics, and the two identical result byte streams.

Each seed has six development role hashes. A train/inner/outer counts are
72/24/24; B counts are 36/12/12. Both sources' final inputs and labels
were replaced by raising sentinels in the real three-seed test and stayed
unread. B source construction occurred after all A wake/replay work and
the A outer score. Inner guard values were not accessed by this fixed
replay-only factor.

## All predeclared development cells

Values are displayed to six decimals; the saved JSON has full precision.
`A/A` is A accuracy after A; `A/B` and `B/B` are final A and B accuracies.
`Mean` is their equal-task final mean. `Forget` is `A/A - A/B` and may be
negative. `R` is applied replay optimizer updates; `W` is fixed width.

| Seed | Arm | A/A | A/B | B/B | Mean | Forget | R | W |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 41 | backprop off | .958333 | .958333 | .833333 | .895833 | .000000 | 0 | 8 |
| 41 | backprop on | .958333 | .958333 | .833333 | .895833 | .000000 | 12 | 8 |
| 41 | PC off | .833333 | .916667 | .833333 | .875000 | -.083333 | 0 | 8 |
| 41 | PC on | .958333 | .958333 | .833333 | .895833 | .000000 | 12 | 8 |
| 41 | neutral circadian off | .833333 | .916667 | .833333 | .875000 | -.083333 | 0 | 8 |
| 41 | neutral circadian on | .958333 | .958333 | .833333 | .895833 | .000000 | 12 | 8 |
| 41 | backprop planned width | .958333 | .958333 | .833333 | .895833 | .000000 | 0 | 12 |
| 41 | PC planned width | .958333 | .958333 | .666667 | .812500 | .000000 | 0 | 12 |
| 43 | backprop off | 1.000000 | .958333 | 1.000000 | .979167 | .041667 | 0 | 8 |
| 43 | backprop on | 1.000000 | .958333 | 1.000000 | .979167 | .041667 | 12 | 8 |
| 43 | PC off | .750000 | .791667 | 1.000000 | .895833 | -.041667 | 0 | 8 |
| 43 | PC on | .750000 | .833333 | 1.000000 | .916667 | -.083333 | 12 | 8 |
| 43 | neutral circadian off | .750000 | .791667 | 1.000000 | .895833 | -.041667 | 0 | 8 |
| 43 | neutral circadian on | .750000 | .833333 | 1.000000 | .916667 | -.083333 | 12 | 8 |
| 43 | backprop planned width | 1.000000 | 1.000000 | .916667 | .958333 | .000000 | 0 | 12 |
| 43 | PC planned width | .958333 | .958333 | 1.000000 | .979167 | .000000 | 0 | 12 |
| 59 | backprop off | .958333 | .958333 | .833333 | .895833 | .000000 | 0 | 8 |
| 59 | backprop on | .958333 | .958333 | .750000 | .854167 | .000000 | 12 | 8 |
| 59 | PC off | .916667 | .916667 | .833333 | .875000 | .000000 | 0 | 8 |
| 59 | PC on | .916667 | .916667 | .833333 | .875000 | .000000 | 12 | 8 |
| 59 | neutral circadian off | .916667 | .916667 | .833333 | .875000 | .000000 | 0 | 8 |
| 59 | neutral circadian on | .916667 | .916667 | .833333 | .875000 | .000000 | 12 | 8 |
| 59 | backprop planned width | 1.000000 | 1.000000 | .666667 | .833333 | .000000 | 0 | 12 |
| 59 | PC planned width | 1.000000 | 1.000000 | .666667 | .833333 | .000000 | 0 | 12 |

## Paired replay contrasts and costs

| Within-method on minus off | Mean accuracy deltas, seeds 41/43/59 | Mean (sample SD) | Signed-forgetting deltas, seeds 41/43/59 |
|---|---|---:|---|
| Backprop | .000000, .000000, -.041667 | -.013889 (.024056) | .000000, .000000, .000000 |
| Ordinary PC | +.020833, +.020833, .000000 | +.013889 (.012028) | +.083333, -.041667, .000000 |
| Neutral circadian | +.020833, +.020833, .000000 | +.013889 (.012028) | +.083333, -.041667, .000000 |

The PC replay gain is small and mixed across the A/B matrix; signed
forgetting moves in opposite directions on seeds 41 and 43. Neutral
circadian equals ordinary PC bitwise in both on and off arms, including
final parameter hashes and metrics. Thus this factor provides no
evidence for a *circadian-specific* benefit. Backprop replay worsens final
mean on seed 59 and ties on the other two. Planned width 12 is also mixed;
it is a capacity reference, not a matched-cost replay contrast. With
only three development seeds, none of these observations is a confirmatory
ranking or significance result.

Per seed/arm, wake work is 24 updates and 1,296 row presentations. PC and
circadian use 48 wake inference loops/2,592 example-iterations; backprop
uses none. Every on arm receives the same 12 selected row IDs, 12 replay
updates/presentations; PC/circadian use 24 additional inference loops.
Off arms receive zero replay work. Six scheduled boundaries per seed
apply all selected rows; both circadian arms attempt six component sleep
events, with replay disabled only in the off arm. All arms keep their
initial width and parameter count throughout: 8/33 or 12/49. Shared
schedule retention is eight labeled rows/192 array bytes at each boundary;
each circadian model also stores its own matching eight-row/192-byte
buffer. Those array-byte counts exclude Python metadata and model memory.
No guard evaluation or rollback occurs by the predeclared design.

The adapter measured whole-worker wall time, not per-arm runtime. Process
peak RSS, per-arm wall time, and event-duration totals were not measured;
they cannot be inferred from update counts or array bytes. P6.10 remains
open. Replay on/off intentionally differs in optimizer work; width 12
intentionally differs in capacity and training compute. The design can
isolate the replay switch at fixed width within a method, but it cannot
claim equal total compute or that replay beats the learning-rule baselines.
