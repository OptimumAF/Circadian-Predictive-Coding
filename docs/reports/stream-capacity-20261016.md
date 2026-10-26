# Capacity boundaries and scheduled split, 2026-10-16

The R2.6 **accounting/scheduled-split engineering prerequisite is accepted** after
independent SOURCE/GO/ACTUAL review. Original R2.6 remains unchecked for its full
split/prune, scheduled-growth, equal-final-width and bounded-renewal comparison
with matched BP controls. This run establishes no efficiency, useful adaptation,
effective-capacity or circadian advantage result.

## Executed behavior

The existing attribution runner now supports an optional capacity helper. It
records actual committed native parameter/auxiliary arrays, completed native
wake calls and row exposures. Means weight the model used for each successful
wake, excluding extra votes for initialization and structural observations.
Original budgets, clocks, measurement callback, ceilings, source and runtime
remain bound. Scalar history is bounded; changed counters/ceilings refuse before
more native training. Default schemas/continuation persist; capacity continuation
and simultaneous helper modes are explicitly unsupported.

One predeclared split reuses the existing raw structural learner/consolidator.
Predicted target parameter/array/width ceilings are checked before expansion;
actual post-publication usage is measured again. Failed measurements/transforms
retain completed wake work. Late publication failure keeps the actual receipt
and available post-state observation, without retry or fresh budget.

Pure snapshot measurement supports ordinary/tracked/renewable coupled, matched
BP and raw/structural/topology state. It validates all native parameter arrays,
chemistry/gates and unit-history arrays without constructing/restoring a model.

## Capacity and outcomes

Four two-row delayed batches A/A/B/A, seed23, raw mean-field gamma1.5, D2/H1,
learning rate0.01 and two fixed inference steps0.05 were frozen before results.
Each method has a scheduled width2→4 split after wake2 and a fixed width4 source
initialized by the same isometric split. All initial heldout scores match within
the declared1e-12 BCE tolerance; accuracy matches exactly. PC/neutral raw parameter
checkpoints and score curves match exactly. Every final width is4.

| Method | Mode | Mean parameters | Peak parameters | Mean native array B | Peak native array B | Final A BCE | Final B BCE |
|---|---|---:|---:|---:|---:|---:|---:|
| pc | fixed_final_width | 17 | 17 | 136 | 136 | 0.793080031 | 0.519308698 |
| pc | scheduled_split | 13 | 17 | 104 | 136 | 0.793065266 | 0.519284808 |
| neutral_cpc | fixed_final_width | 17 | 17 | 200 | 200 | 0.793080031 | 0.519308698 |
| neutral_cpc | scheduled_split | 13 | 17 | 152 | 200 | 0.793065266 | 0.519284808 |
| gated_cpc | fixed_final_width | 17 | 17 | 200 | 200 | 0.793081309 | 0.519309550 |
| gated_cpc | scheduled_split | 13 | 17 | 152 | 200 | 0.793066631 | 0.519285921 |

Growing wake-use parameter counts are9/9/17/17, hence mean13 and parameter-byte
mean104. Fixed width4 has mean17/136bytes. CPC chemistry/gates add mean48/64
bytes respectively, producing152/200native-array bytes. Both modes peak at17
parameters; CPC peaks at200native-array bytes, PC at136.

These are stored candidate field payloads, excluding source/actor/scorer/staging/
checkpoint copies, latent/gradient arrays, metadata, Python and process overhead.
Replicated units count as stored parameters; this does not measure functional
rank. The split maps precision1→0.5, raw rate factor4.5→9 and effective weight
factors approximately(2.25,0.5)→(4.5,0.25); bias factors remain approximately1.
All exact factors are in the raw JSON. Different update-factor paths and
duplicated reference units prevent an equal-compute/effective-capacity claim.

Final A accuracy is0.5 and B accuracy1.0 in every case. B accuracy was already1.0
at initialization; this tiny accounting fixture cannot establish fresh-task
learning or recovery. Slight final loss differences are retained without
selecting a winner or tuning the protocol. All prior negative results persist.

## Every heldout point

Each cell is BCE / accuracy; raw JSON retains full precision.

| Method | Mode | Wake | A BCE / accuracy | B BCE / accuracy |
|---|---|---:|---:|---:|
| pc | fixed_final_width | 0 | 0.793302440 / 0.500000 | 0.519331859 / 1.000000 |
| pc | fixed_final_width | 2 | 0.793142037 / 0.500000 | 0.519274334 / 1.000000 |
| pc | fixed_final_width | 4 | 0.793080031 / 0.500000 | 0.519308698 / 1.000000 |
| pc | scheduled_split | 0 | 0.793302440 / 0.500000 | 0.519331859 / 1.000000 |
| pc | scheduled_split | 2 | 0.793127252 / 0.500000 | 0.519250453 / 1.000000 |
| pc | scheduled_split | 4 | 0.793065266 / 0.500000 | 0.519284808 / 1.000000 |
| neutral_cpc | fixed_final_width | 0 | 0.793302440 / 0.500000 | 0.519331859 / 1.000000 |
| neutral_cpc | fixed_final_width | 2 | 0.793142037 / 0.500000 | 0.519274334 / 1.000000 |
| neutral_cpc | fixed_final_width | 4 | 0.793080031 / 0.500000 | 0.519308698 / 1.000000 |
| neutral_cpc | scheduled_split | 0 | 0.793302440 / 0.500000 | 0.519331859 / 1.000000 |
| neutral_cpc | scheduled_split | 2 | 0.793127252 / 0.500000 | 0.519250453 / 1.000000 |
| neutral_cpc | scheduled_split | 4 | 0.793065266 / 0.500000 | 0.519284808 / 1.000000 |
| gated_cpc | fixed_final_width | 0 | 0.793302440 / 0.500000 | 0.519331859 / 1.000000 |
| gated_cpc | fixed_final_width | 2 | 0.793139411 / 0.500000 | 0.519278387 / 1.000000 |
| gated_cpc | fixed_final_width | 4 | 0.793081309 / 0.500000 | 0.519309550 / 1.000000 |
| gated_cpc | scheduled_split | 0 | 0.793302440 / 0.500000 | 0.519331859 / 1.000000 |
| gated_cpc | scheduled_split | 2 | 0.793124716 / 0.500000 | 0.519254764 / 1.000000 |
| gated_cpc | scheduled_split | 4 | 0.793066631 / 0.500000 | 0.519285921 / 1.000000 |

## Work, overhead and qualification

The sole six-arm case returned all24 native updates/48 row exposures, with24 raw
steps,96 coupled-energy evaluations,36 final-score forwards and six split calls
(three reference initializations plus three scheduled). Source learners remain
unchanged; no final features, labels or scores enter the capacity helper.

Total pilot time0.6340691s, worker0.6704085s, parent1.1643736s; observed process
RSS42,930,176bytes; output100,468bytes, stderr empty. Helper measurement time
0.1282965999s and checkpoint reporting0.0257559000s are included in wall time.
These include snapshot/copy/validation and reporting work; they are not causal
overhead estimates or calibrated throughput comparisons. CPU1/GPU0/60s/256MiB/
2MiB local limits passed with one joined RSS sampler and no download.

123focused cases/369passed phases and11-path Ruff/format/WinLinux mypy passed.
The initial collection attempt failed on a demonstrated import cycle and ran
zero tests/native calls. Moving the existing pure scorer to core removed the
backedge; its old attribution alias and exact clipping/threshold arithmetic
persist. The failed collection and initial test-typing failures are preserved.
Unchanged native math/Torch matrices were reused; affected teacher replay,
default/CLI and fake continuation checks passed. No baseline run, sweep or retry
of the native engineering case occurred.

Next: an owned structural wrapper around the existing scaled matched BP kernel.
Its current fixed-config port cannot follow split/prune histories. Verify native
parity/ownership/continuation before the full R2.6 protocol, while preserving
the old fixed-config restore contract and final-data isolation.

See [usage](../stream-capacity.md),
[evidence and commands](../../artifacts/runs/r26-capacity-observer-20261016/RESULTS.md),
[review](../../artifacts/runs/r26-capacity-observer-20261016/REVIEW.md), and
[all exact values](../../artifacts/runs/r26-capacity-observer-20261016/summary.json).
