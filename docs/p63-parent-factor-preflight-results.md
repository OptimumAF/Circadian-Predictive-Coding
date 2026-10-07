# P6.3c9b parent-control train-only results

## Verified result and scope

Both bounded fresh-process runs of the [frozen contract](p63-parent-factor-preflight.md)
completed all **24 cells/72 opportunities/216 decisions** and repeat exactly:
result SHA-256 **`555fc2fe5dfd86981d87af2d0c15bd8ab925417d9ad748783d95cee95f5a1874`**.
They executed **576 wake updates**, zero replay, **54 accepted guarded sleeps**,
45 proposed/committed splits, nine final zero-add sleeps and 162 not-due decisions.
There were no official rollbacks or failures. Forced-rejection fixtures verify
the separate proposed selector facts, full rollback and identical future retry.

Usage, cyclic and random cells have identical attempted/applied counts and
width trajectories here, while their recorded parent IDs differ. This establishes
the control implementation and reproducibility. No outer/final score was read;
there is no accuracy, forgetting, efficiency or superiority result in this gate.
C9c scoring, original c9/matrix/confirmation and final release remain unfinished.

## Source and identities

Development seeds 347/349/353, initialization seed+1001, selector PCG64 seed+5001
(5348/5350/5354), cursor zero. The ten declared confirmation seeds remain unused.
A/B source 160/160; A train/inner/outer 72/24/24, B 36/12/12. Every cell gets
12+12 common full-batch wakes; B is constructed after all eight A models finish.
Outer/final/arrival sentinels pass. Exact six role hashes per seed, all wake and
after-A/after-B parameter/full-state hashes, selector and lineage histories are
preserved in both complete JSON results. Neutral and ordinary PC parameters
match exactly after every wake and decision. All initialization hashes match
within their prospectively declared widths.

| Identity | SHA-256 |
|---|---|
| Manifest | `a7938028ed3c9279ef74a5f9a2550012927e4bb626b72672861ae64aa71497c9` |
| Selected 31-source map | `280210ea8215b24e3d5238b39daa94725b2091553bfed8d5bb7e5a23977339f9` |
| Adapter bytes | `39bfd61bd50a1abf0a00308ca40028152270b09f99eb401b4176924dd4c6a8b1` |
| First request | `e5c7f0c49d34896409614383db728798bb33cb1b50de239a2a98cad036cfb385` |
| Repeat request | `e269fc8216a5e6b67d85c817daba0e866f7ff8d8c442a5286a80c59faf44f92a` |

Readback parses all finite request/result/audit files, checks every declared
role/model/opportunity, independent ordering/RNG/guard/work/lineage validators,
source/adapter/manifest/request/result digests and observed wall/RSS caps, and
compares exact saved result bytes. Both directories have no failure sidecar:
`artifacts/runs/p63-parent-factor-preflight{,-repeat}/parent-factor-preflight.{request,result,audit}.json`.
These local artifacts are ignored by Git; the contract provides regeneration
commands. All eight c5–c8 bundles and their frozen source/result identities
remain unchanged after this increment.

## Complete work and capacity cells

Widths are initial / after A / after B / peak. Parameters are initial / final /
peak. Wake columns are updates / presentations; latent columns are loops /
example iterations. Guard columns are attempts / forward evaluations / examples.
All replay and rejected optimizer work is zero. Each growth cell commits six
sleep events and five splits; references have zero sleep/guard/split events.

| Seed | Cell | Widths | Parameters | Wake | Latent | Guard |
|---|---|---|---|---|---|---|
| 347 | backprop_off | 8/8/8/8 | 33/33/33 | 24/1296 | 0/0 | 0/0/0 |
| 347 | pc_off | 8/8/8/8 | 33/33/33 | 24/1296 | 48/2592 | 0/0/0 |
| 347 | neutral_off | 8/8/8/8 | 33/33/33 | 24/1296 | 48/2592 | 0/0/0 |
| 347 | usage_growth | 8/11/13/13 | 33/53/53 | 24/1296 | 48/2592 | 6/12/216 |
| 347 | scheduled_growth | 8/11/13/13 | 33/53/53 | 24/1296 | 48/2592 | 6/12/216 |
| 347 | random_growth | 8/11/13/13 | 33/53/53 | 24/1296 | 48/2592 | 6/12/216 |
| 347 | backprop_13_off | 13/13/13/13 | 53/53/53 | 24/1296 | 0/0 | 0/0/0 |
| 347 | pc_13_off | 13/13/13/13 | 53/53/53 | 24/1296 | 48/2592 | 0/0/0 |
| 349 | backprop_off | 8/8/8/8 | 33/33/33 | 24/1296 | 0/0 | 0/0/0 |
| 349 | pc_off | 8/8/8/8 | 33/33/33 | 24/1296 | 48/2592 | 0/0/0 |
| 349 | neutral_off | 8/8/8/8 | 33/33/33 | 24/1296 | 48/2592 | 0/0/0 |
| 349 | usage_growth | 8/11/13/13 | 33/53/53 | 24/1296 | 48/2592 | 6/12/216 |
| 349 | scheduled_growth | 8/11/13/13 | 33/53/53 | 24/1296 | 48/2592 | 6/12/216 |
| 349 | random_growth | 8/11/13/13 | 33/53/53 | 24/1296 | 48/2592 | 6/12/216 |
| 349 | backprop_13_off | 13/13/13/13 | 53/53/53 | 24/1296 | 0/0 | 0/0/0 |
| 349 | pc_13_off | 13/13/13/13 | 53/53/53 | 24/1296 | 48/2592 | 0/0/0 |
| 353 | backprop_off | 8/8/8/8 | 33/33/33 | 24/1296 | 0/0 | 0/0/0 |
| 353 | pc_off | 8/8/8/8 | 33/33/33 | 24/1296 | 48/2592 | 0/0/0 |
| 353 | neutral_off | 8/8/8/8 | 33/33/33 | 24/1296 | 48/2592 | 0/0/0 |
| 353 | usage_growth | 8/11/13/13 | 33/53/53 | 24/1296 | 48/2592 | 6/12/216 |
| 353 | scheduled_growth | 8/11/13/13 | 33/53/53 | 24/1296 | 48/2592 | 6/12/216 |
| 353 | random_growth | 8/11/13/13 | 33/53/53 | 24/1296 | 48/2592 | 6/12/216 |
| 353 | backprop_13_off | 13/13/13/13 | 53/53/53 | 24/1296 | 0/0 | 0/0/0 |
| 353 | pc_13_off | 13/13/13/13 | 53/53/53 | 24/1296 | 48/2592 | 0/0/0 |

Aggregate guard exposure is 108 forward evaluations / 1,944 examples, separate
from 31,104 wake presentations. PC/circadian latent work counts are declared
loop/example counts, not measured FLOPs. Width, algorithms and guard overhead
give unequal compute despite equal wake update counts. Wider references have
53 parameters throughout; growth reaches 53 after starting at 33.

## Every split-parent choice

Columns are global epochs 4/8/12/16/20; children born at these events are
stable IDs 8/9/10/11/12 respectively. Epoch 24 requests zero and preserves the
selector cursor/RNG/last decision, while committing a no-op sleep clock event.
All eligible candidates are chemical-preferred under the predeclared c3 zero
threshold. Scores and before/proposed/applied selector/RNG fingerprints for
every epoch are in the results; the validator rederives all three orderings.

| Seed | Cell | Parent IDs | Final cursor / selection calls |
|---|---|---|---|
| 347 | usage_growth | 0,6,0,8,4 | 0/5 |
| 347 | scheduled_growth | 0,1,2,3,4 | 5/5 |
| 347 | random_growth | 7,6,7,0,10 | 0/5 |
| 349 | usage_growth | 5,2,7,5,1 | 0/5 |
| 349 | scheduled_growth | 0,1,2,3,4 | 5/5 |
| 349 | random_growth | 2,5,8,2,1 | 0/5 |
| 353 | usage_growth | 4,5,7,4,8 | 0/5 |
| 353 | scheduled_growth | 0,1,2,3,4 | 5/5 |
| 353 | random_growth | 6,3,2,8,11 | 0/5 |

Repeated parents across events and splits of earlier children are permitted by
the original eligibility/cooldown policy; within-event selection is unique.
No parent outcome selected a seed or changed the count schedule.

## Memory, resources and validation limits

All four circadian buffers and the shared FIFO retain eight identical arrived
training rows/192 float64 array bytes apiece. **960 bytes/seed** is the retained
replay/supply array scope only; source/role arrays, parameters, metadata and
temporary copies are outside that count and included in whole-worker RSS.
Baselines have no own replay store. All replay exposure/updates are zero.
Before/after supply checks bind all eight contents/order/bytes without invoking
prioritized sampling, whose inherited setting is inert with replay disabled.

First/repeat elapsed times **1.682858/1.784217 s**; observed RSS peaks
**51,748,864/51,417,088 bytes**, **85/82 samples** at 5 ms. Both are below
120 seconds/256 MiB. RSS covers training, independent validation and result
serialization, with stdout and parent writes outside it; short peaks can be
missed. Per-arm wall time/memory is unmeasured. Environment: Windows 11,
Intel Core i7-12700K CPU, Python 3.14.7, NumPy 2.4.6, NumPy training.

The final related suite passes **210 tests, zero skipped**, including 38 new
app/CLI cases, c9a/core/proposal/lineage/sleep/clock and actual NumPy/Torch CPU
atomic rollback cases plus c7 regression. Ruff, mypy (**357 files**), six-file
format and diff checks pass. Tests cover role/arrival seals, actual executed
work, same-cell repeat, all-mode guard rejection/retry, pre/core/post failures,
memory corruption/order, forged parent/RNG/cursor/count/lineage/guard/work/
checkpoint/role/capacity/supply/cell/seal facts, source/manifest refusal and
exclusive/timeout/nonfinite/incomplete worker artifacts. An initial replay
preview mismatch and a red before-hash forgery fixture were corrected without
changing frozen training settings or earlier pinned code; details are in the log.

Full CPU suite, CUDA, large sweeps, outer scoring, independent confirmation and
final release were skipped. No external blocker. **Next: c9c** must prospectively
bind this complete saved all-seed train result, every cell/primary A/B metric and
growth/reference contrast, then globally match all parameter/full selector
checkpoints before reading the first outer value. Repeat bounded scored bytes
and publish every null/negative outcome. This train-only gate closes c9b only.
