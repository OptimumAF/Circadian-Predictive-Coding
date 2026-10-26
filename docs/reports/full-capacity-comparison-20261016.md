# Full capacity mechanism comparison, 2026-10-16

## Frozen scope and interpretation

This is the single prespecified R2.6 local comparison: 23 complete arms, seed23,
18 eight-row A6/B6/A6 wakes, delay2, checkpoints0/6/12/18 and independent final
64-row A/B sets. Training/final RNGs7301/9101 and reserved validation RNG307 are
fixed. All native learning objectives/factors and .01 learning rate/two PC
inference steps at .05 remain unchanged. It is exploratory, without intervals,
equivalence, confirmation, efficiency or circadian-advantage claims.

Raw methods compare fixed width3, fixed width4, scheduled2→4 growth and guarded
split/prune. The width3 reference is a data-blind pure prune of the untrained
split source, with a different initial function. Both references and initial
absolute levels remain visible. Ordinary effective-coordinate fixed/renewal
cohorts initialize from the wide effective parameters; renewal compares only
to its same ordinary method. These coordinates/rate factors are not pooled or
called equal compute/effective capacity. Final data never select updates,
transforms, probes, promotion, schedules or thresholds.

**Negative/limited result:** every guarded prune is rejected under the unchanged
.05 damage limits. Split/prune therefore preserves exactly the recorded growth
checkpoints, receipts and scores, with extra validation work. All three renewal
arms select zero eligible units at nine scheduled events; their scores and
recorded native parameters/clocks match ordinary controls. They add 96 history
bytes per candidate and nine zero-unit native calls/issued consolidations across
the family. No threshold was tuned and no experiment was retried.

All B BCE gains are small and positive; initial B accuracy is already .96875
for wide/growth/ordinary and1 for the width3 reference. Some final B accuracies
improve one of64 rows; width3 B accuracy instead decreases to .96875. Many A
accuracies decline. This is not meaningful fresh-task acquisition evidence.
Active CPC reduces BCE forgetting while also reducing B/return gain relative
to neutral CPC: a tradeoff, not an established successful bundle.

## Every arm: outcomes and candidate costs

B gain=B0−B12 (higher), retention loss=A12−A6 (lower), return gain=A12−A18
(higher). Values below are displayed to twelve decimals; full precision is in
the saved JSON. Every arm completes18 wakes/144 training rows/eight final score
forwards. Byte means/peaks count committed native arrays, with chemistry/history
explicitly included. They exclude copies, metadata, transient activations,
source/actor/scorer payloads and process allocations.

| Method | Mode | B gain | Retention loss | Return gain | Mean / peak parameters | Mean / peak array bytes | Final width |
|---|---|---:|---:|---:|---:|---:|---:|
| bp | fixed_pruned_width | 0.000963480529 | 0.000427698933 | 0.000760012059 | 13 / 13 | 104 / 104 | 3 |
| bp | fixed_wide | 0.001234257733 | 0.000914254866 | 0.001790257295 | 17 / 17 | 136 / 136 | 4 |
| bp | scheduled_growth | 0.001156178183 | 0.000913263991 | 0.001788889151 | 14.3333 / 17 | 114.667 / 136 | 4 |
| bp | split_prune | 0.001156178183 | 0.000913263991 | 0.001788889151 | 14.3333 / 17 | 114.667 / 136 | 4 |
| pc | fixed_pruned_width | 0.000507831481 | 0.000383790418 | 0.000185076593 | 13 / 13 | 104 / 104 | 3 |
| pc | fixed_wide | 0.000360810373 | 0.000786651124 | 0.000364983703 | 17 / 17 | 136 / 136 | 4 |
| pc | scheduled_growth | 0.000371123099 | 0.000786578006 | 0.000364941515 | 14.3333 / 17 | 114.667 / 136 | 4 |
| pc | split_prune | 0.000371123099 | 0.000786578006 | 0.000364941515 | 14.3333 / 17 | 114.667 / 136 | 4 |
| neutral_cpc | fixed_pruned_width | 0.000507831481 | 0.000383790418 | 0.000185076593 | 13 / 13 | 152 / 152 | 3 |
| neutral_cpc | fixed_wide | 0.000360810373 | 0.000786651124 | 0.000364983703 | 17 / 17 | 200 / 200 | 4 |
| neutral_cpc | scheduled_growth | 0.000371123099 | 0.000786578006 | 0.000364941515 | 14.3333 / 17 | 168 / 200 | 4 |
| neutral_cpc | split_prune | 0.000371123099 | 0.000786578006 | 0.000364941515 | 14.3333 / 17 | 168 / 200 | 4 |
| gated_cpc | fixed_pruned_width | 0.000410208860 | 0.000343083340 | 0.000153489563 | 13 / 13 | 152 / 152 | 3 |
| gated_cpc | fixed_wide | 0.000260988653 | 0.000717085180 | 0.000302876539 | 17 / 17 | 200 / 200 | 4 |
| gated_cpc | scheduled_growth | 0.000271061691 | 0.000717055533 | 0.000302849519 | 14.3333 / 17 | 168 / 200 | 4 |
| gated_cpc | split_prune | 0.000271061691 | 0.000717055533 | 0.000302849519 | 14.3333 / 17 | 168 / 200 | 4 |
| bp | ordinary_fixed | 0.003175599650 | 0.003665132347 | 0.002831672590 | 17 / 17 | 136 / 136 | 4 |
| pc | ordinary_fixed | 0.003065360637 | 0.003655866050 | 0.002546565905 | 17 / 17 | 136 / 136 | 4 |
| neutral_cpc | ordinary_fixed | 0.003065360637 | 0.003655866050 | 0.002546565905 | 17 / 17 | 200 / 200 | 4 |
| gated_cpc | ordinary_fixed | 0.002669617370 | 0.003376613242 | 0.002297883364 | 17 / 17 | 200 / 200 | 4 |
| pc | bounded_renewal | 0.003065360637 | 0.003655866050 | 0.002546565905 | 17 / 17 | 232 / 232 | 4 |
| neutral_cpc | bounded_renewal | 0.003065360637 | 0.003655866050 | 0.002546565905 | 17 / 17 | 296 / 296 | 4 |
| gated_cpc | bounded_renewal | 0.002669617370 | 0.003376613242 | 0.002297883364 | 17 / 17 | 296 / 296 | 4 |

All four rejected split/prune arms retain final width4. Growth uses six wakes
at9 parameters and twelve at17: mean14.3333333333/peak17, versus fixed4 mean17/
peak17. The unused counterfactual accepted-prune trace is not substituted.
Fixed3 has mean/peak13 with its different initial function. CPC growth native
array mean168/peak200 B; PC/BP114.666666667/136 B. Ordinary renewable PC232 B,
CPC296 B versus ordinary fixed136/200 B. These measured counts alone do not
establish efficiency. Initial representation transforms and gate work count too.

## All 184 context score points

Each row below contains both context scores at one checkpoint (92 rows total).

| Method | Mode | Wake | A BCE | A accuracy | B BCE | B accuracy |
|---|---|---:|---:|---:|---:|---:|
| bp | fixed_pruned_width | 0 | 0.706831761974 | 0.421875 | 0.594841381113 | 1.000000 |
| bp | fixed_pruned_width | 6 | 0.706080368417 | 0.406250 | 0.596027030183 | 0.968750 |
| bp | fixed_pruned_width | 12 | 0.706508067349 | 0.421875 | 0.593877900584 | 0.984375 |
| bp | fixed_pruned_width | 18 | 0.705748055291 | 0.406250 | 0.594898009445 | 0.968750 |
| bp | fixed_wide | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| bp | fixed_wide | 6 | 0.745673177798 | 0.390625 | 0.512250269826 | 0.984375 |
| bp | fixed_wide | 12 | 0.746587432664 | 0.406250 | 0.509918243043 | 0.984375 |
| bp | fixed_wide | 18 | 0.744797175369 | 0.390625 | 0.511110668488 | 0.984375 |
| bp | scheduled_growth | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| bp | scheduled_growth | 6 | 0.745742424791 | 0.390625 | 0.512330942932 | 0.984375 |
| bp | scheduled_growth | 12 | 0.746655688782 | 0.390625 | 0.509996322593 | 0.984375 |
| bp | scheduled_growth | 18 | 0.744866799631 | 0.375000 | 0.511189390803 | 0.984375 |
| bp | split_prune | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| bp | split_prune | 6 | 0.745742424791 | 0.390625 | 0.512330942932 | 0.984375 |
| bp | split_prune | 12 | 0.746655688782 | 0.390625 | 0.509996322593 | 0.984375 |
| bp | split_prune | 18 | 0.744866799631 | 0.375000 | 0.511189390803 | 0.984375 |
| pc | fixed_pruned_width | 0 | 0.706831761974 | 0.421875 | 0.594841381113 | 1.000000 |
| pc | fixed_pruned_width | 6 | 0.706735403373 | 0.406250 | 0.596186875649 | 0.984375 |
| pc | fixed_pruned_width | 12 | 0.707119193790 | 0.421875 | 0.594333549632 | 0.984375 |
| pc | fixed_pruned_width | 18 | 0.706934117197 | 0.406250 | 0.595457528716 | 0.968750 |
| pc | fixed_wide | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| pc | fixed_wide | 6 | 0.747307553342 | 0.390625 | 0.512633623429 | 0.984375 |
| pc | fixed_wide | 12 | 0.748094204466 | 0.390625 | 0.510791690403 | 0.984375 |
| pc | fixed_wide | 18 | 0.747729220763 | 0.375000 | 0.512212749822 | 0.984375 |
| pc | scheduled_growth | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| pc | scheduled_growth | 6 | 0.747302221499 | 0.390625 | 0.512623297053 | 0.984375 |
| pc | scheduled_growth | 12 | 0.748088799505 | 0.390625 | 0.510781377677 | 0.984375 |
| pc | scheduled_growth | 18 | 0.747723857990 | 0.375000 | 0.512202359399 | 0.984375 |
| pc | split_prune | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| pc | split_prune | 6 | 0.747302221499 | 0.390625 | 0.512623297053 | 0.984375 |
| pc | split_prune | 12 | 0.748088799505 | 0.390625 | 0.510781377677 | 0.984375 |
| pc | split_prune | 18 | 0.747723857990 | 0.375000 | 0.512202359399 | 0.984375 |
| neutral_cpc | fixed_pruned_width | 0 | 0.706831761974 | 0.421875 | 0.594841381113 | 1.000000 |
| neutral_cpc | fixed_pruned_width | 6 | 0.706735403373 | 0.406250 | 0.596186875649 | 0.984375 |
| neutral_cpc | fixed_pruned_width | 12 | 0.707119193790 | 0.421875 | 0.594333549632 | 0.984375 |
| neutral_cpc | fixed_pruned_width | 18 | 0.706934117197 | 0.406250 | 0.595457528716 | 0.968750 |
| neutral_cpc | fixed_wide | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| neutral_cpc | fixed_wide | 6 | 0.747307553342 | 0.390625 | 0.512633623429 | 0.984375 |
| neutral_cpc | fixed_wide | 12 | 0.748094204466 | 0.390625 | 0.510791690403 | 0.984375 |
| neutral_cpc | fixed_wide | 18 | 0.747729220763 | 0.375000 | 0.512212749822 | 0.984375 |
| neutral_cpc | scheduled_growth | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| neutral_cpc | scheduled_growth | 6 | 0.747302221499 | 0.390625 | 0.512623297053 | 0.984375 |
| neutral_cpc | scheduled_growth | 12 | 0.748088799505 | 0.390625 | 0.510781377677 | 0.984375 |
| neutral_cpc | scheduled_growth | 18 | 0.747723857990 | 0.375000 | 0.512202359399 | 0.984375 |
| neutral_cpc | split_prune | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| neutral_cpc | split_prune | 6 | 0.747302221499 | 0.390625 | 0.512623297053 | 0.984375 |
| neutral_cpc | split_prune | 12 | 0.748088799505 | 0.390625 | 0.510781377677 | 0.984375 |
| neutral_cpc | split_prune | 18 | 0.747723857990 | 0.375000 | 0.512202359399 | 0.984375 |
| gated_cpc | fixed_pruned_width | 0 | 0.706831761974 | 0.421875 | 0.594841381113 | 1.000000 |
| gated_cpc | fixed_pruned_width | 6 | 0.706741828732 | 0.406250 | 0.596181701734 | 0.984375 |
| gated_cpc | fixed_pruned_width | 12 | 0.707084912072 | 0.421875 | 0.594431172254 | 0.984375 |
| gated_cpc | fixed_pruned_width | 18 | 0.706931422509 | 0.406250 | 0.595513396184 | 0.968750 |
| gated_cpc | fixed_wide | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| gated_cpc | fixed_wide | 6 | 0.747319252271 | 0.390625 | 0.512620633777 | 0.984375 |
| gated_cpc | fixed_wide | 12 | 0.748036337452 | 0.390625 | 0.510891512123 | 0.984375 |
| gated_cpc | fixed_wide | 18 | 0.747733460913 | 0.375000 | 0.512232837133 | 0.984375 |
| gated_cpc | scheduled_growth | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| gated_cpc | scheduled_growth | 6 | 0.747314037801 | 0.390625 | 0.512610610437 | 0.984375 |
| gated_cpc | scheduled_growth | 12 | 0.748031093333 | 0.390625 | 0.510881439085 | 0.984375 |
| gated_cpc | scheduled_growth | 18 | 0.747728243815 | 0.375000 | 0.512222725298 | 0.984375 |
| gated_cpc | split_prune | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| gated_cpc | split_prune | 6 | 0.747314037801 | 0.390625 | 0.512610610437 | 0.984375 |
| gated_cpc | split_prune | 12 | 0.748031093333 | 0.390625 | 0.510881439085 | 0.984375 |
| gated_cpc | split_prune | 18 | 0.747728243815 | 0.375000 | 0.512222725298 | 0.984375 |
| bp | ordinary_fixed | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| bp | ordinary_fixed | 6 | 0.744855298906 | 0.390625 | 0.514426060625 | 0.984375 |
| bp | ordinary_fixed | 12 | 0.748520431253 | 0.406250 | 0.507976901126 | 0.984375 |
| bp | ordinary_fixed | 18 | 0.745688758663 | 0.375000 | 0.511485490090 | 0.984375 |
| pc | ordinary_fixed | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| pc | ordinary_fixed | 6 | 0.745176303192 | 0.390625 | 0.514424456209 | 0.984375 |
| pc | ordinary_fixed | 12 | 0.748832169242 | 0.390625 | 0.508087140139 | 0.984375 |
| pc | ordinary_fixed | 18 | 0.746285603337 | 0.390625 | 0.511576011399 | 0.984375 |
| neutral_cpc | ordinary_fixed | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| neutral_cpc | ordinary_fixed | 6 | 0.745176303192 | 0.390625 | 0.514424456209 | 0.984375 |
| neutral_cpc | ordinary_fixed | 12 | 0.748832169242 | 0.390625 | 0.508087140139 | 0.984375 |
| neutral_cpc | ordinary_fixed | 18 | 0.746285603337 | 0.390625 | 0.511576011399 | 0.984375 |
| gated_cpc | ordinary_fixed | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| gated_cpc | ordinary_fixed | 6 | 0.745221399944 | 0.390625 | 0.514372519691 | 0.984375 |
| gated_cpc | ordinary_fixed | 12 | 0.748598013186 | 0.390625 | 0.508482883407 | 0.984375 |
| gated_cpc | ordinary_fixed | 18 | 0.746300129822 | 0.390625 | 0.511650391392 | 0.984375 |
| pc | bounded_renewal | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| pc | bounded_renewal | 6 | 0.745176303192 | 0.390625 | 0.514424456209 | 0.984375 |
| pc | bounded_renewal | 12 | 0.748832169242 | 0.390625 | 0.508087140139 | 0.984375 |
| pc | bounded_renewal | 18 | 0.746285603337 | 0.390625 | 0.511576011399 | 0.984375 |
| neutral_cpc | bounded_renewal | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| neutral_cpc | bounded_renewal | 6 | 0.745176303192 | 0.390625 | 0.514424456209 | 0.984375 |
| neutral_cpc | bounded_renewal | 12 | 0.748832169242 | 0.390625 | 0.508087140139 | 0.984375 |
| neutral_cpc | bounded_renewal | 18 | 0.746285603337 | 0.390625 | 0.511576011399 | 0.984375 |
| gated_cpc | bounded_renewal | 0 | 0.747598128726 | 0.390625 | 0.511152500776 | 0.968750 |
| gated_cpc | bounded_renewal | 6 | 0.745221399944 | 0.390625 | 0.514372519691 | 0.984375 |
| gated_cpc | bounded_renewal | 12 | 0.748598013186 | 0.390625 | 0.508482883407 | 0.984375 |
| gated_cpc | bounded_renewal | 18 | 0.746300129822 | 0.390625 | 0.511650391392 | 0.984375 |

All52 declared paired comparisons and416 checkpoint/context differences are
retained in JSON, including exact-zero self/reference and no-activation pairs.
No failed/inconclusive case or unfavorable accuracy/retention outcome is removed.

## Prune observations, including every rejection

Validation is two independent four-row A/B batches, available at tick0; never
final data or native training. The following candidate damage exceeds .05 in
every method/context. Each complete gate performs four score forwards/sixteen
row-forwards/twelve energy evaluations. BP uses the same declared PC-energy
diagnostic without changing its feedforward training objective.

| Method | Validation context | Max prediction change | BCE increase | Inferred-energy increase | Gate |
|---|---|---:|---:|---:|---|
| bp | A | 0.072172501777 | 0.104400813084 | 0.104655358017 | rejected |
| bp | B | 0.075671948686 | 0.119316602651 | 0.119498292180 | rejected |
| pc | A | 0.071902851810 | 0.103903574919 | 0.104160980111 | rejected |
| pc | B | 0.075559579564 | 0.119116452991 | 0.119299518716 | rejected |
| neutral_cpc | A | 0.071902851810 | 0.103903574919 | 0.104160980111 | rejected |
| neutral_cpc | B | 0.075559579564 | 0.119116452991 | 0.119299518716 | rejected |
| gated_cpc | A | 0.071861643295 | 0.103853212288 | 0.104110527217 | rejected |
| gated_cpc | B | 0.075516949353 | 0.119062053802 | 0.119245142336 | rejected |

## Observed work, time and memory

Native controller:414 training entries/returns,3312 training row exposures,
216 raw PC steps/72 raw BP/108 ordinary PC/18 ordinary BP,1296 training-energy
evaluations plus48 gate evaluations=1344,180 BP evaluations.184 final score
forwards;23 splits (15 initialization/eight scheduled),eight prune kernels
(four initialization/four gates),nine zero-unit renewal calls. Validation16
forwards/64 row-forwards is separate from training/final evaluation.

Pilot 5.5527321s, worker5.5962070s, parent6.0820386s; one
joined RSS sampler, peak45,977,600 B, report930,899 B, stderr empty. CPU/BLAS1,
GPU0/60s body90s parent/256MiB RSS/2MiB output limits passed. No Torch, probe,
replay, download, nested subprocess, sweep or experiment retry.

Included boundary times: initialization 0.2892581s,
measurement 1.6581677s, checkpoint reporting
0.1178835s, pruning 0.2240917s,
renewal 0.0505888s. These are included in full wall time;
they are not causal overhead, parallel-independent totals or efficiency estimates.
Reserved validation arrays192 B and binding/copy/temporary memory are outside
candidate array counts; process RSS covers the process without per-arm attribution.
Environment: Python3.14.7/NumPy2.4.6, Windows CPU.

## Qualification and preserved failures

134 distinct supported cases/402 passed qualified phases across135 executions:
initial134 invocation had133 passes and one missing-FakeSampler-field failure
before that case's native work; the repaired sampler's one affected case passes.
All27 other frozen texts and AST outside that sampler remain exact. Original
failed receipt/log preserved; no whole native family rerun. Final eight-path
Ruff/format/Windows+Linux mypy pass; root aggregate18.0939079s. Source-author
initial Optional-local mypy error repaired (14.0997452s total), all outputs kept;
test author first gates pass. Successful native pruning/next updates are verified
by focused behavioral cases, not claimed as an activated outcome in this study.

The first H1 two-coordinate prune draft was invalid for the native uniform API;
corrected prospectively to(2,)4→3. The corresponding split census was corrected
from19 to23 before execution. Invalid drafts remain in SCOPE, with rationale;
no data/score-driven change, relaxed damage threshold or budget renewal.

Full suite/repository gates, clean clone/Linux execution, unchanged raw math/
Torch/checkpoint/legacy pruning matrices were not rerun. Scoped correctness and
this fixed experiment do not establish repository-wide or release readiness.

Evidence: [results/commands](../../artifacts/runs/r26-full-capacity-20261016/RESULTS.md),
[independent review](../../artifacts/runs/r26-full-capacity-20261016/REVIEW.md),
[full saved data](../../artifacts/runs/r26-full-capacity-20261016/summary.json),
[usage and limits](../capacity-comparison.md).
