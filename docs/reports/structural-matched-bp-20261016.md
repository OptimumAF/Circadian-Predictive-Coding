# Structural matched BP — retained capacity comparison, 2026-10-16

## Delivered and accepted scope

The new owned structural BP port uses the unchanged raw matched-BP kernel.
Focused checks execute training after real split, compatible merge, prune and
earlier rollback for one/two hidden layers, with complete root/history and array
ownership. Existing fixed-config BP restore remains strict. Independent SOURCE,
GO and ACTUAL review accepted this prerequisite; original **R2.6 remains open**.

The sole new native case uses `--scenario capacity-bp` in the existing rehearsal
CLI: two BP arms only. Seed 23, D2/H1/W2→4, learning rate .01, four two-row
A/A/B/A updates with two-tick delay, split after update two, checkpoints 0/2/4
and separate final A/B rows are frozen. Existing raw scaling/precision factors,
BCE clip 1e-12 (lower better) and inclusive .5 accuracy (higher better) persist.
Final scoring occurs after training; final data never choose updates, transforms,
probes, promotion or tuning. No new probe, replay, renewal or algorithm is added.

The accepted PC/neutral/active CPC native case is reused after unchanged relevant
math/data/helper texts, six original pilot function ASTs and its original
descriptor were reconciled. No PC/CPC experiment was rerun. Historical scores
enter only external saved-result analysis; the producer never loads them.
Initial parameters, scaling factors and scores match exactly in each same mode;
24 receipt pairs match on nine common non-diagnostic fields. Native diagnostic
units/objectives differ and are not compared as a shared learning metric.

## Every retained score point

Rows below contain all twelve new BP and 36 retained PC/CPC context score points.
Values are displayed to twelve decimals; original precision remains in JSON.

| Method | Mode | Wake | A BCE | A accuracy | B BCE | B accuracy |
|---|---|---:|---:|---:|---:|---:|
| bp | fixed_final_width | 0 | 0.793302439887 | 0.5 | 0.519331858986 | 1 |
| bp | fixed_final_width | 2 | 0.792255967863 | 0.5 | 0.518675094812 | 1 |
| bp | fixed_final_width | 4 | 0.791979818485 | 0.5 | 0.518606757680 | 1 |
| pc | fixed_final_width | 0 | 0.793302439887 | 0.5 | 0.519331858986 | 1 |
| pc | fixed_final_width | 2 | 0.793142036715 | 0.5 | 0.519274334067 | 1 |
| pc | fixed_final_width | 4 | 0.793080030717 | 0.5 | 0.519308697781 | 1 |
| neutral_cpc | fixed_final_width | 0 | 0.793302439887 | 0.5 | 0.519331858986 | 1 |
| neutral_cpc | fixed_final_width | 2 | 0.793142036715 | 0.5 | 0.519274334067 | 1 |
| neutral_cpc | fixed_final_width | 4 | 0.793080030717 | 0.5 | 0.519308697781 | 1 |
| gated_cpc | fixed_final_width | 0 | 0.793302439887 | 0.5 | 0.519331858986 | 1 |
| gated_cpc | fixed_final_width | 2 | 0.793139410614 | 0.5 | 0.519278386828 | 1 |
| gated_cpc | fixed_final_width | 4 | 0.793081309414 | 0.5 | 0.519309549938 | 1 |
| bp | scheduled_split | 0 | 0.793302439887 | 0.5 | 0.519331858986 | 1 |
| bp | scheduled_split | 2 | 0.792256013285 | 0.5 | 0.518672227958 | 1 |
| bp | scheduled_split | 4 | 0.791979882259 | 0.5 | 0.518603959553 | 1 |
| pc | scheduled_split | 0 | 0.793302439887 | 0.5 | 0.519331858986 | 1 |
| pc | scheduled_split | 2 | 0.793127252027 | 0.5 | 0.519250453125 | 1 |
| pc | scheduled_split | 4 | 0.793065265578 | 0.5 | 0.519284807739 | 1 |
| neutral_cpc | scheduled_split | 0 | 0.793302439887 | 0.5 | 0.519331858986 | 1 |
| neutral_cpc | scheduled_split | 2 | 0.793127252027 | 0.5 | 0.519250453125 | 1 |
| neutral_cpc | scheduled_split | 4 | 0.793065265578 | 0.5 | 0.519284807739 | 1 |
| gated_cpc | scheduled_split | 0 | 0.793302439887 | 0.5 | 0.519331858986 | 1 |
| gated_cpc | scheduled_split | 2 | 0.793124716044 | 0.5 | 0.519254764037 | 1 |
| gated_cpc | scheduled_split | 4 | 0.793066631230 | 0.5 | 0.519285921001 | 1 |

## Cost and interpretation

| New BP arm | Wake-use parameter trace | Mean / peak parameters | Mean / peak native array bytes | Auxiliary bytes |
|---|---|---|---|---:|
| Fixed final width | 17,17,17,17 | 17 / 17 | 136 / 136 | 0 |
| Scheduled split | 9,9,17,17 | 13 / 17 | 104 / 136 | 0 |

Means weight actual wake calls using the state that performed each wake. Samples
describe committed candidate arrays, excluding source/actor/scorer/copies,
temporary inference/gradients, Python metadata and process allocation. Retained
PC has the same parameter/array counts; CPC mean arrays are 152 growing versus
200 fixed, equal peak 200, because chemistry is additional native state.

Actual new work: eight native BP calls/returns and raw steps, sixteen training
rows, twelve score forwards, two splits (one fixed initialization, one scheduled)
and sixteen BP evaluations; zero PC inference/coupled-energy calls. One RSS
sampler starts and joins. Pilot/worker/parent times are .2227112/.2618966/.9228935
seconds, RSS peak 43,307,008 bytes and output 36,125 bytes, stderr empty.
Measurement .0447562 and checkpoint reporting .0080146 seconds are included in
wall time; they are not causal overhead or comparative efficiency estimates.
CPU/BLAS1, GPU0, 60-second body/90-second parent, 256 MiB RSS/2 MiB output limits
passed. No retry, seed/metric/baseline selection, downloads or large sweep ran.

All arms retain A accuracy .5 and B accuracy 1, including initial B accuracy 1.
BP has lower BCE on this fixture, but objectives, inference work and gradients
differ. Scheduled versus fixed BP has tiny opposite final A/B changes. Replicated
units, mapped precisions and width-dependent update factors mean equal initial
function/final stored width is not equal effective capacity or compute. This case
qualifies structural ownership/continuation and accounting, not fresh-task
acquisition, useful recovery, efficiency or a circadian advantage.

## Verification and preserved failures

117 focused cases/351 passed phases, zero failures/skips; final five-path Ruff,
format and Windows/Linux mypy pass. Full suite/repository-wide gates, clean clone,
Linux runtime and unchanged raw math/Torch/checkpoint matrices were not rerun.
Their accepted scoped evidence was reused where sources/assumptions stay exact.

The test author's initial Ruff unused-import failure was fixed before root
qualification. The external summary first requested nonexistent `experience_id`;
its KeyError is retained in `summarize-first-failure.json`. Reading the actual
receipt dataclass and comparing its nine real non-diagnostic fields corrected
saved-data analysis only, with no native rerun. No experiment failure is hidden.

See [commands/results](../../artifacts/runs/r26-structural-bp-20261016/RESULTS.md),
[independent review](../../artifacts/runs/r26-structural-bp-20261016/REVIEW.md),
[saved comparison](../../artifacts/runs/r26-structural-bp-20261016/summary.json)
and [usage](../structural-matched-bp.md).

## Exact next implementation

Implement the full R2.6 mechanism/measurement binding in the existing stream.
First allow a measurement-only capacity observer alongside the existing bounded
renewal helper, recording before/after-reset usage and separate reset work/charges
without changing wake-weighted means or renewal eligibility. Reconcile ordinary
renewal versus raw structural state compatibility explicitly; then bind the
split/prune, scheduled-growth, equal-final-width and matched BP cohorts and freeze
the complete local comparison after focused gates. Why this: the matched raw BP
port is now ready, while the runner still excludes simultaneous capacity/renewal
helpers. This is a dependency of the full comparison, not a reason to run another
nearly identical capacity-only example. No new scientific execution is budgeted.
