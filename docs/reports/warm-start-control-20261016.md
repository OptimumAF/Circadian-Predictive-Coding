# One frozen matched warm-start comparison, 2026-10-16

Classification: exploratory local case, seed17; no circadian comparison. Both
arms freshly reproduce the same architecture/eight demonstration packets and
match full native state at8 before any reward collection. All training finishes
before final scoring/live evaluation. No baseline/seed/rate/metric/horizon tuning.

| Endpoint metric | Frozen BC, native8 | BC→reward, native16 | Registered signed difference |
|---|---:|---:|---:|
| Mean CE | 1.5842539963 | 1.5641386719 | BC−RL +0.0201153244 |
| Mean accuracy | 0.390625 | 0.359375 | RL−BC −0.03125 |
| Success rate | 0 | 0 | RL−BC 0 |
| Mean return | 0 | 0 | RL−BC 0 |

Displayed CE values are rounded; raw JSON retains all precision and every signed
context row. A/D CE improve; B CE improves while B accuracy worsens; C CE worsens.
All eight contexts/cues remain included. No metric is substituted to declare a win.

All64 fresh live episodes across checkpoints fail illegal actions with zero
success/return;72 legal steps are actually applied before those failures.
Endpoint8/16 each includes eight episodes, all failures. All eight sampled
training episodes also fail illegal with zero reward, lengths3/1/1/1/1/1/2/1,
11 decisions/three applied steps. All eight reward diagnostics retain −0.0.

Zero reward makes the unchanged reward loss have zero gradients, while carried
BC momentum decays by0.9 each SGD step. Saved two-step parameter changes follow
the0.81 geometric factor within2.6524067e−8 float32 residual. This supports momentum
carry, not reward-gradient learning. Lower CE with worse accuracy and zero return
does not justify the added reward-phase work in this case. It does not establish
an algorithm-wide ranking or useful adaptation/retention/recovery/generalization.

## Costs and limits

- Control8SGD/64teacher rows; reward arm16SGD/64teacher+11sampled rows. Both use
  the same64 teacher rows as separate actual train collections. Original budgets
  charge8/16 permanently; no reset, renewal, replay or probe.
- Total24SGD/139train rows;64 cached forwards/512rows and136live forwards plus
  11sampled forwards =211predictions.139+512+136+11=798unrolled steps.
- Sixteen train teacher episodes128steps +eight final teacher episodes64steps
  +three sampled legal steps +72live steps =267physical steps,96environments.
- Recipe6.3765353s/worker6.958684s/parent7.6732417s; sampled process RSS314019840B,
  1179samples/requested5ms, including Torch import in the recipe window.529716B
  JSON/stderr0; CPU/BLAS/Torch/interop1/GPU0/final Python thread count1.
- All frozen24SGD224row1600unroll928predict1056step96environment90body120parent
  768MiB8MiB caps pass. One native case, no retry or post-result experimental work.

187 distinct focused cases qualify; final five-path statics and independent
SOURCE/GO gate the case. Saved-data analysis checks21frozen texts,24receipts,
both exact teacher sequences,11private draws, all signed endpoint metrics,
native clocks/charges and resource sums without executing learners again.

## Reassessed next action

The existing reserved fresh-task probe reports aggregate CE/accuracy, without
per-decision/class confusion. Earlier E probe accuracy50% matched a majority
rule; this pair also shows CE and accuracy moving in opposite directions.
Add opt-in bounded per-decision/class diagnostics from those already-returned
reserved probe predictions. Keep existing metrics/defaults, source/actor/budget
isolation and no additional native calls. Implement and test this before another
experimental protocol; do not tune from these final scores or extend episodes
until successful. Full R4.2 and broader research/release goals remain open.

[Raw pair](../../artifacts/runs/r42-warm-start-control-20261016/paired.json),
[saved analysis](../../artifacts/runs/r42-warm-start-control-20261016/analysis.json),
[commands/caps](../../artifacts/runs/r42-warm-start-control-20261016/RESULTS.md),
[independent review](../../artifacts/runs/r42-warm-start-control-20261016/REVIEW.md).
