# Slow-teacher replay: fixed local ablation, 2026-10-16

The original **R2.5 implementation and ablation criteria are accepted** after
independent SOURCE/GO/ACTUAL review. This establishes a runnable versioned target
path and a measured comparison, without establishing scientific benefit.

The existing coherent NumPy BP, PC, neutral CPC and active CPC use a parameter
teacher updated only after successful wakes. Replay recomputes probabilities on
the existing selected arrived rows and trains against `(1-lambda)y + lambda*q`.
The frozen settings were lambda0.25, EMA decay0.9, seed23, A/B/A18wake batches,
original initialization/data/rates/replay selection and isolated final scoring.
The retained replay-only control was reused after checking relevant source and
protocol assumptions. No control experiment was rerun or changed.

## Observed outcomes

All four teacher arms have **higher forgetting and lower return gains** than
replay alone. Their A BCE is worse at every trained checkpoint6/12/18. PC-family
new-task B BCE gains improve slightly, but B12 accuracy falls by1/64 in every
method, including BP. BP retains the lowest absolute B12 and A18 BCE. These are
single-seed fixed-data observations; no aggregate improvement, equivalence,
efficiency, causal mechanism or circadian advantage follows.

Gain = B0 BCE minus B12; forgetting = A12 minus A6 (lower is better);
return = A12 minus A18. Accuracy uses the inclusive0.5 threshold; BCE clips at
1e-12. Every point and negative result is retained.

| Learner | Mode | B gain ↑ | Forgetting ↓ | Return gain ↑ |
|---|---|---:|---:|---:|
| bp | Replay only | 0.074078382 | 0.002639345 | 0.066178068 |
| bp | Teacher | 0.070372792 | 0.004527230 | 0.062648979 |
| pc | Replay only | 0.016271081 | 0.009988561 | 0.029150157 |
| pc | Teacher | 0.016871101 | 0.010331848 | 0.027730609 |
| neutral_cpc | Replay only | 0.016271081 | 0.009988561 | 0.029150157 |
| neutral_cpc | Teacher | 0.016871101 | 0.010331848 | 0.027730609 |
| gated_cpc | Replay only | 0.014433366 | 0.008950487 | 0.025944644 |
| gated_cpc | Teacher | 0.015167561 | 0.009333988 | 0.024551706 |

## Full curves

Each cell reports BCE / accuracy; raw JSON retains full precision.

| Learner | Mode | Wake | A BCE / accuracy | B BCE / accuracy |
|---|---|---:|---:|---:|
| bp | Replay only | 0 | 0.763141840 / 0.406250 | 0.834174696 / 0.109375 |
| bp | Replay only | 6 | 0.674373780 / 0.578125 | 0.864972256 / 0.281250 |
| bp | Replay only | 12 | 0.677013125 / 0.500000 | 0.760096314 / 0.312500 |
| bp | Replay only | 18 | 0.610835057 / 0.890625 | 0.713584875 / 0.437500 |
| bp | Teacher | 0 | 0.763141840 / 0.406250 | 0.834174696 / 0.109375 |
| bp | Teacher | 6 | 0.679416998 / 0.609375 | 0.856338918 / 0.218750 |
| bp | Teacher | 12 | 0.683944228 / 0.531250 | 0.763801904 / 0.296875 |
| bp | Teacher | 18 | 0.621295248 / 0.843750 | 0.730375079 / 0.406250 |
| pc | Replay only | 0 | 0.763141840 / 0.406250 | 0.834174696 / 0.109375 |
| pc | Replay only | 6 | 0.719114929 / 0.515625 | 0.856340654 / 0.093750 |
| pc | Replay only | 12 | 0.729103490 / 0.437500 | 0.817903615 / 0.078125 |
| pc | Replay only | 18 | 0.699953333 / 0.578125 | 0.801369929 / 0.078125 |
| pc | Teacher | 0 | 0.763141840 / 0.406250 | 0.834174696 / 0.109375 |
| pc | Teacher | 6 | 0.722880243 / 0.515625 | 0.852072945 / 0.062500 |
| pc | Teacher | 12 | 0.733212091 / 0.437500 | 0.817303595 / 0.062500 |
| pc | Teacher | 18 | 0.705481482 / 0.562500 | 0.806406547 / 0.046875 |
| neutral_cpc | Replay only | 0 | 0.763141840 / 0.406250 | 0.834174696 / 0.109375 |
| neutral_cpc | Replay only | 6 | 0.719114929 / 0.515625 | 0.856340654 / 0.093750 |
| neutral_cpc | Replay only | 12 | 0.729103490 / 0.437500 | 0.817903615 / 0.078125 |
| neutral_cpc | Replay only | 18 | 0.699953333 / 0.578125 | 0.801369929 / 0.078125 |
| neutral_cpc | Teacher | 0 | 0.763141840 / 0.406250 | 0.834174696 / 0.109375 |
| neutral_cpc | Teacher | 6 | 0.722880243 / 0.515625 | 0.852072945 / 0.062500 |
| neutral_cpc | Teacher | 12 | 0.733212091 / 0.437500 | 0.817303595 / 0.062500 |
| neutral_cpc | Teacher | 18 | 0.705481482 / 0.562500 | 0.806406547 / 0.046875 |
| gated_cpc | Replay only | 0 | 0.763141840 / 0.406250 | 0.834174696 / 0.109375 |
| gated_cpc | Replay only | 6 | 0.720630658 / 0.515625 | 0.856036055 / 0.093750 |
| gated_cpc | Replay only | 12 | 0.729581145 / 0.437500 | 0.819741330 / 0.078125 |
| gated_cpc | Replay only | 18 | 0.703636501 / 0.578125 | 0.804306840 / 0.031250 |
| gated_cpc | Teacher | 0 | 0.763141840 / 0.406250 | 0.834174696 / 0.109375 |
| gated_cpc | Teacher | 6 | 0.724198812 / 0.500000 | 0.851855769 / 0.062500 |
| gated_cpc | Teacher | 12 | 0.733532800 / 0.437500 | 0.819007136 / 0.062500 |
| gated_cpc | Teacher | 18 | 0.708981094 / 0.546875 | 0.808833073 / 0.062500 |

## Work and storage

Both families have120 native calls (72wake/48replay),624 training-row exposures,
24 consolidation events and32 final-score forwards. Every selected ID,
retention membership, rotation cursor and exposure matches the retained control.
The teacher adds48 forwards and72 parameter EMA updates. It stores840 parameter
array bytes per arm, with no retained target cache; generated probabilities and
returned mixed-target payloads each total384 bytes across all arms. Peak single
probability payload is8 bytes per arm. The replay cap is384 bytes per arm.

Student parameters840 + teacher parameters840 + full replay384 =2064 bytes per
arm. CPC additionally retains128 bytes of native chemistry/gates:2192 bytes for
those native arrays plus teacher parameters and full replay. These are logical
array subtotals, excluding copies, checkpoints, metadata, Python and other process
memory. Measured RSS was42,377,216 bytes. The four arms ran in0.9323947 seconds
(profiled worker0.9686543, parent1.4668616); these process timings do not establish
efficiency relative to historical timings.

The first post-run summary's2064-byte label omitted CPC auxiliary arrays; that
artifact is preserved. `summary-corrected.json` names the parameter subtotal and
adds the128-byte CPC component. Metrics and original native output are unchanged.
Failed-work tests preserve teacher attempts and permanent replay charges; partial
failed wake native work is not inferred from completed wake counters.

## Qualification and limits

161 focused cases /483 passed phases; scoped Ruff, format and Windows/Linux
mypy pass. Independent review checks actual state, scores, costs, source reuse and
the original R2.5 acceptance criteria. Raw pilot `full_R25_complete=false` is
conservative pre-review self-certification metadata; canonical implementation
acceptance is recorded in the plans and independent review.

Torch distillation, trajectory replay, teacher-helper continuation, durable
recovery, confirmation/generalization and repository-wide release remain
unfinished or unsupported. Existing native Torch and ordinary stream continuation
remain unchanged. This is proposed probability distillation, not DER/DER++.

See [usage](../slow-teacher-replay.md),
[evidence](../../artifacts/runs/r25-slow-teacher-20261016/RESULTS.md),
[independent review](../../artifacts/runs/r25-slow-teacher-20261016/REVIEW.md), and
[full paired values](../../artifacts/runs/r25-slow-teacher-20261016/summary-corrected.json).
