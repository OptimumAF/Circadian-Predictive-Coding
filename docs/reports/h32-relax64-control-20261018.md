# H32 fixed64 PC sensitivity — 2026-10-18

The existing PC/neutral fixed64 control now supports explicit H32 on the original joint TRAIN stream and isolated F16/A16 probes.605focused scripted checks, scoped types/lint/format and independent SOURCE/ACTUAL review pass. A two-update PC-neutral parity check passes; one full **PC** case ran, not a full neutral study.

## Source128 TRAIN observations

| Context/cue | S64 CE | Original S2 CE | S64 correct | S2 correct |
|---|---:|---:|---:|---:|
| A-blue | 0.750268 | 1.110022 | 7/9 | 5/9 |
| A-red | 0.721030 | 1.130986 | 7/9 | 5/9 |
| B-blue | 0.716001 | 1.084554 | 7/9 | 5/9 |
| B-red | 0.766770 | 1.155461 | 7/9 | 5/9 |
| C_train-blue | 0.724026 | 1.138437 | 7/9 | 5/9 |
| C_train-red | 0.748958 | 1.108787 | 7/9 | 5/9 |

Both have zero exact TRAIN sequences. Source0 probe endpoints remain5/9each; all curves, including source32/64/96 and every probe checkpoint, are preserved.

## Source128 isolated adaptation

| Point | F blue/red correct | Old A blue/red correct |
|---|---:|---:|
| Before F | 6/9,6/9 | 7/9,7/9 |
| After F16 | 7/9,6/9 | 7/9,7/9 |
| After A16 return | 6/9,6/9 | 7/9,7/9 |

Original S2PC stays5/9percue at these points. S64 improves partial TRAIN fit, yet loses one F-blue correct action after A returns. Old A was never fully acquired; its unchanged partial accuracy does not establish useful acquired retention/recovery. CE worsens on A after F and improves after A returns. No gameplay, convergence, causal plateau or circadian advantage claim.

Public work192updates7488rows/P94/846/0SGD/10teachers90physics; parity adds2updates108rows/P2/108. Same original F16/A16 budgets, full source/input/RNG/policy guards. Body32.5319341s/outer35.8325574s/peak355336192B. Native authorities bothSPENT; no retry. S64 costs479232 nominal latent-row iterations,32times S2, plus process/copy/validation overhead. No equal-compute or speed claim.

All218signedmetric contrasts independently recomputed. TRAIN exactness observed; original compact probe scores contain CE/accuracy only. Two failed savedreader attempts assumed the wrong keys/extra metric; preserved and corrected to actualschema, with no native rerun or data/metric change. Original H8S64 and S2 negatives stay.

[Full commands, failures, costs and skipped checks](../../artifacts/runs/h32-relax64-control-20261018/RESULTS.md) · [Independent review](../../artifacts/runs/h32-relax64-control-20261018/REVIEW.md)

Next complete the matched full H32 neutral S64 observation through the existing route under a separately frozen local protocol, reusing accepted references before more learner changes.
