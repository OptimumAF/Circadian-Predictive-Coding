# Coupled stationary/recurring attribution pilot — 2026-10-14

Protocol: `coupled_stationary_recurring_pilot_v1`. **Exploratory fixed-data pilot**,
not confirmation or a claim of general superiority. Protocol and selected source
identity were recorded before the single comparison invocation. No settings,
seeds, metrics or baselines were changed after reading outcomes.

## Outcome

All24 declared runs completed:3initializationseeds ×2streams ×4methods. The new
PC/neutral CPC control matches exactly in all six seed/stream pairs, including
metrics, energy diagnostics and training receipts. Gated CPC shows no demonstrated
advantage here. Its mean BCE is slightly worse on stationary A, new B, and returned
A; the old-A retention difference after B is tiny and its nominal interval spans
zero. This negative/inconclusive finding is retained without tuning.

Source/result links: [protocol and evaluation rules](../coupled-attribution.md),
[prospective protocol](../../artifacts/runs/coupled-attribution-20261014/PROTOCOL.md),
[exact source identity](../../artifacts/runs/coupled-attribution-20261014/SOURCE_VERSION.json),
[all seed outcomes and summaries](../../artifacts/runs/coupled-attribution-20261014/pilot.json).

## Common held-out outcomes

Cells are mean [nominal marginal95% t interval] over the three fixed model seeds,
conditional on the same synthetic training/held-out data. Lower BCE and higher
accuracy are better. Every method starts from exactly shared tensors and uses
the same learning rate, arrived rows, update count and target likelihood. BP
optimizes feedforward BCE; PC/CPC optimizes its distinct relaxed total energy.

| Stream/checkpoint/context | Method | BCE | Accuracy |
| --- | --- | --- | --- |
| stationary/18/A | bp | 0.549544 [0.464971, 0.634117] | 0.869792 [0.632611, 1.106972] |
| stationary/18/A | pc | 0.610987 [0.448223, 0.773750] | 0.822917 [0.445898, 1.199936] |
| stationary/18/A | neutral_cpc | 0.610987 [0.448223, 0.773750] | 0.822917 [0.445898, 1.199936] |
| stationary/18/A | gated_cpc | 0.612853 [0.445743, 0.779964] | 0.822917 [0.445898, 1.199936] |
| recurring/12/A | bp | 0.625983 [0.454868, 0.797097] | 0.833333 [0.182243, 1.484424] |
| recurring/12/A | pc | 0.643464 [0.421330, 0.865598] | 0.765625 [0.053025, 1.478225] |
| recurring/12/A | neutral_cpc | 0.643464 [0.421330, 0.865598] | 0.765625 [0.053025, 1.478225] |
| recurring/12/A | gated_cpc | 0.643443 [0.421267, 0.865619] | 0.765625 [0.053025, 1.478225] |
| recurring/12/B | bp | 0.720871 [0.586372, 0.855370] | 0.385417 [-0.282806, 1.053639] |
| recurring/12/B | pc | 0.751770 [0.592348, 0.911191] | 0.354167 [-0.238785, 0.947118] |
| recurring/12/B | neutral_cpc | 0.751770 [0.592348, 0.911191] | 0.354167 [-0.238785, 0.947118] |
| recurring/12/B | gated_cpc | 0.752371 [0.592650, 0.912092] | 0.354167 [-0.238785, 0.947118] |
| recurring/18/A | bp | 0.594653 [0.470323, 0.718982] | 0.864583 [0.512360, 1.216807] |
| recurring/18/A | pc | 0.630936 [0.433494, 0.828377] | 0.791667 [0.219406, 1.363927] |
| recurring/18/A | neutral_cpc | 0.630936 [0.433494, 0.828377] | 0.791667 [0.219406, 1.363927] |
| recurring/18/A | gated_cpc | 0.631922 [0.431960, 0.831885] | 0.786458 [0.192238, 1.380679] |

All four checkpoints (including initialization and phase1), both contexts, both
metrics and every raw seed observation are in the JSON. The displayed points
were fixed by the protocol: stationary finish, old/new contexts after B, and A
on recurrence. A after B measures retention; B measures the new context.
These untruncated t intervals can extend beyond `[0,1]` for accuracy; their wide
endpoints describe the small sample's imprecision, not valid probability ranges.

## Paired BCE differences

Signs are left minus right. Positive BCE differences favor the right method.
No multiple-comparison or significance claim is made.

| Stream/checkpoint/context | PC minus BP | Gated CPC minus neutral CPC |
| --- | --- | --- |
| stationary/18/A | 0.061443 [-0.022336, 0.145221] | 0.001867 [-0.002717, 0.006450] |
| recurring/12/A | 0.017481 [-0.034279, 0.069242] | -0.000021 [-0.000216, 0.000174] |
| recurring/12/B | 0.030899 [0.005535, 0.056262] | 0.000601 [0.000142, 0.001061] |
| recurring/18/A | 0.036283 [-0.038826, 0.111393] | 0.000987 [-0.001676, 0.003649] |

Neutral CPC minus PC is exactly zero in every seed at every point/context/metric;
the summary marks its interval unavailable due to zero observed variance instead
of interpreting a zero-width interval as population certainty.

## Work, limits and interpretation

Observed orchestration wall time was 0.352250s, with sampled absolute
process RSS peak 39,915,520 bytes. Output was
540,247 bytes; all declared resource gates passed. These are
single-process resource observations, not comparative timing measurements.

Total completed work was432updates,3456training row exposures,1296latentsteps,
2160nominal completed gradient evaluations and192held-out forward calls. BP has
0latentsteps and2gradient evaluations perupdate; PC/CPC has4latentsteps and6coupled
energy/gradient evaluations. These counts include candidate gradient evaluations but exclude copies, chemistry,
schema-check overhead and other costs; they are not FLOPs or matched compute. No replay/sleep,
structure/promotion, GPU, download or remote service was used.

The pilot fits the original first-pilot caps:512updates,60seconds,256MiB observed
RSS,CPUthreads1,32MiB output. Safe-boundary checks do not preempt nativecalls or
measure every transient allocation. All failure-vector behavior was tested with
injected failures; this actual invocation had none.

Three model seeds on one fixed simple dataset cannot test interval normality,
provide simultaneous coverage, establish dataset-level generalization, or confirm
that small gating differences are reliable. The short18update schedule, ordinary
fan-in initialization and unequal native work are explicit study conditions.
Published width/depth scaling is still unfinished. No biological or mechanism-wide
claim follows. No extra seed/run was selected or launched after these results.

Reproduce with `python -m scripts.run_coupled_attribution` in the existing NumPy
environment. The saved source/protocol identity and exact initial tensors identify
this invocation; a repeat is additional budgeted work, not a new independent seed.
A separate full pilot repeat was not run in this one-invocation scope.

## Next action

Retain this pilot as the small R1.7 attribution result. Improve the next actual
missing core prerequisite under a new prospective scope (R1.4 scaling or R1.5
bounded inference), preserving these baseline settings and negative outcomes.
Any follow-up comparative/confirmation study needs its own protocol and budget.
