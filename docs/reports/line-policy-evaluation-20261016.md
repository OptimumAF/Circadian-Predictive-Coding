# Live LineNavigation evaluation — 2026-10-16

Original R4.1 contract independently accepted. Training comes from the unchanged
accepted R2.4 record; only frozen prediction copies and actual environment episodes
execute here. This is a tiny seed17 implementation case, not a comparative benefit
experiment. All 64 cached source scores matched before any live episode.

## Every checkpoint and both live outcomes

S = success, return1, two legal steps. I = illegal action, return0, one legal step
followed by an unapplied invalid decision. Every episode made two policy calls;
max4 was frozen. No action repair, clipping, retry or favorable checkpoint choice.

| Method | Replay | Wake updates | A | B |
| --- | --- | ---: | --- | --- |
| bp | False | 0 | S | S |
| bp | False | 2 | S | I |
| bp | False | 4 | S | I |
| bp | False | 6 | S | I |
| pc | False | 0 | S | S |
| pc | False | 2 | S | I |
| pc | False | 4 | S | S |
| pc | False | 6 | S | I |
| neutral_cpc | False | 0 | S | S |
| neutral_cpc | False | 2 | S | I |
| neutral_cpc | False | 4 | S | S |
| neutral_cpc | False | 6 | S | I |
| gated_cpc | False | 0 | S | S |
| gated_cpc | False | 2 | S | I |
| gated_cpc | False | 4 | S | S |
| gated_cpc | False | 6 | S | I |
| bp | True | 0 | S | S |
| bp | True | 2 | S | I |
| bp | True | 4 | S | I |
| bp | True | 6 | S | I |
| pc | True | 0 | S | S |
| pc | True | 2 | S | I |
| pc | True | 4 | I | S |
| pc | True | 6 | S | I |
| neutral_cpc | True | 0 | S | S |
| neutral_cpc | True | 2 | S | I |
| neutral_cpc | True | 4 | I | S |
| neutral_cpc | True | 6 | S | I |
| gated_cpc | True | 0 | S | S |
| gated_cpc | True | 2 | S | I |
| gated_cpc | True | 4 | I | S |
| gated_cpc | True | 6 | S | I |

All16 initial episodes succeed: this case cannot demonstrate acquisition from
failure. All8 final A episodes succeed and all8 final B episodes fail. At wake4,
A succeeds in5 arms and B in6. PC, neutral CPC and active CPC have identical
episode outcomes within each replay mode. Aggregate43/64 success does not establish
robust retention, held-out-layout generalization, a causal replay effect or a
circadian advantage. Every negative probability/transition remains in pilot.json.

## Observed cost and isolation

32 frozen prediction copies;64 import-score forwards plus128 live forwards =192
native forwards,64 live episodes,107 actual legal environment steps. No training,
optimizer update, replay, probe, promotion or policy feedback. Report126,133B;
peak process RSS312,217,600B; pilot7.6162161s, worker7.6536362s,
parent8.6002303s. Caps:60s body/90s parent,768MiB RSS/2MiB JSON,
320 forwards/64 episodes/256 legal steps. All pass; one RSS sampler joined.
RSS includes process/native runtime, not just tensors. Timing includes import,
validation, copies and state checks; this is not an overhead or speed comparison.

The imported weights reproduce original cached predictions but do not reconstruct
historical optimizer, gradients, adaptive clocks, RNG, deployed actor authority or
publication. Original training budgets and receipts are not renewed. `eval:` actor
versions identify these local copies. Final scoring data never enter training,
promotion, tuning or probe selection. Runtime full_R41_complete=false remains
unchanged as pre-review metadata; independent acceptance is separate.

## Evidence and next dependency

[Raw output](../../artifacts/runs/r41-line-rollouts-20261016/pilot.json),
[all outcomes as JSON](../../artifacts/runs/r41-line-rollouts-20261016/summary.json),
[independent criterion map](../../artifacts/runs/r41-line-rollouts-20261016/REVIEW.md),
[commands and failures](../../artifacts/runs/r41-line-rollouts-20261016/RESULTS.md),
[usage](../line-policy-evaluation.md).

Full R4.2 MiniGrid remains unchecked. Actual existing feature learner, attribution
targets and demonstration codec admit only two actions. Next implement an explicit
categorical-action extension through those existing ports, beginning with a bounded
three-action native train/predict/owned-snapshot roundtrip and unchanged binary
regressions. Then build partial-observation key/door demonstrations; preserve the
original A→B→C→A, hidden-cue, recurrent-policy and held-out-layout requirements.
No additional capture/provenance increment or tuning of this known tiny case.
