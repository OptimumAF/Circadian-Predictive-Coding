# P6.3 matched schedule development scoring contract

## Frozen source and global train gate

Protocol: `continual_schedule_factor_outer_development_v1`. This is the
scored continuation of the [c5 train-only gate](p63-schedule-factor-preflight.md).
Keep all eleven model cells on each of seeds **79/83/89**, their initialization,
source/role geometry, 12+12 wake updates, current periodic/adaptive/no-sleep
settings, shared replay supply, inner guard, fixed width 8/12 and work caps.
No c5 threshold, seed, baseline, source or metric changes are authorized.
An inactive adaptive policy remains a valid development outcome.

Bind the complete c5 result at
`artifacts/runs/p63-schedule-factor-preflight/schedule-factor-preflight.result.json`,
byte SHA-256
`87931c5fc3fad5bf50d5b18d07901d04b8a8e827ab4fce4429c0f1644c813c41`.
Its independent repeat is byte-identical. Before source construction,
verify its exclusive request/result/audit bundle, exact bytes, frozen
manifest/source/adapter identities, finite JSON, all cells and work/resource
facts. Failure or an incomplete bundle prevents training and scoring.

Train each seed using the existing c5 model, wake, decision, replay and fact
helpers. Preserve after-A models by deep copy after all A decisions and before
B source arrival. Finish all 33 A→B cells and independently validate the entire
c5 fact object. Require exact equality with the saved reference **globally
before the first outer input or label access**. A mismatch on any seed,
checkpoint, opportunity, clock, guard, row ID, cost or role fails the entire
run with zero outer scores. No second training pass is allowed. Final-source
inputs/labels and confirmation seeds **199/211/223/227/229/233/239/241/251/257**
remain unopened.

## Outcomes and every policy contrast

Only after the gate, evaluate each model's after-A copy on A outer selection
(`A_after_A`), and its after-B model on A/B outer selection (`A_after_B`,
`B_after_B`). All eleven cells receive these three evaluations: **99 outer
evaluations/1,980 examples** (33 × (24+24+12)). No score affects training,
guarding, settings, stopping or inclusion. Verify checkpoint parameter hashes
and exact ordinary-PC/neutral outcome parity for each policy.

Use the existing `phase6_two_task_accuracy_v1` contract: primary
`final_mean_task_accuracy = (A_after_B+B_after_B)/2` and
`signed_forgetting_A = A_after_A-A_after_B`. Publish all three accuracies,
both primary metrics and optional zero-safe A retention ratio. Output
verification rederives the metrics and requires finite values.

For **each** of backprop, ordinary PC and neutral circadian, prespecify the
three **left-minus-right** pairs:

- `periodic − no_sleep`
- `adaptive − no_sleep`
- `adaptive − periodic`

This is nine pairs per seed, 27 total. Publish each seed's differences in all
three accuracies and both primary metrics, plus descriptive three-seed means
and sample SD. Retain duplicate PC/neutral and zero/inactive comparisons;
neither is grounds for omission. Publish both width-12 references alongside
all width-eight cells. Do not define a cost-adjusted winner or a confirmatory
significance test from this small development factor.

The complete c5 facts stay beside scores, including each model's wake,
applied/rejected executed replay, guard, retained-array scope, fixed width
and parameter counts. Policy contrasts have equal wake work and capacity,
but **intentionally different replay/guard costs**. Wider references have
more parameters and per-update compute. The shared neutral/PC guard also
controls backprop replay; it is not a separate backprop guard. Report these
limits when interpreting contrasts.

## Budget, reproducibility and decision

The unchanged c5 ceiling is 1,014 executed optimizer updates under hard cap
1,100 (792 wake plus bounded replay, including rejected executions), at most
63 guarded attempts/126 inner evaluations. Scoring adds only the 99 declared
outer forward evaluations. Keep the 120-second child wall limit and observed
whole-worker RSS cap 256 MiB, sampled every 5 ms. Sampling can miss a shorter
peak; wall/RSS is not attributed to individual arms.

Pin selected source hashes, manifest and scored adapter bytes before the first
outer score, including scored tests. Write exclusive request/result/audit or
failure sidecars to ignored local directories. Independently repeat in a fresh
process and require identical deterministic result bytes. Read back every
cell, contrast, c5 fact and artifact/resource identity; partial runs do not
qualify. Exclude wall/RSS from the deterministic result and label their audit
scope. Preserve every null/negative outcome; no rerun may select favorable
seeds or settings.

**Why this order:** the completed unscored reference witnesses the exact train
path before any score is known. Holding small after-A copies permits the full
global gate without retraining or exposing early-seed scores. Reuse the prior
c4 scoring arithmetic and dataclasses rather than introduce different metrics.

This completes only the c6 development factor when its gates pass. No isolated
schedule result selects a configuration for confirmation. The combined full
model, full-minus-one and scheduled/random growth controls remain open under
P6.3c; a separate prospective protocol must bind those cells and confirmation
roles before they run. Independent confirmation and final release remain open.

## Implementation identities

Before any outer score (including scored tests), the unchanged c5 manifest
digest is `f8b6d60209516c58bc54e729f659c9fdd37ea1b869652b1cac548a71270ea42a`.
Its 20-source map remains fixed. The scored adapter adds:

| Additional source | Byte SHA-256 |
|---|---|
| `src/app/continual_schedule_factor_development.py` | `f2d49461a05defd18b53e5e28f822dcf099405ab6ff716972d41bc765e60c48d` |
| `src/app/continual_sleep_factor_development.py` | `e6a498c107d0071b822309907691e225b1a1ae828c93f8435c94ba0cfa94bf88` |
| `scripts/run_p63_schedule_factor_preflight.py` | `4490f9201b565d3283434da786799cc3d9ac8242218cfc8f9b881de9011b7fbe` |

The combined selected 23-source map digest is
`3e7d8b8086ba881527f21d513015deb30743884b388b37f2eda23a79b4e29c69`;
the scored adapter byte digest is
`94fc133317c9b0954cee47b5d5a431285bcf30da3effc9b457153b6bfe6000c1`.
Four train-only rejection/mismatch tests and repository Ruff/mypy (339 files)
and four-file format gates passed before these identities were frozen.
The source map is selected, not a full dependency-tree hash. The complete
c5 bundle, including actual work and observed resource limits, also verifies.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_schedule_factor_development --output-dir artifacts/runs/p63-schedule-factor-development
.\.venv\Scripts\python.exe -m scripts.run_p63_schedule_factor_development --output-dir artifacts/runs/p63-schedule-factor-development-repeat
```
