# P6.3 combined/full-minus-one development scoring contract

## Frozen source and global train gate

Protocol: `continual_combined_factor_outer_development_v1`. This scored
continuation preserves the complete [c7 train-only gate](p63-combined-factor-preflight.md):
all 17 cells on development seeds **263/269/271**, initialization, arrived
source/role geometry, 12+12 wake updates, existing component switches and
inner guards, full-controlled replay, memory supply, width bounds and costs.
No threshold, baseline, metric, seed or setting may change from a score.
Inactive default full/removal splits remain valid outcomes.

Bind the saved c7 result at
`artifacts/runs/p63-combined-factor-preflight/combined-factor-preflight.result.json`,
byte SHA-256
`79a7d7e09f0ada01ff72d6e266ddee7576aca52316a0f9a8e2ab3dd402251d13`.
Its independent repeat is byte-identical. Before source construction,
verify the complete request/result/audit bundle, finite JSON, exact bytes,
manifest/source/adapter identities, all 51 cells/648 decisions and actual
work/wall/RSS facts. A failed/incomplete/changed bundle prevents execution.

Reuse the frozen c7 model/wake/guard/replay/fact helpers. Deep-copy every
after-A model only after all A decisions, before B source construction.
Finish every seed/arm and validate the entire c7 fact object independently.
Require **exact complete all-seed equality** with the saved reference before
the first outer input or label access. Then check every after-A/after-B
checkpoint's parameter hash and width, and each circadian checkpoint's
complete snapshot fingerprint against its saved epoch facts, globally.
A late-seed parameter/RNG/chemistry/clock/copy mismatch must block even the
first scorer. Check these identities again after scoring. No retraining
or score-dependent stopping is permitted. Final sources stay sealed; all
confirmation seeds **277/281/283/293/307/311/313/317/331/337** and earlier
reservations remain unused.

## Outcomes and every named contrast

After the gate, evaluate each after-A copy on A outer selection
(`A_after_A`), and each after-B model on A/B outer selection
(`A_after_B`, `B_after_B`). Keep all 51 cells: **153 outer evaluations,
3,060 evaluated examples** (51 × (24+24+12)). No metric affects training,
guarding, inclusion or configuration. Reuse `phase6_two_task_accuracy_v1`:
primary final mean `(A_after_B+B_after_B)/2` and signed A forgetting
`A_after_A-A_after_B`, with optional zero-safe A retention ratio. Publish
all three accuracies, both primary metrics and every cost/capacity fact.
Ordinary-PC/neutral outcomes must be exact in off and full-replay states.

Freeze these **22 ordered left-minus-right pairs per seed, 66 total**:

| Group | Left | Right (each named cell, in order) |
|---|---|---|
| Seven removals | `full` | `minus_replay`, `minus_gating`, `minus_structure`, `minus_schedule`, `minus_difficulty`, `minus_homeostasis`, `minus_reset` |
| Nine references | `full` | `backprop_off`, `pc_off`, `neutral_off`, `backprop_full_replay`, `pc_full_replay`, `neutral_full_replay`, `periodic_structure_only`, `backprop_14_off`, `pc_14_off` |
| Replay work | `backprop_full_replay`, `pc_full_replay`, `neutral_full_replay` | Corresponding `backprop_off`, `pc_off`, `neutral_off` |
| Planned capacity | `backprop_14_off`, `pc_14_off` | Corresponding `backprop_off`, `pc_off` |
| Periodic structure | `periodic_structure_only` | `neutral_off` |

Publish every seed's differences in all three accuracies and both primary
metrics, with three-seed descriptive means and sample SD. Preserve duplicate
PC/neutral contrasts, null/negative rows, inactive components and rejected
proposals. No composite cost-adjusted winner, confirmatory significance
test or confirmation configuration is selected from this development run.

Full/minus-one contrasts share wake/source exposure but intentionally differ
in replay/guard cost, topology and per-update compute. Full-controlled
baseline replay receives exactly its accepted IDs/work with no independent
baseline guard; these consumers receive no full chemistry/topology/difficulty.
Difficulty removal changes importance weighting as well as wake scale.
No-schedule removal disables sleep and all downstream sleep effects. Wider
references are prospective width-14 ceilings, not observed-final-width
oracles. Structure-only retains c3's separate 0/1 thresholds and 7/9 bounds;
it does not isolate usage-aware versus random parent selection. Keep these
differences beside the contrasts; do not call the combined cells equal
compute/capacity or relabel the missing c9 controls.

## Budget, reproducibility and decision

Keep c7's **1,548 maximum executed updates / 1,600 hard cap**, including
rejected replay, and 1,224 wake updates. The exact reference records 1,530
executed updates, 144 guarded attempts/288 inner evaluations/5,184 examples,
126 own commits and 18 rollbacks. Scoring adds only the 153 declared outer
forward calls. Keep a **120-second** child wall limit and **256 MiB** observed
whole-worker RSS cap at 5 ms sampling. Include training, retained model
copies, independent validation, scoring and deterministic result
serialization in that RSS interval; final stdout transport and parent
artifact writing are outside it. Sampling may miss brief peaks. These are
whole-worker measurements, not per-arm resource attribution.

Pin the unchanged manifest, selected source map and scored adapter bytes
before any outer score, including scored tests. Keep exclusive request/
result/audit or failure artifacts in ignored local directories. Repeat in
a fresh bounded process and require identical deterministic result bytes.
Independently read back every cell, contrast, c7 fact, metric and artifact/
resource identity. Partial/failing runs remain unfinished. Wall/RSS belongs
in the audit, outside deterministic results. Publish all inactive, rejected,
null and negative outcomes without tuning settings/seeds to change them.

**Why this order:** the saved all-seed train gate witnesses the exact path
before a score is known. Small after-A copies avoid a second training pass
and permit complete checkpoint checks before exposing early-seed results.
Reuse c4 score types/arithmetic and c7 training instead of changing pinned
earlier protocols. This completes c8 development only when its evidence
passes. C9 growth controls, independent confirmation, final release and
the original P6.3c/P6.3 matrix remain unfinished.

## Implementation identities

Before any outer score (including scored tests), the unchanged c7 manifest
SHA-256 is `729bf9df8472752f51696299373555339208af7b5add1892d14154d4f21ad04a`.
The scored adapter adds these selected sources to c7's frozen 26-source map:

| Additional source | Byte SHA-256 |
|---|---|
| `src/app/continual_combined_factor_development.py` | `be4c0156c8921a9f0d4580e93695bd4464c1d6241a9bfa7c3fd531eb199994fa` |
| `src/app/continual_sleep_factor_development.py` | `e6a498c107d0071b822309907691e225b1a1ae828c93f8435c94ba0cfa94bf88` |
| `scripts/run_p63_combined_factor_preflight.py` | `9c3d640638b48d7a2edc9cc4585e205cdb3f2d86bd1c72a6e7a8438bd0f3d404` |

The combined selected **29-source** map digest is
`ee0e2c8cb154f9aa427c0b27680841b64a0dfe3c6a90f254159114ca4ce4d6b5`;
the scored adapter byte digest is
`edf68589d83033d0abdf39c737f78044938f84a97b49d86a25698edce42d8c40`.
Five train-only rejection/mismatch cases passed (one scored case deselected),
followed by repository Ruff/mypy (349 source files) and formatting before
these identities were frozen. The complete c7 reference bundle verifies,
including observed work/wall/RSS. The source map is selected, not a full
dependency-tree hash. No outer score has been read at this point.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_combined_factor_development --output-dir artifacts/runs/p63-combined-factor-development
.\.venv\Scripts\python.exe -m scripts.run_p63_combined_factor_development --output-dir artifacts/runs/p63-combined-factor-development-repeat
```

Both completed results repeat at SHA-256
`2e32c5d7b98b5798f45f5e9b7734088950eec1ef0ae5f19420a282321f3f8714`.
[Every score, contrast and cost row](p63-combined-factor-development-results.md)
is retained, including mixed/negative full-model results. C9 and independent
confirmation remain unfinished; no scored outcome changed the contract.
