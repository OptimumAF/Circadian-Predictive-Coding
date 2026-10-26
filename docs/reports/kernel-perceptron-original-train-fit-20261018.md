# Original TRAIN polynomial-kernel fit — finite full-fit negative

## Question and result

Can the fixed degree-2 kernel learner fit all original 54 TRAIN actions in 128
ordered passes? The single prescribed case completed and qualified; its full-fit
witness is **false**. Accuracy rose from 6/54 to 47/54; exact traces from 0/6 to
3/6; mean CE fell from 1.9459101490553137 to 0.8894917420328361. Every requested
pass ran; only fixed 0/128 readouts were scored. No retry or kernel/cap/dose/seed/
endpoint/metric selection occurred.

| TRAIN episode | Correct / 9 at 128 | CE at 128 | Exact |
| --- | ---: | ---: | ---: |
| A-blue | 9 | 0.403496785591 | 1 |
| A-red | 6 | 1.603641747629 | 0 |
| B-blue | 7 | 1.220063214909 | 0 |
| B-red | 9 | 0.426427831122 | 1 |
| C_train-blue | 7 | 1.268555770326 | 0 |
| C_train-red | 9 | 0.414765102620 | 1 |

Seven final errors (zero-based step, predicted → target):

- A-red, step 2: 0 → 1.
- A-red, step 3: 0 → 3.
- A-red, step 5: 0 → 2.
- B-blue, step 1: 0 → 1.
- B-blue, step 2: 0 → 3.
- C_train-blue, step 2: 0 → 1.
- C_train-blue, step 3: 0 → 3.

The accepted canonical TRAIN bounds report 54 distinct conflict-free inputs,
accuracy ceiling 1 and CE floor 0. This helper evidence is reused, not independently
reconstructed from unpersisted native keys. A finite failed fit does not prove
kernel inseparability or general capacity failure. The simple linear control's
52/54 and all original BP/PC/neutral/active results remain unchanged. Algorithms,
capacities, objectives and computation differ; there is no matched-compute or
circadian advantage claim. No useful neural acquisition, acquired retention/
recovery, unseen-layout generalization, RL or deployment claim follows.

## Execution and costs

Existing `scripts.run_minigrid_linear_fit_control --kernel-perceptron`, original
saved six TRAIN148 episodes, raw296 public histories and original invariant
projection. Empty dictionary, fixed `(1+dot(x,y)/296)^2`, unit supplied-order
mistakes, first-argmax, cap128, no RNG/optimizer/averaging/eviction. The original
delay-one TRAIN stream and source/actor/budget/witness guards remain in force.
Final/probe records never enter learning or selection. All 108 raw probability
rows, 128 receipts and per-pass mistakes are retained.

| Cost | Observed value |
| --- | ---: |
| Native passes / training rows | 128 / 6912 |
| Row mistake updates | 1810 |
| Committed native training kernel pairs | 313371 |
| Retained prototypes / cap | 48 / 128 |
| Prototype / coefficient array-equivalent bytes | 113664 / 2688 |
| Prediction calls / rows | 12 / 108 |
| Derived attempted / returned prediction pairs | 2592 / 2592 |
| Stream input-copy logical bytes | 16422912 |
| Native body / supervised actual seconds | 6.0504181 / 6.7482593 |
| Native parent seconds | 7.0110382 |
| External process RSS bytes | 79458304 |

Native pairs are committed counters; prediction pairs derive from checkpoint
dictionary counts 0/48 times rows. Per-pair dot/accumulation work is source-derived,
not separately instrumented. Python objects, tuples, projection, admission,
validation and copies add memory/computation. Array-equivalent payload is not
resident memory. At most two fit checkpoints, bounded receipts/dictionary, no
full Gram or unlimited success chain. Zero new Torch models/updates, backwards,
optimizer/replay/probe/final/teacher/physics work; original controls were reused.

SOURCE/PROTOCOL and ACTUAL independent GO cover this case. Saved-only readback
recomputes all probability/argmax/clipped-CE/exact/context/gap reductions, original
TRAIN IDs/labels, receipt/diagnostic totals, bounds and stdout/output equality.
No unpersisted dictionary/alpha reconstruction is claimed. All five commands
exit 0 with no resource problem. Unchanged 203 engineering checks and scoped
statics are reused; no repository-wide readiness claim. Science
7.4573552000/120 and engineering 751.8506183000/1000 seconds CLOSED.
[Frozen protocol](../../artifacts/runs/kernel-perceptron-real-fit-20261018/PROTOCOL.md),
[exact commands/receipts](../../artifacts/runs/kernel-perceptron-real-fit-20261018/RESULTS.md),
[independent review](../../artifacts/runs/kernel-perceptron-real-fit-20261018/REVIEW.md).

## Next implementation

The native Torch policy already admits widths through 32, while constructive
TRAIN composition hardcodes width 8. A fixed width-32 quartet tests neural
capacity through existing algorithms/ports and keeps original baselines frozen.
The kernel branch stops at this negative result; no automatic dose extension or
new classifier/observer/audit infrastructure is required. Original R2.1 diagnostics
and isolated fresh-task probes remain accepted; broader acquired retention and
recovery still require a neural acquisition witness.

Add one explicit fixed width-32 TRAIN-fit recipe for the existing BP, PC, neutral CPC and active CPC learners through minigrid_constructive_fit and the saved-source CLI. Preserve the original width-8/default recipes; verify complete owned state, method/width admission and costs before freezing any new comparative fit. No width, seed, dose or endpoint sweep.
