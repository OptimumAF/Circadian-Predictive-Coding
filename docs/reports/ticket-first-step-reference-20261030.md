# First failing BP step: saved-data decomposition - #335 / R2.1-BP.STEP.REF

Previous334 made progress by restoring a mistakenly overwritten legacy API and
preserving the new norm helper separately. That correction remains intact.
This turn executes the next concrete scientific diagnosis, not another native
source reconstruction, checkpoint qualification or F/return attempt.

## Observed mechanism and limits

ONE NumPyfloat32 analytic step matched both saved323 native training losses
EXACTLY at their recorded precision:7.780185222625732 before the first update,
2801.909423828125 on the next minibatch after that update. Two literal gradient/
momentum correctness oracles passed before scientific data arithmetic. Frozen
absolute parity tolerances were .01/.05; neither tolerance was changed or used
to select an outcome. These are unclipped native-CE anchors, not the clipped
probability-CE plateau discussed in323.

Fixed next-minibatch reference comparisons, all using the SAME first-step delta:

| Working-copy parameter change | Unclipped CE | Saturated hidden scalars | Predicted classes | Correct rows |
| --- | ---: | ---: | --- | ---: |
| None | 0.31535154581069946 | 0/9 | 3,1,0 | 2/3 |
| Full SGD step | 2801.909423828125 | 6/9 | 3,3,3 | 1/3 |
| Hidden parameters only | 2801.895263671875 | 6/9 | 3,3,3 | 1/3 |
| Output parameters only | 0.3133639991283417 | 0/9 | 3,1,0 | 2/3 |

Saturation threshold abs(h)>=.999 was frozen in advance. Classes3/1/0 mean
open/defer/commit; next-batch targets are3/2/0 (open/inspect/commit). Output-only
avoids this reference explosion but DOES NOT correct the remaining READY
mistake or demonstrate task acquisition. No native counterfactual tasks ran.

| Parameter group | Initial L2 | New gradient L2 | Prior momentum L2 | After momentum L2 | Actual reference delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hidden | 0.27566720280282253 | 522.5372556852217 | 0.33419998299231357 | 522.5524347812883 | 15.676572473308955 |
| Output | 38222.25981816013 | 0.47181420443189687 | 2.048298084459598 | 1.8494674992704616 | 0.05593515146442672 |

Within the reference calculation, the hidden-block intervention accounts for
nearly all the next-batch loss jump. Error propagation through the fitted output
layer yields a hidden gradient much larger than the original hidden parameter
scale. Native code zeroes gradients before backward, so stale gradient buffers
are not reused. Prior momentum is retained; its norm is small relative to the
new hidden gradient. This is not a causal ablation of momentum or proof that
the absolute readout norm alone causes failure.

Both native scalar loss anchors agree, supporting the reference's relevance;
they do not prove equality of every native tensor. Block counterfactuals and
saturation counts are reference-only. No native output-only execution, reliable
acquisition/retention/recovery, generalization, matched-compute result or CPC
advantage follows.323's native negative old4->0->0/fresh0->0->0 and all failed
attempts/costs remain unchanged;333 capacity and328 LS findings/caveats persist.

## Source and procedure

No production code changed. New stage contains reference.py/run.py/launch.py
and PROTOCOL.md. Nonexistence was checked before creation, preserving existing
modules and avoiding334's collision. Sole scientific data file is accepted323
attempt2 native.json1,594,263B, frozen SHA256
`9dc6ad9bcf0eeab03b7b53b2b91e0d1dc9d1827ef7b246065279778a5ce0315b`.
One direct input digest binds immutable evidence, not a new hashing/admission
framework. Its flags, fixed policy/shape/counters, original SGD parameter group
and output24 derivation were checked; original bytes remained unchanged.

Copy source checkpoint12/72 float32 parameter/momentum data, replace only local
output arrays with original CE coefficients, and use existing build_head_cohorts
for first two fixed private probe_train packets: v1parcel then v2password.
Final data absent. No probe_test targets enter arithmetic or selection; factory
evaluation copies are counted as prep, and only their feature lengths are read.
No private labels become benchmark/source/replay/promotion inputs.

Reference meanCE uses stable log-sum-exp and analytic affine/tanh/softmax
derivatives, old output weights in the hidden derivative, original lr.03 and
momentum.9. Prior gradients are ignored, matching zero_grad. Full-step parity
gates precede both same-delta block ablations. No second scientific gradient,
precision fallback, alternate seed/dose, clipping/rescaling, metric or task
selection. Every branch/result was retained. Saved arrays are diagnostic data,
never a reconstructed runtime owner or a deployable checkpoint.

## Execution, checks, costs and failures

All commands physical `D:/Dev/Repositories/Circadian`, PowerShell TEMP/TMP
`D:/Dev/_Temp`; WSL cwd `/mnt/d/Dev/Repositories/Circadian`, TEMP/TMP/TMPDIR
`/mnt/d/Dev/_Temp`. Existing LinuxPython3.12.3/NumPy1.26.4, threads1. Torch,
Torchvision and SciPy imports are refused; recorded forbidden-import list empty.

One execution, EXIT0; do not repeat after exclusive markers:

```text
.venv/Scripts/python.exe -B artifacts/runs/ticket-first-step-reference-20261030/launch.py
```

Exact WSL argv/OS timeout/exit in shell.json; raw actual.log and actual.json.
Process0.7915907119999996s/51,957,760B RSS; host3.308588999992935s. Frozen30s/
256MiB process,60s host and1MiB stage held; projected stage48,839B at result
write. Marker precedes imports; no retry or reopened scientific allowance.

All work including literal oracles:7referenceforwards17rows,3analyticgradient
calculations5rows,681nominal forward matrix products/199gradient matrix products/
66scalar SGD slots. Bias/softmax/tanh/scalar reductions additional; these are
not complete FLOPs or comparable native-training counts. Factory prep2teachers,
24TRAINcases/24sandboxdispatches/30encodings and78row copies; first6private rows
used for diagnosis. Native models/restores/SGD/optimizer/autograd/random model
draws/LP/SVD/LS/PC-latent/robotphysics/final/promotion/remote all0. Runtime and
memory are diagnostic costs, not evidence of a faster learning algorithm.

Scoped static commands cover exactly reference.py/run.py/launch.py:

```text
.venv/Scripts/python.exe -B -m ruff check <three stage files>
.venv/Scripts/python.exe -B -m ruff format --check <three stage files>
.venv/Scripts/python.exe -B -m mypy --platform linux --explicit-package-bases --follow-imports=silent --cache-dir D:/Dev/_Temp/circadian335-mypy <three stage files>
```

Expanded argv/outcomes in statics-first.json/statics-final.json. Ruff/format
PASS; initial19mypy report-dictionary inference errors fixed before math with
explicit typing; final Linux-target mypyPASS. All static attempts3.301058899931377s
under60s. Linux-target type check ran on Windows; no Windows numerical repeat.
No native/math fixture cohort beyond the two declared analytic oracles.

Broad suite/clean clone/wheel/build/native source12/F-return/full-family
benchmarks/browser/GPU/remoteCI/release checks/sweeps skipped: no production code
change or new native behavior. Existing accepted controls reused in their
original scopes. No installs, deletions, global settings changes, commits/pushes,
new source loader or unrelated resets. Caches Dtemporary, no cache in stage.

## Review and next learner repair

Lead self-reviewed equations, frozen cohort/roles, full before-gradient use of
Who, momentum/zero_grad semantics, source bytes, gate ordering, both oracles,
all comparison arrays and exact counters. Prelaunch independent ACCEPT2reads
0.3424583s; final actual-source/result/interpretation ACCEPT1additionalread,
3total0.5185846s. No numerical execution or independent replication. Finite
exception ended; no native counterfactual acceptance claim.

The existing native learner/private probe has no output-only parameter-block
mode (source inspected). Next precise implementation: add an explicit opt-in
output-only SGD update to the existing disposable BP probe, preserving hidden
parameters and their momentum while applying the original output SGD and
retaining computed gradient diagnostics. Qualify bounded tiny fixtures first,
including default full-update identity and state/budget/failure behavior. Do not
change source training, reset output momentum, tune the frozen323 case or claim
acquisition before observation. This is a conventional learner repair justified
by the diagnosed hidden-step instability, not a new circadian feature.
Full R2/R4/R5/R7/matched science/release/repository goal remain open.

Closeout335: scoped diagnosis closed/completed; canonical6534926621, tracker18
6534950044,parent21 6534950364,parent23 6534950651. Final retained stage58078B
<1MiB. Both plans/log updated; no production changes. Full repository goal stays
active and scientific/release parents remain open.
