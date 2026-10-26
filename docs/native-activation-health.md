# Native activation health

The existing MiniGrid public-history diagnostic can also record native hidden
activations from known TRAIN packets:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --native-activation-health --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --native-activation-health
```

The flag is valid only for this explicit condition. Original no-flag recipes and
all negative findings remain preserved. The descriptor constructs no native model
or environment; execution follows the separately declared local protocol.

## Behavior

`read_native_activations(snapshot, features)` in the new adapter validates and owns
complete native snapshots. It computes actual float32 CPU tanh and returns an
owned float64 matrix. Feedforward values are inference priors shared by BP, PC,
neutral CPC and active CPC; labels do not relax them. Recurrent rows use the
existing native forward in order, starting from zero for every complete episode.
The extra recurrent output projections are computed and discarded, and charged.
Nonfinite affine values fail before tanh can conceal overflow as finite saturation.

The existing `diagnose_activations` reducer retains its definitions: dead units
have mean absolute activity <=1e-3, active is the complement, saturated entries
have absolute activity >=.99. Effective rank uses the entropy of centered singular
values and the original numerical cutoff. These are descriptive properties, not
proof of lost plasticity or causal reasons for failed learning.

Only six known TRAIN packets at each owned0/32/64/96/128 checkpoint enter health.
No probe or final input, label, reward, cue metadata or next frame is supplied.
Source/actor/checkpoint/optimizer/gradient/counter/RNG/input state stays unchanged.
Successful raw scores remain recorded if an optional health read fails; returned
hidden rows remain recorded if SVD fails. Partial status and paid work are retained.

## Costs and structure

The existing800update/7200row stream-plus-probe case adds150native activation
calls/1350rows,120feedforward hidden affine batches,270recurrent hidden steps and
discarded output projections, and150SVD reductions. Raw hidden arrays total86400
float64-equivalent bytes (576B each), plus record/native/copy overhead in time/RSS.
These calls are separate from230original score forwards/2070rows. No extra SGD.
Original180body/210parent seconds,768MiB RSS,32MiB output and CPU1 limits apply.

```text
src/adapters/torch_activation_diagnostics.py native validated hidden-state read
src/adapters/torch_recurrent_learner.py      optional finite preactivation check
src/core/learnability_diagnostics.py         existing unchanged health metrics
src/adapters/minigrid_learnability.py        existing TRAIN checkpoint composition
scripts/run_coupled_attribution.py          explicit CLI flag
tests/test_torch_activation_diagnostics.py  native precision/state/ownership tests
tests/test_minigrid_activation_route.py     role/CLI/cost/failure integration tests
```

From the repository root, scoped checks:

```powershell
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/r21-native-activation-20261016/run_checks.py statics
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/r21-native-activation-20261016/run_checks.py focused
```

These run affected Ruff/format/mypy and focused tests. Repository/release gates
remain separate. See [prospective limits](../artifacts/runs/r21-native-activation-20261016/SCOPE.md)
and ADR-0351. Extend through existing snapshot/feature ports and the unchanged
metric reducer, with isolated readout costs and a new frozen protocol before
further experiments. Do not infer importance/age for models without those fields.

## Observed local integration

The sole case completed100.00parent seconds with439726080B peak sampled RSS.
All preceding receipts/raw scores/probe outcomes are exact; health adds150reads/
150SVDs and11.9161s instrumented time including guards/copies. No units meet the
fixed near-zero-activity definition. Mean six-packet endpoint saturation is.5509
forBP and.5394 forRNN, zero forPCvariants. Effective rank differs, but all initial
both-cue acquisition premises remain false. No causal plasticity or advantage
claim follows from these descriptive measures.

117 unique focused bodies pass across two runs (116 then21publication follow-up),
not one117-case invocation. Scoped statics and independent SOURCE/GO/ACTUAL pass;
failed typing attempts and all scientific negatives remain in
[the exact evidence](../artifacts/runs/r21-native-activation-20261016/RESULTS.md).
Next: public-history296 live support in the existing evaluator, a concrete port
prerequisite for a separately frozen fixed-task acquisition case.

## Encoding-bound TRAIN health

The same reader/route now accepts exact InvariantTorchFeatureSnapshot records.
Complete encoding/native-policy/optimizer/gradient/RNG state validates, then every
raw296 public initial/current row is canonicalized before the unchanged float32
tanh prior. Output is owned float64; no labels, updates or relaxed PC states.

```powershell
# Resource-free singleton preview, executed successfully.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --pc-qualification --native-activation-health --describe
# Execution requires a separate frozen reviewed local protocol.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --pc-qualification --native-activation-health
```

Omit --pc-qualification to describe the existing four bound heads plus rawRNN.
Both recipes score only six known TRAIN traces per checkpoint; no health on E
probes or final D. Original health metrics/cutoffs, objective/seed/dose and isolated
probe budgets remain. Bound gradient geometry still refuses before resources.

For singleton PC, health adds30hidden affines/SVDs,270rows/17280accepted hidden-array
bytes and639360returned complete-reader projection bytes to the original160-update/
46score recipe. Full five-method health adds150calls/1350rows/86400hidden bytes,
120bound affines/2557440projection bytes and270rawRNN unrolls. Time/copies/native/
Python/failed-reader allocations remain additional; no equal-compute claim.
A paid TRAIN score survives optional health failure. Preceding values/record/list
identities and exact-type private costs validate before health publication; returned
hidden work stays charged if a later guard refuses. Diagnostics never renew budgets.

32new zero-update reader cases cover all four native heads and refusal/ownership/
RNG behavior.17new route cases include real source0 six-TRAIN score+hidden+SVD,
six extra reader references, fake full composition and adversarial failure costs.
Initial204 and final154overlapping checks pass; scoped statics/preview and independent
SOURCE/executed review pass. The subsequent trained-PC measurement below is separate
from this engineering acceptance.
[Exact commands, review repairs, skips and costs](../artifacts/runs/minigrid-invariant-health-20261017/RESULTS.md).
The separately frozen trained-PC case completed once; see measured evidence below.
All failed acquisition results, native settings and spent allowances are preserved.


## Measured canonical-PC TRAIN priors

The single2026-10-17 frozen native health case completed/SPENT after independent GO.
All30TRAIN cells at0/32/64/96/128 have dead/saturation fractions0 under existing
cutoffs. Mean per-cue effective ranks are nonzero (all checkpoint/context values
retained), with increased mean activity and lower rank at128. These are tanh priors,
not label-relaxed PC states; health does not establish a causal learning mechanism.
Acquisition still fails for every context:55.6% trained TRAIN accuracy/all forward
predictions/exactsequences0. No retention/recovery or circadian advantage follows.

Actual160updates1440rows/46scores414rows plus30hiddenreads/affines/SVDs270rows,
ten teacher episodes90steps. Parent20.3756786s/actualexternalRSS295735296B. Stored
hidden17280B/healthprojection639360B add to original work; profiling scopes include
guards/validation/copies/instrumentation, not kernel-only compute. Saved-array
arithmetic/SVD checker ran once, with independent narrow review.
[All results, commands and limitations](../artifacts/runs/minigrid-pc-health-measurement-20261017/RESULTS.md).

The explicit existing-method selector and fixed BP control are now complete;
[BP evidence](../artifacts/runs/minigrid-bp-qualification-20261017/RESULTS.md) shows
stronger TRAIN fit but failed acquisition. Full800-native bound protocol remains
unexecuted. Next isolate A-only acquisition through the existing stream, with its
own frozen protocol; no automatic run, objective/feature tuning or health variant.
