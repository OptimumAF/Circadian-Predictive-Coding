# TRAIN hidden health for public-cue checkpoints

The existing public-cue pilot optionally reads native recurrent hidden activity,
dead/saturated fractions and feature effective rank. It reuses
`read_native_activations` and `diagnose_activations`; no learner, objective or
diagnostic definition changes. Preview without data IO or model construction:

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_paired_bc_smoke --describe --public-cue --recurring-training --activation-health
```

The API option is `run_pilot(episodes, recurring_training=True,
activation_health=True)`. It composes with the existing mean-CE, explicit
class-balanced and declared replay recipes. Defaults and all six existing
descriptor byte patterns remain unchanged. Describing a recipe does not authorize
a new experiment or renew any prior allowance.

All fitting completes first. At each original0/32/64/96/128 checkpoint, six
original complete9-row/raw148/K7 TRAIN traces are read before the ordinary
teacher/game evaluation. Actual first public observations select heads; context
names and labels do not route them. Each trace starts recurrent activation at
zero and pays nine native forwards including discarded output projections. The
full recipe adds30reads/270forward calls and rows/30SVDs/17,280returned hidden
array bytes, plus full snapshot/copy/guard costs. Hidden arrays are discarded;
only30bounded metric records are retained across the five fixed points.

Why this: earlier native base-acquisition failures did not establish a missing
cue or explain the learner's representation. Observing actual hidden states
can test activity/saturation hypotheses while preserving the deployed learner.
It is not a causal plasticity test or a task-success measurement.

`src.adapters.minigrid_public_cue_health.read_checkpoint_health` also composes the
same diagnostic with caller-owned live state, original source and readout budget.
It requires the six original ordered TRAIN packets, exact full native policy and
per-head clocks, original cold source and the same200s/768MiB zero-work budget.
The caller guard owns actor, RNG and collection provenance. The pilot checks its
original NumPy/Torch RNG at every diagnostic boundary. Changed source/checkpoint,
arrays/records, source owner, budget identity/clock/charges or preceding health
evidence refuses; the component never restores changed caller state.

The report includes attempted/completed activation reads, admitted forward-call/
row upper bounds, completed native forwards/projections, returned array bytes,
SVD attempts/completions, direct snapshot operations and elapsed time. A missing
native return makes internal forward progress unknown: it is not recorded as
completed. Caller guard work is excluded from the component ledger and included
in the outer execution accounting. `ActivationHealthFailed.report` retains the
last verified paid prefix and uncertainty. The pilot preserves this separately
as `activation_health_failure`, retains preceding accepted records, and stops
before the next game/checkpoint. No changed prior evidence is resealed.

81distinct scoped software cases are qualified on Windows/Linux: Windows76
initial combined plus44final affected cases, with39repeats; Linux81on the final
combined tree. They cover original default/weighted/CLI behavior, actual public
routing, all declared mean/replay native clocks, full-state/role/input/owner/
budget refusal, SVD/lost-return failures, final-callback mutation and preceding
health tampering. Five-file Ruff/format/fresh available mypy pass. Native Linux
Ruff/mypy remain unavailable and were skipped; no installation/full suite ran.

One zero-update actual-Torch qualification passes per platform on the packaged
six TRAIN episodes. All six metrics match direct calls to the existing native/
core functions exactly within each environment, with source/checkpoint/inputs/
budgets/global RNG preserved. Each case pays2native constructors/6private randn
draws2608scalars,108forward rows/calls (54primary+54reference),12SVD,0SGD/backward/
games. Identical initial heads limit native routing discrimination; scripted
distinct heads cover that behavior. This is integration evidence, not learning.

Observed cold TRAIN metrics: no dead units on any trace; saturation0 for Ablue
and1/72 for each remaining trace; effective rank roughly4.90–5.51. Small floating
point differences remain between cached platforms. These data do not support
blanket cold inactivity/saturation, nor establish what happens after training.
All trained failures and unrun endpoints from earlier experiments remain intact.

Exact commands, original budgets, costs, XML metric records, failures/skips and
next action: `artifacts/runs/public-cue-health-20261025/RESULTS.md`. No spent
128-update/probe/game experiment is rerun or resumed by this feature.
