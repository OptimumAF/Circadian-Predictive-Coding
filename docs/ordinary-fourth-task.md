# Ordinary fourth-task TRAIN adaptation

The existing affine workflow can collect a new original TRAIN task, learn it
through the delayed stream while reusing prior TRAIN rows, and play known new
and returning tasks from its actual endpoint. No research model archive or
private-probe promotion is needed. The existing console command exposes this
recipe through a mutually exclusive option:

```powershell
circadian-affine-game --example-affine-recurring --describe
circadian-affine-game --example-affine-recurring --output new-recurring-run.json
```

From a checkout, use `python -m scripts.run_minigrid_linear_fit_control` in place
of `circadian-affine-game`. Describe returns metadata before data/native work.
Execution uses bundled ABC and freshly collected original G TRAIN data, no saved
model/archive or final/private inputs. Output is exclusive, limited to4MiB;
failed runs retain their structured prefix and exit2. Freeze a local allowance
before executing. The normal and private flags/defaults retain their behavior;
the three example modes cannot combine with each other or saved-source options.

```python
from scripts.minigrid_affine_workflow import describe_workflow, run_workflow

protocol = describe_workflow(recurring_training=True)  # Metadata only.
# Run only within a prospective local allowance; this performs collection/fits/games.
report = run_workflow(recurring_training=True)
```

This API option must be an exact bool and cannot combine with
`private_adaptation=True`. The existing entry point and old flags keep their modes;
the new option adds no command, dependency or environment variable. Native play uses
the existing optional pinned MiniGrid requirements and checks them before
collection or model creation. The old normal/private workflows remain separate.

## Original TRAIN source and isolation

`predefined_recurring_train_layout()` declares G_train: unrotated, unswapped grey
layout with keys at(2,5)/(4,5). Its initial visible cue moves to(1,5), preventing
the old cue position from overwriting the left key. The agent, neutral door,
goal, actions and public teacher are unchanged. G accepts only its original
TRAIN role before environment construction. Other presets, reserved F/E probe
roles and D final gate remain unchanged. G is not a renamed F or D record.

Why this: canonical encoding removes compass and wall color. A distinct key
position gives an ordinary new public task without importing reserved labels
or creating a new observation encoder, learner or orchestration framework.
The changed cue position is part of this declared task; there is no isolated
causal claim about key placement alone.

The workflow reuses packaged original ABC TRAIN data and newly collects G
blue/red with seed17/max16 steps. Each teacher source must have nine complete
steps, positive return and confirmed cleanup before the next source is admitted.
Raw148 views become causal raw296 histories using the existing codec, retaining
original episode IDs and role. No target or private state enters observations.

## Learning, return and costs

The fixed sequence is Ablue/red → Bblue/red → C_train blue/red → G_train blue/red
→ Ablue/red. The existing stream owns delay-one labels, receipts and one original
ten-update/90-second/512MiB allowance. Three earlier same-public-cue packets are
joined with each current packet; FIFO retention is at most54 rows and each fit
at most36. It is a finite plan, not an unlimited replay or success-record chain.

Ten analytic fits expose252 rows:90arriving exposures (72distinct source rows,
including18G, plus18return-A repeats) and162joint replay exposures. Input arrays
use598752bytes, peak joint template85536bytes and bank observation/target float64
equivalent256176bytes. Bank equivalence excludes rewards/metadata/Python/native
copies; sampled process RSS has a broader scope. The consolidation-event replay
counter remains0; it does not describe the162joint repeated rows.

All fits finish before detached TRAIN monitor scores at0/2/4/6/8/10 updates.
Eight nine-row sources produce48score calls/432rows. Accuracy/action counts,
cross-entropy and old/new/return states remain separate; none selects later fits,
replay, checkpoints, probes or parameters. TRAIN retention is not heldout recall.

Only a complete observed10-update/252-row/two36-row-head endpoint with complete
receipts enters the existing native evaluator: Gblue,Gred,then original Ablue,Ared
(A preset0, branch3), seed17/max16 each, all seven actions with first-argmax.
Readout has its original150-second/zero-update/zero-replay budget. The whole
workflow additionally has300seconds, a290second work deadline and512MiB cap.
These limits never renew a nested allowance. Complete negative games remain
valid completed observations; failed or unrun games retain uncertainty.

Original source, actor, RNG, clocks, budgets and preceding paid collection/TRAIN
evidence remain authoritative. Corruption or equal replacement stops following
work. Later native failures preserve detached paid training, partial episodes,
cleanup and any possible unobserved changes. Missing evidence is not success.

## Verification and remaining claims

The sole fixed observation completed. G improved from14/18 after ABC to18/18
after its two fits, with old ABC still18/18. All four tasks remained18/18 after
A returned. Native Gblue/Gred/Ablue/Ared all succeeded in nine steps each with
return0.49375; all six environments, including the two teacher sources, closed.
Ten fits exposed252rows, including162joint replay rows. Body128.984seconds,
sampled peak78,073,856bytes. Confidence costs remain: final A CE1.189879 is worse
than its first-fit1.165422, and final C CE1.189879 is worse than1.181246.

| Updates | A correct/18 | B correct/18 | C_train correct/18 | G_train correct/18 |
| --- | ---: | ---: | ---: | ---: |
|0|2|2|2|2|
|2|18|13|11|15|
|4|18|18|13|13|
|6|18|18|18|14|
|8|18|18|18|18|
|10|18|18|18|18|

[Software protocol](../artifacts/runs/ordinary-fourth-task-20261024/SOFTWARE_PROTOCOL.md),
[single native protocol](../artifacts/runs/ordinary-fourth-task-20261024/NATIVE_PROTOCOL.md),
[exact results/costs/failures](../artifacts/runs/ordinary-fourth-task-20261024/RESULTS.md),
[issue124](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/124).
66focused cases qualify per platform, including scripted source/role/grid/budget/
receipt/failure and old default checks. Scoped Ruff/format/fresh Windows types
and single-agent source/procedure/result self-review apply; not independent review
or repository-wide release acceptance. Literal fixtures are not native outcomes.

The native artifact is one fixed known-task observation. It cannot establish
statistical reliability, heldout transfer, recovery after actual forgetting,
matched computation, a comparison against simpler controls or CPC advantage.
The failed/spent120private and108native cases remain unchanged and unretried;
full R2/R4/R7 and release requirements remain open. New comparative work needs
its own frozen data, seeds, metrics, fair controls and budget; no automatic sweeps.

The installed recurring command also completed one bounded Windows delivery
qualification from an empty working directory. The freshly built wheel and
isolated target used cached dependencies; all142 loaded project module origins
were inside that target. Full model checkpoints, TRAIN curves, receipts, both
teacher sources and all four game traces exactly matched the earlier source run.
The installed run performed10fits/252row exposures and54environment steps,
including18teacher steps; all four games succeeded in9steps each. Parent elapsed
time was74.106seconds and sampled actual-interpreter peak RSS83,374,080bytes.
This repeats a fixed working control to test delivery, not independent scientific
replication. The original A/C confidence regressions remain. Final D results
never enter this recipe. See [installed protocol and results](../artifacts/runs/installed-recurring-affine-20261024/RESULTS.md)
and [issue127](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/127).
31focused checks qualify per platform; scoped statics and self-review pass.
Clean-clone/dependency resolution, Linux native gameplay, the generated Windows
launcher and repository-wide release gates were not qualified by this case.

Replay your saved recurring result through the
[shared offline inspector](affine-workflow-inspector.md#ordinary-recurring-reports):
`python -m scripts.build_key_door_inspector --source-artifact new-recurring-run.json --output recurring-learning.html`.
This displays the recorded learning, known G/A public-view traces and original
costs without restoring weights or executing another learner/environment.

The checkout also offers a fixed [A4 development-layout observation](affine-development-layout.md)
from the actual saved ordinary main model, with no fitting or teacher work. The
sole case was1/2: blue truncated after16 steps, red succeeded after13. This
preserves known G/A results while exposing a bounded transfer limitation.
