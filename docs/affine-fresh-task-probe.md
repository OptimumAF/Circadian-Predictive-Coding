# Isolated fresh-task diagnostics for the learned affine model

The existing CLI now binds the after-C model to the original reserved Fprobe:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-probe --describe
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-probe --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --retained-source artifacts/runs/catalogue-capacity-new-c-20261024/actual.json --output new-affine-probe.json
```

The second command needs both qualified local sources, optional MiniGrid
dependencies already available, and a prospective budget. Existing outputs and
fit overrides are refused. Old CLI modes/defaults remain. No new dependency.

It collects the existing Fblue/Fred train/test teacher episodes, seed17/max16,
then fits only the firsttwo predetermined training packets on one disposable
fork. The original16train/2test release is validated before selection. All7
classes, same analytic objective, checkpoints0/1/2; no metric selects a dose,
seed, replay, promotion or training input. Probe_test labels stay in metrics.
Train/test share the same deterministic task and seed: this measures fit on
that reserved task, not heldout generalization or native learned Fgameplay.

Old A/B/C are readonly TRAIN monitors, grouped into two27row batches by public
cue. Full source spans/row IDs and existing row decisions preserve separate
task accuracy/CE. No fake27step episodes or increased monitor envelope; the
affine model uses independent296value initial/current frames. No monitor
labels reach training. All probe updates stay on the private fork; source
native state, benchmark and serving paths remain untouched. Fork0→1 is an
explicit ownership event, distinct from native parameter/counter equality.

Source6fits108rows remains separate from paid upstream10fits180rows and new
probe2fits18rows. Collection, diagnostic forwards/reductions, storage/copies,
wall and RSS count. Failures preserve closed collection prefixes, paid native
work/readouts and one private endpoint; budgets and clocks cannot renew.
No final/controller-validation data, replay, return phase or promotion. The
existing16step return option remains available elsewhere with its own criteria.

## Observed fixed diagnostic

| Private probe updates | F correct | A correct | B correct | C_train correct |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 15/18 | 18/18 | 18/18 | 18/18 |
| 1 | 16/18 | 17/18 | 16/18 | 16/18 |
| 2 | 18/18 | 16/18 | 14/18 | 14/18 |

Fmean CE improves1.413812→1.165422 while old-task CE worsens. The learner can
fit the reserved task after C training, with visible interference on its private
fork. **The original after-C model is unchanged.** Ftrain/test transitions are
identical for each cue under the fixed collector; this is fit, not generalization.

The sole run used2probe fits/18rows,12diagnostic predictions/216rows,12metrics
and12row-diagnostic calls,36teacher decisions/physics steps for4closed sources,
0benchmark/deployed/replay/return updates. Whole body21.742672seconds,
parent22.279737seconds, sampled peak95,539,200bytes. Input arrays406,944bytes;
helper additionally owns85,536Fbytes and128,304monitor bytes. Native source6/108
stays fixed, private endpoint8/126; one fork event explicitly recorded.

32distinct focused checks qualify per platform; scoped lint/format/types and
source/procedure/result self-review pass, not independent acceptance. One check
requires local accepted sources and explicitly skips elsewhere; no cleanclone
or bundled-input qualification yet. No failed numeric attempts or case retry.

No CPC, efficiency, reliability, general plasticity, full recovery, nativeFcontrol
or release claim follows. Earlier failed/spent evidence stays intact.

[Protocol](../artifacts/runs/affine-fresh-task-probe-20261024/PROTOCOL.md),
[results/commands/costs](../artifacts/runs/affine-fresh-task-probe-20261024/RESULTS.md),
[issue113](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/113).

Extend through existing role-separated probe ports and finite monitors. Keep
diagnostics from choosing deployment or altering benchmark/final evaluation.
