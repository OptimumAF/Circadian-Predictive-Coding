# Isolated reserved-task probe from an ordinary native workflow

The existing chronological ticket workflow accepts an explicit `--reserved-probe`
with `--context-return` and `--method bp`, `pc`, `neutral_cpc` or `gated_cpc`.
Run from the physical D: checkout, with project temporary outputs on D:.

```powershell
$env:TEMP='D:\Dev\_Temp'; $env:TMP='D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m scripts.recurring_ticket_workflow_example --context-return --method bp --reserved-probe --output artifacts/new-bp-probe.json
```

Output must be new. This mode uses the existing exclusive bounded report writer
with a4MiB limit and a scalar-cost fallback on serialization/size failure.
Other modes retain their existing output contracts, including the512KiB encoder.
Current native Torch dependencies must already be available; the command does
not install or download anything. Omitting the new flag preserves ordinary behavior.

The original source learns12updates72rows through the existing delayed stream.
Only after its checkpoint12 and receipts are validated does a detached learner
receive16probe updates48rows, then16old-TRAIN return updates96rows. Total44native
updates216row exposures for a complete invocation. Source costs remain separate
from probe costs; row exposure is not matched total computation.

Why this: the source's counted forks carry callbacks tied to its own budget.
The probe instead constructs a detached native owner with the same complete
state/policy and independently counted callbacks. Source/deployed learner, actor,
RNG and original receipts/clocks remain guarded. Probe parameters are never
promoted or restored into the source. No source budget is increased or renewed.

The source keeps its original30s/512MiB limit. Two separate probe/return owners
are each16updates/30s/512MiB, born once at admission; return's clock includes
the preceding probe time. Additional task reads also observe the original
source clock. Paid or failed report objects cannot be reused to reset charges.
Operational failures retain partial work and stop before later phases.

The fixed data comes from the existing TRAIN-derived public-context cohort
builder. New diagnostic tasks are v1parcel/v2password; old tasks are v1invoice/
password and v2invoice/parcel. The original source task schedule and its legacy
readouts remain unchanged. These task sets must not be silently interchanged.
`probe_train` labels alone train F; `probe_test` labels are metric inputs. Old
TRAIN monitors only score during F; separate old TRAIN templates train return.
Final evaluation is unavailable to all of this work. These are known templates,
not unseen-task or held-out generalization evidence.

Read the report fields as follows:

- `training_work`, `budget`, `costs`: original source12/72 only.
- `reserved_probe.preserved_training_work`: all attempted/returned probe+return
  calls/rows, including failed attempts; successful total32/144.
- `reserved_probe.probe`: existing categorical/input diagnostics, fixed probe
  checkpoints0/4/8/16, old TRAIN readouts and nested `return_phase`.
- `reserved_probe.sessions`: both original limits, birth/last clocks and charges.
- `reserved_probe.episodes`: six fixed typed tasks before F, after F and after
  return; `task_readout_work` includes their native forward work. Success must be
  assessed separately from probability scores, losses or class accuracy.
- `probe_updates` is32 after complete work and null when an attempted native
  return was lost; nested paid records retain the attempt. Null is not zero.

The mode rejects replay, private Perceptron adaptation, publication/controllers,
separate heads, alternative LS/probe rules, class-balanced BP and TRAIN-fit
diagnostics. Those combinations need explicit separate qualification.

Implementation: `scripts/ticket_native_probe.py` composes existing cohorts,
`run_fresh_task_probe`, native frozen readers and typed episode evaluator. The
ordinary runner retains source authority. No new learner, scheduler or optimizer.
Before extending it, preserve full native state/shape/role ownership, counted
failures and original budgets, and test the new boundary before experiments.

[Commands, failures, costs, observed outcomes and acceptance](reports/ordinary-native-reserved-probe-20261030.md).

A subsequent zero-model BP diagnosis found row accuracy staying2/3 while
cross-entropy decreased: open/commit examples were correct, every routing
decision remained wrong. The typed dispatcher then refused repeated wrong-phase
actions. This does not establish a unique capacity or optimization cause.
[Saved traces, class diagnostics and limits](reports/ordinary-native-bp-diagnosis-20261030.md).


The TRAIN-only [frozen categorical readout diagnostic](torch-categorical-readout-fit.md)
has scoped software qualification (#317). Its [genuine source12 diagnosis](reports/ticket-frozen-readout-native-20261030.md)
(#318) improves known TRAIN decisions8/12→10/12 and routing0/4→2/4, with two routing
errors remaining. No fitted task rollout was run; existing negative reserved-probe
results stay unchanged.

[Saved TRAIN separation analysis](reports/ticket-frozen-readout-separability-20261030.md)
(#319) found independently verified full/allowed witnesses for all12 rows. This
supports readout learning as the next repair, without adding hidden capacity.
It does not change the native or reserved-probe task failures above.

An optional Python-only `ce_initialization` argument now admits an explicit
[CE-initialized detached native state](torch-ce-probe-initialization.md). The
original source checkpoint/SGD/budget stays separate, all auxiliary fit costs
remain paid, and the existing private factory owns probe callbacks. Default and
CLI behavior are unchanged. Scoped software checks do not establish the new
native F16/return16 outcome; that experiment is still unexecuted.

The subsequent [native attempt](reports/ticket-ce-probe-native-20261030.md) failed
before fitting or F updates because the local observation wrapper misread the
public copied-receipt contract. It remains FAILED/SPENT with no adaptation result;
the optional library path is still qualified only at its recorded software scope.

The ordinary runner now exposes that composition explicitly as
`--reserved-probe --probe-initialization ce` for BP/context-return.
[Usage and software qualification](ordinary-ce-probe.md) retain separate
auxiliary costs, authoritative receipt guards and original source authority.
No genuine native adaptation run or renewal of323 is implied.

The ordinary runner appends `reserved_probe_summary` after original cost/failure
bookkeeping. It separates the fixed source-trained4 and fresh2 across
`before_probe`, `after_probe`, `after_return`. `successes_by_phase` uses only
complete six-task phases. Incomplete phases retain observed prefixes but
aggregates/dependent changes stay null: unknown or inapplicable, never zero.
`recovered_on_return` requires success before F and failure after F. Inspect
`lost_during_probe` to distinguish inapplicable recovery from missing return.

The detached metadata projection needs no learner imports:

```python
from src.app.reserved_probe_summary import summarize_reserved_probe
summary = summarize_reserved_probe(already_returned_report)
```

Wrong/final roles and malformed complete episodes are refused, not re-scored.
Runner records `status: invalid` on projection failure, preserving raw results/
errors/work. Source, auxiliary fit, private F/return and readout counts stay
separate; missing stays unknown. Keep raw timing/memory and original source
guards: this summary authenticates neither source preservation nor total cost.
[Checks and one frozen saved-report projection](reports/reserved-probe-summary-20261030.md)
preserve all four baseline failures and323's unknown private outcomes, with no
new prediction, training or final evaluation.

## Direct checkpoint12 linear control

An explicit simpler LS control can start from the ordinary source before any
serving/rehearsal child training:

```powershell
.venv/Scripts/python.exe -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --reserved-probe --probe-update minimum_change --train-update minimum_change --output NEW-D-PATH.json
```

Omit `--train-update` to retain cold source TRAIN updates; private probe updates
still explicitly use minimum-change. Existing `--paired-serving retained_joint_probe`
keeps its different23/144 donor; the direct route uses12/72 without a serving
child/replay/publication. It shares original F16/return16, six typed task reads and
`reserved_probe_summary`, with source/private costs separate. Source clocks,
receipts, actor and RNG stay guarded. Completed prior readout replacement/
corruption refuses work; bounded `preserved_readouts` retains its trusted prefix
on failure. Native-forward fields absent from LS work stay unknown; raw routing/
affine work remains separate.

The same ordinary checkpoint12 path also accepts a **single shared head**:

```powershell
.venv/Scripts/python.exe -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --reserved-probe --probe-update minimum_change --train-update minimum_change --output NEW-D-PATH.json
```

Omitting `--separate-context-heads` selects that existing ordinary single head;
keeping the flag retains the prior two-head control. Source TRAIN choice stays
separate from the disposable child's explicit minimum-change operation. The
same fixed fresh-F16/old-A-return16 cohorts, six-task readouts and summary apply.
Probe training never updates the source, benchmark or deployed actor. Single-head
replay plus probe, serving, private adaptation and native CE options remain
unsupported. Successful software checks establish admission, transport, isolation
and failure charges, not useful adaptation. [Scope, executed checks and preserved
failed numerical cohort](reports/single-linear-reserved-probe-20261030.md).

The [fixed single-head observation](reports/single-linear-probe-native-20261030.md)
kept the source unchanged but showed old-task success3→2→3: one skill was lost
and recovered, while another was lost during return. Both designated fresh
tasks already succeeded before training and stayed2/2, so this case does not
establish new task acquisition. All loss curves, costs and failures are retained.

[Software qualification and limitations](reports/ordinary-linear-reserved-probe-20261030.md):
133cases on both platforms plus one Windows real-app fixture pass. Ordinary
runtime tests use scripted fits/episodes. The real-app fixture uses a handwritten
source and scripted task reads; those fixtures do not qualify genuine adaptation.
The subsequent [one frozen direct LS observation](reports/ordinary-linear-probe-native-20261030.md)
retained4/4old tasks, improved fresh1/2→2/2 and retained both after return. Only
one task was newly acquired; its initial failure was a near numerical tie.
No loss occurred, so recovery remains unobserved. Return worsened CE despite
retained task successes. This is scoped known-template behavior, without matched
computation/generalization/CPC claims; spent323 remains unchanged.
