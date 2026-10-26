# Context updates through local ticket effects

The existing CLM ticket command can run three declared TRAIN exposures and then
use the resulting linear control to choose actual local workflow tools:

```powershell
python -m scripts.run_clm_ticket_workflow --model-dir <prepared-local-model> --head <prepared-local-head> --output <new-report.json> --linear-context-updates --public-observation-train
```

Use a prospectively budgeted native protocol before running prepared weights.
The option does not grant an experiment allowance. Output creation is exclusive;
an existing report cannot be reused to restart work. This mode requires public
observation TRAIN and cannot combine with the other workflow modes. Defaults
and the original single-fit/contrastive routes retain their behavior.

The fixed order is v1, v2, then repeated v1. Each exposure contains twelve rows
from the original public teacher, with a public policy-context token. The return
uses the original twelve v1 rows/features: it is explicit TRAIN rehearsal with
consent from birth, not new data. One cold minimum-norm solve and two warm
minimum-change residual solves expose36 rows in total. These are least-squares
operations, not native head or optimizer updates.

Why this: complete tasks require `open` and `commit` as well as routing tools.
The control keeps fixed columns for `commit, defer, inspect, open, retire, route`.
Teacher target IDs map explicitly into those six columns. Each episode uses a
new read-only provider binding the permitted five-tool subset. The original
model and all six columns remain intact; active columns are selected before
softmax. The public observation, rather than a private ticket ID or teacher
target, supplies policy context at prediction time.

The existing `ExperienceInbox` receives observations at ticks0/2/4 and labels
at1/3/5. One original three-update budget and clock span the sequence. Training
cannot occur before a label arrives. Applied receipts, learner/encoder work,
source heads/backbone/scale, RNG, features, target bindings and completed report
prefixes are checked before admitting more work. Corrupt or replaced evidence,
unexplained counters and uncertain paid work stop the run. Completed negative
episodes remain valid observations and do not select later training.

After **each** exposure, the existing dispatcher executes these four original
development tasks: v1 invoice/password and v2 invoice/parcel. The old-context
readouts after v2 occur before return rehearsal. Finally, fixed v1 password
stale-state and v2 parcel permission-revocation episodes exercise external effect
checks. There are fourteen episodes, each capped at six steps. Stale-state
episode recovery is separate from relearning a forgotten task. The report claims
learning recovery only if an initially successful old task failed after v2 and
became successful again after return.

Preparation encodes six distinct v1 states plus six action descriptions in one
call, then six v2 states in another:18 sequences. Return features are reused;
their twelve repeated training exposures still count. Each live query encodes
once through the same frozen encoder, whose lifetime ceiling is86 forwards and
sixteen sequences per call. Per-episode provider work includes copying and
guarding the full source control/action bank plus active rows. Sum request/work
fields over the episode records; catalog widths are metadata, not computations.
No native head forwards, gradient/optimizer work, parameter history or model
export is introduced. Only the current control and bounded feature/receipt/report
records are retained. Logical array/copy reserves are not measured total
allocation or FLOPs. The `exposure` fields describe the fixed declared recipe;
on failure consult observed counters/pending records instead of assuming the
whole recipe executed. Local teachers use24 dispatcher transitions for the two
original packets; repeated return rows add no new teacher transition.

```text
src/adapters/clm_linear_catalog_learner.py  explicit union TRAIN option/current control
scripts/clm_linear_context_workflow.py     teacher/inbox/typed episode composition
scripts/run_clm_ticket_workflow.py         opt-in admission and original local loader
tests/test_clm_linear_context_workflow.py  literal CLI, context sequence and failures
tests/test_clm_linear_context_cli.py       refusal before loading/output
```

No new dependencies or environment variables. The generic learner option uses
`candidate_ids=...` and an empty choice-binding tuple. Its TRAIN batches must
carry the full declared catalog. Serving uses `current_control` through the
existing typed provider; attempting to borrow an unfitted or corrupted control
fails. The original temporal learner's four-tool/three-active behavior remains
the default. Extend through these ports rather than adding another scheduler.

Focused tests and static commands, subject to the owning issue's cumulative caps:

```powershell
python -m pytest -q tests/test_clm_linear_context_workflow.py tests/test_clm_linear_context_cli.py
python -m ruff check src/adapters/clm_linear_catalog_learner.py scripts/clm_linear_context_workflow.py scripts/run_clm_ticket_workflow.py tests/test_clm_linear_context_workflow.py tests/test_clm_linear_context_cli.py
python -m ruff format --check src/adapters/clm_linear_catalog_learner.py scripts/clm_linear_context_workflow.py scripts/run_clm_ticket_workflow.py tests/test_clm_linear_context_workflow.py tests/test_clm_linear_context_cli.py
python -m mypy --follow-imports=silent src/adapters/clm_linear_catalog_learner.py scripts/clm_linear_context_workflow.py scripts/run_clm_ticket_workflow.py tests/test_clm_linear_context_workflow.py tests/test_clm_linear_context_cli.py
```

The [software report](reports/clm-linear-context-workflow-20261025.md) retains the
literal failures, measured costs and self-review. Literal one-hot public context
features establish software behavior only. Native acquisition, retention,
learning recovery, an isolated fresh-task probe, matched BP/PC/neutral/active-CPC
comparisons and release readiness remain separate unfinished requirements. Final
evaluation is unavailable to this mode; no probe is trained or deployed here.
