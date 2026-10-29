# R2.1-ID.C — TRAIN input constraints in the existing fresh-task probe

[Issue260](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/260),
2026-10-27. The existing generic disposable probe now optionally reports exact
TRAIN input/label conflicts without changing its learner execution. One data-only
qualification quantified a previously known policy-observation limitation in the
current54-record ticket probe. No model was fitted or rerun.

## Delivered behavior

`src/app/probe_input_diagnostics.py` reuses the unchanged core
`categorical_input_bounds`:within F, within old TRAIN readouts, joint bounds and
cross-cohort conflicts. It preserves supplied frequencies, exact float64 equality,
signed-zero equivalence and adjacent-value distinctions. At most eight deterministic
detached row/label witness pairs are retained; all conflict counts and truncation
remain visible. No history chain, model, IO, RNG or evaluation authority is added.

`run_fresh_task_probe(..., input_diagnostics=True)` attaches the report after its
original probe, old-TRAIN return and endpoint work, using original detached F/old
inputs. DefaultFalse keeps the original report. No probe-test/final labels or
duplicate return templates enter the diagnostic. Valid unsupported data report
`unavailable` with unknown counts; incomplete detachment is `not_run`; diagnostic
exceptions are `failed` with partial work, separate from paid learning status.
The diagnostic never gates learner/arm admission or grants more updates.

The existing ticket `run_arm` forwards this exact Boolean opt-in. The explicit
live catalog probe enables it. Original source/actor/RNG/budget/clock/receipt
guards, return work and default recipe remain; no new runner is introduced.
Array concatenations are counted separately as attempted/returned work, including
a successful allocation before a later failed allocation. Core reductions record
calls, rows and coordinates. Python/core/witness/validation allocations are extra.
Original phase timers exclude this post-work CPU analysis; caller outer time includes it.

Usage and limits:
[generic probe](../categorical-fresh-task-probes.md#optional-train-input-constraints),
[ticket recipe](../ticket-policy-learning-probe.md).

## Fixed data-only result

The existing D7 builder supplies48 F records (16alternating v1invoice/password
three-row packets) and6 old TRAIN readout records (v2invoice/parcel). D9 uses the
existing context TRAIN builder/checker on exactly the same tasks and54records,
with identical targets, roles, contexts, IDs, order and equal record weights.
Every original D7 feature column matches exactly; D9 adds the two public policy
context tokens. No older legacy-task head probe is substituted as a matched case.

| Joint54-record quantity | Original D7 | Existing D9 context encoding |
| --- | ---: | ---: |
| Unique input keys |5|8|
| Cross-cohort conflicting keys |1|0|
| Conflicting rows |9|0|
| Fixed row-only accuracy ceiling |53/54 =0.9814814814814815|1|
| Exactly normalized simplex CE floor |0.05813868264050533|0|

Both individual cohorts have zero internal conflicts under both encodings. D7's
witness is `probe-0-1`/new-invoice, class5=`route`, versus
`read-old-0-1`/old-invoice, class2=`inspect`. One identical input appears eight
times in F and once in old TRAIN with different labels. No preferred outcome was
an admission assertion. D9 removes this exact-input obstruction on these records.

These bounds assume **one fixed row-only predictor**, not a recurrent/stateful
learner. Record frequencies are not optimizer exposure weights or a phasewise
training objective. Zero conflicts do not imply finite-model attainability,
convergence, task acquisition, retention, recovery, causal plasticity or
generalization. No circadian/controller benefit or efficiency is established.

This is a reusable runtime report plus a quantified source constraint, not a new
discovery that public context exists. The original policy limitation was already
documented before #71; #72 supplied public context. Existing #73 D9/H3 BP joint-fit
qualification failed two ready decisions even after context removed aliasing;
do not rerun it or infer context guarantees native fit. #74 shared perceptron
joint fit and #77/#78 paired-head learning/private probes already exist. The latter
demonstrated known-context further learning/loss/return under its own limits.
Those controls differ in representation/capacity/history and are not matched
neural/CPC comparisons. Reuse their accepted evidence rather than rebuilding them.

The newer #259 native catalog-candidate probe remains negative for new invoice
acquisition. Its warm old tasks never failed, so its return is not recovery.
Its native and actual-file read allowances remain spent. This result neither
changes that artifact nor establishes the cause of its failure.

## Verification, failures and costs

Final combined current source:Windows `win-5` and Linux `linux-2` each33PASS,
0FAIL/0SKIP:31new cases plus two unchanged generic probe regressions. They cover
frequency/signed-zero/adjacent-float bounds, witness truncation, malformed roles,
unsupported/failed/not-run reports, original data detachment, unchanged traces,
metrics/budgets/default report, exclusion of probe-test labels, failure accounting
and the actual existing catalog→generic F/return path with scripted native ports.
No real native model is constructed by these checks.

Raw checks/JUnit and exact invocations:
[`artifacts/runs/probe-input-diagnostics-20261027/`](../../artifacts/runs/probe-input-diagnostics-20261027/).
The final commands are in [PROTOCOL.md](../../artifacts/runs/probe-input-diagnostics-20261027/PROTOCOL.md);
each check receipt also records the full target arguments. No automatic full suite.

| Software invocation | Observed outcome | Body/tool-wall seconds |
| --- | --- | ---: |
|win-red|Expected missing-module1FAIL; no model callbacks|2.0551074 tool wall|
|win-1|17PASS1FAIL:flag initially anchored in private score signature|1.5622974 tool wall|
|win-2|17PASS1FAIL:fixture probe-test contexts mismatched F|1.397238800|
|win-3|23PASS|1.335316600|
|win-4|33PASS|8.243573600|
|win-5 final|33PASS|10.001253900|
|linux-1|33PASS|12.789565159|
|linux-2 final|33PASS|13.431483316|

The signature placement was corrected; the invalid fixture received matching
context names with distinct evaluation IDs. Production admission was not weakened.
Windows six-call allowance is spent, cumulative24.594787700s including conservative
tool-wall charges for the first two calls. Linux two calls total26.221048475s.
Observed maximum sampled RSS83,128,320B Windows/77,611,008B Linux; first two
unmetered failed Windows calls have unknown RSS, not a fabricated zero.

All Windows software cumulative work including failed calls/conservative prefix
upper:178scripted updates544rows,344predictions788rows,298sandbox transitions,
143bounds2000rows4908coordinates,58176B diagnostic concatenations. Linux:
168updates534rows,284predictions728rows,298sandbox transitions,
78bounds1104rows3824coordinates,39744B concatenations. Model/reference/history
work0. Raw counters and qualifications are in `cost-rollup.json`.

Seven-file final Ruffformat/lint and default+Linux-target mypy pass
(`statics-verified.json`). Initial lint failed an unused import; mypy failed
witness/optional-keyword typing. The next static run caught new test work-counter
annotations and NumPy's union-typed RNG state; all fixed before the final combined
tests. All failed static receipts remain. Static commands32.879244700s plus
0.7123797s formatting tool wall. Mtime-selected cache payload113352704B mypy+
168496B Ruff, below160MiB; inherited global caches are separate, not growth.

The sole actual data-only command:

```powershell
./.venv/Scripts/python.exe -B artifacts/runs/probe-input-diagnostics-20261027/qualify.py
```

Exit0/complete in0.466345100s; sampled RSS45,039,616B after NumPy/sampler imports,
not a process-tree or hard-enforced peak. Four pure teachers48sandbox transitions,
6bounds216rows1728coordinates,15552B diagnostic concatenations. Retained cohort
feature/target arrays9504B plus768B temporary D7 teacher arrays counted
conservatively. Transient validation/encoding/Python objects are RSS-scoped;
not an exhaustive allocation count. NumPy/Python global RNG unchanged, no model,
fit, prediction, probe execution, heldout scoring, final or historical payload read.
`qualification-attempt.json` and `qualification.json` use exclusive creation.
The15552B concatenation/64KiB cohort-payload distinction was amended prospectively
before execution in issue260 comment6453920557, not retroactively after spending.

One non-scientific inline counter presentation failed with KeyError by choosing
keys from an unmetered empty-count receipt. A key union fixed the rollup; no test,
teacher, model or qualification repeated. Original receipts remain authoritative.

## Self-review and handoff

Single-agent source/procedure/result SELF-REVIEW passed after inspecting combined
admission, detached inputs, reporting failure, callback/timer/budget boundaries,
live-path forwarding, exact same-task/target/column checks, actual witness and costs.
This is not independent acceptance. Prior accepted math/native/control evidence
is reused where unchanged; no new scientific/model/release qualification follows.

Full suite, clean clone, wheel/install, browser/UI, independent review, new
BP/PC/neutral-CPC/active-CPC runs, physics, generalization and release checks were
not run. No dependency/download/GPU/CI/agent/Git mutation or unrelated change.
Historical plans/log remain frozen under AGENTS; live issue260/tracker18 carry
current scope, status and handoff. Broader #21/#23/#26/#33 and full goal remain open.

Critical path:stop adding checkpoint/provenance sequels or repeating the known D9
joint fit. The richer ticket catalog currently binds its native D7 BP recipe;
the existing learnable D9 paired-head live path accepts only the original fixed
references/simple contextual chooser. There is no catalog attachment there.
Next implementation action:scope an explicit wait/wake-only catalog attachment
at `context_head_live.run_case`'s existing selection seam, reusing the shared
executor and conventional learner. Reject unsupported neural rate/gate/replay
arms before construction; preserve existing stream/FIFO/guard/budget behavior.
Start with literal capability/admission tests before any new model allowance.
Why this:it permits the richer controller to be tested on an already learnable,
policy-observable baseline rather than treating D7's conflicting task truth as
evidence about learning plasticity. No promised model/controller advantage.
