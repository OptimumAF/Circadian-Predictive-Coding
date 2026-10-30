# Isolated further learning with contextual LS heads

The existing `scripts.probe_ticket_policy_learning.run_arm` API accepts an
in-memory `ContextLinearFitLearner` and its current owned snapshot:

```python
from scripts.context_head_probe import build_head_cohorts
from scripts.probe_ticket_policy_learning import run_arm

cohorts = build_head_cohorts(spec)
row = run_arm(
    source, source.snapshot_state(), cohorts, spec, check_boundary, memory,
    "warm", separate_context_heads=True, linear_fit=True,
)
```

`source` is the existing owned D9/K6 public-context LS learner, `spec` is the
original contextual ticket feature/catalog spec, `check_boundary` is the caller's
source/benchmark/deployed/environment guard, and `memory` is its existing RSS
sampler (or None in small fixtures). This adds no saved-source loader or CLI flag.
The frozen283raw capture remains closed; do not reopen it to supply parameters.
Run from the physical D: checkout with TEMP/TMP (also Linux TMPDIR) on D:.

The original generic FreshTaskProbe forks the source and updates only that
disposable owner:16alternating3row `probe_train` packets48rows, then16alternating
6row old-TRAIN updates96rows on the same private learner. Source prior updates/
rows are reported separately. Each phase retains original checkpoints0/4/8/16,
CE/top1/decision diagnostics and old-TRAIN readouts. LS update diagnostics remain
the native squared residual, not CE. No model copy is promoted or installed.

Initial, after-probe and after-return frozen typed readouts execute the fixed
four old tasks and two legacy tasks, with maximum6steps each. These are unused
**known TRAIN** legacy templates with shared workflow stages, not unseen data or
final evaluation. Initial ability is measured rather than assumed. Further
learning, retention and recovery need a separately frozen scientific observation;
passing software fixtures alone do not establish those claims. A later
[one-case observation](reports/context-linear-further-learning-20261029.md) learns
one initially failing legacy task, loses three old tasks, recovers those on old
TRAIN return, and loses the new task. The other legacy task was already correct.
This supports further learning/recovery with incomplete joint retention.

Both original16update/30second/512MiB sessions are born once at arm entry; return
time includes preceding probe time. No renewal, benchmark/source training,
final labels, replay, tuning, controller feedback or circadian mechanism. Failed
fits retain their existing owned scalar work receipt; a completed budget count
of0 does not erase an attempted solve. Initial-read failure prevents any probe
fit; failed probe prevents return. Preserve the supplied `row={}` when calling
if the caller needs partial records after a raised completion refusal.

The LS mode binds source policy/state identities and values, supplied snapshot,
original cohorts/spec and NumPy/Python RNG. It seals paid readout/episode list and
row identities/values plus source-history headers, before and after external
callbacks, including throwing callbacks. Changed evidence is restored and refused
before further work; foreign source/input/RNG changes are refused without rollback.
Typed reader routing and selected-head work are retained before post-call guards.
Maximum source/private model coefficients are768B each, with up to three endpoint
snapshots2304coefficient bytes; metadata/copies/validation are additional. No all-in
compute or capacity-matched scientific comparison is implied.

Flags are exact booleans. `linear_fit=True` requires `separate_context_heads=True`,
the fixed D9/K6 schema, ordinary legacy cohorts and no input-diagnostics/transfer
override. Existing native/perceptron API and CLI defaults remain unchanged.

[Qualification, commands, costs, failures, review and limits](reports/context-linear-probe-20261029.md).

Any new scientific work must freeze its distinct question and all source/probe/
return/reuse costs before execution.285is COMPLETE/SPENT, its raw stays closed.
Reuse existing diagnostics and this API; do not create another training runtime
or reconstruct a spent raw capture.


The existing LS API also supports the explicit fixed parcel-v1/password-v2
[task/head transfer mode](context-linear-transfer-probe.md). It measures initial
ability and preserves source/deployed isolation and original probe/return budgets;
legacy/default probes above remain unchanged. Known global concepts/stages are
shared, so this mode does not establish unseen-task generalization.
