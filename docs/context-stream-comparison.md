# Contextual stream controller comparison

The existing serving CLI exposes a pure scenario preview and a finite comparison
over the clean/noisy contextual TRAIN recipes. The comparison reuses the frozen
scalar controller fit and two earlier periodic engineering cases as saved data;
it runs only the ten missing controller applications. No controller is refitted.

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source unused --context-stream-protocol --output local-context-protocol.json
```

The preview opens no source, teacher or learner. It returns the same pure mapping
used by the worker: all24 public task/version/scenario pairs, with explicit stale
ticks4/12/20 and revoked ticks6/14/22. These are finite lists, not modulo rules.
Output must not exist. For a separately budgeted application, the comparison flag
is `--context-stream-comparison`, requiring `--source`, `--periodic-report` and
`--fitted-report`; see the frozen protocol before running any model work.

## Boundaries and interpretation

The single frozen comparison completed all ten missing applications (24updates,
108TRAIN rows). Each retained old4/4 but served new0/2; all ten deployed actors
remained unchanged. Noisy drift/fitted exhausted six updates and failed retention
guards. Candidate drift paid22reads/69rows per recipe yet stalled on an already
correct pending packet, leaving four queued and preventing publication. This is
a scheduling limitation, not evidence of failed capacity. The historical clean
periodic case remains5/6 including the deliberate denial. No controller benefit
was demonstrated. Full costs, negatives and software failures are in RESULTS.md.

`scripts/context_stream_comparison.py` composes existing live execution, paid
report guards and controller objective. It validates preceding paid records and
all12 complete cases before utility. Corruption, equal replacement, incomplete
work or changed source bytes stop admission and retain the paid prefix. Defaults,
source versions, original budgets, guards and clock semantics stay unchanged.

Clean then noisy each use no-learning, drift, budget-aware, fitted and candidate
drift, plus the saved periodic case. Candidate diagnostics see only arrived
pending TRAIN and pay their144-row ceiling; the fitted controller uses its
original actor feedback. Pending work, candidate learning and actor publication
are separate outcomes. No final/probe data or updates enter this path.

All modes use common ticks17–22 against their recipe's no-learning counts and
the unchanged objective. The deliberate denied v2legacy case22 stays in the
denominator; it is neither failed acquisition nor an applied safety violation.
Earlier periodic data lacked a complete prospective scenario description. That
requirement remains open in issue84; this study cannot retroactively satisfy it.
Fixed-order costs include asymmetric publication/rollback/diagnostics and heap
retention, so utility is a descriptive proxy, not an efficiency ranking. The fit
uses a classifier proxy rather than closed-loop task utility. Known templates
and one development protocol establish no generalization or circadian advantage.

Evidence/protocol: `artifacts/runs/context-stream-comparison-20261023/`;
[issue85](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/85).
Extend through the existing recipe/controller ports, preserving source/role
checks and freezing a new protocol before any further comparative work.
