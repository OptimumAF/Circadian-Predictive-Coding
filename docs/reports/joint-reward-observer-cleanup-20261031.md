# Bounded cleanup after an observer work limit

R4.2-RNN.JRL.C / [issue390](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/390).
A narrow observer repair is implemented and executed with literal ports. It is
a prospective patch, not a modification or rerun of the frozen native case.
The original144-update comparison stays FAILED/SPENT with missing readouts and
unknown final preservation. This correctness result supplies no learning claim.

Why this: the original observer checked every cap on every operation. Exceeding
its serialization-work limit consequently refused environment closure too.
Only the exact integer charge `{'closes': 1}` now receives a separate admission
check against the original52-close limit. Other charges, including readbacks,
training and simulator steps, still refuse when any original work cap is spent.
Mixed, zero, negative, Boolean and non-integer close charges refuse before debit.
Attempts and failed closes remain charged. No counter, limit, clock or error is
reset, and cleanup cannot make a failed budget successful.

Deliverables:

```text
tools/patches/joint_reward_observer_cleanup.patch
artifacts/runs/joint-reward-observer-cleanup-20261031/
  PROTOCOL.md, fixture-original.py.txt, fixture-repaired.py.txt
  test_cleanup.py, run_checks.py, first-results.json, logs and static receipts
docs/reports/joint-reward-observer-cleanup-20261031.md
```

The unified patch adds nine lines to `charge`. Ordinary `git apply --check` and
`git apply --directory=artifacts/runs/joint-reward-observer-cleanup-20261031`
applied it only to the separate text fixture. Neither full observer text was
imported or executed. Tests compile only the original `caps`, `TRAIN_KEYS`,
`charge` and `wrap` AST nodes into a literal namespace. They also compare every
other AST node and all original caps for equality. The patched fixture is not
a replacement scientific launcher or a new numerical allowance.

Executed from physical D:, with TEMP/TMP/TMPDIR at D:\Dev\_Temp:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/joint-reward-observer-cleanup-20261031/run_checks.py first
```

**12 methods / 41 literal cases passed,0 failures/errors/skips** on the first
correctness invocation: .1515658s, process peak working set28774400B, PID71604;
host .3136473s/exit0.31 underlying callbacks included27 close callbacks;
1888 terminal-JSON bytes. Native models, training, random training and simulator
work were zero; no project/native imports were attempted. The import blocker
recorded zero blocked attempts. The original cleanup failure is an expected
negative fixture, not a failed new test. No second correctness run was needed.

Cases cover the last permitted close and next refusal; failed-close charges;
multiple exhausted work categories; fabricated/mixed charges; nested learning
and physics refusal; unchanged readout-training prohibition; original non-close
success/failure traces; and cap-trip→bounded close→terminal JSON retaining the
original error, exceeded counter, false budget flag and null outcomes/proofs.
The terminal JSON case is literal correctness, not a recovered native capture.

Scoped format/lint/mypy passed on the two new Python fixture files. First static
call1.7706729s retained two AST-list annotation diagnostics; the explicit
`list[ast.stmt]` annotation fixed both. Final call .572597s/exit0. No ignored
diagnostics or production-validation weakening. Executed final static commands:

```powershell
$taskStage = 'artifacts/runs/joint-reward-observer-cleanup-20261031'
.venv/Scripts/python.exe -B -m ruff format --check "$taskStage/test_cleanup.py" "$taskStage/run_checks.py"
.venv/Scripts/python.exe -B -m ruff check "$taskStage/test_cleanup.py" "$taskStage/run_checks.py"
.venv/Scripts/python.exe -B -m mypy --cache-dir "$taskStage/mypy-cache" --follow-imports=silent "$taskStage/test_cleanup.py" "$taskStage/run_checks.py"
```

Outputs/timings are in `static-first.*` and `static-final.*`; the local cache is retained within
the original4MiB stage allowance. The failed native8GiB allowance is unchanged.

Lead source/procedure/result self-review and independent component review are
recorded separately. Independent scoped component ACCEPT used two complete reads:
2904 tokens/.1581466s and2990 tokens/.2126426s, total.3707892s. The finite exception
ended without tests, imports, model work, replication or allowance carryover.
Receipt is in stage `REVIEW.md` and `reviewer-timings.json`; no native GO is implied.
Skipped full suite, native source retraining/game rerun, Linux, clean clone,
wheel/install, browser/device and release checks because this is an isolated
observer-function repair. No learner/default or production runner changed.

This completes only the scoped prospective cleanup correction.
It does not reduce the serialization cost, restore discarded scores, prove
all-owner native cleanup or qualify the full18-game comparison. No retry,
cap renewal, reconstructed snapshot, favorable seed/dose/metric or final-data
selection is authorized. Full390/parents18/21/23 and the repository goal stay open.

Return to a substantial deliverable after this correction: inspect whether the
current offline installed affine command includes the accepted all-old-task
workflow, then qualify that public delivery path without repeating its native
learning study. Source-checkout task usefulness and package correctness remain
separate; do not automatically start another observer/counter increment.
Bounded orientation already found the source `all_old_tasks=True`/`--all-old-tasks`
route and its accepted known-task8/8 joint result. Its qualification explicitly
skipped current wheel/install. Normal metadata lookup sees checkout egg-info;
isolated `python -I` finds no project installed in this checkout venv. That is
a concrete delivery check to perform, not evidence that other historical
installations are missing or a reason to rerun games.
