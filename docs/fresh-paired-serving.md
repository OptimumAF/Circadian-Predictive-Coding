# Fresh-source ordinary paired serving

Train a conventional separate-context perceptron source from zero, qualify its
four original tasks, then learn and serve the paired new tasks through the
existing guarded runtime. No historical checkpoint file is required.

From the physical repository on D:, with a new output path:

```powershell
$env:TEMP = 'D:\Dev\_Temp'
$env:TMP = 'D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m scripts.recurring_ticket_workflow_example --context-return --separate-context-heads --paired-serving joint --output artifacts/fresh-paired-serving.json
```

This explicit command performs training. `--paired-serving current` selects the
existing current-task recipe instead. Both require the original perceptron,
context-return, separate-head configuration and `--replay none` (default).
They are mutually exclusive with `--private-adaptation`. Invalid combinations
and existing output files are refused before work. Commands without these
options retain their previous behavior.

The Python API is `run_example(context_return=True, separate_context_heads=True,
paired_serving="joint")` in `scripts.recurring_ticket_workflow_example`.
The command uses the existing 4 MiB bounded report writer. Failure summaries
retain available producer costs; an unreturned delegated result has unknown
costs, never an invented zero or continuation authority.

The source pays 12 TRAIN passes and 72 row exposures. All four ordinary source
tasks at checkpoint 12 must complete successfully before child construction.
Its original 12-update, 30-second, 512 MiB session remains active during child
serving. The child separately pays six ordinary TRAIN passes and 48 row exposures
under its original 60-second, 512 MiB session. Neither session is renewed.
The same memory sampler covers both; the source deadline can expire first.

The original source stream stays open until the child finishes. Shared source
qualification checks native state, source receipts, checkpoint/summary identities,
task roles, actor, RNG and original budget witness. Worker and owner observations
serialize access to that witness. Changed or replaced evidence and unobserved
clock, limit or counter changes are refused. The source and benchmark actor
receive no child updates. Source readout roles remain `outer_selection`; their
labels never become child TRAIN data. No private probe weights or final labels
enter training or publication.

The child reuses the [paired learning boundary](paired-learning-boundary.md),
including ordinary delayed TRAIN, at most two guard decisions, at most one
publication, fixed worker tasks and full-state rollback. Further candidate
training does not implicitly change a published actor. Returned provenance is
`current_workflow_train_checkpoint12`, with separate source/child work and no
historical source SHA claim.

The [executed qualification](reports/fresh-paired-serving-20261029.md) passed its
source, new-task, retention and fixed late-worker checks. It is one known-template
product run; joint retention did not require recovery from forgetting. It does
not establish reliability, held-out generalization, a circadian advantage, matched
total computation or installed delivery. A separate [offline installed
qualification](reports/fresh-paired-installed-20261029.md) now passes in clean
Linux Python3.12.3/NumPy1.26.4. Its new wheel contains this option; the older
private-workflow wheel remains unchanged. From a standalone D: directory with
that wheel installed, use `python -I -B -m scripts.recurring_ticket_workflow_example`
with the same flags and a new output path. Other installed platforms and full
release qualification remain open.

Focused developer checks (model fixtures incur costs; spent experimental cases
must not be repeated merely to run these examples):

```powershell
.venv/Scripts/python.exe -B -m pytest -q tests/test_fresh_paired_serving.py
.venv/Scripts/python.exe -B -m ruff check scripts/context_head_rehearsal.py scripts/context_head_live.py scripts/recurring_ticket_workflow_example.py tests/test_fresh_paired_serving.py
.venv/Scripts/python.exe -B -m ruff format --check scripts/context_head_rehearsal.py scripts/context_head_live.py scripts/recurring_ticket_workflow_example.py tests/test_fresh_paired_serving.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent scripts/context_head_rehearsal.py scripts/context_head_live.py scripts/recurring_ticket_workflow_example.py tests/test_fresh_paired_serving.py
```

Extend through the existing source guard, TRAIN factory and runtime ports. Keep
new task recipes separate from probes and preserve original clocks and charges.
Why this: sharing the already-qualified in-memory source removes a concrete
delivery dependency without adding another training or orchestration framework.
