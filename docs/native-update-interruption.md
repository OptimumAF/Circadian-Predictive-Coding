# Native update interruption

The bounded engineering example interrupts a registered native worker after it
computes and encodes one admitted update but before it delivers checkpoint B.
The surviving Windows parent continues serving actor A and terminally refuses
uncertain work without refunding admission, copy or checkpoint-attempt charges.

```powershell
python -m scripts.run_coupled_update_interruption --describe
python -m scripts.run_coupled_update_interruption --directory artifacts/my-new-interruption-example
```

Use a new dedicated directory. Description performs no operational IO, model or
process work. Actual execution is Windows only, CPU1, three sequential workers
(PC, neutral CPC and active CPC). Nine training calls/36 energy evaluations include
six warm-up calls before authority creation and three admitted child updates;
twelve actor predictions span pending work and terminal refusal. Frozen seed23,
D2/H(3,2), precisions(1,1), training rows and rates appear in the description.
No evaluation data, comparison metric or learning-benefit claim is involved.

## What happens

1. Prepare and save complete checkpoint A, then start and retain the physical
   child registration. Bind the existing coordinator and SQLite authority.
2. Persist update1, copy262144 and checkpoint-attempt1 before sending A.
3. The explicit worker flag `--pause-before-result` computes and bounded-encodes B,
   emits only `{event:"computed",training_calls:1}`, and waits for exact stop.
   It does not send B, its hash, energy or predictions. Stop exits without B;
   the default worker invocation keeps the healthy completion protocol.
4. While the worker is paused and observed live, serve A. Kill and reap only the
   owned child inside the preparation callback; observe ended through its retained
   registered handle. The post-preparation coordinator check fails before the
   checkpoint validator can run.
5. Require stopped/uncertain authority with admitted1/completed0 and unchanged
   manifest and charges. A second request refuses before either callback. No B
   file is created; complete actor state, A bytes and all four predictions match.

Why this: completed computation does not establish acknowledged completion.
The pause event is engineering control-flow telemetry, never a completion witness.
The parent does not guess B, clear uncertainty, hand off or retry the admission.
The terminal authority can use its last valid coordinator time/RSS stamp: worker
binding fails before a fresh coordinator sample. Separate later live/ended host
observations prove the controlled interruption and are reported distinctly.

## Structure and checks

```text
src/adapters/coupled_update_worker.py         opt-in pause; default healthy worker
src/adapters/coupled_update_interruption.py   native/Windows/coordinator composition
scripts/run_coupled_update_interruption.py   bounded JSON CLI
tests/test_coupled_update_interruption.py    independent protocol and native controls
```

The adapter composes existing codec, file store, bounded child transport,
retained Windows registrations and authority helpers. Core models, equations,
authority rules and completion validation are unchanged. No new dependency or
environment variable is introduced; thread settings use the existing defaults.

```powershell
python -m pytest -q tests/test_coupled_update_interruption.py tests/test_coupled_update_completion_cli.py
$interruptionFiles = @('src/adapters/coupled_update_worker.py', 'src/adapters/coupled_update_interruption.py', 'scripts/run_coupled_update_interruption.py', 'tests/test_coupled_update_interruption.py')
python -m ruff check @interruptionFiles
python -m ruff format --check @interruptionFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @interruptionFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @interruptionFiles
```

Actual process/native checks require a prospective local budget. Current evidence
is under `artifacts/runs/coupled-update-interruption-20261015/`; see its scope,
results and independent reviews. Extend recovery through explicit original
admission and complete managed state; never infer ownership or renewed quotas
from current files. Full managed-bundle recovery, actor promotion,
coordinator-loss recovery and long-run criteria remain unfinished.
