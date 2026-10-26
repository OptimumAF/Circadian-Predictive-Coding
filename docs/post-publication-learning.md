# Fresh learning after a guarded publication

The existing native ticket runner has an explicit two-window engineering mode:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --continue-training-after-publication --output continued-serving.json
```

Use a fresh output path. The qualified source is required locally; the command
refuses changed source bytes. No download or source refit is performed. Original
single-window modes and guard criteria retain their behavior. This mode cannot
combine with fitted controllers, comparative recipes or the outer/final study.

## Behavior and boundaries

`FollowupTrainingPlan(first_batches=8, start_tick=24)` is fixed before work. The
first eight six-row TRAIN batches teach the new contexts; eight original old-task
return batches begin at tick24. The fixed periodic2 policy, delay1, 48 live
opportunities and contention ticks2/6 are declared in advance. All16 update
allowances, inbox capacity and48 decisions belong to one original runtime,
budget, clock, sharing gate and action envelope from construction onward.

The first publication requires eight completed native receipts, every first-window
label consumed, no pending predictions and an idle joined worker/gate. The
original guards/prepare/commit and exact typed receipt checks decide publication.
Refusal, lost return or inconsistent generation prevents second-window learning.
There is no fallback retry, renewed budget or outcome-selected start time.

After verified publication, old feedback remains in the invalidation record with
its original actor version. New predictions record the actual observed version;
new labels retain that version. Current verified generation is checked before
observations, after predictions and before admission. Stale/mixed versions and
non-TRAIN sources refuse. The finite plan is revalidated, including exact types,
before further work. Incomplete runs retain applied receipts, pending indices,
original charges and action/sharing progress.

The original private learner keeps its full native optimizer history and owner
ancestry. Its candidate ID remains `c0`; the first deployed version is also named
`c0`, but these are separate owned models with explicit native update clocks.
The serving model stays at the first published snapshot while the private model
continues training. There is one publication; second promotion/rebasing is not
implemented by this option. Original full rollback remains available.

Why this: the inbox already pairs observation/label versions. Declaring both
windows up front allows fresh feedback through existing ports without another
runner, a budget reset, or relabeling old evidence. There is no final/probe/replay
access or new learning rule. Live outcomes do not select updates or guards.

## Executed evidence and limits

The sole frozen Windows case passed first launch. First publication at tick20
deployed native104/624rows; fresh sources at24..31 and labels25..32 produced
updates9..16 at26..40. Private state reached112/672rows while every later live
request still used deployed104. Full rollback restored native96/576rows.
Source and caller NumPy/Torch RNG were unchanged;16SGD96rows were paid once.

The live trace completed48 episodes with40 successes. Legacy failed before
publication and succeeded afterward; three revoked parcel commits refused.
The frozen private readout solved all four known ordinary tasks. Old tasks were
already retained at publication, so the extra old-task updates establish neither
recovery from forgetting nor an incremental utility gain. These repeated tiny
templates establish no statistical/generalization, controller, efficiency, CPC,
safety or release advantage. Instrumented150 live calls: p50=15.61ms,
p95=21.03ms, max=22.63ms; this is not a latency SLO or controlled comparison.

125 distinct scoped pure checks/platform pass on Windows and cached WSL, plus
scoped Ruff/format and fresh default+Linux mypy. Source/procedure/result review
is single-agent self-review, not independent acceptance. Native math/baseline,
full repository, clean-clone and release suites were deliberately not rerun.

```console
python -m pytest -q tests/test_post_publication_learning.py -k "not native_fixed"
python -m ruff check scripts/ticket_serving_timeline.py scripts/run_bp_ticket_serving.py tests/test_post_publication_learning.py
python -m ruff format --check scripts/ticket_serving_timeline.py scripts/run_bp_ticket_serving.py tests/test_post_publication_learning.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/ticket_serving_timeline.py scripts/run_bp_ticket_serving.py tests/test_post_publication_learning.py
```

Full commands, budgets, failures, source pins and cost ledgers:
`artifacts/runs/post-publication-learning-20261023/RESULTS.md`, issue66 and
ADR-0582. R7.7 long-duration/poisoning/broader utility and R7.5 remain open.

Extension: retain the lifetime owner/envelope and exact source/label version
pairing when adding a new predeclared recipe. Any second publication needs an
explicit ancestry/guard contract; do not infer it from successful private training.

## Explicit clean or noisy second window

The optional `--followup-training-recipe clean` or `noisy` changes only the
second window. It requires `--continue-training-after-publication`:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --continue-training-after-publication --followup-training-recipe clean --output clean-followup.json
```

Both recipes retain the original first eight six-row new-context packets and
their delay1 labels. At tick24, eight twelve-row packets combine the same new
contexts with the retained old contexts, with delay2 labels. IDs, features, row
order and opportunities match between recipes. Only the noisy recipe changes
retained invoice row7 from `inspect` to `defer`, explicitly marked experimental
false labels in `followup_training_corruption`. These labels never replace the
original promotion guard truth. Each arm pays16 updates/144 TRAIN rows from
the original lifetime budget; private history reaches112/720 if complete.

Recipes and owned TRAIN arrays are checked on the owner and worker boundaries.
Non-TRAIN data, changed payloads or stale actor generations refuse. The deployed
actor stays at its first guarded104/624 snapshot through subsequent private
learning. Omit the option to retain the original old-task six-row second window
and delay1 behavior. Original single-window modes remain available.

Why this: a separate port preserves the first window and original defaults;
combining the older live recipe flag would replace both windows. This is a
bounded feedback-noise experiment, not a general poisoning defense. Results,
physical work and limitations are recorded in issue92 and
`artifacts/runs/followup-training-recipes-20261023/RESULTS.md` (ADR-0608).

The sole frozen pair completed: clean private tasks4/4, noisy3/4 (password
failed), with invoice/parcel retained in both. Both deployed actors completed
40/48 live episodes from the identical first publication. Each paid16 updates/
144rows; the pair paid32 SGD/288rows in26.376651113s with350572544B process
peak. Source/RNG/full rollback and causal clocks passed;59 focused checks on
each platform and finite independent source/procedure/result review passed.
These are repeated known-template outcomes, not an estimate of general noise
robustness. Serial startup/residency makes the two measured run times unsuitable
for a speed comparison. No second publication or deployment of the noisy state.
