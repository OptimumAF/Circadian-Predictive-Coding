# MiniGrid chooser learning composition

The ordinary fresh A4 command exposes the existing fitter:

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-controller retain_arrived --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-controller retain_arrived --output fresh-controller.json
```

Use `current_only` instead of `retain_arrived` for the original alternative.
`--describe` performs no collection, source reading or fitting. Execution uses
the optional MiniGrid dependencies, reads bundled A3 TRAIN and collects fresh
A4 TRAIN. It creates the original zero-update/replay150-second/512MiB readout
owner before preparation, then passes it to the existing fitter. It does not
accept saved sources, fit overrides, feedback switches or another mode.

The Python API also supports one separate-source TRAIN nuisance recipe:

```python
from scripts.minigrid_controller_learning import run_controller_learning

report = run_controller_learning(
    original_grey_a3_a4_train_pairs,
    readout_budget=original_readout_owner,
    retain_arrived=True,
    validation_recipe="nuisance_redwall_delay2",
    acknowledge_correct=True,
    share_readouts=True,
)
```

The caller must start the original zero-update/zero-replay150-second/512MiB
owner, with an attached RSS sampler, before source preparation. The new recipe
adds no budget or renewal. It completes the original four grey A3/A4 fitting
trials and freezes the chooser before collecting two fixed red-wall A3 TRAIN
episodes (blue/red, seed17/max16). Both validation domains use A3 branch3;
wall color is grey versus red. No final/probe layout is converted to TRAIN.

Five independent policy windows receive red-blue/grey-blue/red-red/grey-red,
delay2/six opportunities/four update credits. The old fitting episodes remain
separate and unchanged. A fresh warm-model baseline in grey/red A3 and all
candidate game endpoints are read only after every validation training window
finishes. The earlier grey A3/A4 fitting baseline cannot substitute for this one.
Endpoint group keys are `old_A3` and `shifted_A3`. They do not imply new task
semantics or recovery. Endpoint outcomes never enter chooser fitting or tuning.

Readout sharing remains optional and bounded to six complete states for this
invocation. A reused nuisance endpoint has `utility=None` and `measures=None`,
including baseline reuse when state sharing is disabled. Physical calls and
logical reused slots stay explicit. Failed collection/readout, unknown returned
work and the last paid evidence are retained; corruption stops the invocation.
The new recipe is Python-only; existing CLI/known-recipe behavior is unchanged.
[Software checks, costs and limits](reports/minigrid-nuisance-validation-20261027.md).
No native nuisance experiment or generalization claim is established by those
scripted checks. Freeze source, collection, physics, fitting and evaluation
costs before any new native observation; earlier failed budgets remain spent.

The JSON preserves full warmup, four discarded trials, frozen fit, application,
source provenance, bank/diagnostic/game costs and failed/unknown work. `summary`
projects the original choices/estimates, reported completed updates/rows,
outcomes and unfitted packets. It computes no new utility. A complete window
can leave correct packets unfitted; identical-state readout reuse is explicit.
No deployed actor is changed. A child call without a return is marked unknown,
and a later capture error keeps overall `summary.complete` false.

Output is compact finite JSON, at most8MiB including its final LF, created
exclusively without overwriting. Every report field is retained.
An existing path refuses before work; a creation race preserves its bytes.
Failed workflow reports are emitted/saved with exit2. Do not retry numerical
work automatically after an output/serialization failure. The existing fresh
default, fixed reference and other command modes are preserved.
[Public-command software qualification](reports/a4-controller-cli-20261026.md)
uses scripted outcomes; the earlier native result below qualifies the shared
composition. The first [native public invocation](reports/a4-controller-cli-native-20261026.md)
completed learning but failed its output size limit; that attempt is spent.
The [lossless encoding repair](reports/a4-controller-cli-serialization-20261026.md)
passes37software checks per platform and preserves the8MiB limit. The repaired
public path has not been qualified by another native invocation; no automatic
retry or broader reliability is implied.

The Python API also supports a distinct schedule validation mode:

```python
result = run_controller_learning(
    episodes, readout_budget=original_owner, retain_arrived=True,
    validation_recipe="interleaved_delay2",
)
```

The caller must create the original zero-update/replay150s512MiB owner before
preparing inputs. This mode keeps the original four-trial fit and replaces its
known-pattern application with five isolated windows: fixed no-learning,
periodic, drift, budget-aware and the same frozen chooser. All share
A4blue/A3blue/A4red/A3red, delay2 and six opportunities. It reads validation
endpoints only after all windows complete, without draining pending packets.
These are known TRAIN games on a different schedule; recovery is unmeasured.
Default budget-aware reserves all four credits, so it may perform no updates.
The [software qualification](reports/minigrid-controller-validation-20261026.md)
covers69cases per platform with literal learners/scripted games. One
[native schedule observation](reports/minigrid-controller-validation-native-20261026.md)
found fitted and drift each learned1/2new cue tasks and retained2/2old tasks;
periodic had the same endpoints. Correct queued work blocked the later new cue.
No chooser advantage, unseen-task generalization, tuned-heuristic comparison or
net benefit is established. This option grants no experiment budget or retry
of failed209.

For explicit schedule validation, add `acknowledge_correct=True` to the Python
call to permit proved-correct TRAIN consumption without an optimizer update.
DefaultFalse retains the original waiting behavior. Each request performs a
fresh independent read-only prediction under the existing candidate lease;
previous zero-mistake feedback is not authorization. At most one fit or
acknowledgement happens per opportunity, without draining or renewed credits.

The report separates `consumption_complete`, `training_complete` (all packets
actually fitted), acknowledgements, pending/unfitted packets and proof costs.
Readout geometry follows actual committed cue receipts when packets are skipped.
[Software qualification](reports/correct-ready-window-20261026.md) covers197cases
per platform; its scripted learner reaches both new cues and retains both old
cues. The subsequent [native acknowledgement observation](reports/correct-ready-window-native-20261026.md)
completed every training window: periodic and drift acquired both new cues and
retained both old cues. The run exceeded its original150-second body limit during
the fitted chooser's final readout, leaving that outcome and final global checks
unknown. It is failed and spent; partial results do not establish a completed
comparison or chooser advantage. The earlier partial result above is preserved.
No CLI flag, retry or additional experiment budget is added.

For that validation API, optional `share_readouts=True` reuses a preceding
successful endpoint only when its complete native state and readout arguments
match. All five training windows still finish before validation endpoints.
The original warm-baseline reuse remains. The local cache retains at most the
warm source plus five candidate entries and is discarded on return or failure;
there is no persistent or cross-run cache. Different parameters, fit diagnostics,
policy, counters, failure state or readout arguments require a fresh readout.
Original source, paid-result identity/value and lifetime-budget guards apply
before and after reuse; corrupted evidence refuses continuation.

`reused_from` identifies the original physical readout. Each endpoint's
`readout_work` separates physical four-game calls, four logical episode slots
and reused slots; `readout_seconds` includes lookup time. Every reused endpoint
has `measures=None` and `utility=None`, including the shared baseline, to avoid
manufacturing a cheaper utility ranking. Physically new readouts retain the
original objective and timing/RSS limitations. Training and controller fitting
receive no endpoint feedback. DefaultFalse retains the original schema and
readout behavior. [Software qualification](reports/minigrid-readout-sharing-20261026.md)
uses literal fits and scripted game IO; native savings and complete fitted
outcomes remain unmeasured. This option grants no retry of the spent native run.

`scripts.minigrid_controller_learning.run_controller_learning` connects the
existing four-trial `CounterfactualController`, `AttributionStream` and affine
game evaluator. It accepts the four original typed A3/A4 blue/red TRAIN episodes
in that order. It does not read source files or provide an experiment allowance.

The caller must start and attach the existing process sampler to a
`ToyBudgetSession(ToyExecutionBudget(max_training_updates=0,
max_replay_examples=0, max_wall_seconds=150,
max_process_rss_bytes=512 * 1024**2), monotonic)` **before preparation**.
Then pass those same episodes and owner:

```python
result = run_controller_learning(
    episodes, readout_budget=readout_budget, check_boundary=check_original_inputs
)
```

This call performs real training and games when using ordinary adapters. Freeze
its data, protocol and cumulative native limits before invocation. Software
qualification uses literal adapters and supplies no native result.

Two current-only delayed A3 fits create an owned warm state with native counters
2 updates/18 rows. Four disposable trials start from that identical complete
state: clean wait/wake, then mistakes wait/wake. Each owns four new update
credits, a 90-second clock and the same process RSS sampler. Clean uses A3/A3;
mistakes uses A4/A3, with four packets and five opportunities. No warm history or
native counter is reset. The first A3 blue and A4 blue predictions must actually
produce the declared contexts; unavailable contexts fail without a search.

Why this: closed-loop task success supplies the utility, while prediction
mistakes identify only the TRAIN context. New-task gain and old-task drop use
complete four-game before/after-task outcomes, the unchanged objective, measured
whole-trial time and cumulative sampled process RSS proxies. After-return goals
and recovery remain separate. Identical complete states can reuse the common
baseline or an earlier within-window readout, explicitly labeled; equal weights
with different native history cannot. Each window retains at most three readout
states. These cost proxies and asymmetric readout reuse do not establish fair
total compute or amortized controller benefit.

After four valid returns, the same frozen owner is applied through
`run_contextual_window` to A4/A3. Oldest-ready packets, causal labels, partial
training and pending work retain their original meaning. A policy that waits
does not drain the queue. This is known-TRAIN development and application;
there is no controller validation, final release, unseen-task or CPC claim.

The report separates warmup, contexts, discarded trials, fitted estimates,
application, each original budget, native versus relative checkpoint counters,
readout reuse and unknown/failed work. A complete window can have incomplete
training. Prior successful evidence, source and budget controls are checked
before subsequent procedural work; corruption or equal container replacement
fails and preserves the last as-read completed prefix, without authorizing a
retry. Returned reports are evidence, not resumable execution authority.

## Optional bounded TRAIN bank

Pass `retain_arrived=True` to `run_controller_learning` (or its description)
to reuse the existing one-prior-packet-per-cue composer. The default remains
current-only and the negative current-only observation remains unchanged.
Two actual warmup fits still cost18 rows. Each trial/application prepares the
same six-arrival plan: the two completed warmup packets, then its four local
packets. Only the four suffix batches enter the new window. Warm templates are
allocated but never fitted again. Each selected new fit now uses9current and
9earlier same-cue TRAIN rows. Its complete native counter is2+n updates and
18+18n rows, admitted by the existing joint reference evaluator, including odd
prefixes with one9-row and one18-row head. No native state is reconstructed.

Warm labels retain their already-observed warmup arrival coordinates. New
labels retain local delay1; the report records both epochs explicitly. Each
trial/application starts an independent bank from the same warm sources; no
discarded trial bank leaks into another trial. Outer context diagnostics score
the original9current rows; window ready feedback scores its actual joint18
rows. Comparisons must account for that difference in observation and compute.

Each window's `catalogue` separates preparation attempts/returns/unknown work,
the planned six-wake source spans, bytes/peak-bank equivalents and the verified
committed current/prior/native exposure counts. Preparation includes unused
warm templates and unselected packets; waiting costs preparation but0fit rows.
Budgeted-but-unverified commits remain separate. Prior rows are joint native-fit
exposures; they do not invent separate consolidation events or increment the
consolidation replay counter. The original four-update budget and clock start
before plan preparation. Damaged or replaced planned evidence is refused and
its as-read record retained on failure.

The [bank software report](reports/minigrid-controller-bank-20261026.md) qualifies
this composition with literal outcomes; it does not establish native benefit.
Use the same focused commands with `tests/test_minigrid_controller_bank.py`
alongside the original29case file. No new dependency or learning algorithm is
introduced. Freeze native data, limits and analysis before any real bank study.
The [native bank observation](reports/minigrid-controller-bank-native-20261026.md)
subsequently selected wait/clean and wake/mistakes. Its private application
acquired both new cues and retained both old cues with2joint fits/36rows
(18prior). Including warmup and discarded trials, the run paid12fits/198rows.
This is known-TRAIN usefulness, not independent acceptance, a comparison that
isolates controller benefit, generalization or a CPC advantage.

Run focused checks from the repository root with ordinary test imports:

```powershell
$env:PYTHONPATH = ".;tests"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -B -m pytest -q tests/test_minigrid_controller_learning.py
python -m ruff check scripts/minigrid_controller_learning.py tests/test_minigrid_controller_learning.py
python -m ruff format --check scripts/minigrid_controller_learning.py tests/test_minigrid_controller_learning.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/minigrid_controller_learning.py tests/test_minigrid_controller_learning.py
```

For the public command, run `python -B -m pytest -q tests/test_a4_controller_cli.py`;
use the same Ruff and mypy commands with `scripts/run_minigrid_linear_fit_control.py`,
`scripts/minigrid_a4_adaptation.py` and that test file. The measured finite
Windows/Linux commands and resource receipts are in
`artifacts/runs/a4-controller-cli-20261026/`.

For the measured finite qualification, use the commands and receipts in
`artifacts/runs/minigrid-controller-learning-20261026/PROTOCOL.md` and
`RESULTS.md`. [Scoped report](reports/minigrid-controller-learning-20261026.md).
The [native observation](reports/minigrid-controller-learning-native-20261026.md)
completed after one paid runtime-selection failure and a prospective correction:
the current-only learner forgets/relearns, all trial utilities are negative, and
the fitted chooser waits in both contexts. It demonstrates no controller benefit.
Extend through the existing stream/evaluator ports; freeze any new comparative
or withheld protocol separately and retain setup, discarded and diagnostic costs.
