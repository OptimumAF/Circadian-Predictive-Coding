# Native old-TRAIN relearning and F interference

R2.1-A4.R.N / [issue222](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/222).
One fixed native run of the ordinary221 API recovered old-TRAIN fit, while
losing the additional F fit. Source/procedure/result SELF-REVIEW, not independent
acceptance. This is a fit/relearning result, not demonstrated game recovery.

## Observed outcome

The sole call was `run_fresh_adaptation(reference_policy=ReferencePolicy("drift"),
retained_probe=True, return_old=True)`. The actual source completed4updates/
54rows, acquired known A4 and retained A3: native old/new goals2/0→2/2 on two
fixed cues. Logicaltick6 reused the unchanged4/54 endpoint/readout, not extra
training. The causal window completed with two unapplied packets; the six-wake
90-row plan was not fully trained.

The same single private fork then fitted two nine-row F batches, reaching6/72,
and fitted each original18-row old-TRAIN cue template once, reaching8/108.
All checkpoints were fixed before execution, with no metric-based selection.

| Private phase / relative updates | Old TRAIN correct /36 | F correct equivalent /18 |
| --- | ---: | ---: |
| Before F /0 | 36 | 14 |
| F /1 | 34 | 16 |
| F /2 | 31 | 18 |
| Return /0 | 31 | 18 |
| Return /1 | 33 | 16 |
| Return /2 | 36 | 14 |

Old TRAIN recovered five correct decisions. Final A3blue/A4blue/A3red/A4red
were each9/9, with their original approximately1.165422CE restored. During F,
they had fallen to8/9,8/9,8/9,7/9. F blue/red CE changed
1.277968/1.273307→1.165422/1.165422→1.277968/1.273307. F counts in the table
are the original per-cue accuracy multiplied by its fixed nine rows; no
predictions or losses were rescored. Old correct counts are original readouts.

This is a tradeoff, not joint retention: old fit returns and F fit loses four
correct decisions. F training/test use the same known geometry and seed.
The original source and actor remain unchanged at4/54. Paid F6/72/two9-row-head
state is separately retained from the actual final8/108/two18-row-head state.
No deployment, benchmark update, promotion, final labels, consolidation replay
or new controller/CPC mechanism. The return uses explicitly authorized old
TRAIN labels; probe_test labels remain readonly. No F game or post-return old
game was executed, so native task recovery and generalization remain unknown.

## Paid work and resource scope

The first and only child/checked-parent invocation exited0. All original-port
attempts returned; no lost native work, retry, timeout or budget revision.

| Native work | Fits | Exposed rows |
| --- | ---: | ---: |
| Source | 4 | 54 =36current+18joint prior |
| Private F | 2 | 18 |
| Old TRAIN return | 2 | 36 |
| Total, also8LAPACK solves | 8 | 108 |

All36return rows are old-TRAIN relearning, not18replay rows inferred from their
batch length. Phase counters followed the original budget progress. No separate
consolidation events/replay-budget charges occurred. The packaged A3 source
production is historical/sunk, not a newly collected teacher trace; one94705B
read decodes the release before selecting original A3 episodes.

Other paid original ports:15empty native constructors,1931snapshots,
116predictions/500rows,14environments/14closes,140physics steps. The14environments
comprise two fresh A4 teacher episodes, eight physical source games and four F
teacher episodes. Steps are18+86+36. The return adds no teacher, environment or
source read. F and return each use six F-scoring calls/54rows plus six old-TRAIN
monitor calls/108rows:24calls/324rows together; source calls cover the remaining
176rows (90ready-diagnostic+86game). No random parameter draws.

Diagnostic body9.0008878999s, parent9.3531408000s. The parent also bounds complete
child startup/final bookkeeping below150s, within its180s cap. Whole helper
F+return diagnostic2.5166486001s; its F budget's2.4782788001s elapsed spans the
helper including return, not isolated F compute. Return-only peak RSS is
unavailable; no fabricated phase peak or per-fit efficiency comparison.

Sampled process peak94,740,480B was recorded after both raw and scalar report
serialization/writes and the sampler's final exit observation. Its scope starts
before optional imports. The final3000B accounting receipt/console follows the
sampler and is explicitly excluded from this RSS measurement; parent timing
includes it. This is sampled current-process RSS, not instantaneous peak,
process-tree/allocation attribution or a whole-system guarantee.

The four original owners retained exact bindings, starts, limits and permanent
charges: whole0/0/150s, source6/None/90s(actual replay0), F2/0/90s, return2/0/90s,
all512MiB. Return owner was born before F, not renewed at return. Source TRAIN
owner was sealed before later games/probe; F clock progress used the existing
guarded observer. Global Python/NumPy RNG and all instrumented bindings passed
final preservation checks. Original source, actor, paid F report/state and
return-input guards remained active. Return0 equals preceding paid F/old scores.

Retained input arrays449712B; the short-return input witness adds213840B.
These are scoped data arrays, not every state/report/native copy or FLOP.
All prospective8fit/108row/8solve/512prediction/2048row/18environment/288step/
256owner/40000snapshot/time/RSS ceilings held.

## Qualification and evidence

Native command, executed exactly once:

```text
.venv/Scripts/python.exe -B artifacts/runs/short-affine-return-native-20261026/execute.py
```

Stage `artifacts/runs/short-affine-return-native-20261026/` retains PROTOCOL,
frozen producers, qualification22pins/runtime/data metadata, all gate receipts,
native-start/parent/work, emitted terminal-summary and result review/command.
The parent uses218 same-call metadata validation and subprocess(check=True);
no failed gate was ignored. Installed Windows Python3.14.7/NumPy2.4.6/minigrid
3.1.0/gymnasium1.2.3/pygame-ce2.5.7/cloudpickle3.1.2; numerical thread limits1.

Reused22156distinct checks/platform after all eight current source pins matched.
Solver/learner/budget/simulator/sampler remain clean to HEAD. New two-file final
Ruff format/check and default/Linux-target mypy pass. Two pure projection checks
per Windows/cachedLinux passed: no optional imports, native model, label read,
physics or marker; raw/teacher/weight arrays excluded and metric endpoints kept.
Windows.3091804sbody/.8633485sparent, Linux.221954107s/7.5883822s. The failed first
Windows fixture expected two predictions but its poison dictionary overwrote
them with one; explicit final fixture assignment fixed it. Failure.5770729sparent,
body unknown/upper-bounded by parent, preserved; no producer behavior changed.

Initial mypy had one None-narrowing diagnostic; typed local elapsed/peak fixed
it. One subsequent format check failed a multiline conditional; the formatter
fixed it. All pre-native failures/commands are in gate-receipts.json. No ignore,
gate weakening or native launch on failure. All static attempts12.6587036s <60s,
touched existing cache111161344B <160MiB; inherited global cache separate.

Raw actual.json1222244B is compact strictfinite UTF8+LF, exclusively written
once and **never reopened**. Its hash is the producer's in-memory hash, not a
later file read. The allowlisted17422B scalar/metric projection contains no
teacher transitions, observations, weights or prediction arrays. Final work
receipt is3000B. Postflight read only these emitted scalar records, qualification
and source hashes; exact assertions pass in.015706s internal/.411785s wrapper.
They verify22pins, roles/order/seed, original resources/counters, native work,
phase conservation, return0 continuity, output allowlist and all declared caps.
No model reconstruction or new scorer. No product code changed this session.

Original217 and220 failures/spent budgets remain unchanged, including217's
failed setup/false GO and220's unqualified capture RSS/projection scope. This
new return question does not repair or requalify either case. Combined costs
217+220+222:14fits180rows,30environments298physics,3TRAINreads284115B,
22.1608187000sbody23.5657884001sparent. Failed work is not forgiven or called
successful; no unused prior quota carries forward. Other negative/failed
controller, baseline and scientific results remain preserved.

Skipped unchanged suites, another Linux/native cohort, fullsuite, cleanclone/
wheel, F/post-return games, comparison, independent/browser/release checks.
No installs/downloads/GPU/Torch/agents/CI/Git mutation/sweeps/old raw reader.
Historical plans/log remain frozen under AGENTS; live owning issue/tracker hold
handoff. Full R2.1, reliable native acquisition/retention/recovery, comparative
science and release parents stay open.

## Next useful implementation

Fit recovery cannot establish task recovery. The original game loop and guarded
endpoint observer already exist, but the F port admits retained4/54→6/72 and
the matched rehearsal-return port requires8/180/head36. The new final8/108/head18
cannot be silently passed as either older recipe.

Next action: add explicit short-return phase admission for paid6/72 and8/108
to the existing readonly native evaluator, covering F plus the four original
A3/A4 TRAIN tasks. Compose it through the existing private endpoint observer
and original whole zero-update/150s owner, with source/native-state preservation
and paid game-prefix/cost checks. Start literal same-run/role/failure tests.
No new training recipe, optimizer, runner, checkpoint chain or historical-state
loader. A later native game observation needs its own frozen allowance; this
case grants none and will not be resumed or rescored.
