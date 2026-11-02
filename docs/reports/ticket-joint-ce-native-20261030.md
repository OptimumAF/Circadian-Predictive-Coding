# Native joint TRAIN readout solves six workflows

R2.1-TW.CE.JOINT.N / issue338. Previous337 was progress: output-only SGD retained
the four old tasks but acquired neither fresh task at the fixed dose.333 already
proved affine separation on reference joint TRAIN features. This qualification
tests the existing native CE optimizer and typed execution, without repeating
that LP, changing the SGD dose or using its witness coefficients.

## Useful behavior observed

One fixed native fit classified all18 joint TRAIN rows correctly, globally and
among allowed actions. Its frozen reader solved all six ordinary workflows.

| Task | Actions | Applied transitions | Outcome |
| --- | --- | --- | --- |
| v1 invoice | open / route / commit | ready / staged / committed | success |
| v1 password | open / inspect / commit | ready / staged / committed | success |
| v2 invoice | open / inspect / commit | ready / staged / committed | success |
| v2 parcel | open / defer / commit | ready / staged / committed | success |
| v1 parcel | open / route / commit | ready / staged / committed | success |
| v2 password | open / inspect / commit | ready / staged / committed | success |

All18 predictions returned and all18 effects applied. Every task used three
steps under the unchanged six-step bound, full capabilities and version-allowed
five-action catalogs, including phase-illegal actions. There was no phase mask,
target-fed decoder, fallback, rollout training or outcome-based task selection.
The source's original24 development episodes and final old four-task failures
remain in source.json.gz. No extra before-fit fresh-task rollout was performed;
fresh initial success is not assumed to be zero.

Final native float32 mean CE is0.000317386700771749; the existing summary from
saved logits computes float64 mean CE0.0003173899968078002. Both are retained.
READY allowed/global raw margins, in the table order:5.29931640625,
9.755859375,7.30419921875,22.60400390625,23.544921875,22.3974609375.
All18 global margins are positive. Raw logits, coefficients, all34 closure
losses, feature rows, targets and complete task observations/actions/transitions
remain in control.json. No coefficient rescaling or numerical tolerance changed.

This demonstrates native learned behavior on a known joint TRAIN cohort. The
fit already included both formerly source-untaught tasks and all old examples.
It is not sequential fresh acquisition, retention during new learning, recovery
after forgetting, held-out generalization, statistical reliability or CPC benefit.
It does not make337's chronological0/2 acquisition result successful.

## Existing implementation and isolation

No production code changed. Existing joint_old_fresh_teacher supplies the fixed
old4 then fresh2 original TRAIN templates, three phases each. Existing
fit_categorical_ce_readout already accepts1..128 exact released TRAIN cases.
The new stage adapts321's established actual.py only to use that teacher18,
JOINT_TRAIN_TASKS6 and the report scope. The observer differs only in declared
18-row/six-task caps. scoped.diff preserves those exact changes.

The existing ordinary run_example(context_return=True, method="bp") prepares
the original live source12SGD72rows, D9/H3/K6 seed17/lr.03/momentum.9/inference2,
A4/B4/A4. A passive observation of that existing stream retains its actual owner,
as in321; no new stream runner, bootstrap system or serialized parameter loader.
Complete source/checkpoint12/receipt/budget qualification precedes the fit.
Saved-only common-source.json then confirms all four complete checkpoints,
training data, source work and original limits exactly match337, before using
that historical source context. Different output-fit protocols are not treated
as matched controls. Saved checkpoints serve data equality only; this necessary new live
preparation is charged again.

One unchanged float32 LBFGS step fits only24 detached output coefficients over
the frozen30 hidden parameters, starting from original source output weights.
Settings remain lr1/max_iter32/max_eval48/history10/strong_wolfe/tolgrad1e-7/
tolchange1e-9/hard64closures. There is no witness warm start or alternate solver.
All six tasks run after computationally complete fit/work/source binding,
regardless of accuracy. The fit remains frozen throughout the typed readouts.

Original source, seed and actor are distinct complete owners. The inherited
guards check their identities, payloads, full parameters/gradients/optimizer
buffers, policies, versions,12 receipts, original budget/clock/birth/counters,
TRAIN cases/features/targets, RNG and prior task evidence before and after
callbacks, fitting, reader creation and predictions. Final unchanged() passes;
corruption is refused rather than overwritten. Source/actor flags are true.
Source NumPy/Torch RNG booleans are true; control preservation is evidenced by
live _rng equality guards and final unchanged(), not separate after-control RNG
boolean fields. No fitted weights are installed into source or deployed actor.
Original source capture bytes are rechecked unchanged at finalization.

Only original TRAIN definitions enter the fitter. Neither probe_test nor final
labels, task outcomes or witness coefficients enter training, replay, promotion,
tuning or selection. The old source replay counter remains0, but this does not
mean zero rehearsal:34 closures reread12old+6fresh rows each, paying408old
rehearsal and204fresh TRAIN gradient-row exposures. Memory/copy/guard overhead
is present in wall/RSS and bounded captures; copy bytes and total FLOPs are not
separately measured. No matched-compute or efficiency claim is supported.

## Commands, costs and limits

New stage/report paths were verified absent before creation. Stage:
artifacts/runs/ticket-joint-ce-native-20261030/. Frozen protocol is PROTOCOL.md
and issue338, before model work; prior captures remain untouched.
All commands use physical D:/Dev/Repositories/Circadian and process-local
PowerShell TEMP/TMP=D:/Dev/_Temp. WSL TEMP/TMP/TMPDIR=/mnt/d/Dev/_Temp.
Existing producer Python3.12.3/NumPy1.26.4/Torch2.14CPU, one CPU thread.

One native invocation, EXIT0; session94578 was polled once to terminal, never
restarted. Exact WSL argv and status are in shell.json:

```text
.venv/Scripts/python.exe -B artifacts/runs/ticket-joint-ce-native-20261030/launch.py
```

Native7.320168813s/338124800B peak RSS, host10.809408500004793s, within native60s/
768MiB and host75s. Original source30s512MiB birth5.352722680 and counters12/0
remain unchanged; last_clock advances8.980556797->10.016749142 through the fit
and readouts. Control15s512MiB starts before teacher preparation after source
qualification; measured control0.952532844s, including fit0.523353290s.
These overlapping times are not additive. Scientific limits were not extended.

Physical native-work.json counters, all frozen caps verified:

- 12native SGD/backwards72rows; one detached CE optimizer/step,34forward/backward
  closures612rows plus one final TRAIN read18rows. Total gradient rows684.
- 27components54named seeded draws1215values;153native head forwards213rows;
  159typed calls,30episodes,207sandbox dispatches including original source.
- 19hidden reads36rows: one fit18 plus18typed one-row reads.18fitted affine reads.
- Zero LS/LP/SVD/autograd.grad/random mutations/PC latent/stream replay/probe/
  final/robot physics/GPU/remote work. Auxiliary rehearsal is separately counted.

Exclusive launch/native SPENT markers precede work. No native failure, retry,
fallback, budget renewal or source clock reset occurred. A hardkill could leave
uncaptured work unknown and conservatively charged; no such kill occurred here.

Metadata-only preflight EXIT0/.070155812s/24391680B, no numerical/project imports.
Final Ruff check/format and Linux-target mypy on five stage scripts PASS; expanded
argv/output/times are statics.json/statics-final.json. Linux-target mypy runs on
Windows. Cumulative engineering commands37.507132200s within90s.
Saved-only analyze.py EXIT0 inspects all18 rows, all6 raw task traces, guards,
clocks and costs without numerical-library/project imports or new model work.

Supporting failures remain explicit:

- Two initial guessed reads requested absent321launch.py and
  scripts/ticket_policy_diagnosis.py. Bounded rg--files discovery replaced guessing;
  no dependency, archive reconstruction or tooling framework was introduced.
- All five initial commands passed, but the post-command cache gate FAILED:
  46403829B exceeded the agent-created32MiB limit. The cache and failure receipt
  remain. Prospective amendment6535632986 permits future engineering admission
  and growth to64MiB; it does not make the original overrun pass. Final cache
  46407909B fits the amendment;90s engineering and all scientific limits stay
  unchanged. ENGINEERING_AMENDMENT.md/cache-limit-failure.json retain both states.
- Review PRE read2 response was discarded during reviewer context compaction;
  its exact exit and time are UNRETAINED/UNKNOWN. Known PRE1/3 elapsed.4503786s;
  nominal10s charge for read2 is accounting only, not proof of actual total.
  Original30-second aggregate review-time compliance cannot be fully certified.
  No archive recovery or repeated PRE/model work was used to conceal that gap.
- POST read1 output was truncated. POST read2 checked all18 raw logits, then a
  reviewer-only extraction assumed a nonexistent candidates field and raised
  KeyError (.252476s). This does not change a scientific artifact. The final
  allowed read uses the actual selected-id/transition schema; no extra model
  work or unbounded review retries are authorized.

Accepted333 teacher tests,320 fitter tests and unchanged321/336 boundaries are
reused in their original scope. No repeated numerical fixture cohort, Windows
numerical case, four-family comparison, wheel/clean-clone/install, browser,
GPU/remoteCI, release or sweep was run. No production edits, dependencies,
global settings changes, deletes, commit/push or unrelated reset. Stage below
4MiB; static cache on D: under the amended64MiB allowance.

## Acceptance and next implementation

Lead self-reviewed the combined driver/observer diff, preceding live source
qualification, TRAIN-only inputs, all18 raw classification rows, all6 task
transitions, unchanged ownership/RNG guards, original clocks and paid costs.
Independent source/protocol and final actual-result/report ACCEPT. Six reads
were issued: five retained timings total1.1488778s, plus PRE2 with unretained
exit/time. POST2 has the recorded reviewer KeyError/EXIT1; final POST3 verifies
all6 actual transitions, saved-source equality and caps. Nominal total11.1488778s
includes the10s accounting charge for PRE2 and is NOT measured proof of aggregate
30s compliance. That supporting timing gap remains unqualified. The physical
REVIEW.md receipt was verified after its write. No numerical replication,
project imports, tests, models, solves, network or children; finite exception ended.

Native optimizer and known six-task execution are now observed, beyond333's
reference capacity witness.323's full-update failure,337's failed acquisition,
331's FAILED/SPENT capture and328's LS result/caveats remain unchanged. Broader
R2/R4/R5/R7/scientific/matched comparison/release requirements remain open.

Inspection found the concrete next delivery gap: ordinary --train-readout ce
still hardcodes the old12 teacher and four tasks in ticket_ce_tasks.py. The
qualified joint18/six-task behavior is currently available only in this local
qualification driver. Next precise implementation: expose an explicit default-off
joint TRAIN cohort through that existing ordinary CE-readout path/API/CLI,
using the existing joint teacher and fixed six-task order. Preserve old defaults,
strict TRAIN roles, original source/fit clocks, paid-prefix guards and completed
evidence ownership. First run focused admission/default/role/late-failure fixtures;
do not launch another native case until software gates and a new protocol pass.
This makes useful existing learning behavior accessible without a new runner or
another gradient-norm/capture increment; it is still not sequential adaptation.

Scoped disposition: native execution/results and independent scientific review
are complete. Issue338 remains open with the original aggregate review-time
qualification unchecked: PRE2 exact receipt is missing. Do not infer that gate
passed from the nominal accounting charge or rerun science to replace it. This
supporting gap does not block the next ordinary joint-cohort implementation.

## Tracking receipts

Issue338 remains OPEN solely for the unchecked original supporting review-time
qualification; native execution/results and scientific review are accepted.
Canonical https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/338#issuecomment-6535813926 .
Parent updates: #18 comment6535824093, #21 comment6535824468,
#23 comment6535824811; all broader goals remain open.
