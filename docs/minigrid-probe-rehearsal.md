# Public MiniGrid old-A rehearsal

## Matched four-method observations

The six frozen PC-family cases now complete the accepted BP pair. All eight
qualify; PC-neutral matches270TRAIN probability rows and32F/A score+diagnostic
points perallocation. Source128PC-family stays5/9percue; activeendpointCEhigher,
noCPCadvantage. F48/replay tradeoffs and all costs/signs remain in the
[human report](reports/matched-exposure-controls-20261018.md). This is fixed
teacher-forced evidence, not live acquired-task recovery/generalization.

## Matched native-exposure control

The existing F48-only route now provides a distinct extra-exposure control with
same256native calls/8064rows/P94/846rows, allocating48 F updates perprobe and no
old-A replay. The original recipe below remains unchanged. One fixed BP pair shows
F-only exposure improves F fit but has higher A CE; neither fully recovers A-red.
This is not a totalprocesscost, gameplay, generalization or CPC advantage claim.
See [F48 usage, costs and evidence](f48-exposure-control.md).

## Explicit H32 composition

Add `--wide-neural` to the command below for the original H32/seed17 recipe.
All four original methods are admitted; omission keeps H8 unchanged. Replay adds
64native calls/576rows across the two probes:256/8064total, BP256optimizer steps
orPC256parameter updates/16128nominal latent rowsteps. Games remain refused.
One original-data BP case verifies clocks48/64/176/192 with separate original
wake/return16 and replay288rows perprobe. Full states, source/actor/inputs/RNG and
original budgets pass. 472 focused checks/scopedstatics/independent review pass.
Source128 A-red9/9→8/9→8/9 remains incompletely retained/recovered; no livegames,
matched-exposure study or CPC advantage. [Exact commands, costs, limits and next
control](../artifacts/runs/wide-old-a-rehearsal-20261018/RESULTS.md).

The existing MiniGrid history pilot exposes fixed old-A TRAIN replay during its
disposable F probe, followed by A return on the same candidate. Example from the
repository root (describe performs no collection or native work):

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts/run_coupled_attribution.py --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --joint-catalogue --distinct-probe --old-task-readouts --return-to-a --old-a-rehearsal --describe
```

Remove `--describe` to execute collection and training only under a separately
declared local run budget. The other admitted methods are `pc`, `neutral_cpc` and
`gated_cpc`, with the original H8/seed17/native objectives unchanged. This option
requires the full explicit route above. Pure `--train-fit-bounds` is allowed;
other encodings, weighting, bounded inference and extra diagnostics/games refuse
before resource construction. Omitting the option preserves the existing route.

## Behavior and costs

The benchmark performs128 joint updates on its six original TRAIN episodes,
54rows each. Two isolated probes start from original source checkpoints0/128.
Each performs F16, two nine-step old-A TRAIN episode replays after every F wake
(32replay native calls/288rows), then A16 return with no replay. F remains
`probe_train`; replay remains `train`. Source, benchmark, actor, inputs and global
RNG are protected. Final evaluation remains unrun and unavailable to updates.

Every probe receives distinct original F, replay and A-return budgets. Replay
owner, limits, clock, start, progress and sampler are pinned before its factory;
charges must agree with observed events. Boundaries protect preceding report,
outcome, work and projection evidence, including equal replacements. No budgets
are renewed or permanent charges erased.

Planned and observed totals expose **256native calls/8,064training rows**, including
**64replay calls/576rows**, and94score forwards/846rows. `work.old_a_rehearsal`
separates attempted, returned and committed calls; charged replay rows; returned
training rows; logical bank bytes; and validation projection costs. A failed or
unreadable record remains partial/unknown; numerical partial sums are lower bounds.

Each bank has18transitions/87,280encoded array bytes. Two bank admissions sum to
174,560logical bytes, not peak memory. They add72validation projection rows/
170,496projected-array-equivalent bytes. Per-slot projection accounting includes
returned replay rows; validation is separate. Native states, event reports,
copies, Python objects, guard work, wall time and process RSS are additional.

Why this: reuse the accepted fresh-task API and existing stream/consolidation;
the CLI adapter adds composition and accounting, without a second update runner.

## Verification and limits

Two synthetic native BP engineering cases execute the direct probe and complete
public CLI. Together:320updates/8,640training rows/P126/1,134. Collection uses
existing explicit fixtures; physics, teacher construction and final evaluation
are forbidden. Native work and assertion outcomes are persisted in JUnit; full
CLI output was parsed in the test and was not saved as a scientific dataset.
The accepted four-method native API evidence is reused without repeating models.

Focused commands, final regression/static outcomes, independent review and exact
budgets are in [results](../artifacts/runs/minigrid-rehearsal-cli-20261018/RESULTS.md).
These scoped checks do not establish repository-wide readiness.

Replay increases old-A exposure and computation relative to the original
no-replay control. This is not a matched-dose/compute comparison or evidence of
acquisition, retained acquired skill, recovery or circadian benefit. Preserved
scientific runs still have unacquired old A. Next freeze/review a small comparison
using exact original source episodes, explicit extra-exposure controls and full
cost accounting before executing any comparative experiment.

## Files and extension

```text
src/adapters/minigrid_learnability.py       original pilot composition/budget guards
src/adapters/minigrid_probe_rehearsal.py    option admission, description, cost reducer
scripts/run_coupled_attribution.py         existing public CLI flag
tests/test_minigrid_rehearsal_route.py      public admission and native integration
tests/test_minigrid_probe_rehearsal_work.py partial receipt/cost behavior
tests/test_minigrid_rehearsal_cli.py        CLI dispatch/refusals
```

Extend the declared recipe only with separate acceptance tests and a prospective
protocol. Keep default dispatch unchanged and separate replay costs from F/A
charges. See [ADR0430](adr/ADR-0430-public-isolated-old-a-rehearsal.md) and the
[accepted API](isolated-probe-rehearsal.md).
