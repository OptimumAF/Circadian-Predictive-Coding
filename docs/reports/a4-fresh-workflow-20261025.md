# R2.1-A4.P — fresh first-party A4 workflow, 2026-10-25

[#173](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/173).
Single `/root` implementation/tests/source-procedure-result self-review, not
independent acceptance. Full repository goal and broad parents remain open.
Historical plans/log frozen under AGENTS; all earlier negative/failed/spent
evidence preserved. HEAD064fb328, inherited dirty documents preserved.

## Working behavior

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --output fresh-a4.json
```

The same flag is exposed by `circadian-affine-game`; a newly installed wheel
has not yet been qualified. Existing optional MiniGrid environment is required.
The normal command no longer needs ignored historical170data. It collects
fresh A4blue/red TRAIN, uses bundled A3, runs the existing delayed joint
A3→A4→A3 six-update stream, and reads all three causal checkpoints through
the original game loop. No new runner, learner, solver, source-loader framework
or dependency. Source provenance records its own origin/collection time/counts,
with no historical byte/SHA claim. Both fixed-source research commands remain
strict and compatible; their scientific budgets are not renewed.

Changed source: `scripts/minigrid_a4_adaptation.py` factors the existing native
record decoder and source producer handoff; `scripts/minigrid_affine_workflow.py`
adds distinct exactbool `a4_adaptation`; `scripts/run_minigrid_linear_fit_control.py`
adds mutually exclusive `--example-affine-a4`, describe-before-IO, conflict
refusals, exclusive8MiB output and honest exit2 failure reports. Default/private/
G modes retain their original behavior and4MiB ceiling. New tests live in
`tests/test_a4_fresh_workflow.py`; usage is `docs/minigrid-a4-adaptation.md`.

Two collection records and three phase records only, original identity/value
proofs checked before further work. Wrong/partial/unknown collection stops before
fit. Failed records, source/actor state, receipts, original clocks, limits and
permanent charges are retained. One150s/512MiB zero-update/replay owner begins
before collection/admission/training and spans all games; the original inner
six-update90s/512MiB owner is unchanged. No clock reset or old budget renewal.

## Native result, including the failed attempt

Original product qualification **FAILED/SPENT**, EXIT2 after both teacher
episodes completed9steps/.49375/closed. It performed18teachersteps/2env,
**0fits/readouts/bundled payload reads**. Reason: actual in-memory native
observations are tuples, while the historical JSON transition codec accepts
lists. The literal fixture had JSON-shaped lists and missed this boundary.
All learner metrics are unknown for this attempt, not zero-success outcomes.
Body1.0393154s/parent1.3425254s, outer sampledRSS77537280B (inner78020608B),
JSON/countUTF8272403B each; original clock/limits/NumPy RNG preserved.

Repair normalizes only exact tuple observations to lists at the producer/codec
boundary. Exact shape, scalar, role, layout, return, native closure and typed
packet equality checks remain. Five targeted zero-fit checks per platform
verify actual tuple handoff and short/string/NaN/out-of-range refusal.

Before further native work, #173 and `REPAIR_PROTOCOL.md` prospectively amended
agent-created **product engineering** limits for one explicit corrected follow-up.
Original attempt/protocol/receipts remain failed and permanently charged; this
is not a new scientific comparison, hidden task rename or171/172retry. No
seed, metric, solver, horizon or algorithm changed. Original and corrected
clocks/reports are distinct; successful correction does not pass the original.

Corrected qualification **COMPLETE**, EXIT0, body7.9364371s/parent8.261291s,
outer sampledRSS86224896B (inner82169856B). Fresh blue/red teacher episodes
each completed9steps/.49375/closed. Collection interval .0842669s is a nested
interval, not total import/preflight/workflow time. All12game records closed:

| Checkpoint | A3 goals | A4 goals |
| --- | ---: | ---: |
| After initial A3, update2 | 2/2 | 0/2 |
| After A4, update4 | 2/2 | 2/2 |
| After A3 return, update6 | 2/2 | 2/2 |

Ten successful games take9steps/return.49375; two initial A4 failures complete
16steps/return0. Acquisition gain+1, old retention delta0, return gain0,
recovery[false,false] because A3 was never lost. This verifies useful ordinary
known-TRAIN product behavior. It is not independent replication, generalization,
live deployment, matched compute, CPC benefit or release readiness. Accepted
#171/#172 scalar evidence remains unchanged; neither old model/source was read
or rerun. No final/probe/private input, evaluation-selected updates or promotion.

Corrected cost:6LSE90exposures =54arrivals (including18oldreturn) +36jointreplay,
36distinct selected TRAIN records.16TRAINdiagnostic calls144rows plus122live
game calls/steps =138predictcalls266rows,356port projections,5031snapshotcopies,
15196617typedscalar bound,9owners;14env/reset/closes and140physicssteps include
18new teacher steps. JSON/countUTF83538623B each includes16KiB conservative
terminal reserve. No optimizer/backward/Torch/random model training.
Source/actor/clock/budget/paid-prefix/closure/RNG guards pass. These scoped
counters/reserves are not universal physical allocation or FLOPs.

Across failed+corrected attempts:36teachersteps,16env/closes,158physicssteps,
6fits90rows, body8.9757525s/parent9.6038164s; RSS reported separately. One bundled
94705B payload total, as-read SHA5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508
via existing reader, no new hash preparation.54bundledTRAINrows decoded,
18A3selected/unused36notfitforward; bundled original generation cost not
remeasured. Historical170/171/172payload/hash/model reads0. Both native
allowances now SPENT; no further automatic attempt or extra raw reader.

## Checks and evidence

50distinct focused checks/platform qualified (new workflow plus original
first-party CLI controls), then5targeted tuple-boundary checks/platform, all
within original cumulative engineering caps. Windows initial49PASS/1FAIL,
0skips,4.2048763s/RSS103059456B; descriptor expectedJSONtuple/list assertion
alone failed. Expected value normalized, only that pure check reran PASS
1.4426757s/RSS96088064B,0fits. Linux50PASS13.939922565s/RSS98848768B. Each initial
cohort6literalLSE90rows/40predict168rows/258projs/1420copies4051081scalarbounds,
native physics forbidden/NumPy RNGsame. Numeric fixture not replayed.

Five boundary checks PASS each: Windows1.6956933s/RSS90673152B, Linux7.534904694s/
RSS95137792B,0fits/readouts/physics. They execute the corrected actual collector
handoff with tuple records; old50 reused for unchanged mathematics/controlflow,
native corrected run verifies the complete final path. Original+targeted tests
use46scriptedteachersteps/18scriptedenv/70scriptedsteps per platform. Source/
training/prior-phase corruption, replacement, clock/budget changes, cleanup/
partial failure, typed packet/role/seed/row mismatch, exclusive output, missing
runtime and size-limit behavior covered. No raw historical scientific input.

Final nine-file Ruff/check-format/default+Linux-targetmypy PASS. Static/format
time and cache/storage metadata are in `final-metadata.json`. No product/test
code changed after final checks/native follow-up. Fullsuite, cleanclone/newwheel,
independent review, multiseed/comparative/statistical/release checks skipped;
forced OS-timeout cleanup branch source-reviewed only.

Exact commands: `.venv/Scripts/python.exe -B
artifacts/runs/a4-fresh-workflow-20261025/qualify.py windows-1`, followed by
`windows-repair-1` and `windows-boundary-1` (affected only). Cached WSL uses
`wsl -d Ubuntu -- env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES= timeout 140s
/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B
artifacts/runs/a4-fresh-workflow-20261025/qualify.py linux-1`, then
`linux-boundary-1`. Exact scoped statics commands/outcomes in statics JSON.
Original `execute.py` once FAILED; explicit corrected `execute-repair-1.py`
once COMPLETE. No installs/downloads/GPU/browser/CI/commit/push/delete/reset.

Stage: `artifacts/runs/a4-fresh-workflow-20261025/`, original and repair protocols/
reviews, source observers, both actual/work/terminal/execution records, JUnit,
parent/quality/statics/metadata. Only same-process terminal scalar summaries were
consumed, no extra raw actual/source/model reader. Errors/costs remain visible.

## Critical path / exact next action

This closes the ordinary fresh-source product gap; no new context/control/
checkpoint/audit clone is next. Current packaging includes scripts and bundled
TRAIN resources, but the last qualified wheel predates these A4 source changes.
Next reconcile existing build/install qualification and run a bounded cache-only
fresh-wheel isolated-install **zero-model** CLI describe/resource/dispatch check
outside the checkout, repairing only demonstrated packaging failures. Reuse
this native outcome rather than rerunning training/games for packaging. This
is a concrete clean-user/release dependency, not broad release completion.
Fullgoal/R2/R4/R5/G5/R7/P9 and #168 RNG gap remain open; prior CLM/native/resource/
matched CPC failures and all scientific requirements remain preserved.
