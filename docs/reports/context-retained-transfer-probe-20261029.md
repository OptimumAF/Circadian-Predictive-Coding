# Actual retained donor: isolated transfer and return, 2026-10-29

R2.1-CTX.LS.T.O, [#303](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/303).
This completes a scoped composition and one descriptive observation. Full R2,
retention, learned-controller, model-family, scientific and release requirements
remain open. Local bytes are not committed or published evidence.

## Working behavior and observed outcome

Explicit `retained_joint_probe` uses the existing owned workflow. Original source
12 fits/72 rows precedes six ordinary wakes/27 rows and five joint fits/45 rows.
After all serving workers close and the preceding terminal readout is validated,
the real candidate23/144 is copied into an independently owned plain LS donor.
The accepted transfer API forks that donor for16 probe updates/48 rows followed
by16 old-TRAIN return updates/96 rows. No new loader, runner or training loop.
[Usage, isolation contract and frozen protocol](../context-retained-transfer-probe.md).

Fixed probe_test task outcomes, in the original order:

| Phase | Old invoice/password-v1, invoice/parcel-v2 | Selected parcel-v1/password-v2 |
| --- | --- | --- |
| Before probe | 4/4 | 0/2 |
| After16 probe updates | 2/4 | 2/2 |
| After16 old-TRAIN return updates | 4/4 | 0/2 |

All18 episodes completed, three steps each, zero training updates and zero external
actions. Old password-v1/parcel-v2 failed after probe; both selected tasks failed
before probe and after return. These failures remain in `actual.json`. Selected
classification accuracy was2/3 each initially,1 after4/8/16 probe updates, then2/3
after4/8/16 return updates. Initial CE1.2580700206584228/1.2263485469698603;
probe CE1.0435917781858575/1.0435917781858577; return
CE1.22634854696986/1.2263485469698603. No seed, order, metric or direction changed.

This supports new-task acquisition and recovery of previously observed old-task
ability in this fixed set. It does NOT show simultaneous retention: new tasks
were forgotten on return. Prior legacy-task retention is outside these readouts
and unverified. Concepts/workflow stages are shared; no disjoint/unseen-task,
plasticity-cause, CPC advantage, fair-control or generalization claim follows.

Original actor's live serving completed24 episodes with15 successes/9 failures;
private probe learning did not improve or replace that deployed actor.
Original candidate remained23/144. Source, serving actor, original runtime scope,
paid evidence, donor and Python/NumPy RNG checks passed unchanged. Private work
returned32/144 with no unknown unreturned work. Original source/root guard and
candidate lease bind receipts, original inbox/clock, labels/consent, budget
origins/limits/charges, actor configuration/version and report owners/values.
Probe updates do not enter the bank, deployed learner, benchmark or promotion.
Final evaluation exposure and root deployed probe updates were zero.

## Executed checks, costs and preserved failures

Physical checkout `D:\Dev\Repositories\Circadian`, HEAD0aae1d6 with inherited
dirty/untracked work preserved. Windows3.14.7; existing Linux3.12.3/NumPy1.26.4.
Every project shell used TEMP/TMP `D:\Dev\_Temp`; Linux TMPDIR `/mnt/d/Dev/_Temp`.
Artifacts: `artifacts/runs/context-retained-transfer-probe-20261029/`.

```powershell
$tests = @('tests/test_context_retained_probe.py','tests/test_context_linear_same_context.py','tests/test_context_owned_reference_validation.py')
.venv/Scripts/python.exe -B artifacts/runs/context-retained-transfer-probe-20261029/check.py artifacts/runs/context-retained-transfer-probe-20261029/windows-final-work.json @tests
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-retained-transfer-probe-20261029/check.py artifacts/runs/context-retained-transfer-probe-20261029/linux-final-work.json @tests --capture=sys
```

Final combined tree:71PASS,0fail/skip per platform. Windows4.147482300s,
82,219,008B sampled peak; Linux15.776255032s,82,395,136B. Each11 actual LS fits/
72 rows/11lstsq. The integration uses a scripted private probe port: its32/144
metadata is NOT physical numerical work. Each cohort also observed22 nested
training-port calls/144 rows (wrapper+leaf, not22 solves),18 affine calls/18 rows,
6 probability calls/27 rows,80 routes/258 rows,9 teacher traces,38 training-case
builders,6 task episodes/501 dispatches, no Torch. Behavior checks cover original
clock/cursor/label/consent/receipt/actor/budget/paid-evidence corruption, wrapper
replacement, donor mutation, lost result/unknown costs and pre-work mode refusal.

Earlier commands use the same check.py with these output/test arguments:

| Receipt | Test suffix | Outcome | Seconds | Actual LS fits/rows |
| --- | --- | --- | --- | --- |
| windows-pure-work.json | test_context_retained_probe.py -k 'not finish_retained_serving' | 0PASS/1FAIL | 1.387972900 | 0/0 |
| windows-pure2-work.json | same pure selection | 19PASS | 1.791543000 | 0/0 |
| windows-pure3-work.json | same pure selection | 21PASS | 1.876283500 | 0/0 |
| windows-combined-work.json | same three-file cohort as final command | 21PASS/1FAIL | 3.794943300 | 11/72 |

Exact argument arrays are authoritative in each receipt. First failure: synthetic
SimpleNamespace receipt incompatible with the unchanged strict comparator; fixed
fixture to dataclass. Combined failure: nonexistent runtime._clock after11 paid
retained fits, before any private probe fit; fixed to original runtime._inbox._clock
under its candidate lease and made fixtures reflect the real interface. Both
failed attempts remain charged. All fixtures total33LS216 rows. Original forecast
77/432 became88/504 after the failed11/72, still under SAME96/560 allowance. No
budget reset, weakened comparison, source renewal or native retry.

```powershell
$checked = @('scripts/context_head_rehearsal.py','scripts/context_head_live.py','scripts/recurring_ticket_workflow_example.py','scripts/context_retained_probe.py','tests/test_context_retained_probe.py')
.venv/Scripts/python.exe -B -m ruff check @checked
.venv/Scripts/python.exe -B -m ruff format --check @checked
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent @checked
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux @checked
```

All four terminal0,3.538391100s total; static RSS uninstrumented. `statics.json`
contains exact commands/outcomes. No source edit after final combined checks.

## Single native observation

After focused gates and source/protocol review:

```powershell
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-retained-transfer-probe-20261029/native.py
```

ONE `native.attempted`, terminal status0/noerror,39.441647718s,65,449,984B sampled
peak,55LS fits/288 rows/55lstsq. Combined engineering88/504 under96/560. Native
39 teacher traces,72 episodes,365 dispatches,248 typed/496 delegated calls,
212 affine calls/rows,38 probability calls/171 rows,354 routes/704 rows,
36 unfitted reads,0 scalar replay,0 Torch. Counters cover named paths, not every
scalar/array operation. Rehearsal's45 reused TRAIN rows are included in55/288;
zero scalar replay does not imply no replay exposure. Monitor/validation/diagnostic
time and memory are included in invocation totals.

Native source120s was declared prospectively AT BIRTH, only in this explicit
combined mode, with measured #30113.589s+#30217.366s rationale. Existing30s source
modes/catalog-reference240s and retained60s/probe-return30s unchanged. Actual
source age38.187337616s<120; retained age34.090110477s<60. Probe/return sessions
both born before probe at14.139305640/14.139307986; final readings
41.595657335/41.595757545, ages27.456351695/27.456449559<30. Neither renewed.
Their phase-only measured times2.612205814/2.836807583 exclude intervening guard/
readout overhead; phase times must not replace full session-age accounting.

Output1,963,856B; stage12 files1,983,290B, within8MiB. `actual.json` and
`native-work.json` are authoritative; no rerun after a tool-output truncation.
Broader tests, unchanged native baseline, raw-backed/model-family/GPU/remote CI,
clean clone/current wheel/release checks skipped. #279 wheel predates these
changes. No installs/downloads/deletions/global changes/commit/push. #285 negatives,
#287 FAILED/SPENT and unrelated/frozen/ignored work preserved.

Lead self-reviewed actual source, procedure, task vectors, clocks and all returned
costs. Finite independent review found no blocking source/integration/protocol/test
defect before native, then directly confirmed native totals/root isolation flags.
Four readonly calls, no imports/tests/models/network/writes. Known command wall
0.5620596s plus an unavailable second-read duration; do not claim complete measured
review time. Final projection used the wrong JSON level and missed nested donor/
endpoint/clock fields; those are lead-verified, NOT independently result-accepted.
No extra reviewer retry; exception ended. No independent scientific replication.

## Next substantial outcome

The demonstrated gap is simultaneous retention, not another checkpoint/capture
increment. Existing `prepare_joint_rehearsal` and `context_head_rehearsal` already
provide TRAIN provenance and joint/current-only controls; do not recreate them.
Next implementation action: reconcile and extend that existing private rehearsal
composition to accept the fixed transfer-task cohort and actual retained donor,
using bounded same-context joint TRAIN exposure on the disposable learner with
matched current-only exposure. Start with role/ownership/row-charge and old/new
behavior checks before freezing any small comparative protocol. Preserve current
sequential mode/negative results, original session limits and final isolation;
never relabel probe data as retained TRAIN or infer equal compute from equal rows.
No new scientific run, #287 retry, tuned baseline or sweep is authorized by this
report. Full roadmap/model-family/control and release claims remain open.
