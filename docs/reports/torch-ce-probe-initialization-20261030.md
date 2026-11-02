# R2.1-TW.CE.P / issue322

Previous turn PROGRESS: accepted321 solved all four known TRAIN tasks with an
additional CE-output fit, not native SGD alone. This increment supplies its
concrete missing continuation seam: existing isolated native probes can receive
the fitted output while retaining complete original SGD momentum/gradients.
Full goal, #18/#21/R2.1 scientific acquisition/retention/recovery/comparisons and
release remain open. No new genuine source12/task experiment was executed.

Physical checkout D:/Dev/Repositories/Circadian, HEADd9119e80926a139a2f38460564d7eab1b6d2a2b0
plus preserved dirty/untracked work. Lead changes only new adapter/tests,
minimal optional input in existing ticket_native_probe, usage/report/README,
this local stage and archival plan/log pointers. No commits/pushes, installs,
downloads, deletions or global settings/cache changes.

New adapter src/adapters/torch_categorical_ce_initialization.py derives an owned
snapshot from actual BP owner+actual complete fit, changing only output weights
and bias. No model/random/forward/fit/update work for derivation. Native counters
retain SGD meaning; full source/fit/derived provenance and auxiliary work remain
separate. Preserve even stale momentum and gradients; no silent reset/rescale or
fake SGD credits. Optional existing probe keyword uses its independent factory,
not source-counted fork; original checkpoint12 qualification is unchanged.
fit/source/input/RNG/ownership/callback guards refuse foreign changes and retain
them. Final data, selection, promotion and deployment are unavailable here.

Stage artifacts/runs/torch-ce-probe-initialization-20261030/ contains the frozen
SOFTWARE_PROTOCOL.md, check.py work observer and exclusive process receipts.
Budget fixed in322 before numeric work: components128/draw256/8192values;
24native SGD/backwards128rows;8CEoptimizers/512closures1536rows/8final32rows;
hidden128/512rows, affine128/512, nativehead128/512, typed128. Each pytest120s/
768MiB, cumulative240s; scoped static aggregate120s, stage4MiB. TEMP/TMP D,
WSL TMPDIR D; no new library-temporary-byte evidence claimed.

All pytest calls below ran from the physical checkout, CPUthreads1, no cache
provider and --capture=sys. Windows Python3.14.7/NumPy2.4.6/Torch2.14CPU; Linux
existing Python3.12.3/NumPy1.26.4/Torch2.14CPU. No runtime installation/change.
Model work is small2D/H3/K3/seed17, never the genuine9D source/task cohort.

Commands (prefix W = `.venv/Scripts/python.exe -B`; prefix L =
`wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B`):

```text
W artifacts/runs/torch-ce-probe-initialization-20261030/check.py artifacts/runs/torch-ce-probe-initialization-20261030/windows-first-work.json tests/test_torch_categorical_ce_initialization.py -p no:cacheprovider --capture=sys
W artifacts/runs/torch-ce-probe-initialization-20261030/check.py artifacts/runs/torch-ce-probe-initialization-20261030/windows-combined-work.json tests/test_torch_categorical_ce_initialization.py -p no:cacheprovider --capture=sys
L artifacts/runs/torch-ce-probe-initialization-20261030/check.py artifacts/runs/torch-ce-probe-initialization-20261030/linux-combined-work.json tests/test_torch_categorical_ce_initialization.py -p no:cacheprovider --capture=sys
W artifacts/runs/torch-ce-probe-initialization-20261030/check.py artifacts/runs/torch-ce-probe-initialization-20261030/windows-guard-work.json tests/test_torch_categorical_ce_initialization.py -k 'derivation or callback_or_copy' -p no:cacheprovider --capture=sys
L artifacts/runs/torch-ce-probe-initialization-20261030/check.py artifacts/runs/torch-ce-probe-initialization-20261030/linux-guard-work.json tests/test_torch_categorical_ce_initialization.py -k 'derivation or callback_or_copy' -p no:cacheprovider --capture=sys
```

Pre-review outcomes: Win15PASS then19PASS; Linux19PASS; final5affectedPASS/platform after
explicit equal-fit-work-replacement guard.20distinct tests qualified/platform,
63PASS executions/0FAIL/0SKIP. Initial Windows pass preceded script handoff cases;
the combined cohort qualifies those changes. Final affected checks cover the
source-bound initializer's added identity pin; unchanged numeric/probe results
are reused, not rerun a sixth time. Each process EXIT0/budget_verifiedTrue.
Durations3.4891775/3.5964022/7.715734579/3.2408203/7.511448662s;
all25.553583241s/max344,952,832B. All shared numeric caps held, no renewals.

Paid aggregate112components224namedseededdraws1680values/14SGD-backwards42rows;
5CEoptimizers178forward+backwards534rows/5final15rows;14hidden24rows/9affine9rows/
32nativehead78rows/18typed.0LS/LP/SVD/autograd.grad/PC latent/probe16/genuine
source12/physics/final/GPU/remote. Three of those private native updates are tiny
one-update FreshTaskProbe fixtures, included in native14; not zero probe work.
The fixed CE solver uses34closures Windows and38Linux on this fixture; no
cross-platform bitwise fit claim or favorable selection. Platform parity tests
compare each native reader with its own complete float32 fit.

Behavior: populated original SGD/RNG/ownership unchanged; exact raw logits and
typed predictions equal the CE reader without rescaling; manually installed
private reference and initialized private learner have identical complete native
continuation. Existing FreshTaskProbe makes one private paid update, leaving
both initialized seed and original source unchanged. Corrupt hidden/momentum/
gradient/counter/coefficient/work records, spoofed public source snapshots and
source/fit/deepcopy callback mutations refuse. Equal fit-work replacement refuses.

Four script handoff modes use EXPLICITLY SCRIPTED source12 counters, clock,
schema/cohort/readout seams and stop at probe entry: default, optionalCE,
equal-fit-work replacement and foreign-source fit. They prove original checkpoint
versus derived seed routing and rejection before work, not genuine native12
training, full32updates, six-task success or source-budget measurement. Their
actual native seed constructor/restores are included in observed components.

Scoped static commands target the new adapter, script and new test file:
`W -m ruff check <files>`; `W -m ruff format --check <files>`;
`W -m mypy --explicit-package-bases --follow-imports=silent <files>` and same
`--platform linux`. Initial5.994973s: formatPASS, Ruff unused replace import,
both mypy3test annotation errors; corrected annotations and the import is used
by new integration fixtures. Combined5.9281316s all4EXIT0. Failures retained in
statics-first.json; no model/scientific failure or spent experiment retry.
Final static/review receipts follow below.

Independent review found a concrete script pin-order blocker: references were
captured after initial deepcopy, allowing equal-value fit.work replacement during
a copy callback to become the baseline. Fixed by capturing initialization refs
before ANY initial input copy. Added literal script deepcopy callback regression;
foreign replacement is preserved and rejected before cohorts/models/probe. No
mathematics/default/protocol change. Earlier passing fixtures lacked this case;
they were not evidence for that callback boundary.

Final affected commands use the same W/L observer prefix and new exclusive
windows-copy-work.json/linux-copy-work.json, with the same test file and
`-k scripted_probe_handoff -p no:cacheprovider --capture=sys`. Both5PASS EXIT0,
3.3551588s/7.606763522s, peak327,868,416/344,096,768B. These qualify all five
scripted handoff modes (including default and optionalCE) after the order fix.
Final4statics PASS5.8922032s; preceding final5.8593762s PASS also preserved.
All static work23.674684s<120; no cap amendment or overrun.

FINAL totals:21distinct tests qualified/platform,73PASS executions/0FAIL/0SKIP;
7pytest processes36.515505563s/max344,952,832B, allbudget_verifiedTrue.
124components248namedseededdraws1860values/16nativeSGD+backwards48rows;
7CEoptimizers250forward+backwards750rows/7finalTRAIN21rows/16hidden30rows/
9affine9rows/34nativehead84rows/18typed; no forbidden numeric work. All shared
original software caps held. summary.json is a stdlib-only receipt sum, no new
model/fit/forward call. Native study/adaptation outcome still unexecuted.

Finite independent reviewer used4read-only commands/.6388582s, no project or
numerical imports, models/fits/tests/writes/network/children. Review found the
pin-order blocker above; actual fix/regression and all7numeric work receipts
were then independently checked. Its final metadata helper hit AssertionError
while projecting guessed static receipts. This was a review-helper projection
failure, not a project test or source failure. Lead supplied exact filename and
observed exits/times without another command: statics-copy-final.json is final
5.8922032s/[0,0,0,0]; statics-final.json5.8593762s and statics-combined.json
5.9281316s also all0; statics-first.json is preserved5.994973s/[1,0,1,1].
No model/static rerun or review-budget extension for that metadata mistake.
Self-review remains distinct from independent acceptance; final disposition
recorded below after the review handoff.

Final independent disposition: ACCEPTED scoped322 software after pin-order fix.
Actual source/private continuation/receipt guards and all7work receipts accepted;
latest4static passes corroborated by lead's exact receipt projection. Exception
ended, no standing worker. This is not independent acceptance of a genuine
adaptation result, because that case has not been executed.

Continuation prompt's old Downloads multi-agent file was missing; bounded repo
filename search found no replacement. Current AGENTS/AGENT_PROMPT supersede its
coordination workflow; scientific/ownership/resource requirements were preserved.
Physical D checkout reverified (no reparse target). Final new source and README
whitespace checks emitted no errors; README LF/CRLF advisory only.

Skipped broad repository/unchanged320math/nativefamily/baseline/cleanclone/wheel/
release checks, genuine known-task repetition, F16/return16 native case and
comparative/statistical studies. The tiny fixtures establish trainable plumbing,
not useful adaptation, retention/recovery or CPC benefit. Earlier FAILED/SPENT
results,318unverified temporary-byte criterion,319supporting static overrun and
321's paid408oldTRAIN rehearsal remain unchanged.

Exact next action: freeze/review ONE original reserved-probe runner nativeBP
source12/72→fixedCEfit→explicitderivedprivateinitialization→existingF16/return16
case, with source qualification before auxiliary work and all fit/probe/readout/
source-budget/callback costs declared. Use original source report/receipts and
independent factory; don't rewrite checkpoint12 with derived weights. Source
must be live-owned in that one invocation, not reconstructed from saved scalars.
Preserve original large logits/momentum/metrics/seeds/directions and accept a
negative adaptation result; no new runner or near-identical readout/capture study.

Closeout receipts: issue322 canonical comment6528014637 and state closed/completed,
four scoped criteria checked; tracker18 comment6528021477 and parent21
comment6528021881 leave the full goal open. Final stage15files35,680B<4MiB.
