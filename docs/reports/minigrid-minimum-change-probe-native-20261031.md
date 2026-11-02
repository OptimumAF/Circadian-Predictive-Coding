# #370 / R2.1-MG.MIN.P.N — fixed F fit with incomplete old-task retention

The single actual minimum-change probe completed. F improves from 15/18 to
18/18 correct decisions in two fits. A stays 18/18; B and C each fall to 17/18.
The original source is unchanged. This demonstrates reserved-task fit on the
disposable learner and incomplete retention of old TRAIN decisions. It is not
native learned gameplay, heldout generalization or recovery after forgetting.

| F updates | F correct/18 | F mean CE | A correct/18 | B correct/18 | C correct/18 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 15 | 1.4138118310 | 18 | 18 | 18 |
| 1 | 16 | 1.3639628839 | 18 | 18 | 18 |
| 2 | 18 | 1.1654221805 | 18 | 17 | 17 |

Frozen F CE gain is 0.24838965054953688; correct-count gain is 3. Old-task mean
CE changes (final minus initial) are A -0.0280813441, B +0.0086390406 and
C -0.0286530620. B becomes worse by both measures. C's CE improves while its
correct count falls: neither measure replaces the other. All three points,
both F cues, old source spans and row decisions remain saved.

## Historical cold evidence and limits

The saved #113 cold case has the same F curve, but its old-task final counts
are A16/B14/C14. Its one-update old counts are17/16/16. No cold learner was
refitted. Saved-only comparison passes the frozen applicability checks: same
original source/TRAIN bindings, runtime, four F episode records, monitor spans,
protocol after removing the two new choice fields, and initial scores/actions
(CE absolute tolerance1e-12). This is descriptive historical reuse, not a new
paired experiment or independent replication.

The incremental case has less interference in this fixed observation, with
remaining B/C errors. Equal fit counts and row exposure do not establish matched
computation: this run performs six scheduled training design products/54 row
visits. The older observer used broad profiling and current guards differ, so
the elapsed-time difference cannot establish efficiency. There is no CPC,
statistical reliability or general benefit claim.

F train/test transition sequences are exactly equal per cue under the original
deterministic task/seed. Thus18/18 measures fit on that reserved domain. All36
simulator steps below belong to teacher collection, not learned-policy games.
There was no old-context return phase or observed A loss, so no recovery claim.

## Actual command, source and costs

Prospective start6545934470/freeze6545979516; stage
`artifacts/runs/minigrid-minimum-change-probe-native-20261031/`. No production
source changes. Existing #369 software/#367 math/#368 known-game evidence is
reused within its producer scope. #217/#365 and all other failed/SPENT captures
remain unchanged. No old runtime or spending authority was restored.

Physical `D:\Dev\Repositories\Circadian`, TEMP/TMP on `D:\Dev\_Temp`.
Windows Python3.14.7/NumPy2.4.6, original two BLAS threads; MiniGrid3.1.0,
Gymnasium1.2.3, pygame-ce2.5.7. Linux native work was skipped; no installation.

```powershell
& .venv/Scripts/python.exe -B artifacts/runs/minigrid-minimum-change-probe-native-20261031/producer.py --preflight
& .venv/Scripts/python.exe -B artifacts/runs/minigrid-minimum-change-probe-native-20261031/launch.py
& .venv/Scripts/python.exe -B artifacts/runs/minigrid-minimum-change-probe-native-20261031/analysis.py
```

The launcher invokes the current CLI exactly once:

```text
--context-linear-probe --probe-update minimum_change
--source artifacts/runs/joint-catalogue-study-20261018/bp.txt
--retained-source artifacts/runs/catalogue-capacity-new-c-20261024/actual.json
--output <stage>/actual.json
```

Original after-C source377,836B/SHA05ee674... and readonly TRAIN2,346,812B/
SHA4b29e1... are strictly bound; full digests are in checks.INPUTS and preflight.
Historical cold930,239B/SHA7d01ab... is read only for saved comparison. The
existing loader decodes training_sources, not evaluation sections. Upstream
10fits/180rows remain historical paid work, including unused branches. Selected
source6updates/108rows is distinct from new private2updates/18rows; final8/126.

Sole native/CLI exit0, no host expiry/retry. Child2.676860400s/sample peak
88,272,896B; host2.859214500s under original90s/512MiB/105s host bounds. The
original probe2-update/zero-replay/90s/512MiB session has elapsed2.020269400s
with original owner, limits and birth clock. Source/caller state, global
NumPy/Python RNG and original authority checks pass. No source/deployed/benchmark
learning, replay, return, promotion, final or controller work occurred.

| Named observed work | Attempts = known returns |
| --- | ---: |
| Update ports / rows | 2 / 18 |
| Warm fits / rows; cold fits | 2 / 18; 0 |
| LS solves / rows | 2 / 18 |
| Scheduled training products / row visits | 6 / 54 |
| Learner/core/probability predictions, each | 12 / 216 rows |
| Metrics and decision diagnostics, each | 12 / 216 rows |
| Nested projections | 696 |
| Owner construction / restore / fork | 2 / 3 / 1 |
| Snapshot API calls / nominal coefficient visits | 243 / 1,002,078 |
| Named copies / fit validations | 250 / 520 |
| Environment construction/reset/local seed/unique successful close | 4 each |
| Teacher choices / physical steps | 36 / 36 |
| Source/TRAIN decodes | 1 each |
| Named qualified reads / expected byte visits | 314 / 427,769,736 |
| Pickle output calls / byte visits | 355 / 20,593,113 |
| Forbidden work | 0 |

These are named API/product/expected-source/output visits, not all allocations,
copies, heap, FLOPs or OS I/O. Other source/integrity/hash/serialization/Python
work is inside native elapsed/RSS bounds. Four closed environment references
prevent lifetime-ID reuse; there is no unlimited success-record chain.

Input arrays406,944B comprise raw193,104B, selected history85,536B and old
monitors128,304B. The generic helper additionally owns85,536F bytes and128,304
monitor bytes. Metadata, signatures, copies, solver/native/Python objects and
reports are extra and included in RSS/time, not those logical array figures.

CLI929,353B and captured original19,227B gzip pass canonical equality plus
same-file identity/content pins. Native stderr is empty; stdout933,111B is
retained. Exclusive native.attempted/SPENT markers permanently close the case.
Saved-only analysis ran once, exit0/complete true,0.015625400s/27,070,464B peak,
with no model/simulator imports or further fits. Its second slot is unused.

## Engineering evidence and unresolved resource proof

Pure gates use the existing Windows/WSL interpreters with `-B <stage>/check.py
<label> <stage>/cache/<label>`; targeted nodes/actual argv and outcomes are in
each checks receipt. They forbid project model/numerical imports and perform
no solver, learner, teacher or environment work. Final32distinct cases qualify
per platform;99casecalls/5cohorts10.223191246s measured Python-body time.

- Windows1:29PASS2FAIL,0.704938100s/50,286,592B. Text-mode temporary files were
  CRLF while the fixture expected LF hashes; write_bytes fixes the fixture,
  preserving the strict byte gate and original failed receipt.
- Windows2:31PASS,0.576648800s/49,201,152B.
- Linux1:31PASS,4.270117407s/38,428,672B; same handle60784 to terminal0.
- Final targeted Windows3:3PASS,0.617876200s/47,960,064B.
- Final targeted Linux2:3PASS,4.053610739s/37,593,088B; same handle40869 to
  terminal0, never restarted.

Source self-review corrected draft expected source IDs before models: selected
F training records have original :copy:0/:copy:1 release IDs; physical sources
and probe_test keep unsuffixed IDs. The collector/data were not changed. One
wrong-copy refusal case plus positive/negative acceptance were run under the
prospective Windows extra-three-pure-case amendment6546142658; Linux used its
original second slot. Distinct36, aggregate180body seconds, native and storage
limits stayed unchanged. A patch anchor mismatch was corrected using actual
formatted context. No test or criterion was weakened.

Final five-file Ruff/format/default plus Linux-target mypy pass on first and
final trees. Three static invocations12.6289766s plus two formatting calls
0.2237420s are below90s. Sole Windows source-only preflight exits0, complete,
0.159727500s/45,604,864B, binding31named source/input/control paths and4drivers.
External preflight host0.3216283s/no expiry under30s. This is named binding,
not transitive source authentication. No project/numerical imports occurred.

The final WSL launcher was observed starting09:35:32, with test attempt09:38:05
and body receipt09:38:09: roughly153s pre-interpreter waiting. Total launcher/
host-inclusive aggregate was not instrumented and is not claimed conformant to
a host-inclusive180s ceiling. Reported10.223s is interpreter-body time only.
No user process was stopped and no emulator/WSL reset was performed. Actual
learning uses the separately timed Windows90s/105s path, not this launcher.

PRE gives scoped source/protocol GO: six reads, known outer0.7552451s plus
unknown read4 outer time after context loss. Lost/truncated output and limited
full-test-body coverage are retained; exact aggregate review-time conformance
remains unverified/open. Final31+4 hashes/preflight/statics were independently
checked. Independent visibility of the unspent marker projection was truncated;
the root observed it and the exclusive producer marker enforced one attempt.
These limitations are not retroactive passes or scientific budget renewals.

## Review, open claims and next action

Lead source/procedure/result self-review passes for actual source, outcomes and
native limits. Independent POST ACCEPTS the scoped completed observation and
interpretation: four readonly calls0.7896992s exact outer. It checked actual31+4
hashes, saved SHA/size pins and parsed CLI/raw equality, counters/caps, curves,
source/role/budget metadata and storage. OS inode/time enforcement is supported
by inspected admission/analysis and executed saved checks, not independently
reimplemented. Initial report-output truncation was resolved by targeted cost
reads; full test bodies/row diagnostics were not exhaustively reread. No models
or numerical replication ran. See REVIEW-POST.md/timings; finite exception ended.
The original administrative resource-proof limitations above remain open; they
must not be collapsed into a claim that every frozen engineering gate passed.
All native numerical/collection/capture/time/RSS caps are observed held.

Full R2/R4/R5/R7, reliable acquisition/retention/recovery, unseen transfer,
comparative/CPC benefit and repository/release readiness remain open. Skipped
unchanged mathematical/old model-bearing fixtures/control refits, full suite,
clean clone/wheel/install, Linux native games, GPU/robot/browser/remote CI and
final/controller studies. No dependencies, downloads, Git publication, deletion
or global changes; unrelated dirty/ignored work and every old negative remain.

The case's original engineering resource qualification stays PARTIAL/OPEN; the
successful observation does not supply missing historical timing proof. Preserve
that unmet requirement without rerunning the spent case or inventing a timing.

Critical-path reassessment selected [planned #371](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/371):
compose the new isolated F operation in the existing first-party workflow and
target-free native readout, including F and all original old A/B/C contexts.
The current workflow private path hardcodes the two-arm rehearsal comparison
and has no probe_update option; the native F evaluator already supports8/126.
This is a concrete missing end-to-end capability, not another saved-source
capture or repeat fit. Include all old tasks rather than choosing A after seeing
B/C errors. Reuse the existing stream/probe/native loop, defaults and accepted
evidence; no new runner/source-loader. Exact next action: inspect that private
handoff and mixed-role evaluator, freeze minimal selector/composition and scripted
fixture costs, then implement and execute checks before any later actual game
qualification. No new numerical allowance, baseline rerun or promised win.

[Canonical scoped observation and partial-qualification handoff](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/370#issuecomment-6546489450).
Readback: #370 OPEN/five checked/one partial unchecked resource criterion;
#371 OPEN/five unchecked/planned/no numerical allowance. Parent update receipts
#18/6546503279, #21/6546503520, #23/6546503734; all parents remain OPEN.
