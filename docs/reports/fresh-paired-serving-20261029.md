# R4.8-CTX.FS — Fresh-source ordinary paired serving

Issue [273](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/273)
delivers fresh TRAIN to guarded ordinary serving. Full goal and parents
18/21/23/26/33 remain open. Live status is in GitHub under AGENTS.md; the two
historical plans and development log remain frozen. Checkout HEAD 0aae1d6 and
inherited 270–272 dirty changes were preserved. No commit, push, install,
dependency, global-setting or historical-capture change occurred.

## Implemented and executed behavior

`context_head_rehearsal.prepare_owned_source` extracts the existing native source
qualification for both private and ordinary callers. It retains exact D9/K6
schema, source 12 passes/72 rows/16 mistakes, head counts 8/48 and 4/24,
checkpoints 0/4/8/12, original summary and row identities, source/actor/RNG
preservation, four successful terminal ordinary tasks and `outer_selection`
roles. Original budget witness observation/checking is serialized across owner
and worker checks. It rejects changed or replaced prior evidence, changed limits,
unobserved clocks and counters before admitting more work.

`context_head_live.run_owned_case` composes that guard with the existing paired
serving path, sharing the source sampler. Historical-source APIs stay intact.
The owned path has truthful in-memory provenance and no historical source SHA.
The workflow's explicit API/CLI `paired_serving=current|joint` keeps the source
stream open through ordinary child serving, validates it again, then seals its
training-only stream. It cannot combine with private adaptation. The child uses
original ordinary TRAIN teachers, guards, worker schedule, publication and
rollback. Source and benchmark actor receive no child updates; probes/final
evaluation are unavailable. Output uses the existing bounded 4 MiB writer.

Independent source review found one blocking accounting defect: a lost delegated
child report could be represented as zero work. It is repaired: only rejection
before delegation has known zero child work. Lost delegated work is explicitly
unknown, with null aggregate costs. If a child report returned before a final
source-guard rejection, its paid prefix remains. Missing child fields are null;
a truly absent child after source failure contributes known zero. Producer
summaries confer no receipts, publication or continuation authority.

## Final focused software evidence

Physical cwd `D:/Dev/Repositories/Circadian`, TEMP/TMP `D:/Dev/_Temp`, Python `-B`.
Linux uses `/mnt/d/` paths and TMPDIR too, with existing interpreter
`/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python`.
The reused finite observer declared at most five pytest invocations per platform,
180 seconds/512 MiB sampled RSS and 192 native attempts/1280 row exposures.
Other model/copy/readout limits are retained in stage receipts. No allowance
was reset after a failed attempt. Tests used no historical scientific JSON,
neural/physics workloads or final evaluation.

With `<stage>` = `artifacts/runs/fresh-paired-serving-20261029`:

```text
<python> -B <stage>/check.py <win|linux>-final tests/test_fresh_paired_serving.py tests/test_context_head_fresh_workflow.py::test_should_refuse_corrupted_owned_evidence_before_model_work tests/test_context_head_fresh_workflow.py::test_should_refuse_invalid_api_modes_before_data tests/test_context_head_fresh_workflow.py::test_should_refuse_invalid_cli_before_output_or_model tests/test_context_head_fresh_workflow.py::test_should_keep_failed_source_native_work_and_never_enter_private
```

47 passed per platform. Windows 8.0588414 s / 84,271,104 B sampled peak RSS;
Linux 10.977461797 s / 76,423,168 B. Each paid 75 native attempts/477 rows,
38 constructors/76 zero heads, 240 matrix rows, 245 contextual and 783 head typed
requests, 1023 sandbox steps, 20 teachers, 373 packets, 15,391 context/21,390 head
copy visits and 2,504,256 logical coefficient slots. All forbidden/historical
counters were zero. Literal source-budget fixture charges are distinguished from
actual native fitting. Current and joint fresh end-to-end fixtures both passed.

After the accounting-only repair, the unchanged native cases were reused and
these affected scalar cases were run on the final tree:

```text
<python> -B <stage>/check.py <win|linux>-accounting tests/test_fresh_paired_serving.py::test_should_preserve_unknown_work_when_delegated_report_is_lost tests/test_fresh_paired_serving.py::test_should_not_count_missing_child_work_as_zero_in_failure_totals tests/test_fresh_paired_serving.py::test_should_protect_existing_output_and_forward_exact_option
```

Four passed per platform: three new distinct cases and one repeated forward,
for 50 distinct focused cases per platform, zero skips. Windows 1.1965792 s /
74,256,384 B; Linux 5.005625234 s / 69,238,784 B. No models/fits/teachers/history
work; one context/two head snapshot visits, 192 logical slots.

Four-file Ruff check/format and mypy passed, with default and Linux mypy platform:

```text
.venv/Scripts/python.exe -B -m ruff check scripts/context_head_rehearsal.py scripts/context_head_live.py scripts/recurring_ticket_workflow_example.py tests/test_fresh_paired_serving.py
.venv/Scripts/python.exe -B -m ruff format --check scripts/context_head_rehearsal.py scripts/context_head_live.py scripts/recurring_ticket_workflow_example.py tests/test_fresh_paired_serving.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent scripts/context_head_rehearsal.py scripts/context_head_live.py scripts/recurring_ticket_workflow_example.py tests/test_fresh_paired_serving.py
```

The last command also passed with `--platform=linux`. Whitespace checks passed
with inherited LF/CRLF advisories. Lead reviewed actual source, procedure and
results. A bounded independent readonly source reviewer used four calls, found
the accounting issue, then two followup calls confirmed its repair with no
further blocker. No reviewer tests/models/raw JSON/writes; both exceptions ended.
This is independent source review, not independent experimental acceptance.

Retained failed Windows attempts: `win-owned` passed 16 cases before a malformed
vocabulary fixture failed in its constructor (1.1407189 s / 75,444,224 B, zero
native work); `win-combined` passed 26 before expecting a nonexistent stream
`complete` field after successful native work (3.3639093 s / 81,121,280 B,
18 attempts/120 rows, 11 constructors/22 heads, 120 matrix rows, 114 contextual/
212 head typed requests, 260 sandbox steps, four teachers, 73 packets,
4210 context/5906 head visits, 687,648 slots). Fixtures were corrected to valid
foreign schema and actual stream counters, without weakening production criteria.
Initial typing/fixture API errors and one no-op patch-anchor failure were fixed.
The accounting repair first failed nullable-sum typing; explicit narrowing fixed
it before final tests. All receipts/JUnit/temporary failures remain.

Windows used four of five invocations: 13.7600488 s cumulative, peak 84,271,104 B,
93 native attempts/597 rows. Linux used two: 15.983087031 s, peak 76,423,168 B,
75 attempts/477 rows. Software work is separate from the genuine case below.

## One prospectively frozen native qualification

Protocol, 60 declared slots and seven source/procedure bindings were frozen after
software/source-review gates. GitHub GO receipt 6493370883 preceded execution.
The single producer command was:

```text
.venv/Scripts/python.exe -B artifacts/runs/fresh-paired-serving-20261029/native.py
```

It invoked the actual existing parser/main/writer once with
`--context-return --separate-context-heads --paired-serving joint --output
<stage>/native.json`. This qualifies the main entry function; it did not separately
launch an installed or module subprocess. The observer refused historical source
reads, private adaptation, neural/physics/test imports and final/probe work.
Original source 12-update/30 s/512 MiB session remained active through child
6-update/60 s/512 MiB serving. Whole-case caps were 18 passes/120 rows,
120 s/512 MiB sampled RSS, metadata 30 s, output 4 MiB and stage 24 MiB.
Costs were flushed before the actual writer; the same sampler covered imports,
training, serving, writer and the single allowed raw metadata read.

EXIT 0. Measured 2.504336 s, sampled peak 57,544,704 B, output 1,427,895 B,
metadata 0.0104872 s. Exactly 18 new passes/120 rows: source 12/72 and child 6/48.
Nine constructors/18 zero heads, 84 matrix rows, 212 head typed requests,
260 sandbox steps, four teachers, zero traces/history/private calls,
4178 context/5842 head copy visits and 681,504 logical coefficient slots.
No random model initialization. All prospective limits held; sampled RSS is not
a hard OS peak. Source clock start 20838.9521311, last 20841.0158101, committed 12;
child start 20839.7659474, last 20841.0022574, committed 6. Neither origin nor
limits were renewed. Source replay zero, final source summary 12/72 with zero
score forwards. Source/actor/RNG preservation and shared witness checks passed.

| Fixed outcome | Observed result |
| --- | --- |
| Source ordinary four tasks at checkpoints 0 / 4 / 8 / 12 | 0/4, 2/4, 4/4, 4/4 |
| Source terminal stale / revoked-commit cases | 4/4 successes; 4/4 expected refusals |
| First original guard at tick 10 | Accepted, new 1 / old 1; publication c0, generation 1 |
| Child workers | All 24 complete; 19 successful |
| Fixed late ordinary ticks 13,15,16,17,18,19,21,23 | 8/8 successful |
| Published readouts | Old 4/4, new 2/2 |
| Full warm-source rollback at generation 2 | Old 4/4, new 0/2 |

Prepublication wrong-new choices at ticks 1/5 and expected permission refusals at
6/14/22 remain failures. Return TRAIN observations 12/13, labels 13/14, applied
at updates 14/16 alongside earlier 4/6/8/10; no second publication was needed.
Published actor success is not a separate final-candidate diagnostic. Source and
child full-state counters differ; no identical-state or equal-compute claim.

The case is now SPENT. Its one raw `native.json` metadata read is consumed; no
raw reread, rerun, rescore, capture repair, partial recovery or budget renewal.
Use `native-summary.json` and scalar work receipts afterward. Original failures
and spent 270/271/236/262 cases stay preserved. This result is one conventional
known-template delivery qualification. Joint retained the old tasks; it did not
recover from actual forgetting. Prior current-only forgetting/recovery evidence
is reused from 271 without retraining. No held-out, reliability, circadian,
statistical, matched-total-compute or repository-wide readiness conclusion.

Evidence: `artifacts/runs/fresh-paired-serving-20261029/`, including the protocol,
declared slots, source bindings, software receipts/JUnit, native producer,
attempt/body/final work receipts, scalar summary and closed raw capture.
Native outcome GitHub receipt 6493468807.

Final closeout verified all seven frozen source/procedure bindings unchanged.
The owned stage contains 35 files / 8,671,648 B, below its 24 MiB cap; this
check reads file sizes and pinned source/procedure bytes, not the closed raw
capture. Documentation-only additions followed the qualified source freeze.

Skipped: broad suite/CI, new model-family comparisons, nuisance retry, simulator,
another native platform/control/cohort, raw historical reconstruction,
independent experimental replication, wheel/install/clean-clone qualification.
These are separate gates, not implied by scoped tests or source review.

## Critical path and exact next action

Fresh ordinary serving now removes the historical-checkpoint delivery dependency.
Accepted learnability/probe and conventional learned-control work is reused;
there is no need for another update-count or provenance increment. Reliable
adaptation, broader held-out science, comparative model-family claims and full
release remain open. Nuisance 236 is FAILED/SPENT; 237 was a software repair only.

The existing installed wheel from 269 predates 270–273, so it cannot establish
portable delivery of the new useful command. Next action: inspect package
inclusion and the existing offline wheel workflow, open a bounded owning issue,
then build the current source into a new physical D directory and verify an
isolated installed command's origin/help/pre-I/O/output gates with zero model
work. Preserve the older wheel/environment. Freeze any necessary single installed
native qualification only after those gates pass. This advances a concrete
delivery requirement; it grants no fresh scientific control or old-case retry.
