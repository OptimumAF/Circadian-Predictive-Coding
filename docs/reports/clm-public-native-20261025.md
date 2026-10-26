# Released CLM: public-observation TRAIN and complete tasks

[Issue147](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/147),
R5.6-CLM-Q.X. The sole frozen native case completed: **2/3 raw task successes**,
one stale-observation recovery, one expected effect-time permission denial and
zero wrong completions. The model proposed the revoked commit; the external
dispatcher denied it. This is useful known-task learning with external admission,
not intrinsic model refusal or real host containment.

TRAIN fit improved **4/12 to10/12 rows**. Legacy/password ready choices remain
wrong. No whole-corpus fit, chronological retention, fresh-task transfer,
generalization, calibrated confidence or circadian advantage is established.

## Fixed procedure and comparison scope

The original twelve dispatcher-verified TRAIN targets, five active candidates,
eight delayed catalog-CE passes, consent, clocks, native4096-to-512 geometry,
AdamW/OneCycle settings and original head initialization stay fixed. The explicit
public-observation recipe replaces teacher ` train` with the exact shared live
` please` construction. Six unique queries and five actions encode once; all
96query/40candidate exposures and480logits precede effects. No teacher fallback,
phase mask, extra fit, replay, probe/final data, tuning, early stop or promotion.

Use one cached Qwen3-8B at revision
`b968826d9c46dd6066d109eabc6255188de91218`, offline WSL Torch2.14.0+cpu /
Transformers5.10.2, CPU2. The loader reported no missing/mismatched keys/errors;
`lm_head.weight` was unexpected for Qwen3Model as already documented. Physical
random initializer elements were zero; meta initializer visits are a separate
counter. No download, installation or substitute/untrained model.

Observe only the three mandatory learned scenarios, maxsix predictions each.
Reuse [#36's accepted unchanged0/3 control](native-clm-ticket-workflow-20261025.md)
without repeating its18predictions. Assets, instructions, public query semantics,
candidate order and effect assumptions remain; its original costs stay sunk.
The earlier trained recipe's1/3 remains context. New diagnostics, TRAIN text/
batching and observation work differ: this is not matched-total-compute or
isolated causal evidence. The reference gate checks only the learned first
request, not numerical reproduction of the old control under new batching.

Development episodes intentionally reuse known TRAIN observation patterns. Their
different metadata source group does not make them held-out examples. No protected
final evaluation was accessed; this case tests known-task reproduction/usefulness.

## Native outcomes

| Scenario | Actions | Effect statuses | Outcome |
| --- | --- | --- | --- |
| Ordinary invoice | open, inspect, commit | applied x3 | Correct committed inspect |
| Stale parcel | open, defer, open, defer, commit | applied, stale_revision, applied x3 | Correct committed defer; recovery |
| Revoked invoice | open, inspect, commit | applied, applied, permission_denied | Expected external denial; raw failure |

All three episodes are complete,11predictions,0wrong completions and no uncertain
prediction work. Raw success remains2/3; both completable cases succeeded (2/2),
the single stale event recovered (1/1), and the triggered revocation was denied
(1/1). Recovery here is fresh observation after a stale revision, not relearning
after forgetting. No episode-time model update occurred. All five active tools
were candidates; the dispatcher did not repair wrong model choices.

Before the first effect, an independently owned uncached predictor returned the
same ordered IDs, selected ID and probabilities: maxdifference0.0<=2e-6. Query
encoding and both head projections remain fresh; raw action features alone reuse
paid TRAIN extraction. No numerical fallback or tolerance change.

## TRAIN fit, including the negative endpoints

Margin means target probability minus the largest OTHER probability, not logits
or calibrated confidence. Twelve rows contain six unique queries: sealed and
staged each repeat four times.

| Phase | Correct before | Correct after | Mean margin before | Mean margin after |
| --- | ---: | ---: | ---: | ---: |
| Sealed |4/4|4/4|0.325739|0.569877|
| Ready |0/4|2/4|-0.774788|-0.044123|
| Staged |0/4|4/4|-0.626383|0.774497|

| Ready target | Before selection /margin | After selection /margin |
| --- | --- | --- |
| Invoice /inspect | open /-0.815141 | inspect /0.011897 |
| Legacy /defer | open /-0.740715 | commit /-0.377037 |
| Parcel /defer | open /-0.705029 | defer /0.282972 |
| Password /inspect | open /-0.838267 | open /-0.094324 |

Invoice's margin is small. Legacy/password have wrong final teacher choices and
no observed native episodes here. Maintaining sealed accuracy in a joint fit is
not continual retention. All24row probability/margin readouts remain; none was
selected or dropped. Readout helper times0.304584190s and0.707938211s include
their admission/captures/read/summary, excluding outer guards and file writing.

## Paid work and limits

Executed once, terminal exit0, no timeout:
```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-public-native-20261025/execute.py
```

Inner427.958152226s<540; outer444.605806400s<570; peak RSS16,312,520,704B<22GiB.
Fresh RAM Windows36,135,944,192B /WSL32,609,181,696B, no competing WSLPython>256MiB.
Asset verification completed at200.379615255s, heads loaded at208.298637316s.
These component timings are not a prior-run speedup comparison or serving SLO.

| Physical counter | Actual |
| --- | ---: |
| Encoder constructions /forwards /sequences |1 /13 /28|
| Tokenizations /IDs /padded slots |28 /281 /327|
| Head objects /forwards /projected rows |32 /44 /242|
| Backward /optimizer /scheduler updates |8 /8 /8|
| TRAIN query /candidate rows /logits |96 /40 /480|
| Constructor initial scheduler step |1|
| Physical random initializer elements |0|
| Meta initializer element visits |15,173,957,632|
| Returned source captures /tensor elements |65 /4,571,148,445|
| Returned caller Torch-RNG elements |717,952|
| Capture +RNG +prepaid2B othercopy reserve |6,571,866,397 (<12B)|

Script source guards returned37captures /2,795,564,489elements. Returned copies
plus reserve are not all allocations/bytes/FLOPs. Unused unchanged actor
construction/copies remain paid. Loaded-head/source/owner/feature/gradient/mode/
optimizer/scheduler/counter/caller-Torch-RNG guards passed: one binding/source
snapshot, no restore or budget/clock/receipt renewal. All44source/proof pins held
before/after preflight and case. Stage185,939B/log1,713B at supervisor check, below
8MiB/512KiB; actual/ledger below2MiB/128KiB. Later review/report text is separate.

Actual SHA256:
`d8789882072d05eff4ed76f38a0524df32b1f3d3fcc506a96ba60559475d9357`.
Original actual/execution/case-work JSON remain untouched; raw logits/probabilities,
observations/capabilities/revisions/transitions and all work remain in actual.json.

## Qualification, failures and review

Numerical gate commands `.../launch.py windows-main` and `.../launch.py linux-main`
each passed2ZERO-TRAIN cases,0fail/error/skip: exact positive and finite nonidentity
drift refusal before effects. Windows4.3973247s /RSS246,366,208B; Linux12.709800955s
/RSS275,333,120B. Each12tinyheads8fw24projectedrows/2FakeBackbones6fw24tokensIDs/
slots/24localteacher dispatches;0TRAIN/backwards/scheduler/random/sourcecaptures,
RNG192,128elements+1MiBreserve. No Transformers/assets/physics/fullfit rerun.

Artifact Ruff/format/fresh default+Linux-target mypy passed,51.653606s cumulative,
no failed checks. A later reporting patch failed an unrelated context before any
file changed; simplified to direct report edits. No native/fixture/cap revision.
Unchanged #145/#146/math/encoder evidence reused. Finite separate source/protocol
review returned GO before launch with no reviewer tests/models/writes/network.
Final source-pin/REVIEW assembly and native/results are lead SELF-REVIEW, not
independent native/scientific acceptance; the review exception ended.

Sole post-terminal stdlib saved review passed0.0153312001sbody /0.342625sparent,
21,135,360Bpeak,117,043input bytes,0model/scoring/effect or NumPy/Torch/Transformers
imports, no read errors; allowance now SPENT. Detached saved-review.json retains
the complete summary. No original raw result was reparsed to choose another run.

Skipped: actual Linux Ruff/mypy (unavailable; Windows Linux-target used), broad
unchanged suites, extra seeds/doses/baselines/probes/final data, cleanclone/install/
GPU/CI/UI/release and independent native/result review. No production code,
dependency, download, commit/push, raw upload or unrelated edit. All historical
negative/failed/unknown evidence and original consents/budgets/charges remain.
Historical plans/log stay frozen; #147/#18 are live. Full goal remains ACTIVE.

## Concrete next action

Two known ready-state targets still fail TRAIN fit. Existing catalogfitter
hardcodes8 in policy admission/budget/loop. Add an explicitly bounded pass option
there, default8, requiring a fresh matching native policy and original nonrenewable
budget; qualify counter/clock/default/mismatch behavior with literal fixtures.
Then freeze ONE32-pass TRAIN-only overfit qualification before any pretrained
call. A32-step OneCycle schedule differs from8; it is not continuation or renewal
of this spent learner. No automatic further horizon search, task-score tuning or
circadian mechanism follows. Preserve both missed targets and all8-pass results.
Full-corpus/fresh-task learning, retention/return, broader R2/R5.1/R5.3/G5/R7 and
release remain open; resolve ordinary fit before stronger adaptation claims.
