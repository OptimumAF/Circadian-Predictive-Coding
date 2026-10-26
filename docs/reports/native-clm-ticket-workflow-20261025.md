# Released CLM: complete local ticket episodes

Issue [36](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/36),
R5.6-CLM.X. The original single native observation completed. The unchanged
released heads solved **0/3** episodes; the trained heads solved **1/3**, recovering
from a stale parcel observation. Both failed ordinary invoice and never reached
the commit-revocation event. Zero permission denials therefore demonstrate no
native permission-refusal behavior in this case. This is a completed bounded
investigation with limited task usefulness, not reliable adaptation or a CPC win.

## Fixed procedure and source assumptions

All eight catalog cross-entropy updates precede all effects: the original twelve
consented TRAIN demonstrations, 96 query-row exposures, 40 candidate-row exposures
and 480 logits. Frozen Qwen3-8B encoder, pinned revision
`b968826d9c46dd6066d109eabc6255188de91218`, original 4096-to-512 heads, CPU two
threads, cached Torch 2.14.0+cpu / Transformers 5.10.2 in WSL. No extra fits,
teacher fallback, legality mask, final/probe examples, promotion or external tool
handlers. The two frozen head owners see the same five active candidates.

Twenty of 21 earlier action-reuse source pins remained unchanged. The unrelated
current checkout appends ` please` to public episode queries; TRAIN queries retain
their original ` train` suffix. This change was preserved and explicitly frozen
before the native run in `CURRENT_PROTOCOL_20261025.md`. Earlier BP/Open-Jev
results are contextual references, not matched-current-input comparisons. The
suffix difference is a visible TRAIN/runtime distribution difference; this run
does not identify it as the cause of failure.

Raw action embeddings already paid during TRAIN extraction are reused. Query
encoding and both head projections remain fresh. Each head owner's first request
was independently encoded uncached before its effects: both numerical references
returned the same ordered IDs, selected ID and **zero probability difference**
(frozen tolerance 2e-6). This checks those two requests, not all possible inputs.

## Actual episode outcomes

Every episode is complete, with all prediction returns validated and no unknown
unreturned prediction work. `wrong_phase` and stale refusals apply no local effect.

| Owner | Scenario | Actions | Transition statuses | Outcome |
| --- | --- | --- | --- | --- |
| Unchanged | Ordinary invoice | open x6 | applied, wrong_phase x5 | Failed, ready |
| Unchanged | Stale parcel | open, open, commit, open, open, open | applied, stale_revision, wrong_phase, applied, wrong_phase x2 | Failed, ready |
| Unchanged | Revoked invoice | open x6 | applied, wrong_phase x5 | Failed; revocation not reached |
| Learned | Ordinary invoice | open x6 | applied, wrong_phase x5 | Failed, ready |
| Learned | Stale parcel | open, defer, open, defer, commit | applied, stale_revision, applied, applied, applied | Correct committed defer; recovery |
| Learned | Revoked invoice | open x6 | applied, wrong_phase x5 | Failed; revocation not reached |

There are 35 typed predictions, zero wrong completions, zero permission denials
and zero external effects. For the learned stale case the effect-time revision
guard refused the obsolete defer, then a fresh open/defer/commit sequence solved
the task. Both actors encountered one stale event; only learned recovered. This
is error recovery within one known task, not evidence of old-task recovery after
forgetting, unseen-task transfer, calibration or general reliability.

Original observations, requests, returned raw probabilities/logits, all actions,
environment events, effect-time capabilities and transitions remain in
`artifacts/runs/clm-native-ticket-workflow-20261022/actual.json` (SHA256
`79293949c303a2b2c2de9fee07cb74b3dce8a124b286579b2df189bfc2e64e6e`).
No task outcome selected optimization or further work within the fixed run.

## Paid work and limits

Command, executed once, terminal exit 0:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-native-ticket-workflow-20261022/execute.py
```

The existing supervisor checked the actual case, numerical references, work
equalities, current source pins and output sizes. Inner elapsed **534.566583159s**
under 540s; outer **555.4470415000105s** under 570s. Sampled peak RSS
**16,303,136,768 bytes**, under 22GiB. Fresh available Windows/WSL memory exceeded
24GiB with no competing WSL Python process. No timeout or native retry occurred.

| Meter | Actual |
| --- | ---: |
| Physical encoder constructions / forwards / sequences | 1 / 38 / 58 |
| Tokenizations / token IDs visited / padded slots | 58 / 629 / 695 |
| Head objects / forwards / projected rows | 30 / 90 / 358 |
| Backward / optimizer / scheduler update attempts | 8 / 8 / 8 |
| Scheduler constructor initial step | 1 |
| Physical random initializer elements | 0 |
| Meta initializer element visits | 15,173,957,632 |
| Returned source captures / tensor elements | 117 / 8,726,702,129 |
| Returned caller-RNG elements | 1,375,232 |
| Capture + RNG + prepaid 2B other-copy reserve | 10,728,077,361 (<14B) |
| Raw action elements per predictor owner | 20,480 |

Copy element counts are scoped to observed returned captures/RNG plus the
declared reserve, not all allocations, total bytes or FLOPs. Source guard captures
alone returned 97 times / 7,328,912,309 tensor elements. The source learner retained
eight commits, one serving snapshot, no restores or episode-time training; its
bundle/state/owner/work and loaded heads/caller RNG were checked throughout.
Both predictor owners returned all their requests (18 unchanged, 17 learned);
binding and predictor unknown-work flags are false. Parent source pins remained
exact. No budget, clock, receipt or source authority was renewed.

Native stage/log size at supervisor check: 340,397 / 7,320 bytes, below 8MiB /
512KiB; postflight documentation adds small separate storage. The log retains
`lm_head.weight` unexpected for Qwen3Model (no missing/mismatched keys) and the
existing staged OneCycle optimizer-step warning. Neither warning was hidden or
repaired in this experiment.

## Engineering qualification and review scope

The previous observer's literal [1,1] swap injected no drift. The repaired
negative fixture now proves finite, nonidentity drift before testing refusal;
the tolerance/order/selected-ID/no-effect criteria are unchanged. Fresh negative
and full-flow positive checks passed once each on Windows and Linux, two distinct
cases per platform. Each platform paid eight Tiny TRAIN updates / 96 query rows;
pretrained work zero. Current-source qualification and exact commands, source
override, counters, timings, failures and scoped statics are in
`artifacts/runs/clm-workflow-drift-fix-20261025/RESULTS.md`.

The finite separate read-only source/protocol review returned GO before launch.
It did not independently accept this native result; the final qualification
builder and assembled receipt were lead-reviewed only. This report is lead
source/procedure/result **self-review**. Older accepted math/encoder/baseline
evidence was reused within unchanged assumptions, not rerun.

The first stdlib saved-result summary failed a Windows ctypes pointer-width
signature after reading its three files: exit1, parent0.595473s, RSS unknown.
Failed source/receipt remain. One prospectively amended engineering pass added
the signature and succeeded: exit0, body0.0118133001s / parent0.3672198s,
21,254,144B peak RSS, 220,172 input bytes, zero NumPy/Torch/Transformers imports
or model/scoring/effect work. `saved-review.json` is the detached summary; the
original raw reports were not rewritten. That review allowance is spent.

The summary helper's initial Ruff check passed and format check failed. Ordinary
formatting corrected it; final Ruff/format passed and parsed AST exactly matches
`review_saved_executed.py`, preserving executed behavior without another result
read. The failed helper snapshot remains unformatted historical evidence. No new
production module or public typed interface was introduced by this helper.

Skipped: actual Linux Ruff/mypy (unavailable; Windows fresh default/Linux-target
checks passed), broad unchanged suites, clean-clone/release/install/GPU/remote CI
and independent native/result review. No dependency changes, downloads, commits,
pushes, raw uploads or unrelated source edits. Historical failed budgets and
negative/incomplete studies, including #34/#143, remain unchanged and unrenewed.

## Next useful work

Original R2.1 diagnostics and isolated probes already exist and are accepted;
do not rebuild them. Broader reliable new-task learning, retention, return,
R5.1/R5.3/G5/R7 and release remain open. This case supplies no new circadian,
matched-compute, pretraining-exclusion or real host-containment conclusion.

Why this next: the native full-task trace exposes a concrete failure at the
ready-invoice decision, while the current runner reports only aggregate TRAIN
loss. Reuse the existing catalog `predict_choices` and TRAIN-readout pattern to
add an explicit default-off, phase-specific TRAIN fit/action-margin diagnostic
to the existing ticket runner. Qualify source/head/RNG isolation and paid readout
cost on tiny fixtures first. Do not reconstruct/refit this run's unsaved heads,
retry its native cohort, change its training strings or tune dose from these
outcomes. Any later native diagnostic requires its own distinct justified frozen
protocol; the present native allowance is spent.
