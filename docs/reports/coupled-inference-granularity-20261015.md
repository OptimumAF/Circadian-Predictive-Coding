# Granularity-aware inference comparison — 2026-10-15

## Outcome

The new whole-iteration admission/qualification boundary passes115 focused
tests and scoped lint/format/Windows+Linux type checks. The ONE prospectively
frozen six-case experiment is **incomplete**: fixed seed59 exceeded its3ms
observed complete-iteration ceiling. Its scores remain null and known work is
retained. No stopping savings or complete three-pair comparison is demonstrated.

The original10ms experiment remains unchanged. This distinct v2 allocation was
chosen only from its retained timing envelope, not learning scores. The existing
fixed-step baseline and default adaptive thresholds, seeds, data, metrics and
comparison direction are unchanged. Protocol, source/design reviews and receipts:
artifacts/runs/coupled-inference-granularity-20261015.

## Frozen method and all results

Allocated30ms/case, observed30..33ms inclusive; maximum known complete interval
3ms inclusive. First interval starts at window_start; later starts partition full
native iterations; final span includes terminal checks. All native calls, copies,
stopping/check arithmetic and in-loop bookkeeping count inside the wall window.
Common setup/deferred scoring count only in total execution. No preemption,
calibration, warmup, retry, replacement or extension. Seeds23/41/59 with fixed
counterbalanced order, exact paired initialization and train/final isolation.

| Seed | Method | Status | Updates | Window ms | Max interval ms | Final BCE | Accuracy |
|---|---|---|---:|---:|---:|---:|---:|
|23|fixed|complete|26|30.4745|2.8328|0.709463|0.476563|
|23|adaptive|complete|31|30.5173|1.2694|0.693234|0.523438|
|41|adaptive|complete|31|30.0619|1.3106|0.545074|0.953125|
|41|fixed|complete|32|30.1391|1.6749|0.543403|0.953125|
|59|fixed|failed: interval ceiling|7|8.4460|3.1088|null|null|
|59|adaptive|complete|30|30.1121|1.6164|0.595619|0.789063|

Every adaptive update (92total) used all6latent steps, with no fallback. Fixed
and paired complete means/intervals are null because seed59fixed is unqualified.
Partial observed means retain diagnostic meaning only; they are not complete
three-seed estimates. The JSON retains every initial score, final score, interval,
work counter, ordered failed row and complete/partial seed statistic.

Actual157attempted=157committed updates; execution0.1813525s, sampled absolute
peak RSS39,833,600B, output192,396B, CPU1. Process command exit2/empty stderr is
the expected incomplete-report signal. All global resource caps pass. Known
work942latent steps,1099inference+157parameter energy evaluations,644gradient
norm checks,552energy comparisons and2512training-row exposures. These counts
are not FLOPs and do not replace whole-window timing. Eight engineering calls
plus157scientific=165 of512; unused24engineering/323science CLOSED.

## Interpretation and next work

An indivisible iteration can still exceed a prospective timing ceiling despite
a conservative prior envelope. The new boundary detects and retains that case;
it does not turn it into an admissible comparison. One local process and three
seeds on fixed data do not isolate scheduling, cold-start and initialization
variation or support a general speed/quality claim. Different update counts also
mean different training-row exposure. No final-score feedback enters decisions.

Keep original R1.5/R1.5b unchecked and preserve both failures without widening
tolerances or running replacements. Continue independent runtime functionality:
add an explicit bounded codec for actual coupled PC/CPC full snapshots through
the existing checkpoint codec port, with complete detached round-trip/corruption/
continuation controls before any disk/crash recovery claim. This does not spend
the closed timing allowance or resolve the incomplete comparison by assertion.
