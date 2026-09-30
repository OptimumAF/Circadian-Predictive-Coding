# ADR-0136: Stop toy execution on observed process RSS

## Context

The opt-in toy budget already limits wake updates, elapsed time, replay
exposure, and adaptive hidden width. Width does not cover the dataset, Python
runtime, baseline models, or temporary allocations. The repository already
has a Windows/Linux `ProcessRssSampler` used for benchmark telemetry. Its
samples are absolute current-process resident bytes, not memory attributable
to a particular model.

## Decision

Add a positive `max_process_rss_bytes` to the execution budget and a matching
toy baseline CLI flag. Start one sampler before toy dataset/model construction
for each invocation. It takes a 5 ms background sample and explicit samples
before wake updates, before sleep, and before final-test release. The runner
stops with `max_process_rss_bytes` when the observed high-water exceeds the
cap. It also checks the final sampler observation before returning a scored
result, so the CLI cannot publish an over-cap result. A host that cannot
measure RSS raises a typed error before constructing toy resources; the
CLI state records `error/process_rss_unavailable`.

The CLI writes `work.process_rss` with the scope, PID, start, observed peak,
sample count, and interval for that attempt. A resume starts a new sampler
and records a new segment, even if it restores training from a checked
checkpoint. RSS is never described as durable checkpoint work. Older v1
run states remain resumable with their original hash/cursor identity; the
new budget and work fields are additive. The scientific config and fixed
v14 protocol do not change.

## Alternatives

- Infer memory from model parameter bytes: rejected because Python, data,
  replay, baseline models, and temporary arrays are outside that estimate.
- Subtract the start RSS: rejected because the requested limit is absolute
  current-process residency; subtracting would silently change its scope.
- Put the RSS peak in the checkpoint: rejected because a resumed process
  gets a new RSS baseline and possibly a different PID.
- Claim a hard allocation bound: rejected because polling cannot prevent a
  transient allocation before the next checked boundary.

## Consequences

The cap is a measured soft ceiling. The sampler may observe a transient
peak between checked boundaries, causing a stop at the next check. A short
peak that no sample sees can be missed. The process may temporarily exceed
the cap before a stop, and cleanup can change RSS after the final check.
The final-test role remains sealed when the before-final check stops a run;
an over-cap observation during final scoring suppresses publication but
cannot undo that scoring. No baseline, seed, metric, old result, or fixed
v14 artifact is changed.
