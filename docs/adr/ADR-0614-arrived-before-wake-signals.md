# ADR-0614 — Observe arrived labels before deciding to wake

## Context

The accepted categorical observer runs after training and scores previous
parameters. It cannot decide whether an idle candidate should train. Existing
candidate feedback repeatedly scores the oldest unapplied FIFO batch, which is
useful for regression checking but cannot describe newly arriving distributions
while learning waits. Issue99 demonstrates partial signals and missed return
onsets; it does not establish controller utility.

## Decision

Use an explicit bounded outer reader in the existing live reference schedule.
Read each newly arrived ordinary TRAIN packet once on a detached current
candidate; reuse the existing metrics/detector/incidents with unchanged defaults.
Keep one current-state pin, bounded scalar records, paid costs and native-origin
binding. Carry only scalar alarms/clocks/versions to the pure reference choice.
Check the candidate and report again before existing wake admission.

Why this: observing arrivals must not require spending an update. Any current
alarm can request wake1, while the original FIFO, credits and shared gate retain
authority. No fresh labels means no new observation or stale alarm request.
Opt-in is restricted to explicit contextual live drift mode. Existing defaults,
candidate/task feedback and acknowledgement protocols remain separate.

## Alternatives

- Reuse the after-update observer: requires training to obtain a wake signal.
- Rescore the oldest ready batch: misses new arrivals while the queue waits.
- Change reference thresholds, skip FIFO data or add a new scheduler: unnecessary
  to expose this observation boundary and would confound interpretation.

## Consequences

Diagnostics spend prediction/copy/time/RSS even when no training occurs. A
current alarm may train an older pending batch and produce no task improvement.
Record that as a result. Keep final/probe data unavailable, failures permanent,
and source/actor permissions and credits unchanged. This is a fixed heuristic
input, not evidence of a deployable detector or a learned-controller advantage.
