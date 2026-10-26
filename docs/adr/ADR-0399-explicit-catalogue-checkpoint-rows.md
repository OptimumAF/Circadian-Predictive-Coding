# ADR-0399: Explicit catalogue rows at checkpoint and probe entry

## Context

The accepted joint catalogue composer executes54-row wakes through existing
native learners and the attribution stream. MiniGrid readout/probe admission
assumes nine-row wakes and therefore rejects these checkpoints. Ignoring native
row counters would break the existing state/exposure contract.

## Decision

Add optional `catalogue_rows=None` to the existing counter, bound snapshot,
TRAIN readout and isolated probe helpers. Default behavior, points and wire stay
unchanged. Explicit mode admits only exact native54 for the six original9-row
TRAIN sources and bound neural methods, native points0..128, and matching native
update/row clocks. All complete policy, encoding, optimizer, source, input,
checkpoint, budget and paid-evidence checks remain.

Report the original expected rows using guarded exact primitive fields. Probe
entry binds only the initial benchmark state; F and A-return retain their own
additional9-row update law and original separate charges. Refuse metadata changes
before later work and after success. Publish canonical initial exposure evidence
with permanent charges after a failed callback; retain failure and paid partials.

Why this: the actual dependency is an explicit exposure law, not a missing learner
or runner. Extending the existing probe in this increment avoids handing off
another almost identical counter prerequisite.

## Alternatives

- Broadly skip row counters: would admit unobserved state changes.
- Change the default to54: would invalidate accepted online clocks and evidence.
- Add another probe/runtime framework: duplicates existing owned-fork isolation,
  old-A readouts, return templates and budget handling.

## Consequences

Four native methods can score joint checkpoints. One native BP case executes
2joint updates,16isolated F updates with old-A readouts, and16A-return updates.
It verifies execution/isolation on synthetic fixtures, not useful acquisition.
163distinct scoped cases qualify through retained/extended/repaired gates, with
two original fixture failures kept; final scoped statics/review qualify.

Next expose this opt-in composition via the existing learnability API/CLI,
retaining default objectives/seed/metrics and source/probe-support admission.
A separately frozen small protocol must precede comparative scientific use.
Prior negative results and SPENT scientific grants remain unchanged.

[Evidence](../../artifacts/runs/catalogue-checkpoint-rows-20261018/RESULTS.md).
