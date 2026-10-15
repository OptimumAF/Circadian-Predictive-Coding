# ADR-0258: whole-window coupled inference comparison

## Context

R1.5a implemented guarded inference. Its latent/evaluation counts do not measure
stopping arithmetic, parameter validation, resource checks or orchestration.
R1.5 requires a comparison against fixed PC at a declared matched basis.

## Decision

Use equal allocated training wall-time windows, timing all in-loop work and whole
native calls. Admit updates only before the deadline, without preemption; qualify
observed windows prospectively within 10% overshoot. Preserve complete failures
with no scores/retries/extensions. Clone state/data before callbacks and score
detached initial/final states only after training. Keep exact shared initialization,
fixed data/default policies, ordered seeds, all counters and paired summaries.

Why this: whole-window timing directly includes overhead that diagnostic counters
cannot represent. It permits an honest negative or incomplete outcome.

## Alternatives

Equal update count or latent-step count would not match time/work including policy
overhead. Timing only the native kernel excludes orchestration. Padding a completed
short run would obscure cap exhaustion. Retrying overruns would select timings.

## Consequences

Actual wall times can differ by up to the explicit tolerance; exposure can differ.
Short windows, scheduler/cache effects and n3 constrain inference. Fixed-data
intervals mix initialization/runtime variability. Neither CPC benefit nor a general
speed/convergence claim follows. Broader protocols require separate frozen budgets.
