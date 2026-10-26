# ADR-0358: Continue fixed MiniGrid BC with the existing reward runtime

## Context

The cold MiniGrid reward route is implemented and accepted, but all eight sampled
episodes returned zero. The fixed unweighted A128 learner supplies17/18 raw
labels and one greedy native goal; the distinct weighted objective reverses the
failed cue without joint acquisition or live-return gain. Neither result proves
useful retention/recovery. The typed demonstration/reward learner already owns
one native recurrent optimizer; the MiniGrid recipe lacks a warm composition.

## Decision

Add an explicit warm-start option to the existing reward recipe and CLI. Reuse
the fixed unweighted reference/source DTOs, original128 A packets, seed, objective
and checkpoint law. The recorded JSON contains no native tensors, so regenerate
the required transient warm checkpoint once as newly charged work. Check all
five raw points and both native cues at0/128 against exact saved observable
values before admitting reward work. A failure is retained and stops the case.

Restore the complete native checkpoint into TorchDemonstrationRewardLearner,
including optimizer momentum, gradients and counters. The existing actor/inbox
runtime and TRAIN collector then process the unchanged eight-episode schedule
with the original PCG64 draws and delayed matching-ID releases. Keep warm charges
and reward phase charges/receipt clocks explicit while carrying native counters
from128. The whole-case deadline also bounds the existing45-second reward phase.
Cold defaults and their descriptor remain unchanged.

Why this: it supplies the missing real-domain continuation using existing ports,
without introducing another trainer or tuning a loss after a negative result.

## Alternatives

A larger cold reward dose or selected rewarding seed would not test this missing
composition. Reusing JSON metrics cannot restore opaque optimizer state. Loading
weights alone discards momentum/gradients and changes the continuation. A new
checkpoint serialization framework is unnecessary for one local integration.

## Consequences

Reconstruction and parity readouts consume new time, memory, updates and physics;
historical budgets stay spent. Greedy one-cue success does not guarantee sampled
reward, and TRAIN collection returns are not independent policy evaluation.
The smoke establishes integration only. A future reward-effect comparison needs
shared full warm forks and carried-momentum/additional-update controls under a
separate frozen protocol. Keep all negative cues and unfinished scientific,
retention/recovery/generalization and release requirements.

## Observation

The sole original smoke reproduces all exact warm readouts and restores native
state, then stops at the45-second reward budget after six commits. Preserve that
incomplete result and134 paid updates. Saved-record component timing supports
removing duplicate parent serialization only inside the callback whose caller
immediately validates the entire warm record. All outer/state/receipt/counter
checks remain.191 focused cases pass after repair; no native retry followed,
so eight-commit completion and whole-route speed remain unproven. Detailed
source windows, failures and costs are in the stage RESULTS.md.
