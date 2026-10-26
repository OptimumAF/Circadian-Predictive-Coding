# ADR-0608: Declare second-window feedback separately

## Context

The existing two-window runtime accepts current-generation feedback after one
guarded publication, but its original second window has only old-task rows.
The older live recipe port replaces the entire stream and has one label delay;
using it for a noisy second window would silently change the first window.

## Decision

Add an explicit `followup_training_recipe` port and clean/noisy frozen choices
through the existing builder, schedule and CLI. Retain original first8 packets;
compose8 twelve-row new+old packets and distinct second-window delay2. Both
choices have identical IDs/features/order; only retained invoice row7 differs.
Validate immutable recipe and owned TRAIN payloads at owner/worker boundaries.
Reuse original lifetime runtime/budget/clock/gate, verified publication,
current actor receipts, original guards and full rollback. No second publication.

Why this: a separate small port expresses the actual experiment while keeping
existing defaults and authority. A timeline method pairs labels with the correct
window delay without adding another scheduler. Omitted second delay inherits
the original global delay, including existing global-delay2 calls.

## Alternatives

Relaxing the old flag exclusion would alter first-window data and clocks.
A separate orchestrator would duplicate admission and publication logic.
Ignoring generation changes or renewing the budget would weaken the contract.

## Consequences

The caller explicitly declares false-label metadata; guard truth stays intact.
Private adaptation and deployed behavior remain separate. One matched local
pair may describe known-template noise effects but cannot establish robustness,
CPC/controller benefit, generalization, long-duration safety or release readiness.
Issue92 and its local protocol/results preserve paid failures and open parents.
