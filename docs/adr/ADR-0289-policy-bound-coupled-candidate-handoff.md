# ADR-0289: Policy-bound coupled candidate handoff

Date:2026-10-15. Status: accepted design; validation recorded in R3.7s results.

## Context

Complete learner checkpoint envelopes now preserve native state and separate
learning/inference settings. The existing candidate controller already owns
same-process capture, preparation charging, cursor/resource checks and atomic
owner publication. Its callbacks still need a concrete coupled implementation.

## Decision

Add a small adapter that owns a detached policy/binding/limits tuple, builds and
digests with the existing complete bounded envelope, and admits only exact
coupled learners whose complete policy equals the original policy. Compose it
with the existing controller without changing app/core/native contracts.

The runnable integration demonstrates wrong-policy refusal, successful second
preparation, continued delayed-label updates under original resource objects,
old-owner/token refusal, unchanged serving actor, and matched full-state controls.

## Alternatives

Native-only digests omit future update behavior. A new controller or recovery
framework duplicates accepted ownership logic. Encoding live runtime authority
in the learner envelope confuses data validation with trusted in-process rights.

## Consequences

The adapter supplies reusable complete-policy callbacks with existing byte/array
bounds and no training or prediction. The original controller remains the sole
handoff authority and charges failed preparation. This slice requires surviving
actor, clock, budget and sharing objects; it does not close process/lifecycle
recovery or establish a scientific learning benefit. No dependency or environment
configuration is added.
