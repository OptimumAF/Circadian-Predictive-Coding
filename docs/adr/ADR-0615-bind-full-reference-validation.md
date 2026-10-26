# ADR-0615 — Bind full fixed reference execution for validation

## Context

Existing controller chronology separates train, validation and sealed evaluation,
but its ReferencePolicy does not describe optional candidate feedback and receipt
acknowledgement. The accepted102 outcomes are already known development evidence.

## Decision

Bind three explicit complete execution modes before any source access in the
existing live outer composition. Reuse102 only as controller_train metadata;
admit one new stress validation per binding through ControllerChronology. Keep
its train reader closed afterward, issue the ordinary later freeze receipt and
never supply/release a sealed callback. Preserve original inner TRAIN ownership,
clocks, source, guards and budgets. Validate all paid cases before public scores.

Why this: it fills the missing flag binding with ordinary existing interfaces and
does not require a controller, scheduler or role-framework replacement.

## Alternatives

Binding ReferencePolicy alone omits two behavior-changing flags. Relabeling102 as
validation would misrepresent already known outcomes. A new orchestration layer
would duplicate the existing live runner and chronology.

## Consequences

Fixed outer validation can contain ordinary arrived-TRAIN inner adaptation. It
does not fit a controller or prove unseen semantic-task generalization. The sole
joint stress cohort rejected both trained candidates and showed no deployed gain.
Configuration freeze before validation is distinct from the later core freeze
receipt. Trusted local Python composition is not a hostile-code sandbox.
