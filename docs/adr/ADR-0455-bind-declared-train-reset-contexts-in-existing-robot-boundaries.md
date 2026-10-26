# ADR-0455: Bind declared TRAIN reset contexts in existing robot boundaries

## Context

The nearest-observation control can reproduce the qualified seed17 trajectory.
Acquisition and retention require a second TRAIN context and a pre-append score.
The low-level port already accepts reset seeds, but the teacher and frozen smoke
previously hardcoded17. Probe23/final29 are reserved. Reset contexts of PickCube
are not distinct tasks.

## Decision

Use a small pure mapper accepting only exact native17/19, with default17 context
metadata preserved. Add keyword-only teacher seed configuration and matching CLI
options to the existing teacher and frozen actor boundary. Bind actual port
reset, episode identity, context, TRAIN role and receipts. Validate before scene
creation; preserve existing teacher/actor algorithms and state guards.

Freeze the single A17→B19→fresh A17 protocol before observing19. Require teacherB
qualification and executed A-only Bbefore before appending allB labels once.
Retain allA labels/frozenA statistics, account for memory and distance work, and
preserve every negative result. Reuse the accepted unchanged A-before reference.

## Alternatives

Arbitrary caller seed/role/context combinations provide unnecessary authority.
Relabeling probe data would compromise isolation. Another runner would duplicate
existing native orchestration. Automatically trying seeds after a teacher failure
would select the protocol from observed outcomes.

## Consequences

Existing default17 behavior remains available. One signature-only test is revised
prospectively to require the sole keyword option; qualification and behavioral
criteria are unchanged. Extension to another context requires an explicit protocol
and isolation review, rather than selecting observed successful seeds. The lookup
control has different memory/compute/capacity from neural controls. Neither source
correctness nor same-initial-state replay proves scientific or release readiness.
