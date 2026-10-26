# ADR-0407: Dispatch weighted PC through the existing schedule

## Context

Weighted categorical PC math and complete atomic PC/neutral episode updates have
passed independent correctness and full-state continuation checks. The existing
TRAIN schedule previously admitted feature BP only. It already owns complete
episodes and validates input arrays, prefixes, detached candidates and donors.

## Decision

Extend feature policy admission to PC/neutral and dispatch its guarded candidate
to the accepted complete weighted-PC episode callback. Use a distinct objective
ID with the original sealed native method/policy. Preserve the BP callback and
recurrent path, schedule witnesses, current-episode-only weight derivation,
source/actor isolation, detached publication and fork/restore behavior.

## Alternatives

A separate PC scheduler would duplicate state and release checks. Calling native
train_batch would omit explicit class weighting. Reusing BP's diagnostic ID would
misstate the objective. None is needed; the existing guarded candidate path can
compose the accepted episode implementation directly.

## Consequences

The existing delayed TRAIN stream now runs complete weighted PC/neutral schedules.
Public API/CLI composition and scientific studies remain unfinished. The current
public weighted BP factory admits128online nine-row episodes; joint54-row mode
explicitly rejects weighting and needs separate aggregate semantics. Extend the
online public route explicitly before considering a separately frozen comparison.
Correctness does not establish task adaptation or a circadian advantage.
