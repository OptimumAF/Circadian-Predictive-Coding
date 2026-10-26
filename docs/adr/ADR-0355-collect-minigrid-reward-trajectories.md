# ADR-0355: Collect native MiniGrid reward episodes into the existing DTO

## Context

The existing recurrent native objective and generic reward-update port support
148 public coordinates and seven actions. Actual MiniGrid routes so far use
demonstrations and greedy evaluation; the custom8/K5 reward collector cannot
represent this native task. Native rollouts often truncate and cannot use the
demonstration conversion, which requires genuine termination.

## Decision

Add an outer adapter that samples TRAIN-only predefined MiniGrid contexts with
the existing raw cumulative-mass law and caller-owned PCG64. Bind the full frozen
behavior checkpoint and zero-reset recurrent session. Construct the existing
PolicyGradientEpisode directly from returned native transitions. Preserve both
native end flags and use terminal precedence if both are true; the existing
reward objective has no bootstrap. Reuse existing state comparison and native
environment lifecycle rather than adding another trainer or runtime.

## Alternatives

Teacher conversion rejects truncated episodes and changes sampled behavior.
Legal masking adds information and changes the policy law. A new orchestration
framework duplicates the existing reward DTO, learner and actor/inbox runtime.

## Consequences

Collection admits no D/E/final/probe inputs or update authority. Failed attempts,
returned work, draws, cleanup and full state are retained with bounded storage.
Unknown constructor work is explicit; failures never produce a trainable packet.
This qualifies a missing native collection interface. It does not establish
useful adaptation, matched exposure, learning benefit or a circadian advantage.
The separately reviewed generic runtime route remains future work.
