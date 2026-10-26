# ADR-0440: Inspect saved isolated learnability through existing playback

## Context

The saved H32 joint128/F16/A16 quartet is incomplete. The fixed-A inspector cannot
represent its separate F/A roles, failed observed tails or unrun game endpoints.
The evidence must remain usable without rerunning a learner or widening clocks.

## Decision

Add an explicit pure saved-report projection for these four exact protocols.
Share the existing native XY/action/history/transition parser with narrow optional
context/role/failed-input-tail parameters; preserve old defaults. Retain unchanged
source JSON and exact pointers; planned missing slots have no source pointer.
Reuse the existing exclusive offline publisher/CLI and MiniGrid template. Show
recorded costs/nulls/lower bounds and false proof flags with their source scope.

Why this: one format-specific projection supplies the missing runnable view with
no learner changes, new orchestration or scientific selection. The pretty saved
BP file slightly exceeds8MiB, so permit16MiB raw only for these exact protocols;
compact app bounds and16MiB output remain unchanged.

## Alternatives

Converting records into fixed-A would mislabel F roles and lose provenance.
Replaying an environment substitutes new states for evidence. Another experiment
or larger clock does not solve the missing saved-data view.

## Consequences

No model, physics, network or final evaluation is used. All failed/incomplete
studies remain unqualified. Scoped implementation checks do not establish
retention, recovery, circadian benefit or repository-wide readiness.
