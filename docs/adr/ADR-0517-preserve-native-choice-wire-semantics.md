# ADR-0517 — preserve native Choice wire semantics

## Context

R5.1 primary inspection identified released Open-Jev native Choice probabilities
and chance-relative confidence. Existing DecisionResult requires raw scores and
lexical ID ties, which the wire neither supplies nor follows.

## Decision

Add one pure Open-Jev request/reply codec using existing immutable request input
and a separate frozen decoded observation. Preserve supplied candidate order,
native probabilities/confidence and the original bounded byte body. Check native
schema/normalization/selection; do not run a second softmax or invent evidence.
Source/model contracts remain separate from loaded model and trained-task claims.

## Alternatives

Converting probabilities to invented logits would misstate observation. Replacing
native ties/confidence with the local reference contract would change behavior.
Installing or serving models before resource/native-contract gates would not be
needed to validate this pure boundary. No new runner or dependency is required.

## Consequences

The first bridge supports only the existing bounded text Choice subset. No
transport, native model loading or Noul/Score implementation is implied. Question
and candidate checks do not authenticate correlation or provenance. Tests use
declared analytic replies; actual native inference remains a separate gate.
