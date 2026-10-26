# ADR-0354: Project recorded MiniGrid public views through the existing inspector

## Context

The accepted MiniGrid fixed-A case contains public148 observations, causal296
feedforward inputs and seven native actions. The existing custom-game inspector
uses eight features, five actions and a derived5x5 map. Its assumptions cannot
describe these saved successes, negative terminals and timeouts faithfully.

## Decision

Dispatch the exact fixed-A protocol to a pure MiniGrid projection, retaining the
unchanged detached source and explicit JSON pointers. Decode native[x][y][channel]
public observations; validate recorded decisions, continuity, history, metrics,
endings, cleanup and displayed costs. Distinguish a before-decision input from
an after-action observation. Keep unknown or failed evidence unknown.

The existing file builder chooses a MiniGrid template only for the new projection
schema. Existing custom-game branches/template, file caps and exclusive writes
remain. The new renderer uses the agent-centric public view and separate global
direction metadata; it reconstructs no private world map or environment.

## Alternatives

Treating this as the custom schema would transpose semantics and action masks.
A new inspector framework duplicates file safety and publication. Running the
environment during playback replaces observed evidence with a new simulation.

## Consequences

All five controls and all failures remain inspectable with raw fit, live outcomes,
input history and paid work separated. Source text is inert and escaped. No
learner, native step, final evaluation, tuning or scientific advantage follows.
Only one explicit saved protocol is supported; future formats need their own
tests and honest unknown fields. Full topology/lineage/sleep requirements remain.
