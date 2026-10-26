# ADR-0385: public-history reports in the existing offline inspector

## Context

Accepted memory live/E16 and four-method matched-history reports expose runs and
raw TRAIN episodes. The older inspector expects policy_evaluation or a different
legal-conditioned TRAIN schema. Complete reports therefore cannot be displayed.

## Decision

Add one explicit app projection for the three saved protocols and one offline
HTML template. Extend existing dispatch/template mapping. Reuse recorded geometry
and playback validation through a separate public_history raw-TRAIN selector;
preserve the older conditioned selector and default TRAIN refusal. Keep full raw
source, exact pointers and owned copies; validate bounded counts/roles/clocks and
preserve partial/null evidence. Embedded whole-report costs only; refuse unrelated
sidecars. Do not recompute scientific metrics, run physics or load learners.

## Alternatives

Relabeling reports as final/conditioned data would change their meaning. Another
inspector server or execution framework duplicates existing IO and boundaries.
Hiding negative/inactive controls or fabricating topology would misstate evidence.

## Consequences

Accepted learning evidence has a common offline presentation with explicit
benchmark/probe/serving distinctions. Raw source and computed board geometry
remain distinct. Browser policy currently blocks agent local-file navigation;
interactive acceptance stays open until manual observations exist. No security
rerouting, new dependency, model work or scientific benefit claim is introduced.
