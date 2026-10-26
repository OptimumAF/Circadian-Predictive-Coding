# ADR-0498: Compose arrived head updates through the existing runtime

## Context

The GN stream learns known A control but loses it after B; B grasps/transports
without completing. Existing goal-consistent supervision can label newly visited
B states, while the correction adapter currently applies SGD or PC updates.

## Decision

Add an explicit BP-only head_gauss_newton opt-in at the existing correction and
native composition boundaries. Reuse the bounded native recorder and eight
arrived runtime updates, original receipts/clocks/charges and teacher. Summarize
actual native work in one small adapter helper. Preserve all default behavior.

## Alternatives

A new runner duplicates receipt/ownership rules. A new teacher or retuned seed/
threshold/dose would change the current scientific question without evidence.
Broad counter exceptions or budget renewal would weaken provenance.

## Consequences

Focused software checks and independent acceptance precede one frozen native
question. Bounded scalar prefix and one latest success pin preserve prior work
without a model chain. Unsupported replay/binary/diagnostic compositions refuse
early. Scientific usefulness remains an experiment, and negatives stay valid.
