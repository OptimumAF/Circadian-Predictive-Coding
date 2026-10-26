# ADR-0602 — Explicit correct TRAIN acknowledgement

## Context

The frozen recurrence comparison exposed a queue-progress gap. Candidate drift
learned through two updates, then repeatedly inspected a correct oldest pending
packet. Waiting did not consume that packet, so later TRAIN remained queued and
the all-updates drain condition prevented publication. More epochs or threshold
tuning would conceal this scheduling dependency.

## Decision

Add a scalar no-update receipt at the existing inbox/runtime boundary, verified
against current detached candidate state under the exclusive lease. Use a lazy
bounded history to preserve the default owner schema. The explicit candidate
feedback option separates settled packets from optimizer updates, permits one
settled packet per tick, and composes the existing guarded publication after all
arrivals and receipts. Recheck actor/base/candidate version after verification.

Why this: an arrived packet can be correct without requiring optimizer work.
Recording that fact explicitly resolves queue progress without inventing update
charges or weakening the original guard and resource envelopes.

## Alternatives

Forcing a wake on every correct packet fabricates a learning requirement. Keeping
an acknowledgement only in the script would let the inbox later train the same
packet. Erasing the payload would misstate consumption and free authority or
capacity. A new orchestration framework or checkpoint format is unnecessary.

## Consequences

Point-in-time correctness is not lasting retention; final clean guards remain
mandatory. Diagnostics/copies are paid; actual updates and acknowledgements have
separate clocks/provenance/counts. Unmanaged no-sleep TRAIN only. Unsupported
cursor/handoff/erasure formats refuse the new history instead of dropping it.
Persistence, broader scientific benefit and deployment readiness remain open.
Independent review found and prompted the post-verifier serving-version check;
the repaired source and scalar actor-swap refusal test preserve that finding.
