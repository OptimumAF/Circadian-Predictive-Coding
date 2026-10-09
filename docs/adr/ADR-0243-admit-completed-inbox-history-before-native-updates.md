# ADR-0243: Admit completed inbox history before native updates

## Context

Native replay capacity may legitimately evict a trained row while the original
inbox source, label and committed receipt remain live. Canonical replay rows
cannot certify the complete inbox history in that case. Observing current
records at capture would grant retrospective provenance.

## Decision

Add `retain_inbox_origins=False` to the original `ManagedReplayOrigins`
constructor. Enrollment requires fresh work and empty canonical replay. An
opted-in ledger installs a separate bounded weak history before training.

At the actual update's started observation, compare the original numeric
source/label contents with the actual detached consumed inputs. Reserve history
metadata and a live record through the same original replay admission before
publishing the provisional witness. Bind the actual inserted receipt at
completion; qualify only receipts returned by a successful final poll.

Native row pruning remains separate. Canonical eviction does not refund history
or permanent copied-record charges. Tombstones may prune weak history only when
both original payload maps have removed that pair. Metadata-only counting lets
legitimate expired weak payloads reach that leased pruning path; counting grants
no access. Complete verification still refuses missing non-tombstoned records.

Checkpoint copies admit each complete inbox pair under the original metadata
and live limits, in addition to existing native row and raw-copy reservations.
Actual copier memo identities bind materialized records and arrays. Prepare the
rebound weak history before publication, validate after the final opaque probe,
and commit only prepared plain field assignments after the controller's existing
four publication statements. History instances have fixed fields; trust
boundaries invoke the exact implementation rather than an instance callback.

## Alternatives

- Inferring history from matching keys, current contents or receipts at capture
  cannot establish what the original update consumed.
- Keeping all canonical native rows would change native retention semantics and
  raw payload ownership.
- A new history budget would renew authority across the same operation chain.

## Consequences

Default-off preserves the conservative original capture requirement. Opt-in
history consumes additional original metadata/live capacity. Capacity must be
chosen at ledger birth; an exhausted original allowance remains exhausted.
Content stamps establish integrity of already observed identities, never consent
or ownership. The two-pair/one-canonical-row validation slice does not qualify
mixed erased/live receipt capture, compound erasure, other native families,
durable recovery, disk/RSS accounting or scientific effectiveness.

Qualification and exact commands live in
`artifacts/runs/r35b2e5b3d2b2b2e-inbox-history-20261008/`; the parent task remains
open until all of its preserved criteria pass.
