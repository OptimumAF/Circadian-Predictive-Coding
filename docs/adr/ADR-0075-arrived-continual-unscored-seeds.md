# ADR-0075: Persist completed v6 seeds without final data

## Context

The ordinary v6 runner holds every trained seed until global final-test
release. An interruption after seed 17 must not require retraining that seed
or expose its final labels before seed 19 trains. V5 format 5 stores a
different development split and no observed guard/access ledger, so its
payload cannot represent v6 without changing verified v5 identity.

## Decision

Add a distinct `ArrivedRunnerCheckpoint` payload and trusted local file
header for format 6. At each completed-seed transaction, save detached
trained models, the six arrived development role IDs/hashes, the actual
release/update/guard/task ledger with a content digest, and the ordered run
config/seed digest. Save no source object, final-test value/hash, or score.
Before training a later seed, regenerate only prior development roles and
validate their IDs/hashes, baseline step counts, circadian wake state,
observed-example replay membership and budget, and the completed event
cursor. Terminal resume repeats final scoring without training or writing
another checkpoint. The file remains trusted local pickle with a checksum;
the checksum is not authentication.

The format reserves active cursor/state fields for the next intra-seed
increment. This increment accepts only `seed_complete` transactions and
rejects nonempty active fields, so it cannot silently present a partial
checkpoint as a completed seed.

## Alternatives

- Reuse format 5. Its role hashes and scored-result layout do not express
  the v6 four-role audit and would blur established v5 identity.
- Save final-test arrays for later scoring. That would put held-out labels
  inside a training-side payload.
- Retrain every earlier seed after interruption. That wastes work and
  weakens exact continuation evidence.

## Consequences

Two-seed completed-boundary and terminal resumes match the ordinary v6
report in both model orders. Raising source sentinels keep final input and
labels closed until both seeds train. Changed earlier A/B roles, future
replay, and event-cursor damage reject before another update; changed final
labels alter only final reporting after terminal resume. Saved state and
file size grow with completed seeds. Wake, before-sleep, after-sleep, and
A/B intra-seed interruption remain P1.3c3b2c2b2. Outer setting selection
and all parent strict-online claims remain open.

ADR-0076 subsequently filled the reserved active fields with A/B model
and sleep transactions; the completed-seed record contract remains intact.
