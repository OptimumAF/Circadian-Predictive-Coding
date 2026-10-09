# ADR-0213: Anchor owned metadata age independently of array bytes

## Context

The full R3.6 completion audit demonstrated that scalar-only metadata received
no elapsed anchor. The array byte port legitimately measures zero, but that
does not mean a dictionary is empty. Initial `any()` also stopped validating
remaining auxiliary graphs after finding a positive array byte count.

## Decision

Validate every exact supported initial auxiliary dictionary. Anchor any nonempty
dictionary to the original budget clock independently of its measured array
bytes. Apply the same rule before promotion metadata copies. Cache first-copy
anchoring remains conservative. Never renew existing age on copy/discard/rejection;
reset only after actual all-holder cleanup.

Why this: retention describes owned content; copied-array bytes describe a
separate conservative quota. Keeping the existing metric and port boundary
preserves reproducibility and avoids pretending scalar metadata occupies no RAM.

## Alternatives

Adding string/Python/RSS estimates would change the byte metric and still miss
zero-size arrays or nested empty containers. Anchoring every empty dictionary
would expire an idle actor with no content. Selecting only positive arrays leaves
the confirmed bug and incomplete graph validation intact.

## Consequences

Supported nonempty metadata/cache content expires under the original authority,
including scalar-only and zero-size arrays. Expiry conservatively purges all
owned copies and may precede a sample's own deadline. Original work/byte/holder
quotas, consent, IDs and parameters remain unchanged. The original process,
quiescence/trusted-port, caller-copy/RAM/unlearning and durable restart limitations
remain explicit. No new interface, dependency or native/scientific equation.

Tests first reproduced the bug and validation short circuit. Controls cover all
zero-byte content classes, failed/discarded/prepared copies, reset, actual worker
and one separate fixed native handoff successor. Evidence:
`artifacts/runs/r36b2b-auxiliary-20261007/`.
