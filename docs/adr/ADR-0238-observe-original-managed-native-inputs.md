# ADR-0238: Observe original managed native inputs

## Context

Native replay snapshots contain arrays, priority and positive fraction, but no
source keys. Content-derived sample IDs cannot prove which original managed
source produced a retained row. The inbox detaches inputs before native work,
and native replay storage copies them again. Conservative erasure currently
clears all replay rows because it has no subject index.

## Decision

Add an optional typed observation port at the original detached inbox invocation,
propagated through existing runtime/sharing gates and a managed owner convenience
method. Report original identities, actual native inputs, receipt and spent count
through synchronous, expiring, original-thread access. Preserve original update
order, exception behavior on the default path, schemas and instance fields.
Release access-owned references; preserve primary and secondary observer failures.

Why this: the producer must supply origin at the time of the actual update.
Separating this boundary from persistent replay retention permits small fake-only
verification before modifying native models or codecs.

## Alternatives

Matching content hashes loses producer identity for equal inputs. Adding persistent
raw references to inbox instances would change source schemas and introduce
unbudgeted retention. Changing native storage and every checkpoint path together
would make the first implementation difficult to qualify independently.

## Consequences

Callbacks are trusted observational code and own references they retain. Completed
observations precede final checks; failure stages must invalidate any provisional
consumer result. This increment does not certify provenance or grant consent,
restore or recovery permission. Full persistent replay origin and every retained
row path remain unfinished under R3.5b2e5b3.
