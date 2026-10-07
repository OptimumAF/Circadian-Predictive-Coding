# ADR-0180: Bind complete stream claims to the prospective design

## Context

The existing prospective replica API computes the 400 required streams from all
60 ordered bindings and 50 planned source groups. Its tests already cover base
types, shared groups and all cross-stream offsets. A caller's separately supplied
derived-stream records had no public validation boundary. The helper's toy
single-replica envelope leaves 25 cases unbound and cannot establish the full
study's source/history/request/role proof.

## Decision

Add one pure public app function accepting the full design, its expected whole
canonical identity, complete ordered bindings and all 400 typed stream claims.
Reuse the existing public scope/binding/collision validators and compare each
claim with its required record using exact field types and ordering. Return the
existing immutable declaration with independence unknown and fresh authority
false. Raise useful errors on detached identities or invalid records.

Why this: full numeric metadata must agree before a future envelope consumer can
use it. Exact types prevent boolean/integer and float/integer equality from
concealing a malformed declaration. Whole layout checks preserve shared views
and all matched settings; no single-row toy or mock replaces them.

## Alternatives

- Extend the original core binder to parse external JSON: adds IO-facing schema
  concerns to a stable pure ordered-binding interface.
- Bind the helper fixture to a dummy inspector: would conceal missing envelope
  implementation and incorrectly count skipped cases as coverage.
- Add only more base-collision tests: duplicates behavior already verified and
  leaves external stream claims unchecked.

## Consequences

One app module and one full-layout test module are added. Existing source/test
files, original derived offsets, matched settings, scientific caps and negative
results remain intact. App dependencies point to app/core, with no infra import.
Actual prior usage, RNG-domain provenance, source/request/role identity, release
chronology, independence, original failed resource/precision acceptance and full
b3 execution gates remain unfinished. No helper integration or scientific
execution is authorized by successful metadata validation.
