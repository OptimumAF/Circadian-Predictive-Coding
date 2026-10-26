# ADR-0416: Prove pure scalar report values and owners together

## Context

The completed50-step game receipt takes14.92146ms per paid_report_guard call.
Profiling identifies39142 recursive general comparisons/reflection calls; the
following native readout previously expired20s after10steps. That observation
does not causally attribute the timeout to this guard.

## Decision

Give the existing guard a private proof for exact built-in scalar values,
dict/list/tuple containers and string dict keys. It verifies values and original
owner identity in one traversal and checks that every original owner path was
visited. Only known primitive paths receive a lookup index, avoiding additional
custom-key hash callbacks. It proves the report anew on every call.

Any unsupported value/key, value mismatch, replacement or missing path falls
back to the unchanged original value proof followed by original owner paths.
Why: arrays/tensors/dataclasses may execute comparison code that changes an
earlier owner. Keeping their two-pass proof preserves closing behavior, value-
error precedence and existing mutable expected-report semantics.

No cache across ports, new serialization, hashing, budget authority, dependency,
public interface or learner objective. The additional path index refers only to
the same bounded original owner objects already retained by the guard.

## Alternatives

Dropping full values, checking owner membership without its path, or applying
one-pass ownership to opaque comparison callbacks would weaken prior protection.
Changing all generic _same_value consumers has a broader regression surface.

## Consequences

Final saved-receipt median10.263318ms is31.2177% lower than baseline, meeting
the prospectively fixed20% guard-only criterion. It does not establish native
20s/full16-game completion or satisfy old whole-case performance targets.
New behavioral parity tests cover original exceptions/identities/opaque side
effects. Future leaf types can use the original fallback until a separately
justified pure comparison is verified.

[Checks, exact commands and preserved failures](../../artifacts/runs/paid-report-guard-20261018/RESULTS.md).
