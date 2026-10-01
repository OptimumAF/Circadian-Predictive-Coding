# ADR-0159: Read training references sequentially and validate all scored JSON

Date: 2026-10-01
Status: Accepted for P6.7c1c1 implementation; scoring execution remains gated.

## Context

P6.7c1a/b verify reproduced training and the fixed final evaluation matrix.
Their proofs explicitly carry no external source, file or resource authority.
Each saved complete training result is 134,554,378 bytes. The unchanged public
training reader already validates its full body, derived costs, request,
observed work, resource facts and complete audit. No reserved final value has
been opened. The original held-model worker memory cap remains 512 MiB.

## Decision

Split c1c into reference/payload readback correctness and the complete bounded
worker/artifact boundary. Keep every original criterion and cap. Add a pure
app validator for the whole scored JSON representation, with no partial-scope
production flag. Independently derive every cell accuracy/failure and total
from the ordered endpoint counts, check the fixed three global state proofs,
and require all final-role IDs/counts/signatures and their endpoint/cell links.
Shared phase/seed identities retain equal final signatures. Match the original
final ID declaration already required by the unchanged training validator.

The validator establishes declared JSON links only. It cannot prove source
arrays, live checkpoints, observed prediction calls, file bytes or resources.
It preserves the app result's explicit limits on authority. The process
boundary must still establish those facts independently before publication.

A new infra reader accepts the unchanged complete-reader port. The inspection
adapter supplies the actual pinned public training reader. Check both bundles'
exact request/result/audit bytes and result lengths before reading, after each
read, and again after both reads. Refuse failure/claim/missing artifacts.
Independently hash canonical decoded JSON incrementally and compare it with
the file identities, catching a changed or detached body returned by a reader.
Bind the original manifest/scope/source-map/adapter declarations as well.
Keep only reference, cost and historical resource metadata between reads;
discard each large decoded graph before reading the next bundle.

Why this: reuse the complete existing validator while avoiding two retained
134-MB decoded graphs and an additional whole encoded string. Parent reference
inspection is read-only correctness work, with no new measured RSS claim.
The future bounded child can recheck reference bytes without decoding them
beside held models; its entire resource and serialization gate remains c1c2.

Freeze the expanded available composition before fabricated scored JSON
fixtures. This increment's freeze is not the future complete scored worker
closure. Any implementation repair retains its superseded freeze and reason.

## Alternatives

- Retain both decoded results through scoring: unnecessary duplicate state
  beside held models under the unchanged memory budget.
- Trust result/audit hashes without complete readback: misses body/cost/schema
  checks that already exist in the public training reader.
- Reuse the app's cell derivation for JSON validation: would fail to establish
  independent endpoint-to-cell readback links.
- Treat JSON readback as scoring provenance: cannot demonstrate actual source
  contents, live models, call counts or resource enforcement.

## Consequences

No old pinned source, dependency, seed, model, baseline, metric, failure policy
or budget change. Tests use fabricated complete JSON and explicit IO-only
reader spies, never reserved sources or final values. The actual existing
unscored bundles will also be independently read with data/model/update/final
sentinels. C1c2/c1/c2/P6.11b and original final/reporting gates remain open.
