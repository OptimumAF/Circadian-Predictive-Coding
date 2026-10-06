# ADR-0178: bind original observed release traces with partial order

## Context

The complete saved chronology binds every historical witness and alias but
newly verifies no execution event. It does not assert that execution was absent.
The original scoring audits contain source input/target reads, ordered successful
release records and every endpoint call. The unchanged scoring reader validates
the entire fixed result/audit and both unchanged complete training readers.

The request UTC is recorded before the parent begins its measured invocation.
The observer records causal event order without individual UTC timestamps.
Adding elapsed duration to request UTC would invent an exact release time.
Separate deterministic requests, copies and numeric seed counts also do not
establish independent source replications or cross-run release order.

## Decision

Add a pure causal-order module, a complete decoded-scoring witness consumer and
a separate fixed original-reader boundary. Reuse the unchanged full scoring
validators; require every original family, seed, arm, checkpoint, role, endpoint,
failure, source/request/audit/resource link. Retain each causal record by its
exact original pointer and full canonical record digest, with full input identities.

The fixed public boundary reads one canonical or repeated scoring bundle through
the original scoring reader, supplying both original complete training readers.
Verify all whole scoring bytes and request sources before/after readback. Its
private reader/IO ports remain fixture seams and grant no original-reader proof.
Only actual supervised full original-reader dispatch with complete source/input/
proof/current/history preservation supplies recorded historical verification.

Keep exact UTC release, cross-run ordering and independent replication counts
unknown. Preserve the entire prior saved witness ledger and every original
unknown in a separate bound input; actual traces never erase earlier unknowns.
Keep all fresh-role and original prior-usage/resource acceptance flags false.

Why this: recover the supported actual evidence without upgrading documentary
timestamps, callback counts or copied records to stronger chronology claims.
The within-run order follows the pinned observer: training state barrier; each
input/target read and completed release; the next whole state barrier; every
ordered prediction; final whole state check. No source/model/RNG is constructed.

## Alternatives

- Treat request UTC as actual release UTC: unsupported by the recorded clock.
- Extract just a role header: omits full endpoint/state/failure/audit links.
- Use a cached reference dictionary instead of complete training readers:
  drops the original source-bound reader requirement.
- Count repeated callbacks/copies/shared seeds as independent trials:
  overstates replication and ignores common source identity.
- Promote this evidence to fresh-role admission: the original resource gate
  and complete future role/execution contract remain unfinished.

## Consequences

New P6.7d2b2b2 is one original b2 component. Its full scope must pass before its
checkbox changes. Original b2a/b2/b3/d2b/d2/P6.7/d3 remain open. Each of the two
complete original scoring read stages has a 120-second cap, sharing 240 seconds
including failures; both reconstruct both full original training references.
Inspection/source/test/static/closing/document families have 180-second limits;
closing includes preparation and outputs are capped at 32,000,000 bytes.

Original failed semantic repeat remains failed: 350.7925872/360 seconds spent,
9.2074128 remaining. Scientific caps, metrics, baselines, seeds and contrasts
remain unchanged. No dependency or environment variable is added. The next
extension must preserve these records while completing verified unknown effects
and the full future independent role/config/count/caps/analysis/stopping/source/
request contract before any fresh source or confirmation release.
