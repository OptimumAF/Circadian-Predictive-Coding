# ADR-0186: Observe runtime code through a live process proof port

## Context

V2 files and native request ownership are implemented. File manifests do not
observe executing objects, detached Python functions or native executable memory.
The complete trusted runtime/source-version gate must precede source construction.

## Decision

Add core immutable process records/live ports, app composition over unchanged full
V2/native context and separate focused Python/native outer observers. Enumerate the
whole actual loaded process and actual code/closure/default/namespace/entrypoint
bindings; pin complete native images/executable memory. Keep observed bytes separate
from original source/version attestation. Explicit audit events are denied while
held, full late checks run, immutable observations remain historical after exit.

## Alternatives

Caller manifests/selected dependencies would not observe real execution. Mixing
native/process IO into app would violate layer direction. Rewriting original
source/version closures would invalidate historical evidence. None was chosen.

## Consequences and current evidence

No prior source edit/rekey/dependency/scientific change. Real positive, detached
object, native memory/denied load/release/inactive/file and current invalid/late
Python controls provide component evidence. Grouped child timeout and failed
static/fixture versions retained. Full20/502 related and complete source-version/
transient/continuity variants remain open; P6.7d2b2j is unchecked. Actual complete
captures are11.18MB with measured Python/native/serialization costs. Preserve the
168.7993043/180spent correctness family; diagnose lossless graph costs and explicitly
declare any corrected bounded protocol before further gates. No silent weakening
or reset of original350.7925872/360 scientific resource failure. All parent prior/
arrival/ledger/b3 requirements and held proposals remain required.
