# ADR-0237: Check enrolled pending source bindings before copying

## Context

Complete capture projected pending runtime references without checking that the
runtime belonged to the original registry and was leased. Consent validation
covered retained live inboxes, but omitted inbox payloads in checkpoint views.
Source schemas alone did not establish these relationships.

## Decision

Validate exact bounded original runtime/controller/pending/token schemas and
relationships under the existing original gates. Retained runtimes must share
the original actor, lineage, budget, clock and installed consent/copy guards.
Pending runtimes must be enrolled and leased. Checkpoint resource/probe bindings,
revision and sampler history must belong to that original interval.

After the complete graph preflight bounds supported values, validate complete
checkpoint views, original budget chronology, stored integrity and live consent
for every retained checkpoint payload. Validate promotion token/report/builder
relations through pure original checks. Include portable promotion ticket fields
in the copied graph; retain the actual ticket and rollback receipt as original
references outside it. Copied tokens never acquire restore or rollback rights.

Why this: legitimate stale observations are retained data. Calling a controller's
restore guard would reject them and invoke native callbacks or reacquire gates.
Capture validates retained observations without granting permission to reuse them.

## Alternatives

Schema-only projection misses foreign original references. Per-component checks
outside the common interval can mix owners. Controller restore validation invokes
callbacks and applies a different contract from complete retained capture.

## Consequences and limits

The new app module checks relationships; the outer NumPy adapter independently
bounds complete snapshots and checkpoint inbox payload shapes. No new dependency,
native algorithm, allowance reset or owner construction occurs in production.

Tests include bounded checkpoints issued by the existing capture path, plus
separately labeled synthetic promotion and retired-history graphs. Native
checkpoint snapshot calls are explicit bounded fixture work. Configured digest
probes in those fixtures are deterministic stubs, not native provenance proof.
Synthetic graphs do not establish actual promotion issuance or checkpoint restore.
Full native variants, original provenance, replay consent, durable bytes and
live/disk/model recovery acceptance remain open.
