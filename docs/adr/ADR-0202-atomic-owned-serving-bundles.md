# ADR-0202: Publish complete owned serving bundles atomically

## Context

R3.3 owns independent actor/candidate models, and R3.4a measures all matched inner
guards. A report can become stale while a candidate trains or another promotion
changes the actor. Restoring a live model can fail halfway through mutation.
Weights alone cannot restore cache, complete native state or serving context.

## Decision

Keep the fixed actor API and add a compatible promotable actor with optional
composition in the candidate runtime. Freeze candidate base version and lease
candidate snapshots through approval and commit. Issue private identity-checked
local tickets only through the configured guard evaluator. Use one native builder
for evaluation and preparation, recheck full state/revision/policy and monotonically
increasing actor generation before commit. Publish one owned slot under the read
gate; the slot contains current complete bundle, generation and prior bundle.

The bundle includes native learner, version, fixed bounded TTL cache configuration,
cache entries, passive serving metadata and cache tick. Reads update cache by
replacing their bundle under the same gate. Promotion invalidates cache; rollback
publishes the exact retained prior bundle with a new generation and consumes the
latest genuine receipt. Do not expose a public raw model setter or accept foreign
guard reports. Preparation happens off the serving gate. No callback follows the
publication point. Nested operations fail explicitly rather than deadlocking.

## Alternatives

Live restore needs recovery from arbitrary partial native mutation. Independent
assignments of model/cache/context allow inconsistent reads. Model version alone
allows old approvals to become valid after rollback. Persistent approval files
would require new physical provenance/security boundaries. Unbounded history/cache
would introduce unnecessary lifetime/resource policy.

## Consequences

Reads see one supported complete generation, stale approvals are refused, and
rollback does not depend on native restore succeeding. Native implementations and
factory/probe/feature-identity callbacks remain trusted contracts; this does not
certify malicious Python graphs or provide final-label/deployment authority.
Metadata cannot change algorithms or action policy. Payload hard allocation limits,
fair scheduling and measured serving p50/p95 belong to the subsequent resource
sharing task. Start a fresh candidate runtime after promotion; never silently rebase
the previous inbox. One prior rollback generation and finite pending/cache entry
counts bound retained models and records without introducing deeper histories.
