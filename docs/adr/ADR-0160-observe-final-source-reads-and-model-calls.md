# ADR-0160: Observe final source reads and model calls

Date: 2026-10-01
Status: Accepted for the P6.7c1c2 execution observer increment; worker gated.

## Context

The complete app scoring matrix counts calls to injected release/evaluation
ports. The process boundary must independently observe actual source reads,
model predictions and examples, compare them with each saved endpoint, and
retain final views for checks after serialization. Existing global checks
verify original role metadata and all live model state without reading outer
or original final fields. The original numerical failure policy and 0.5
prediction threshold are already frozen. No reserved final value is open.

## Decision

Add an infra observer over the supplied held training inventory, with a
budget-check callback from the unchanged optimizer/resource observer. It
establishes execution facts only, never source/request provenance, a global
training proof or scientific authorization. The future full worker remains
required; no partial scientific run command or manifest override is introduced.

During final evaluation, temporarily replace each original sealed role's
source with a transparent source-field observer and its outer role with a
raising guard. Preserve all train/inner/ID/hash/release metadata. Read each
original final input/label field exactly once, only within its scheduled
release, and bind the returned view to those same observed arrays/IDs/content.
The original roles remain final-unreleased. Remove owned guards on every exit,
preserving any other corrupted metadata for the enclosing gate to reject.

Wrap the actual BP/PC/circadian prediction methods (including inherited parent
controls) only during evaluation. Require the exact scheduled held model and
released input object, one prediction per endpoint and all role releases
before predictions. Record attempts/examples, successful returns/numerical
exceptions and model-kind partitions outside model state. Independently
derive the same correct/count or already-declared numerical null from each
returned probability array. Compare the unchanged adapter's result to it.
Block optimizer calls and outer/source reads outside their prescribed context.
Other errors propagate and never become substituted scientific outcomes.

Keep source-returned arrays and final views, plus immutable capture facts,
through the future worker's serialization. Recheck installed role/source
bindings, cached source arrays and released content without reopening fields;
compare every observed release/read/prediction fact and total with the app
result. A separate pure app verifier requires the full fixed scored manifest,
complete validated scored JSON and independently declared model kinds.

Budget checks precede and follow actual source reads/releases/predictions.
During post-evaluation verification, callbacks precede the final content and
link checks. A fabricated regression exposed content mutation by a callback
after verification; preserve the first freeze and link a corrected V2 before
rerunning fixtures. Completed prediction facts are recorded before a later
budget stop. Restore every method/source/outer guard on success,
numerical failure, contract/state failure, resource stop or cancellation.

Why this: counting port invocations alone cannot prove the original prediction
ran, received the declared model/input, or produced the reported count. Scoped
guards and direct method observations preserve the original algorithms and
make these facts independently reviewable. Cached arrays avoid a second final
source read during post-serialization checks.

## Alternatives

- Trust app-local counts: cannot observe a missing/extra/wrong model prediction
  or a substituted source view/result.
- Reopen final source fields after serialization: violates exact access counts.
- Persist counters in model snapshots: rollback/state copying could change
  them, and observation would alter the scientific training state.
- Catch resource/contract errors as numerical nulls: silently changes the
  frozen failure policy and can publish incomplete scientific work.

## Consequences

No old pinned source, model, baseline, metric, seed, failure policy, dependency
or cap change. Freeze new source identities before fabricated final fixtures.
Tests use genuine first-development-seed held models and fabricated source
fields only. Whole dispatch fixtures require an explicit global-state spy;
they demonstrate observations, not reserved training/scoring or provenance.
P6.7c1c2 stays unchecked until its full request/worker/resource/artifact gates
pass; all original c1/c2/reporting acceptance criteria remain unchanged.
