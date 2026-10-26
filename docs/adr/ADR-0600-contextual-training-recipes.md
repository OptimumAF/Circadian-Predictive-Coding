# ADR-0600 — Explicit contextual recurrence/noise data adapter

## Context

The existing native stream builder binds six-row D7 packets and native label
offsets. The contextual conventional learner uses three/six-row D9 prototypes;
silently relaxing the native builder would change frozen experiments. Broader
controller validation needs explicit recurring/noisy TRAIN streams.

## Decision

Add a separate small ContextTrainingRecipe/build_context_stream adapter with
exact original trace identities, public feature and clean-target validation,
owned output packets, and named synthetic corruption provenance. Reuse the live
schedule, gate, publication and rollback; new recipe cases declare24opportunities,
six-update/60second budgets at creation. Preserve original defaults and the
84-row diagnostic allowance; new recipe144-row allowance is explicit.

Why this: the demonstrated dependency is data shape/semantics, not a missing
runtime. Native recipes and existing negative controls retain their original
meaning and evidence.

## Alternatives

Reusing native offsets would corrupt the wrong semantic row. Treating synthetic
noise as trusted teacher labels would misstate truth. Renewing a two-update owner
would violate its budget. None is necessary.

## Consequences

The initial semantic parser failed because it assumed concepts appeared at all
workflow stages. Source/saved features confirmed intentional masking outside
ready; corrected trace-ID binding preserves the encoder and reorder support.
The failed validation and costs remain. New engineering cases do not establish
comparative utility/generalization/CPC benefit or long deployment. A separate
prospective comparative protocol remains required.
