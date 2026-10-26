# ADR-0487: causal executed-action inputs with an equal-width control

## Context

Readiness teaching uses bounded phase/closing memory. Current conventional neural
policies have zero native acquisition despite lower imitation loss. Current-state
features may omit useful execution history; this is a hypothesis, not established
failure causality. Original observations/actions and negative evidence must remain.

## Decision

Add an explicit recipe to existing training and frozen rollout paths: raw or
sign-canonical42 prefix, previous actually executed nativefloat32 action4, capped
consecutive negative-grip count/6. Reset independently per episode and advance
only after a matching successful native return. Validate preceding state against
the existing bounded native record prefix before more work. Fit first42 only;
append fixed zero means/unit scales. An equal-width five-zero mode shares the
architecture, initialization and prefix preprocessing. Keep original raw traces.

## Alternatives

Elapsed-step features have existing negative results and are retained. More dose,
new circadian mechanisms or another audit framework do not address this specific
representation hypothesis. A recurrent learner remains possible future work but
has greater state and training complexity than this bounded explicit input.

## Consequences

History may help or hurt; no task benefit is inferred from correctness. The closing
count is a proxy that can saturate before proximity/contact.47D changes capacity
and computation versus42D; compare with equal-width zero history. Live storage is
bounded while prefix revalidation adds computation. No new dependency, runner or
evaluation access. Unsupported native compositions fail before paid SDK work.
