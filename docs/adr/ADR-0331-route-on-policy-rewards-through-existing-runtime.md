# ADR-0331: Route on-policy rewards through the existing generic runtime

## Context

The recurrent reward objective and actual collector pass their correctness gates.
AttributionStream admits supervised arrays and class targets, while its underlying
ActorShadowRuntime and inbox already support opaque training payloads. A separate
RL CLI needs actual sampled updates without reinterpreting teacher labels.

## Decision

Add a small reward-only NativeLearner wrapper accepting an exact owned RewardUpdate
plus a matching ID release. Delegate existing fullstate/on-policy reward admission,
fixed gamma1, native SGD and snapshot format. The separate recipe feeds those
packets through existing Experience/LabelArrival/inbox/runtime budget receipts.
Only a fixed eight-task schedule lives in the adapter; no new runtime framework.

Use existing key/door scoring/live helpers after all training, adding explicit
context metadata and prediction-only type requirements to the private scorer.
Admit recurrent_rl in common evaluation identity and fresh-session evaluation.
Pin bounded admitted records, source/candidate state, original budget and sampler
progress before collecting more. Preserve failed work and unreached slots.
Freeze/review one tiny native protocol, with no quality gate, baseline rerun,
post-result update or final-data feedback.

## Alternatives

Array/teacher dummy targets would conceal the native reward interface. A second
stream framework would duplicate role admission, ownership and permanent charges.
Off-policy reward replay would violate the existing full behavior checkpoint guard.
Changing supervised defaults would weaken accepted controls unnecessarily.

## Consequences

One pure-RL/no-replay arm is separately evaluated with variable policy exposure;
it is not an equal-work comparison to previous BC/PC/CPC methods. Real negative
training/final outcomes remain valid. The small observed case has only zero rewards,
so unchanged parameters do not establish useful retention or recovery. A later
explicit demonstration warm start must preserve typed objective/phase accounting
and control for optimizer momentum before any comparative claim. Original full
R4.2/useful learning/generalization/release and repository goals remain open.
