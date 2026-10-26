# ADR-0442 — Match native exposure with an explicit F48-only control

## Context

The existing H32 F16/old-A replay/A16 composition adds64native calls/576rows
across two isolated probes. Comparing it with the cheaper192call route confounds
old-A exposure with additional training. The fresh-task return currently admits
the original16batch envelope and a16update F budget. Expanding input packets to48
would violate its original input identity/storage contract.

## Decision

Add explicit exact-bool `extra_f_exposure=False` to the existing app/facade and
`--extra-f-exposure` to its CLI. True validates/copies the original16 input batches
once, repeats the owned tuple by reference three times through the existing loop,
and requires a fresh48update F budget. Checkpoints0/12/24/48 align with actual
native calls at replay logicalF0/4/8/16. Same-candidate A16 keeps its distinct
original budget/optimizer; return/observer guard the explicit48 authority. F16
defaults and replay/game recipes remain unchanged. True refuses rehearsal/games.

Why this: use the existing learner loop without increasing the original65536entry
input bound, converting reserved roles, renewing budgets or broadening tolerated
counter progress. The pure adapter helper owns public recipe/cost descriptions.

## Alternatives

Implicitly train more F in the existing baseline changes its frozen dose. Treating
F48 as16logical updates hides paid work. Relaxing all return counters accepts
unobserved changes. A second runner duplicates ownership and isolation rules.

## Consequences

Both explicit routes execute256native calls/8064rows/P94/846rows; allocation to
fresh F versus old A differs. Native count matching does not equalize validation,
bank/copy/memory or process costs. Freeze comparative metrics/protocol before
execution, retain all cues/negative results, and keep gameplay/generalization/
acquired recovery/CPC benefit separate from teacher-forced fit. Full research and
release requirements remain open.
