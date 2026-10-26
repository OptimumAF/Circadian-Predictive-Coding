# ADR-0403: Fixed finite-relaxation controls in the existing acquisition route

## Context

The accepted joint runner supports isolated F and A-return updates, but original
S2 PC/neutral methods did not acquire the six TRAIN traces. One qualified replay
matches all270 original TRAIN probabilities exactly, finds local descent and
nonzero latent residual. This motivates a fixed sensitivity control, not a
causal conclusion or a tuned winning baseline.

## Decision

Admit separately named `pc_relax64` and `neutral_cpc_relax64` recipes through
existing learner, delayed-label stream, checkpoint and disposable probe ports.
Use the exact original native policy with64 inference steps, the existing native
maximum. Preserve base native algorithm identities and original S2 defaults.
New outer names bind the complete fixed policy with explicit54-row clocks;
neither aliases nor arbitrary expected-policy parameters weaken admission.

Keep the mode narrow: joint catalogue, distinct F, old-A readout and A-return;
no additional diagnostics or alternate training/rollout modes. Charge64 times
returned updates/rows after probe/return aggregation, with unknown partial failed
work explicitly unknown. This differs in inference computation from S2 and BP.

## Alternatives

- Relabeling the original PC method destroys baseline provenance.
- A sweep, larger learner or changed objective confounds this finite-relaxation
  question and needs separate scientific justification.
- Another runner duplicates existing isolation, budget and state authorities.
- Another diagnostic-only increment would not exercise learner adaptation.

## Consequences

One module owns fixed recipe/metadata/cost composition; existing adapters and CLI
admit the explicit names. No core objective, native optimizer law, dependency or
default changes. Native parity and full isolated synthetic-route tests qualify
engineering behavior. Separate frozen real-data execution remains necessary for
acquisition/retention/recovery claims. All historical negatives, SPENT budgets,
final-data isolation and unfinished probability-evidence criteria are preserved.
