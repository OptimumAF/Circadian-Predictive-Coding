# ADR-0361: Explicit zero-loss SGD on original observed episodes

## Context

The accepted warm MiniGrid run completes all eight reward updates and both TRAIN
readouts, but CE worsens and goal successes disappear. A frozen warm reference
has fewer optimizer updates. Carried momentum can move weights during zero-return
REINFORCE episodes; the first observed episode also has nonzero reward. Existing
frozen-BC controls do not match this additional-update exposure. Editing rewards
or behavior records would falsify the original observations.

## Decision

Add a distinct TRAIN-only `MomentumOnlyUpdate` to the existing typed recurrent
adapter, delegating to a small native zero-loss objective module. Preserve the
complete original episode evidence. Use original observation rows and a zero
recurrent initial state, dense exactly-zero gradients on all five parameters,
existing native SGD configuration and staged full-state validation before atomic
publication. Keep ordinary reward/CE math and default routes unchanged.

Why this: connecting elementwise-zero finite logits to backward lets ordinary
SGD momentum act on every parameter. Skipping backward or leaving `None`
gradients would implement a different control. Zero before summation avoids
reduction overflow while preserving the intended gradient.

## Alternatives

- Frozen BC: useful reference, but does not match optimizer update count.
- Relabel original rewards or manufacture behavior: invalid evidence.
- Force stale behavior probabilities through REINFORCE: violates its on-policy
  contract and confuses an explicit control with the reward objective.
- New orchestration framework: unnecessary; use existing typed stream ports.

## Consequences

Native numerical/atomicity qualification is separate from comparative evidence.
A future paired runtime must start independent learners from one owned shared
warm state, keep distinct evaluator identity and record repeated-row exposure,
updates, memory and compute. This component supplies no admission authority,
paired CLI, replay ledger or scientific benefit. Final data, probe updates and
benchmark learners remain isolated. Preserve all negative and failed results.


## Runtime composition evidence, 2026-10-16

The existing warm/stream CLI now implements separate post-reward zero-loss
control updates with original episode DTOs and bounded exposure/receipt charges.
Native8+8updates pass on shared-warm forks; the full three-readout comparison
fails at its declared90second diagnostic cap. Keep that failedcase spent and
comparisons null. Later narrow duplicate-proof removal is fake/static qualified
only; any larger engineering allowance requires a separate prospective versioned
protocol and review. No scientific benefit follows from correct integration.


## Explicit v2 allocation evidence, 2026-10-16

The originalv1 readout timeout remains SPENT. A separately selected version2
uses210readout/300whole/330parent caps from measured90s/95prediction cost and the
qualified duplicate-proof reduction. Original45reward/control phases, scientific
choices, old default paths and full3readout requirements remain. The native v2
case completes all readouts: zero-gradient control has lowerCE than reward but
worsens fromwarm; allbothcueacquisition false. This completes a conditional local
comparison, not usefuladaptation or circadianbenefit. No newnativecase follows.
