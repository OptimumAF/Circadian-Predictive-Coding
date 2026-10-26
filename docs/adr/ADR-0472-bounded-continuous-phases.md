# ADR-0472: Bounded update counts in the existing continuous A/B/A runner

## Context

Default24-update public21/RGB MLP cases fail the fixed native task despite
improved offline imitation. A simpler public21 nearest control works locally.
Neither result establishes why neural learning fails. The existing generic
runner hardcodes8updates per phase and24-based budgets/costs, preventing an
explicit bounded optimization-depth diagnostic without source edits.

## Decision

Add keyword `phase_updates` with unchanged default8 to both existing entry
points, and an optional `--phase-updates` to the existing MLP stream CLI.
Accept exact native integers1..128 before episode admission/fit/model work;
CLI count/scenario checks precede teacher IO. Derive budgets/inbox/ticks,
receipts and all training-dependent cost formulas. Four endpoint checkpoints,
eight context readouts per kind, objective/optimizer/seed/statistics/source and
actor guards remain unchanged. Omission/default preserve report/call behavior.

Why128: it bounds an ordinary run to384retained training packets, below the
ordinary inbox default1024, and768logical ticks. It is a finite API guard,
not a memory guarantee or scientific compute allowance. No old budget renews.

## Alternatives

Editing the constant per experiment hides the selected count and leaves24-based
work/budget assumptions. A new runner duplicates orchestration and state guards.
An adaptive convergence criterion would choose work from observed outcomes and
change the experiment. The explicit fixed count is the smaller reusable change.

## Consequences

Later experiments can freeze a single count and report its exposure/computation
honestly. This implementation does not show that more training helps, prove
native/generalization/CPC benefit or authorize a sweep. Historical24-update
stage readers/protocols and negative results remain unchanged. Full roadmap
requirements stay open. See [guide](../continuous-phase-updates.md) for costs,
interfaces and exact verification evidence.

## Subsequent fixed p32 evidence, 2026-10-19

One prospectively reviewed96-update arm improves final A/B mean TRAIN fit with4xtraining exposure and higher math, while three metric differences and worst gripper errors worsen. 72checks perplatform and independent readback accept scope. All negatives retained; native/generalization/CPC benefit unproven. See docs/public-vector-mlp-depth.md and its linked protocol/results/review.
