# ADR-0268: Uniform dyadic raw coupled split and compatible merge

## Context

The coupled nonlinear model supports scalar hidden precision, uniform raw width
and complete CPC wake state. An isometric linear split does not commute with
tanh. Existing pruning/runtime reconstruction cannot derive a nonlinear energy-
preserving split by simply copying linear geometry rules.

## Decision

Duplicate every hidden coordinate into adjacent pairs, divide middle/output
effective contributions by2, and halve all hidden precisions. Preserve biases,
policy and count with duplicated chemical/gate coordinates. Re-encode raw factors
at the new width. Require reversible effective halving and bounded encoding.
Merge only exact raw pair images under the complete declared target config.
Keep the pure operation and explicit lineage separate from runtime prune history.

## Alternatives

Unequal unit duplication requires a new coordinate precision/objective contract
for the same diagonal energy identity. Linear isometric maps change tanh values.
Tolerant merging would require separate observed-damage criteria. Weakening the
existing runtime precision binding would admit configurations outside its scope.

## Consequences

Function and diagonal energy preserve in exact arithmetic; each latent gradient
halves, requiring doubled fixed inference step size for the replicated proposals.
Finite representation, exact compatibility and full owned CPC state are checked.
There is no bitwise raw roundtrip, raw SGD trajectory, guarded stopping, general
learning-benefit or serving authority claim. Mixed-history runtime transfer and
original parent R1.6 acceptance remain separate unfinished work.
