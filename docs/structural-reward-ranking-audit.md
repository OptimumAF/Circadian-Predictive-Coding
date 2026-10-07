# P4.7a fixed reward-to-structure rank audit

This rank-only protocol was fixed before observing its scores or choices.
It measures the existing implementation. It does not select a new policy,
train a model, apply a structural mutation, or open held-out data.

Both shallow NumPy and CPU Torch heads have four neurons. The only
eligible split or prune candidates are original neuron IDs 0 and 1;
the relevant cap is one. Both have equal output-weight norm. For split,
their chemistry is 1.00 and 0.99 while other neurons have 0.00, above
the default 0.80 threshold only for 0/1. For prune, chemistry is 0.00
and 0.01 while other neurons have 1.00, below the default 0.08 threshold
only for 0/1. All cooldowns are zero and minimum age is zero. No actual
split or prune is executed.

The hidden-output gradient magnitudes are `[6, 0, 0, 0]` on step one
and `[0, 2.5, 0, 0]` on step two. EMA decay is 0.5. Historical reward
factors are 1.0 then 1.5, both allowed by the existing switch. Compare:

1. no importance in ranking (`split_importance_mix=0`,
   `prune_importance_mix=0`), using the same reward-weighted history;
2. existing importance mixes (0.20 split, 0.35 prune) with both reward
   factors set to 1.0, a counterfactual importance history;
3. the same existing mixes with reward factors 1.0 then 1.5, the
   historical reward-weighted history.

Weights, chemistry, eligibility, initial EMA, proposal caps, and score
components other than the stated factor are identical within each
backend. The audit checks exact EMA values, split/prune scores and
candidate IDs, plus a constant-factor control where multiplying both
gradient steps by the same positive reward factor cannot alter the
min-max-normalized importance ranking. Backend differences, if any,
must be recorded rather than adjusted. A changed candidate proves a
distinct ranking signal, not a held-out learning benefit. P4.7b owns
the outcome comparison before an additional heuristic can be accepted.

## Observed fixed-rank result

Both NumPy and CPU Torch produced the same result. The unweighted
importance EMA after two steps was `[1.5, 1.25, 0, 0]`; the historical
reward-weighted EMA was `[1.5, 1.875, 0, 0]`. With equal norms and the
fixed chemistry, the two eligible candidate scores and one-slot choices
were:

| Ranking lane | Split score ID 0 / ID 1 | Chosen split | Prune score ID 0 / ID 1 | Chosen prune |
|---|---:|---:|---:|---:|
| No importance mix | 1.000 / 0.993 | 0 | 0.700 / 0.693 | 0 |
| Plain gradient importance | 1.000 / 0.961667 | 0 | 0.350 / 0.404833 | 1 |
| Historical reward-weighted importance | 0.960 / 0.995 | 1 | 0.420 / 0.3465 | 0 |

Thus the historical factor is already a distinct structural-ranking
input when its value changes across gradient histories. Applying the
same positive 1.5 factor on both steps scaled the EMA uniformly and
left normalized scores unchanged. No neuron was actually split or
pruned; wake and sleep clocks remained zero and all four original IDs
remained active. The result is a causal rank check, not evidence that
any lane improves held-out accuracy, retention, compute, or capacity.
P4.7b remains necessary before accepting an additional heuristic.
