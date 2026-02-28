# Circadian Model Review Notes

Review source: `C:\Users\Avery\Downloads\Circadian model review.pdf`

## Scope

This note maps review recommendations to concrete repository changes and follow-up work.

## Implemented in this pass

1. Reward-modulated wake learning
- Added optional reward scaling in `CircadianPredictiveCodingNetwork` wake updates.
- Batch difficulty is measured from mean absolute output error relative to an EMA baseline.
- Scale is clipped via config to keep updates stable.
- Why this: gives a simple task-relevance signal without changing the core predictive-coding math.

2. Adaptive sleep budget scaling
- Added optional scaling for split/prune budgets based on:
  - energy plateau severity
  - hidden chemical variance
- Preserves `max_split_per_sleep` and `max_prune_per_sleep` as hard caps.
- Why this: reduces manual schedule sensitivity while staying deterministic and lightweight.

3. CLI exposure for new controls
- Added baseline CLI flags to toggle reward modulation and adaptive sleep budget behavior.

4. ResNet circadian parity
- Added the same reward-modulated wake learning and adaptive sleep budget scaling to `CircadianPredictiveCodingHead`.
- Exposed benchmark CLI/config knobs so ResNet benchmark runs can enable the mechanisms.

5. Test coverage
- Added unit tests for reward scaling behavior and adaptive budget expansion/contraction in both NumPy and Torch circadian paths.

## Still pending (next iterations)

1. Explore deeper-layer structural plasticity (current ResNet adaptation remains head-focused).
2. Prototype faster inference variants (reduced steps or amortized inference).
3. Evaluate reward-biased split/prune ranking directly (currently reward influences wake updates and importance EMA).
