# ADR-0083: Confirm the frozen representative matched study by separate scopes

## Context

P1.8p1 fixed three matched head settings by outer validation before final
access. P1.8p2a restored their exact result, journal, request, source hashes,
and manifest without a dataset read. The existing repeated-confirmation API
ran all scopes in one call, so it could not enforce the saved 240/240/600
second caps separately or retain a completed wall-time or memory seed after
a later worker failure.

## Decision

Use one local parent adapter and a bounded child for each fixed-data,
wall-time, and isolated-memory scope. The parent repeats the exact-artifact
preflight, requires three quiet CUDA readings under the frozen thresholds,
and saves a gate record with the runner SHA-256 before any final access.
Every child restores the same freeze and gate. The parent applies hard
subprocess timeouts of 240, 240, and 600 seconds, limited further by the
1,080-second total. Fixed-data writes each attempt to a synced journal;
wall-time and memory write a completed seed immediately. A failure saves
the available artifact hashes and scope error, without changing any seed,
candidate, metric, or budget.

The fixed-data scope uses one epoch per selected head. The wall-time scope
uses the saved five-second deadline per head and epoch cap 1,000. The
isolated-memory scope uses the existing 60-second child limit for each
fresh head process; nine such limits fit within the frozen 600-second
scope. After each scope, the parent checks complete seed/head coverage and
paired role, feature, backbone, initial-head, and capacity identity before
starting the next. The final audit also checks actual work, relaxation,
guard and sleep counts, deadline stop/overshoot, distinct memory PIDs,
separate RSS/CUDA allocator telemetry, and score dispersion.

## Alternatives

- Run the existing three-scope API under only one outer timeout. It would
  not enforce each predeclared scope limit.
- Save only one final JSON. A later timeout would erase already completed
  seed evidence.
- Change seeds, learning rates, or the scoring metric after the larger
  validation result. That would make the confirmation test-informed.

## Consequences

The unchanged seeds 181/191/193 completed in 142.232 seconds fixed-data,
178.568 seconds wall-time, and 356.843 seconds isolated memory; the total
was 677.735 seconds. The launch gate observed 2%/3%/8% GPU utilization
with at least 8,199 MiB free, and post-run utilization was 9%.
All nine fixed-data training/test rows, nine deadline head reports, and
nine different memory-child PIDs passed the saved-result audit. Every head
kept 32,954 trainable parameters. Fixed-data mean final accuracy was
0.8793 backprop, 0.7749 predictive coding, and 0.7087 circadian predictive
coding. Five-second wall-time means were 0.8934, 0.8757, and 0.8040 in
the same order. All per-seed values and population standard deviations are
retained in the local result. The circadian model remains below matched
backprop; no retuning followed final access.

Fresh-child observed trainer RSS means were approximately 1.909, 1.936,
and 2.152 GB in that order. CUDA allocated peaks and reserved peaks are
reported separately for each child. These are scoped observations, not an
exact incremental memory cost. Equal epochs produced different work:
each fixed-data head saw 16,384 examples/512 wake batches, while PC and
circadian each performed 1,024 latent steps and circadian made one sleep
attempt. Under equal wall time, heads processed different sample counts;
all stopped at the deadline with recorded overshoot.

This comparison covers a 16,384-example CIFAR-10 training subset, resized
224-pixel inputs, and a frozen ImageNet ResNet-50 representation. It does
not rank full-data or end-to-end trainable-backbone systems. The existing
image-level unmatched reference remains separately labeled. The negative
result answers this declared matched-feature study without establishing a
general head-family ordering.
