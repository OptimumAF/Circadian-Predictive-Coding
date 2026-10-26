# ADR-0316: Measure committed capacity on the existing stream

## Context

Split, prune, growth and bounded renewal are implemented, but the R2.6 comparison
needs actual parameter/auxiliary bytes and work accompanying outcomes. Existing
growth examples exercise publication and ceilings without a training-weighted
capacity trace. Starting width alone cannot establish an efficiency result.

## Decision

Add an optional single-use `StreamCapacity` helper to the attribution runner.
Read validated native snapshot arrays through an injected adapter measurement.
Record bounded scalar observations and successful wake calls/rows. Weight means
by the model used for each wake; initial and post-split samples affect peaks and
future work. Keep measurement time, completed work and unknown failed work
distinct. Pin original budgets, measurement port, bounds and schedule.

Support one predeclared guarded raw split through the existing structural
consolidator, checking predicted width/parameter/array limits before expansion
and measuring the actual committed result. Retain receipts and post-publication
observations on late failures. Keep the default runner and helper choices intact.

## Alternatives

- Counting only initial/final width misses the path used for training.
- Averaging every snapshot gives structural events an artificial extra vote.
- Reconstructing models during measurement adds hidden native work.
- A new runner or durable audit framework duplicates existing boundaries.

## Consequences

The fixed native example can verify scheduled growth and peak/wake-weighted
candidate capacity. Stored-array observations exclude temporary/process memory;
RSS and wall measurements remain separate. This helper initially excludes replay,
renewal coexistence and continuation. The full mechanism study and matched
structural BP port remain unfinished. No efficiency or learning benefit follows
from fewer stored parameters in early wakes.
