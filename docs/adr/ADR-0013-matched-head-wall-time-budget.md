# ADR-0013: Version the matched-head wall-time budget

## Context

The fixed-feature three-head route equalizes examples, initial parameter
tensors, and representation, but an equal epoch cap does not equalize compute.
PC heads perform latent relaxation; the circadian head can also perform guard
checks and sleep work. The existing `train_seconds` is an observation from an
epoch-limited run and cannot itself enforce a fixed-time comparison.

## Decision

Add `vision_three_head_fixed_feature_wall_time_v1` with one positive, finite
per-head training deadline after the shared feature bank and head
initialization. It includes optimizer setup, wake updates, latent relaxation,
guard checks, and circadian sleep/rollback. It excludes the shared feature
setup, outer validation, and final test. The same budget is given to each
matched head. Target-accuracy stopping is disabled; the epoch cap is a
safety limit and a run fails before final test if any head reaches that cap
before its deadline.

Check the deadline before launching each wake batch, before guard/sleep work,
and after completed guard/sleep work. Synchronize device work before checks.
An in-flight operation can finish after the deadline; report that overrun,
the actual elapsed time, completed epochs, partial-epoch examples and batches,
and the reason training stopped. Keep the epoch-limited protocol and timing
scope unchanged. Preserve equal-initialization hashes as parameter-only
hashes; trained-head hashes additionally cover PC traffic and circadian
adaptive/structural RNG state. Neither hash is an optimizer checkpoint.

## Alternatives considered

- Comparing `train_seconds` from equal-epoch runs would leave training work
  unequal and cannot guarantee a common time allowance.
- Terminating a kernel or sleep event mid-operation would leave model state
  uncertain. Boundary checks preserve completed updates and expose overrun.
- Changing the existing epoch protocol in place would alter the meaning of
  earlier reports without a versioned boundary.

## Consequences and open work

The wall-time route compares head training on one shared cached feature
view. It is not an end-to-end image-training comparison; setup costs must be
reported separately. A small CPU smoke verifies wiring and deadline status,
not model ranking. Peak host/device memory, capacity-matched budgets,
equal-effort tuning, repeated confirmation runs, and local GPU checks remain
open under P1.8/P1.7. Wall-clock timing can vary with system load, so a
single timed run cannot support a comparative claim.
