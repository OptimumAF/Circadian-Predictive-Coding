# ADR-0038: Restore Torch head state and split randomness together

## Context

The Torch circadian head already had an in-memory dictionary snapshot used
by vision and matched-head sleep guards. It copied weights, adaptive
per-neuron tensors, counters, reward baseline, and diagnostic history, but
omitted its model-owned split generator. After a rejected noisy split,
restoring weights alone changed the next split draw. Restore also assigned
fields one at a time without checking snapshot compatibility or alignment.

## Decision

Keep the existing flat dictionary API so current guard and trained-state
hash callers continue to work. Add a format version, static dimensions,
device, immutable configuration values, and a copied split-generator
state. Snapshot tensor values are detached clones and list/config values
are separate copies. Restore verifies the complete field set and static
metadata, stages detached tensor copies and scalar/list values, checks
topology and width bounds on a candidate head, and loads the generator
state into a separate generator. Only then does it replace live state.

Why stage the generator separately: `set_state` can reject malformed byte
data. A failed restore must not advance or replace the live generator.
The existing classifier `snapshot_state()` remains a head-only guard
primitive; copying the backbone at each sleep attempt would change its
measured time and memory budget. P3.3c adds a separate whole-classifier
boundary.

## Alternatives and consequences

A new snapshot dataclass would break the current dictionary-based guards,
hashing, and tests. Keeping the dictionary makes its keys a maintained
in-memory schema; the version and complete-field validation reject old or
malformed snapshots instead of silently accepting partial state. Current
trained-state hashes also include the generator directly, so it is
represented twice in those hashes; this is deterministic and does not
affect training. The format is not a durable checkpoint. P3.7/P3.9 own
atomic guard acceptance and file-backed resume.

## Evidence

`tests/test_torch_full_snapshot.py` first failed because the saved
generator state was absent. It now checks full-state equality after
changed-width noisy split and prune, next-draw continuation against a
control, copy isolation in both directions, and unchanged live state
after format/config/device/topology/generator rejection. The development
log records focused integration and full quality gates.
