# ADR-0124: Resume v14 only from a checked unscored trial prefix

## Context

The fixed v14 NumPy comparison has six independent seed/arm trials and
one global final-role release. P5.3a makes completed artifact directories
atomically visible, but interrupted training previously restarted all
six trials. A trial's arrived role object carries a deferred source
reference, so serializing the object without care would also copy
unopened final fields into an unscored checkpoint.

## Decision

Use one complete seed/arm trial as the checkpoint unit. A separate
format-10 trusted local pickle contains only a Cartesian prefix of
unscored trials. Strip deferred source references before persistence;
reconstruct them deterministically from the fixed manifest and seed
only after all six trials are present. Validate every stored trial's
role, replay, work, clock, structural lineage, and matched-arm facts
before another training update. Re-run the existing global preflight
before final release.

Keep a hidden, atomic `run-state.json` cursor with exact immutable
checkpoint filename and SHA-256, full captured environment, config and
protocol hashes, capture mode, next cell, and incomplete/failed/
canceled/completed status. Save a new checkpoint before advancing the
cursor; a crash between the two leaves an orphan file and the earlier
valid prefix. Use a local OS file lock that releases on process exit.
An interrupted trial restarts from its beginning. If a completed
public bundle already exists after a later failure, verify its exact
manifest and payload bytes before completing the sidecar or status.

## Alternatives

- Save an epoch cursor inside each v14 trial: this would duplicate a
  more complex mutable model/RNG/role state machine for six short
  independent trials.
- Put partial training files under the public run ID: report readers
  could mistake them for successful outputs.
- Pickle arrived roles unchanged: the deferred final source would be
  stored before the global seal.
- Rewrite a single checkpoint path: a crash between replacement and
  run-state update could point the old cursor at new bytes.

## Consequences

The opt-in `--resumable`/`--resume` route preserves original v14
training, scored, and measured data bytes in the recorded environment.
It keeps immutable checkpoint history and may redo one interrupted
trial or global scoring. A changed source, runtime environment,
configuration, protocol, capture mode, checkpoint file, or malformed
trial prefix stops before training. Pickle remains trusted-local only;
the hashes detect accidental edits and drift, not an adversary who
can rewrite both state and checkpoint. The public bundle and measured
sidecar keep their existing schemas and verifiers.
