# ADR-0123: Publish complete artifact directories with a sibling rename

## Context

The P5.1 bundle and P5.2 sidecar/projection writers created their
public directories before writing all files. Their verifiers rejected
a missing last manifest, but a partial public path could remain and
block a run ID after interruption. The existing trusted checkpoint
store already uses atomic file replacement, while artifact bundles
need a whole-directory visibility boundary.

## Decision

Stage each validated file set under a hidden sibling directory,
record staged-file progress and terminal failure/cancellation there,
and publish by a same-parent rename only after exact byte checks.
Use a target-specific exclusive local lock for cooperating writers,
refuse occupied public paths, and preserve failed stages for later
inspection. Keep the original manifest contents, data bytes,
directory names, and verifiers unchanged.

## Alternatives

- Continue writing a manifest last in the public directory: its
  verifier is safe, but partial public paths still look like runs.
- Write separate atomic files into the public directory: readers
  could observe a mixed set between replacements.
- Delete failed directories automatically: this discards evidence
  needed to diagnose or resume local work.

## Consequences

Readers see a complete public directory or none after a caught
failure. Hidden stages distinguish incomplete, failed, and canceled
publication states. A process crash can leave a hidden stage or
lock; P5.3b must define checked checkpoint resume and lifecycle
handling. Same-volume rename gives atomic visibility, while broader
filesystem crash durability and cross-platform tolerance are not
claimed here.
