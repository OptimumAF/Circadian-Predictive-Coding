# ADR-0157: Bind all reproduced training facts before independent final release

Date: 2026-09-30
Status: Accepted for P6.7c1a implementation; final execution remains gated.

## Context

Both complete P6.7b train results have identical 134,554,378-byte JSON and
the exact recorded digest. P6.11a already fixes every seed, contrast, metric
and inference rule. The existing trainer retains independent A/B checkpoints
and their full state fingerprints. Its final-role API returns a new view;
the original sealed roles can still be checked after future evaluation.
Loading another whole decoded result beside all held models risks the
unchanged observed 512-MiB cap. No final values have been opened.

## Decision

Bind both original request/result/audit byte identities and paths, the scope
and old source identities, and the complete analysis declaration in a
separate immutable scientific scoring manifest. Preserve all 560 cells,
580 pairs, 1,680 final calls/67,200 examples and original resource caps.

Incrementally serialize the reproduced training dataclasses in the exact
existing artifact encoding: sorted keys, two-space indentation, default
ASCII escaping, finite JSON, UTF-8 and one LF terminator. Convert a dataclass
to a shallow field mapping when reached; do not deepcopy its entire graph.
Compare the entire fingerprint with the bound training results. Then verify
the exact held inventory and fact attachments, every sealed role, and every
live A/B parameter/width/full controller/selector checkpoint. The same
verification can run again after evaluation without releasing any role.

Split P6.7c1 into a scientific manifest/live-state gate, final-role/evaluation
composition, and full source/request/process/artifact gates. P6.7c1a uses
only original development training fixtures and no final evaluation.

## Alternatives

- Decode both saved results beside the new live model graph: unnecessary
  duplication of already byte-identical complete evidence under a fixed cap.
- Compare only checkpoint parameter hashes: omits raw costs, roles, controller,
  selector and other state; fails the original global acceptance.
- Increase the memory cap: unsupported before measuring the new bounded
  process and would change the frozen execution contract.

## Consequences

SHA-256 equality binds every fact under the existing byte contract, assuming
its collision resistance. A separate boundary must first verify both real
files, complete old audits/sources/resources and every new source/request.
The app proof alone does not establish file provenance or authorize final
release. Public verification accepts only the complete frozen production
scope; private development fixtures do not constitute reserved execution.
Incremental serialization reduces duplication, but makes no measured RSS
claim. Resource enforcement and complete scored repetition remain required.
