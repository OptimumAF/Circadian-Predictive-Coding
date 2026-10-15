# ADR-0273: Bounded atomic file transport through existing codec ports

## Context

The complete coupled PC/CPC codec now supplies bounded, detached native bytes.
Existing file stores use trusted pickle and unbounded reads; artifact publication
creates new directories rather than replacing one checkpoint. None supplies the
needed bounded raw-byte replacement reader for this native state.

## Decision

Add a small generic infra store depending only on the existing core codec port.
Validate owned controls, encode bounded bytes before mutation, stage beside the
target, require full write/flush/fsync/stream close/raw close, then atomically
replace. Retain sole descriptor ownership using closefd=False. Load regular files
with size/identity checks and a bounded actual-size-plus-one read; verify the
independently supplied digest before trusted codec decode. Failures preserve the
old file before publication and clean only owned resources with primary errors
retained. Successful replace has no fallible postcommit cleanup.

Actual Windows tests found differing `lstat`/`fstat` ctime values on unchanged
replaced files. Compare device/inode/size/mtime across those APIs on Windows;
retain ctime in descriptor-before/after checks and on other platforms. Mandatory
expected hashes and quiescence remain unchanged. Why this: those APIs provide
incomparable ctime observations on the tested runtime; suppressing every identity
check would discard useful change detection.

Provide a fresh-directory PC/neutral/active file example with injected fsync
failure and exact restored continuation. Keep all legacy stores and native
formats/settings unchanged; no new journal, codec or authority framework.

## Alternatives

Modifying the legacy pickle reader would risk working callers and still require
a new bounded transport contract. Using embedded self-reported checksums as
authority would not bind a checkpoint to independently expected state. Direct
overwrite could destroy the last valid checkpoint during partial write/failure.

## Consequences

Model bytes gain capped reads and atomic visible replacement with explicit
failure handling. Parent directory trust/quiescence and a single cooperating
writer are preconditions. Hostile races, ambiguous OS outcomes, power loss,
runtime lifecycle/cursors/cumulative allowances and durable restart ownership
are not solved by this component. Full R3.7 recovery and both incomplete timing
comparisons remain open; no scientific or release claim follows.
