# ADR-0190: Retain static reader callbacks across runtime failures

## Context

The actual late-file error preserves V2's newly created selector lambda in its
traceback, adding a callable to complete held process membership and changing the
containing marshal hash. Full-suite pytest basetemp additionally retained221MB of
unrelated fixture output, violating the declared owned64MB cap. Both failures and
all bytes are preserved; archive/cleanup does not turn either gate into a pass.

## Decision

Replace both equivalent V2 selector lambdas with one typed module function returning
the same immutable metadata sources. Keep public ports/file checks/diagnostics/
native audit/whole runtime observations intact. Ordinary tests use pytest tmp_path
and report complete artifact locations through JUnit properties. Single-use local
supervisor copies every runtime group byte after completion; unrelated pytest
fixtures use default temporary locations. Preserve whole failed tree in a verified
ZIP with all regular bytes/directories/link targets before checked native cleanup.

Prospectively bound this child by500s correctness including preparation/failures/
focused/current full548, separate180s static/metadata/shared whole closing and64MB
saved output. Only this leaf may complete after current597-file static/full548/
complete readback/whole original source-history-criterion preservation.

## Alternatives

Ignoring the added lambda, reducing observations, filtering GC, normalizing code
hashes, weakening the audit or dropping failed output would weaken acceptance.
Reserving whole-suite basetemp for metadata unnecessarily duplicates unrelated
fixtures into owned proof storage. None is accepted.

## Consequences

All96 controls pass with actual first2/other3 observations and releases. No provider,
source/array/model/final execution occurs in the selector. Original j3/j3a/resource/
provenance/scientific gates remain unchecked/unaccepted. Broader code constant
reference-count stability and non-audited transient callables still require proof;
this static callback does not claim to close those global obligations.
