# ADR-0302: Complete coupled data bytes separate from authority

## Context

Complete managed capture detaches one bounded graph with original authority
separate. Component native/managed-record codecs cannot independently preserve
cross-component aliases and all serving/pending state.

## Decision

Add a fixed-version codec using explicit allowlisted record/container nodes and
alias references. Reuse pure coupled graph/lifecycle/record checks and bounded
canonical JSON. Admit the complete wire graph and aggregate array metadata before
array materialization. Preserve ordered mappings, numeric dtype/bytes/layout/flags
and PCG64 state. Return detached data, with no live authority input or output.

Why this: one explicit node graph retains aliases across component boundaries,
while byte input cannot select executable classes. Existing independent binding
and digest contracts suffice; no authority protocol is added.

## Alternatives

Generic object loading permits executable construction paths. Separate component
envelopes lose shared objects without another alias representation. An authority
protocol would blur recovery restrictions and is unnecessary for a data codec.

## Consequences

The caller can carry complete supported captured data without restoring owners.
Encoded/decoded copies remain outside the source's logical raw-copy ledger.
Ambiguous ties between distinct identity-based unordered members refuse explicitly;
broader canonical unordered-graph qualification remains unfinished. No algorithm,
scientific claim or recovery completion restriction changes.
