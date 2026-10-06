# ADR-0192 — Retain observed code values before runtime freeze

Status: accepted for the bounded retention mechanism; full runtime admission open.

## Context

Whole observations rejected legitimate first-use/exception paths because raw marshal
sharing flags can change with references. ADR-0191 proved the cause with complete
native-value/raw controls, including line-table aliases. Existing retention held
functions/classes but missed code-only untracked native containers and metadata.

## Decision

Before baseline, use the existing native namespace reader to traverse full GC and
loaded-module roots plus function globals/defaults/keyword defaults/closures/
attributes, and retain every reached code object's public fields/recursive values.
Keep the observer identity, complete graph/native membership and audit/finally
contract unchanged. Move original diagnostic helpers verbatim to a pytest-free
fixture; never import a test runner into the real observed process. Record and
charge every failed control/fixture/static attempt before correcting it.

## Alternatives

- Constants-only lifetime control missed native metadata aliases.
- Hash normalization/node filtering would change the observation contract.
- Guessed private ABI pointers would not provide trusted build correspondence.
- Saved manifests cannot grant current live process/source authority.

## Consequences

581 preserved+new cases0 skips and complete real runtime/raw independent readback
verify the declared first-use/exception paths and old mutation denials. Strong
references live for the freeze and release afterward. Native public getters are
read even when deprecated; only their newly introduced warning noise is locally
suppressed with filters restored. Additional traversal/lifetime work is charged
to fixed budgets. Private localsplus/kinds/build/source correspondence and
unobserved/opaque/nested/transient coverage remain separate required gates.
No scientific method/source/seed/baseline/metric or previous failed cap changes.
