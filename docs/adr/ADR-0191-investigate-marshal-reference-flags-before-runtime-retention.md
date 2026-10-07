# ADR-0191 — Investigate marshal reference flags before runtime retention

Status: accepted for diagnostic controls; production retention/source admission open.

## Context

Whole real runtime observations identified marshal hashes changing on legitimate
first-use/exception paths. No historical raw code serialization was saved. A hash
change alone does not identify a semantic change or prove source correspondence.

## Decision

Split mechanism investigation from observer stabilization. Preserve production
code/audit and all prior failed acceptance gates. Compare complete fresh raw code
bytes, every native public field, recursively encoded constants, reference/alias
lifetimes, actual first return/controlled exception, and genuine content changes.
Keep every failed fixture/source/typing/readback attempt and budget. Strengthen
the experimental retention control to include native metadata after a line-table
alias changed raw bytes. Decode saved bytes independently; normalize no observer
bytes and reconstruct no unsaved historical executable content.

## Alternatives

- Ignoring changed hashes or removing nodes would weaken complete observation.
- Normalizing or hashing only selected fields would change the identity contract.
- Retaining constants alone misses native metadata sharing.
- Editing production before mechanism controls would conflate two acceptance gates.

## Consequences

18 controls pass and genuine content changes remain distinguishable. This supports
a bounded production-retention follow-up, without proving private interpreter
fields, complete graph capture, disk-to-loaded source/version or transient coverage.
Current69 code/39 docs and original150 closure remain whole and preserved. Scientific
admission, old failed resources, seeds, baselines and metrics remain unchanged.
