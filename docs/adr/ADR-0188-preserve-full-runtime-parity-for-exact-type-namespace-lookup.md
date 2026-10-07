# ADR-0188: Preserve full runtime parity for exact-type namespace lookup

## Context

j1 profiled608760 namespace calls/0.743523s cumulative. Most ordinary exact builtin
containers have no native instance dictionary. Full runtime/source-version gates
remain unfinished; a filtered GC/module/native graph would weaken their scope.

## Decision

Shortcut only exact dict/list/tuple/set/frozenset using identity checks, avoiding
custom metaclass equality/hash. Preserve the complete old file before the one
permitted edit; all other59 lead files/original150 source closure remain unchanged.
Full actual original/candidate per-object namespace identities and whole canonical
body equality, independent readback/corruption controls and fixed matched lookup
benefit precede adoption. A slotted full GC inventory adapter exists only in parity
fixtures to provide identical unfiltered actual inputs; production uses native GC.
Require all current544 case IDs/current590-file static gates before completion.

## Alternatives

Persistent type/MRO caches require invalidation for legal type/descriptor/member
drift and are unnecessary here. Sampling or skipping native/file reads loses proof
coverage. j1 lossless subtree encoding is22.7x slower and remains unadopted.

## Consequences

One fixed complete-input pair shows27.819% lower namespace time and
4.377% lower whole capture time, without a variance/future speed
claim. Complete bodies/actual45793-object native lookup ledger retained. Separate
300s current regression protocol is explicit and justified by prior502 alone
131.0861267s plus unchanged runtime children; old j180s/scientific failed gates and
budgets are unchanged/unaccepted. Separate64MB full before/two bodies/closing
preserves old j16MB/j1_32MB. Current cases pass; nested schema/source-version/
MAKE_FUNCTION transient/full runtime and every scientific admission gate stay open.
