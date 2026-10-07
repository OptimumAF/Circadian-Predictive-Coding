# ADR-0082: Restore the exact selection freeze before representative confirmation

## Context

P1.8p1 saved six development-only trials, a durable attempt journal, and a
two-layer confirmation manifest. P1.8p2 is the first phase allowed to open
the final CIFAR source. A valid manifest alone does not prove that the
persisted trial rows, journal, request, and local source files are still the
same evidence that led to its choices.

## Decision

Make read-only restoration a separate gate. Before any dataset construction,
check the exact saved result, journal, and manifest file SHA-256 digests.
Then restore the unchanged P1.8o request, probe result, CIFAR archive, and
ImageNet weight hashes. Validate six complete equal-grid outer trials,
the 12 ordered start/completion events, common role/feature/backbone/initial
hashes, the zero-final-access record, the original quiet CUDA gate, each
head's independent outer choice, the typed manifest digest and selected
configs, and the outer digest binding all three confirmation budgets.

Expose this as `python -m scripts.restore_cifar_representative_selection
--preflight`. It prints only hashes, trial count, and zero final iterations.
It does not create a CIFAR dataset, train a head, or read a final label.
Confirmation adapters must call this same gate before the next quiet-device
check and before final access.

## Alternatives

- Restore only the typed confirmation manifest. That would leave the saved
  trial/journal evidence unaudited.
- Trust JSON fields without checking exact file bytes. A rewritten selection
  and recomputed internal digest could silently replace the predeclared run.
- Fold restoration into the first final-test worker. Isolating it makes
  missing or altered evidence fail before the irreversible test access.

## Consequences

The actual saved P1.8p1 artifacts pass the new read-only preflight with six
trials and zero final iterations. Tests reject changed result or manifest
bytes, a missing journal, a changed completed trial, malformed journal
events, and a nonzero final iteration before dependent work. The exact
confirmation scopes remain open: 240 seconds fixed-data, 240 seconds
wall-time, 600 seconds fresh-child memory, and 1,080 seconds total, using
only seeds 181/191/193. No new final score exists yet.
