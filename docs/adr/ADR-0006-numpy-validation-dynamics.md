# ADR-0006: Version NumPy dynamics around a validation holdout

## Context

The historical hardest-mode animation measured phase-B final-test accuracy,
predictions, and latency at intermediate epochs, including during phase A.
Its checked-in figures must remain attributable to that test-informed route.
The NumPy dataset generator exposed only training and test arrays.

## Decision

Add a deterministic, class-stratified validation partition of the original
training split. Record content hashes for each train, validation, and test
role, including the actual subsampled phase-B training data. The default
`validation_dynamics_v1` protocol uses phase-B validation examples for all
intermediate plots, probes, accuracy, and latency; plot bounds use training
and validation inputs. It computes a final test score only after all training
and sleep events. New default filenames include the protocol version and
existing files are never overwritten by the command. Preserve the original
route as explicit `legacy_test_informed_v0` for historical reproduction.

## Alternatives considered

- Removing the historical route would make old figure behavior difficult to
  reproduce.
- Keeping the test set in intermediate plots while merely excluding scores
  from learning decisions would leave test labels visible during training.
- Reusing training examples for plots would conflate fitted and held-out
  behavior.

## Consequences

The corrected route trains on fewer phase-A and phase-B examples than the
historical route, so their curves and rankings cannot be compared as if only
the label source changed. Intermediate phase-B validation is an offline
visualization even during phase A; it is not used as a sleep guard or training
signal and does not claim strict-online operation. Final test scores are
descriptive for this single fixed run. Matching budgets and a complete
strict-online protocol remain separate plan tasks.
