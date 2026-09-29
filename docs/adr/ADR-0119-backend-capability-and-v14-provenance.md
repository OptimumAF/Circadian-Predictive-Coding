# ADR-0119: Record backend capability and v14 provenance separately

## Context

P4.9 asks for a verified backend feature matrix and explicit result
scope. The completed v14 replay artifacts already have recorded SHA-256
identities. NumPy implements sleep replay and the v14 three-method
arrived-role runner; the Torch head has no replay. Phase 5's next task
defines an artifact schema, not a Torch replay experiment.

## Decision

Record current NumPy/Torch differences in
`docs/backend-capability-matrix.md`. Add a tracked JSON provenance
sidecar binding the fixed v14 train-only and scored protocol IDs, exact
artifact hashes, NumPy backend, method list, and absence of Torch.
Preserve the scored artifact bytes and protocol IDs. Do not port replay
to Torch without a separately predeclared matched experiment.

## Alternatives

- Add a `backend` field to the already scored v14 JSON: this would
  change the recorded byte identities solely for metadata.
- Port replay into the Torch head now: no current protocol defines a
  Torch counterpart, replay rows, compute caps, or matched baseline.
- Rely only on prose: readers and scripts could miss the backend scope.

## Consequences

The sidecar is machine-readable but separate from the JSON result. A
consumer must verify its hash against the artifact before using the
backend annotation. Future Phase 5 schemas should embed backend and
algorithm identity in the result itself, with a new schema/protocol
identity rather than silently rewriting fixed v14 evidence.
