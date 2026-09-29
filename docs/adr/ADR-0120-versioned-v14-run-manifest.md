# ADR-0120: Attach a versioned run manifest to opt-in v14 output

## Context

The fixed v14 train-only and scored JSON preserve role isolation and
matched work, but do not record the executing commit, dirty source,
runtime versions, machine, precision, or complete seed derivations.
Older trusted checkpoints serialize resumable model state, not a result
provenance contract. Rewriting the published v14 JSON would change its
recorded hashes without changing the experiment.

## Decision

Define a pure `circadian_run_manifest_v1` validator and a separate local
v14 bundle producer. Run the fixed six-trial study once; serialize its
train-only JSON before global final release, score the same study after
the existing full preflight, and write both original-format payloads
with a hash-bound provenance manifest. Capture Git/workspace,
dependency, and CPU facts before training and reject source drift
before writing. The disk verifier checks the complete cell grid,
protocol/config/role identities, hashes, and manifest status.

## Alternatives

- Add provenance fields to v14 outcome JSON: changes the fixed result
  byte identity and conflates P5 metadata with the v14 science protocol.
- Package old local JSON with newly collected machine information:
  the machine record would describe packaging rather than execution.
- Reuse pickle checkpoint formats: they require trusted input and do
  not represent inspectable cross-run result metadata.

## Consequences

The manifest is honest about `not_applicable` pretrained weights,
unavailable Git metadata, unmeasured timing, and same-environment
determinism. An incomplete write has no completed manifest and is
rejected. Recoverable interruption, atomic directory publication,
per-step observation streams, and broad artifact reports remain P5.2–P5.7.
