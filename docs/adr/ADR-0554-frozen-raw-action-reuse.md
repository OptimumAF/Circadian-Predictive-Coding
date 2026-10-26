# ADR-0554 — Reuse bounded raw action embeddings for a frozen catalog

## Context

The accepted native typed case spent61.21seconds after TRAIN feature extraction
on four queries plus head work. Full workflow episodes may require36queries.
Each query was re-encoding the same five public action descriptions; this made
the existing540second full-workflow ceiling doubtful. Native batch shape can
also cause small numerical differences (previous observation9.6223e-7).

## Decision

Add an explicit optional raw-feature declaration at the existing predictor and
binding. The workflow derives it from its already-paid TRAIN action rows. Every
decision still encodes its current query and projects both query and raw actions
through its own frozen heads. Default uncached behavior stays available.

Bound exact identity/catalog/contiguous storage and copy once per predictor.
Guard original and owned raw data on use, using owner/pointer/version and value
equality. Existing encoder guards remain authoritative. A detected mutation or
uncertain return stops the owner permanently; no TTL refresh or budget renewal.
The cache lasts only for that frozen encoder and exact catalog. Report raw
copies/reads/equality checks separately from head and encoder work.

Why this: it removes measured repeated backbone work without retaining stale
learned projections or changing examples, dose, optimization, decisions or effects.

## Alternatives

Increasing wall limits would retain redundant computation. Caching projections
would couple reuse to learned head versions and risk stale scores. A general
cache service with eviction/refresh is unnecessary for a fixed five-row catalog.

## Consequences

Workflow retains one original five-row matrix and one copy per actor. The backend
API bounds2..64rows with exactly logical backing storage. The declaration is
trusted producer input, not encoder provenance certification or a TRAIN port.
Checks do not contain malicious same-process code or certify tokenizer internals.
Literal parity must be tested; native parity/usefulness needs its own prospectively
budgeted observation. Reducing encoded rows does not establish a CPC advantage,
native latency bound, task success, calibration or repository-wide readiness.
