# ADR-0537: Connect released CLM features through existing ports

## Context

The conventional CLM head update has numerical parity evidence, while its first
engineering resource gate failed and remains preserved. Useful native adaptation
still requires features from the exact released encoder/head. The checkout had
pinned source/weight metadata but no executable CLM encoder adapter. The installed
WSL environment supplies Torch/Transformers; no new dependencies are needed.

## Decision

Add a local-only loader for pinned assets and a synchronous frozen-encoder
adapter. Implement the explicit upstream plain-text TRAIN recipe with CPU HF,
last nonpadding pooling and native normalization. Reuse core frozen identity,
paired TRAIN features and typed choice schemas. The outer script injects the
native head factory and composes those boundaries. Native serving/vLLM parity
remains separate and unqualified.

Transfer the already frozen large backbone rather than copying its weights.
Own only small head copies and bounded one-slot diagnostics. Preserve caller
CPU RNG and physical attempt counters. Missing/replaced assets refuse loading;
uncertain native work stops its owner. Safetensors and safe tensor-only head
loading exclude remote Python and random model substitution.

## Alternatives

- Claim native runtime from fake features: would not test the missing dependency.
- Add another wrapper/queue/capture framework: existing learner/inbox and typed
  decision ports already express the needed boundaries.
- Install vLLM or quantize/offload the8B graph: adds distinct dependency/kernel
  parity requirements. CPU BF16 fits the separately measured local capacity and
  allows one small, explicitly bounded request without those changes.

## Consequences

A passing frozen request would prove one executed local readout and source
recipe consistency, not useful adaptation or generalization. A wrong/failed
request remains evidence and grants no retry or tuning. Follow with a separately
frozen native head-learning protocol through the existing stream. Keep all other
R5.1 terms/runtime requirements, R5.3 consolidation, the failed original budget,
broader roadmap and release claims open until their own evidence exists.
