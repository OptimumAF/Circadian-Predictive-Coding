# ADR-0129: Resolve the single-run ResNet CLI configuration

## Context

The documented `resnet50_benchmark.py` command constructs a 110-field
`ResNet50BenchmarkConfig` from broad parser defaults and flags. It prints
a descriptive unmatched report but saves no resolved request. P5.4
requires a named typed preset, early validation, and complete provenance
for new artifacts without changing this historical CLI's results.

## Decision

Use the unchanged `ResNet50BenchmarkConfig` defaults as the named
`historical-single-unmatched` preset. The parser translates every config
field to its existing flag and obtains defaults from that preset. It keeps
the old dataset-dependent `--classes` omission and negative
`--target-accuracy` sentinel. Existing flags retain their behavior;
repeatable typed JSON overrides apply afterward to those same 110 fields.
The app validates strict scalar types, finiteness, and runner constraints
before a Torch runner call. An explicit override requires an artifact path.

New `--resolved-config` and `--json-result` destinations are exclusive.
The config record contains the complete request, preset, seed, fixed model
execution order, unmatched track, exact input tokens, and overrides. A
new result JSON retains the runner report fields and embeds the same
record. The adapter checks that
the runner's reported config and order equal the request before writing.
It keeps the original stdout formatter and performs no model selection.

## Alternatives

- Save only the old stdout report: rejected because it omits many settings.
- Add a narrow new flag surface: rejected because the existing command
  already exposes the full broad configuration, including learning rates.
- Change baseline rates, metric names, protocol identity, or historical
  result files: rejected because P5.4 is a provenance change.

## Consequences

Default and README flag-set full-config hashes remain unchanged. Invalid
requests fail before data or model construction. The route remains a
descriptive unmatched reference; the new record does not establish a
matched baseline or a result ranking. Future setting additions need an
explicit CLI mapping and validation update.
