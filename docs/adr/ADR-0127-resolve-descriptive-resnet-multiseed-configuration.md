# ADR-0127: Resolve descriptive ResNet multi-seed configuration

## Context

The actively documented multi-seed ResNet CLI is an unmatched reference.
It accepts many flags whose defaults lived in argparse, while its JSON
result showed only a subset of `ResNet50BenchmarkConfig`. Other model
settings and learning rates came from the dataclass but were omitted.
P5.4 requires a validated named preset and the fully resolved settings.

## Decision

Move the original no-flag defaults into one typed
`historical-unmatched` preset. Keep existing flags and precedence; add
repeatable typed JSON overrides only for fields those flags already
exposed. Validate the resulting config and ordered seeds before the
runner starts. Save the full base and per-seed configs, exact input
tokens, and overrides inside the existing JSON result. Require the
runner's reported config to equal the requested trial config.

## Alternatives

- Leave defaults in argparse and save only the current dataset/runtime
  subset: rejected because inherited model settings remain invisible.
- Open every dataclass field, including baseline learning rates, to
  generic overrides: rejected because this descriptive route has no
  frozen selection protocol for those new tuning choices.
- Rewrite old result files into the new schema: rejected because those
  are historical observations and should retain their recorded bytes.

## Consequences

New JSON results contain a complete resolved config; existing summary
and CSV fields remain. The no-flag and representative flagged configs
retain their pre-change hashes. The route remains an unmatched,
validation-selected reference and provides no new model ranking claim.
Future matched studies need their own frozen selection and provenance
contract rather than importing this preset as a matched baseline.
