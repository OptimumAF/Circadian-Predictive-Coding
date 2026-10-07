# ADR-0128: Resolve the root toy CLI configuration

## Context

The documented root toy command has baseline and indepth modes. Its parser
held sample, epoch, seed, and scenario defaults; other settings came from
`ExperimentConfig` and `CircadianConfig`. Its optional baseline JSON saved
the report without those settings, while indepth had no config artifact.
P5.4 requires a named, validated, fully resolved configuration.

## Decision

Declare the original defaults as the sole typed `historical-toy` preset,
with the three `PC_*` environment values applied before legacy flags.
Keep the existing flags and their precedence. Permit repeatable typed JSON
overrides only for config fields already exposed by those flags; require an
artifact path when overrides are supplied. Validate the resulting config
and ordered seed/noise request before training. The indepth runner and
artifact builder share the exact per-cell config constructor.

New CLI baseline JSON keeps its existing top-level report fields and adds
`resolved_config`. `--resolved-config` writes the same complete record for
either mode to an exclusive new path after a successful run. The record
contains the mode, preset, explicit tokens and overrides, base config, and
the ordered per-cell configs. It is built before training and does not
read scores or select a winner.

## Alternatives

- Add only a config field to `ExperimentResult`: rejected because indepth
  needs a grid and results built directly through the API need not know CLI
  inputs.
- Offer overrides for learning rates, model order, or nested circadian
  settings: rejected because the existing CLI did not expose these choices
  and this descriptive route has no frozen tuning protocol for them.
- Rewrite old result files: rejected because they remain historical evidence.

## Consequences

The no-flag and representative flagged config hashes, baseline rates,
model order, default protocol, and stdout workflow stay fixed. Invalid
settings now reject before training. The toy comparison remains descriptive;
the new config record is provenance, not a matched-model result or a
test-informed selection rule.
