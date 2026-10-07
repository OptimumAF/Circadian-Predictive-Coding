# ADR-0126: Declare the fixed v14 preset without rekeying results

## Context

P5.4 requires explicit, validated configuration at active experiment
entrypoints. The v14 bundle already saves its full typed manifest as
`resolved_config` and binds two historical raw result hashes. The v14
validator accepts exactly `fixed_trigger_replay_manifest()`.

## Decision

Expose one typed app preset, `fixed-v14`, and an optional CLI `--preset`
that accepts only that value. Resolve it before source capture, training,
or resume. Keep the existing manifest schema and raw JSON bytes; the
saved `resolved_config` and digest remain the exact fixed manifest.
Unknown preset names and settings fail before training. Audit the other
documented CLIs separately, since this fixed route does not satisfy the
configuration needs of a genuinely configurable benchmark.

## Alternatives

- Add setting overrides under the existing v14 ID: rejected because a
  changed setting would inherit the old scientific identity and hash
  expectations.
- Add a redundant preset field to the P5.1 manifest: rejected because it
  would change a strict public schema and invalidate existing completed
  bundles without providing more setting detail than `resolved_config`.

## Consequences

Default and explicit `fixed-v14` runs preserve the two raw SHA-256 values.
The CLI and direct app call reject unknown presets before training. A
future configurable v14-like study needs a new protocol and artifact
identity. P5.4 remains open for the multi-seed ResNet reference route.
