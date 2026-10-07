# ADR-0125: Typed overrides for the existing configurable continual study

## Context

The fixed v14 manifest rejects every changed field and its bundle binds
the original source, baseline, seed, and outcome identities. Allowing
new tunable values under that protocol ID would weaken its result.
The older continual-shift CLI is genuinely configurable, with named
`baseline`, `strength-case`, and `hardest-case` profiles and many
individual flags. Its JSON result already contains the resolved typed
config, but it had no compact override interface or separate resolved
config artifact with preset and override provenance.

## Decision

Keep v14 fixed. Add a pure resolver for the existing
`ContinualShiftConfig` hierarchy. A documented allowlist covers only
the data/phase, shared width, noise/transform, validation-fraction,
and sleep-interval fields already exposed by the CLI. Exclude
`protocol_id`, `model_order`, whole `circadian_config`, and baseline
learning-rate fields. Parse repeated `--override FIELD=JSON` inputs
with duplicate-key rejection and exact Python integer, finite numeric,
or integer-array validation. Apply overrides after the named profile
and existing flags; run the existing config validator before any data
construction. The resolved record must also serialize as finite JSON.

When overrides are used, require `--json-result` or
`--resolved-config`. The latter writes an exclusive artifact with
schema ID, preset, seeds, explicit override values, and the full
resolved config. The existing result JSON continues to embed that
same full config. Default CLI arguments and historical result schema
remain unchanged.

## Alternatives

- Make v14 configurable in place: its fixed protocol identity and
  prior evidence would become ambiguous.
- Add more one-off CLI flags: this scales poorly and hides the final
  merged configuration.
- Permit arbitrary dataclass field overrides: users could silently
  change baseline rates, model order, or protocol under a familiar
  preset name.

## Consequences

The configurable continual route now has explicit precedence and a
reviewable full config artifact. Unknown, duplicate, malformed,
nonfinite, and type-invalid overrides reject before training. This
does not turn the historical `continual_validation_v1` result into a
matched v14 comparison; its existing comparison-scope label remains
descriptive. No seed, baseline, metric, or trigger was selected from
the new smoke result.
