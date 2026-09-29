# ADR-0065: Keep a future-phase arrival gate separate from offline checkpoint recovery

## Context

The corrected ordinary continual runner constructs Phase B roles after all
Phase A models train. Its trusted-file checkpoint route constructs both
phases before Phase A because the v1 data digest binds both sets of
development roles. The two routes therefore have different label-arrival
behavior, despite using the same training helpers and protocol ID.

## Decision

Add an executable canary that denies Phase B source construction until
Phase A training finishes. Check both ordinary model orders and retain the
checkpoint route's observed early-access rejection as a passing negative
regression test. Keep the existing offline protocol and v1 checkpoint
unchanged. P1.3b will define phase-specific checkpoint identity before any
strict-online implementation claim.

## Alternatives

- Label the current corrected runner strict-online because its ordinary
  training helpers receive only current-phase roles. This would hide the
  checkpoint route's future-data prefetch and its known full A+B schedule.
- Change the v1 checkpoint digest in this gate. That would break a trusted
  recovery boundary without a format/version and resume migration design.

## Consequences

The canary gives a reproducible Phase A access test and a precise failing
boundary for the next change. Existing offline scores and checkpoint files
retain their meaning. Full strict-online replay budgets, schedule timing,
label-arrival reporting, and checkpoint continuation remain open.
