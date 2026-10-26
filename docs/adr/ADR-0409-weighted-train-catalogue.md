# ADR-0409: Separate weighted TRAIN catalogue updates

## Context

The online weighted study completed but did not acquire exact A/B/C skills.
Testing joint weighted fit requires a valid catalogue boundary first. The joint
composer produces independent rows with episode=None and original source spans;
the weighted episode adapter admits one complete1..16-row trajectory. Joining
six trajectories would discard their actual episode semantics.

## Decision

Add a separate adapter for an owned TrainCatalogue:1..6 complete TRAIN raw296/K7
sources, total at most54 rows, and exact original half-open spans. Reuse existing
catalogue causal/schema validation and native PC54-row numerical primitive.
Derive fixed aggregate weights N/(present_classes*catalogue_class_count), absent1,
separately from the original per-episode rule. Explicit validated weights feed
one atomic BP/PC/neutral native update with distinct catalogue diagnostic IDs.

Preserve the complete native transition, donor identity/state/gradient/RNG/mode,
source evidence and counters before one payload publication. Foreign changes
cause refusal without rollback. Uniform observed BP weights use the ordinary
unweighted reduction order because the constant cancels; this gives exact native
all-one state parity, including momentum. PC reuses the accepted staged update.

## Alternatives

- A fictitious54-step episode would misrepresent source boundaries.
- Expanding the existing episode port would silently change its contract.
- A new stream framework is unnecessary; the existing schedule/runner can later
  compose this bounded update with its original release/budget/probe guards.

## Consequences

The original episode, schedule, public route and baseline math remain unchanged.
Full-state native parity, weighted oracles, PC-neutral continuation and atomic
refusal checks pass in scoped composite gates; initial fixture failures remain.
Source validation/copy/projection costs add computation and memory. The adapter
does not enforce a lifetime budget or authenticate historical origin of arbitrary
caller data/weights; released TRAIN authority and budgets remain caller duties.
Public weighted joint streams and scientific evidence are separate next work.
