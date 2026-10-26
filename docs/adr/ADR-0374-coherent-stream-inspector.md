# ADR-0374: Show saved coherent streams in the shared inspector

## Context

The accepted independent-data replay-removal report saves all eight arms and
their A/B trajectories, signed outcomes and unequal costs. The shared offline
inspector had no route for this report. Full R4.7 also needs other events that
this source does not contain.

## Decision

Add one exact protocol projection in the app layer and one standalone template
through the existing bounded file adapter/CLI. Project detached source-addressed
cells without importing learners or reconstructing scientific metrics. Preserve
all outcomes/failures across curve filters. Distinguish missing from null and
planned from measured work; separately supplied process costs are unverified.
Require frozen count/parity consistency before accepting complete report claims.

## Alternatives

A new dashboard/runner would duplicate working publication machinery. A generic
schema guesser could silently mislabel metrics or admit unsupported protocols.
Rerunning the experiment supplies no necessary evidence for a saved-data view.

## Consequences

Existing routes, 8MiB input/16MiB output limits and exclusive publication remain.
The new pure app module owns coherent-report validation/projection; infra owns
filesystem and presentation. No configuration/dependency/model changes. Partial
and synthetic failure fixtures are supported without fabricating completed work.
Topology/lineage/sleep stay unknown and full R4.7 stays open. See the linked
[executed evidence](../../artifacts/runs/r47-coherent-stream-inspector-20261017/RESULTS.md).
