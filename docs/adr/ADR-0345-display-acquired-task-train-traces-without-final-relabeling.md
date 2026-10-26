# ADR-0345: Display acquired-task TRAIN traces without final relabeling

## Context

The accepted acquired-task and matched replay recipes expose twelve recorded
TRAIN episodes and explicit costs. The existing inspector requires a final
policy_evaluation readout for complete reports, so it cannot display this useful
adaptation evidence. Raw fit gains coexist with failed live retention.

## Decision

Extend the existing app projection and offline HTML/file CLI for a single
explicit acquired_return_diagnostics readout. Preserve original JSON pointers,
source records, execution phase and role; display TRAIN separately from final
evaluation. Validate the three supported protocol/control contracts, recorded
clocks, metrics and legal conditional action evidence. Put the added consistency
checks in a focused pure app module to keep the playback layer focused.

## Alternatives

Renaming TRAIN data into policy_evaluation would misstate its role. A second
inspector or runner would duplicate working playback and safe file publication.
Re-running learning would add no evidence needed for this presentation change.

## Consequences

Legacy final and paired off/on schemas remain distinct. Missing native counts,
topology, lineage and sleep events stay unavailable. All successful and failed
cues remain inspectable, alongside raw/live metrics and measured costs. This
makes recorded adaptation evidence accessible; it establishes no retention,
generalization, recovery-savings or circadian advantage by itself.
