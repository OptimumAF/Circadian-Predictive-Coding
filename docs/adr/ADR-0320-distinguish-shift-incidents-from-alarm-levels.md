# ADR-0320: Distinguish shift incidents from alarm levels

## Context

R2.7 persistence emits a level alarm on every qualifying batch. The accepted
prefix-calibrated trace has residual alarms8..13; update13 is the existing B
streak, so calling it new returning-context detection is misleading. Thresholds,
raw measurements and the negative experiment must remain unchanged.

## Decision

Add one pure bounded `ShiftIncidentTracker` consuming exact consecutive
`ShiftDecision` records. Report independent per-channel inactive/onset/continued/
cleared states and the original onset index. Clear on the first non-alarm;
refuse gaps, duplicates and a return to prefix phases after monitoring. Store
only three active onset indices, count and monitoring flag.

Expose this through default-off observer and existing shift pilot/CLI options.
Retain the old alarm-level outcomes verbatim and add separately named onset
outcomes after stream completion. Context annotations never reset tracking or
enter the detector. No changes to calibration, learner updates or authority.

Why this: event consumers need an honest new-trigger contract; a pure adapter
over existing decisions supplies it without changing the scientific question.

## Alternatives

Changing alarms to edges would silently change existing reports. Resetting at
known task boundaries would use unavailable context information. Refitting or
adding cooldowns would change detection policy without evidence or a protocol.
All are rejected for this increment.

## Consequences

Default schemas and native work remain unchanged. Opt-in tracking/recording is
included in observer time; reports gain bounded scalar records. Saved-data
reclassification can verify the contract without new learner execution. False
feature incidents and missed fresh returns persist, so full R2.7 stays open.
Successful observation and state tests do not demonstrate deployable detection,
adaptation, efficiency, scientific benefit or repository-wide readiness.
