# ADR-0367: Compose incident feedback in the existing shift pilot

Date:2026-10-17. Status: accepted implementation; native tensor capture incomplete.

## Context

The existing calibrated stationary/recurring pilot observes concept changes but
does not act on them. The accepted core incident gate and shared stream rehearsal
already support future-boundary admission. The fixed feature channel has known
stationary false onsets; selecting a different channel from observed results would
change the scientific question.

## Decision

Add explicit `--calibrated --incidents --incident-replay` to that pilot. Keep
all original modes/settings. Route completed copied incidents after the observer
to the existing feature gate; period4/recent4rows/retention8rows192numeric bytes.
Separate four-row replay budgets are born at setup and share original clock,
RSS sampler and global deadline. Predictions name the preceding snapshot's actual
committed replay count. Use existing stream finish progress for partial paid finals.

## Alternatives

A new runner would duplicate accepted orchestration. Retuning the feature channel
would hide false-trigger costs. Same-wake admission would use unavailable evidence.
Rerunning a completed experiment merely for its failed tensor export would spend
more learner work without supplying a new learning outcome.

## Consequences

Production behavior passes386focused cases and scoped static checks. Eight actual
cases completed, with costly stationary false incidents and no circadian advantage
established. Whole native qualification stays incomplete because the stage wrapper
failed saving BP gradient tensors after scalar results; its spent attempt and
original acceptance criterion are preserved. See
[failure and evidence](../../artifacts/runs/r27-shift-incident-feedback-20261017/FAILURE_NOTE.md).
Next build on the existing isolated probe observer, with separately charged probe
work and fixed checkpoints, to examine learnability after actual replay.
