# ADR-0484: Explicit bounded readiness teacher roll-in

## Context
The accepted student-only correction study queried50ALIGN labels, improved
pre-update training diagnostics, and retained0native task success. Later-phase
acquisition coverage on student trajectories is missing. Readiness controller and
raw-label/private-update/frozen-return paths already exist.

## Decision
Add a fixed1..49teacherprefix to the existing collector with per-control student
proposal, teacher target, actual command, and attempted/returned source receipt.
Then return to the frozen student; no adaptive outcome-driven switching. A pure
selector and small bounded app ledger preserve layer direction. Mixed success
cannot become student-only success; candidate return remains purestudent.

Existing native controller sends/storesfloat32; preserve float64 proposals/labels
but encode the selected executed command exactly in binary32. Why: the initial
fixture exposed actual precision mismatch; tolerance relaxation would lose exact
raw provenance. No default policy/controller behavior changes.

## Alternatives and consequences
Repeating offline dose or another ALIGN-only correction batch supplies no later
phase supervision. A new collector/orchestration framework duplicates existing
routes. Teacher roll-in changes behavior/data and must be explicitly budgeted;
labels remain provisional off-reference. Predictions during teacherprefix remain
paid; actual executed gripper controls stage history. No guaranteed autonomous
learning/reliability/generalization/safety/CPCbenefit follows from coverage or
mixedbehavior task completion. All prior negatives remain.
