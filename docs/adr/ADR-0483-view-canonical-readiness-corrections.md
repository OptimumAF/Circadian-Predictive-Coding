# ADR-0483: Canonical readiness corrections retain raw arrival authority

## Context
Canonical quaternion features now train/serve consistently, but the bounded native
control remains negative. Existing readiness corrections provide labels on the
student trajectory, while the correction projection only supported public21/RGB.
Canonical smoke explicitly rejected queries and the driver hardcoded19→17→19.

## Decision
Derive a detached canonical model view from an owned raw readiness batch at the
existing admission boundary after original arrivals. Preserve both representations,
provisional provenance, executed student commands and exact targets/ticks. Reuse
the existing eight private updates/frozen runner. Add explicit fixed17 acquisition
rather than labeling base acquisition as B-learning/retention. Bind new canonical
observer boundaries to original native state, budgets, paid receipts and bounded
prior evidence; retain defaults.

## Alternatives
Changing raw observations in-place loses measurement provenance. Requalifying
geometry labels as expert evidence would overstate supervision quality. A new
orchestration framework or more offline dose would not address missing student-
trajectory coverage. Existing19 adaptation remains available as the default.

## Consequences
This delivers a usable correction path but no promised native task improvement.
Projection adds computation and pairing payload; account separately from optimizer
work. Final evaluation/probe/deployed learners remain isolated. Negative results
and existing acceptance gates remain. See state42-canonical-corrections usage.
