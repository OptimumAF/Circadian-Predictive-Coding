# ADR-0465: Select the existing continuous objective explicitly

## Context

The default visual MLP improves saved TRAIN loss but fails both native resets;
large TRAIN gripper errors already exist, so a coverage-only explanation is
unproven. The existing core supports strict binary-gripper CE, whose earlier
42D study also had negative translation/grasp tradeoffs. The current fixed stream
recipe hardcodes MSE and rejects mixed objective receipts.

## Decision

Expose an explicit optional axis through the existing recipe/result/JSON APIs
and a flag on the current MLP CLI mode only. Preserve default calls/reports,
validate both episodes' binary labels before fitting, and check receipts against
the actual policy objective. Reuse core kernels and all common physical metrics.
Report added CE row work without claiming equal computation.

## Alternatives

New loss kernels or a runner are unnecessary. Automatically extending epochs,
changing seeds or collecting more data would confound the immediate objective
hypothesis. Changing the default would silently alter the baseline.

## Consequences

A small predeclared visual control can reuse accepted MSE evidence and add only
the mixed arm. Unlike training losses remain separately named; all common-axis
tradeoffs, previous negative results and unfinished parent claims are preserved.
Any future native evaluation needs its own fixed budget/protocol. This is an
explicit baseline option, not a claimed repair or circadian advantage.
