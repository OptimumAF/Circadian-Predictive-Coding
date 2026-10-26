# ADR-0452: Explicit public episode clock at the observation boundary

## Context

Matched fixed1024 TRAIN optimization leaves phase-specific action errors. The
teacher chooses phases by its public step counter. This motivates an input
hypothesis; all29 original observations are unique, so it does not establish
that a clock is necessary or caused the previous native failures.

## Decision

Add a pure owned episode-clock projection and compose it in the existing frozen
robot smoke. Append either step/horizon or zero with the same input width.
Require complete TRAIN data for training projection, preserve raw/action/flag
fields and terminal seams, and record the explicit recipe separately from native
observations. Guard the admitted configuration and full learner state before
controls and after cleanup. Preserve the unclocked default and policy schema.

Why this: test one public input with a matched control through working ports,
without another learner, runner, simulator adapter or persistence framework.

## Alternatives and consequences

Inferring feature meaning from width would confuse equal-width controls.
Embedding schedule labels or teacher geometry would change information access.
Changing the policy schema would require broader persistence changes. The caller
therefore explicitly retains and supplies the matching feature recipe. This
recipe is not automatically included in a learner snapshot. Frozen standardizer
statistics remain part of full model state.

475 scoped cases and source type/lint/format gates pass. A separately frozen
single TRAIN study compares equal-width policies with identical initialization,
data and update budgets. Neither these gates nor offline fitting qualify robot
skill, broader learnability or CPC benefit. No real native case is admitted by
the TRAIN-study protocol.
