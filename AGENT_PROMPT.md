# Circadian — Current Continuation Prompt

Continue the authorized repository goal using AGENTS.md, the supplied goal
objective, the [GitHub tracker](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/18),
the owning issue and its latest comments, and the actual checkout. Follow
[the issue workflow](docs/github-issue-tracking.md). The two plans and
docs/development-log.md retain historical requirements/evidence, not live status.

## Execution policy

Use one agent by default for planning, implementation, tests and self-review.
Do not create standing implementation/reviewer agents, routinely assign a second
agent, or delegate recursively. At most two agents total are permitted only for
a concrete useful bounded exception with explicit scope, disjoint ownership,
shared resource limits and a finite handoff. Close it when the task ends.

Perform source, procedure and result review yourself. Do not impose independent-
subagent or repeated orchestration gates; do not call self-review independent.
Preserve actual human/external review requirements and every substantive product,
testing, safety, scientific and resource acceptance criterion. Older multi-agent
execution prompts and dispatch records are superseded for future workflow.

## Continue from evidence

1. Classify the previous turn as progress, verified wait or no progress. Reuse
   accepted evidence when its relevant source and assumptions remain valid.
2. Inspect the next useful unmet behavior and relevant implementation. Preserve
   dirty work, unfinished requirements, historical failures and negative results.
3. Implement or validate the smallest useful end-to-end increment through existing
   components. Do not stop after planning, writing unexecuted tests or delegation.
4. Freeze local experiment limits, data, seeds, metrics and failure policy before
   execution. Keep final evaluation and disposable probes separate from TRAIN,
   promotion and tuning. Do not select favorable runs or manufacture a CPC win.
5. Run focused tests and applicable scoped lint/format/type checks. Review the
   combined source, procedure and observed results yourself; distinguish code,
   passing behavior, usefulness, scientific claims and release readiness.
6. Post start, meaningful milestone, scope-change, failure/blocker and session-end
   updates to the owning GitHub issue with task IDs, exact commands/outcomes,
   failures/skips, artifacts/costs, open parents and one precise next action.
   Update its checklist/status from evidence; link PRs and close only when its
   complete scoped criteria pass. Keep the full goal active until achieved.
   Do not append duplicate current sections to the historical plans or log.

Proceed with authorized local work without routine approval questions. External
publication, paid services, destructive actions or messaging require their
applicable authorization. Do not change other repositories as part of this prompt.

The current next product task belongs to the owning GitHub issue, not this prompt.
Background on repository guidance and optional prompt files:
[official OpenAI documentation](https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex).
