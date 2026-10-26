# ADR-0551: Use GitHub issues for project tracking

Date: 2026-10-22
Status: Accepted by Avery's tracking migration request

## Context

The development plan, research roadmap and development log repeated current
handoffs across large historical records. Avery requested GitHub tickets and
agent progress there while another agent continued implementation.

## Decision

Use [tracker #18](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/18)
and bounded child issues for live scope, acceptance, dependencies, progress and
handoff. Require start, meaningful milestone, scope-change, failure/blocker and
session-end comments. Preserve stable legacy IDs and full acceptance criteria.
Close only with observed scoped evidence and self-review; keep broader parents
open. Retain evidence/protocols in existing files and link them from issues.

Preserve the historical plan/log bodies with prominent supersession notices.
Change AGENTS.md, AGENT_PROMPT.md and contribution guidance so agents stop
maintaining mirrored live status. A research-task issue form supports new work.
The working agent confirmed its frozen run was terminal before policy edits.

## Alternatives

- Continue updating three current sections: recurring duplication and ambiguity.
- Delete historical plans/logs: loses scientific criteria, provenance and failures.
- Add a separate hosted board or tracker service: unnecessary for the requested
  issue-based workflow and introduces another status surface.

## Consequences

GitHub access is needed for live updates. When unavailable, retain a pending
local update and sync it explicitly later. Imported checks describe historical
scoped evidence, not fresh baseline validation. Existing scientific protocols,
budgets, negative outcomes, single-agent policy and genuine external review
requirements remain in force. The initial migration changed no source/tests or
experiments and ran no models or dependency installs. Avery subsequently
authorized committing and pushing only the tracking documentation/templates;
unrelated implementation work and raw experiment artifacts remain excluded.
