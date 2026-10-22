# GitHub issue tracking

GitHub issues are Circadian's live project tracking system from 2026-10-22.
Start with [the project tracker](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/18)
and the owning issue's latest comments, then inspect the actual checkout.

Why this: one task history makes current progress visible without keeping three
competing current sections synchronized. Scientific evidence stays in its
existing files, with issue links connecting scope, decisions and outcomes.

## Ownership and status

Use one issue per bounded implementation or investigation. Keep stable IDs such
as `R5.3` in titles; link broader parent tickets and dependencies. Imported lane
issues retain original requirements and scoped acceptance, including negative
results. Their historical checked items do not establish new scientific success
or close broader acquisition, retention, recovery, resource or release gaps.

Before starting, search existing issues and read comments to avoid duplicate
work. Identify the agent/chat and owned files in a start comment. A different
agent must explicitly hand off active ownership. The default remains one agent
implementing, testing and self-reviewing under AGENTS.md.

Use `Planned`, `In progress`, `Blocked` or `Deferred` in the latest progress
comment; open/closed is GitHub's durable completion state. A blocker includes
its cause, unblock condition and one next action. Do not represent deferred,
duplicate, canceled or unexecuted work as completed implementation. Record that
disposition explicitly if closing without completion. Reopen regressions.

## Required updates

Post at task start, meaningful milestones, scope/protocol changes, failed checks,
new blockers and session end. Do not post repeated unchanged polling updates.
Amend scope before executing changed scientific work. At handoff, record:

- Behavior changed and relevant files/boundaries.
- Exact commands, exit outcomes, pass/fail/skip counts and unrun checks.
- Evidence paths or committed links, observed costs, failures and limitations.
- Acceptance criteria still open, parent dependencies and one exact next action.

Example comment:

```text
Status: In progress
Owner: <agent/chat identifier>; task: <stable ID>
Scope/files: <bounded behavior and owned paths>
Progress: <observed change or result>
Validation: <exact commands, exit status, counts and skips>
Evidence/costs: <paths, source binding, resource observations and failures>
Open criteria/blockers: <remaining requirements and unblock condition>
Next action: <one precise command or coding step>
```

Close only when all scoped criteria are evidenced and self-reviewed. A completed
research investigation may have a negative result; its implementation or benefit
parent can remain open. Link a PR with `Refs #N`; use `Closes #N` only when that
PR satisfies the whole ticket. Do not close the project tracker for one increment.

Use connected GitHub tools with the OptimumAF account matching the repository
origin, or authenticated `gh` if already available. Verify write receipts. If
GitHub is unavailable, keep a clearly marked pending update locally, disclose
the failure and sync it when access returns. Never invent a successful update.

## Historical records and evidence

`DEVELOPMENT_PLAN.md`, `RESEARCH_ROADMAP.md` and `docs/development-log.md` retain
their historical bodies and original acceptance criteria. Their supersession
notices take precedence over old live-status and mandatory log-update text.
Do not add new current sections or maintain mirrored progress checklists there.
Read relevant preserved requirements when splitting an imported ticket, and copy
its complete acceptance criteria to the bounded child issue before starting.

Keep protocols, result files, ADRs, module docs and reproducibility evidence in
the existing locations. Update product documentation when behavior changes.
Issue updates must preserve budgets, sealed evaluation, adverse outcomes and
genuine human/external review requirements. Never describe self-review as
independent review. Local evidence paths do not imply committed/published bytes.
Do not upload secrets, private raw data or bulky artifacts just to update status.
Tracking does not authorize experiments, paid work, deployment or other messaging.

## Initial migration crosswalk

| Work | Issue |
| --- | --- |
| Project index and progress contract | [#18](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/18) |
| R5.7-CLM.F.I implementation closeout | [#19](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/19) |
| R5.3 paired-head TRAIN replay candidate | [#20](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/20) |
| R2 plasticity/replay/capacity | [#21](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/21) |
| R3 adaptation runtime | [#22](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/22) |
| R4 closed-loop demos | [#23](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/23) |
| R5 native heads/memory/consolidation | [#24](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/24) |
| R6 additional model families | [#25](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/25) |
| R7 adaptation controllers | [#26](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/26) |
| R8 capability/market hypotheses | [#27](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/27) |
| R9 circuit research | [#28](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/28) |
| P2 deferred deeper attribution | [#29](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/29) |
| P6 remaining controlled studies | [#30](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/30) |
| P7 measured performance work | [#31](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/31) |
| P8 gated extensions | [#32](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/32) |
| P9 validation/docs/release | [#33](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/33) |

R0 and R1's base roadmap checklists were recorded complete at migration; their
evidence stays preserved without creating artificial new implementation work.
Broader scientific applicability and outstanding legacy P requirements remain
explicitly open in the imported tickets. Future tasks belong in issues, not this
static crosswalk.
