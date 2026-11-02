# Single-head LS with bounded arrived-TRAIN replay

The existing ordinary context-return command can opt into its existing recent
or coverage episode replay policy with the single-head linear-fit learner:

```text
python -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --train-update minimum_change --replay coverage --output NEW.json
```

Use the updated checkout and a new output path. Retained development wheels
predate this option; a new installation has not been qualified. The command
performs training and needs a prospectively budgeted local run. It is not a
request to repeat a spent experiment. `--replay recent` selects the existing
recent policy. Omitting `--train-update` keeps cold fits for both wake and replay;
omitting `--replay` keeps the original no-replay path.

This composes existing `AttributionStream`, `StreamRehearsal`, episode codec and
candidate consolidation. Only original arrived TRAIN observations/actions are
retained. Selection follows the existing declared recent/coverage policy, with
no final/probe labels or task-outcome feedback. The same selected LS operation
is bound to all owned forks, including replay proposals. No new solver,
optimizer, row-selection score, replay framework or capacity is introduced.
Separate-head, paired, private, reserved-probe, controller and native BP/PC/CPC
replay combinations remain unsupported by this option.

For explicit **shared-head joint packet** replay, use the same command with
`--replay coverage --replay-joint joint`. The equal-row `--replay-joint current_only`
control repeats the current packet. Both cold and explicit minimum-change TRAIN
operations remain available; neither becomes a new default. This option requires
ordinary context-return shared LS alone. Its input experiences receive original
replay consent before source creation; changing that consent or prior paid work
refuses further updates. It does not renew inbox permissions, source clocks or
an expired earlier run.

Why coverage: the existing two-key bank retains the latest arrived TRAIN episode
per declared context. Recent-only retention can discard the old context before
the next two-update replay boundary. Joint mode reuses that existing bank and
selector, drops whole duplicate packets already in the current wake, and never
selects by task outcomes, loss or final/probe labels. No distinct retained packet
means an explicit skipped event. Current TRAIN6 + retained TRAIN6 fit together;
the control repeats current6 twice. There is no probe-train bank or promotion.

This new mode starts with its own replay limit **72 fitted-row exposures/30s/
512MiB**, while the original source remains **12 wake updates/30s/512MiB**.
The fixed source schedule has two early skips then four additional 12-row fits,
so a complete case is expected to fit16times/120rows: wake12/72 plus replay4/48.
The replay charge includes current rows already fitted in wake. Separate clocks
and repeated exposures are not a matched-computation or efficiency claim.

`replay.joint_shared` identifies the explicit policy and bounded bank-guard bytes.
Each eligible event records actual current/retained/total fit rows and output
array bytes. Proposal applications, unknown applications, callback returns and
candidate consolidation commits remain different readings; none is a serving
promotion. Failed/lost returns stay paid. The bank pin retains one bounded current
value/ownership signature, refreshed only after owned arrival/selection or an
authorized continuation. Changed data, policy, budget or prior event refuses.
Bank-guard bytes exclude temporary serialization/copy/metadata/process overhead.

[Shared joint software evidence](reports/shared-joint-train-20261031.md) includes
tiny genuine one/two-row fits and full **scripted** ordinary sequences. A separate
[fixed native comparison](reports/ordinary-shared-joint-comparison-20261031.md)
then observed both arms acquiring A and B from prior failures. Joint retained
both pairs; current-only lost one old task during B, recovered it on return,
then lost one B task. Each arm used16fits/120rows. This is one known development
template case with an unchanged deployed actor, not generalization, efficiency
or CPC evidence. Earlier #354 remains PARTIAL/SPENT; no spent run was retried.

The fixed ordinary recipe plans twelve wake fits on72rows. Existing replay
cadence adds up to six fits on36retained-row exposures. Replay uses its own
original exposure budget and30s clock; the wake budget remains12updates/30s/
512MiB. These are separate authorities, never a renewed wake allowance.
Source and deployed actor remain unchanged; work updates the owned candidate.
Extra replay exposure and computation are not a matched-compute comparison.

The report's original `training_work` counts ALL delegated wake and replay fit
ports. Its wake budget counter still counts wake commits only. New
`costs.fit_exposure_breakdown` presents wake committed updates/rows, replay
admissions, returns, commits and charged rows alongside the total fit counters.
A replay admission may fail before reaching a native fit. Charged exposures
survive failure, uncertain physical call counts remain unknown, and missing or
malformed cost fields are not replaced with zero. Recorded replay mode and
consent, prior checkpoints/receipts and source/actor/RNG/budget guards are checked
before more work. Changed foreign intent stays visible in the refused report.
Committed wake rows come from captured original batch lengths and actual update
receipts, so a refused edit to the displayed row counts cannot rewrite paid cost.
Without those original bindings, the derived committed row count stays unknown.

[Software checks and evidence](reports/linear-train-replay-20261030.md) cover
tiny actual consolidation/failure cases plus explicitly scripted complete and
partial captures. The later shared-joint comparison above supplies a separate
ordinary outcome, not qualification of the earlier partial sequential capture.
Earlier single-head no-replay results and the failed #352 observer
coverage gate remain preserved. Do not infer joint retention, generalization,
efficiency or CPC benefit from correct software composition. A small separately
frozen outcome protocol is required before any new study.
