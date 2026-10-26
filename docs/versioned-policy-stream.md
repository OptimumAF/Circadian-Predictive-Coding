# Policy changes in the recurring ticket stream

The separate [native reference-trial option](native-policy-trials.md) now accepts
this verified corpus with actual-update phase records and isolated development
readouts. It preserves this scalar recipe and its observed parcel-retention failure.

Run the existing chronological runner with a fresh output path:

```console
python -m scripts.recurring_ticket_workflow_example --policy-change --output policy-change.json
```

Default remains the conventional perceptron. Existing native method ports remain
available through --method bp/pc/neutral_cpc/gated_cpc, but new-policy neural
outcomes have **not** been qualified or compared in this increment. This option
requires replay=none. It makes no controller choice or final/probe read.

## Fixed task and data flow

| Updates | TRAIN domain and policy | Desired behavior |
|---|---|---|
| 1–4 | Invoice + password, v1 | Invoice routes; password uses inspect |
| 5–8 | Parcel + password, v2 | Parcel defers; password still uses inspect |
| 9–12 | Invoice + password return, v2 | Invoice now uses inspect; password unchanged |

Four six-row passes per phase give12updates72rows. Each row comes from an
explicit open→workflow→commit demonstration; labels arrive one tick after
observation through the original AttributionStream. Checkpoints0/4/8/12 feed
owned frozen candidate readouts into actual sandbox episodes. Source and stable
actor retain their original states; this is not a deployed publication trace.
Each ticket has an explicit policy version. Readouts use the completed phase's
version; the original delayed runner may already have registered the next phase's
unlabeled observation. Its label remains unavailable until its own arrival tick.

The existing v1/v2 sandbox is reused. Its union catalog stays fixed; v2 adds
defer and removes retire. Route remains permitted, so obsolete invoice routing
can be committed and correctly scored as failure. Capabilities, revisions and
workflow phase admission stay outside the learner; truth does not repair a wrong
permitted choice. Current-policy correctness is separate from unchanged-password
retention. A current v2 return never restores obsolete v1 targets.

`run_ticket_workflow.training_cases(policy_version=...)` generates TRAIN requests
from original sandbox transitions and refuses a teacher trace unless every effect
applies and final workflow_success holds. Its default v2 output is exactly equal
to the accepted saved teacher payload. `ticket_controller_streams` builds finite
owned packets with unique phase/version IDs. TRAIN groups/queries remain separate
from development episode groups/queries. Both use overlapping concepts/grammar;
this is not statistical task generalization or globally unseen data.

Why this: changing labels must reflect real changed task rules, rather than
silently treating marked noise as truth. The temporal core, dispatcher, learner,
stream, snapshots and task readouts already exist; this option connects them.
The original fixed-v2 stream, its replay modes and previous native evidence remain.

## Observed conventional result

One prospectively fixed perceptron case passed first launch; no seed, dose or
metric search was performed. Readouts use current policy v1 at0/4 and v2 at8/12:

| Checkpoint | Invoice | Parcel | Password |
|---|---|---|---|
| 0 | Fail | Fail | Fail |
| 4 | Pass: route | Fail: inspect | Pass: inspect |
| 8 | Fail: obsolete route | Pass: defer | Pass: inspect |
| 12 | Pass: current inspect | Fail: inspect | Pass: inspect |

Parcel is learned and then lost during return. Password is acquired and retained.
Invoice adapts to revised current truth; its failure at8 is not evidence of
forgetting an unchanged target. Two stale-revision scenarios recover; permanent
commit revocation refuses. Overall8/15 episodes succeed; all15 complete with
zero applied permission/revision violations in this fixture. Failure traces stay
in the report. No task correctness assertion forces a positive experimental sign.

Physical work:12perceptron updates72rows,58typed predictions,82dispatches including
24teacher transitions,3coefficient owners144zero slots,555snapshot-copy entries.
Zero random parameters, Torch updates, replay, probes, final or controller fitting.
Internal .4311s, metered2.2670s, sampled peak78962688B. These are instrumented
partial costs, not a comparative efficiency claim.

100selected software checks/platform, four-file Ruff/format/fresh default+Linux
mypy and saved-result readback pass. Initial unused test imports and a v2-only
scripted-fixture failure were fixed and preserved. Source/procedure/results were
self-reviewed; this is not independent acceptance. New-policy native family and
matched controller experiments remain unexecuted; #54negative utilities remain.

```console
python -m pytest -q tests/test_versioned_policy_stream.py -k "not sole_fixed"
python -m ruff check scripts/run_ticket_workflow.py scripts/ticket_controller_streams.py scripts/recurring_ticket_workflow_example.py tests/test_versioned_policy_stream.py
python -m ruff format --check scripts/run_ticket_workflow.py scripts/ticket_controller_streams.py scripts/recurring_ticket_workflow_example.py tests/test_versioned_policy_stream.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/run_ticket_workflow.py scripts/ticket_controller_streams.py scripts/recurring_ticket_workflow_example.py tests/test_versioned_policy_stream.py
```

Evidence/commands/budgets/failures: issue67 and
`artifacts/runs/versioned-policy-stream-20261023/RESULTS.md`; ADR-0583.
Extending the recipe requires truthful current-version demonstrations, explicit
finite exposure/receipt counts and separate unchanged-task outcomes. Do not reuse
obsolete labels, convert final/probe cases, or infer matched compute from passes.
