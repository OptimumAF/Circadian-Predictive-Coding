# Fresh same-context joint TRAIN serving

Use the existing workflow entry point with the explicit LS option:

```text
python -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --paired-serving same_context_joint --output NEW.json
```

The output must be new. Despite sharing the existing `--paired-serving` entry
point, this option uses the original legacy task family: invoice/password-v1,
invoice/parcel-v2 and legacy-v1/v2. It requires LS and excludes private probes,
incident serving and catalog modes. Existing current/joint options and
their four-packet learning window keep their behavior.

This mode reuses `context_joint9_return`: two ordinary TRAIN packets each contain
the three new legacy-v1 rows and six old invoice/password-v1 rows. Both tasks
therefore participate in the final learning fit. These are nine paid row
exposures per fit, including six repeated old rows. The original return packet
still contains only six old rows; returning may forget the new task.

Why this: a replay fit followed by a narrow current-only LS fit can overwrite
old knowledge. The existing joint recipe addresses that final solver input
without changing the catalog's frozen replay-before-wake contract.

The normal source learns12updates/72rows and keeps its original30second budget,
guard and sampler throughout the child run. The existing followup plan publishes
after two learning packets and admits the return at tick12, using one original
child budget. Guards can reject publication. Reports distinguish the candidate,
published actor, continued serving and exact rollback to the source. Neither
private probes nor final evaluation supply labels or choose the protocol.

Add `--adaptive-serving` to use the existing arrived-error/current-candidate
acknowledgement control instead of periodic fits:

```text
python -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --paired-serving same_context_joint --adaptive-serving --output NEW.json
```

The two declared windows remain fixed. Each correct arrived packet is verified
against the current private candidate under its original lease and acknowledged
without charging an optimizer update. Exact original observation/arrival clocks
and the complete verified actor boundary stay binding after publication; the
candidate's source version is not rebased. One bounded candidate pin accepts
changes only after observed successful poll receipts. Ordinary serving can defer
publication. Diagnostics, acknowledgements, fits and all worker costs are
reported separately. A final frozen private TRAIN task readout occurs after the
stream, without selecting training or publication.

The [bounded acknowledgement observation](reports/context-same-context-ack-20261029.md)
used one joint fit and two acknowledgements. Both the final candidate and served
actor passed all six known tasks; rollback restored the original legacy failure.
The report preserves the pre-strengthening run, review findings, explicit
engineering allowance amendment and both runs' costs.

This is paid model work on known TRAIN templates. It does not establish held-out
generalization, matched compute, circadian advantage or release readiness. See
the [execution report](reports/context-linear-same-context-20261029.md) for the
bounded observation, costs and remaining limitations.
