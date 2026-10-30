# Fresh owned LS reference validation

Preview the fixed comparison without preparing a source or running models:

```text
python -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --reference-validation --reference-validation-protocol --output NEW-PROTOCOL.json
```

Run the six-case comparison only under an explicit local experiment budget:

```text
python -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --reference-validation --output NEW.json
```

Outputs must be new. This separate mode excludes private probes, paired/adaptive/
incident serving, catalog modes, replay and tuning diagnostics. It uses the normal
workflow's freshly trained LS source12updates/72rows, retaining its original
guard/sampler and12-update/240-second/512-MiB budget throughout. That long lifetime
is created before training; no live budget is renewed. Existing30second source
defaults and historical Perceptron comparisons keep their behavior.

The existing comparison runner executes the two existing withheld block recipes,
each with no_learning, periodic and acknowledgement_drift references. All use
the same source, ordinary TRAIN packets and24serving opportunities. Each private
candidate is separate; workers keep serving the original actor, with no promotion.
Exact declared order, delay, control settings and all diagnostic/worker costs
remain in the report.

The collector retains at most six detached candidate snapshots as data. It
validates preceding paid results and snapshots before more work, rejects replaced
or corrupted evidence, and reports missing delegated work as unknown. Only after
all six execution cases complete does it read the source and six candidates on
the existing six known TRAIN task episodes. Those endpoint labels cannot choose
training, controls or admission of later cases. Ordinary live serving traces are
operation evidence and never select cases or parameters.

Why this: historical comparison sources and fitted policies belong to the
Perceptron family. Reusing them as LS authority would change their assumptions.
This explicit route builds a current source and reuses the existing runner and
reference controls instead of adding another orchestration system.

This three-reference foundation leaves the full learned-controller comparison
open. It is temporal-order/delay validation on known TRAIN templates, not new-task
generalization or final evaluation. It does not establish matched compute,
overall efficiency, circadian benefit or release readiness. The first bounded
[observation](reports/context-owned-reference-validation-20261029.md) found fewer
ACK fits but no retention advantage over periodic fitting; both lost old tasks
on one order. The earlier successful joint-packet mode remains a different input
recipe, not a substitute control for this comparison.
