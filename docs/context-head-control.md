# Conventional public-context separate-head control

The recurring runner can keep independent learned heads for two known public
policy contexts:

```powershell
python -m scripts.recurring_ticket_workflow_example --context-return --separate-context-heads --output context-heads.json
```

Use a fresh output path. This option requires context-return, perceptron and no
replay; incompatible combinations fail before work/output reservation. Old shared,
D7default, policy-change, replay and native modes retain their existing behavior.

## Delivered behavior

The D9public query has policy_v1 or policy_v2. Its exact one-hot marker selects
one existing D7/K6perceptron, then the context columns are stripped. The original
supplied-order unit mistake update and permitted-candidate affine readout remain.
Each homogeneous TRAIN pass changes only its selected head, including its bias.
Missing/ambiguous/unknown contexts and mixed-context batches are refused.
Readout tokens select learned coefficients, never a teacher action or permission.
Valid syntax does not independently attest to an external context's truth.

Complete immutable snapshots contain both head policies/matrices/counters and
global counters equal to their sums. Snapshot/fork/restore revalidate and detach
state; valid whole updates commit atomically. Original stream, actor, budgets,
clocks, delayed receipts and paid-evidence guards remain in charge. Contexts are
predeclared; this does not infer or create a head for an unseen context.

New modules and responsibilities:

```text
src/core/context_perceptron.py       pure policy/state/routing, original head rules
src/adapters/context_perceptron.py   learner port and frozen typed public-context readout
tests/test_context_perceptron.py     ownership, refusal, rule/readout parity
tests/test_context_head_workflow.py  original stream, task protocol, costs and failure stops
```

Standalone configuration uses explicit column indices from the same fixed spec:

```python
policy = ContextPerceptronPolicy(
    len(spec.vocabulary), len(spec.catalog),
    (spec.vocabulary.index("policy_v1"), spec.vocabulary.index("policy_v2")),
)
learner = ContextPerceptronLearner(policy)
# Features must share this spec/order and contain one context throughout a batch.
learner.train_batch(features, targets)
predict = ContextPerceptronDecisionPredictor(spec, learner.snapshot_state())
```

Core depends on the original pure perceptron. Adapter depends on core and the
original typed affine adapter; existing runner composes it with AttributionStream.
No new runtime, optimizer, context detector, dependency, IO or scheduling framework.

## Sole frozen case: acquisition with retention

Issue77/R4.1-H/R2.1-W, fresh zeros, same75data/order/truth,12updates72TRAINrow
exposures,delay1. Fixed0/4/8/12 readouts and endpoint stale/revocation:

| Updates / rows | v1 invoice | v1 password | v2 invoice | v2 parcel |
|---|---|---|---|---|
|0 /0|fail|fail|fail|fail|
|4 /24 after A|pass|pass|fail|fail|
|8 /48 after B|pass|pass|pass|pass|
|12 /72 after A returns|pass|pass|pass|pass|

Both B tasks failed before B, then succeeded. The entire A head remains unchanged
during B; the entire B head remains unchanged during A return. No acquired A task
was lost, so recovery is null, not a demonstrated forgotten-skill recovery. Four
return passes add no mistakes. Endpoint4/4ordinary and4/4stale success;4/4revoked
commits correctly denied, which is not task success. Across all24episodes there
are10ordinary+4stale successes; initial/untrained B failures remain in the report.

This is useful acquisition/retention on these known contexts. Saved shared75and
recent/coverage76controls end at3/4; their negative results remain unchanged.
The new control has96float64coefficients versus60 (+60%), separate biases and
different input representation/routing. Model bytes768versus480, four full
checkpoint matrices3072versus1920bytes; explicit known context required. No matched-
capacity, causal attribution solely to routing, total-compute, CPC, unseen-task,
statistical or release claim follows. Costs and broader applicability stay open.

## Measured costs and checks

12headpasses72rows16mistakes: A8,B8,return0. Three composite-learner constructors
and six zero-head initializations;0original PerceptronLearner objects (original
pure head functions are reused).1215context snapshot validations,2650perceptron
snapshot copy entries127200logical coefficient slots,110routing entries,
98contexttyped+98delegated typed calls (98physical task decisions),2teachers13
packet constructions292encodes122sandboxdispatches. No full-class predict-port,
Torch/SGD/PC, replay/probe/final updates, random draws, promotion or external action.

Nominal TRAIN3456affine terms and task3920terms;256coefficient additions,768model
coefficient bytes and3072four-checkpoint bytes. Double query projection, routing,
copies/validation/Python overhead additionally counted. Recipe10.79960s,
meter11.39769s, sampled RSS45,109,248B, boundary44,826,624B. Sampling is not a hard
OS peak guarantee. Raw timings from separate runs do not establish efficiency.

110selected checks pass Windows/WSL,0skips; five-file Ruff/format/fresh default+
Linux mypy pass. Source/procedure/result review is self-review, not independent
acceptance. Fullsuite/nativefamilies/math/physics/packaging/cleanclone/CI unrun.

Frozen protocol, raw state/effects, ledgers, zero-model readback and exact commands:
`artifacts/runs/context-head-control-20261023/RESULTS.md`. Live handoff:
[issue77](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/77).

## Subsequent diagnostic

Issue78implemented the [isolated further-learning probe](context-head-probe.md)
through existing components. Warm learns v1legacy, temporarily loses password,
then restores it without losing legacy endpoints. Fresh loses v1legacy on return;
v2legacy was already correct for warm. These private probe weights do not update
this qualified source or deployed/benchmark learners; all known-data/cost limits
and the original77result remain unchanged.
