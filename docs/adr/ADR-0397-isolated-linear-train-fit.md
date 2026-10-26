# ADR-0397: Disposable fixed linear TRAIN-fit control

## Context

Accepted online native learners still have zero exact TRAIN traces. Saved action
margins show an action tradeoff under weighted BP. These observations do not
identify whether labels can be fit by a simpler linear head, or why online
acquisition fails. Existing full state, matched controls and probes remain valid.

## Decision

Add a bounded pure one-hot least-squares core and a thin saved-TRAIN adapter/CLI.
Use all admitted rows, equal weights, float64, an appended intercept and fixed
`numpy.linalg.lstsq(rcond=1e-12)`, with no alternative solver or tuning. Reuse causal
history/canonical encoders and complete source witnesses. Validate next views but
do not learn from them. Retain every raw prediction, label, model coefficient,
rank/SSE, attempted/completed cost and available paid partial on failure.

Why this: a separate simple fit question is useful evidence without modifying
accepted online objectives, seeds or metrics. It needs no new stream framework,
dependency, model selection or evaluation access. Core stays independent of IO;
the CLI owns JSON and the adapter owns TRAIN admission/reporting.

## Alternatives

- More SGD dose, changed class weights or favorable checkpoints: would change the
  scientific question and cannot repair the previous spent experiment.
- Declare a failed least-squares fit proof of inseparability: mathematically invalid;
  least-squares reproduction and classification feasibility are different claims.
- Add another capture/audit or margin variant: does not address acquisition.

## Consequences

The reviewed sole54-row use completes with52/54 actions and4/6 exact traces,
rank25/SSE4.212268570088096. The all-label witness criterion fails; this negative
result and its SPENT grant remain permanent. Correct implementation does not
establish native learner acquisition, adaptation, generalization or advantage.

Next, compose an explicit full-catalogue native acquisition control using existing
training-only attribution and owned native learner ports, then existing disposable
F/old-A/A-return probes. Repeated catalogue exposure differs from the online
baseline and must carry separate row/memory/compute costs and objective receipts.
Freeze a small protocol before any native comparative use; no solver variant or
retry of the prior pair is justified by this result.

[Evidence](../../artifacts/runs/linear-train-fit-20261018/RESULTS.md).
