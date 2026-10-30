# Adapt LS serving from arrived TRAIN errors

The existing owned LS workflow can use its existing supervised error/ACK control:

```powershell
$env:TEMP='D:\Dev\_Temp'; $env:TMP='D:\Dev\_Temp'
python -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --paired-serving joint --adaptive-serving --output artifacts/my-new-adaptive-result.json
```

Run from the physical D: checkout with a new output path. This performs training
and serving; prospectively budget future executions. The accepted captures in
the report need no repetition. Omit `--adaptive-serving` for the unchanged fixed
periodic LS path; the Perceptron default remains unchanged.

This option requires `linear_fit`, context return, separate heads and explicit
paired serving. Replay/private/catalog/task-feedback/arrived-shift mixtures are
unsupported. The original owned TRAIN source12/72, schema, original clock,
budget/sampler, paid receipts and source/actor checks remain mandatory.

Only the FIFO-front, already-arrived ordinary TRAIN packet is diagnosed on a
detached current candidate. The unchanged fixed drift reference requests one
fit when its mistake fraction exceeds.1. A packet with no mistakes is consumed
through the existing verified `acknowledge_ready` receipt, without a fit. Labels
that have not arrived, private probes, final evaluation and task readouts cannot
trigger work. Diagnostics themselves cost forwards, copies, memory and time;
an unapplied packet may be diagnosed again after a deferred admission.

ACK counts are separate from actual updates. `receipt_progress` records fits,
acknowledgements, consumed and total. In adaptive mode `training_complete`
means all original TRAIN packets were consumed; `training_complete_scope`
explains that ACK is not a fit. The unchanged six-wake ceiling remains a ceiling;
unused credit is not renewed or converted into successful updates. Publication
waits for all original labels, drained fit/ACK receipts and the unchanged inner
promotion guard. Post-training workflow outcomes never select training or guards.

The [executed case](reports/context-linear-adaptive-serving-20261029.md) on both
platforms uses two child9row fits, four correct ACKs and seven diagnostic reads
covering57rows. The published candidate completes all six known workflows and
serves78worker requests, then exact rollback restores the source. These are
known TRAIN templates with public policy cues and explicit old-TRAIN reuse.
The `current` recipe is admitted but has not received a full adaptive CLI case.

This is a delayed supervised-error baseline. It does not establish that the
existing uncertainty/feature/residual ShiftDetector is calibrated or deployable.
Prior shift misses, scientific negative results and the spent287comparison
remain intact. No autonomous unlabeled discovery, CPC benefit, generalization,
matched-compute advantage or release readiness is claimed.
