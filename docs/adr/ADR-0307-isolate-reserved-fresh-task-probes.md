# ADR-0307: Isolate reserved fresh-task learning from the benchmark

## Context

The accepted matched A/B/A stream measures new-task outcomes, retention and
context return. Fixed loss/gradient diagnostics do not test whether a checkpoint
can still learn a reserved task. Final-test feedback must remain unavailable to
training and probe selection. R2.1 also requests feature-health observations.

## Decision

Use the existing stream checkpoint observer, which receives copied state and
only arrived training metadata. At each fixed checkpoint, run feature diagnostics
on a separate label-free reference and restore a disposable fork of an independent
probe template. Use separately declared probe training/evaluation data and budget;
complete every fixed observer before the existing final benchmark scorer runs.
Retain all learning curves, failures and separate costs. Do not alter the native
learner, stream runner or original matched baselines.

## Alternatives

Training the deployed/benchmark model on the probe would contaminate retention.
Choosing probes from final scores would contaminate evaluation. Adding another
stream orchestrator duplicates an already tested isolation boundary.

## Consequences

The new probe answers a bounded exposure question, not causal or universal
plasticity. Diagnostic rank definitions and numerical cutoffs are explicit.
Native importance and neuron ages are unavailable in coupled models and remain
unfinished R2.1 requirements. Memory/time/work observations include diagnostics
and probe costs; equal updates are not claimed to be equal native compute.

## Native history amendment

Legacy CPC already stores importance EMA, float update ages, and stable unit and
parent IDs. Expose these through a bounded read-only adapter and pure immutable
summaries, with an opt-in initial/final observation in the existing stream demo.
Why this: observe actual histories rather than manufacture ages or importance
from unrelated chemistry/global counters. Current adaptive width handles native
splits/prunes; pruned parents remain valid lineage. No native methods, model
snapshot, RNG mutation, new training rule or stream runner is required. Reads
require a quiescent model and do not certify full-model validity or provenance.
The legacy demo is an engineering check, separate from the frozen matched pilot;
coupled fields stay unavailable and original negative results remain unchanged.

## Fixed exposure follow-up

The original four-update experiment improved C at intermediate points but had
fifteen negative fixed endpoints. Add a separately named sixteen-update scenario
with a prospectively fixed full curve and the same first four batches, settings,
evaluation data and seed. Why this: the short exposure leaves later learning
unobserved; a new bounded exploratory protocol can measure it without selecting
an earlier best point or changing baseline settings. Preserve the original
protocol/source snapshot/results. Reuse existing stream/probe isolation and
failure accounting, with 328 total updates and separately reported probe costs.
This is not independent confirmation, a controller or a circadian-benefit claim.
