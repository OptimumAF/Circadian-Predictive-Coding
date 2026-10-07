# ADR-0064: Separate CUDA reproducibility from matched timing confirmation

## Context

The verified repository environment used CPU-only Torch despite an RTX 3080
being present. P1.7 had CPU model-order, replay, and stochastic-loader
evidence but no actual CUDA numerical check. P1.8 had a frozen pretrained
matched-head CPU confirmation; CUDA wall-time and allocator behavior had not
been observed. Desktop applications can share this GPU during experiments.

## Decision

Keep the CPU `.venv` unchanged and install pinned matching Torch 2.14.0+cu130
and torchvision 0.29.0+cu130 in a separate gitignored `.venv-cuda`. First
run a synthetic-only device smoke. Then use the seeded v3 unmatched vision
route on one small, verified local-CIFAR split to reverse model execution
order with deterministic CUDA algorithms and a fixed CUBLAS workspace. Require
exact role/trained-state hashes and a predeclared `1e-6` absolute metric
tolerance, with final test sealed until all models finish.

Only after that gate, freeze a distinct pretrained shared-feature CUDA
selection request and manifest. Preserve the CPU study's role sizes and
equal candidate grid, but use independent selection/confirmation seeds. A
later fixed-data/wall-time/process-memory confirmation may start only after
three five-second-apart GPU readings each show at most 10% utilization and
at least 5 GiB free. A busy window saves a deferred artifact and reads no
final-test batches. The selected manifest, metrics, and budgets stay fixed.

Why this: a working CUDA wheel is necessary but does not prove seeded model
behavior. A shared GPU can distort equal-wall-time observations even when
each head receives the same nominal deadline. Recording a launch gate before
test access makes a busy window an explicit deferral rather than a reason to
change seeds or select a favorable run.

## Alternatives

- Replace the CPU environment with CUDA packages. Rejected because it would
  disturb an already verified baseline.
- Infer CUDA reproducibility from CPU tests or a tensor-only smoke. Rejected
  because device kernels and stream behavior differ.
- Run the wall-time confirmation during observed 24–37% background GPU use.
  Rejected because it would confound the local timing scope.
- Change selected candidates or confirmation seeds while waiting for an idle
  window. Rejected because the validation manifest is already frozen.

## Consequences

The isolated CUDA smoke passed after a telemetry-only retry, and the original
failure remains saved. The actual-CIFAR CUDA reversal produced identical
role and trained-model hashes and zero differences in all declared metrics;
both orders kept final test sealed through training and attempted forced
circadian sleep once. The CUDA matched-head selection completed six equal
validation trials with no final-test iteration and saved a digest-checked
manifest. Its first confirmation launch was deferred by three busy-GPU
readings, so no CUDA matched final-test result is claimed. The RTX 3080,
32-pixel inputs, tiny reversal, and later process-wide memory observations
limit any broader inference.
