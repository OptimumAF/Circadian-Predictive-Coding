# P6.3 replay factor: prospective development pilot

## Question and roles

Protocol `continual_mechanism_replay_dev_v1` asks whether adding a small,
shared, recent-row replay opportunity changes the two-task development
outcome at fixed model width. It uses the v14 arrived-source geometry and
role fractions, with A training arriving before B. Only the outer-selection
development roles are scored, after A and after B. Inner guard is not
accessed because this replay-only feasibility factor predeclares acceptance
of all six periodic opportunities; any core error fails the run. Final
source/labels remain sealed. Development seeds are **41, 43, 59** and
reserved independent confirmation seeds are **101, 103, 107, 109, 113,
127, 131, 137, 139, 149**. The latter are not run here.

## Fixed arms and work

At width eight, compare paired replay off/on for backpropagation, ordinary
predictive coding, and neutral circadian predictive coding. Two additional
planned-width **12** controls, backprop and ordinary PC without replay,
test a 1.5x width reference chosen before this study; they are not equal
capacity or equal FLOP controls for width eight. All shallow width-eight
models use seed `source_seed + 1001` for identical initial parameter
tensors; both width-12 controls also share that model seed with each other.
Neutral circadian disables all chemical plasticity and non-replay sleep
effects. The only circadian on/off switch is `sleep_enable_replay`.

Each arm receives 12 full-batch A and 12 full-batch B wake updates; A train
has 72 rows and B train 36. Backprop uses wake rate 0.12; PC/circadian use
rate 0.05, two inference steps, inference rate 0.2. This preserves the
historical v14 method-specific optimizer settings while keeping on/off
work matched *within* each method. A single prediction-independent FIFO
retains at most eight labeled float64 rows/192 array bytes. Every fourth
wake epoch of each phase selects the two newest retained IDs for every on
arm: six opportunities and 12 one-row replay updates per on arm/seed.
Replay uses rate 0.01; PC/circadian use two inference steps at rate 0.15.
Off arms receive zero replay optimizer updates. Both circadian arms observe
identical bounded memory and attempt the same six forced component sleep
events. Structure, homeostasis, and chemical reset are disabled; width
stays eight. No guard/rollback, adaptive schedule, or score-dependent
selection is permitted.

The planned optimizer work is **684 updates**: 3 seeds x 8 arms x 24 wake
updates = 576, plus 3 seeds x 3 on arms x 12 replay updates = 108. Cap is
**720 updates** and one public child-process wall limit is **120 seconds**.
Each width-eight model has 33 trainable parameters; width 12 has 49.
Retained IDs, selection IDs, applied IDs/counts, wake and replay work,
capacity, and explicit A/B outer-selection accuracies must be read back
for every seed. Abort on a missing boundary, ID mismatch, nonfinite score,
parity or capacity failure. Do not change rates, seeds, width, intervals,
or metric definitions in response to results. Run the exact protocol a
second time in a fresh directory and require byte-identical deterministic
result JSON. A failed or null result remains published as such.

Primary outcomes are `final_mean_task_accuracy` and `signed_forgetting_A`
from [the Phase 6 metric contract](phase6-metric-contract.md). Report every
seed, arm, and paired within-method on-minus-off contrast. Backprop versus
PC is a method comparison, not an equal optimizer-work contrast; replay
on versus off intentionally differs by 12 replay updates per seed.
The planned-width controls intentionally differ in parameter count. The
historical v9/v14 final outcomes are background only and cannot select
this pilot's factor settings.

**Why this design:** v14's periodic contrast bundled replay and pruning.
A fixed width and neutral circadian replay switch make the replay treatment
visible while shared selected IDs give backprop and PC the same memory
access. The absence of a guard is specific to this development feasibility
factor; later full-mechanism/confirmation studies must explicitly define
their guard and selection policy. This pilot alone does not close P6.3c.
The decision and alternatives are recorded in
[ADR-0142](adr/ADR-0142-isolate-fixed-width-replay-before-structural-matrix.md).

## Artifact contract

The public adapter writes an exclusive request before its bounded worker,
then a verified finite result and audit or a failure sidecar under
`artifacts/runs/p63-replay-factor-pilot/`. A second fresh directory is
required for the repeat. The request binds exact source and adapter SHA-256,
manifest digest, environment versions, declared work and wall limits.
No saved v9-v14 identity or artifact is modified. Source hashes and the
manifest digest are recorded in the session log before scored execution.
Before the two public scored runs, the frozen manifest digest is
`221465f5f0825c2a4c9bd5b01804008b3f6e51b5bb0549be4126ef0025f1cbed`,
the sorted 12-source SHA-256 map digest is
`153f5e7fd287b4cfd6940d3367cee6275fada4e5192b9c09d5e8186c2eee8cbe`,
and the adapter byte SHA-256 is
`9255ad6615b3c41953fc5f76ad4c5597389724abf57ad996f2c1945c83b8826a`.
The adapter pins each source hash and writes them into its request. This
selection covers the pilot app, v14 source factory, arrived/data splitter,
three model cores, metric helper, shared retention, and role/data sources;
it is not a complete environment or dependency-tree hash.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_replay_factor_pilot --output-dir artifacts/runs/p63-replay-factor-pilot
.\.venv\Scripts\python.exe -m scripts.run_p63_replay_factor_pilot --output-dir artifacts/runs/p63-replay-factor-pilot-repeat
```
