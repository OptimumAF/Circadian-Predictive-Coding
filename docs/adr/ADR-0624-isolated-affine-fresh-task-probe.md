# ADR-0624 — Reuse the isolated probe for the learned affine control

## Context

The conventional affine model now performs known B/C native tasks after C
learning. Its future learning and interference remain separate questions.
R2.1already supplies owned-fork probes, reserved Fcollection and readonly old
TRAIN monitors. Those generic ports support this learner; a new runner is
unnecessary. The monitor envelope permits two batches, not six episodes.

## Decision

Bind the exact after-C model to the original helper with two fixed Fpackets,
checkpoints0/1/2 and one disposable fork. Validate the collector's full original
16train/2test envelope, select the declared firsttwo packets, and preserve all
four original selected raw sources/roles. Train/test share fixed deterministic
Fgeometry/seed17; these are fit diagnostics, not independent generalization.

Use feedforward independent-row transport for F and two cue-grouped27row A/B/C
readonly monitors. Preserve original episodes and source spans, and use existing
row decision diagnostics to recover separate A/B/C accuracy/loss without extra
forwards. Retain the original two-monitor/65536entry limit. No invented long
trajectory, monitor-label replay, final access, promotion or benchmark updates.

Guard the original native source, one-fork event, budget/clock/monotonic charges,
actual private commits, source reports and arrays before later work. Keep at
most four known closed collection reports and one private endpoint, preserving
paid prefixes/failures. Declare source6/108 separately from paid upstream10/180
and new probe2/18. Original optional16update return semantics stay unchanged;
do not invoke repetitive analytic updates merely to satisfy that old recipe.

## Alternatives

Increasing monitor capacity or faking joined episodes weakens existing contracts.
Reimplementing the probe repeats accepted infrastructure. Reading only new-task
accuracy hides old-task interference. Sending probe labels into the benchmark
or serving path invalidates isolation. None is needed for this bounded slice.

## Consequences

The software binding and a separately frozen small local observation must pass
before usefulness or cost is claimed. Report every outcome, source/role
qualification and failure. No causal plasticity, nativeFcontrol, reliability,
matched compute, CPC benefit, full recovery or release claim follows from
correct isolation or one fixed fit diagnostic.
