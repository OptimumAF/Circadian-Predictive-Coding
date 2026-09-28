# ADR-0039: Expose a separate full circadian classifier snapshot

## Context

The ResNet circadian classifier's `snapshot_state()` delegates to its
head. That is the guard snapshot taken around each candidate sleep event.
The backbone has parameters, buffers such as batch-normalization running
statistics, per-submodule training modes, and parameter gradient flags.
The head-only guard is sufficient for current sleep mutations but cannot
represent the whole classifier for continuation.

The circadian vision route calls `CircadianPredictiveCodingResNet50Classifier`
`train_step()`, which extracts features and updates the head directly. It
does not create or step an optimizer, scheduler, or mixed-precision scaler.
The fixed-feature matched-head route likewise updates this circadian head
directly. Backprop comparators have an SGD optimizer in their separate
routes; it is not owned by the circadian classifier.

## Decision

Keep `snapshot_state()` and `restore_state()` as the head-only sleep-guard
API. Add `snapshot_full_state()` and `restore_full_state()` for explicit
in-memory whole-classifier continuation. The versioned
`CircadianClassifierSnapshot` copies the backbone `state_dict`, module
type/name structure, each module's training mode, each parameter's
`requires_grad` flag, and the full head snapshot including its local
split generator. Restore validates classifier/device identity, backbone
keys, tensor shapes/dtypes/devices, modes, and flags, and stages the head
before changing live state. It copies snapshot values again on restore.

Why a separate API: copying ResNet-50 at every sleep guard would change
the guard's time and memory budget. Sleep currently changes only head
state. A caller requesting a full classifier snapshot can pay the
backbone copy cost explicitly.

## Alternatives and consequences

Expanding the existing guard API would silently alter benchmark cost.
Saving only `backbone.state_dict()` would omit train/eval mode and
gradient flags; that could change subsequent batch-normalization
behavior. This snapshot is still in memory. It does not capture external
loader/sampler state, process-global Torch/NumPy/Python RNG, or app-level
epoch/report counters, and it has no optimizer/scheduler/scaler state
because the circadian route owns none. P3.9 defines file-backed resume
and those external boundaries. P3.7 adds atomic acceptance/rollback
across all sleep effects and append-only attempt telemetry.

## Evidence

`tests/test_torch_classifier_full_snapshot.py` first failed on the
missing API. A synthetic Linear/BatchNorm backbone now verifies copied
parameters, changing buffers, mixed module modes, gradient flags,
snapshot detachment in both directions, next fixed-batch update and
noisy split after restore, and rejection of incompatible classifier or
head state before mutation. The development log records the full gate.
