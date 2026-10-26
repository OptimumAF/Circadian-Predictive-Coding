# Public cue-relative key encoding

This optional representation keeps the original public cue and encodes whether
each visible key matches it. It is available as a pure codec and as a separately
typed binding of the existing BP, PC, neutral CPC and active CPC learners.
The original encodings, native objectives and defaults remain unchanged.

## Representation and boundary

The input is an ordinary normalized public initial/current history of296
coordinates. Both full frames and directions undergo the existing canonical
validation. Exactly one initial Ball(object6), red(color0) or blue(color2), must
be visible. Its original color remains in the output. Every Key(object5),
including a held key, must be red or blue in either frame. Its color becomes2
when matching the initial cue and0 otherwise. Current-frame balls never choose
or replace the initial cue. Spatial positions, object IDs, states, neutral door
color and other non-key colors remain intact, with the original wall-color and
direction normalization.

Why retain the initial color: the key permutation is reversible relative to the
old canonical representation. On a red-cue history, applying it twice undoes the
key permutation. **Apply exactly once to raw histories.** The296-coordinate
shape cannot identify whether data was already projected.

Choose one boundary:

1. Feed raw296 histories from the ordinary `PublicInitialFrameHistory` to
   `CueRelativeTorchFeatureLearner`. The wrapper projects internally for both
   training and prediction.
2. Use `cue_relative_public_history`, `CueRelativeInitialFrameHistory`, or
   `encode_cue_relative_history_episode` to project explicitly, then feed that
   output directly to an unwrapped `TorchFeatureLearner`.

Never feed an explicit codec's output to the bound learner. A caller that
constructs native weights manually remains responsible for their provenance;
an encoding tag cannot authenticate arbitrary weights.

```python
from src.adapters.cue_relative_torch_feature_learner import (
    CueRelativeTorchFeatureLearner,
    CueRelativeTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeaturePolicy

learner = CueRelativeTorchFeatureLearner(
    CueRelativeTorchFeaturePolicy(
        TorchFeaturePolicy(method="bp", feature_dim=296, hidden_dim=8, seed=17, num_classes=7)
    )
)
# learner.train_batch(raw_public_histories, actions)
# learner.predict(raw_public_histories)
```

Use the existing native policy's explicit configuration for a matched comparison.
The new encoding identity is
`minigrid_public_cue_relative_key_color_history_296_v1`. Its policy and snapshot
types are distinct from the original invariant family. Complete native state
(optimizer, gradients, chemistry and RNG) is copied and restored through the
existing native implementation. Invalid later rows are rejected before any
native batch work. A failed stateful reset clears the preceding episode's cue.

The pure episode helper accepts complete original148-frame episodes, copies
and validates original continuity, and independently transforms current and next
views while preserving metadata and roles. It grants no authority to train on
an episode: the existing orchestration boundary still controls role admission.

## Structure and checks

```text
src/core/minigrid_cue_relative_encoding.py       pure codecs
src/adapters/cue_relative_torch_feature_learner.py native binding and owned state
tests/test_minigrid_cue_relative_encoding.py      public-frame behavior
tests/test_cue_relative_torch_feature_learner.py  native parity and stream behavior
```

The dependency direction is adapter → core and adapter → existing native adapter;
the core codec contains no filesystem, environment, teacher, model or RNG access.
No dependencies or environment settings were added. Existing contribution and
release policies apply. [ADR0425](adr/ADR-0425-cue-relative-key-encoding.md)
records why this uses a separate state family.

Focused verification, using the repository virtual environment:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_minigrid_cue_relative_encoding.py tests/test_cue_relative_torch_feature_learner.py tests/test_minigrid_invariant_encoding.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/minigrid_cue_relative_encoding.py src/adapters/cue_relative_torch_feature_learner.py tests/test_minigrid_cue_relative_encoding.py tests/test_cue_relative_torch_feature_learner.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/minigrid_cue_relative_encoding.py src/adapters/cue_relative_torch_feature_learner.py tests/test_minigrid_cue_relative_encoding.py tests/test_cue_relative_torch_feature_learner.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/minigrid_cue_relative_encoding.py src/adapters/cue_relative_torch_feature_learner.py tests/test_minigrid_cue_relative_encoding.py tests/test_cue_relative_torch_feature_learner.py
```

Repeat the last command with `--platform linux` for the second platform check.
The recorded checks used the existing bounded serial supervisor; exact argv,
resources and outputs are in
[RESULTS](../artifacts/runs/cue-relative-encoding-20261018/RESULTS.md).

## Evidence and remaining work

110distinct scoped cases passed (81new and29original codec cases), including
all four methods against direct projected native references, full owned state,
PC-neutral identity, malformed-input refusal, cross-family refusal, and an
actual two-update delayed training-only `AttributionStream` run. The latter
does not alter its source learner or actor. These are unit/integration checks,
with24native updates/48rows and18predictions/36rows across the combined run.

The sole saved-source representation preflight passed on the original six
TRAIN episodes:54rows,27per cue,27changed,54unique projected inputs and zero
conflicts. Explicit cell-level mapping, inverse mapping and source/RNG
preservation passed without constructing a model. Parent wall0.9856741seconds,
actual0.7080489seconds and sampled RSS45088768bytes include admission/copies/
Python work beyond the384048bytes of reported arrays. This does not demonstrate
learnability or convergence.

The subsequent [constructive recipe](constructive-train-fit.md) now exposes
`--constructive-bp --cue-relative` through the existing TRAIN catalogue/stream,
with projection-aware bounds, counters and probability readouts. Its one frozen
128update run reaches48/54rows and0/6exact traces; full fit remains unmet. The
codec-only increment above used no scientific native training. The subsequent
[public probe integration](cue-relative-probe.md) now registers the family for
joint training and isolated F/old-A/A-return, with scoped native engineering
checks. Scientific adaptation remains open. Existing gradient and
activation readers still bind the old representation and must explicitly
support the new projection before reading the new snapshot family. The earlier
original52/54negative, unestablished old-A retention/recovery and lack of evidence
for a circadian advantage remain unchanged.
