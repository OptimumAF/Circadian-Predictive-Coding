# Recorded paired game inspector

The existing standalone inspector now accepts
`paired_key_door_own_policy_quota_v1` reports. It reads saved JSON and reconstructs
recorded playback without running a learner, policy, or environment.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/paired-wall-allowance-20261020/paired-0-23.json --output output/paired-inspector.html
```

The output path must be new; the existing writer refuses replacement. Open the
standalone HTML in a browser. The saved local examples from this increment are
[complete 0/23](../artifacts/runs/paired-inspector-20261020/paired-0-23.html),
[complete 6/17](../artifacts/runs/paired-inspector-20261020/paired-6-17.html), and
[original failed attempt](../artifacts/runs/paired-inspector-20261020/paired-failed.html).

Choose the frozen or trained arm, then an episode. Physical TRAIN steps and
native optimizer updates have separate labels. All forty episode slots and all
recorded signed comparison cells remain visible. A failed or unreached episode
has no invented frames, outcome, or return. The original incomplete comparison
and missing timing remain unavailable. Frozen zero updates remain visible.

Mapped control labels show both the action index and resulting direction.
Spatial observation channels keep their original basis. Legal-conditioned
action probabilities and the original raw model probabilities are shown
separately. The **Source** links open exact embedded JSON records; trained raw
physical fields remain absent where originally absent. Stage associations use
separate pointers to original quota records.

Costs include recorded training exposures, score/live work, unknown failed work,
pair time, the original wall cap, and sampled process RSS. Inclusive timing spans
can overlap and must not be added. Replay is inactive for these paired arms.
Topology-change history, lineage and sleep proposals remain unavailable.
No useful learning, generalization, or circadian advantage is established.

## Checks and limits

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_paired_quota_inspector.py tests/test_key_door_inspector_projection.py tests/test_key_door_inspector_files.py
.venv/Scripts/python.exe -m ruff check src/app/key_door_inspector.py src/app/paired_quota_inspector.py tests/test_paired_quota_inspector.py
.venv/Scripts/python.exe -m ruff format --check src/app/key_door_inspector.py src/app/paired_quota_inspector.py tests/test_paired_quota_inspector.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent src/app/key_door_inspector.py src/app/paired_quota_inspector.py tests/test_paired_quota_inspector.py
```

The original150 focused cases pass on Windows and pinned Linux; the recorded-state
extension passes187 combined cases per platform (37new+150prior). These include
saved local artifact fixtures; clean-clone readiness is not established.
Independent source and actual presentation review accepted functional behavior.
The browser lifetime exceeded the declared 240-second allowance; resource
conformity is withheld, with the failure preserved. The last role-note wording
change was reviewed separately; saved pages retain the browser-tested wording.
[Commands, evidence, failures and remaining work](../artifacts/runs/paired-inspector-20261020/RESULTS.md).

## Selected recorded native checkpoint

The selected episode determines its physical-stage ordinal. The view shows the
original observation at that ordinal, then checks its native index against the
quota. This distinction matters because all five frozen stages have native0.
Architecture, native/wake/training counts, actual five saved array shapes and
parameter/optimizer/gradient/activation payload counters resolve to original
source links. The recorded memory-scope exclusions remain visible; payload bytes
are not process RSS. Nothing reconstructs a model or optimizer from JSON.

Missing observations or fields stay unavailable. A raw observation without an
authoritative quota stays in embedded source but is not presented as an admitted
checkpoint. Changing selection clears previous state and source links. A failed
live readout may still have a genuinely recorded native checkpoint; its failed
live outcome remains unchanged.

[Complete example](../artifacts/runs/paired-state-view-20261020/complete.html) ·
[Original failed example](../artifacts/runs/paired-state-view-20261020/failed.html).
The null-state page in that stage is a labeled presentation-only fixture, not
another study or scientific sample.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_paired_quota_inspector_state.py tests/test_paired_quota_inspector.py tests/test_key_door_inspector_projection.py tests/test_key_door_inspector_files.py
.venv/Scripts/python.exe -m ruff check src/app/paired_quota_inspector.py tests/test_paired_quota_inspector_state.py
.venv/Scripts/python.exe -m ruff format --check src/app/paired_quota_inspector.py tests/test_paired_quota_inspector_state.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent src/app/paired_quota_inspector.py tests/test_paired_quota_inspector_state.py
```

Final187 cases/platform and statics passed with independent source/actual review.
The first launcher failed before functional assertions. A separately reviewed,
prospectively bounded output-collection repair passed; browser lifetime12.54s
under60s, all cleanup checks true. Both attempts and the prior unrelated240s
lifetime failure remain preserved. [Exact evidence](../artifacts/runs/paired-state-view-20261020/RESULTS.md).

## Extending this view

Keep source binding in the pure app projection and render exact source pointers.
Missing topology-change/lineage/sleep records need actual producer evidence before
they can be displayed. Do not infer these from parameter arrays or run a new
cohort solely to fill a presentation field. FullR4.7 remains open.
