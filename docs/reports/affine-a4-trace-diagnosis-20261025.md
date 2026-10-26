# R4.2-A4D.D — observable A4 failure, 2026-10-25

[#169](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/169).
Single-agent source/procedure/result self-review, no independent acceptance.
Analysis of both original168 DEVELOPMENT traces; no new game or learner work.

Blue enters an exact public two-state/action cycle at step6. It chooses left
facing south, then right facing east, returning to the first public frame.
Steps8–16 repeat the pre-states/actions from6/7. All29 original decisions match
firstargmax and have a unique maximum; exact tie-breaking does not explain this
failure. Blue's repeated left margin is0.006897168851388552 and right margin is
0.26322106169979087. The small left margin is descriptive, not a tuned threshold.

The visible blue key moves from the front cell at step3 to the agent cell in
step4, consistent with the recorded pickup. Step4's left turn and step5's forward
move precede the eventual loop. No blue door-toggle action occurs. These public
observations do not establish hidden inventory/position or prove which feature,
training omission or coefficient caused the choices. There is no expert/optimal
action comparison, causal intervention or post-hoc label.

Red has no repeated pre-state/action pair. Its sequence is forward, right,
pickup, right, forward, left, left, forward, forward, forward, toggle, forward,
forward. The public door changes from locked to open after toggle; the original
game terminates with goal success at13. Every step changes the public frame in
both cues. A per-step unchanged-frame detector alone would miss blue's cycle.
Original blue16/truncated/return0 and red13/terminated/return0.26875 stay intact.

## Execution and limits

Frozen local `artifacts/runs/affine-a4-trace-diagnosis-20261025/PROTOCOL.md` and
issue169 preceded data access. Inspected producer and installed constants as
source text. One incorrect three-filename search returned rg2; resolved with
`rg --files`, no data/model work. Producer inspection showed duplicated prediction
and transition pre-frames plus next-frame; original4736 public scalar-slot cap
was prospectively corrected to14208 in comment6404749009 before any JSON access.
No additional model/source/step allowance or retrospective pass.

Commands from repository root:

```powershell
.venv/Scripts/python.exe -B -S -m py_compile artifacts/runs/affine-a4-trace-diagnosis-20261025/analyze.py
.venv/Scripts/python.exe -B -S artifacts/runs/affine-a4-trace-diagnosis-20261025/analyze.py > artifacts/runs/affine-a4-trace-diagnosis-20261025/terminal.txt
```

Both EXIT0. Sole analysis completed0.01532140001654625s; command wall0.591948s,
peak working set23105536B, below15s/30s/256MiB caps. Exactly one NEW168 actual
payload read532179B;203 probability scalars/12876 public scalar slots/29 original
steps examined. Source JSON is parsed but old provenance is opaque; no saved
parameter extraction/decoding, canonical projection or learner reconstruction.
Actual saved-file bytes differ from168's counted logical serialization bytes;
these are separate observations, not additional model work.

No project/NumPy/Torch/MiniGrid/Gym imports, fitting/forward/optimizer/teacher/
simulator/TRAIN/replay/random work. Exclusive result receipt created before read;
result.json stores both per-step pre/post public summaries, margins and repeated
pairs; terminal.txt retains emitted derived evidence. No extra raw-reader or
rerun. Source/procedure/result self-review checked decoder scales/coordinates
against producer/installed constants, continuity, action/probability alignment,
positive and complete negative outcomes, finite caps and unknowns. No product
code changed in169. Skipped model tests, full suite, Linux/native reruns,
independent/statistical/causal/release qualification: no relevant model change.

The168 global NumPy RNG witness remains unobserved; its acceptance item2 remains
unchecked and issue168 OPEN. This analysis neither supplies nor waives it.
167's timeout and all earlier negative/spent evidence remain. No benefit/CPC/
generalization/recovery or repository-wide readiness claim. Historical plans/log
stay frozen under AGENTS; full goal and broad parents remain OPEN/ACTIVE.

## Precise next implementation action

Extend existing `scripts/minigrid_affine_workflow.py` TRAIN collection seam
`_collect_recurring_train_sources` with an explicit default-preserving layout
input and fresh A4 TRAIN episode identities. Add literal role/cleanup/failure
checks before any native collection. Reuse the existing teacher, inbox, learner
and stream rather than building another runner. Preserve original G/default
behavior and its nine-row contract; determine A4's own row contract from its
declared TRAIN procedure rather than assuming nine.

Why: useful next evidence needs learning on fresh changed-geometry data and
old-task retention, rather than extending a failed readout. Do not reuse or
relabel168 probe_test records as TRAIN. Any subsequent learning comparison needs
a separately frozen small protocol for from-birth TRAIN collection, source
authority, new/old/return outcomes and total/replay/computation costs. It must not
renew168/167, reread127 under spent authority, tune a baseline to force CPC benefit
or claim recovery unless an old task was actually lost and then regained.
