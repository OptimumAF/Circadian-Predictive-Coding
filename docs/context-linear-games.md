# Frozen affine policy in the actual game

The existing saved-source CLI can execute the qualified two-head affine model
without refitting:

```powershell
python -m scripts.run_minigrid_linear_fit_control --context-linear-game --describe
```

A fresh run requires a declared local protocol and the retained qualified fit:

```powershell
python -m scripts.run_minigrid_linear_fit_control --context-linear-game --source artifacts/runs/context-linear-train-fit-20261024/actual.json --output fresh-affine-games.json
```

This research command accepts the exact qualified fit report, bounded to 512 KiB.
It is not a bundled model or a clean-install/release qualification. Describe
accesses no source, model or game. Conflicting modes, fitting overrides and
existing output files are refused.

## Executed known-task result

One fixed native pair completed on the first launch:

| Known TRAIN task | Success | Steps | Return |
|---|---|---:|---:|
| A, blue cue |Yes|9|0.49375|
| A, red cue |Yes|9|0.49375|

The actual MiniGrid environment terminated both episodes, without truncation.
The policy used all seven available native actions and first argmax; no action
mask or teacher supplied a decision. It retained the first public frame when
the cue disappeared, as declared by the existing representation. These are
learned affine predictions, not input-key lookup or scripted teacher actions.

The source was the preceding offline joint fit on all known TRAIN traces.
This pair establishes known-A gameplay for that model. It does not establish
chronological acquisition, old-task retention, recovery, unseen-layout
generalization, recurrent RL, statistical reliability or circadian advantage.
All earlier unsuccessful controls remain unchanged.

## Ownership and budgets

The new readonly adapter owns complete fitted-model snapshots and independent
forks. Immutable coefficient tuples may share leaves; model wrappers and binding
records are detached and revalidated. It rejects a different restore binding,
input mutation and corrupted snapshots. It exposes no update method or invented
perceptron/optimizer clocks. Initial public cue chooses the head; current cue,
metadata and labels cannot choose it.

A narrow wrapper calls the original frozen evaluator loop with known A only,
blue then red, seed 17 and maximum 16 steps per episode. A fresh readout budget
permits zero training/replay updates, 90 seconds and 512 MiB; its original
configuration, clock and owner bindings remain checked. The source's two prior
fits / 54 rows are separate from the new zero-update readout. Source bytes remain
bound throughout. Failed later episodes retain preceding results and unknowns.

Why this: the original two-head affine fit already classifies all 54 TRAIN
actions. An ordinary readonly port makes those learned weights executable in the
existing game loop without altering their learning history or adding a runner.

## Evidence and checks

The native case used 18 predictions and 18 physics steps, two environment resets,
zero fits/updates/teacher/probe/final/replay work. Readout elapsed was 35.9349295 s;
parent elapsed was 37.8132167 s; sampled peak RSS was 71,888,896 B. All frozen limits passed. Source
bytes, fitted snapshots and caller NumPy RNG remained unchanged. Copy/validation
and repeated source-read costs are retained; no efficiency comparison is made.

34 distinct focused cases qualify per platform. Windows had 33 passes/one malformed
exception fixture, then its single repair passed; both receipts remain. Linux
passed 34. All54 saved TRAIN probabilities reproduced within fixed1e-12 absolute
tolerance on both platforms. Source/procedure/result self-review and scoped
Ruff/format/mypy passed. Self-review is not independent acceptance.

Two tests require local accepted fit/TRAIN artifacts. They executed and passed
in this qualification; a checkout without those artifacts reports explicit skips.
That checkout has not reproduced the parity/source-integration qualification.
Other tests use literal models and scripted environments, not learned gameplay.

The spent local case is under `artifacts/runs/context-linear-games-20261024/`:
PROTOCOL.md, actual.json, RESULTS.md, all receipts and the original profiler/
fixture failure history. Verify saved transitions and counts without models or
physics:

```powershell
python -B artifacts/runs/context-linear-games-20261024/verify_result.py
```

[Issue106](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/106)
contains scoped acceptance and handoff. Broader R4.2, adaptation, scientific and
release requirements remain open. The next learner task is causal TRAIN updating
through the existing stream; another frozen readout cannot establish adaptation.
