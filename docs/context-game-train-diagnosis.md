# Contextual game TRAIN alignment and conflicts

The existing saved-source diagnostic CLI can now inspect the two public-cue
heads without constructing a learner or running a game:

```powershell
python -m scripts.run_minigrid_linear_fit_control --context-bounds --describe
```

After declaring a local diagnosis budget, supply the original TRAIN source:

```powershell
python -m scripts.run_minigrid_linear_fit_control --context-bounds --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --output fresh-context-bounds.json
```

The flag is mutually exclusive with existing fit modes. Omit it to retain the
original CLI behavior. Output is exclusive; an existing file is refused.

## What it checks

The adapter reuses the existing six-source TRAIN admission and raw/canonical
input bounds. It checks that each episode-array row contains the public initial
frame and its own current observation, paired with the original current action.
A deliberate next-observation shift is rejected. Next views are validated as
evidence and never used as current inputs.

The initial visible ball cue chooses the head, independently of metadata names
or targets. Each head receives the canonical296 coordinates after the two routing
bits are removed, matching the existing native routing. Reports retain every
source episode/step/action, visible cue, name agreement and exact-key conflicts.
No dimension expansion, learner, fitting callback, final/probe data or evaluator
is added. The output is an empirical lookup ceiling and normalized CE floor;
it does not prove linear separability, optimizer convergence or live success.

Why this: pooled catalogue bounds already existed. Per-head groups and an
explicit current-input/action check fill the actual diagnostic gap without
another learning or orchestration framework.

## Executed result and limits

The single readonly original-source diagnosis found54distinct raw keys,
54distinct canonical keys and27distinct keys in each public-cue head. Every group
had zero conflicting keys, lookup accuracy ceiling1 and CE floor0. All54 input/
action pairings and visible/named cues agreed. No learner defect was demonstrated.

The earlier contextual control's already-paid TRAIN scores remain1/9 at0 and5/9
at checkpoint8 for each source. Its failed game case remains failed/spent; no
later fit or game outcome was reconstructed. Removing exact aliasing and input
pairing as explanations does not establish that its linear heads can fit the
catalogue, or that more passes would help.

14focused tests passed on each platform, with scoped Ruff/format/mypy and
source/procedure/result self-review. Actual work was four pure reductions over
162rows and270nested projection entries, with no model initialization, fits,
native score forwards, updates, physics, probes or final evaluation. The initial
launch used a nonexistent meter filename; its failure is preserved beside a
prospectively authorized path-only repair. No numeric outcome was retried.

Evidence: `artifacts/runs/context-game-train-diagnosis-20261024/` contains the
frozen protocol, actual.json, original and repaired launch receipts, software/
work receipts, reused prior TRAIN metadata and RESULTS.md. Source/prior bytes
remain unchanged. [Issue104](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/104)
tracks scoped acceptance. Self-review is not independent review; full R4.2,
scientific benefit and release readiness remain open.
