# Inspect private learning, retention and partial games

Export a saved first-party private-adaptation report through the existing offline builder:

```powershell
python -m scripts.build_key_door_inspector --source-artifact private-adaptation.json --output private-learning.html
```

Open the new HTML file in a browser. It contains its data and makes no network
request, model call or physics interaction. The source stays unchanged; an
existing output is refused. The normal affine and other inspector routes remain
available. Existing input, JSON-structure and 16 MiB output bounds still apply.

The overview keeps the frozen TRAIN reference and both private controls visible.
Fit completion and native-game completion are separate columns. Select a learner
to inspect its recorded scores and games; source buttons open the original JSON
record at an exact pointer. Learning is inactive throughout the viewer.

The reference shows TRAIN checkpoints 0/2/4/6, six causal receipts and catalogue
exposure. Each private learner shows F probe-test and old A/B/C TRAIN readouts at
private updates 0/1/2. Old accuracy is recorded correct actions divided by nine
rows; CE stays as recorded. Dots represent observed complete points, with no
interpolation across missing work. Reused TRAIN diagnostics are not heldout
retention. Worse confidence and incomplete or absent points remain visible.

Native playback uses the existing seven-action, first-argmax public-view decoder.
F precedes original returning A for both learners; A is branch row 3 and F is
row 2. There is no action repair, rerun, reconstructed model or inferred map.
The qualified local example contains five completed games: F-only solves F but
times out on A; joint F-blue completes before the original deadline. Joint
F-red fails before environment construction and A-blue/A-red are unrun.
These slots retain zero frames, disabled playback and unknown outcomes. An
omitted outcome and an explicit null both remain unrecorded, with original
source keys preserved. The joint final `state_unchanged=false` and possible
unobserved failure work remain visible rather than being accepted as proof.

Costs retain original study and arm clocks, attempted and committed charges,
fork counts, matrix storage, diagnostics and native work. The protocol's ten-fit,
252-row plan is labeled as a plan, separately from paid receipts. The permanent
54 extra-F versus 54 old-TRAIN reuse exposures do not disappear after a failure.
Workflow and phase RSS/time keep their recorded scopes; external whole-process
costs are unavailable in this report. Missing quantities are not displayed as zero.

Why this is a separate app projection: private roles and clocks differ from the
normal TRAIN-only workflow. The original definition and source stay intact while
both projections share the existing HTML, native decoder and TRAIN validation.
The viewer checks consistency of saved data, not authenticity, budget authority
or scientific acceptance. It selects or promotes no private model and does not
complete the failed native qualification, establish generalization, matched
computation or a circadian advantage.

Scoped Windows/Linux checks, the saved-report export and desktop/narrow browser
checks pass under single-agent self-review. No new fitting or gameplay occurred.
See [local evidence](../artifacts/runs/private-affine-inspector-20261024/RESULTS.md),
[issue121](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/121) and
[ADR0632](adr/ADR-0632-private-affine-inspector.md). The generated local example is
[private-adaptation.html](../artifacts/runs/private-affine-inspector-20261024/private-adaptation.html);
it is not a bundled pretrained product or a release qualification.

Extend the existing projection with explicit recorded-schema checks and source
pointers. Preserve unknowns and role boundaries; add a separate frozen protocol
before changing learning schedules, metrics or experiments.
