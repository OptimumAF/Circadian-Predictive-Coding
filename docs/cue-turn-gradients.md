# Selected TRAIN-row prior gradients

The existing encoding-bound prior-gradient reader can isolate specified rows of
two original complete TRAIN episodes. The disposable fit mode can request the
two original A cue-turn rows that remain wrong after512 BP updates. This is a
readout of the prediction CE gradient; it does not change a learner.

## Use

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_linear_fit_control --constructive-bp --cue-turn-gradients --describe
```

Explicit execution example, subject to a prospectively budgeted local protocol:

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_linear_fit_control --constructive-bp --cue-turn-gradients --fit-updates 512 --fit-seconds 90 --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --output cue-turn-fit.json
```

The flag requires `--constructive-bp`. Defaults keep the previous report and
work unchanged. No new seed, optimizer, objective, dose or baseline is selected.
The fit mode completes all training and original fit scores first, then reads
A-blue/A-red row2 at fixed0/128/512 checkpoints. The same budget covers reads,
copies and guards; partial diagnostic work and raw scores remain on refusal.
It does not start a new clock or grant more work.

API:

```python
from src.adapters.torch_feature_gradient_diagnostics import diagnose_prior_gradients

report = diagnose_prior_gradients(snapshot, complete_train_pair, row_indices=(2,))
```

`row_indices=None` preserves the existing full-episode definition, loss and call
shape. A selection must be an exact nonempty tuple of increasing distinct native
integers valid in both episodes. The full original causal current and next
views—including unselected rows—are admitted before optional Torch access.
Selection only copies numeric arrays. Episode roles, IDs, histories and labels
are not rewritten into fictitious short episodes. Full-source admission and
selected-row projections have separate work counts.

The selected definition is
`torch_invariant_feature_selected_rows_prior_ce_gradient_geometry_v1`.
Each loss names its original episode and selected row count. The four detached
parameter gradients use the same mean CE/tanh/logits and scalar geometry as
the existing reader. Snapshot optimizer state, gradient buffers, CPC chemistry,
RNG and caller grad mode remain unchanged; inference mode is refused.

## Interpretation and cost

Positive cosine means local alignment and negative cosine means local
opposition between the two supplied gradients. Neither establishes causal
interference, incapacity, convergence or circadian benefit. These are prediction
CE gradients at the prior, not native PC latent-energy update directions.

At the default three checkpoints, the option adds3 full snapshot validations,
6 forward calls/6rows,6 backward calls/6rows and zero optimizer steps or published
updates. It also adds108 full current/next admission projections and6 selected
projections. Full source arrays, detached copies, graphs, validation, Python
objects, timing and RSS are additional. Existing18 classifier calls/162rows and
512updates/27648rows remain separately counted. `elapsed_seconds` on each
diagnostic runs from its opening clock through copies and closing clock, excluding
the surrounding first/last guard; the enclosing invocation includes all guards,
admission, work and output. It is not kernel-only timing.

The earlier512 negative remains authoritative:52/54rows,4/6exact, both Astep2
errors, no acquired-old-A premise. Do not select B/C afterward to claim retention
or recovery. Reconstruction needs native states absent from the earlier saved
probability-only artifact; its entire0/128/512 fit must bind the original run.

## Executed reconstruction

One frozen reconstruction completed with all prior probabilities identical
(maximum difference0.0). Fit stays52/54rows,4/6exact and `fit_witness=False`.
The selected A-step2 gradients have joint cosine−0.10636947 at0, +0.43659278
at128 and−0.99571962 at512. Final norms are12.71988/10.48137 and dot−132.75113;
the two selected losses are0.74855739/0.91442025. Saved arithmetic qualified
loss consistency and all work counts. These measurements describe local
opposition and do not establish its cause.

The run used512updates/27648TRAIN rows,18classifier calls/162rows and6additional
prior forwards/6rows plus6backwards/6rows. It made zero diagnostic optimizer steps,
replay, teacher, game, reserved or final calls. Parent wall16.4270935seconds,
sampled peakRSS449073152bytes; sole saved arithmetic0.4985402seconds. The new
120second grant closed at16.9256337seconds spent; unused time was closed.

The next explicit hypothesis is a public cue-relative color representation.
Its implementation must first reconcile actual cue visibility and object codes,
preserve essential initial-cue information, and use no context IDs, labels,
future views or oracle state. Keep the original encoding and results intact,
with a distinct encoding tag and the same transform for all four matched
methods. The present gradients do not prove that representation caused the
errors or that a new transform will improve learning.

## Verify and extend

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_cue_turn_gradients.py
```

Independent NumPy analytic derivatives and directional finite differences cover
all four existing native methods, including stored gradient/chemistry/RNG/mode
preservation. Scripted constructive integration covers paid failures, source and
checkpoint mutation, deadlines, report ownership and default behavior. See
[exact scoped checks, costs and scientific evidence](../artifacts/runs/cue-turn-gradients-20261018/RESULTS.md).

Extend through the same complete-episode API and explicit row selector. Keep
diagnostics separate from training decisions and final data. A new comparative
experiment needs a frozen protocol; this API does not establish its conclusions.
