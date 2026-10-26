# TRAIN input bounds and observed fit gaps

The existing MiniGrid joint-catalogue pilot can report how much row error is
forced by identical inputs with different labels. This report is optional and
does not choose probes, tune a learner, or change benchmark metrics.

## Use

Add `--train-fit-bounds` to an existing joint-catalogue command. For example,
this command describes the protocol without constructing a model:

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --distinct-probe --old-task-readouts --return-to-a --joint-catalogue --train-fit-bounds --describe
```

The API equivalent is `run_pilot(..., joint_catalogue=True,
train_fit_bounds=True)` in `src.adapters.minigrid_learnability`. The default is
`False`; it adds neither fields nor work to existing reports. The report is
computed after benchmark training and before reserved probe collection, using
only the six original complete TRAIN sources. Existing source and report seals
reject replacements or corruption. Successful method publications add five
`fit_gap_outcomes` records alongside the unchanged `outcomes`. A later failure
retains completed bounds and paid records without inventing a successful gap
publication.

For an existing saved report with top-level `training_sources`, use the original
linear-control CLI's explicit bypass:

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_linear_fit_control --bounds-only --describe
.venv/Scripts/python.exe -m scripts.run_minigrid_linear_fit_control --bounds-only --source artifacts/runs/weighted-relax64-study-20261018/weighted_pc_relax64/native.json --output train-bounds.json
```

The second command requires a new output path. It decodes only the existing
TRAIN source records, computes bounds, and bypasses fitting and prediction.
It does not compute observed gaps from an arbitrary saved summary; the scoped
experiment driver separately verifies the original saved readout means before
computing its gaps. Final and reserved labels are not arguments to the bounds
helper. Running the ordinary pilot still incurs its original training/probe
budgets; adding this report does not grant more work.

## Meaning

`src.core.categorical_input_bounds` accepts finite float64 feature matrices with
1..128 rows, 1..296 coordinates in [-1,1], hard labels and 2..16 classes. It
returns an immutable scalar `CategoricalInputBounds`. It has no model, RNG, IO,
or role authority. The existing adapter supplies strict TRAIN admission and
causal public-history validation.

For each exactly equal numerical input key, let its class counts be `n[k]` and
its total count be `n`. Signed zeros compare equal; other distinct finite values
are not merged by a tolerance.

- Maximum correct rows: sum of `max(n[k])` over keys.
- Row accuracy upper bound: maximum correct rows divided by all rows.
- Normalized CE lower bound: sum of `-n[k] * log(n[k]/n)` over positive counts,
  divided by all rows.

An unrestricted deterministic lookup attains the accuracy ceiling; the empirical
conditional label distribution attains the normalized-simplex CE floor. Neither
establishes attainability or convergence for the finite H8 network, exact episode
success, live behavior, retention, recovery, or generalization. Original score
validation allows absolute 1e-6 probability-sum roundoff. Signed CE gaps can
therefore be slightly negative and are preserved. Metrics are not renormalized
or clamped. Equal context and cue means equal row mass here because all six
original episodes contain nine rows. Gaps use original **unweighted** scores,
including when the selected training objective is weighted.

## Cost and evidence

Each successful bounds report counts two 54-row reductions, 108 existing
current/next admission projections, 54 additional canonical projections, and
256176 bytes of raw/canonical/target arrays. Logical key and count payload bytes
sum both transient reductions; they are neither retained storage nor a measured
peak. Python objects, copies, validation, failed work, and enclosing wall/RSS are
additional. There are zero native updates, model forwards, or solver fits.
The helper's elapsed field excludes initial admission and the final witness
check; the enclosing supervised wall time includes them.

The saved-source case confirms the previously accepted 54 distinct raw and
canonical keys with zero conflicts: ceiling1, floor0. The original weighted-PC
update128 has accuracy31/54 and CE1.345738544779821, hence accuracy gap23/54 and
the same CE gap. This is a report of existing underfit, not new learning evidence
or a circadian advantage. The earlier least-squares control52/54 is not a proof
of linear inseparability. All earlier failures and negative results remain.

Checks:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_categorical_input_bounds.py tests/test_minigrid_train_fit_bounds.py
```

The integration tests run the real pilot publication path with scripted paid
ports, without native learning or prediction. The final combined scoped check
has335 passing cases. See [exact commands, failures and costs](../artifacts/runs/train-fit-bounds-20261018/RESULTS.md).
Whole-repository, release, and scientific adaptation claims remain open.

Extend this by composing the pure reducer with an existing role-admitting
adapter; keep it report-only. A model-specific capacity or optimization
diagnostic needs a separately named disposable learner and prospective work
cap, with its data isolated from benchmark, deployed and final evaluation state.
