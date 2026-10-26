# Native games from chronological affine learning

The existing game evaluator can restore the genuine causal affine TRAIN states
from issue107: after learning A, after learning B and after A returns. It runs
two known current-phase games at each state. No new fitting or teacher actions.

Why this: exact TRAIN action fit alone does not establish complete task
performance. The offline joint model's earlier A success already knew B/C;
these states originated from the actual delayed A/B/A training sequence.

## Use

From the repository root with the accepted local107report available:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-causal-game --describe
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-causal-game --source artifacts/runs/context-linear-stream-20261024/actual.json --output new-causal-games.json
```

Each invocation needs its own prospective readout budget. The mode requires the
exact qualified source bytes, accepts no fit overrides and never replaces an
existing output. It does not open the original TRAIN148 data or final/probe/
controller-validation data. Input is bounded1MiB, output4MiB. The original issue108
native allowance is FAILED/SPENT; the example is usage, not permission for an
automatic repeat of that failed experiment.

## Contract

The original `_run_checkpoint` loop is reused, including seven legal native
actions, raw first argmax, seed17/max16steps, one public initial frame per game,
owned scorer forks, state guards and attempted environment closure. New exact
binding accepts states2/4/6 with18/36/54 cumulative source row exposures and two
nine-row heads. Source remains untrained; snapshots do not grant new training
or reset lifetime authority.

The three pairs are fixed: state2→Ablue/red, state4→Bblue/red, state6→Ablue/red.
All share one original history of6fits54rows, not twelve fits or108rows.
New fit/optimizer/LAPACK/teacher/replay/probe/final/promotion work is zero.
Independent environment resets seed their own local generators; caller NumPy
RNG is checked unchanged. All physics, projections, copies, model validations,
source IO and report work are included in the measured readout costs.

One original zero-update/replay150s512MiB budget covers ALL phases. Original
clock/start/progress/sampler/config and zero charges remain pinned. Before more
work, verify the source and exact detached prior complete reports. A normal
exception retains known prior reports and current paid prefix, and stops later phases.
A complete negative game remains a valid outcome; success is not forced.

After the failed hard-stop attempt, the command now stops work at140seconds,
reserving10seconds of the same150second limit for cleanup and serialization.
The simulated deadline path passes; real native timeout cleanup remains unproved.
External process loss can still lose reports that have not been serialized.

## Evidence and limits

See [frozen protocol](../artifacts/runs/context-linear-causal-games-20261024/PROTOCOL.md)
and [results/commands](../artifacts/runs/context-linear-causal-games-20261024/RESULTS.md).
32distinct checks qualify on Windows/Linux; scoped statics and self-review are
distinct from independent acceptance or release readiness. Two local-artifact
checks ran here and explicitly skip if their source is unavailable elsewhere.
Four initial Windows fixture failures and all extra costs are retained.

The sole native attempt exhausted150seconds before writing its result. Observed
54predictions/53physics returns/6resets,0new fits, sampled peak81,420,288B.
No per-game traces, closure proofs or final source/RNG checks were saved.
Complete-game successes and all-six acceptance are therefore **unavailable**;
do not infer them from counters or rerun/rescore the spent case.

Known current-task games do not measure native old-task retention. Existing
TRAIN old-task fit falls to13/18 after the other task, and is not hidden by
successful current-task play. Generalization, statistical reliability, CPC
advantage, online deployment/promotion and repository-wide release are separate
requirements. Existing BP/PC/neutral/active controls and all negative/spent runs
remain unchanged; no comparison or tuning against final evaluation is performed.
