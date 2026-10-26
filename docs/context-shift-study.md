# Fixed contextual shift study

`scripts/context_shift_study.py` runs three small, fixed TRAIN-only conditions
through the existing contextual learner, delayed stream and categorical shift
observer. It checks stationary learning, a fixed stationary label error and
A→B→A recurrence. It adds no adaptation controller.

From the checkout with the existing NumPy CPU environment:

```powershell
python -m scripts.context_shift_study --output NEW.json
```

The destination must not exist. Budget limits are fixed: three twelve-update
streams with six rows per update, 45 seconds per stream, 120 seconds for the
whole case and 512 MiB RSS. The experiment protocol additionally bounds every
model/diagnostic/copy/readout operation and owns one native attempt. Do not rerun
the spent study to select outcomes.

All three conditions restore the same qualified native12 context source. The
noise condition changes only the first label to the next class in every packet,
from its first update. This is a fixed label corruption, not sampled random
noise. New-task packets repeat three rows twice; all exposures count.

Signals use the preceding parameter state on just-arrived TRAIN labels. The
unchanged detector has three reference batches, a two-batch window and two-batch
persistence. Reports include bounded raw metrics, signed score changes, alarms
and onset/continued/cleared incidents. Signals cannot choose training or use
final/probe data.

All training completes before six fixed known-template task readouts at each
of updates0/4/8/12. Readouts use detached private owners and never publish.
Task outcomes describe those TRAIN templates, not unseen-task generalization.

## Observed result

The single frozen run completed36 updates/216 training-row exposures and72
complete workflow episodes. On recurrence, successes were5/6 at updates0/4/8,
then6/6 at12: legacy was acquired, password was lost, and password recovered
while legacy remained. Both stationary conditions stayed5/6 (legacy failed).

Uncertainty and residual first alarmed at6, one update after B arrived.
Uncertainty cleared7; residual continued through the return and cleared11.
There was no separate return onset; feature drift never alarmed. Both stationary
conditions had zero alarms. The fixed corruption caused24 within-pass mistake
updates that canceled in endpoint coefficients; a relative shift detector did
not flag this stable reference-level error. This is not a poisoning detector.

Observer durations totaled0.5899611 seconds, including its copy/validation and
profiling. Wholecase47.20555 seconds, sampled peak47,378,432 bytes; no timing
benefit is inferred. No final/probe/replay work, controller feedback or CPC claim.

Protocol, actual outcomes, costs, failures and checks are retained locally under
`artifacts/runs/context-shift-study-20261023/`. Live task and acceptance history:
[issue99](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/99).
The broader deployable-signal and scientific requirements remain separate.

To extend the study safely, first declare a new question and protocol. Reuse
the existing stream/observer ports; keep final/probe isolation, all earlier
negative results, fixed readout endpoints and paid attempt accounting.
