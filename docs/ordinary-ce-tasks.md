# Ordinary BP tasks with a TRAIN-fitted CE readout

The existing chronological command has an explicit optional mode:

```powershell
$env:TEMP='D:\Dev\_Temp'; $env:TMP='D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m scripts.recurring_ticket_workflow_example --context-return --method bp --train-readout ce --output D:/Dev/_Temp/ordinary-ce-tasks.json
```

Choose a new output path. The mode requires ordinary BP context return with no
replay, probe, private adaptation, separate heads, balancing or other experiment
options. Invalid combinations and existing output paths are refused before data
preparation or model execution. The original command defaults are unchanged.

The existing stream first completes its original twelve updates, 72 TRAIN row
exposures and unchanged development episodes. Only after source qualification,
by default the existing CE fitter learns a frozen output readout from twelve already
released old TRAIN examples. The existing typed predictor then runs all four
known tasks in fixed order, with the original allowed actions and six-step bound.
It runs later tasks after a negative task outcome, provided the preceding
computation and evidence are complete. Operational failure stops further work.

Add `--train-readout-cohort joint` to fit the fixed eighteen released TRAIN
examples and run all six tasks, including v1parcel and v2password. The API option
is `train_readout_cohort="joint"` with `train_readout="ce"`, BP and context return.
Omitting it retains the old twelve-row/four-task behavior. The joint option
requires CE mode and refuses other modes before preparation or output work.
Each joint auxiliary closure rereads twelve old and six fresh TRAIN rows; count
that old-data rehearsal explicitly. These are TRAIN definitions, never relabeled
probe or final examples. The fit includes both task groups before readout, so
joint success does not demonstrate sequential acquisition or recovery.


Read `train_readout` separately from the original `summary`. The latter still
describes the unchanged raw BP source. The added section records the auxiliary
fit work, final TRAIN CE, frozen prediction work and complete task transitions.
It never installs the fitted weights into the source or actor. It supplies no
probe checkpoint, promotion or SGD credit.

Auxiliary fitting and task reads have a separate 15-second/512 MiB budget. The
original 30-second/512 MiB source clock continues throughout; do not add these
overlapping elapsed times. Repeated old TRAIN gradient-row exposures are paid
auxiliary work, not independent new examples or free rehearsal. Failed work stays
recorded. `preserved_train_readout` and `preserved_train_readout_budget` retain
canonical paid records when later callbacks corrupt or replace current evidence.
Foreign mutations remain visible; the mode refuses completion instead of
restoring them. Storage is bounded to one fit and at most six task records (four by default).

Why this: the accepted standalone CE control can execute known tasks, but was
previously available in the ordinary command only inside private probe
initialization. This exposes the same conventional component through the existing
task path. It does not change source training or establish a circadian advantage,
novel acquisition, generalization, retention, recovery or matched computation.

[Software checks and limitations](reports/ordinary-ce-tasks-20261030.md) qualify
the new composition and command admission. The
[first genuine checkpoint-12 attempt](reports/ordinary-ce-tasks-native-20261030.md)
solved all four known tasks, then failed final evidence validation and returned
EXIT1. That old four-task attempt remains unqualified; its source/actor final flags are unknown.
That attempt is permanently spent, with no automatic retry. Existing #321
evidence and all failed or inconclusive adaptation results remain preserved.

A [model-free finalization repair](reports/ordinary-ce-guard-diagnosis-20261030.md)
now retains the public open-only snapshot boundary and validates the observed
finished CE source through a sealed read-only port. It preserves the initiating
error if a later check also refuses. The original native case stays failed;
this software repair is not native requalification or proof of its cause.


Joint mode records `train_readout_cohort_requested="joint"` at the root and
`training_cohort`, `training_case_count=18`, `declared_tasks` in the guarded task
record. Corrupted intent or replaced paid task containers refuse more work and
keep canonical failure evidence through source finish and final callbacks.
[Joint software checks](reports/ordinary-joint-ce-tasks-20261030.md) passed107
model-free cases per platform. The separate [native joint TRAIN qualification](reports/ticket-joint-ce-native-20261030.md)
solved6/6 known workflows with the same existing teacher/fitter; its supporting
review-time gap stays open. One [genuine ordinary joint integration](reports/ordinary-joint-ce-native-20261030.md)
now completes real fitting, all six successful three-step workflows, source
finish and final capture with source/actor/both RNG flags true. It pays34CE
closures612rows (408old rehearsal/204fresh TRAIN), plus source12SGD72rows.
This qualifies that fixed joint mode, not331's old failed capture or sequential
adaptation/generalization/CPC benefit. All historical allowances remain spent.
