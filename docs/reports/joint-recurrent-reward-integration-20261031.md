# Live joint TRAIN to reward/control integration

R4.2-RNN.JRL.I / [issue390](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/390).
The complete software path is implemented and executed with literal native and
environment ports. Independent scoped source/procedure/software-result review
accepted the integration; lead self-review is recorded separately.
No genuine native experiment has run for this new composition. Full390 and
broader scientific/release requirements remain open.

## Delivered behavior

```powershell
python -m scripts.recurrent_joint_example --reward-training --describe
python -m scripts.recurrent_joint_example --reward-training --output joint-reward.json
```

Execution requires a separately budgeted native protocol; describe is metadata
only. This exclusive modifier preserves the original unweighted joint128/1152
schedule, raw148/H8/K7/seed17/lr.03/momentum.9 and two64-update/576-row heads.
The existing stream's128 paid receipts, source/actor/benchmark state, inputs,
RNG and original authority/births are checked throughout. Original prefix0/128
scores/games are skipped in this mode. Other modes/defaults are unchanged.

The small `minigrid_joint_warm` adapter transfers the actual current-run full
endpoint to `PublicCueRewardLearner`, including parameters, optimizer momentum
and gradients. No historical JSON weights, duplicate source fits or new runner.
It binds original and newly restored warm owners/values around callbacks;
single-use preparation refuses corrupt/equal-replaced state. Pure live checks
avoid snapshots that could return old values after mutation. Failed final
handoff proof retains the paid pilot report rather than accepting continuation.

The existing reward pilot collects eight TRAIN A/B/C_train/A cue-pair episodes,
sampling29/environment17/max16, then releases matching identifiers through its
existing delayed runtime. Eight original gamma1 reward updates are followed by
eight existing native zero-loss updates on the unchanged observed trajectories
in a separate candidate. Original actions/rewards/probabilities remain evidence,
not relabeled on-policy control targets. The warm source and BC control stay
frozen. Both candidates share warm values/full optimizer, but the zero control
is conditional on reward-arm observations, not independent exploration.

After all updates, the existing readout runner/kernel scores and plays all six
known TRAIN tasks for frozen BC128, reward136 and zero136. The new facade admits
only exact128/136, default policy, no failed work, per-head64/68 and row clocks
derived from the exact eight observed TRAIN trajectories. It pins that cohort
and preserves the old fixed-row facade. Readout has zero update/replay authority;
no D/probe/final data, promotion, dose/seed/arm selection or early stopping.
Each returned record is labeled `frozen_joint_bc`, `reward` or `momentum_only`
before sealing, with original warm/continuation objective IDs. Architecture
method remains `recurrent_public_cue`; the two136 arms are explicitly distinct.

The source128/1152 lineage is reported separately from the pilot's new work,
avoiding double counting. Whole committed updates are144 on completion; failed
attempts/unknown work remain in original budgets/slots. Source180s, outer480s
and readout authority420s are created with original births for this new mode;
inner existing reward45s/control45s/readout210s/whole300s ceilings remain. The
outer768MiB supervision is stricter than the inner1GiB limit. New CLI output
cap8MiB; historical caps unchanged. These are ceilings, not measured native cost.

## Executed software evidence

All project commands use physicalD and process-local TEMP/TMP `D:\Dev\_Temp`:

```powershell
./artifacts/runs/joint-recurrent-reward-integration-20261031/run-checks.ps1 -Attempt core -Suite core
./artifacts/runs/joint-recurrent-reward-integration-20261031/run-checks.ps1 -Attempt combined -Suite final
./artifacts/runs/joint-recurrent-reward-integration-20261031/run-checks.ps1 -Attempt readout -Suite diagnose
./artifacts/runs/joint-recurrent-reward-integration-20261031/run-checks.ps1 -Attempt final -Suite final
./artifacts/runs/joint-recurrent-reward-integration-20261031/run-checks.ps1 -Attempt handoff -Suite handoff
./artifacts/runs/joint-recurrent-reward-integration-20261031/run-checks.ps1 -Attempt labels -Suite labels
```

| Attempt | Pass/fail/skip | Body seconds | Sampled peak bytes |
| --- | --- | ---: | ---: |
| core |9/2/0|9.5161702|99258368|
| combined |78/2/0|40.7406761|111804416|
| targeted readout diagnosis |0/1/0|16.5168857|97320960|
| combined before final helper refinement |80/0/0|80.4014422|108802048|
| final targeted handoff tree |12/0/0|57.2460028|99397632|
| comparison labels only |1/0/0|2.1593802|80924672|

Final targeted PID63892exit0/57.410153s host/no expiry. That run executes the
complete actual joint stream128→full-state transfer→eight reward/eight zero
runtime commits→three six-game readouts, plus new handoff corruption and strict
readout/metadata cases. The earlier80-case combined pass includes69 historical
default/warm/control checks and paid-failure paths; those components remained
unchanged by the final helper-only refinement. This is not a claim of a fresh
83-case full-tree run. Fake environment outcomes are not learned task evidence.

All failures are preserved. First fixture constructor omitted positional
seed/maxsteps; second used a collector-only environment without legal_actions
and queried a finished stream through its closed API. Constructor/property and
runtime snapshot access were corrected. One targeted diagnosis exposed the
missing property without repeating the full baseline cohort. Production review
also supplied actual native row counts to the kernel for variable-length136.

Lead final review found the new warm owner needed a full-state lock and the
original helper's last snapshot could hide equal endpoint replacement. Direct
live values/strong owner bindings now bracket the same parent guard positions,
including the restored warm source. A prospectively recorded fifth targeted
slot (max14cases) covered that production change. Original four slots remain
spent. All original cumulative400-execution/300s/counter/storage caps unchanged;
no scientific quota renewed. Total184executions179PASS5fixtureFAIL/204.421177s.
Independent review then required explicit comparison identity after kernel
records replace placeholders. A sixth prospectively bounded label-only slot
(max2cases, no joint-prefix repeat) passed the actual readout-loop labeling and
post-seal corruption check. No data, metrics, directions, dose or native budget
changed. Final cumulative185executions180PASS5fixtureFAIL/206.5805572s; the
earlier184/204.421177 subtotal remains above to explain the original fifth slot.

Cumulative conservative literal ports:2872constructors,1922supervised updates/
17298rows,574public reward/control requests/1148rows (including refusals),
48prediction calls/432rows,150live calls. Successful final run has128 source
updates/1152rows,16new reward/control requests/32rows,18raw scores/162rows,
52live calls/26closed literal environments. Genuine models/initializations,
SGD, solvers, physics and source payload work are zero; blocked imports zero.
These are fixture port measurements, not native cost, FLOPs or task usefulness.

## Static scope, review and limitations

```powershell
$files = @('scripts/recurrent_joint_example.py', 'src/adapters/minigrid_joint_warm.py', 'src/adapters/minigrid_reward_pilot.py', 'src/adapters/minigrid_momentum_control.py', 'src/adapters/minigrid_warm_readouts.py', 'src/adapters/minigrid_policy_evaluation.py', 'tests/test_joint_reward_integration.py')
.venv/Scripts/python.exe -m ruff format --check @files
.venv/Scripts/python.exe -m ruff check @files
.venv/Scripts/python.exe -m mypy --follow-imports=silent --cache-dir artifacts/runs/joint-recurrent-reward-integration-20261031/mypy-cache @files
```

Final seven explicit files format/lint/scoped types0,.7060188s; intermediate
scoped0/3.4832026s. Initial8.8593962s had1unused-import lint and12type diagnostics:
three owned issues fixed; nine imported-body diagnostics outside scope remain
(return_policy_evaluation3, imported test_recurrent_readout6). Imported annotations
remain available under follow-imports=silent; this is not repository-wide type
acceptance. Unrelated files were preserved and the initial diagnostics recorded.
The final label-only production delta also passed the same seven-file scoped
format/lint/type commands,1.0327693s in `statics-labels.json`.

The peer used three reads/.6891628s. Its first95-token handoff omission was
recovered in read2. Read3 omitted258 middle tokens in a cohort/default-admission
test setup; those tokens are not independently claimed inspected and no extra
read followed. Production source/deltas, substantive integration/failure/label
checks and actual80/12/1 receipts were reviewed. No numerical replication or
native GO; the finite exception ended. This is software acceptance, not a
native learning, scientific-benefit or repository-wide claim.

Stage `artifacts/runs/joint-recurrent-reward-integration-20261031/` retains five
before-copies, prospective protocol/amendment, existing adapted checker/launcher,
all SPENT/log/host/test/static receipts and final source pins. Stage42files/
20,477,790B/cache20,197,605B before later review/report receipts, within64/40MiB.
Lead self-review is separate from independent review. No new dependency/install,
Git mutation, download, GPU, WSL, CI, baseline rerun or historical-data recovery.
Full suite/clean clone/Linux/packaging/browser/device/native/release skipped.
PriorP/M failures/SPENT, #389 terminal D, #388 negative weighting and all prior
scientific/review limitations remain unchanged.

Next: freeze ONE small TRAIN-only native protocol from this joint prefix using the existing
fixed BC and zero controls. Account every source/candidate/control/diagnostic/
copy/RNG/physics/time/RSS/storage attempt. No run until those gates pass. Negative
or inconclusive results are valid; no tuning from D, favorable seeds, dose
search, CPC advantage, statistical generalization or release claim. Full390 and
parents18/21/23, reliable adaptation, separately evaluated useful recurrent RL,
R5.3 and all broader roadmap requirements remain open.
