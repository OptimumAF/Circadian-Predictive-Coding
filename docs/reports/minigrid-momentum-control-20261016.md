# MiniGrid matched momentum control — 2026-10-16

The existing runner completed one prospectively reviewed local case: a shared
warm128 learner, eight reward updates and eight separate zero-loss momentum
updates on exactly the same122observed TRAINrows, then all three owned readouts.
Source/servingactors remain unchanged. This is a conditional same-observation
control, not a CPC comparison or independent exploration experiment.

## Known TRAIN endpoints

| Context | Warm CE | Reward CE | Momentum control CE | Goals: warm / reward / control |
|---|---:|---:|---:|---:|
| A | .145778 | 1.904723 | .166440 | 1/2 / 0/2 / 1/2 |
| B | .946894 | 1.837955 | .975339 | 0/2 / 0/2 / 0/2 |
| C_train | .505898 | 1.843649 | .614850 | 1/2 / 0/2 / 0/2 |

The simpler momentum control has lower CE than reward continuation everywhere
and keeps A's red-cue goal that reward continuation loses. Yet the control also
worsens CE from warm everywhere and loses C's red-cue goal. All both-cue joint
fit+goal acquisition flags are false for warm and both continuations. No fulltask
retention/recovery or useful newtask adaptation is established.

A meanreturn is−.253125 for warm/control and0 for reward. Warm/control combine
one−1wrong-key failure with one+.49375goal; reward produces two zero timeouts.
Thus reward has higher meanreturn and fewer goals. Preserve both measures;
zero-return timeouts are not successful task performance. C also reaches0return
by losing its goal/wrong-key pair. No metric, seed, baseline or dose was changed
to obtain these outcomes; all prior negative/incomplete cases remain.

## Cost and qualification

This case pays144SGD1396rows (128warm1152 +8reward122 +8control122repeatedrows),
28rawforwards252rows/410physics/30envs/2058recurrentpositions. Added diagnostics
18raw162rows/18sessions/240livephysics take169.812405s; control training5.514985s
and reward collection/training41.269373s are different workloads, not equalFLOPs
or directly comparable SGD timings. Whole231.046233/300s; parent236.858730/330s.
Peak sampledprocessRSS372776960B<1GiB, CLIimportpeakunmeasured; JSON9125894B<16MiB.
No final/probe/teacher/replayalgorithm/promotion/benchmark learning occurs.

The failed original90/180case remains spent. An explicit nondefault v2 option
prospectively grants210readout/300whole/330parent while preserving original45s
reward/control caps and frozen completed budgets; oldv1/default routes unchanged.
The qualified duplicate-entry-proof repair retains finalcallback validation.
206distinct focused cases qualify across205unchanged passes plus one corrected
receipt-reader test; firstfailure preserved. Ruff/format/win32+Linux-target mypy
pass. Independent SOURCE/GO/nativeACTUAL accepted; scope does not prove release
readiness. [Exact commands, artifacts and limitations](../../artifacts/runs/r21-minigrid-momentum-v2-20261016/RESULTS.md).

All six warmcase costs total828updates7824rows, permanently charged; separate
syntheticunit8physical7published14rows+2failedrows remain separate. All allowances
spent. Next investigate frozen TRAIN per-cue CE gradient geometry on isolated
parameter clones, before choosing any learner change. Existing R2.1probes/health
and matched BP/PC/neutral/active evidence are already accepted and remain reused.
