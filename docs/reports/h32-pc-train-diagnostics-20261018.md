# H32 PC TRAIN update diagnostic — 2026-10-18

The existing TRAIN-only readout now runs on the explicit H32 PC/neutral route.
One original-data **PC** run completed; neutral routing has scripted coverage.
The benchmark and isolated F16/A16 results match accepted evidence exactly:
128 update receipts,270 TRAIN probability rows and32 F/A checkpoints.

| Source update | Prior CE | Counterfactual CE change | Gradient · actual delta | Final latent gradient L2 |
|---:|---:|---:|---:|---:|
| 0 | 2.011718512 | −0.08215618134 | −0.08424370318 | 1.795455587 |
| 128 | 1.121374488 | −0.002229809761 | −0.002229333985 | 3.022359943 |

All four parameter groups point downhill at both states; latent objectives
fall across two relaxation steps. This gives no sign-inversion evidence there.
Nonzero residuals do not establish the cause of the observed5/9-per-cue plateau,
convergence or a useful/CPC advantage. All raw signed values remain saved.

Native work:192updates/7488rows,94predictions/846rows,0SGD,10teachers/90physics.
Diagnostics add108projected rows/255744arrayB,216priorforward rows,108backward
rows,324relaxforward rows,2temporary parameter sets and0learner updates. Source,
actor,input,RNG,original budgets and paid evidence remain guarded. No final data.
Body32.341128s, outer36.2543013s, peakRSS358440960B; oneSPENT/no retry.

525 focused scripted checks, scoped Ruff/format and Windows/Linux source types
pass. Independent SOURCE review preceded native execution; independent ACTUAL
review is recorded with the evidence. No fullsuite/release/gameplay claim.

Next implement explicit H32 composition of the already existing fixed S64
control, preserving original H8/S2 and all negative results. H8 S64 already gave
35/54TRAIN and0/6exact; this is a finite-relaxation sensitivity question, not a
promise of improvement. Review a separate bounded protocol before more models.

[Full commands, failures, costs and skips](../../artifacts/runs/h32-pc-train-diagnostics-20261018/RESULTS.md) ·
[Independent review](../../artifacts/runs/h32-pc-train-diagnostics-20261018/REVIEW.md)
