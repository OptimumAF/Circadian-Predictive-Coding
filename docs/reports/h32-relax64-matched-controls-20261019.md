# Matched H32 finite-relaxation observations — 2026-10-19

One new neutral S64 run completes the existing PC S64 pair. Both use the original
joint TRAIN stream and isolated F16/A16 probes. Four accepted original controls
are reused. All six cases qualify. PC and neutral CPC match exactly at S2 and S64:
270 TRAIN probability rows and 32 score/decision checkpoints per pair.

| Learner | Relaxation steps | TRAIN correct / 54 | Exact sequences / 6 | Mean CE at 128 |
|---|---:|---:|---:|---:|
| BP | — | 51 | 3 | 0.186233 |
| PC | 2 | 30 | 0 | 1.121374 |
| Neutral CPC | 2 | 30 | 0 | 1.121374 |
| Active CPC | 2 | 30 | 0 | 1.192282 |
| PC | 64 | 42 | 0 | 0.737842 |
| Neutral CPC | 64 | 42 | 0 | 0.737842 |

BP uses CE backpropagation, without PC latent relaxation. S64 improves partial PC
fit; BP still fits more TRAIN actions with lower CE at the same native learning
doses. Neither PC recipe fully acquires the tasks. Active CPC has no demonstrated
advantage. All 1,526 signed contrasts and 1,620 TRAIN rows were independently
recomputed. Every curve, cue, and negative result remains available.

At source128, S64 F fit is 6/9 per cue before adaptation, 7/9 blue and 6/9 red after
F16, then 6/9 each after A16 returns. Old A stays 7/9 per cue, with CE worsening
after F and improving after return. This is partial fit, not useful acquired-task
retention or recovery. Source0 endpoints remain 5/9 per cue.

S64 uses 32 times the nominal latent-row work: 479,232 versus 14,976 for S2 PC.
The original controls have additional game observations, so timing and total
computation are not matched. New neutral work: 192 updates / 7,488 rows;
94 predictions / 846 rows; no SGD; 10 teachers / 90 physics steps. Body37.9755s,
outer41.0672s, peak RSS360808448B. One authority spent, no retry. Full state, input,
RNG, and original budget checks pass. No final data or new game observations.

28 saved-reader checks, scoped statics, and independent SOURCE/ACTUAL review pass.
A synthetic rounding assertion failed and was corrected; a reviewer comparison
scope error is also recorded. Neither changed native data or metrics. Production
and the accepted605 focused/source type evidence are unchanged. Full research,
release, neural acquisition, retention/recovery, and other roadmap work remain open.

[Commands, costs, failures, and limitations](../../artifacts/runs/h32-relax64-matched-controls-20261018/RESULTS.md) ·
[Independent review](../../artifacts/runs/h32-relax64-matched-controls-20261018/REVIEW.md)

Next finish the existing FIFO36 acquired-task positive control with zero-training
readouts of its five actual saved snapshots. Preserve the original incomplete run
and charges; use a separate reviewed local budget.
