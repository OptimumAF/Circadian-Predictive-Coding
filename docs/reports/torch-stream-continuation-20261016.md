# Torch labeled-stream continuation — 2026-10-16

The existing attribution runner now reconstructs complete labeled-stream state
at a committed wake boundary under its original live authority. This is actual
candidate/inbox/buffer replacement, with native optimizer/gradient/local RNG,
delayed labels, checkpoints, replay rotation/events, original clocks, budgets
and receipts retained. No disk/process recovery or trajectory claim is made.

The sole fixed eight-arm CPU run paused/resumed at wakes2/4, then finished at6.
All16 reconstructions completed with one pending unlabeled arrival at each pause.
External saved-data analysis compared **every common arm field**, not just scores,
against the accepted uninterrupted artifact: all eight arms match exactly.
No baseline experiment was rerun and no historical metric entered training.

Costs:60 native calls/returns (48wake/12replay),108 row exposures,45 PC-family
steps/15 SGD steps,64 final-score forwards,16pauses/16resumes, one joined RSS
sampler. Pilot8.5310614s, worker8.5686591s, parent9.4505944s; observed peak process
RSS312242176 bytes,203121-byte JSON. Same60s/768MiB/2MiB Torch engineering cap;
no revision of scientific budgets. These timings are not an efficiency comparison.

All8 negative B gains and7 negative return gains remain exactly unchanged; only
replay-on BP has positive return gain. See the [original full interpretation](torch-feature-rehearsal-20261016.md).
Implementation equivalence does not establish useful acquisition, a causal replay
effect or circadian benefit. Off/on work and BP/PC objectives still differ.

Ninety-nine current cases qualified via98completed broad cases plus one narrow
renewal regression. The original broad invocation's guard86 failure is preserved:
its epoch counter mistakenly included BP under the PC cap. There was no assertion
failure or unguarded tail, and all120 Torch correctness-control training calls
were reused without repeat. Native paired tests establish full state equivalence
and global RNG isolation; fake tests cover corrupted state, stale/foreign/reused
tokens, changed authority, wrong-return restore, failures and final isolation.
Seven-path Ruff/format/Windows+Linux mypy and independent SOURCE/GO/ACTUAL pass.

R2.4 snapshot/resume acceptance is met for the supported owner-local labeled path.
Trajectory replay remains unchecked. Process durability is an unsupported limit,
not a newly invented original-task blocker. No repository-wide readiness follows.
Next substantial learner outcome: R2.5 versioned slow-teacher distillation with
explicit staleness/schema rules and teacher storage/forward costs, then a separate
frozen ablation. No further pause/capture/provenance increment is automatically next.

[Evidence, commands, failures and skips](../../artifacts/runs/r24-stream-continuation-20261016/RESULTS.md),
[review](../../artifacts/runs/r24-stream-continuation-20261016/REVIEW.md),
[all scores](../../artifacts/runs/r24-stream-continuation-20261016/checkpoint-levels.csv),
[guide](../attribution-stream-continuation.md).
