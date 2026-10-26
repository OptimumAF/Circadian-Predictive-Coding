# Guarded publication while live serving continues

The native serving demo can publish once at its causal TRAIN drain, then serve
later requests on the learned actor:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --publish-during-serving --output drained-serving.json
```

Use a fresh output path. This opt-in mode uses the same24tick live plan, original
drift0.1 controller, eight6row new-context TRAIN batches and qualified96update/
576row BP source as the [original live mode](live-ticket-serving.md). It changes
publication timing; it does not refit the source or change guards, thresholds,
optimizer settings, labels or native learning rules. Existing CLI modes retain
their defaults. The Python equivalent is `run_case(source, reference_mode="drift",
serving_plan=LiveServingPlan(), publish_on_drain=True)`; other study/deferred/
counterfactual paths cannot enroll this option.

## Causal boundary and authority

Publication is attempted once after every original TRAIN update has committed,
all labels have arrived, pending predictions are empty and the worker has joined.
The original gate must be idle and native completion/receipt counts complete.
No live workflow result chooses this boundary. `_publish_candidate` reuses the
original development guards, training IDs, metadata, prepare and commit methods.
No alternative guard, repeated evaluation or fallback promotion is introduced.

`_publish_at_drain` checks the actual actor generation/version before and after
the callback and the original typed commit receipt. Metadata uses exact primitive
types; bool/float generation coercions are refused. A lost/mismatched return
retains unknown-publication evidence and stops further work. Refusal retains the
old actor and does not cause a later retry. The tiny genuine native correctness
fixture demonstrates guard refusal, not forced publication.

After a verified swap, the old scalar TRAIN feedback is copied to an invalidation
record and removed from subsequent controller observations. Its old actor version
is preserved; it is never relabelled as new. Earlier controller events and label/
update receipts remain intact. Because this boundary has consumed the original
envelope, subsequent choices wait; no budget renewal, candidate replacement or
new learning epoch is implied. Future data after a publication needs its own
explicit original authority, not reuse of this spent envelope.

The separate [two-window option](post-publication-learning.md) now declares both
windows and all lifetime allowances before construction. It admits newly observed
feedback after the first verified publication while preserving this single-window
mode, its original spent envelope and its historical evidence.

Each later typed request reads its actual atomic publication, as before. Existing
complete rollback still restores original model/optimizer/configuration/metadata.
Live outcomes never enter TRAIN, controller feedback, guards, final/probe selection
or tuning. No final/probe source is opened. The original post-stream mode remains
available and unchanged.

Why this: the preceding live trace completed learning at tick10 but withheld the
candidate until the window ended. Reuse the existing verified publication boundary
at a causal drained point rather than adding another runner or renewing budgets.

## Executed evidence

97selected combined checks/platform plus19focused scalar metadata checks/platform
pass, alongside scoped Ruff/format/fresh default+Linux mypy. Seven old reference
cohort/failure cases were deliberately reused/deselected. This is scoped
single-agent self-review, not independent acceptance or repository-wide testing.

The sole frozen native trace used8updates48rows. Guard acceptance at tick10 changed
generation0/qualified-96/native96 to generation1/c0/native104; all requests from
tick11 used that publication. Old batch-7 feedback was preserved/invalidated, and
no more training ran. Legacy still failed at1/5/9; it succeeded at13/17/21. Live
success was18/24, including all invoice/password/ordinary-parcel requests; all
three revoked parcel commits refused and three stale invoice episodes recovered.
Applied permission/revision violations were zero in this fixed fixture.

The learned full candidate is exactly equal to the saved original live trace's
candidate, verified from unchanged artifacts with zero new model work. The
deployment change exposes the already learned behavior sooner. Complete rollback
restored the original96state and legacy failure. Source and caller RNG unchanged.
This is known-template task usefulness, not recovery from forgetting, controller
advantage, generalization or CPC benefit.

Physical case:8SGD/48rows/8backwards,588head constructor entries22932constructor
matrix-slot upper,168forward invocations1347snapshot constructions180240logical
elements,148typed requests172local dispatches,0PC/finalTRAIN/probe/replay/newfit.
Internal7.9785s,meter9.4542s,outer13.8445s,sampled peak335024128B. Instrumented
78live predictor calls:p50=14.00ms,p95=15.99ms,max=19.55ms, with queue/reservation/
workflow durations recorded separately. Constructor/element/forward accounting is
partial, RSS sampled current-process only; profiling/startup/order prevent latency
SLO or comparative-efficiency claims.

```console
python -m pytest -q tests/test_drained_ticket_publication.py
python -m ruff check scripts/run_bp_ticket_serving.py tests/test_drained_ticket_publication.py
python -m ruff format --check scripts/run_bp_ticket_serving.py tests/test_drained_ticket_publication.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/run_bp_ticket_serving.py tests/test_drained_ticket_publication.py
```

Exact commands, source pins, failures, scope/costs and sole-run receipts:
`artifacts/runs/drained-ticket-publication-20261023/RESULTS.md`. Full R7.7 still
requires noisy/poisoning-feedback and long-duration/broader utility validation;
R7.1 gate/sleep/combinations and scientific/release goals also remain open.
