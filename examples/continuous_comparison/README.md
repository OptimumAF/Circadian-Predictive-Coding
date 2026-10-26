# Saved continuous replay comparison

Exact copy of the accepted continuous-replay-comparison-20261020 summary.
Original numbers, directions, environment, baseline disposition FAILED/SPENT,
qualification and all24benchmark/128probe contrasts are retained. Project-generated
numeric evidence under the repository MIT notice; no SDK assets or weights.

Use `python -m scripts.build_key_door_inspector --example-comparison --output comparison.html`.
This reads saved JSON, not the experiment driver. It trains no learner. The fixed
A/B/A stream and disposable fresh-task probes show learning/retention tradeoffs;
replay adds optimization and exposure. This does not demonstrate a CPC advantage,
matched computation, final evaluation, robot success or generalization.

Original commands/protocol/independent review (repository-local, not bundled):
artifacts/runs/continuous-replay-comparison-20261020/{PROTOCOL.md,RESULTS.md,REVIEW.md}.
No old baseline or model work is repeated for display. Additional costs excluded
by the original summary stay unknown, not zero. Extend via the existing inspector
source-cell contract, preserving failures/contrasts rather than selecting winners.

## Accepted saved controls

`controls.json` is an exact copy of the original accepted saved no-replay control
cohort from continuous-probe-continuation-20261020/remaining-arms.log (1,647,144B).
It includes complete reports, scalar state summaries and original FAILED/SPENT
history; it contains no parameter checkpoint. It is project-generated numeric
comparison evidence under the project MIT notice. Never reconstruct native
parameters or retrain old controls to display/validate this evidence.

`python -m scripts.continuous_replay_example --example-baseline --validate-baseline-only`
checks saved-control consistency without a new study or training. Normal execution
still requires the fixed original producer profile and a separate local protocol.
