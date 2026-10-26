# Bounded recurrent replay: software delivered, native result unavailable

The existing recurrent stream pilot now supports recent-episode extra fitting
and context-coverage replay. Both use the same incoming sequence, bank caps and
extra updates/rows/per-public-cue exposure. Both finish training before any
readout. Strict native/logical clocks, causal episode selection, source/actor
isolation and preceding evidence checks are retained. See [usage](../recurrent-episode-replay.md).

Scoped qualification passed59replay checks plus9compact-export checks perplatform,
Ruff/format/fresh Windows mypy and lead self-review. These are scripted/pure
fixtures, not independent review or numerical learning evidence. Linux mypy
was unavailable in the cached environment; no install or full release suite.

The ONE native comparison failed at publication: its indented JSON exceeded
the original12MiB cap before the detailed report was saved. The retained meter
records320native updates2880rows,196physical steps and16closed environments in
341.9048s with385236992B peak RSS. Detailed outcomes and terminal app status/
preservation/RNG proofs are UNKNOWN. The run is failed-spent; it is not retried,
reconstructed or rescored. This supplies no comparative retention result.

The necessary output fix now uses compact bounded JSON in the existing paired
CLI, preserving every value and its original caps. A literal fixture shrinks
9147386to1765993bytes with exact parsed-value equality on both platforms.
That repair does not recover the lost report or prove those unavailable bytes
would fit. All original failures, charges and earlier negative results remain.

Full commands, failures, caps and receipts are in
`artifacts/runs/balanced-recurrent-replay-20261025/RESULTS.md` and
[issue133](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/133).
[Issue134](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/134)
is the scoped output repair; the scientific comparison criterion remains open.
No circadian advantage, generalization or repository-wide readiness is claimed.

Next: [135/R2.1-PCU](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/135),
qualify the existing isolated fresh-task probe with this concrete recurrent
state before adding any missing binding. No new native experiment is authorized
by that software task. Historical plans/log remain preserved; current handoff
belongs to GitHub issues under AGENTS.md.
