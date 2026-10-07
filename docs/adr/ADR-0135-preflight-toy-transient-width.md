# ADR-0135: Preflight transient circadian width in toy runs

## Context

The NumPy circadian core has an intrinsic `max_hidden_dim`, but the toy
execution budget had no independent run-level capacity ceiling. Sleep applies
selected splits before selected prunes. A sleep can therefore temporarily
grow from width 12 to 14 and finish at width 12. Checking only its final
width would miss the allocation and violate a run-level cap. A final external
neuron proposal is another possible growth path outside sleep.

## Decision

Add an opt-in positive `max_hidden_width` to `ToyExecutionBudget`, outside the
scientific `ExperimentConfig`. It covers the adaptive circadian hidden layer,
including its transient split width, not baseline model widths or RSS. The
core checks the entry width and `entry_width + selected_splits` after its
existing split/prune decisions and before mutation. The same check applies
to an external circadian proposal before it splits. An over-cap proposal
raises `HiddenWidthLimitExceeded` with the proposed width and cap; the toy app
turns this into an incomplete stop, normally at the checked `before_sleep`
cursor. The app checks initial width before any wake update and checks the
restored current and historical peak widths before continuing a checkpoint.

The budget session records actual current and peak width separately from a
rejected proposed width. A checked checkpoint's current width comes from its
validated circadian snapshot; its historical peak is reconstructed from
initial width and applied split counts in validated sleep telemetry. The CLI
records observed and checkpointed current/peak widths separately. Older
version-1 run states without these additive fields remain resumable only when
their original checkpoint hash, cursor, and prior identity fields match.
The unbudgeted call paths keep their original argument shape.

## Alternatives

- Use the core's intrinsic `max_hidden_dim` as the run cap: rejected because
  changing it changes model/checkpoint identity and split selection.
- Check final sleep width or a checkpoint alone: rejected because split and
  prune can cancel after transient growth.
- Clamp the number of splits to fit the execution cap: rejected because it
  would change the scientific decision instead of stopping the run.
- Count all three toy models under this cap: rejected because this limit
  specifically bounds circadian structural adaptation; the shared process
  memory ceiling is separately scoped under P5.5d3.

## Consequences

An over-cap proposal produces no partial result or final-test access. A
resume may raise its run-level cap without changing model settings. A new
cap below the historical observed peak stops before more training even if
the current width is now smaller. The cap does not bound replay work or
process RSS; P5.5d1 handles the former and ADR-0136 handles the latter.
No baseline, seed, metric, fixed-v14 artifact, or old scientific
result is changed.
