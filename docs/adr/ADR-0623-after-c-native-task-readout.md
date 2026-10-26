# ADR-0623 — Read the learned after-C model in known B/C games

## Context

The bounded third-task study learns C_train and retains A/B TRAIN class accuracy
under capacity2. It also costs more replay/storage and worsens current B/C
confidence. TRAIN accuracy alone cannot prove successful new-task control.
Known A/B games before C already passed; earlier failed native budgets stay spent.

## Decision

Add one exact after-C flag to the existing saved-state evaluator/CLI. Select
only qualified111capacity2state6/108/two27-row heads and use the ORIGINAL native
loop for Bblue/Bred/C_trainblue/C_trainred, no refits. Keep the untrained source,
owned scorer, full snapshot/source/budget/clock/closure guards, public-history
features and all7rawfirstargmax behavior. Old defaults remain. Share only the
plain dataclass conversion already used by the older saved-state mode.

Separate paid109+111source histories10fits180rows, selected native6fits108rows,
and new readout0fits. The native command opens only111JSON, never original TRAIN,
final/probe labels,105fit or109report. No new schema, runner or replay service.

## Alternatives

Inferring native C from imitation scores is unsupported. Refitting the model
or rerunning the A/B baseline wastes accepted work. Fullhead equality permits
compatible105/106known-A reference reuse only while model/encoding/environment
assumptions remain unchanged; it is not newly executed after-C A or C success.

## Consequences

One small native case must be frozen after software qualification. Retain every
outcome and cost, including partial failures; no automatic retry or changed
seed/metric/dose. Known-task usefulness stays separate from heldout layouts,
reliability, general recurring recovery, RL/CPC benefit and release readiness.
After this functional gate, reassess the full roadmap instead of automatically
starting another checkpoint/provenance/capture increment.
