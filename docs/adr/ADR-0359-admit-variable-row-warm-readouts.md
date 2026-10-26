# ADR-0359: Admit explicit native rows for a distinct warm reward readout

## Context

The repaired MiniGrid warm route completes136updates/1274rows. The existing
evaluator only admits original BC checkpoints and nine rows per update. Reward
episodes have variable lengths, so applying that original law would reject the
real checkpoint or require changing its counters. Saved JSON lacks native weights.

## Decision

Add a distinct recurrent_warm_reward method to the existing evaluator, with
explicit native_training_examples metadata:128/1152 or136/1160..1280. Require
matching actual snapshot counters and exact original TRAIN A/B/C triple/six
148/K7 sources. Keep original methods, default fields and final gates unchanged;
old methods reject nonNone overrides. Reuse existing raw/live/fresh-session/
partial-cost/cleanup behavior, and add only the new report's native row field.

Why this: it removes a demonstrated evaluation-port dependency without changing
the learner, native objective, counters or metrics. Metadata does not prove
receipts: the caller continues to bind completed work, state, budgets and evidence.

## Alternatives

Relabeling the warm reward checkpoint as old recurrent BC conceals its objectives.
Changing training_examples to9*136 corrupts actual exposure. A new evaluator
duplicates owned fork/session/cleanup logic. A generic counter override broadens
old/final routes unnecessarily.

## Consequences

The port can evaluate owned snapshots before teardown. It does not itself add
CLI composition, new native evidence, plasticity or benefit claims. Later
composition must charge isolated readout work, preserve completed training
budgets and evaluate only after all updates. Two endpoints do not prove context
return recovery or causal reward effect. Preserve every negative and open parent.
