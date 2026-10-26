# ADR-0590: Test the existing simpler control before more neural tuning

## Context

Contextual BP completes its fixed96updates but fails two of four workflows.
Inputs no longer alias; this does not identify whether capacity, optimization or
class frequency caused the failure. The existing local perceptron and stream can
use the same data without a new framework or additional BP optimization.

## Decision

Share fixed joint teacher preparation in the existing teacher module. Native BP
delegates to that helper; local example adds mutually exclusive joint-context
mode with original8passes,96rows and zero60-coefficient initialization. Keep all
defaults, typed effect rules, original source/RNG/budget and paid-prefix checks.
Read both fixed states only after TRAIN; charge all teacher/task reads separately.

Why this: useful task behavior is the prerequisite for the next adaptation
demonstration, and a simpler supported control is valid. Increasing BP dose or
selecting favorable seeds is unnecessary for this bounded control question.

## Consequences

The sole case fits12/12 and solves all four ordinary/stale workflows, with four
correct revocation denials. Preserve the BP negative. Different models, parameter
counts/dtypes, update rules, total work and runtime environments prevent causal
or equal-compute/efficiency claims. Neither result supports CPC advantage or
generalization. Joint data at outset is not chronological retention/recovery.

Next test a genuine public-context A→B→A stream through the existing recurring
runner, starting fresh zero. Do not pass the jointly pretrained model off as new
acquisition. Reuse known ability evidence, keep every task at every phase and
separate stale-ticket recovery from recovery of a previously lost learned task.

83distinct focused checks/platform, scoped statics and actual accounting/effect
readback pass. Self-review is not independent acceptance or release readiness.
See issue74 and artifacts/runs/context-perceptron-control-20261023/RESULTS.md.
