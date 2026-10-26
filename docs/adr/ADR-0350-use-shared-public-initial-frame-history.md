# ADR-0350: Share public initial-frame memory across feedforward controls

## Context

The accepted MiniGrid diagnostic retains all negative results. Its recorded E
TRAIN traces contain identical current observations with opposite step2 turn
labels. More updates cannot resolve that deterministic recorded-data ambiguity.
This is not a proof about closed-loop policies, which can take other trajectories.

## Decision

Add an explicit opt-in condition using each episode's first public148 frame plus
its current public148 frame. All four feedforward methods receive the same296
inputs. Reuse the complete-episode codec, native learner ports, existing delayed
stream, owned checkpoint scoring and disposable fresh-task probes. Preserve the
original148 recipes and the recurrent148 control. Admit exactly296 at three
existing dimension gates; retain all other bounds, roles and budgets.

Why this: a bounded causal memory supplies the publicly observed cue without
extracting semantic cue labels, exposing private state, or adding another runner.
The pure online encoder resets per episode and clears old state before rejecting
a malformed reset. The offline transform uses only its own first/current frame;
next-observation evidence never supplies current features. Metadata and outcomes
are preserved as evidence and targets, separate from feature construction.

## Alternatives

More dose or favorable seed selection cannot remove the recorded contradiction.
Using a private cue field would change information access. Replacing every model
with a recurrent architecture would change more than the demonstrated dependency.
Full history or a generic frame-stack framework is unnecessary for this control.

## Consequences

Feedforward parameters increase1255 to2439, versus1319 for the recurrent control.
Derived probe arrays occupy384912 bytes, with193104 original arrays also retained.
Equal rows/updates are not equal memory or computation. This exploratory condition
was motivated by observed TRAIN data; it is not an independent confirmation or a
matched-capacity comparison with the old condition. Run only a separately frozen
small protocol after scoped gates and independent review. Useful live skill,
activation health, retention/recovery and circadian advantage remain open claims.

## October17 recurrent acquisition prerequisite

Joint acquisition still failed with the raw paired recurrent learner; the actual
reserved-E probe reduced CE but its warm17/18 action accuracy did not improve.
Extend native recurrent policy admission explicitly to296 for a separately chosen
public-history control. Keep all existing raw recipes/routing unchanged. Reuse
the encoder, paired array owner and generic stream, without a new orchestration
layer or changing the paired objective.

Initialize only from a validated cold raw paired snapshot: zero added first-frame
input weights, copy original current-frame and remaining parameters. This
preserves the initial raw law within float32 roundoff instead of changing all
random draws with a wider tensor. Refuse trained state rather than erasing paid
clocks or silently migrating momentum. Recurrent capacity becomes2503 rather
than1319 parameters; row matching is not capacity/compute matching. Qualification
does not prove missing memory caused failure or admit a new comparative run.

The existing MiniGrid evaluator now has an explicit paired296 history branch.
It still admits only original148 TRAIN-A packets and original paired checkpoint
labels/clocks; the shared raw `_paired_counter` is unchanged. Existing public
encoder and fresh recurrent sessions supply raw/live features. This avoids a
new evaluator or relabeling source data. A cold native readout verifies the
boundary without training or establishing acquisition; comparative study remains
a separately frozen/reviewed step.

The separately identified fixed0/64 acquisition condition is now implemented at
the existing paired pilot/CLI. It preserves original64updates/128exposures/1152rows,
checks the copied initial law before training, and reuses accepted raw paired
endpoints without completing their failed full-five protocol. The one local case
was negative: jointfalse, same17/18actions/onegoal, smaller CE improvement with
more parameters. No additional dose/architecture tuning follows from this result.
The next integration direction is the roadmap's sensor/classifier demo; original
MiniGrid old/new/return goals remain open and all negative evidence is retained.
