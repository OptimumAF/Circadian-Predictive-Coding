# ADR-0570 — Train a contextual chooser from disposable native trials

## Context

R7.4 calls for a small trained adaptation controller, version-bound counterfactual
snapshots, causal stale-outcome rejection and discarded-work accounting. The
existing native learner already owns complete snapshots/forks; the existing
stream already admits finite wait/wake choices and observes delayed TRAIN errors.
R7.3-A supplies the fixed scalar utility but not learned-policy training.

## Decision

Fit a two-context/two-arm table from four bounded full-horizon disposable trials.
Use the original objective unchanged; one sample per cell, maximum empirical
utility, fixed wait tie-break. Core owns scalar choices only. App binds the
original versioned complete native source, admits exact causal trial returns,
retains failed charges, stops retries and freezes one detached fitted record.

Compose through an opt-in branch of the existing serving CLI and a finite recipe
helper. Counterfactual mode uses the original delayed native admission but cannot
publish; fresh application owners use the unchanged guard/publication/rollback.
Require the existing exact qualified source even for a supplied detached base.
Keep default/reference/outer-study paths and report limits intact. Never read
final/probe sources or old scientific outcome artifacts for fitting/selection.

## Alternatives

A new runner, hashing system or source loader would duplicate existing ports.
Online reward attribution to individual wake decisions would be unsupported by
these delayed whole-task effects. A larger policy/sweep would obscure whether
the small controller can fit and act at all. Selecting an objective after results
or dropping losing trials' costs would invalidate the intended comparison.

## Consequences

The fitted chooser learns different wait/wake preferences and acts through the
working native runtime. All16discarded fit updates remain paid; on observed
applications its choices match the existing simple drift reference. There is no
isolated or amortized benefit. Shared templates and noisy one-sample costs leave
held-out generalization unanswered. Source binding is to the original complete
model, not a permission to rebase arbitrary later versions. Broader empirical
chronology and genuine held-out comparisons remain separate unfinished work.
Single-agent self-review is not independent acceptance.
