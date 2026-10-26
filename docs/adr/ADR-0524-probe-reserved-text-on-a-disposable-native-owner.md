# ADR-0524 — Probe a reserved text task on a disposable native owner

## Context

Native warm/cold/reset/rehearsal controls show context-dependent learning
failures; first-B group norms are nonzero. The reserved-task question remains
open. Disposable native ownership is now accepted on tiny fixtures. Existing
ExperienceInbox already admits probe_train with a fresh exact bounded budget.

## Decision

Freeze one new marker task C before execution. Warm the original model on the
accepted A6 stream and verify exact original/after-A bindings. Fork once, inherit
all paid native history and optimizer state, and deliver six fixed probe_train
exposures to the child through the existing probe-role inbox. Predict C/oldA/
anchor endpoints on the child before and after; score labels only afterward.
The source receives no predictions or probe updates. Exact snapshot/buffer/
RNG/mode/work checks run after copy and after child work, including failure.

Why this: observe useful task-level learnability and retention without changing
the benchmark/deployed owner or creating another runner framework. Reuse the
existing native owner/delivery/metric interfaces and keep every paid prefix.

The typed owner prediction batches candidates together and differs from native
serving Scorer's batching. Therefore fresh endpoints use the same typed path on
both sides; historical serving metrics are not reused. Only accepted state
bindings are reused. No calibration, dose, seed or target selection from results.

## Alternatives

- Updating/restoring the source violates independent probe ownership.
- Repeating observed B/development data cannot create a fresh reserved task.
- Reusing serving endpoint scores assumes unchanged prediction batching.
- A new orchestration framework duplicates the existing inbox's probe contract.

## Consequences

One protocol and one prospective native invocation; no retry/alternate task if
the outcome is negative, incomplete or zero-shot correct. Counts include copies,
source-proof diagnostics, predictions and child deltas from inherited history.
Ordinary trusted frozen-base guards remain; Python/tokenizer overhead and
internal RNG/checkpoint work retain explicit unknown qualifications. New C data
becomes observed evidence once executed and must not later be relabeled fresh.
Task recovery, broad generalization, calibration, circadian advantage and full
R2.1/R5.4/release acceptance remain separate claims.

## Follow-up: same-task cold control

The accepted warm C study failed acquisition and anchor NLL preservation. Add
an explicit cold source history to the same CLI, with zero prior work, empty
AdamW and six-update native cap; retain warm A6/twelve-update defaults. Reuse
the exact accepted warm report and now-observed corpus, recipe and prediction
path. Why this: cold B uses a different task and cannot answer C's history
association. Cold A measures untrained transfer, never retention. Different
prior history and total computation prevent matched-compute or weights-versus-
optimizer causal claims. Use a separately frozen one-run allowance, no warm
rerun, new framework, seed selection or relabeling C as untouched.
