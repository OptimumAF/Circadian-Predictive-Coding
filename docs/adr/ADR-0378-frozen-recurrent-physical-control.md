# ADR-0378 — Frozen recurrent control with physical checkpoints

## Context

The mapped quota route supplies equal physical exposure and explicit environment
instances. Existing negative B results do not show whether reward updates improve
on initialized weights. A nonlearning own-policy control is missing. Counting
collection episodes as native updates would fabricate work and collapse stage
identity in existing dictionaries indexed by native update number.

## Decision

Use an explicit frozen option on the existing mapped quota recipe. A zero native
update budget and complete state guards protect initialized weights, optimizer,
counters and source/actor/candidate ownership. Collect through the existing
episode collector; never submit prepared packets to the actor inbox. Validate
collections separately from native commits. Preserve every physical quota and
failure charge. Learner17 and private sampler29 remain unchanged.

Represent five physical stages separately with native index0. A small adapter
composes existing checkpoint scoring/live helpers for one stage at a time;
it preserves partial paid evaluation progress and prior completed stage evidence.
No shared evaluator schema or native objective is changed for old routes.

Why this: unchanged initialization gives a clear nonlearning reference while
retaining actual own-policy data. Physical stage identity describes exposure;
native counters describe real training and must remain honest.

## Alternatives

Zero-rate SGD would still perform gradients and alter optimizer/counters. A
teacher-trajectory control would not collect its own data. Relabeling physical
phases as native updates would conceal the control's defining behavior. A new
runner/framework adds no useful capability here.

## Consequences

Compute/reset costs remain unequal and explicit. Frozen behavior is not a trained
matched baseline and cannot complete full R4.3. Separately freeze/review a small
repeated-instance protocol before comparison; final evaluation cannot select
seeds, metrics, training or probe conditions. Implementation tests and a bounded
engineering qualifier cannot establish useful adaptation or circadian benefit.
