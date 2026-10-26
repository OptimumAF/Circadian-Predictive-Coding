# ADR-0446: Bind normalized ManiSkill controls to continuous records

## Context

R4.4 requires an actual simulated robot task and working BC before CPC adaptation.
Pure continuous records/CPU BC already exist, while categorical episode records
cannot express the Panda's continuous controller. Actual startup dependencies
must be distinguished from fake boundary checks.

## Decision

Add an optional infra port for pinned ManiSkill3.0.1 PickCube-v1/Panda CPU state
with pd_ee_delta_pos, exact normalized4axis controls and separate flags, plus
a fixed one-episode smoke CLI. Use existing ContinuousEpisode/NativeLearner ports.
Validate actions before quantization, store actual float32 commands, refuse final
role and incomplete/poisoned export. Count successful native returns and failed
physics upper bounds separately. Preserve unmeasured reset costs explicitly.

Why this: PickCube is a real supported task permitted by the roadmap. Explicit
normalization avoids falsely labelling controller coordinates as metres; official
time-limit wrapping avoids autoreset ambiguity. Native startup demonstrated a
missing optional Pinocchio CPU IK dependency despite SAPIEN importing, so check
that requirement before creating scenes and preserve the failed native attempt.

## Alternatives

Do not invent a reach task, relabel categorical actions, substitute toy dynamics,
switch controller to avoid IK silently, use a new runner, or treat zero-action
rollouts as expert training. Do not retry a spent native protocol to erase failure.

## Consequences

Adapter boundary tests and source review can pass while native behavior remains
blocked. Compatible CPU IK installation, full process-tree memory observation,
new prospectively bounded native smoke, expert data/BC quality and eventual
matched adaptation remain open. Code and noncommercial asset terms stay separate.
