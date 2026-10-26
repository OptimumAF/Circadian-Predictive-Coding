# ADR-0635: Original ordinary TRAIN task through existing affine ports

## Context

The ten-update arrived-TRAIN runner supports a fourth context and return, but
the native layout whitelist has no ordinary G source. Reserved F probes and D
final data cannot be converted to TRAIN. Useful native adaptation needs an
original source and an actual completed endpoint, not another capture increment.

## Decision

Declare fixed G_train branch5/unrotated/unswapped/grey, move G-only initial cue to
(1,5) to preserve the left key, and admit original TRAIN role only. Preserve old
presets/teacher/encoder/learner/native loop. Existing workflow API adds distinct
recurring_training=True mode: preflight, original G collection, bundledABC+G
through the existing ten-update stream, then original native evaluator G/A readout.
No new CLI, runner, optimizer, source-loader or replay framework.

Retain existing witnesses, paid-report proofs, actual native endpoint handoff,
zero-update readout authority and failed-prefix/cleanup reporting. Original
TRAIN90s/readout150s budgets remain; whole300s/workstop290s512MiB supervision is
prospective for this new ordinary TRAIN case, not renewal of spent private120.

## Alternatives

Relabeling F or D would break evaluation isolation. Merely rotating/recoloring
old layouts would be removed by canonicalization. Changing defaults or creating
another runner would duplicate working behavior. None is adopted. This fixed
source is chosen before outcomes; no search for a favorable G layout is allowed.

## Consequences

Original ordinary fourth-task learning/retention/return is executable through
existing ports, with162joint replay rows and full collection/diagnostic/native
costs explicit. Classification and complete native gameplay are distinct claims.
One known-task observation is not reliable recovery/generalization/CPC benefit
or release readiness. Software self-review is not independent acceptance; all
failed attempts and unmet broad parents remain.

See [usage](../ordinary-fourth-task.md) and
[evidence](../../artifacts/runs/ordinary-fourth-task-20261024/RESULTS.md).
