# Reference rehearsal during private corrections

The existing correction adapter and robot runner accept optional fixed TRAIN
reference replay for BP, PC, neutral CPC and active CPC. Eight corrective wake
updates remain the default. With replay enabled, complete reference episodes are
trained as separately charged consolidations between those wake updates. No
serving promotion occurs. Nearest-prototype correction does not support replay.

Why this: a fully student taskspace12 trial collected only ALIGN labels. Its eight
private updates lowered correction-batch loss but damaged original A imitation fit
and still failed the native task. Reference rehearsal makes preservation an
explicit, measurable treatment. Useful control or retention is not guaranteed.
[Preserved negative result](../artifacts/runs/taskspace12-failure-state-corrections-20261020/RESULTS.md).

## Programmatic usage

```python
from pathlib import Path
from src.app.continuous_stream_rehearsal import ContinuousReplayPlan
from scripts.maniskill_continuous_smoke import run_corrective_adaptation

# source and reference_a are already-owned TRAIN model and original TRAIN episode.
# For history inputs, reference_a must already use the exact causal model recipe.
plan = ContinuousReplayPlan(every_updates=2, max_replay_examples=4 * len(reference_a.transitions))
report = run_corrective_adaptation(
    source,
    output_directory=Path("new-exclusive-output-directory"),
    visual_observation=False,
    readiness_corrections=True,
    acquisition_only=True,
    action_history_recipe=recipe,
    replay_plan=plan,
    reference_episodes=(reference_a,),
)
```

Native execution needs its own prospectively reviewed local protocol and budget.
The example is not an experiment authorization or a qualification of its inputs.
The direct private update API accepts the same two replay keywords plus its usual
arrived batch/now arguments. Omit both keywords to retain old behavior. A bank
without a plan is an error. No new CLI flag or environment variable is introduced.

Banks are exact tuples of 1–4 complete TRAIN episodes with distinct IDs, one
shared spec equal to the frozen model spec, and bounded rows/bytes/exposure.
Default retained caps are128 rows/1MiB; maximum exposure allowance65536. Inputs
are copied before private/native work. The caller supplies existing model-space
episodes; replay does not refit statistics or silently project raw observations.
History validation checks actual-action causal tails, not authenticity of dropped
raw coordinates or expert qualification. Final/probe/guard episodes are refused.
The bank and replay plan never reach rollout callbacks.

## Counts, ownership and failure

With a20-row reference bank and every_updates=2, eight wake updates add four
consolidations and80 replay-row exposures. A50-label correction batch contributes
400 corrective-row exposures. The candidate therefore adds12 optimizer updates /
480 training rows; the wake budget still records exactly8 updates and0 replay rows.
A distinct original replay budget records80 exposures. This is extra optimization
and information exposure, not matched compute.

The existing ContinuousStreamRehearsal owns the bank, plan, original budgets and
consolidation receipts. Correction admission binds the already-validated native
wake state before helper callbacks; later checks validate preceding success before
new work and before detached release. Source/actor state, clocks, consent, paid
wake receipts, CPC state, native reports and failed replay charges remain bound.
Failed or incomplete runs release no candidate and do not admit a later wake.

Reports expose the helper under replay, applied_consolidations separately from
wake receipts, and reference_rehearsal request metadata in the robot summary.
Work counts include replay optimizer/row/kernel costs and bank storage; scalar
validation, copies, containers and process memory are additional. No total
performance comparison follows from these selected-kernel counts.

## Files and checks

```text
src/app/continuous_stream_rehearsal.py       existing bounded replay owner
src/adapters/continuous_correction_demo.py  optional private correction composition
scripts/maniskill_continuous_smoke.py       collect/update/frozen-return API
tests/test_correction_reference_rehearsal.py four-family behavior and refusal checks
```

Focused commands and evidence: [software increment](../artifacts/runs/correction-reference-rehearsal-20261020/PROTOCOL.md).
The scoped tests use small real learners and fake simulator IO. They establish
software behavior only. No native replay benefit, autonomous success, generalization,
CPC advantage, safety or repository-wide readiness is established by these checks.
To extend, reuse the same fixed-bank contract and preserve separate exposure and
consolidation accounting; do not import later evaluation labels into the bank.

## Executed evidence

The final196 focused cases passed on both Windows and pinned Linux, with scoped
statics and independent combined-patch/actual-evidence acceptance. [Commands and
limits](../artifacts/runs/correction-reference-rehearsal-20261020/RESULTS.md).
A subsequent one-use native BP12 trial added four A20 reference updates80 rows to
the eight correction updates. Both student episodes still failed. A reference fit
was slightly less damaged than correction-only with extra work, but remained worse
than its source. [Accepted negative trial](../artifacts/runs/taskspace12-reference-rehearsal-acquisition-20261020/RESULTS.md).
These results do not establish useful acquisition, behavioral retention or a
matched-compute benefit. Further replay ratios/doses are not the next priority.
