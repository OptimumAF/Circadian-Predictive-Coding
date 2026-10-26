# Fresh-task diagnostics after four-phase replay

Use the existing runner with two disposable E16 probe cohorts:

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_attribution --scenario torch-key-door-four-phase-coverage-probe --describe
```

Omit `--describe` to execute. Explicit `recent48-probe` and `partitioned48-probe`
scenarios are also available; they are not automatically executed together.
The public adapter functions `describe_four_phase_probe_protocol(control=...)`
and `run_four_phase_probe_pilot(control=...)` also accept the original `recent`
and `partitioned` controls. Existing scenarios retain their defaults.

The benchmark still runs A64/B64/C_train64/A64, 256 wake plus 128 replay updates.
Existing reserved E train/test packets are collected separately. Each probe starts
from an owned initial or final checkpoint (wake0/256, native0/384) and uses a
separate original 16-update budget on a disposable learner fork. It never updates
the benchmark, replay buffer, deployed learner or promotion policy.

Why this: the accepted coverage control has a mixed retained-task result, while
new-task learnability at its endpoint has not been measured. Existing helpers
already support this endpoint. One coverage diagnosis re-creates its benchmark
because saved reports have no owned endpoint weights; the deterministic benchmark
projection must match the accepted report exactly. This does not rerun the whole
three-control comparison or select a control using new probe results.

The initial probe must satisfy the complete-cohort contract before any benchmark
wake. One detached initial record is checked before and after each wake admission,
checkpoint scoring and later probe/final work. Changed evidence causes refusal;
completed wake/probe/replay charges remain spent. The existing generic two-probe
predicate shares its unchanged single-cohort checks with this admission boundary.
Final packets remain unavailable until training finishes and both probes pass.

## Costs and interpretation

The complete route permits 416 native SGD updates/3,328 training rows: 384 benchmark
updates plus 32 probe updates. Probes add 16 reserved score forwards/128 scored
rows, 18 teacher episodes/144 physical steps and 10,368 reserved input-array bytes.
They share the original invocation deadline and RSS ceiling. Native snapshots,
optimizer state, Python objects, copies and reports are included in total time
and process RSS, outside the encoded replay-array ceiling.

All reserved E CE, accuracy, per-class, decision and exact-sequence curves at
0/4/8/16 updates are retained. A larger CE gain from a worse starting score does
not alone demonstrate better plasticity. The deterministic E trace domain is not
held-out generalization, and this route does not compare CPC mechanisms.

```text
src/adapters/four_phase_probes.py       fixed prospective protocol composition
src/adapters/train_acquisition_pilot.py existing stream/probe/final orchestration
src/adapters/key_door_learnability.py   shared complete-cohort predicate
scripts/run_coupled_attribution.py     explicit CLI dispatch
```

Extend a future protocol through these boundaries, preserving role separation,
independent charges and the complete preceding-cohort checks. Freeze any new
comparative protocol before execution. Full useful adaptation and release claims
remain separate from scoped implementation acceptance.

## Executed evidence

123 focused cases and six-path Ruff/format/win32+linux mypy pass. Independent
SOURCE/GO/ACTUAL accepted; sole native diagnosis completes within original limits. Every deterministic old benchmark/final report field
matches exactly. Initial E CE1.618712→1.325128, accuracy.125→.5; endpoint CE
3.499030→.402721, accuracy.375→.875. All exact sequences0. This establishes
partial fitting under the fixed dose, not complete task learning or generalization.
[Commands, costs, failures and next action](../artifacts/runs/r21-four-phase-probes-20261016/RESULTS.md).
