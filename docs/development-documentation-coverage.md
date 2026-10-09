# P9.4 documentation coverage

Current readback: 2026-10-06, HEAD `182077545d12d880e918f73cbf142c2279c211da`
with the preserved working changes. Earlier document review headers name their
recorded commits; current source/document identities are bound separately by the
local `artifacts/runs/r31-learner-ports-20261006/p94-coverage-entry.json`.

| Original requirement | Current guidance | Accepted review scope |
|---|---|---|
| Learning equations | [Learning mathematics](learning-mathematics.md), NumPy/Torch kernels and diagnostic definitions | P9.4c; all original equations retained, completed P2.8 gates and unresolved deeper/backend limits explicit |
| Biological-inspired terminology | [Model card](model-card.md), summary and circadian mechanisms | P9.4a/c; chemistry, reward, wake/sleep explained as programmed numerical rules |
| Algorithm variants | [Model card](model-card.md), model-family table and [README](../README.md) workflows | P9.4a/e; ordinary backprop/PC/CPC and components/legacy/disabled sleep distinguished |
| Benchmark tracks | [Evaluation protocols](evaluation-protocols.md), [model card](model-card.md) | P9.4d full revalidation/e; binary, frozen shared representation, unmatched image and practical backprop scopes remain separate |
| Dataset information access | [Evaluation protocols](evaluation-protocols.md), [model card](model-card.md) | P9.4d/e; arrived roles, inner guard/outer selection/final release, dataset construction versus feature/scoring access and historical test-informed routes explicit |
| Resource accounting | [Evaluation protocols](evaluation-protocols.md), [outcome/cost presentation](p610-outcome-cost-presentation.md), [README](../README.md) | P9.4d/e/f; wake/latent/replay/guard/sleep/rejected work, capacity, setup, elapsed/RSS/CUDA scopes and soft sampling limits retained |
| Backend capability differences | [Capability matrix](backend-capability-matrix.md) | P9.4b/c; six rows, distinct objectives/shapes/clocks, NumPy-only replay/v14 and conditional parity preserved |
| Compatibility and migration | [README](../README.md), [evaluation protocols](evaluation-protocols.md), [capability matrix](backend-capability-matrix.md) | P9.4d full revalidation/e; current checkpoint formats, historical format 7 versus current 8, trusted same-environment trial-prefix continuation and unsupported cross-version claims explicit |

Architecture/contribution guidance was independently reconciled by P9.4f;
reproducibility/dependency boundaries by P9.4h; current runtime and original-release
interpretation by P9.4i. The latter preserves the negative foreign-exit witness and
user-deferred owning-entry repair. Later native port additions are documented in
[learner ports](learner-ports.md), with existing kernels and protocol identities intact.

The coverage readback binds eight accepted receipt families and current complete
document bodies. It does not rerun historical readers, certify missing original
facts, validate remote CI, or infer scientific completion. Historical tables,
equations, results, seeds, metrics, tolerances, compatibility and failed budgets
remain preserved. P9.4g's code dependency debt, P9.5/P9.6, deeper learning work
and scientific/source/native/runtime admission are independent unfinished tasks.

Why this: completion of the documentation requirement requires every original
topic to be explained at its actual implemented scope. It does not require those
documented unresolved research mechanisms to become successful results.


## Dependency status at closeout

R0.3/G0 has reopened for the repair PR remote CI diagnosis. R3.1 remains unchecked
although its new software ports pass local targeted tests and both type targets.
The existing owner retains that diagnosis; no dependent experiment or additional
port implementation is launched. Earlier baseline/document receipts keep their
recorded scope. See [the active roadmap](../RESEARCH_ROADMAP.md).
