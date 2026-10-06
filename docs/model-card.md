# Model Card: Circadian Predictive Coding

## Summary

Circadian Predictive Coding is an experimental family of predictive-coding-style
learners with plasticity modulated by hidden activity and scheduled structural
adaptation. The repository provides binary NumPy networks and multiclass Torch
image heads, alongside backpropagation and ordinary predictive-coding references.

"Chemical" state is a numerical proxy accumulated from mean absolute hidden
activity. "Wake" and "sleep" name training and consolidation phases; their clocks
are completed training calls and runner epochs. The optional scale named reward
uses supervised error relative to a moving baseline. These labels describe
programmed rules for supervised learning. See the
[implemented learning equations](learning-mathematics.md).

Documentation scope: the current checkout at `28e71ee` plus its retained local
changes, reviewed on 2026-10-06. Saved results keep their original source and
environment scope in the [publication register](published-experiment-register.md).

## Intended Use

- Research and educational experimentation with biologically inspired learning dynamics.
- Controlled benchmarks against backpropagation and traditional predictive coding.

## Not Intended For

- Safety-critical production decisions.
- Unreviewed deployment in medical, legal, or financial decision pipelines.

## Model Family and Comparison Boundaries

| Route | Implemented behavior | Interpretation |
|---|---|---|
| NumPy binary networks | Backpropagation MLP; ordinary PC relaxes every hidden layer; circadian PC relaxes the final hidden layer and updates earlier feedforward layers through their prior gradient. | A neutral one-hidden circadian control matches ordinary PC locally. Deeper NumPy PC and circadian updates differ; their descriptive rankings do not isolate a circadian mechanism. |
| Torch `frozen_shared_representation` | One frozen ResNet-50 feature bank; backpropagation tanh MLP, PC and circadian heads start from identical parameter tensors and share cached batches. | This is a head comparison. Equal starting tensors and examples do not by themselves equalize relaxation work, guard scoring, elapsed time or later capacity. Separate fixed-width and resource protocols record their own controls. |
| Torch `unmatched_reference` | Image-level ResNet runner; backpropagation can train its backbone, while PC and circadian use manual head updates on a frozen backbone. | Different trainable representations and head architectures prevent direct mechanism attribution. |
| Torch `end_to_end_backprop` | Separate practical backpropagation reference with a trainable ResNet backbone. | Its image training costs and scores have a separate scope from cached-feature head training. |

NumPy models use binary sigmoid outputs and float64 arrays. Torch heads use
multiclass softmax outputs and initialize float32 tensors. The
[backend capability matrix](backend-capability-matrix.md) records additional
clock, structure, replay and snapshot boundaries. Matching feature names do not
establish numerical equivalence across backends.

### Circadian mechanisms

Wake training can accumulate and decay chemistry, gate local updates, apply
supervised-error scaling, and track importance. Sleep can reset chemistry,
rescale weights through homeostasis, split/prune neurons and, in NumPy, replay
retained labeled training examples. Trigger, budget, replay and structural
settings are configurable and must accompany a result.

`sleep_mode="legacy"` preserves the existing budget-gated behavior.
`sleep_mode="components"` enables independently switched components even when
split/prune budgets are zero. `sleep_mode="disabled"` makes sleep a no-op.
The Torch head has no sleep replay buffer or replay updates. An isolated split's
function-preservation check does not imply that a combined sleep event preserves
predictions: pruning, homeostasis, replay and guard rollback have separate effects.

## Training Data and Information Access

- NumPy studies use generated binary toy data, including two-phase distribution shifts.
- Vision routes use synthetic images or torchvision-backed CIFAR-10/CIFAR-100.
  Cached-feature studies use one resolved random or pretrained backbone;
  pretraining and weight provenance remain part of the experiment specification.
- Corrected guard-separated vision protocols distinguish wake training, a labeled
  inner guard, outer validation and final test. The guard can drive stopping or
  sleep rollback; outer validation selects settings in the declared tuning route.
  Final test is scored after the required training and selection gates, and does
  not set training, stopping or configuration-selection decisions.
- NumPy data access depends on the named protocol. Arrived-role studies construct
  phase B after the required phase-A work; global final-role gates cover their
  declared trial families. Earlier corrected profiles and offline dynamics plots
  have different information access and selection limits. In particular, plotting
  future-phase validation offline does not demonstrate strict-online learning.

Prediction uses feedforward states. Label-conditioned latent relaxation belongs
to training and is not used to compute held-out predictions. The
[evaluation protocols](evaluation-protocols.md) specify exact roles, label arrival,
selection timing, replay exposure and final-release gates for each route. Historical
test-informed outputs and reproduction protocols retain their original labels.

## Evaluation and Resource Accounting

Use the declared held-out metrics: accuracy and cross-entropy, and for continual
studies the specified retention, forgetting and adaptation endpoints. Preserve
the predeclared seeds, comparison directions, uncertainty family and failed or
undefined outcomes. A low training diagnostic alone does not establish better
held-out performance.

PC/circadian `energy` and vision `final_energy` are training diagnostics from
relaxed states. They differ from held-out feedforward cross-entropy; normalization
also depends on hidden width and, in Torch, class count. They are not a common
optimization loss across methods or backends. Exact formulas and local gradient
limits are in the [learning mathematics](learning-mathematics.md).

Resource comparisons must identify their scope: feature extraction/setup, wake
updates, latent steps, replay examples, guard evaluations, sleep attempts and
committed or rejected work. Record capacity trajectories and trainable parameter
counts as well as latency/throughput. Head-training time does not include every
image-pipeline cost. Process RSS and CUDA allocator measures have different
ownership and sampling scopes; a sampled RSS ceiling is a soft observed limit.
Checkpoint and observation costs must be included when the protocol measures them.
See [original outcomes against compute and memory](p610-outcome-cost-presentation.md).

## Recorded Results and Known Limitations

- No universal circadian advantage is established. The corrected seven-seed
  [hardest toy profile](p62-corrected-profile-results.md) has lower mean circadian
  balanced accuracy than PC. It remains a descriptive reproduction of a previously
  tuned profile, with unmatched deeper updates and unequal capacity/work.
- The retained [primary confirmation findings](p612-confirmation-findings.md)
  leave H1–H4 unresolved within their declared primary evidence. Intervals crossing
  zero do not establish equivalence. Derived reports and repeated identical inputs
  do not add independent replications or establish fresh seed/source/runtime provenance.
- Sleep, replay and structural policies can underperform. Inactive components,
  rejected events, null results and negative outcomes remain part of the evidence.
- Historical chart source/per-seed records, some original run records and complete
  historical dependency environments are missing. Current constraints and an earlier
  feasibility request cannot supply another run's original versions. Consult the
  [original publication register](published-experiment-register.md).
- Broader confirmation admission, deeper matched NumPy attribution and repository
  release gates remain scoped by the [development plan](../DEVELOPMENT_PLAN.md).
  The deferred guard repair has not been completed. Local correctness or metadata
  checks do not establish those unfinished scientific requirements.

## Checkpoints and Reproducibility

Full-state snapshots and trusted local continuation exist for supported toy,
continual and Torch routes. The fixed v14 NumPy route resumes complete unscored
trial prefixes, then repeats its global gate before final release. It does not
provide a mid-trial cursor or cross-version checkpoint compatibility. See
[checked v14 resume](v14-checked-resume.md) and
[atomic artifact publication](atomic-artifact-publication.md); atomic directory
visibility does not imply full filesystem crash durability.

[Dependency reproducibility](dependency-reproducibility.md) records the tested
Windows NumPy and CPU Torch environments, installation checks and their limits.
Current dependency constraints and historical saved-environment evidence are
separate records. Preserve protocol IDs, exact source/config/input identities,
environment facts, original artifacts and failed attempts when reproducing a result.

## Ethical Considerations

- No personal data is required by default benchmark workflows.
- Public benchmark claims should include dataset, seeds, and configuration details for reproducibility.

## Maintenance Status

Active research repository; APIs and defaults may evolve.
Use release tags for stable references in external projects, and retain the exact
commit plus local source identity used for any reported experiment. See the
[README](../README.md) for current commands and compatibility routes.
