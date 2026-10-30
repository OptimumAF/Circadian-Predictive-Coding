# Owned LS trials in the existing live catalog

Use the normal workflow to run a fixed private catalog action while the original
actor continues serving:

```text
python -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --catalog-action replay6-wake1 --catalog-context mistakes --output NEW.json
```

Both catalog options are required. Actions are wait, wake1 or replay6-wake1;
contexts are clean or mistakes. This mode requires ordinary owned LS context
heads and excludes paired/private/adaptive/incident modes. Existing commands
retain their defaults. Output must be new. This is a fixed counterfactual trial,
not a trained action chooser or deployment command.

The parent workflow keeps its original12update/72row source, sampler, budgets
and lifetime guard active. run_owned_case constructs the trial from that current
complete snapshot/digest with context-ls-12, full base and explicit LS family.
The accepted [source/replay binding](context-linear-catalog.md) verifies actual
source/candidate history before authentic ordinary TRAIN bank construction.
No saved Perceptron/probe weights are relabeled as LS.

The original14opportunity live timeline continues to serve the source actor.
The private candidate runs the selected catalog action using original shared
admission and independent wake/replay budgets. Existing selected-wake
continuation can finish a deferred replay→wake without charging another replay.
No private candidate is published. Source/actor/RNG/budget origins stay binding.

Before/after readouts use private frozen LS actors and the same six known
controller_train workflows, with the original64typed-attempt allowance. Their
typed/native reader work is retained even on a failed readout. Perceptron defaults
use the original source and decoder. Task counts do not choose or tune the fixed
action; final/probe data remain unavailable. Extra readout/candidate/source/
replay/worker costs are separate from task utility.

The low-level run_case existing catalog_trial/catalog_policy ports also require
an original owned LS source guard/sampler when _linear_fit=True. This adds no LS
saved-source loader or policy rebasing. A fixed trial does not establish learned
selection, meta-generalization, circadian advantage or release readiness.

See [executed evidence, exact commands and limitations](reports/context-linear-live-catalog-20261029.md).
