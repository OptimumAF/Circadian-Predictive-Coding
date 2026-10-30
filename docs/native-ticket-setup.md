# Installed native ticket and weighted BP support

The optional `native` extra declares Torch using the same `>=2.2,<3.0`
compatibility range as the existing ResNet requirements. Base NumPy installation
is unchanged. Native array/ticket heads do not need torchvision or a simulator.
The whole dependency range is not qualified by a single installed check.

Install your supported Torch build using your platform's normal procedure, then
install the wheel with its `native` extra in a dedicated environment. For example:

```text
python -m pip install "circadian-predictive-coding[native]"
python -m scripts.recurring_ticket_workflow_example --help
```

For a local wheel, use `"/path/to/circadian_predictive_coding-0.1.0.dev0-py3-none-any.whl[native]"`
instead of the package name. This is a local development package; no public
registry availability is implied. CPU/GPU Torch selection and fresh SDK setup
are separate platform requirements; the ticket learner currently executes on CPU.

The recurring API accepts native `bp`, `pc`, `neutral_cpc`, `gated_cpc`, with
`context_return=True` for the D9 public-context stream. Optional
`record_train_fit=True` reads fixed TRAIN diagnostics. `class_balanced_bp=True`
requires ordinary BP/context-return/replay none and weights only each current
arrived TRAIN batch; it does not expose final or probe data. Separate-head and
paired-serving options remain perceptron-only.

The fixed275/276/278development experiments are complete/SPENT, including the
negative ready-action/task results. Installing these options does not authorize
repeating those experiments, tuning seeds/dose or claiming useful native
acquisition, retention, recovery, CPC advantage or release readiness. See
`docs/reports/context-class-balanced-20261029.md` in the source checkout.

The279installed qualification uses a fresh project wheel/environment and an
explicitly reused existing Linux Torch2.14.0+cpu SDK. Every project module must
come from the new environment, never the checkout. This is installed project
compatibility, **not a fresh Torch dependency installation qualification**.
