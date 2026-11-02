# Installed CPU ticket workflows and optional native heads

## NumPy-only paired minimum-change workflow

The base package includes the conventional linear-fit workflow; Torch is not
needed for this command. In an isolated Python3.11+ environment, install a local
project wheel and its declared NumPy dependency, then run from any directory:

```text
python -m pip install /path/to/circadian_predictive_coding-0.1.0.dev0-py3-none-any.whl
python -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --paired-serving joint --train-update minimum_change --output paired.json
```

Retain `paired.json` as the recorded trajectory. The current source includes the
ordinary paired projection in `src.app.context_linear_workflow_inspector_projection`:

```text
python -m scripts.build_key_door_inspector --source-artifact paired.json --output paired.html
```

Open `paired.html` locally to inspect recorded outcomes, model identities, costs
and original source pointers. Saved JSON inspection requires no new learning;
running the workflow does. The first #349 development wheel predates this
projection and refuses the report; use a wheel built from the updated source
or run from this checkout. Do not edit the saved definition/status or pretend
it is a reserved probe to bypass an old inspector. Output files are exclusive: choose new
names rather than overwriting paid evidence.
A failed run is incomplete evidence, even when a partial report was saved.

The fixed development workflow performs twelve source fits on72rows and six
child fits on48rows, including old TRAIN rehearsal exposure. It uses ordinary
arrived TRAIN labels, not final/probe labels. `minimum_change` selects the
explicit incremental least-squares update; omitting it preserves the cold-fit
default. Existing source30s/child60s clocks, consent, receipts, guard and rollback
remain active. No historical checkpoint file, native extra or simulator is needed.

An accepted development case retained four old tasks and learned the previously
failing v2 password task through old-context return. The v1 parcel task was
already successful at source and receives no acquisition credit. Published
model16 and later private candidate18 are distinct; rollback returns to source12.
No forgetting was observed for old tasks, so this is retention through return,
not recovery after forgetting. All failures/refusals remain in raw live outcomes.
These are fixed known templates, not held-out generalization, matched compute,
CPC advantage, dependency-range or release qualification. The separate CE probe
qualification remains failed under its original clock; this command does not
resume that run or renew its allowance.

The module entry points already ship in the wheel. Installing or inspecting
reports does not authorize extra experiments or change default learner behavior.

## Optional Torch heads

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
paired-serving modes support the conventional perceptron and linear_fit recipes,
not the native BP/PC/CPC methods. See the NumPy-only command above for the
explicit paired minimum-change mode.

The fixed275/276/278development experiments are complete/SPENT, including the
negative ready-action/task results. Installing these options does not authorize
repeating those experiments, tuning seeds/dose or claiming useful native
acquisition, retention, recovery, CPC advantage or release readiness. See
`docs/reports/context-class-balanced-20261029.md` in the source checkout.

The279installed qualification uses a fresh project wheel/environment and an
explicitly reused existing Linux Torch2.14.0+cpu SDK. Every project module must
come from the new environment, never the checkout. This is installed project
compatibility, **not a fresh Torch dependency installation qualification**.
