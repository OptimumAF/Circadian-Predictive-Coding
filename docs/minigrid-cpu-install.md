# Installed FIFO36 game control

The existing conventional count-memory learner can run the fixed recurring
MiniGrid task from an installed wheel. It collects six public TRAIN traces and
uses the same delayed stream/learner/game ports as the accepted development
control. No research archive, source bundle, Torch or pretrained weights are
needed. The default base package remains NumPy-only.

## Install and inspect the protocol

Use Python3.11+ and the built local wheel. The package is not published by this
workflow. In an activated environment, from the directory containing the wheel:

```console
python -m pip install "./circadian_predictive_coding-0.1.0.dev0-py3-none-any.whl[minigrid]"
python -m scripts.run_coupled_attribution --scenario minigrid-action-memory --invariant-fifo-memory --describe
```

The `minigrid` extra declares the existing pins: MiniGrid3.1.0,Gymnasium1.2.3,
pygame-ce2.5.7,cloudpickle3.1.2,Farama-Notifications0.0.6. In a checkout,
`python -m pip install ".[minigrid]"` or the original
`python -m pip install -r requirements-minigrid.txt` supplies those dependencies.
No dependency downloads or installs are performed by describe. It also works
with the base package and all optional game/model imports blocked.

For offline installation, supply compatible cached dependency wheels and add
`--no-index --find-links WHEEL_DIRECTORY`. Dependency availability/resolution is
an installation requirement; this scoped validation uses existing runtimes and
does not claim a fresh dependency install on every platform.

Why this: installed users do not receive the checkout requirements file. An
explicit optional extra gives pip an executable setup contract while preserving
the base package and original pinned environment boundary.

## Run the existing bounded game

After inspecting its resource limits, run the actual fixed case:

```console
python -m scripts.run_coupled_attribution --scenario minigrid-action-memory --invariant-fifo-memory
```

This prints JSON to stdout. Inspect its status, failed prefixes and work fields;
a printed report is not itself a success claim. The unchanged recipe uses
128storage updates/1152TRAIN rows, six teacher episodes,30live episodes and at
most480live prediction/physics steps, with the original180s body/210s parent
and32MiB output allowances. The protocol's resource fields remain authoritative.
Memory is limited to36retained entries/87264logical payload bytes, with retirement
metadata, native/Python copies and RSS additional. Old/new/return readouts remain
separate. There are no neural gradients, finalD/probe access, controller tuning,
rendering windows or physical hardware actions in this control.

The metadata increment (issue94) executed only describe and metadata/error checks.
The subsequent installed native qualification (issue96, below) was incomplete.
Any future delivery execution must be separately capped before models/environments.

## Evidence and limits

The accepted FIFO36 known-game acquisition/retention/return evidence is reused.
That evidence combines the original incomplete 180-second run with separately
paid saved-state readouts. A fresh complete run within the same limit remains
unverified. A snapshot-copy optimization passes focused checks and a fixed local
microbenchmark ([ADR-0611](adr/ADR-0611-validate-fifo-entry-once.md)). A new current
wheel and separately budgeted installed native case (issue96) still reached the
original180s body limit:128updates1152rows completed, checkpoints0/32 completed,
checkpoint64 was partial and96/128 were unrun. Both A32 cue games succeeded in
9steps; full acquisition/retention/return and installed delivery did not qualify.
There is no automatic retry or cap extension.
FinalD failures, neural negatives and the incomplete contextual-head case93
remain unchanged; no generalization, CPC, matched-compute or release benefit is
established. Current package checks/results are in issue94 and
`artifacts/runs/installed-fifo-game-20261023/RESULTS.md`.
The incomplete native delivery's exact commands, costs and immutable failed
prefix are in `artifacts/runs/installed-fifo-native-20261023/RESULTS.md`.

No simulator code/assets or third-party weights are bundled in this wheel.
Optional distributions retain their own licenses/notices. MiniGrid3.1.0's local
metadata says MIT while its shipped LICENSE header is Apache2.0; both observations
are preserved in THIRD_PARTY_NOTICES.md without resolving the discrepancy.

Extension: keep optional pins aligned with requirements-minigrid.txt and runtime
direct-version checks. Test installed describe with optional imports blocked,
inspect final wheel metadata, and preserve the default NumPy dependency set.
Do not add another runner or duplicate TRAIN resource for this existing route.
