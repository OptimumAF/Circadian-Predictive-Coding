# Install and check the affine workflow

The base package needs Python 3.11+ and NumPy. It supports bundled TRAIN data,
command descriptions and saved-result inspection. Playing native games also
requires the pinned `minigrid` optional dependencies declared in pyproject.toml.

For an ordinary local wheel installation in a new environment:

```text
python -m venv .venv
<venv-python> -m pip install /path/to/circadian_predictive_coding.whl
<venv-bin>/circadian-affine-game --example-affine-play --describe
```

Use the actual wheel filename. An offline installation must also supply a
compatible NumPy wheel, with `--no-index --no-deps`; this project does not bundle
NumPy. For native play, install the local wheel's `minigrid` extra from your
approved package source or local dependency cache. The checkout alternative is
`python -m pip install -r requirements-minigrid.txt`.

The play command now checks the existing exact runtime versions before reading
TRAIN data or creating a learner. Missing or mismatched packages return a failed
JSON report and exit 2 before training. Existing output is never overwritten.
`--describe` does not require the optional runtime. The saved inspector also
works in a base-only environment:

```text
<venv-python> -m scripts.build_key_door_inspector --source-artifact saved-run.json --output saved-run.html
```

The result file is explicit user input; inspecting it does not restore weights,
train or run a simulator. A failed preflight report can be inspected in the same
way. Unknown outcomes and absent work records stay unknown.

The bounded local qualification uses a fresh Linux Python3.12 virtualenv,
cached NumPy1.26.4 and the current project wheel, with no system site packages,
`.pth`, PYTHONPATH or checkout imports. It verifies actual installed entry points,
bundled TRAIN decoding, saved-result export and missing-runtime refusal with
zero fits or physics. [Commands, executed evidence and limitations](../artifacts/runs/affine-clean-install-20261024/RESULTS.md)
record whether that qualification completed; no broad clean-clone, optional
native-install, full-suite or release claim follows from the base-only check.

Earlier [native learning/play evidence](portable-affine-workflow.md) and
[browser playback checks](affine-workflow-inspector.md) retain their original
scope. Installation correctness establishes no circadian or generalization benefit.

The [current A4 package qualification](reports/a4-installed-package-20261025.md)
built a new wheel and installed it with cached NumPy into a fresh base-only
Linux environment outside the checkout. The actual console command
`circadian-affine-game --example-affine-a4 --describe`, bundled TRAIN decoding,
zero-model workflow dispatch, exclusive output and missing-runtime refusal all
passed. All180 loaded project modules came from the new environment. No fits or
simulator steps ran; the [fresh A4 native result](reports/a4-fresh-workflow-20261025.md)
remains separate checkout evidence. Install the pinned `minigrid` extra before
using the fresh learning/game command. This does not qualify a clean clone,
fresh dependency resolution, optional native installation or a full release.

The newer [deployment package qualification](reports/deployed-a4-installed-package-20261025.md)
includes the current-actor transition and `--example-affine-deploy-a4` command.
Its fresh offline Linux base installation passes the actual console descriptor,
installed-only imports, zero-model dispatch/exclusive output and missing-runtime
refusal before data or budgets. Packaged TRAIN availability is checked by metadata
only; no payload is decoded. The private/published/fresh-A4 descriptors remain
available. This package check preserves the earlier Windows test-time overrun
and missing native witness; it does not qualify installed native learning.

The [retained rehearsal installation check](reports/retained-rehearsal-installed-20261026.md)
qualifies a newer local wheel with the fixed private rehearsal commands:

```text
<venv-bin>/circadian-affine-game --example-affine-a4 --context-linear-retained-rehearsal diagnostics --describe
<venv-bin>/circadian-affine-game --example-affine-a4 --context-linear-retained-rehearsal games --describe
```

Both descriptions work in a fresh base-only offline Linux installation outside
the checkout. Default/controller descriptions, registered dispatch and exclusive
output also pass. Without the optional runtime, execution exits2 with failed
empty prefixes, unknown outcomes and install help; no data decoding, learner
construction or simulator work occurs. This existing fresh workflow creates its
original zero-update/replay budget before checking the runtime. No budget is
renewed and no training/replay charge is made.

The check performs no native learning or private games. Install the pinned
optional dependencies and budget a full run before executing either mode.
Rehearsal diagnostics still include the ordinary source games. The earlier
native comparison remains failed/spent; installed command access establishes
no joint-retention, CPC, generalization or full-release benefit.

The [installed acknowledgement check](reports/affine-acknowledgement-installed-20261027.md)
qualifies a newer local wheel containing the ordinary correct-feedback option:

```text
<venv-bin>/circadian-affine-game --context-linear-reference drift --ready-feedback --acknowledge-correct --describe
<venv-bin>/circadian-affine-game --context-linear-reference drift --ready-feedback --reference-retain-arrived --acknowledge-correct --describe
```

Both descriptions, unchanged ordinary defaults and early invalid-option refusal
pass in a fresh base-only offline Linux installation outside the checkout. No
training or payload decoding ran in that installation check. The separate
[native command observation](reports/affine-acknowledgement-native-20261027.md)
uses the source checkout and records four fits54rows plus two fresh ACKs; it
does not establish installed learned execution, improved efficiency, game
success, circadian benefit or release readiness.
