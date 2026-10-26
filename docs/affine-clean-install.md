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
