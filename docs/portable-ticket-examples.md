# Portable saved-ticket examples

From a source checkout:

```console
python -m scripts.build_key_door_inspector --example-ticket bp --output bp.html
python -m scripts.build_key_door_inspector --example-ticket pc --output pc.html
python -m scripts.build_key_door_inspector --example-ticket neutral_cpc --output neutral.html
python -m scripts.build_key_door_inspector --example-ticket gated_cpc --output active.html
```

The same commands work after installing the project wheel, from an unrelated
directory without `artifacts/` or the research tree. Use the project's normal
isolated Python3.11+ environment; the distribution declares NumPy, while this
viewer reads only saved JSON and requires no Torch/native SDK. Install the wheel
and declared dependencies with pip before running these module commands.
No new training or comparison is triggered. Open the fresh HTML locally.

For actual fresh candidate learning instead of saved inspection, see the
separately budgeted [fresh BP route](fresh-bp-ticket-demo.md), which requires
an existing CPU Torch environment. It leaves this read-only option unchanged.

Each choice reads a byte-identical original report through the existing bounded
JSON loader, source-cell projection and exclusive exporter. Source checkout
prefers its `examples/ticket_workflow` files; an installed wheel uses
`importlib.resources.as_file` for the whole publication lifetime. Corrupt selected
data fail loudly, never silently switch to another baseline. Output is never
overwritten. `--source-artifact` and `--example-comparison` remain supported;
select exactly one source option. Optional separate costs remain UNVERIFIED.

[Resource provenance and licensing](../examples/ticket_workflow/README.md) retain
all source hashes, original negatives and costs. The JSON includes project-trained
numeric state, not third-party released weights or SDK assets. It is presentation
evidence, not a model loader or execution permission. Full task steps distinguish
observation/effect/result revisions and validated action selections; raw predictor
returns remain pre-validation records in the complete source.

Saved BP learns a new input context while retaining the old tasks; all three
native controls fail full tasks at this BP-informed dose. Equal exposure is not
matched compute/equal optimization. No general method ranking, CPC benefit,
fresh-run demonstration, physical safety or release readiness is implied.

Why this: saved examples should be usable without a developer's local artifact
paths. This reuses existing package resources and inspector, adding no dependency,
runner, network service or model work. Full R4.8 fresh-run/GPU/setup/clean-committed-
clone criteria and other release requirements remain separate open work.

Focused checks:

```console
python -m pytest -q tests/test_ticket_example_packaging.py tests/test_continuous_comparison_inspector.py tests/test_key_door_inspector_files.py
python -m ruff check scripts/build_key_door_inspector.py tests/test_ticket_example_packaging.py examples/ticket_workflow/__init__.py
python -m ruff format --check scripts/build_key_door_inspector.py tests/test_ticket_example_packaging.py examples/ticket_workflow/__init__.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/build_key_door_inspector.py tests/test_ticket_example_packaging.py
```

Scoped validation:72scalar checks/platform and four statics pass. One wheel and
source archive contain byte-exact resources and project notices. A fresh Linux
Python3.12 prefix with only project+cached NumPy1.26.4 passes pip check and exports
all4examples under `-I` from an unrelated directory, with no research tree on its
import path or native libraries loaded. This is working-tree package verification,
not clean committed-clone, Windows fresh-install or whole-release acceptance.
[Exact commands, costs, artifacts and skips](../artifacts/runs/ticket-example-package-20261023/RESULTS.md).
