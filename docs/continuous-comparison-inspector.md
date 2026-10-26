# Inspect the saved four-family comparison

```powershell
python -m scripts.build_key_door_inspector --example-comparison --output comparison.html
```

The standalone HTML contains the bundled accepted BP, PC, neutral CPC and active
CPC continuous-task replay contrasts. Open it locally to inspect learning a new
task, retaining the old task and returning to its context, plus isolated fresh-task
probe gains and recorded costs. It runs no learner, simulator or network request.
The output must be new. `--source-artifact PATH` instead accepts an explicit saved
report; invalid input is refused before publication. The two source options are
mutually exclusive. Source checkout and installed wheels use the same example.

All24benchmark and128probe contrasts remain visible, including adverse deltas,
negative gains and both inherited/zero-velocity probe views. Numeric links open
exact saved JSON pointers; root report preserves the full input. Unknown, recorded
null and zero differ. Filters for unrelated legacy curves are hidden; aggregate
contrasts are not reconstructed into fabricated learning curves. A source value
is a recorded claim, not authentication by this viewer.

Replay adds12updates/24row exposures to this run relative to its saved controls;
total observed work includes56updates/116rows with32disposable probe updates.
These are recorded counters, not matched total computation or whole-process
memory. The original work exclusions and incomplete/failed baseline disposition
remain visible. Source README documents original evidence (repository-local).
Final evaluation, training and promotion are unavailable from this display.

The accepted study found stronger old-task endpoints and weaker new-task
endpoints under this fixed replay recipe. It does not establish a CPC advantage,
generalization, robot success or uniform repair of fresh-task learning. Full
release, native/GPU setup and individual asset permissions remain separate gates.

## Files and extension

```text
src/app/continuous_comparison_inspector.py    pure saved-result projection
examples/continuous_comparison/              exact summary and provenance notice
scripts/build_key_door_inspector.py           existing exclusive HTML exporter
```

Why this: ship already accepted evidence through the existing inspector rather
than rerun a cohort or introduce an experiment framework. Extend the saved schema
explicitly, retaining raw values/directions and source pointers; do not import the
experiment script to render data (it creates training inputs at module scope).

Focused checks (scalar JSON/IO only; no models):

```powershell
python -m pytest -q tests/test_continuous_comparison_inspector.py tests/test_coherent_stream_inspector.py tests/test_taskspace_memory_inspector.py tests/test_key_door_inspector_files.py
python -m ruff check src/app/continuous_comparison_inspector.py src/app/key_door_inspector.py src/infra/key_door_inspector_files.py scripts/build_key_door_inspector.py tests/test_continuous_comparison_inspector.py
python -m ruff format --check src/app/continuous_comparison_inspector.py src/app/key_door_inspector.py src/infra/key_door_inspector_files.py scripts/build_key_door_inspector.py tests/test_continuous_comparison_inspector.py
python -m mypy --follow-imports=silent src/app/continuous_comparison_inspector.py src/app/key_door_inspector.py src/infra/key_door_inspector_files.py scripts/build_key_door_inspector.py tests/test_continuous_comparison_inspector.py
```
