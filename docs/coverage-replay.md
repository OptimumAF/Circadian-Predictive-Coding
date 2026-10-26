# Declared training-context replay

The existing four-phase runner has three additional explicit controls:

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_attribution --scenario torch-key-door-four-phase-coverage --describe
.venv/Scripts/python.exe -m scripts.run_coupled_attribution --scenario torch-key-door-four-phase-recent48 --describe
.venv/Scripts/python.exe -m scripts.run_coupled_attribution --scenario torch-key-door-four-phase-partitioned48 --describe
```

Omit `--describe` to execute one local case. Each case permits 256 wake updates,
128 replay updates and 1,024 replay transition exposures. The original `recent`
and `partitioned` controls retain their 16-transition settings and results.
All three new controls cap encoded storage at 48 transitions/12,096 array bytes.
Partitioned replay divides that ceiling into 24 recent and 24 reservoir rows;
duplicates across partitions can reduce unique retained content.

Coverage keeps the latest complete released training episode for each declared
key, in order A-red, A-blue, B-red, B-blue, C_train-red, C_train-blue. Every second
wake update, it selects one eligible key after the preceding selected key,
wrapping and skipping unseen keys. A replacement does not reset that cursor.
Unknown contexts are refused. Final and probe packets cannot enter this path.

Why this: prior recent replay selected only blue cues and both selectors failed
to retain B/C live skill. Fixed coverage makes training exposure explicit. It
uses extra declared task/cue information; equal capacity and update counts do
not make this an information-matched comparison or isolate a reservoir effect.
No schedule, seed, metric or baseline was selected from new final scores.

Structure:

```text
src/core/coverage_replay.py          bounded encoded episode retention/selection
src/app/stream_rehearsal.py          released-update admission, native charges
src/adapters/four_phase_replay.py    fixed controls and prospective protocol
scripts/run_coupled_attribution.py  existing CLI dispatch
```

The store owns its arrays and preserves the existing episode codec/content IDs.
The codec itself does not prove physical provenance: admission of train role and
exact completed arrays belongs to the existing rehearsal boundary. Encoded array
bytes exclude Python metadata, temporary copies and event records; total process
RSS and wall time account for those separately. Reports expose selected episodes,
eligible keys, cursor, receipts, committed work and permanent exposure charges.

Extend this through the existing policy boundary with a separately declared
training-only key set and adequate capacity. Do not infer keys from final data.
Full R4.2 retention/recovery/generalization and release readiness remain open.

## Executed evidence

211 focused cases and affected Ruff/format/Windows+Linux mypy pass. Sole frozen
three-arm case completes within original budgets; independent SOURCE/GO/ACTUAL accepted.
Final pooled B/C success coverage1/4 vs0/4 ordinary controls, mean return0 all.
Coverage gains B-red success while losing original B-blue success; C is acquired
then forgotten, D always fails. One fixed seed/extra metadata/no CPC attribution.
[Complete costs and signed results](../artifacts/runs/r42-coverage-replay-20261016/RESULTS.md).
