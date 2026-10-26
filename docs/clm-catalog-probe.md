# Disposable catalog-objective CLM probe

`CLMCatalogProbe` is an isolated warm-weight learner for future fresh-task
diagnostics. It admits an exact `CLMCatalogLearner` source and copies its head
weights into separate modules/storage, with fresh AdamW/OneCycle state and cleared
gradients. Source identity, policy, modes, gradients, optimizer/scheduler, private
RNG, logical progress and permanent work remain intact. The probe retains source
progress metadata, not a reference to the source learner.

## Boundaries and roles

```text
src/core/clm_catalog_probe.py          explicit probe_train catalog record
src/adapters/clm_catalog_probe.py     catalog-specific source/feature admission
src/adapters/clm_head_probe.py        shared owned probe lifecycle and limits
src/adapters/clm_catalog_learner.py    shared indexed-catalog tensor admission
tests/test_clm_catalog_probe.py       isolated ownership/role/failure behavior
```

`FrozenProbeCatalogBatch` carries identity, query/candidate features and ordered
candidate IDs, with role exactly `probe_train`. `CLMCatalogTargets` supplies one
index per query; repeated target indices are valid. Exact probe records reach
the shared catalog validator directly, without conversion to benchmark TRAIN.
The catalog learner still admits only `FrozenCatalogTrainingBatch`; paired
learning and the paired probe retain their original exact source/record kinds.

The probe uses the unchanged `clm_full_catalog_cross_entropy_v1` objective and
atomic native update engine. Candidate rows and logit values stay permanently
charged, including failed paid work. A paid update/readout failure stops the
owner. Readouts use existing `FrozenProbeChoiceBatch` roles `probe_train`,
`probe_test` or `outer_selection`; final evaluation is not an admitted role.

The public surface contains `train_batch`, `predict_choices`, metadata, progress
and a copied work ledger. It has no snapshot, restore, export, installation,
fork or paired-prediction interface. Private Python objects are trusted; this is
an ownership boundary, not containment of hostile Python code.

## Use and validation

Construct from an existing catalog learner, with a fresh probe policy:

```python
from dataclasses import replace
from src.adapters.clm_catalog_probe import CLMCatalogProbe

probe = CLMCatalogProbe(
    source,
    policy=replace(source.policy, max_updates=6, max_restores=0),
    max_readouts=6,
)
```

All other policy settings must exactly match the source. The source may have
spent its own training allowance; probe work does not renew that allowance.
Probe policy admits 1..16 updates except 10, where the original OneCycle setup
has a singular phase, and requires zero restores. Readout allowance is 1..64.
Choose data, consent and independent orchestration budgets before using the
probe through the existing inbox/learner-step ports.

```powershell
python -m pytest -q tests/test_clm_catalog_probe.py
python -m ruff check src/core/clm_catalog_probe.py src/adapters/clm_catalog_probe.py tests/test_clm_catalog_probe.py
python -m ruff format --check src/core/clm_catalog_probe.py src/adapters/clm_catalog_probe.py tests/test_clm_catalog_probe.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/clm_catalog_probe.py src/adapters/clm_catalog_probe.py tests/test_clm_catalog_probe.py
```

The [reserved-task runner](clm-reserved-probes.md#optional-full-catalog-objective)
now composes this boundary through `--catalog-training --reserved-task-probes`.
The standalone boundary's correctness protocol grants no pretrained experiment;
integration/native observation has a separate frozen protocol and gates.
Native task usefulness, retention and recovery remain unproved; all prior
negative results remain. See [protocol and evidence](../artifacts/runs/clm-catalog-probe-20261022/).

The scoped Windows/WSL cohort passes221cases/platform, with lint/format/types and
independent software acceptance. A static-only typing failure is retained. The
executed input meter omitted the second module's seeded RNG-state reserve; its
raw receipt remains preserved and a separate conservative correction gives43670
elements within the unchanged131072cap. This is source-derived reserve accounting,
not a fully measured RNG ledger. No numerical fixture was repeated.

## Integration

The existing reserved-task runner uses explicit catalog probe records, indexed
labels, consent, independent budget and exact catalog exposure receipts. Preserve
the benchmark learner and completed probe prefixes. Scoring occurs only after every
fixed source/probe update; native observation requires its reviewed small protocol.
