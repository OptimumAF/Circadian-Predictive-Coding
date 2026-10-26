# Fixed conventional CLM head learning

`scripts/run_clm_head_adaptation.py` composes the released frozen CPU encoder,
existing CLMHeadLearner and ExperienceInbox. It addresses the native learning
prerequisite before consolidation; there is no new orchestration framework.

## Fixed diagnostic

Encode three public hand-authored TRAIN queries and three candidate descriptions
once under the pinned plaintext recipe. Train password/parcel for four updates,
refund/parcel for four, then password/parcel for four. Parcel is a shared anchor;
password is absent during B. Pair alignment already supplies positive-label
information; delayed group-code receipts gate updates. The native paired loss
uses two-way in-batch negatives, while readouts rank all three candidates.

The released heads and encoder remain frozen. Only separate learner-owned
head/scale copies use clipped AdamW/OneCycle; projection dimension remains512.
Raw frozen features are reused under exact identity and head projections are
recomputed. Four unscored probability matrices are saved at0/4/8/12, then scored
after the fixed training dose. No final/probe data, early stopping or selection.
The unchanged released control is scored from one same-feature readout.

Report per-task and aggregate accuracy/CE (existing target-probability clip at
1e-12), full probabilities, update receipts, physical work, sampled RSS and time.
Endpoint success, new acquisition, retention and actual loss/recovery have
separate flags. Already-correct answers are not newly learned. These are known
TRAIN examples: a successful endpoint would not prove generalization, useful
closed-loop behavior, calibration or a circadian advantage. Negative results
remain valid without changing the frozen corpus/dose/metrics.

## Modules and extension

```text
scripts/run_clm_head_adaptation.py   fixed outer composition and CLI
tests/test_clm_head_adaptation.py   tiny stream, scalar scores and refusals
```

No change to existing learner/encoder defaults or dependencies. Keep separate
learner/control ownership and permanent costs when extending the protocol. New
comparisons require prospective bounded protocols; do not turn development
results into final-evaluation inputs or silently reuse changed-head projections.

## Commands

With the prepared optional WSL environment and verified assets, the CLI accepts
`--model-dir`, `--head`, and `--output` exactly as the
[frozen encoder example](clm-frozen-encoder.md#run-using-prepared-local-assets).
Replace its script with `scripts/run_clm_head_adaptation.py`; refuse existing
output paths. This reproduction description does not grant a new model run.
No download, service, GPU or quantization is performed.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_clm_head_adaptation.py
.venv/Scripts/python.exe -m ruff check scripts/run_clm_head_adaptation.py tests/test_clm_head_adaptation.py
.venv/Scripts/python.exe -m ruff format --check scripts/run_clm_head_adaptation.py tests/test_clm_head_adaptation.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent scripts/run_clm_head_adaptation.py tests/test_clm_head_adaptation.py
```

Session execution must follow the
[frozen protocol](../artifacts/runs/clm-head-adaptation-20261022/PROTOCOL.md)
and [independent review](../artifacts/runs/clm-head-adaptation-20261022/REVIEW.md).
The preceding132/128 head-baseline failure remains preserved and unchecked.
This new integration does not retroactively renew its allowance.

## Observed scoped result

The sole reviewed native case completed within budget. Learned correct counts
were1/3→3/3→3/3→3/3 atbaseline/A/B/return; frozen released control stayed1/3.
Password was acquired and retained, and parcel corrected. Refund was already
correct beforetraining; no newrefund acquisition is demonstrated. Passwordnever
becamewrongafterA, so no recovery is demonstrated. AggregateCE .91751→.17213→
.01865→.01595 on these known TRAIN strings is separate from held-out usefulness.

25tests/platform and scoped lint/format/type checks passed, with original two
annotation failures and existing scheduler warnings retained. Sole native work:
1encoderforward/126paddedslots,12headupdates24rows,19lifetimeheadpairs,
34headforwards;230.466139s wrapper and sampledRSS15,785,938,944B (nottruepeak).
[Full evidence](../artifacts/runs/clm-head-adaptation-20261022/RESULTS.md).

The next implementation is rectangular frozen-feature readout for the existing
temporal-rule corpus, keeping native paired updates/defaults strict. That corpus
can supply genuine rule/tool changes; this already-correct refund question cannot
establish newtask acquisition. No further experiment budget follows from the
completed known-example stage. FullR5.1/R5.3/G5/release remainopen.


The historical rectangular interface next action above is now completed; the
[temporal-rule pilot](clm-temporal-sequence.md) has also executed. Its current
critical path is full-catalog TRAIN diagnosis of the persistent invoice rule,
with the known-example result and budget unchanged.
