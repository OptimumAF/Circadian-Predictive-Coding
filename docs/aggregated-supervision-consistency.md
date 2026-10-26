# Grouped TRAIN supervision consistency

Inputs:2..6 complete distinct named TRAIN episodes,1..50 rows each/total<=198,
shared normalized four-axis action spec/full observationsD<=64. The pure core
`diagnose_grouped_input_targets(groups)` composes the existing100-row kernel.
Outputs: all intended global pairs once, raw/native32 deltas/conflict flags,
source identities, deterministic nearest indices and actual repeated work.
No model, learner update, RNG, source role conversion or IO occurs in core.

Outer script `diagnose_aggregated_supervision` admits the fixed accepted198 study
plus its original correction file via the existing public loader. Both files
remain byte-bound through publication; incomplete diagnostics retain paid prefixes.
It grants no new training, collection, probe or evaluation authority.

```text
python -X utf8 -B -m scripts.diagnose_aggregated_supervision FRESH_OUTPUT.json
pytest -q tests/test_grouped_input_target_consistency.py tests/test_aggregated_supervision_diagnosis.py
python -m ruff check src/core/grouped_input_target_consistency.py scripts/diagnose_aggregated_supervision.py
python -m ruff format --check src/core/grouped_input_target_consistency.py scripts/diagnose_aggregated_supervision.py
python -m mypy --follow-imports=silent src/core/grouped_input_target_consistency.py scripts/diagnose_aggregated_supervision.py
```

Use the actual diagnosis only under its prospective protocol/budget; the output
must be fresh in an existing directory. All19,503 supplied pairs were checked:
three identical Push-reset pairs, zero raw/native32 target conflicts. This rules
out exact contradictions only; it does not prove learnability or useful control.

Why this: better correction fit still failed native control and lost old-task
retention; inspect concrete label ambiguity before changing the learner. Preserve
the100-row kernel cap and compose bounded blocks rather than extending its scope.
New applications should use their own explicit data authority and prospective
costs, keep complete groups and all outcomes, and preserve failed-call work.
No new dependency/environment/configuration is required.

[Protocol, full pair evidence, commands, costs and review](../artifacts/runs/aggregated-supervision-consistency-20261021/RESULTS.md).
