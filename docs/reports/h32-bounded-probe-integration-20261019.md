# H32 isolated weighted PC probes

The existing runner supports original H32 PC or neutral CPC with two explicit
weighted probe modes. Benchmark learning remains original; F/A updates belong
only to an owned disposable candidate. Ordinary scalar evaluation metrics remain
visible separately from the weighted training objective. This is implementation
correctness evidence, not useful recovery or circadian advantage.

Preview without constructing a learner:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method pc --joint-catalogue --distinct-probe --old-task-readouts --return-to-a --wide-neural --probe-inference bounded-weighted --describe
```

`fixed-weighted` uses the same objective/cap with tolerances disabled. Both retain
safety fallback; fixed is not a guarantee of accepted unsafe steps. Configured
S2 maxima and actual attempted/accepted/rejected iterations, candidate work and
elapsed/storage/process costs are separate. The planned whole recipe remains
192updates/7488rows and94scoreforwards/846rows; this session executed only four
single-update engineering cases, not that full recipe.

New structure: `tests/test_minigrid_wide_bounded_probe.py` owns pure/scripted
admission/isolation tests; native fixtures/receipts live in the stage below.
Reuse existing adapters/app solver boundaries for extensions; do not enable new
methods/objectives or mix replay/games without a separate reviewed recipe.

[Full commands, failures, costs and skipped gates](../../artifacts/runs/h32-bounded-probe-integration-20261019/RESULTS.md) ·
[Independent review](../../artifacts/runs/h32-bounded-probe-integration-20261019/REVIEW.md).
