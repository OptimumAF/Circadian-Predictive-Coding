# Bounded replay capacity when C_train arrives

The arrived catalogue now accepts an explicit `retained_per_cue=2` option.
The default remains1. Both retain only previous label-arrived TRAIN packets
chosen by the public initial cue, with FIFO eviction and explicit source spans.
The two-packet bank fits A+B+C rows jointly when C_train arrives; it does not
invent a long trajectory or change the affine objective.

The existing CLI has one fixed third-task study mode:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-new-c --describe
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-new-c --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --retained-source artifacts/runs/arrived-cue-catalogue-20261024/actual.json --output new-c-capacity.json
```

The second command needs both exact accepted local sources and a prospective
study budget. Existing output files and fit overrides are refused. Original
commands/defaults remain available. No new dependency or runner.

Each capacity starts from the same genuine after-B4fit54row model and fits only
C_trainblue/red through the original delayed stream. Native counters keep their
history; local suffix receipts start at0observed/1arrived, separately identified
from historical offset4. This is a new bounded study fork, not resumed consent,
publication or budget authority. Source and stable actors are unchanged.

Capacity1 uses2newfits36rows(18current+18replay); capacity2 uses2newfits54rows
(18current+36replay). The shared prefix is not refitted. The original source's
whole6fits90rows, including unused later work, stays paid and separate. Both
arms finish training before fixed TRAIN diagnostics, including the original
after-B model's previously unmeasured C_train reference. No scores select the
schedule/capacity, tune the learner or access final/probe evaluation.

The bank holds at most4episode references/36logicalrows. Metadata gives complete
retained IDs/arrival ticks and source spans; legacy scalar IDs identify the most
recent prior packet. Numeric-equivalent bank bytes include current and next
observations and targets, exclude reward/metadata/Python overhead, and share
source schedule objects. Full plans, suffix arrays, temporary inputs and process
RSS are reported separately.

## Observed fixed TRAIN outcome

| Model | A correct | B correct | C_train correct | New fits / rows |
| --- | ---: | ---: | ---: | --- |
| Original after-B source | 18/18* | 18/18* | 13/18 | 0 / 0 |
| Capacity1 after C | 13/18 | 18/18 | 18/18 | 2 / 36 |
| Capacity2 after C | 18/18 | 18/18 | 18/18 | 2 / 54 |

*A/B starting scores reuse accepted109evidence without rescoring. The C reference
is measured from an unchanged source copy after both arms finish fitting; it
cannot influence the fixed recipe. This is a TRAIN acquisition/retention result.

Retention has a cost: capacity2 adds18replay rows, while B/C cross-entropy
worsens from1.165422 to1.181246. Its27-row heads have rank20/squared error about
2/3, versus capacity1's18-row rank14/near-zero squared error. Class accuracy
does not imply perfect interpolation or an overall winner. Bank numeric
equivalents increase from85,392 to170,784bytes; suffix input arrays from85,536
to128,304bytes. These exclude other memory; whole sampled peak69,951,488bytes.

The sole outcome used4new fits/90rows,14diagnostic predictions/126rows,
1,176nested projections, no physics/teacher/RNGdraws/otherlearners. Body15.640799
seconds, parent16.219275seconds, no timeout/softstop/retry. Full source, actor,
budget/clock, bytes and callerNumPyRNG guards pass.56distinct checks per platform,
scoped lint/format/types and self-review pass, not independent acceptance.

These TRAIN scores alone do not establish native gameplay. The separate
[after-C native readout](after-c-linear-games.md) now solves four known B/C games
without refitting. Generalization, reliable retention/return, matched-compute
and CPC benefit remain unproved. Earlier failures,
including issue108's different spent native case, remain preserved.

[Protocol](../artifacts/runs/catalogue-capacity-new-c-20261024/PROTOCOL.md),
[results/commands/costs](../artifacts/runs/catalogue-capacity-new-c-20261024/RESULTS.md),
[owning issue111](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/111).

Extend through the existing composer and stream. Keep the capacity finite,
arrivals causal, labels role-separated and old/new work explicit. Qualify a
distinct native task outcome before claiming useful after-C game behavior.
