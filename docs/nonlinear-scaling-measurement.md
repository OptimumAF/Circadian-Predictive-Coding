# Isolated nonlinear raw scaling measurements

The opt-in runner measures the [local nonlinear mapping](coupled-pc-scaling.md)
using owned raw BP, PC, neutral CPC and active CPC learner ports. The matched
[raw BP control](scaled-matched-backprop.md) shares initialization and all weight
and bias update factors. This does not establish a published nonlinear theorem.

## Run

```powershell
.venv/Scripts/python.exe -m scripts.run_nonlinear_scaling_pilot --describe
# Separate prospective legacy-only extension; does not include original cells:
.venv/Scripts/python.exe -m scripts.run_nonlinear_scaling_pilot --legacy-only --describe
# One budgeted scientific invocation, only after correctness and source review:
.venv/Scripts/python.exe -m scripts.run_nonlinear_scaling_pilot > pilot.json 2> command-resources.json
# The legacy extension requires its separate frozen allowance:
.venv/Scripts/python.exe -m scripts.run_nonlinear_scaling_pilot --legacy-only > legacy-pilot.json 2> legacy-command-resources.json
```

The frozen protocol is `nonlinear_raw_scaling_fixed_pilot_v1`: 96 cases, widths
2/8, hidden depths 1/3, fan-in/mean-field, seeds 23/41/59, four methods and three
arrived updates each. Data, rates, inference caps, CPC policies, metrics and seed
order are constants. There is no CLI parameter search. Limits: CPU1, 288 updates,
60 seconds, observed process RSS256MiB and output16MiB. The CLI sets numerical
thread counts before NumPy imports. Do not rerun failed rows or select settings
using the final scores.

The first [retained report](reports/nonlinear-scaling-measurement-20261015.md)
contains all 96 completed fan-in/mean-field cases. The separate legacy-only
protocol adds 48 cases and 144 updates with the same settings, data and metrics.
The [complete report](reports/nonlinear-scaling-complete-20261015.md) combines saved
rows into the full 144-case matrix without rerunning the original models. Neither
study demonstrates circadian benefit. The default protocol remains unchanged.

`planned_keys` and `summarize_nonlinear_scaling` accept the exact original pair,
legacy singleton or canonical all-three tuple via keyword `schemes`. Reordered
groups, arbitrary subsets and duplicate/missing cases are refused. The saved-data
combiner checks each origin's protocol, exact data and prescribed keys before
building all 672 summaries. It retains failures and origins, with no learner calls.

## Isolation and measurements

`src/app/nonlinear_scaling_measurement.py` validates explicit train/final roles,
unique printable IDs, disjoint storage, float64 data and bounded geometry. A
trusted learner snapshot controls its own internal allocations; geometry is
bounded before further wrapper copies/reconstruction. The application forks and
owns complete raw state, then uses the real actor/shadow delayed-label stream.

An optional checkpoint observer in `src/app/coupled_attribution.py` receives
independent state copies. The initial observation has no batch or labels; a
target-free probe uses first training features. Later probes use only the last
arrived training batch. All training finishes before any saved checkpoint is
scored on final data. Callbacks cannot mutate saved scoring state or future
training batches. Failures retain known work, states and diagnostic failure rows,
but expose no partial final scores. Invalid public inputs raise before execution.

Diagnostics include raw/effective parameter norms (including biases), activation
and logit RMS, relaxed latent RMS, true ungated gradient norms, native objective
IDs, task loss/penalties, actual floating factors, tensor bytes and CPC chemistry.
PC uses a fresh fixed two-step diagnostic probe, not equilibrium; BP has only
feedforward activations. Diagnostic gradients differ from applied CPC gated
updates. Postupdate probes are separate from preupdate native training receipts.

`src/app/nonlinear_scaling_summary.py` retains all planned cases and complete
three-seed vectors. Incomplete groups/pairs have null complete means/intervals.
Pair directions are PC-minus-BP, neutral-minus-PC and active-minus-neutral.
Intervals are nominal marginal95% t intervals, df2/critical4.303, conditional on
these fixed data/seeds; constant differences have no estimated interval. They
are not multiplicity-adjusted or population generalization guarantees.

## Cost and limitations

The completed nominal workload is 1008 training objective calls plus 720 probe
calls. Counts do not include every copy, failed attempt or arithmetic operation;
they are not FLOPs or equal computation. Total wall/RSS includes orchestration,
factories, probes, scoring and report assembly. JSON report telemetry stops at
assembly. A separate stderr CLI receipt extends measurement through actual JSON
encoding, stdout and sampler shutdown; its wall clock also covers numerical
imports. Transient import RSS is unobserved. RSS is sampled process memory, not
an exact allocation ceiling or model-only memory. Native tensor bytes exclude
copies and Python overhead. An external command timer covers full subprocess
startup. Nonzero exit means an incomplete pilot or final command resource gate.

Three updates on tiny synthetic data cannot establish convergence, broad width
stability or comparative circadian benefit. All outcomes, including negative and
incomplete ones, must remain in the report. Extend this runner only with a new
prospectively declared protocol and the same role/state/failure controls.

## Focused checks

```powershell
.venv/Scripts/python.exe -m pytest tests/test_nonlinear_scaling_measurement.py tests/test_attribution_checkpoint_observer.py tests/test_nonlinear_scaling_pilot.py tests/test_coupled_attribution.py -q -o addopts= -p no:cacheprovider
.venv/Scripts/python.exe -m pytest tests/test_nonlinear_legacy_pilot.py -q -o addopts= -p no:cacheprovider
```

This command is a useful future regression selection, not a request to rerun
already accepted engineering cases or the scientific pilot. Exact session checks
and retained failures are in `artifacts/runs/nonlinear-scaling-measurement-20261015`.
