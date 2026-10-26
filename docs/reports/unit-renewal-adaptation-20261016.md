# Fixed renewal setting: no units qualified

The new existing-CLI renewal/learnability scenario passed its integration checks
and one prospectively frozen run. All nine scheduled events found no eligible
units under age≥6 and outgoing gradient EMA≤.001. **No replacements occurred.**
The threshold was left unchanged; this run does not evaluate efficacy after reset.

The original seed23 A→B→A and reserved sixteen-step C probes were reused.
Only three renewable PC-family arms executed (246updates/1968rows). Accepted
BP/PC/neutral/active no-renewal controls were read only after the new run.
Shared initial parameters, every common benchmark report/outcome, all72new probe
points and activation health exactly match retained same-method controls.

| Observation | Result |
| --- | --- |
| Learning B | Positive B gain in all3 arms; retained BP has larger local gain. |
| Retaining A during B | All3 show positive A loss; no improvement from inactive renewal. |
| Returning to A | All3 recover A; retained BP has larger local return gain. |
| Fresh task C | All12 curves improve at8;9 trained16-step endpoints are negative. All points retained. |
| Renewal cost |9native no-op operations,0charges/units; native history192arrayB/model. |
| Total run cost |54wake+192probe calls,1476energy,96predict,24SVD; .853456s/RSS42909696B. |

The simpler no-renewal path produces the same trajectories here. This is not
general equivalence or evidence against every renewal policy. Probe headroom,
unequal BP computation and different profiled cohorts limit broader comparisons.
There is no circadian advantage claim.

Source/probe isolation, permanent allowances and original defaults pass78focused
tests and scoped Ruff/format/WinLinux mypy; independent SOURCE/GO/ACTUAL accepted.
Initial annotation failures are preserved. Raw levels, curves, history, commands
and costs: `artifacts/runs/r22-renewal-adaptation-20261016/RESULTS.md`.
API/CLI: [unit renewal](../unit-renewal.md).

Next investigate the existing replay policy's long-stream retention/exposure
criterion using the same stream framework; do not loosen this frozen threshold.
