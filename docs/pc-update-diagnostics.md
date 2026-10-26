# TRAIN-only realized PC update diagnostics

## Explicit H32 composition

Add `--wide-neural` to the original explicit PC/neutral joint catalogue command:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method pc --distinct-probe --old-task-readouts --return-to-a --joint-catalogue --wide-neural --pc-update-diagnostics --describe
```

This pure preview constructs no model or environment. Removing `--describe`
executes the full pilot and requires its own prospective local experiment budget.
Use `neutral_cpc` for the neutral route. Only original H32 F16/A16 is admitted;
replay, F48, games, weighting and other diagnostics refuse. Original H8/False
behavior remains. The recipe metadata declares D296/H32/K7/seed17,9735parameters,
38940array bytes, LR.03/momentum.9/S2eta.04; full policy/clocks stay exact.

Two TRAIN catalogue reads at0/128 add108projected rows/255744arrayB,
216prior-forward rows/108backward rows/324relaxation rows, two temporary candidate
parameter sets and0learner/optimizer/published updates. Source/input/RNG/paid
preceding evidence and original budgets remain guarded. Validation/copies/guards
and process costs are additional. Final/probe packets never enter the readout.

525scripted checks and scoped statics pass. One native PC case preserves270TRAIN
probability rows/128receipts/32F/A checkpoints. Neutral native execution is not
claimed for this increment. Both local proposals descend; residual gradients
remain nonzero and no cause or benefit is established.
[Findings](reports/h32-pc-train-diagnostics-20261018.md) ·
[Commands/costs/failures/review](../artifacts/runs/h32-pc-train-diagnostics-20261018/RESULTS.md).

The readout explains what one existing ungated PC update would do on supplied
TRAIN rows. It measures the direction after latent relaxation, while the existing
prior-gradient readout measures the prediction model's CE gradient. Neither
readout teaches a task, chooses a method or establishes why acquisition failed.

## Files and boundaries

```text
src/adapters/torch_pc_update_diagnostics.py       bound admission and readout
tests/test_torch_pc_update_diagnostics.py        independent math and state checks
tests/test_torch_pc_update_native.py             actual native assignment oracle
src/adapters/minigrid_gradient_readouts.py       grouped paid checkpoint composition
src/adapters/minigrid_learnability.py            existing pilot/API and budget guards
scripts/run_coupled_attribution.py               optional public CLI flag
tests/test_minigrid_pc_update_route.py           independent routing/guard checks
tests/test_minigrid_pc_update_native.py          complete native public route checks
```

The adapter reuses the complete invariant/native snapshot copier, TRAIN episode
validator, canonical public-history encoder and pure gradient-geometry reducer.
No runner, learner, objective, native checkpoint format, dependency or environment
variable is changed. Opt-in MiniGrid CLI composition is described below.

## API

```python
from src.adapters.torch_pc_update_diagnostics import diagnose_pc_update

# An existing owned invariant PC or neutral-CPC checkpoint, and original TRAIN
# episodes already encoded as causal initial/current public histories (296/K7).
report = diagnose_pc_update(checkpoint, train_history_episodes)
assert report["status"] == "complete"
assert report["work"]["published_updates"] == 0
print(report["prior_ce_change"], report["gradient_delta_geometry"])
```

Inputs require an exact encoding-bound snapshot for `pc` or `neutral_cpc`, and an
exact tuple of1..6 complete distinct TRAIN episodes, each1..16 rows, total at most54.
Probe/final roles, malformed records/arrays/counters, unsupported methods and
inference mode refuse. The encoder validates current observations; this API does
not authenticate historical external weights or the origin of supplied episodes.
Use the existing collected/owned TRAIN sources when composing it into the pilot.

Why only these methods: the readout evaluates the ungated local PC law. Active
CPC additionally changes chemistry/plasticity/reward state and is not silently
treated as its neutral control. BP uses its existing CE/optimizer path. No
baseline is replaced or tuned.

The adapter clones native parameters, enables autograd locally for one prior-CE
gradient, performs the configured finite latent relaxation with detached tensors,
and forms a local candidate using the original manual parameter gradients.
It constructs/steps no optimizer, calls no learner training/prediction port and
publishes no candidate. It preserves complete caller snapshot state, gradient
metadata, Torch/NumPy/Python RNG and grad mode, including `no_grad` callers.
Before/after guards refuse observed caller changes; they do not rewind external
changes. Once native work begins, failure raises `GradientDiagnosticFailed` with
an incomplete scalar report and the available paid prefix. Malformed envelope
or unsupported mode may raise `ValueError` before native work.

## Math and interpretation

For B rows and prior `h=tanh(XW+a)`, the fixed-state objective is

`E(z,theta) = mean_row CE(zV+b,y) + 0.5 * mean_row sum_hidden (z-h)^2`.

The native latent step uses `B*dE/dz`, without a division by hidden width. The
parameter step uses the mean-over-B derivative with the final relaxed z fixed.
Every point0..S reports mean CE, summed-hidden quadratic penalty, E, L2 latent
gradient (native unaveraged convention) and displacement. Local residual/stationarity
is not a promise of convergence for arbitrary finite step sizes.

The candidate delta is **actual float32 assignment minus original parameter**,
including rounding. Global and all four parameter-group geometries compare the
prior-CE gradient with that delta. A negative dot/cosine indicates a local descent
direction; zero norms have `cosine=None` and a reason. This is not the sign of an
already negated gradient. `prior_ce_change` is candidate prior CE minus original
prior CE on these same TRAIN rows; a negative value is lower counterfactual fit
loss, not an observed deployed improvement or selection authority.

`legacy_diagnostic` retains the original identifier and value:
`0.5*(mean_all softmax_output_error^2 + mean_all hidden_error^2)`.
It is not relabeled E. [ADR0021](adr/ADR-0021-training-energy-diagnostic-contract.md)
and existing local-autograd tests already establish that distinction.

## Costs and verified scope

One successful B-row/S-step readout reports one complete snapshot validation,
one canonical projection of B rows (`B*296*8` bytes), two prior forwards/2B rows,
one prior backward/B rows, S+1 relaxed forwards/(S+1)B rows and one candidate
parameter set. Native train calls, optimizer steps and published updates are0.
Snapshot/parameter/autograd/candidate copies, validation and Python records add
memory and computation; projected bytes are not a process memory total. Caller
must admit diagnostic work under the existing budget before pilot integration;
this function supplies no new clock or budget renewal. Measure RSS separately.

Final combined43 tests pass, including independent NumPy dynamics/derivatives,
neutral parity, exact54-row admission, isolation/refusals/failure-prefix/overflow
and two actual native one-step assignment oracles. Scoped Ruff, formatting and
Windows/Linux mypy pass with independent source/actual review. Initial three
type-check errors are retained. The original standalone API increment performed
no scientific run; the new qualified TRAIN replay is described below. The prior
four-method negative study and its missing-probability criterion stay unchanged.

[Exact commands, failures, work and review](../artifacts/runs/pc-update-diagnostics-20261018/RESULTS.md).

## Opt-in joint checkpoint integration

The existing learnability API now accepts `pc_update_diagnostics=True` with the
joint catalogue route and explicit `qualification_method="pc"` or `"neutral_cpc"`.
It refuses action-margin combinations and all unsupported joint modes early.
Omitting the option preserves existing reports, work counts and CLI forwarding.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method pc --distinct-probe --old-task-readouts --return-to-a --joint-catalogue --pc-update-diagnostics --describe
```

Removing `--describe` executes the complete native pilot; admit its experiment
budget before doing so. The existing original budget/clock remains in force.
Only owned source checkpoints0/128 receive one all54-row diagnostic;32/64/96
retain the original six raw scores. `pc_update_diagnostics` is a one-record list
with original source IDs and detached native DTO; its separately named work is
summed at readout, method and report levels. Benchmark/F/return update and score
counts are unchanged. Two successful readouts add108 projections,216 prior-forward
rows,108 prior-backward rows and324 latent-row forwards, plus validation/copies/
guards under original wall time/RSS. No new diagnostic training updates.

The existing grouped readout helper gains optional arguments with old defaults
unchanged. Exact new scalar types and fixed DTO container owners, including native
work/source IDs, are guarded during and after publication. Refusal preserves all
six preceding raw scores and known paid diagnostic prefix. No fresh budget, new
runner or full-state/counter bypass is introduced.

167 scripted/default-regression cases and two complete synthetic PC/neutral public
cases pass on the combined tree; five-path statics and actual CLI describe pass,
with independent review. The two native cases complete384 total updates14976rows,
188 predictions1692rows and four diagnostic calls. These are engineering checks.
[Commands and costs](../artifacts/runs/pc-update-checkpoint-route-20261018/RESULTS.md).

## Qualified real TRAIN replay

One separately frozen PC-only TRAIN replay reconstructed the original fixed head
because the historical report has no native tensor checkpoints. It consumed saved
TRAIN sources and TRAIN readouts only, without repeating F/return/physics. All270
new raw probability rows reproduced the original PC values exactly. At0/128,
all four group directions point downhill; counterfactual CE changes are
-0.0153620243 and-0.00152027607. The final latent-gradient norms remain0.804481878
and2.933402431. Exact acquisition is still0/6 traces.

This supplies no sign-inversion evidence at the observed states, but does not
identify the cause of failed acquisition or prove convergence/benefit. Candidate
probability arrays are not saved; their math scope is the accepted independent
NumPy/native oracle, not a new saved-array recomputation claim. The previous
four-method study stays SPENT/negative and its missing-probability criterion open.
[Frozen protocol, full scalar records and review](../artifacts/runs/pc-realized-train-replay-20261018/RESULTS.md).

The separately named H8 fixed64-step PC/neutral acquisition control is now implemented; historical motivation follows.
Introduce a separately named fixed64-step PC/neutral acquisition control
through the existing generic native/stream/probe ports.64 is the current policy's
fixed maximum, a single finite-relaxation sensitivity endpoint, not a tuned winner.
Preserve the original S2 recipes/identity and exact policy/clock/source guards;
charge the increased latent computation. Test before freezing any new study.
