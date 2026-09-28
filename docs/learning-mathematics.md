# Implemented learning mathematics, version 1

This is an implementation specification for the current NumPy toy/continual
models and Torch image heads. It describes the equations the code executes;
it does not assert that every reported `energy` is the objective differentiated
by the update. Protocol, data-role, and fairness definitions are in
`docs/evaluation-protocols.md`. ADR-0021 records the P2.2 decision to
preserve and label the existing diagnostics. P2.3 finite-difference and
test-only autograd checks are recorded in `tests/test_local_gradient_contracts.py`;
P2.6 restricts attribution to the verified shallow control, while the
unimplemented shared deeper formulation remains P2.6a in `DEVELOPMENT_PLAN.md`.

## Notation and common conventions

Examples are rows: `X` has shape `(B, D)`, `B > 0` is the batch size, `W`
maps columns of one layer to the next, and biases broadcast across examples.
`η` is the weight learning rate, `α` the latent relaxation rate, and `K` the
number of relaxation steps. `p` denotes a feedforward hidden prior, `h` a
relaxed training state, `e = h - p` a hidden residual, and `o = q - T` the
output residual. All gradients below use the state and weights computed
before the parameter update unless explicitly stated. The input, labels,
relaxed states, and priors remain fixed while those computed parameter
gradients are applied; the model does not re-relax between parameter
assignments. NumPy uses float64 arrays; image heads initialize float32 tensors.

The NumPy models use binary targets `Y` of shape `(B, 1)`, sigmoid output
`q`, and batch-mean binary cross-entropy

```text
BCE(q,Y) = -(1/B) Σ_i [Y_i log(q_i) + (1-Y_i) log(1-q_i)].
```

The NumPy training entry contract now requires a finite real feature
matrix of the model's input width, a same-row one-column target matrix
with values in `[0,1]`, and a positive finite weight learning rate. Soft
labels inside that interval remain valid. Validation occurs before model
state changes; see ADR-0026 and P2.8a. Torch input validation is described
below; post-update boundaries remain P2.8c.

The displayed NumPy BCE clamps probabilities to `[1e-8, 1-1e-8]`; the sigmoid
also clips its input to `[-50, 50]`. The code's `q-Y` update is the usual
unclipped sigmoid/BCE derivative and is not the literal derivative of these
clamped diagnostics in saturation regions. The Torch heads use class indices
`y_i`, one-hot rows `T_i`, logits `z`, softmax `q`, and batch-mean
cross-entropy `CE(z,y) = -(1/B) Σ_i log q[i,y_i]`. Torch held-out metrics use
cross-entropy summed over examples and divided by the actual example count.
Torch PC/circadian training now requires a nonempty `(B,D)` floating feature
tensor matching the head's weight dtype/device, and a same-row one-dimensional
`torch.int64` class-index tensor with labels in `[0,C)`. It rejects invalid
steps/rates and nonfinite features before any head state changes; see
ADR-0027. ADR-0028/0029 record the guarded post-update boundaries.
At finite logits of magnitude `1e300`, the NumPy wrong-class binary
diagnostic remains near `18.42` because of its probability clamp; Torch
held-out logit CE can instead report `2e300`, while the three-class PC
training squared-error diagnostic is `1/3`. ADR-0030 keeps these
executed metrics separate and records the small positive/negative
binary-tail rounding difference. Accepted split/prune topologies are
checked for aligned widths before the next training step. Both circadian
configuration validators reject NaN and infinity in every numeric field
at construction, including a currently disabled mechanism; their finite
range rules remain unchanged (ADR-0031).
Model constructors also require positive integer input/feature,
hidden, class, and circadian min/max widths before allocating tensors;
NumPy integer scalars remain valid, while boolean or fractional widths
are rejected (ADR-0032).

## NumPy binary models

### Backpropagation MLP

For hidden layers `l=1,...,L`, with `p_0=X`, the forward pass is
`a_l=p_(l-1) W_l+b_l`, `p_l=tanh(a_l)`, and
`q=sigmoid(p_L V+c)`. There are no free latent states. One `train_epoch`
call evaluates BCE on the supplied whole batch, computes ordinary chain-rule
backpropagation, and applies one gradient-descent step:

```text
δ_out = (q-Y)/B
δ_L   = (δ_out Vᵀ) ⊙ (1-p_L²)
δ_l   = (δ_(l+1) W_(l+1)ᵀ) ⊙ (1-p_l²)
∇V = p_Lᵀ δ_out                 ∇c = sum_rows(δ_out)
∇W_l = p_(l-1)ᵀ δ_l             ∇b_l = sum_rows(δ_l)
parameter ← parameter - η gradient.
```

The returned loss is the **pre-update** feedforward BCE. Hidden traffic is
the per-unit mean absolute feedforward activation. Source:
`src/core/backprop_mlp.py`. The toy and continual runners call this method
once per entire training role in each outer epoch.

### Multilayer predictive-coding-style MLP

The forward priors `p_l` are the same as above and are computed once at the
start of the step. Initialize every latent `h_l=p_l`. For each of `K` steps,
compute `q=sigmoid(h_L V+c)`, `o=q-Y`, and `e_l=h_l-p_l`, then update every
latent from the **old** latent array of that step:

```text
g_L = e_L + o Vᵀ
g_l = e_l + e_(l+1) W_(l+1)ᵀ, for l<L
h_l ← h_l - α g_l, simultaneously across layers.
```

The weights and all feedforward priors stay fixed throughout relaxation.
After the final latent step, recompute `q`, `o`, and `e`. The local parameter
updates, each divided by `B`, are

```text
∇V = h_Lᵀ o;                       ∇c = sum_rows(o)
r_l = -e_l ⊙ (1-p_l²)
∇W_l = p_(l-1)ᵀ r_l;               ∇b_l = sum_rows(r_l).
```

All parameters are then stepped by `-η` times these gradients. Earlier
weights receive their own local residual, using the **feedforward**
`p_(l-1)` as input; the code does not backpropagate later-layer residuals
through the full feedforward prior chain during this weight update. The
reported diagnostic is `BCE(q,Y) + Σ_l Σ_i,j e_l[i,j]² / (2 B Σ_l H_l)`
(`numpy_pc_bce_plus_half_mean_all_hidden_error_sq_v1`). It uses the pre-update `q,e`
even though it is returned after the parameter assignments. The extra
top-down term in the latent update lacks a tanh derivative and uses fixed
feedforward priors; a coherent multilayer scalar objective has not yet been
established for this implemented rule. Source: `src/core/predictive_coding.py`.

### Circadian predictive-coding-style MLP

When there are earlier hidden layers, they form only a feedforward chain:
`u_0=X`, `u_j=tanh(u_(j-1) A_j+a_j)`. Let `U` be its last activation, or `X`
when there is no earlier layer. The single adaptive hidden prior is
`p=tanh(U W+b)`; initialize `h=p`. The `K` latent steps use
`q=sigmoid(h V+c)`, `o=q-Y`, `e=h-p`, and
`h ← h-α(e+o Vᵀ)`, with `U,p,W,V,b,c` fixed. Recompute residuals after the
last step. Before circadian scaling, the batch-mean local gradients are

```text
∇V = hᵀ o / B;                    ∇c = sum_rows(o) / B
r = -e ⊙ (1-p²)
∇W = Uᵀ r / B;                    ∇b = sum_rows(r) / B.
```

For earlier feedforward layers, the code propagates `r Wᵀ` backward through
their tanh derivatives and uses the preceding feedforward activation for
each gradient. Earlier layers are not relaxed latent states. This is distinct
from the all-latent NumPy PC path when `L>1`.

After computing these gradients, the model updates chemical activity from
the relaxed `h`, optionally updates a batch-difficulty reward baseline, and
updates an importance moving average. With current chemical value `C_j` and
configured or adaptive sensitivity `s_j`, plasticity is
`G_j=clip(exp(-s_j C_j), min_plasticity, 1)`. The effective learning rate is
`η` times reward scale `R` (`R=1` when reward modulation is off). `G_j`
multiplies column `j` of `∇W`, row `j` of `∇V`, and `∇b_j`; it does **not**
multiply `∇c` or gradients of earlier feedforward layers. All those
gradients still receive the common factor `ηR`. The exact chemical,
sensitivity, reward, and importance options are in
`src/core/circadian_predictive_coding.py`; they are algorithmic state, not
latent optimization variables.

In the default single-chemical mode, activity is
`A_j=mean_i(abs(h[i,j]))` and `C_j ← chemical_decay*C_j +
chemical_buildup_rate*A_j`. Optional dual and saturating modes replace this
accumulator before `G` is calculated. When reward modulation is enabled,
`d=mean(abs(o))`, `R=clip((d/max(baseline,1e-8))^exponent, R_min, R_max)`;
the baseline is then updated by an exponential moving average (the first
baseline is `d`). Importance is also an exponential moving average of
`R*mean(abs(∇V[j,:]))`. These states influence later gating and sleep
decisions but are not differentiated through the current weight step.

The returned diagnostic is `BCE(q,Y) + Σ_i,j e[i,j]² / (2 B H)`
(`numpy_circadian_bce_plus_half_mean_final_hidden_error_sq_v1`) on pre-update relaxed
outputs. Ordinary wake steps then update traffic, age, counters, and replay
memory. A separately scheduled `sleep_event` can change topology, rescale
weights/chemistry, and replay stored labeled wake snapshots. Replay calls the
same training kernel with its own rates and steps: it updates weights,
chemistry, reward/importance, cooldown/prune state, and traffic, but does not
increment wake epoch/age/history or store another snapshot. A sleep attempt
can return early when its trigger or structural budget blocks it. These
operations are outside the per-batch latent/weight equations above.

## Torch multiclass image models

### Backpropagation references

The practical image-level reference uses ResNet-50 features followed by a
linear classifier: `z=f_θ(images) C+c`. It minimizes mean multiclass CE by
Torch autograd and SGD with configured momentum. Its backbone can be frozen
or trained according to the practical route's configuration. The matched
fixed-feature reference instead takes cached features `F` from one shared
frozen backbone and uses a tanh MLP head:
`p=tanh(F W+b)`, `z=p V+c`. It starts with independent tensors identical to
the PC head's tensors for a shared seed, then trains by autograd CE and SGD
with configured momentum. It has no latent relaxation or hidden penalty.
Sources: `src/core/resnet50_variants.py`, `src/app/resnet50_benchmark.py`,
and `src/app/matched_head_benchmark.py`.

### Predictive-coding head

Given fixed features `F`, form `p=tanh(F W+b)` and initialize `h=p`.
For `K` steps compute `z=h V+c`, `q=softmax(z)`, `o=q-T`, and `e=h-p`, then
update `h ← h-α(e+o Vᵀ)`. Features, parameters, targets, and prior stay fixed
through relaxation. After the last step, recompute `q,o,e`, then use

```text
∇V = hᵀ o / B;                    ∇c = mean_rows(o)
r = -e ⊙ (1-p²)
∇W = Fᵀ r / B;                    ∇b = mean_rows(r)
parameter ← parameter - η gradient.
```

These are manual local head updates; no optimizer steps the backbone in this
path. In particular, the image-level PC wrappers' `freeze_backbone=False`
option does not itself introduce a backbone update. The returned `energy`
is the **diagnostic** `Σ_i,c o[i,c]² / (2 B C) + Σ_i,j e[i,j]² / (2 B H)`
(`torch_pc_half_mean_output_error_sq_plus_half_mean_hidden_error_sq_v1`). The
two terms have separate class and hidden-unit denominators. The supervised
term driving relaxation and output weights is `o=q-T`, the derivative of
softmax CE with respect to logits, not the derivative of `mean(o²)`.
Source: `PredictiveCodingHead` in `src/core/resnet50_variants.py`.

### Circadian predictive-coding head

The Torch circadian head uses the same one-hidden-prior relaxation and local
gradients as the Torch PC head. It updates per-unit chemistry from relaxed
activity, optional reward scale, and importance before applying weights.
With its own `G_j=clip(exp(-s_j C_j), min_plasticity, 1)` and reward scale `R`,
it steps `V[j,:]`, `W[:,j]`, and `b[j]` by `-ηR G_j` times their corresponding
local gradients. Output bias uses `-ηR ∇c`. Its diagnostic is the same
`0.5*(mean(o²)+mean(e²))`. A separately scheduled sleep event may change
topology, weight norms, and chemistry. This Torch head has no NumPy-style
labeled replay buffer or replay update. Source:
`CircadianPredictiveCodingHead` in `src/core/resnet50_variants.py`.

## Prediction and evaluation boundary

Training-time `q` from a relaxed `h` uses a target to infer that state.
Prediction and role evaluation never perform target-conditioned relaxation.
The NumPy methods `predict_proba`/`predict_label` recompute feedforward
priors under current weights and threshold probability at `0.5`; toy and
continual accuracy uses those labels. Torch `predict_logits` similarly uses
the feedforward tanh head. Image accuracy is argmax of feedforward logits;
reported validation/test CE is the example-weighted mean of those logits.
The corrected runners keep final-test roles out of training helpers and
reserve them for final scoring. Guard-role decisions and outer validation
selection are separate in their versioned protocols. A training `energy`
must therefore not be read as held-out CE or as a head-to-head comparable
loss across the binary and multiclass tasks.

## Diagnostic normalization and update objective

The NumPy backprop loss is the pre-update batch-mean BCE
(`numpy_binary_bce_preupdate_v1`). The three PC/circadian identifiers above
name **training diagnostics**, not held-out objectives. A vision report's
legacy `final_energy` is only its last training-batch diagnostic; its
`final_cross_entropy` is held-out feedforward test CE. New reports and exports
carry the relevant identifier without modifying the numeric series.

For one hidden layer with circadian scaling neutralized, a useful local
update objective is `J = mean_i(CE_i) + Σ_i,j e[i,j]²/(2B)`. Holding weights
and prior fixed, its latent gradient is `(o Vᵀ + e)/B`; the code omits the
common `1/B` in its latent step, which can be absorbed into that step's
learning rate. Holding the relaxed state fixed, its head weight gradients
match the local formulas above away from NumPy sigmoid/BCE clipping. The
**reported** hidden penalty instead divides by `H` (or `Σ_l H_l`), so its
gradient is not the executed residual term. If an unchanged nonzero error
occupies one unit while other units are added with zero error, its
contribution to the diagnostic decreases as width grows. For a fixed
sum of squared hidden residuals, the P3.2c fixture measures a 0.002
diagnostic drop solely from changing width four to five. Component-mode
adaptive sleep therefore restarts its diagnostic history after an actual
width change; legacy retains its historical series (ADR-0036). Torch also
divides
its output squared-residual term by class count `C`; its logit derivative
contains the softmax Jacobian and differs from the executed CE residual
`o=q-T`. The multilayer NumPy PC top-down term and circadian gates/sleep do
not extend this one-hidden gradient interpretation to a single static
objective. The P2.3a float64 fixtures compare the one-hidden binary and
multiclass latent/held-state parameter partials with central differences and
test-only autograd; actual PC and neutral circadian parameter steps, two-layer
NumPy backprop, duplicated batches, and NumPy chemical-gate scaling also
agree. The P2.3b fixtures also verify the two-hidden NumPy circadian
feedforward-prior chain and actual parameter step, plus Torch circadian
chemical/reward scaling of local partials. A two-layer ordinary-PC fixture
shows that its nonzero lower-latent top-down move is not the derivative of
the fixed-prior diagnostic, whose lower-latent derivative is zero. These are
local contracts, not a gradient claim for the multilayer all-latent update
or the sleep transaction.

## P2.4 local relaxation stability boundary

With one hidden layer, fixed prior `p` and fixed output weight `V`, the
local objective above is strongly convex in each relaxed state row. The
executed update is `h ← h - α B ∇_h J`. For binary sigmoid or multiclass
softmax cross-entropy, a conservative per-row gradient Lipschitz bound is
`L ≤ 1 + ||V||₂²`. Therefore `0 < α ≤ 1/L` is a sufficient small-step
descent condition for this fixed-weight, fixed-prior local objective.
`tests/test_latent_relaxation.py` checks that its explicit `α=0.2` fixtures
satisfy this bound, that every tested local-objective step decreases, that
the gradient norm falls below `10⁻⁴` of its initial value after 60 steps,
and that the actual NumPy/Torch final state matches an independent
test-only autograd trajectory. A zero output drive leaves the latent at
its prior; a `10⁻⁸` drive produces a bounded small displacement. Calls
with one and 60 steps verify the exact configured iteration count; zero
steps are rejected.

For ordinary multilayer NumPy PC, the same test file instead verifies the
specified simultaneous fixed-prior update and a `10⁻⁴` residual-norm
reduction on one small, weakly coupled fixture. It does **not** assert
monotonicity of the logged diagnostic or a global convergence theorem for
the unmatched rule. The bounds above do not cover arbitrary larger step
sizes, changing weights, chemical/reward gates, or sleep.

The training kernels reject nonfinite input data and controls before the
parameter update, and detect nonfinite prior linears, states, or logits
during relaxation. NumPy checks each step; Torch performs a combined final
finite check to avoid per-step GPU synchronization. ADR-0022 records the
decision and its timing/atomicity limits. Broader numerical input and
post-update boundaries remain P2.8. Under P2.8c1, NumPy backprop and
ordinary PC also reject nonfinite existing parameters, forward
intermediates, gradients, staged parameters, traffic, or pre-update
diagnostics before committing a step (ADR-0028). The circadian and Torch
post-update boundaries are guarded by staged circadian and Torch PC
candidate checks under P2.8c2 (ADR-0029); finite extreme-logit and
topology checks remain P2.8c3. A rejected NumPy circadian wake step
restores provisional chemistry/reward state and any active gradual-prune
decay. A rejected Torch head update leaves parameters and adaptive state
unchanged. The Torch candidate reductions share the diagnostic's existing
`.item()` host read but add kernels to the timed training path.

## P2.5 shallow no-circadian control

`CircadianConfig.matched_pc_control()` (NumPy) and
`CircadianHeadConfig.matched_pc_control()` (Torch) make the circadian wake
gate exactly one, reward scale one, and forced sleep a no-op; the NumPy
preset also gives replay zero steps and zero memory. With identical seed,
width, inputs, and rates, a one-hidden NumPy circadian network then has the
same binary local objective and wake update as the one-hidden ordinary PC
network. The Torch circadian head has the same multiclass local objective
and wake update as its ordinary PC head. Chemistry may still accumulate,
but it cannot affect these neutralized updates. The paired local CPU tests
compare initialization, four successive parameter states, feedforward
predictions, traffic, and no-op sleep. NumPy keeps separate diagnostic IDs
even where their shallow numeric values agree.

This equivalence does not extend to two-hidden NumPy paths: ordinary PC
relaxes both latents and uses local earlier-layer residuals, whereas
circadian relaxes only the final latent and propagates its prior gradient
through earlier feedforward layers. A same-seed two-hidden fixture records
their earlier-layer weight divergence. ADR-0023 limits the control claim.
ADR-0024 records the reporting decision: new NumPy outputs identify
ordinary PC as `numpy_pc_all_latent_fixed_prior_v1` and circadian PC as
`numpy_circadian_final_latent_feedforward_prior_v1`; deeper runs are labeled
`numpy_deeper_unmatched_descriptive_v1`. This does not change either update.

## P2.7 cross-backend boundary

In the float64 one-hidden binary/two-class gauge,
`softmax([-z/2,z/2])[1]=sigmoid(z)` and the represented BCE/CE and latent
drive agree. At equal scalar learning rates, two free Torch output columns
move the margin by `-2ηg` while the NumPy scalar column moves by `-ηg`;
hidden updates and matched neutral chemistry agree for one step. Separate
zero-momentum backprop MLP and zero-noise split/prune fixtures compare the
corresponding forward/update and structural boundaries. ADR-0025
limits the result; it is not full production-backend parity.

## Pending mathematical gates

- **P2.6a:** Specify and verify a common deeper formulation before claiming
  matched deeper NumPy attribution. P2.6 currently limits the comparison
  scope while preserving existing numerical algorithms.
- **P2.8:** Validate the remaining shapes, target ranges, model parameters,
  and post-update numerical boundaries across all training paths.

No numerical or benchmark result changes are made by this specification.
