# Frozen categorical TRAIN readout diagnostic

`src.adapters.torch_categorical_readout_fit` diagnoses an existing native BP
representation without updating it. Pass an owned `TorchFeatureLearner`, exact
released TRAIN `DecisionCase` records and their fixed `DecisionFeatureSpec`:

```python
from src.adapters.torch_categorical_readout_fit import diagnose_categorical_readout_fit

diagnostic = diagnose_categorical_readout_fit(learner, train_cases, feature_spec)
print(diagnostic.fit.rank, diagnostic.fit.squared_error)
print(diagnostic.training_logits.argmax(axis=1))
```

Use `read_hidden_features(learner, features)` for a label-free native prior read.
Both ports read CPU float32 tanh activations and return detached float64 arrays.
They validate the actual complete owner and optimizer binding rather than its
replaceable public snapshot callback. A private baseline checks the working
snapshot and actual source before further fitting and on exit. Source, optimizer,
gradients, counters, inputs and global RNG changes are refused and preserved;
there is no restoration to hide a foreign change. Python private objects are not
a security sandbox or a concurrent access protocol.

The diagnostic checks **every** case's TRAIN role before encoding labels. The
caller still owns lawful label release and must supply only already admitted
TRAIN data. This API cannot establish arrival history from a role string. Do not
supply reserved probes, final evaluation, selection data or unreleased contexts.

The original hidden representation stays fixed. One existing equal-row-weight
one-hot least-squares solve uses the unchanged intercept and `rcond=1e-12`, then
reads the fitted logits on the same TRAIN rows. The disposable `(H+1)*K` output
coefficients match the original output layer's coefficient count. Float64 LS and
float32 CE have different objectives and precision. There is no promotion,
deployed learner update, optimizer construction or new native training.

One successful call reports one hidden read, solve and affine readout with their
row counts; coefficients, singular values, rank and residual remain available.
Caller must budget wall time, memory, encoding, validation and all attempted
work, including failures before a result returns. No automatic clock or paid
experiment budget is created by this adapter.

A successful readout is a witness of **known-TRAIN** feasibility for those frozen
features. A failed LS fit does not rule out other CE optima. Neither proves
closed-loop usefulness, generalization, retained plasticity or a CPC benefit.
The initial [software qualification](reports/torch-categorical-readout-fit-20261030.md)
does not execute the genuine source12 TRAIN diagnosis.

The subsequent [native diagnosis](reports/ticket-frozen-readout-native-20261030.md)
(#318) ran once: BP TRAIN8/12 versus frozen-readout10/12, routing0/4 versus2/4,
unchanged source/actor/budgets. Two routing errors remain; no fitted task rollout
or generalization claim. The subsequent
[separation witness](reports/ticket-frozen-readout-separability-20261030.md) (#319)
proves an affine readout can separate all12 saved TRAIN rows. Next implement
bounded disposable CE readout learning and qualify all four known tasks through
the existing ports; no native task success is established by that witness.

The [CE readout fitter and typed reader](torch-categorical-ce-readout.md) now have
scoped software qualification (#320), including TRAIN isolation and unchanged
source continuation. The [genuine four-task case](reports/ticket-ce-readout-native-20261030.md)
(#321) now succeeds4/4 after additional joint TRAIN fitting. It does not establish
new-task adaptation or retention without rehearsal.
