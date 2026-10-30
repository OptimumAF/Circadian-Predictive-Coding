# Explicit minimum-change updates on the generic linear owner

```python
from src.adapters.linear_fit_learner import LinearFitLearner
from src.core.categorical_linear_state import LinearFitPolicy

source = LinearFitLearner(LinearFitPolicy(feature_dim=7, num_classes=6))
private = source.fork()
diagnostic = private.train_batch_minimum_change(released_features, released_targets)
```

The caller still owns admitted TRAIN or isolated probe roles, label release,
exclusive runtime lease, budgets and permanent attempted-work accounting. Arrays
alone confer no role/replay/promotion authority. This operation changes only the
owned learner; keep probes separate from deployed/benchmark learners and final
data. Existing `AttributionStream` can use an explicitly composed learner port;
its role/label/counter behavior is unchanged. No new production stream mode here.

Why: the generic default cold-refits each packet and can discard prior coefficient
directions invisible to that packet. The wide categorical core already implements
minimum-change residual fitting. The shared arithmetic now also supports the
bounded generic model and its existing atomic learner, without widening the wide
or CLM role interfaces or creating another update loop.

With augmented design Z and current one-hot targets Y:

`W_next = W_prior + pinv(Z) @ (Y - Z @ W_prior)`

The fixed intercept/RCOND1e-12/one-hot equal-row objective are unchanged. This is
current-batch least squares with minimum displacement from prior coefficients,
NOT cross-entropy optimization. It preserves prior components invisible to Z;
old predictions can still change when their directions overlap the new batch.
Conflicting old/current labels cannot both be satisfied by this rule.

An unfitted owner makes exactly the ordinary cold fit. `train_batch` ALWAYS remains
the original minimum-norm refit; the new operation is explicit and NONSTICKY.
Forks/restores/snapshots do not set an update mode. Existing policy/snapshot fields,
prediction format, cumulative committed counters and no-history storage stay.
Caller must choose the new port on every intended update. Diagnostic definition
is `minimum_change_one_hot_least_squares_sum_squared_residual_v1`; the value is
current-packet residual squared sum, not lifetime fit/retention or CE.

Pure `core.categorical_linear_fit.update_categorical_linear(prior, x, y)` requires
an exact valid fitted CategoricalLinearFit. Generic current rows1..128/features
1..296 remain; a prior whose last fit used128rows is valid. Wide current rows
1..16/features1..4096 remain separately enforced. Shared private residual arithmetic
does not grant wider public geometry or CLM encoding/role admission.

Warm owner updates detach a full prior snapshot separately from the unchanged
comparison pin, plus feature/target arrays. Success publishes one validated model/
counter replacement. Source/prior/input/RNG or owner replacement refuses publication;
foreign mutations are not silently restored. Failed/lost fits expose the existing
TrainingUpdateFailed scalar work/unknown flags under the new definition. Budget
successful-update counters are not rewritten into attempt counters: the caller
preserves attempted work and a failed existing stream remains stopped.

This is a conventional software update capability, not evidence that native task
retention is repaired. Earlier minimum-norm native outcomes/negative results stay
frozen. A separate prospectively budgeted protocol is required before measuring
the new rule in a native task or comparing model families. No scientific allowance,
CPC/generalization/deployment/release claim or probe-to-replay conversion here.
