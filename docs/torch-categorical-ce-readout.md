# Disposable TRAIN CE readout and typed decisions

Use this development control when an owned native BP representation exists and
the output layer needs a bounded TRAIN fit. It leaves the supplied learner,
original SGD optimizer/gradients/counters and RNG unchanged. It does not promote
or replace a deployed model. The caller owns label release, actor guards and
wall/RSS limits, including failed fitting work.

```python
from src.adapters.torch_categorical_ce_readout import (
    CategoricalCEReadoutPredictor,
    fit_categorical_ce_readout,
)

fit = fit_categorical_ce_readout(source, train_cases, spec, check_boundary=check_budget_and_actor)
reader = CategoricalCEReadoutPredictor(fit)
result = reader(public_request)
print(result.selected_id, fit.final_loss, fit.work, reader.work)
```

Supply only exact, already released TRAIN `DecisionCase` records with their fixed
schema. Every role is checked before labels, encoding or optimization; selection,
probe and final cases are refused. A role string alone cannot prove label arrival.
The source is pinned before input copying and every external boundary callback.
Actual complete state and optimizer binding are checked rather than a replaceable
public snapshot callback. Changed source/input/RNG evidence is refused and retained;
there is no restoration to hide it.
The caller must serialize owner operations; these Python guards are not a
concurrent-access protocol or a security sandbox.

The fitter reads native CPU float32 tanh features once and starts detached output
weights from the supplied source. It fits the same full-class mean cross-entropy
with the same `(H+1)*K` coefficient capacity. Hidden parameters stay fixed. One
temporary Torch LBFGS step uses fixed lr1/max_iter32/max_eval48/history_size10,
strong-Wolfe line search, gradient tolerance1e-7 and change tolerance1e-9. A hard
64-closure ceiling precedes paid forwards/backwards. These constants have no
caller tuning interface. No random initialization, LP warmstart or new dependency.

This is a separately costed output-learning control, not the original whole-head
SGD update. `fit.work` names attempted/returned hidden reads, optimizer construction
and step, CE forwards/backwards and final TRAIN read. `closure_losses` and final
TRAIN logits/loss are descriptive. `CategoricalCEFitFailed.work` retains attempted
work when no fit returns; label rejection occurs earlier, before numeric work.
An optimization return does not guarantee that every TRAIN decision is correct.

The reader copies one complete source checkpoint and fitted coefficients/schema.
It has no training, restore, publication or promotion interface. Storage is constant
in request count; it keeps counters rather than an unlimited success-record chain.
It reads float32 hidden features and output logits, then shares the original native
typed decoder's allowed-logit softmax and lexical tie rule. Do not use its private
native learner's original output as the fitted readout. Public `reader.work` is a
detached copy of canonical paid counters. Corruption and equal-value owner/work
replacement latch a fault; prior attempts stay paid and foreign changes remain.

The [software qualification](reports/torch-categorical-ce-readout-20261030.md)
covers small planted D2/H3/K3 TRAIN and typed fixtures. A subsequent
[frozen native case](reports/ticket-ce-readout-native-20261030.md) solves all four
known TRAIN tasks, three applied steps each, after34 CE gradients/408 auxiliary
TRAIN rehearsal rows. The original source's matched tasks still fail. This is a
known-task control with extra optimization; unseen adaptation, retention/recovery,
generalization and CPC advantage remain open. Preserve original baseline/SPENT
evidence and count this extra learning before any later comparison.

The next private-probe handoff needs an explicitly initialized trainable native
fork; this reader intentionally has no fork/train interface. Keep source/SGD/RNG
unchanged, record auxiliary initialization costs and use independent private
callbacks rather than charging probe work to the original source.

The existing fitter also accepts the fixed joint old+fresh TRAIN18 cases from
`joint_old_fresh_teacher`. One [native joint qualification](reports/ticket-joint-ce-native-20261030.md)
classified18/18 rows and solved all six known workflows using one frozen fit.
Its612 auxiliary gradient rows include408old rehearsal and204fresh TRAIN.
This establishes joint TRAIN usefulness, not chronological acquisition, recovery,
generalization or CPC benefit. The ordinary --train-readout ce command currently
uses old12/four tasks; a default-off joint cohort option remains unfinished.
