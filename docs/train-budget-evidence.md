# Original training budgets in failed reports

The existing `minigrid_learning_pilot._train_arm` accepts the explicit option
`record_budget_evidence=True`. Default-off calls preserve their existing schemas,
model behavior, receipts and budgets.

The option retains original limits, clock identity and start immediately after
the training budget is born, before sampler/native callbacks. At training exit,
before model readback, it observes current clock values and committed update/
replay charges. A nested `finally` exports detached `row["budget_evidence"]` even
when the arm never returns or candidate readback fails. If source construction
fails before budget creation, it reports `born=False` with no invented clock.

Why this: the issue143 diagnosis stopped at an underestimated optimizer limit.
Its caller never received the arm, so the original budget clocks were unavailable
in its JSON. The ten committed receipts survived. This repair supports future
callers; those missing historical clocks remain unknown.

The evidence distinguishes original limits/start from observed values and reports
session/config/clock/start/limit binding discrepancies. Invalid non-native or
nonfinite observed values are null with named invalid fields. It does not repair
changed runtime state, renew authority or supply a resumable budget. The existing
guards still refuse changed state/configuration. Changed or replaced exit records
also refuse through the completed guard before more work. One bounded scalar
record is retained, without a success-history chain or model tensors.

Sixteen distinct scripted cases qualify on Windows/Linux, including full32-step
streams, lost returns with two committed receipts, unavailable model readback,
source-initialization failure, changed clocks/start/config/limits, modified or
replaced report evidence, option/type/prior-evidence refusal and pre-change
default schema parity. Scoped Ruff/format/fresh mypy default and Linux-target
checks pass. No real Torch/model/SGD/physics/SVD/data-file work ran. This is
single-lead self-review, not independent/native/release acceptance.

Exact commands, original limits and outcomes:
`artifacts/runs/train-budget-evidence-20261025/RESULTS.md`.
The issue143 model experiment stays FAILED-SPENT, with no health readout or retry.
