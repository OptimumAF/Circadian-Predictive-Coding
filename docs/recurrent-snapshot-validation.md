# Recurrent snapshot validation

`torch_recurrent_learner` binds its existing SGD options explicitly in
`_sgd_options`. Both actual optimizer construction and exact serialized group
validation use that configuration. Readonly snapshot validation no longer
constructs an optimizer or clones parameters just to discover its metadata.

The snapshot format and numerical update rule are unchanged. Validation still
checks every group key and native value type, ordered parameter IDs, update and
row clocks, complete finite CPU float32 parameters/gradients/momentum, and owned
copies. Unsupported metadata fails. There is no cache or relaxed state reader.
The explicit options were checked against both cached Torch2.14 CPU runtimes;
new library metadata needs qualification rather than silent acceptance.

Issue137 qualification passed43focused cases on Windows and Linux. It includes
two-update equality with implicit-default native SGD for float rate/momentum
and native integer rate/zero momentum, corruption/refusal, frozen sessions, and
16full copies with optimizer construction and Tensor.clone forbidden. Both
platforms paid24SGDsteps/72training rows, including the independent reference.

Reproduce the bounded selected checks with the original commands in
`artifacts/runs/recurrent-snapshot-cost-20261025/RESULTS.md`. That engineering
allowance is spent; do not automatically repeat its model work. Scoped Windows
Ruff/format/fresh mypy pass. Linux Ruff is unavailable (one failed attempt);
Linux mypy was already known unavailable. No dependency install was made.

This verifies a measured component repair, not whole-game speed, task acquisition
or repository readiness. Issue136's original deadline failure, majority-only
fit and unrun return remain unchanged. No native cohort was retried.

To extend the optimizer contract, change the explicit options and exact metadata
tests together, qualify numerical/state parity prospectively, and keep final
evaluation unavailable to configuration decisions.
