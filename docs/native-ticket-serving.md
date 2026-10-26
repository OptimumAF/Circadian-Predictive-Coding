# Publish a learned native BP task policy

In the existing CPU Torch environment, using the original qualified old-task
report from #43:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --output serving.json
```

The output must be fresh. The input's original exact hash and complete96-update
state are checked through the existing BP qualification reader. This command
requires that local research artifact; use `--fresh-bp` in
[the fresh training demo](fresh-bp-ticket-demo.md) for training without history.
No downloads or external tools/services are involved.

## Behavior and boundaries

The existing `AttributionStream` receives a `PromotableActor`. It owns one shadow
candidate and eight delayed new-context TRAIN batches,48row exposures, using the
unchanged seed17/native BP settings. The actor executes old tasks before and
midway through those updates. Its original weights remain stable during learning.

`TorchServingPredictor.serve` takes one atomic complete actor snapshot per typed
request. The existing native decoder masks raw logits before normalizing allowed
actions. Results identify actor version, generation and native clock. No readout
model/cache/history survives a request; reconstruction costs are paid each time.
The adapter latches faults and keeps attempt/return counters. A request may finish
after a concurrent publication, but still reports the snapshot it actually used.
This is snapshot-based local serving, not a measured concurrent latency service.

After the fixed eight updates, existing `PromotionGuardEvaluator` and
`ServingPromotionController` measure new-task categorical accuracy1.0, gain>=0.1,
old-task drop<=0, finite native state/probabilities, allowed catalog actions,
prediction<=0.25s and complete native tensors<=4096bytes. Failed guards produce a
detached refusal report and leave the actor unchanged. There is no extra training
or automatic retry. Accepted publication switches the full native model and
metadata through the existing atomic transaction. Task episodes and stale/revoked
cases use that actor. One-step rollback then restores the complete prior model,
configuration, metadata and cache, with an increasing generation.

Guard rows have independent inner_guard identities, but their feature templates
repeat the tiny synthetic teacher domain. These development guards establish
configured behavior on known templates, not heldout generalization or broad
safety. Actual tool permission/revision checks remain at effect time. Outer task
observations do not choose guards, update count, promotion thresholds or seeds.
No final labels, disposable probe updates, replay, circadian mechanism or learned
controller participates. Historical96updates/576rows remain separately charged;
only the new8updates/48rows belong to this invocation's budget. Neither source
state nor old runtime/budget authority is renewed.

The stream must finish before the demo publishes; this does not implement a
long-running post-promotion candidate rebase or durable restart. Default streams
keep a0/c0 identities. Injected actors are restricted to ordinary TRAIN-only
streams; probes and streams with final evaluation cannot enroll a serving actor.
Receipts use the candidate's original actor version, including delayed labels.

## Verification and extension

The one frozen BP case passed: all eight new updates returned; guard new-template
accuracy rose from5/6 to6/6 while old accuracy stayed6/6. All four ordinary tasks
passed after publication. Both stale cases recovered and both revoked commits
were refused. Rollback restored the original full native state and original
legacy-task failure. This is useful learned serving and deliberate restoration,
not recovery from forgetting. New work was8updates/48rows/70native forwards,
including58typed decisions and four guard forwards; prior96updates stay paid.

Final Linux39checks passed fresh. Windows coverage is17unchanged qualified passing
checks, one repaired recipe check, and21fresh pure guard checks. Two failed native
fixture cohorts remain in the ledger; it is not39fresh passes on the final Windows
tree. Scoped Ruff/format/default+Linux mypy passed. All review here is self-review.

```console
python -m pytest -q tests/test_torch_serving.py tests/test_promotion_guard.py
python -m ruff check src/adapters/torch_serving.py src/app/coupled_attribution.py scripts/run_bp_ticket_serving.py tests/test_torch_serving.py
python -m ruff format --check src/adapters/torch_serving.py src/app/coupled_attribution.py scripts/run_bp_ticket_serving.py tests/test_torch_serving.py
python -m mypy --explicit-package-bases --follow-imports=silent src/adapters/torch_serving.py src/app/coupled_attribution.py scripts/run_bp_ticket_serving.py tests/test_torch_serving.py
```

These tests reuse full paid BP96/104snapshot fixtures; they do not refit warmup.
They still execute bounded native copies/prediction and two tiny TRAIN updates
per full cohort. Test recipes are engineering checks, not independent studies.
See `artifacts/runs/native-ticket-serving-20261023/` for frozen limits, retained
failures, exact final commands and observed costs. Do not extrapolate scoped
checks to clean installs, GPU support, release readiness or a CPC advantage.

Why this: learning useful candidate weights is insufficient when the live actor
never receives them. Compose the existing guard/atomic publication ports rather
than introducing another controller or state framework. Extend native typed
serving via those same ports; declare a new protocol before testing other methods
or longer contexts, and preserve this baseline and its limitations.
