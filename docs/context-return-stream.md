# Fresh contextual acquisition, retention and return

The existing recurring ticket runner supports `--context-return`. It starts a
zero perceptron; it does not load the jointly trained model. The original default
and `--policy-change` recipes retain their behavior. This mode requires perceptron;
the later [queue rehearsal extension](context-queue-rehearsal.md) also supports
explicit recent/coverage replay. Incompatible options fail before work/output reservation.

```powershell
python -m scripts.recurring_ticket_workflow_example --context-return --output context-return.json
```

Use a fresh output path. Internal limits are twelve updates,30seconds,512MiB RSS;
JSON is bounded to512KiB. No extra dependencies. Both public context and task
truth return: A=v1invoice/password; B=v2invoice/parcel; A_return=v1invoice/password.
Each phase uses four six-row TRAIN passes, labels delayed one tick. Checkpoints
0/4/8/12 read all four tasks under their original policies. Endpoint reads also
exercise stale-ticket invalidation and revoked commit permission.

## Executed result

Issue75/R4.1-F/R2.1-U, one frozen local case:

| Updates / exposures | v1 invoice | v1 password | v2 invoice | v2 parcel |
| --- | --- | --- | --- | --- |
|0 /0 (unchanged control)|fail|fail|fail|fail|
|4 /24 (after A)|pass|pass|fail|fail|
|8 /48 (after B)|fail|pass|pass|pass|
|12 /72 (after A returns)|pass|pass|fail|pass|

Both B tasks failed before B training, then succeeded: this is new task gain,
not inherited ability. A invoice was acquired, lost, then recovered under the
same v1 truth. Password was retained, so its recovery field is null. Returning A
loses B invoice while retaining parcel. Thus retention remains incomplete.
Endpoint stale results match the three ordinary successes; all four revoked
commits are correctly denied, including the wrong staged invoice. Permission
refusal is not task success, and stale-ticket recovery is distinct from skill
recovery. Eight ordinary plus three stale successes,24total episode records.

Twelve passes72exposures13mistakes (A8/B4/return1), no random draws, probe/final
updates, replay, native SGD/PC, controller selection or promotion. The original
seed and stream actor remain unchanged; the stream trains its owned candidate.
All twelve original delay1 receipts and four full states are retained. Paid
results, input packets, frozen predictor, source/RNG and budget origin are checked
before more work; changed/replaced paid records are refused and restored.

Nominal lower bounds:4320TRAIN affine terms,4600task affine terms,260coefficient
additions,1920bytes for four retained coefficient matrices. Physical observer:
3learners/zero initializers,1155snapshot copy/validation entries69300logical
coefficient slots,92typed reads,2teachers13packet constructions188encodes116sandbox
dispatches. Copies/validation/Python overhead are additional. Recipe8.33685s,
meter8.94138s, sampled RSS44,879,872B, boundary44,609,536B; sampling is not a hard
OS peak guarantee. Actual262946B SHA256
`581a68d8badbaed81fb6dbf85b78aab3e08798149de86448feb31936e3d7b4b0`.

The zero checkpoint is reused as a no-learning control on identical deterministic
tasks. No equal-compute, statistical, unseen-task, CPC advantage, general neural
plasticity or release claim follows. The previous joint fit and native BP negative
remain unchanged; they answer different questions. This result motivates bounded
TRAIN rehearsal against measured interference, not a longer checkpoint chain.

## Validation and next extension

105distinct selected checks pass on Windows and Linux (one intentional separate
old-policy-case skip each); three affected source/test files pass Ruff, formatting
and fresh default/Linux mypy. Software failures and consumed work are preserved.
Source/procedure/result review is self-review, not independent acceptance.
Full suite, native/physics/math, packaging, clean-clone and CI gates are unrun.

Protocol, exact commands, failed checks, ledgers, raw results and zero-model
readback: `artifacts/runs/context-return-stream-20261023/`. Live handoff:
[issue75](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/75).

Subsequent issue76 implemented contextual queues through the existing rehearsal
path and preserved D7defaults. Both fixed replay cases keep the same3/4finalsuccess;
see [the extension and limitations](context-queue-rehearsal.md). This original
no-replay result remains unchanged; explicit extra exposure is charged separately.
