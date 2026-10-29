# Bounded replay followed by a causal fresh update

[Issue242 / R7.1-F.RW](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/242)
delivers the scoped composite action through the existing shared executor.
Source/procedure/result **self-review**, not independent acceptance. Full goal,
broader R7.1 and scientific/release requirements remain open. Prior240/241
native command and installed delivery evidence is reused within its old scope.

## Working behavior

An ordinary owner can declare `AdaptationAction("rehearse_then_learn", 1, 3)`
alongside its existing wait/wake/replay IDs. The original finite envelope admits
the complete three-row TRAIN replay bank and one causal fresh update. The action
uses one decision, two separate priority-gate admissions and original separate
replay/wake budgets. Larger wake bounds retain the existing per-poll limit.

`src/app/resource_sharing.py` now allows this default-rate/unchanged-gate/zero-sleep
composition. Its existing replay code validates/commits the bank; the existing
full candidate/receipt/source/clock/budget guard runs again before wake admission.
An extracted original wake observer/poll helper runs unchanged FIFO updates.
All other original methods AST-match the saved pre-edit tree, preserving the
inherited working changes. No core algorithm, state schema, action envelope,
replay store, runner, dependency or default policy change.

Both cumulative envelope limits are checked before work. Initial replay deferral
performs no wake. If replay commits and wake defers or has no arrived labels,
the returned poll preserves the real replay receipt and actual empty wake/defer
result. The caller can choose wake-only next. Repeating the composite ID requests
another paid replay; there is no implicit continuation, transaction rollback,
automatic retry, budget renewal or ignored counter change. A thrown late failure
retains original known receipts and all attempted/unknown work. Corrupted or
replaced prior evidence refuses further native work. Mixed replay+rate/gate/sleep
and missing executors remain refused before resource/payload work.

Usage: [fixed TRAIN replay actions](../categorical-replay-actions.md#replay-followed-by-fresh-learning)
and [finite actions](../adaptation-actions.md). Existing scalar fixture setup
accepts an optional explicit envelope; its defaults are unchanged. The obsolete
combined-unsupported regression now checks replay+wake+rate is still refused.

## Focused software qualification

Final combined tree: **66 distinct checks/platform pass**:28new composite cases,
27 existing replay cases and11 affected wait/wake/sharing/rate-gate/sleep controls.
Tests cover execution order/full state/charges, FIFO and poll cap, absent/future
labels, both-phase pause/serving/resource/quota deferrals, envelope exhaustion,
unsupported effects, missing executor, failed replay/restore, late budget stops,
lost wake return and changed receipt/candidate/source/clock before wake admission.

| Platform, all attempts | Instances | Pass/fail/error/skip | Body seconds | Sampled peak RSS bytes | Scripted train calls/rows |
|---|---:|---|---:|---:|---|
| Windows | 187 | 184/3/0/0 | 12.586564800 | 76,943,360 | 193/469 |
| Linux | 132 | 132/0/0/0 | 18.658638739 | 68,259,840 | 124/292 |

Windows failures: intended preimplementation RED unsupported/0work; a new fixture
called a locking public property from an already leased budget callback; another
expected a lost wake return to increment the completed-update budget. The latter
correct behavior is completed0, irreversible wake/sharing admission1, unknown
native attempt and stopped owner. Fixtures corrected to assert original rules;
no production guard weakened. All failed attempts/XML retained. Final tree has
zero failing/error/skipped cases; one legacy actual native example is deselected.
No optional Torch/environment imports or genuine model work in these cohorts.

Original allowances held: <=64distinct new and160affected checks, <=224total
instances/platform including repair repeats;90sbody/platform/512MiB and512scripted
calls2048rows. Five-file Ruffformat/check/default+Linux-target scoped mypy pass
first attempt8.885087900s/120; conservative touched-cache113,283,724B/160MiB, not
growth. Original core replay/math/other-method source evidence reused unchanged.

Exact command arrays/outcomes/XML in `artifacts/runs/replay-wake-action-20261027/`.
Representative commands (the final tree selection also includes the11 explicit
affected node IDs recorded in `*-tree-checks.json`):

```powershell
.venv/Scripts/python.exe -B artifacts/runs/replay-wake-action-20261027/check.py win-combined tests/test_replay_wake_actions.py tests/test_categorical_replay_action.py -k 'not native_wake_replay_wake_example'
.venv/Scripts/python.exe -m ruff check src/app/resource_sharing.py tests/test_replay_wake_actions.py tests/test_categorical_replay_action.py artifacts/runs/replay-wake-action-20261027/check.py artifacts/runs/replay-wake-action-20261027/native.py
```

Linux uses the existing offline Python environment, `timeout110s`, CPU1 thread
variables and the same `check.py linux-tree` selected nodes from the checkout.
No new environment/install or old trained-source fixture. Source copies precede
edits;11helper/test/source/protocol pins qualify the actual final tree.

## Sole tiny native equivalence case

Frozen PROTOCOL/qualification and source/procedure self-review preceded GO.
Exactly once under60s parent:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/replay-wake-action-20261027/native.py
```

Exit0/complete;6.531006100s through scalar capture preparation, enclosing parent
7.228166200s including final output. Reported sampled RSS311,246,848B before the
small terminal JSON encode/write; sampler continues through writing, no separately
exported final peak or exhaustive process-tree/instantaneous maximum claim.
Both observed clocks stay below the30s global allowance/1GiB. CPU1, no retry.

Two fresh identical conventional BP owners, seed17/feature7/hidden4/K6, native
default LR.03/momentum.9. One literal three-row one-hot TRAIN bank (labels0/1/2),
one two-row all-ones TRAIN experience (zeros), observed0/label-arrived1. Composite
choice versus original separate replay/wake choices produce **exactly equal
complete endpoints**, including momentum, gradients, configuration and clocks.
Each endpoint has2native updates5row exposures; one wake receipt/one replay
receipt, wake budget1update/replay0, replay budget0update/3rows, two sharing
admissions. Decisions differ1 versus2 as declared; no unknown work/publication.
Each original source and actor, budget limits/clock/start/sampler/progress and
global Python/NumPy/Torch RNGs remain unchanged. Runtime NumPy2.4.6/Torch2.14.0+cpu.

All native attempted calls equal returned calls:22parameterized heads,
44seeded weight-draw calls1144scalars,66public snapshots,6forks,4BP updates10rows,
4forward/CE10rows,4backward and4SGD steps. Source initialization, private staging,
restores/reference work all paid; snapshots/constructors are separate counters,
internal validation/copy allocations are paid time/RSS rather than claimed full
FLOP/kernel measurements. No predictions, PC/CPC/latent inference, task scores,
source files/archives, replay selection, teacher/physics/probe/final/promotion.
No model weights or observations/targets serialized. This is wiring equivalence,
not a fresh scientific comparison, useful-task or matched-compute benefit result.

## Scope, evidence and next action

Stage contains pre-edit copies, helpers, frozen protocol/qualification, all test
and static receipts, native exclusive start/scalar result/log/parent, postflight
and issue receipts. Existing studies/negative/spent results remain unchanged,
including236; no retry, historical model reconstruction, new dependency/download,
GPU/remote CI/agent/Git mutation or unrelated reset. Scoped whitespace passes
with inherited LF/CRLF advisories. Skipped full suite/clean clone/installed wheel,
Linux native duplication, independent review, scientific/model cohorts, native
game/teacher/robot/controller-utility/CPC comparisons and release gates.

Historical plans/log remain frozen under AGENTS; GitHub owns current status.
Parent26/full R7.1, reliable acquisition/retention/recovery, scientific benefit
and repository readiness remain open. The old installed wheel predates this
optional API capability; this increment makes no new installed qualification.

Critical path moves to controller use, not another composite/capture/packaging
micro-case. Inspection of `src/core/contextual_controller.py` finds a globally
fixed `ARMS=(wait,wake1)` and exactly four context/arm trials; existing app
counterfactual/trial builders inherit it and cannot select replay+wake.
Next precise implementation action: add an explicit opt-in bounded arm catalog
including replay-then-wake to this existing selector/trial path, while preserving
the default two-arm/four-trial contract and saved-controller schemas/evidence.
Require version/envelope binding, complete catalog coverage, original utility
weights/tie direction, stale-return refusal and all discarded replay/native costs;
test metadata/causality first, freeze any small model trial separately. Why this:
the new learner capability needs a real controller choice before usefulness can
be tested. No new runner, broader baseline tuning, old failed-case retry or claim
that rehearsal will improve the declared utility.
