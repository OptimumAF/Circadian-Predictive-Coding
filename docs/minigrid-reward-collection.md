# TRAIN-only MiniGrid reward collection

`collect_minigrid_reward_episode` collects a sampled native cue/key/door episode
for the existing recurrent reward learner. It accepts only the exact predefined
A/B/C_train layouts, blue/red cues and train role. D and E are unavailable.
The source supplies independent zero-reset episode sessions and full snapshots.
The native Torch recurrent source implements that contract; an arbitrary source
implementation is not attested by the collector.

```python
import numpy as np
from src.adapters.minigrid_reward_collection import collect_minigrid_reward_episode
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.adapters.torch_reward_update_learner import RewardUpdate, TorchRewardUpdateLearner
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_key_door import predefined_layouts

source = TorchRewardUpdateLearner(TorchRecurrentPolicy(148, 8, 7, 17))
candidate = source.fork()
sampler = np.random.Generator(np.random.PCG64(29))
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1, max_wall_seconds=45))
report = collect_minigrid_reward_episode(
    candidate, predefined_layouts()[0], "blue", episode_id="train-A-blue",
    sampler=sampler, check_boundary=budget.before_final,
)
budget.before_update()
diagnostic = candidate.train_batch(
    RewardUpdate(report["episode"], report["behavior_state"]), report["episode"].episode_id,
)
budget.record_update()
```

Requires the existing optional Torch and pinned MiniGrid dependencies described
in requirements-resnet.txt and requirements-minigrid.txt. No new dependencies,
environment variables, install step or training authority is created here.
The example is one engineering episode/update, not a learning protocol or a
matched comparison. Do not choose rewarding seeds or repeat it to tune results.

## Contract and costs

All seven actions are sampled from the returned raw probabilities using one
caller-owned PCG64 uniform and cumulative mass per decision. No legal mask,
argmax fallback, retry or teacher correction. Sessions use148 public features
with a maximum16 predictions. Behavior probabilities and the full original
checkpoint accompany the existing PolicyGradientEpisode. Its native admission
rechecks the checkpoint and action law before an atomic reward SGD.

The collector records prediction attempts/returns, actual uniforms/actions,
environment construction attempts/returns, physics attempts/returns, snapshot
readbacks and cleanup. The original native report retains reset/step counts,
rewards, public transitions and native terminated/truncated flags. A truncated
episode is a valid reward packet; to_demonstration() is deliberately unused.
If both native end flags are true, reward DTO termination uses terminal, while
the native report keeps both. There is no bootstrap in the existing objective.

Caller guards are checked around work; budgets and RNG draws are never renewed
or rolled back. Source/session/sampler mutations, invalid clocks, native faults,
cleanup or readback faults raise MiniGridRewardCollectionFailed with a detached
observed report and no completed reward packet. Failed constructors may have
unobserved reset work; this remains explicit unknown work. Mandatory cleanup
runs even after the caller guard refuses more work. The primary fault remains
separate from cleanup/readback faults. Storage is bounded by16 decisions and
one behavior snapshot. Process RSS includes native allocations; report payload
bytes alone are not a process-memory estimate.

## Structure and extension

```text
src/adapters/minigrid_reward_collection.py  native outer collection boundary
src/infra/minigrid_key_door.py              existing owned native environment
src/core/policy_gradient_episode.py         existing reward trajectory contract
src/adapters/torch_reward_update_learner.py  existing reward update packet/port
tests/test_minigrid_reward_collection.py    collection/failure/isolation behavior
```

Run scoped checks with the stage run_checks.py focused/statics commands recorded
in artifacts/runs/r42-minigrid-reward-collection-20261016/RESULTS.md. Preserve
receipts before corrective checks; its native branch is a sole frozen case.
The collector itself has no update authority. The route below composes the
existing ActorShadowRuntime, Experience, LabelArrival and RewardUpdate interfaces,
with separately reviewed small protocol and permanent costs; keep training,
final evaluation and disposable probes separate. Useful acquisition, retention,
recovery and comparative benefit remain scientific requirements.

## Executed qualification

Final combined170focused tests pass with no skips; scoped Ruff/format and mypy
win32/Linux pass. Earlier typing failures and successful pre-correction checks
remain in the stage. Independent SOURCE/GO preceded the sole native case;
independent ACTUAL checked saved trajectories and sampling arithmetic.

The frozen A/blue seed17/PCG64(29) case collected16native steps and truncated
with return0. One existing reward SGD commits16rows with diagnostic−0.0; stale
reuse is refused before another SGD. Source/actor/global RNG assertions pass.
Body4.2260109s/parent5.0215401s; process peak329871360B. No retry or learning-
benefit claim. Native budgets are spent. Full R4.2 and release gates remain open.

On corrupt nonfinite native return headers, failed evidence retains counters
and transitions while replacing only the invalid header with null and an
invalid_total_return string. The original drives refusal; no valid return is
invented. Strict JSON export excludes opaque behavior-state tensors; the saved
native artifact contains root-executed state-check results, not resumable states.

## Delayed TRAIN reward runtime route

Preview the fixed protocol without model or environment work:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_reward_smoke --describe
```

Run only this small local engineering recipe, publishing to a new destination:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_reward_smoke --output artifacts/minigrid-reward-route.json
```

The new CLI composes the existing ActorShadowRuntime and reward interfaces.
It uses fixed seed17 RNN148/H8/K7, learning rate.03/momentum.9 and one PCG64(29)
over A/blue,A/red,B/blue,B/red,C_train/blue,C_train/red,A/blue,A/red. Each episode
has at most16 sampled native actions. One original budget permits8updates,
45body seconds and1GiB process RSS; qualification uses a60second parent timeout.
CLI sets CPU intraop1/interop1/BLAS1/GPU0. No dose, seed or method knobs.

An experience alone causes no training. Its exact ID is released as a train-role
label one tick later; native reward admission checks the full original behavior
checkpoint. Original receipt, native counter/row progression, source, actor,
sampler and budget must validate before the next collection. Late faults retain
committed receipts and permanent charges and stop the stream. No final data,
teacher, probe, replay, promotion or evaluation is available to this route.

Reports retain observed draws/physics, every slot and original receipts, including
failed prefixes. Opaque checkpoints remain transient; existing bounded8-payload
inbox retention includes up to8behavior snapshots and counts toward RSS. This
is not a checkpoint-resume artifact or a measured learning comparison.
JSON output is strict finite, at most16MiB and never overwrites an existing file;
existing output/missing parent is refused before native setup. Incomplete runs
produce nonzero CLI status with their observed report where serializable.

```text
src/adapters/minigrid_reward_pilot.py      fixed recipe over existing actor/inbox
scripts/run_minigrid_reward_smoke.py      describe/run/exclusive bounded JSON
tests/test_minigrid_reward_pilot.py       actual runtime with fake native boundary
```

Run focused tests with `python -m pytest -q tests/test_minigrid_reward_pilot.py
tests/test_minigrid_reward_collection.py tests/test_key_door_reward_pilot.py
tests/test_torch_reward_update_learner.py`. Scoped lint/type commands and native
receipts are recorded in artifacts/runs/r42-minigrid-reward-route-20261016/RESULTS.md.
Extend this recipe through the existing runtime interfaces, preserving delayed
on-policy release and permanent costs. Any comparative learning protocol needs
separate prospective freezing/review; no favorable-seed or dose selection.

Executed qualification:110focused cases/0skips and scoped Ruff/format/win32+
Linux mypy pass. After independent SOURCE/GO the sole native route completes
8updates/128rows, with8empty pre-release polls and original delayed receipts.
All8episodes truncate with return0/diagnostic−0.0. Source and serving actor remain
unchanged; no acquisition/retention/recovery is demonstrated. Pilot15.2546648s,
parent18.7156s; sampled pilot RSS peak338956288B (transient import peak unmeasured).
Output3145938B. Native allowance spent/no retry; original failures remain.

## Fixed BC warm prefix (opt-in)

Preview the warm composition with no reference read, model or environment:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_reward_smoke --warm-start --describe
```

The opt-in recipe regenerates the fixed accepted unweighted A128 native state
from the original TRAIN sources. Saved JSON has no native tensors. This required
reconstruction and its five raw/two-endpoint native parity checks are new paid
work. Only exact accepted observable parity admits continuation; a mismatch
stops with its observed costs. Both-cue acquisition remains false in the accepted
prefix; no cue, seed or weighted arm is selected.

The complete128-update/1152-row snapshot, optimizer momentum and gradients carry
into the typed demonstration/reward learner and its independent runtime forks.
The existing eight TRAIN reward collections/releases use the same order and
PCG64(29). Native counters continue from128; reward receipt numbers/logical ticks
and its separately charged eight-update budget are phase-local. Original warm
charges remain permanent. Cold defaults and their45-second budget are preserved.

The warm whole-case limit is180body/210parent seconds/1GiB sampled RSS; the warm
training arm retains768MiB. Reward work is bounded by45seconds and the remaining
whole deadline. New work is128 warm SGD plus at most8 reward SGD/128 reward rows,
10 raw forwards/90rows,4 warm live episodes/at most64steps and8 reward episodes/
at most128steps. No new teacher, final/probe, replay or promotion work is allowed.
Existing five checkpoints/eight inbox payloads remain bounded; report bytes are
not total memory. Native state remains transient.

After separate protocol qualification, an invocation publishes to a new file:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_reward_smoke --warm-start --output artifacts/minigrid-warm-reward.json
```

This is an integration smoke. Sampled reward is not guaranteed, and collection
returns do not establish acquisition or a reward-update benefit. Future effect
comparisons need shared warm forks and carried-momentum/additional-update controls.
See ADR-0358 and artifacts/runs/r42-minigrid-warm-reward-20261016/ for prospective
limits, checks and observed outcomes; no large sweep follows this option.

```text
src/adapters/minigrid_warm_start.py    accepted-source warm reconstruction/parity
src/adapters/minigrid_reward_pilot.py existing cold/opt-in warm runtime recipe
scripts/run_minigrid_reward_smoke.py  existing CLI with explicit warm option
tests/test_minigrid_warm_reward.py    phase/state/isolation/failure/CLI checks
tests/test_minigrid_warm_prefix.py    exact saved probability/physics parity
```

Focused qualification commands from the repository root (PowerShell):

```powershell
$warmScope = @('src/adapters/minigrid_warm_start.py', 'src/adapters/minigrid_reward_pilot.py', 'scripts/run_minigrid_reward_smoke.py', 'tests/test_minigrid_warm_reward.py', 'tests/test_minigrid_warm_prefix.py')
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_minigrid_warm_prefix.py tests/test_minigrid_warm_reward.py tests/test_minigrid_reward_pilot.py tests/test_minigrid_reward_collection.py tests/test_torch_demonstration_reward_learner.py tests/test_minigrid_weighted_bc_pilot.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check --no-cache @warmScope
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check --no-cache @warmScope
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @warmScope
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @warmScope
```

### Original warm qualification and preserved failure

The original native smoke is **incomplete**. Warm128/1152 reconstruction, all
five exact saved raw/both endpoint native parity checks and full-state restore
pass. Six reward commits then apply90rows; the seventh collection reaches the
original45-second phase limit and the eighth is not run. First collection ends
with−1; the next five truncate with zero return. All134 new updates/1242rows,
failed collection15draws/14physics and original budgets remain paid and recorded.
No acquisition or reward-update benefit is established.

The case took58.1887parent/55.0919pilot seconds with363728896B sampled peak RSS.
Source/actor flags remain unchanged and seven collection environments close.
The secondary outer RNG readback reflects its pre-failed-collection expectation;
the actual spent draws are retained. It is not evidence of unseen RNG advances.

After a saved-only component measurement, a narrow duplicate serialization repair
retains full warm/state/counter/receipt validation.191 final focused cases and
scoped statics pass; independent SOURCE POSTREPAIR accepts it. The old native
failure qualifies the pre-repair source only. At that historical stage no native
run had followed the repair, so eight-update completion and45-second fit were
open. The separate repaired validation below resolved those scoped gates;
the failed case is not renewed. [Original evidence](../artifacts/runs/r42-minigrid-warm-reward-20261016/RESULTS.md).

### Repaired native route completed

A separately frozen and independently accepted validation completed all eight
reward commits under the same limits:55.1099parent/52.2274pilot seconds,
136new updates/1274rows,10raw90rows/170total predictions and physics. Warm parity
and full-state restore pass; source/actor remain unchanged. First sampled Ablue
ends−1 in10steps; all seven later episodes time out with return0. This validates
the runtime path, with no useful learning or reward-effect claim. The original
134paid updates/1242rows remain permanent; combined work is270updates/2516rows.

All18 source assumptions were independently checked after the native case and
before subsequent evaluator changes.191 tests/statics were reused, with no
repeated teacher/baseline/probe studies. [Commands, costs and source window](../artifacts/runs/r42-minigrid-warm-reward-20261016/repaired-case/RESULTS.md).
The distinct [TRAIN readout port](minigrid-warm-readout.md) for actual136-update
variable-row reward checkpoints now passes scoped qualification; existing BC
readout clocks remain unchanged. MINIGRID-WARM-READOUT-COMPOSE/CASE is still open:
wire owned before/after snapshots into this recipe after all training, then
freeze and review a small native-use protocol before execution.
