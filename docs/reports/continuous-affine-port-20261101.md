# R2.1-PUSH.LS / #398: readonly affine robot port

This continues the accepted single198-row TRAIN fit. Previous goal turn was
progress: actual full-rank fit and mixed residual evidence, not robot success.
HEAD remains d9119e80926a139a2f38460564d7eab1b6d2a2b0 in the inherited dirty
checkout. No old raw source/result/model artifact was read or recomputed here.
Full goal and broader acquisition/adaptation/release parents remain open.

## Working software

`src/adapters/readonly_continuous_affine.py` owns a detached, validated physical
map, separate immutable policy/snapshot and strong native identity pins. It has
snapshot/fork/predict, no fit/update/restore/optimizer/RNG authority. Prediction
uses accepted core clipping and returns detached arrays. Source fit rows are
metadata; optimizer update count remains zero. Replaced or corrupt native state
is refused without restoring it.

`scripts/maniskill_continuous_smoke.py` admits this exact type in its existing
frozen Pick rollout, requiring executed TS12 history and ordinary TRAIN resets.
It refuses reserved scoring, correction/teacher queries and interventions. It
uses existing physical features and actual float32 command history. Source and
private fork are checked before further control and final release. The existing
`TrainableLearner` union remains separate, so correction workflows gain no new
update authority. Existing BC/PC/nearest/default dispatch branches are preserved.

`scripts/push_cube_continuation_study.py` admits the same readonly reader through
its existing bounded TRAIN Push rollout, with direct frozen source checks before
prediction/control and after work. It does not create a second runner or solver.

## Focused execution

Prospective engineering budget:100pytest executions,120s aggregate body,60s/body,
512MiB observed process RSS,512literal prediction rows,0LS/SGD/model RNG draws or
physics;8MiB software artifacts plus32MiB type cache,180s scoped statics. The check
observer blocks Torch/Gymnasium/Sapien/ManiSkill imports; source tests forbid
`numpy.linalg.lstsq` and core solve. Live counters cap literal matrix work.

Every command below starts physically on D:, with TEMP/TMP/TMPDIR=D:\Dev\_Temp,
Python bytecode off, and OMP/OpenBLAS/MKL threads1. Prefix `python` is the existing
`.venv/Scripts/python.exe`. Stage is
`artifacts/runs/continuous-affine-port-20261101/`.

| Command after `python -I -B artifacts/runs/continuous-affine-port-20261101/check.py` | Outcome | Observed body seconds | Peak bytes | Matmuls / rows |
| --- | --- | ---: | ---: | ---: |
| `initial` |17PASS,1FAIL|1.5061738000|82919424|8/9|
| `push-fix -k drive_existing_push` |1PASS,17deselected|1.1614300000|77991936|2/2|
| `combined` |18PASS|1.3104566000|82681856|10/11|
| `late-observer -k last_history_observer` |4PASS,18deselected|1.0664952000|72925184|6/6|
| `final-combined` |22PASS|1.2615830000|82849792|16/17|
| `final-history -k last_history_observer` |4PASS,18deselected|1.1212070000|74014720|6/6|

Final combined22 passes, followed by affected4 passes after strengthening the
final-history observer target. All22distinct cases pass;67executions66PASS and
one preserved initial failure,0skips (deselected cases are not skips). Aggregate
body7.4273456001s;48literal matmuls/51rows. All observers report budget_ok=true and
blocked=[];0genuine fits, optimizer updates, model draws or physics.

Coverage includes clipping, detached source/snapshot/output, immutable reader,
no update/restore methods, corrupt rank/role and equal state/coeff replacement,
separate private fork, all3 Pick TRAIN seeds using actual command history,
Push typed native-shaped port, unsupported recipe/zero history/query/probe/
decoding refusals before native creation, mid-episode replacement, and late
source/private replacement before a command and at final history release.
The inherited constant-zero default also passes without prediction work.

Scoped final commands (all exit0):

```text
python -m ruff format --check src/adapters/readonly_continuous_affine.py scripts/maniskill_continuous_smoke.py scripts/push_cube_continuation_study.py tests/test_readonly_continuous_affine.py
python -m ruff check src/adapters/readonly_continuous_affine.py scripts/maniskill_continuous_smoke.py scripts/push_cube_continuation_study.py tests/test_readonly_continuous_affine.py
python -m mypy --follow-imports=silent --cache-dir artifacts/runs/continuous-affine-port-20261101/type-cache scripts/maniskill_continuous_smoke.py scripts/push_cube_continuation_study.py src/adapters/readonly_continuous_affine.py tests/test_readonly_continuous_affine.py
```

Final logs: format-closeout.log/lint-closeout.log/types-closeout.log. Software
stage measured17629761B including17379557Bcache before this report; noncache
250204B, within separate8/32MiB caps. No clean-clone/installed distribution check.

## Failures and review

Initial Push fixture omitted required `counts`; fixed the fixture without
relaxing acceptance. Initial typing reported16fixture/test-annotation errors,
then one substantive union error: adding readonly to the rollout union leaked
into the correction source annotation. Fixed with a separate trainable union.
One later empty-list test annotation was fixed. Statics stdout before closeout
is retained; initial diagnostics were observed in the tool transcript.

Finite independent source reviewer read the adapter, affected existing source
and tests in4read-only batches, without executing or reading model/data artifacts.
It requested changes: an owner check preceded the last history callback, allowing
late replacement before control or stale success at final release. Added the
affine-only closure in the existing `_check_mass_history_frozen` composition,
which runs after that callback. One additional bounded source batch ACCEPTED the
production correction but identified that the final-phase regression initially
targeted the first post-close observer. Lead strengthened it to the second/last
observer and executed all4fault cases successfully. The peer did not re-read
that final test-only counter adjustment or independently execute any tests.
This is independent source acceptance plus lead-executed tests/self-review,
not independent numerical or native acceptance. Exception ended; no standing
reviewer or numerical/native allowance remains.

The historical missing downloaded multi-agent prompt was checked and is absent;
current AGENTS/AGENT_PROMPT explicitly supersede it. Two guessed inspection
paths failed; subsequent discovery used actual filenames. One documentation
patch context mismatch was corrected. None started model/native work.

## Open requirements and exact next action

398criterion2's readonly boundary is implemented and scoped software-tested;
criterion5's physical qualification remains open. Accepted198fit, all negative
historical Push/Pick/MiniGrid outcomes, budgets, original consent/receipts/clocks/
charges and retired publication history are unchanged. The former3actual-byte
passes and original5source opens remain spent. No final/probe data are available
to this port for learning, tuning or selection.

Next SAME398: freeze/review ONE separate native TRAIN qualification of the
cached new FITTED-MODEL.json on ALL Push17/Pick17/Pick19/Pick20,0fits/0old source
or raw result reads, existing ports only. Use the cached Linux NumPy1 CPU/Mesa
environment from physical `/mnt/d/Dev/Repositories/Circadian`, D: temporary output;
the old `/mnt/c/...` command paths in historical receipts must not be reused.
Declare the single cached-model read, <=200predictions/controls,4environments,
memory/time/storage/capture budgets and failure policy before execution. Keep
all endpoints and complete negative outcomes; no refit/retry/seed/weight search.
No native allowance was created or consumed here. Fullsuite/Linux/WSL/native,
wheel, browser/device/GPU/CI/release checks were skipped because this increment
is a literal/fake-boundary software gate. No installs/downloads/deletions/Git
publication/global settings changes occurred. Native task usefulness remains
unproved; old fit residuals are not robot evidence or a CPC advantage.
