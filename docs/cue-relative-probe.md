# Cue-relative public pilot and isolated fresh-task probes

The existing MiniGrid learnability pilot can now select the public cue-relative
family for BP, PC, neutral CPC or active CPC. It runs the existing joint TRAIN
stream and disposable F/old-A/A-return diagnostics. The original route remains
the default. No new runner, optimizer, dependency or environment setting is added.

## Describe and configure

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --joint-catalogue --distinct-probe --old-task-readouts --return-to-a --cue-relative-history --train-fit-bounds --describe
```

Select `bp`, `pc`, `neutral_cpc` or `gated_cpc` with the existing method option.
Remove `--describe` only within a prospectively budgeted, reviewed local protocol.
This increment executes synthetic engineering checks; it issues no new scientific
run. The route's existing180second body,210second parent and768MiB RSS limits
still apply, with separate permanent budgets for benchmark, F and A-return work.

Programmatic selection uses `describe_protocol` or `run_pilot` in
`src.adapters.minigrid_learnability`, adding `cue_relative_history=True` to the
same public-history, invariant-history, joint-catalogue, explicit-native-method,
distinct-probe, old-task-readouts and return-to-A options. It is an exact bool;
false preserves the original report/call behavior. The existing history flags
remain transport prerequisites. The selected method, encoding identity and
checkpoint family explicitly identify the new representation.

Weighting, relaxation controls, memory/recurrent routes, rollouts and old
projection-specific activation/gradient/PC-update readers are refused. Optional
`--train-fit-bounds` reports the cue-relative input keys. Future readers must
explicitly support that projection before being enabled.

## Behavior and isolation

- The same original54 TRAIN rows feed128 delayed1 joint wakes. Inputs remain
  raw296; the bound learner applies the non-idempotent cue transform once.
- Complete causal current and next frames are admitted before native training,
  checkpoint readouts or probe work. A unique initial red/blue Ball supplies the
  cue; key colors must be red or blue. The original cue is retained.
- At fixed benchmark checkpoints, detached readers score all six TRAIN traces.
  At0/128, independent disposable forks perform the existing F16 updates and
  old-A readouts at0/4/8/16. Separately charged A16 return continues from F16.
- Native policy, complete optimizer/gradient/chemistry/RNG state and9/54-row
  clocks bind the new exact snapshot family. Cross-family restore/counter use is
  refused. Receipt, source/actor, original-budget/clock and prior-evidence guards
  remain active. Refusals retain completed work and partial records.
- No final D data, promotion, probe selection or deployed learner enters this
  path. Benchmark state is protected while disposable learners update. Old-A
  readouts measure fit on the supplied old traces; they do not establish that
  the old task was previously acquired.

`invariant_history` and `invariant_projection` remain legacy transport/accounting
field names. The protocol marks the selected encoding and the legacy key. Old-A
and return storage maps use the new method name. Original canonical support
tables remain labeled as such: their equality/overlap counts also apply to the
new representation only after full domain admission, because the retained cue
makes the key permutation bijective. They are not newly measured cue-relative
projection work or a generalization guarantee.

See [apply-once boundary](cue-relative-encoding.md),
[constructive TRAIN result](constructive-train-fit.md), and
[ADR0427](adr/ADR-0427-reuse-isolated-probes-for-cue-relative-family.md).

## Structure and extension boundary

```text
src/adapters/minigrid_cue_relative_route.py     recipe, owned snapshots, raw admission
src/adapters/minigrid_learning_pilot.py        existing counters and paid receipts
src/adapters/minigrid_joint_catalogue_route.py existing raw TRAIN composition
src/adapters/minigrid_learnability.py          existing public runtime/probes
scripts/run_coupled_attribution.py             existing CLI
tests/test_minigrid_cue_relative_probe.py      behavior and native integration
tests/test_minigrid_cue_relative_probe_cli.py  dispatch/refusal checks
```

The new helper imports the existing bound/native policy records and pure core
episode/encoding APIs. It owns no training, IO, clock, budget or role authority.
The existing orchestration supplies those boundaries. Extend a diagnostic reader
through an explicit projection-aware port; do not widen an old snapshot reader
to silently accept a different encoding.

## Executed checks and limitations

254distinct scoped cases pass (47new,207regression).292test executions include
one retained fixture failure; no errors/skips. Seven paths pass Ruff, format and
Win/Linux mypy, with the changed fixture file checked again after correction.

Four native methods each complete F16+A16 (32updates/288rows/32predictions288rows),
preserving source, checkpoint, benchmark budget, packet inputs and RNG. A complete
public BP integration completes192updates7488rows94predictions846rows, with new
bounds, all five checkpoint readouts and isolated return. Native metrics are
synthetic engineering outputs, not scientific acquisition or adaptation evidence.

The first public check correctly refused synthetic F novelty after128 TRAIN
updates. Its initial extra Ball had been removed to satisfy the one-cue schema,
which also removed the fixture's novelty marker. The corrected public-only
fixture uses a non-cue Box7 marker; the default direct-probe fixture is unchanged.
A pure full-domain/9-of-9-support check passed before rerunning only that public
case. No production gate was relaxed. Failed work remains charged. The prospective
engineering cap revision and review are preserved in
[SCOPE](../artifacts/runs/cue-relative-probe-20261018/SCOPE.md).

Total engineering native work, including that failure:448updates15552rows,
222predictions1998rows, below the revised500updates16000rows/300predictions3000rows.
The successful public invocation used23.9331963parent seconds and sampled
RSS427126784bytes. Arrays, snapshots, full causal validation, copies and Python
objects cost additional memory/computation beyond row/call counts; enclosing
wall/RSS includes those costs. Exact commands, partial failures, XML and resource
receipts are in [RESULTS](../artifacts/runs/cue-relative-probe-20261018/RESULTS.md).

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_minigrid_cue_relative_probe.py tests/test_minigrid_cue_relative_probe_cli.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/adapters/minigrid_cue_relative_route.py src/adapters/minigrid_learning_pilot.py src/adapters/minigrid_joint_catalogue_route.py src/adapters/minigrid_learnability.py scripts/run_coupled_attribution.py tests/test_minigrid_cue_relative_probe.py tests/test_minigrid_cue_relative_probe_cli.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/adapters/minigrid_cue_relative_route.py src/adapters/minigrid_learning_pilot.py src/adapters/minigrid_joint_catalogue_route.py src/adapters/minigrid_learnability.py scripts/run_coupled_attribution.py tests/test_minigrid_cue_relative_probe.py tests/test_minigrid_cue_relative_probe_cli.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/minigrid_cue_relative_route.py src/adapters/minigrid_learning_pilot.py src/adapters/minigrid_joint_catalogue_route.py src/adapters/minigrid_learnability.py scripts/run_coupled_attribution.py tests/test_minigrid_cue_relative_probe.py tests/test_minigrid_cue_relative_probe_cli.py
```

Use `--platform linux` for the second type-check platform. Recorded validation
split scalar and native checks, reusing unchanged accepted cases to stay within
the prospective resource caps. Full suite, clean-clone and release gates were
not run. Existing scientific negatives remain: new128fit48/54/0exact and old512
fit52/54. Acquisition, useful old-A retention/recovery and circadian benefit are
unestablished. The subsequent [real BP case](cue-relative-bp-adaptation.md) now
executes that fixed protocol. It reduces F loss but does not fully acquire either
task; the preserved invariant control has larger warm F gain and smaller old-A
loss. Next extend the existing rehearsal path to public296/K7 old-A replay,
with explicit matched exposure/cost controls. No additional fit dose or broad
sweep follows from these integration checks.
