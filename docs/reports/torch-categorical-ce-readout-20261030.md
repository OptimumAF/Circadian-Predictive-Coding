# Disposable CE readout learns TRAIN fixtures and serves owned typed decisions

R2.1-TW.CE / [#320](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/320),
following accepted319's separation witness. Source HEAD
`d9119e80926a139a2f38460564d7eab1b6d2a2b0` plus inherited dirty work preserved.
This is software qualification; the genuine all-four-known-task case is unexecuted.

Delivered `src/adapters/torch_categorical_ce_readout.py`, focused tests and a direct
extraction of the original scalar normalization block in `torch_decision.py`.
The new fitter reuses accepted317's actual-owner/private-baseline/hidden ports;
the old LS fitter, native SGD, model policy, runner flags and defaults are unchanged.
Why a separate adapter: keep the accepted LS module stable while coupling the new
bounded CE fit with its owned typed reader. [Usage/contracts](../torch-categorical-ce-readout.md).

On planted D2/H3/K3 fixtures the detached float32 output learns all three supplied
TRAIN labels. First closure equals the source's original full-class CE, confirming
source-weight initialization. Final losses across four successful fits range
1.192092824e-7 to3.178913914e-7, with24/25 closures. Original populated SGD/gradient
state is unchanged and the next native update matches the control exactly.
Typed readouts choose all supplied targets and remain independent of subsequent
source training and caller-result mutations. The original native typed reader
retains exact literal raw-score/allowed-softmax behavior; forbidden logits and
lexical ties remain unchanged.

No native four-task success, generalization, benchmark advantage, matched compute
or CPC benefit is established. This control adds detached optimization and is
explicitly separate from the original BP/PC/CPC protocols. Source features and
output capacity stay fixed; no LP coefficients, new seed, weighting or extra
whole-source training was selected from task/probe outcomes.

## Executed checks

Windows Python3.14.7/NumPy2.4.6 and existing Linux Python3.12.3/NumPy1.26.4,
both Torch2.14.0+cpu. One process per invocation, CPU thread environment1.
Start physical `D:/Dev/Repositories/Circadian`, PowerShell TEMP=TMP=D:/Dev/_Temp.
Let `S=artifacts/runs/torch-categorical-ce-readout-20261030` and
`T=tests/test_torch_categorical_ce_readout.py` in these command suffixes:

```text
.venv/Scripts/python.exe -B S/check.py S/windows-first-work.json T --basetemp=D:/Dev/_Temp/circadian-ce-readout-win-first
.venv/Scripts/python.exe -B S/check.py S/windows-guard-work.json T -k "input_copy_mutation or boundary_mutation or roles_rejected" --basetemp=D:/Dev/_Temp/circadian-ce-readout-win-guard
.venv/Scripts/python.exe -B S/check.py S/windows-role-work.json T -k foreign_role --basetemp=D:/Dev/_Temp/circadian-ce-readout-win-role
```

Replace literal S/T with paths defined above. Linux prefix:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B
```

Same suffixes with `linux-{first,guard,role}-work.json`, same selections,
`--capture=sys` and
`--basetemp=/mnt/d/Dev/_Temp/circadian-ce-readout-linux-{first,guard,role}`.
Receipts store exact argv; yielded handles were resumed, no process restarted.

| Invocation | Passed | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| Windows first |21|3.811240300|328388608|
| Linux first |21|7.894208894|348200960|
| Windows guard |6|3.374819300|325890048|
| Linux guard |6|7.793605872|338907136|
| Windows foreign role |1|2.059762100|245653504|
| Linux foreign role |1|5.926099238|263401472|

All56 executions passed, zero pytest failures/skips. **23 distinct qualified checks
per platform:** initial21, new copy-mutation refusal and new foreign-role callback
refusal. Five prior role/boundary checks repeated in each six-check guard run.
After review, only copy/role admission changed; unchanged numerical/reader evidence
is reused. No full numerical suite rerun after that guard-only change is claimed.

Tests cover the hard closure limit (65th attempt refuses before forward,64 actual
forwards/backwards paid), lost optimizer return after one paid gradient, source/
RNG/case/batch mutations before backward, spoofed public snapshot, equal-value
owner/fit/coeff/work replacement, fault latching and later source independence.
Mutation tests deliberately preserve their changes; no donor repair/restoration.

Shared all-attempt cost, **30.859735704s**, maximum348200960B:

- 64 native component constructions,128 named seeded draws/960 values.
- 8 native SGD/backward updates,24 training-row exposures.
- 16 detached CE optimizer constructions and step attempts; four successful
  learning fits, remaining attempts intentionally fault/cap tests.
- 235 CE output forwards/705 row exposures;227 backward calls/681 rows. Eight
  mutation cases stop after a forward and before backward. Windows118/114 versus
  Linux117/113 forward/backward counts reflect retained platform convergence
  differences, not selected runs or changed settings.
- 4 final TRAIN reads/12 rows;26 hidden reads/58 rows;10 fitted affine inference
  reads/10 rows;14 other native head forwards/34 rows;32 typed call attempts.
- Two intentional scalar global NumPy RNG mutations are refused and retained,
  separate from named model-initialization draws. Zero LP/LS/SVD/autograd.grad,
  PC latent, simulator, replay, probe/final, GPU or comparative cohort work.

All six receipts have budget_verifiedTrue; per-process120s/768MiB, cumulative240s,
and shared numeric caps in SOFTWARE_PROTOCOL.md held. The observer wraps physical
work before ops and retains returned/failed fitter work, including missing step
returns. It samples RSS before Torch import. These counts do not enumerate every
validation/copy/line-search scalar/tensor operation or establish matched compute.
No new library-temporary-byte ceiling is claimed; all intentional writes are in
the bounded D stage and process temporary locations are on D.

Final scoped commands target new adapter, `src/adapters/torch_decision.py` and
the new test file:

```text
.venv/Scripts/python.exe -B -m ruff check src/adapters/torch_categorical_ce_readout.py src/adapters/torch_decision.py tests/test_torch_categorical_ce_readout.py
.venv/Scripts/python.exe -B -m ruff format --check src/adapters/torch_categorical_ce_readout.py src/adapters/torch_decision.py tests/test_torch_categorical_ce_readout.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent src/adapters/torch_categorical_ce_readout.py src/adapters/torch_decision.py tests/test_torch_categorical_ce_readout.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux src/adapters/torch_categorical_ce_readout.py src/adapters/torch_decision.py tests/test_torch_categorical_ce_readout.py
```

Pre-review and guarded batches all four EXIT0,5.967632s and5.830978s respectively.
One final allocation argument explicitly pins reader coefficients to CPU,
preserving the tested default-CPU behavior. All four CPU-final static commands
EXIT0/6.3490388s; total18.1476488s under120s. Independent reviewer confirmed prior
acceptance remains applicable to this exact one-line change without additional
tools; changed Torch-default-device behavior was not separately executed. Normal
CPU evidence is reused. No static or numerical budget overrun/renewal here.
Older319's supporting static-batch violation remains preserved in its own report.

## Review, tracking failures and limits

Independent review found input copying preceded the owner pin, allowing a copying
callback to change the donor before its baseline. Fixed before acceptance: move
the copy inside the entered owner guard. The literal regression refuses hidden/
optimizer work and preserves the mutation. Also require exact native str roles
before equality, refusing a foreign role's comparison callbacks before admission.

Finite reviewer accepted the actual final source, test/cost receipts and direct
decoder extraction: exactly four read-only commands0.5702292s, all EXIT0; no
imports, models, tests, writes, network or children. Exception ended. Lead separately
reviewed source/procedure/results; self-review is not independent review.

Connected GitHub issue creation returned no issue/internal errors; the premature
assumed320 comment returned404, not success. Local pending protocol and parent18
start/milestones preserved the authorized scope and evidence. Authenticated
ordinary GitHub CLI fallback verified userOptimumAF, then
`gh issue create --repo OptimumAF/Circadian-Predictive-Coding --title 'R2.1-TW.CE: disposable readout learning and typed decisions' --body-file D:/Dev/Repositories/Circadian/artifacts/runs/torch-categorical-ce-readout-20261030/ISSUE_BODY.md`
returned320/EXIT0. Connected fetch/comment confirmed it, receipt6525675955.
Pending note is marked resolved; initial failures remain. One local receipt typo
was corrected immediately to that actual returned receipt before closeout.

Skipped full repository suites, installed wheel, clean clone, other native
families/cohorts, genuine four-task rollout and release checks. Changed decoder
logic is identical and method-independent; focused original-BP/native subset and
pure underflow/tie parity checks cover the extraction. No install/download/delete,
commit/push or global settings/cache change. All inherited user changes and old
negative/SPENT evidence remain intact. #318 temporary proof stays unchecked/open.

Next: freeze and review ONE existing-runner native source12/72→disposable CE fit
on all12 admitted old TRAIN rows→ALL four known-task rollouts, with original
source/actor/receipt/budget guards and separate closure/backward/inference costs.
No source scalar-JSON loader, task selection, alternative solver or retry. A
negative case remains valid. This genuine task witness is unfinished, while scoped
software learning/typed behavior is accepted. Fullgoal/#21/#18/R2.1 acquisition,
retention/recovery, scientific comparison and release requirements remain open.


Closeout: stage14files49335B under4MiB; scoped source/new-file
whitespace checks passed with LF/CRLF advisories only. Trackercreationfallback
is resolved with verified320receipt; all initial failed attempts remain in the
local pending-history note. FinalCPU allocation argument enforces reviewed
contract; actor/concurrent-owner serialization remains the caller's responsibility.
