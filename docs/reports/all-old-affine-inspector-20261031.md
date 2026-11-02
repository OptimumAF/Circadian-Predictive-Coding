# Inspect both all-old affine controls

R4.7-MG.ALL / [issue392](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/392).
The source-checkout inspector now projects all16 recorded games through the
existing private player. Software, actual saved export and code-level player
checks pass. Genuine desktop/narrow browser acceptance remains open under the
recorded #154 file-URL restriction; no workaround or browser launch occurred.

The explicit `portable_affine_matched_all_old_v1` route admits two eight-game
phases: F blue/red probe_test, then old A/B/C_train blue/red TRAIN. It validates
phase intent, roles, original8/180 endpoints, charges, clocks, layouts and source
pointers. B/C reuse the existing exact rotated/swapped layouts and `_episode`
validator. Present optional plans must match the fixed roster; missing optional
metadata stays unknown. Detached presentation data grants no authenticity or
resume authority.

Shared native-cost validation now accepts exactly4 or8 expected episodes. The
default remains4 resets/64 steps; explicit all-old uses8/128. The original
four-game output and labels remain unchanged. New player text identifies all old
rooms without claiming recovery or generalization. Original false/null/missing
proofs, paid failures, score/return values and costs remain visible. There is no
new renderer, learner, dependency, training, rollout or parameter reconstruction.

Changed source: `src/app/minigrid_private_affine_inspector.py`,
`src/app/minigrid_affine_inspector.py`, `src/app/key_door_inspector.py`,
`src/infra/minigrid_inspector.html`, plus `tests/test_all_old_affine_inspector.py`.
All unrelated dirty work and original scientific artifacts are preserved.

Executed from physical D:\Dev\Repositories\Circadian with TEMP/TMP/TMPDIR
at D:\Dev\_Temp. Stage `artifacts/runs/all-old-affine-inspector-20261031/` retains
before copies, protocol, logs, SPENT markers, fixtures, exports and receipts.

| Validation | Body seconds | Peak RSS bytes | Host seconds / exit |
| --- | ---: | ---: | --- |
| Initial55 combined cases | 1.8120341 | 56320000 | 1.9526988 / 0 |
| Two new cost boundaries | 1.1136627 | 55164928 | 1.248349 / 0 |
| Final38 all-old cases | 1.8334379 | 55504896 | 1.9842711 / 0 |
| Initial actual export | .4098618 | 42438656 | .5640174 / 0 |
| Final actual export | .4062778 | 43982848 | .5644599 / 0 |

There are62 distinct passing cases/95 executions,0 test failures or skips.
This is55 combined +2 added boundaries +38 final all-old cases; unchanged24
old-route checks are reused, not a fresh62-case combined run. Original saved
legacy-report test was deliberately deselected, avoiding an unrelated native
archive read. All guarded processes recorded no numerical/runtime import attempts.

```powershell
./artifacts/runs/all-old-affine-inspector-20261031/run-check.ps1 -Attempt first
./artifacts/runs/all-old-affine-inspector-20261031/run-check.ps1 -Attempt boundary -Selection test_should_keep_each_rosters_original_step_ceiling_on_partial_work
./artifacts/runs/all-old-affine-inspector-20261031/run-check.ps1 -Attempt final -Selection test_all_old_affine_inspector
./artifacts/runs/all-old-affine-inspector-20261031/run-export.ps1
./artifacts/runs/all-old-affine-inspector-20261031/run-export-final.ps1
```

Final export uses the existing public `scripts.build_key_door_inspector` CLI and
its strict bounded loader/exclusive writer. `all-old-final.html` is2764641 bytes
and is byte-identical to the first HTML exercised by the Node fixture. The second
allowed export verifies the final plan-consistency refinement, not a repeated
scientific experiment. The original1483873-byte #373 JSON stayed byte-identical
at the post-first-export check, SHA256
`587d112d8d9e5e796977fa93f446c87f63dc7e4b6b1f876691119aa6e3fb3748`.
Its embedded source equals the original JSON. Four original-source reads total
were used for orientation, first export, preservation check and final export.

The existing scripted DOM harness passed61 assertions in .060896s/70496256B.
It exercised both eight-option selectors, all16 recorded public-frame SVGs,
frame stepping, source dialog, unchanged2/8 versus8/8 stored goals and terminal
frames without an invented final prediction. This is a fake-DOM code test,
not a browser, screenshot, layout, accessibility, device or visual result.

```powershell
node --check artifacts/runs/all-old-affine-inspector-20261031/scripted-dom.cjs
node artifacts/runs/all-old-affine-inspector-20261031/scripted-dom.cjs
```

Scoped Ruff format/lint/mypy pass. Initial3 owned diagnostics were retained and
fixed by separate loop variable names, explicit layout keyword and a JSON type
annotation; initial3.0664709s, repaired six-file.9610327s. Added boundary/final
test/helper statics also pass. No ignored diagnostics or validator weakening.
Types use `--follow-imports=silent`; this is not a full repository type result.

```powershell
$files = @('src/app/minigrid_private_affine_inspector.py', 'src/app/minigrid_affine_inspector.py', 'src/app/key_door_inspector.py', 'tests/test_all_old_affine_inspector.py', 'artifacts/runs/all-old-affine-inspector-20261031/check_once.py', 'artifacts/runs/all-old-affine-inspector-20261031/export_once.py')
.venv/Scripts/python.exe -B -m ruff format --check @files
.venv/Scripts/python.exe -B -m ruff check @files
.venv/Scripts/python.exe -B -m mypy --cache-dir artifacts/runs/all-old-affine-inspector-20261031/mypy-cache --follow-imports=silent @files
```

Lead source/procedure/result review is separate from the finite independent
software/data-interpretation review recorded in stage `REVIEW.md` and
`reviewer-timings.json`: scoped SOFTWARE ACCEPT, two bounded reads/.8968700s,
no original native JSON reads. The peer directly parsed final HTML scalars and
confirmed all16 outcomes, roles, pointers and closures. Full issue392 remains
OPEN/PARTIAL. Original criterion2's broad replacement-refusal wording is not
fully satisfied: internal consistency and detachment pass, but authenticity,
equal-byte replacement detection and resume authority are neither implemented
nor proved by this viewer. Preserve that requirement unchecked; do not silently
reinterpret it as authentication. Browser acceptance also remains unchecked.
Full suite, Linux, clean clone, new wheel/install, native
learning/game replay, GPU/remote CI and release checks were skipped. No source
cohort, seed, metric or model was selected or changed. Original #391 partial
package qualification and #390X failed/spent native comparison remain unchanged.

This is a saved-data presentation improvement, not new adaptation, recovery,
generalization, matched-compute or circadian benefit evidence. The earlier #391
wheel predates these source changes. Broader parents and the full goal stay open.

Critical path: the existing learnability/probe/stream implementations and their
accepted results are reused; this increment supplies their missing saved-game
inspection route. Do not start another numbered-update, export, wheel or capture
increment by default. A concrete remaining software-quality dependency is the
typing failures retained by #390I in the learner/readout boundary
(`src/adapters/return_policy_evaluation.py` and `tests/test_recurrent_readout.py`).
The earlier nine diagnostics are historical, not freshly confirmed here.
Exact next action: read those two current files and reconcile their current
targeted mypy diagnostics, then fix typed interfaces with focused behavior
checks, preserving all learner math, protocol doses, seeds, metrics and the
spent native experiment. This selects a real quality-gate defect rather than
renewing a scientific budget or constructing another audit framework. A new
comparative experiment still requires its own small frozen protocol and review.

Final canonical comment6554529676; parent18/21/23 receipts6554533208/6554533418/6554533541. Owning issue remainsOPEN, criteria1/3/5 checked and2/4 unchecked. Independent exception ENDED. SESSION-END.json records scoped closure; final stage20668299 bytes within64MiB. No source change since final checks/review; full goalACTIVE.
