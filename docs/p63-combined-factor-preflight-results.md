# P6.3c7 combined/full-minus-one train-only results

Date: 2026-09-30. Checkout: `master` at
`86cd5bff71b9c70da94ddcf69d8f62316f2d3382`, with the preserved dirty
c5/c6 increments and this c7 addition. Runtime: Windows 11, Intel Core
i7-12700K CPU, Python 3.14.7, NumPy 2.4.6.

## Scope and decision

The [prospective contract](p63-combined-factor-preflight.md) and
[ADR-0147](adr/ADR-0147-compose-full-minus-one-before-new-growth-controls.md)
fixed protocol `continual_combined_factor_train_only_v1`, all 17 cells on
development seeds **263/269/271**, existing full switches and seven removals,
full-controlled replay references, periodic structure-only and planned
width-14 references before public training. No outer-selection or final
accuracy was read. The inner guard values below are decision evidence.

Two independently bounded public processes produced identical deterministic
result bytes. All **51 cells, 72 shared opportunities and 648 own decisions**
passed independent fact validation. Exact ordinary-PC/neutral parity holds
at initialization and after every wake/replay. B source construction follows
all 17 A models. Rejection restores complete snapshot fingerprints, including
RNG, chemistry, lineage, memory and clocks, while executed replay stays in
the work ledger.

**Complete c7 only.** C8 must freeze and globally verify scored development
against this saved all-seed gate. Required scheduled/random parent growth
controls (c9), independent confirmation and the original P6.3c/P6.3 matrix
remain unfinished. Current add-count policy still chooses usage-ranked
parents; this periodic structure row does not implement the missing control.
No seed, baseline, threshold or metric was tuned or selected from outcomes.

## Artifacts, identities and limits

Both ignored directories contain `combined-factor-preflight.request.json`,
`.result.json`, `.audit.json` and no failure:

- `artifacts/runs/p63-combined-factor-preflight/`
- `artifacts/runs/p63-combined-factor-preflight-repeat/`

The identical saved result SHA-256 is
`79a7d7e09f0ada01ff72d6e266ddee7576aca52316a0f9a8e2ab3dd402251d13`.
Request hashes are
`9001aeee45dceba0286f3bbd97eb1c77bab5fcf550b935588e365529dc2d44bf`
and
`6453651598f794926fe3d1de93febe4994e75615e61f50793233efda2ea54acb`.
Frozen manifest SHA-256 is
`729bf9df8472752f51696299373555339208af7b5add1892d14154d4f21ad04a`;
selected 26-source map digest is
`457df802859b68e42b2a54a31e7a9a49f4e893d07062074bb22d3069a96c737f`;
adapter byte digest is
`9c3d640638b48d7a2edc9cc4585e205cdb3f2d86bd1c72a6e7a8438bd0f3d404`.
This selected map binds the older c5 map and six added dependencies, not
the complete repository/dependency tree. Older c5/c6 bytes remain unchanged.

| Process | Elapsed seconds | Observed peak RSS bytes | RSS samples |
|---|---|---|---|
| First | 2.902358 | 56,459,264 | 159 |
| Repeat | 2.949283 | 58,642,432 | 158 |

Each elapsed time is below **120 seconds** and each sampled whole-worker
RSS peak below **256 MiB**, with 5 ms sampling. RSS covers training,
independent validation and deterministic result serialization, including
runtime/fact allocations. Final stdout transport and parent artifact writing
are outside the labeled sampling interval; short peaks can be missed.
These are whole-worker observations, not per-arm resource attribution.

## Work, capacity and component facts

Each cell executed **24 wake optimizer updates, 1,296 wake example
presentations**. PC/circadian cells used 48 wake inference loops and 2,592
wake example-inference iterations; backprop used zero inference loops.
Total work is **1,530 executed optimizer updates = 1,224 wake + 280 applied
replay + 26 rejected replay**, below the prospective maximum **1,548** and
hard cap **1,600**. Seed totals are 510/504/516. Rejected replay still used
52 latent loops. There were 144 guarded attempts, 288 inner accuracy calls
on 5,184 examples, 126 own commits and 18 rollbacks.

Available labeled memory is eight rows/192 array bytes for the shared FIFO
and each of eleven circadian buffers: **2,304 persistent labeled-array
bytes per seed**. The scope excludes parameters, metadata and temporary
copies. Offered IDs, order and contents match at every boundary; backprop/PC
receive detached copies. Only full-controller commits supply the three
matched consumers, with **10/8/12** replay updates each on seeds 263/269/271.
The neutral replay consumer has no independent guard, explaining its
commits with zero own attempts.

Initial width/parameters are 8/33 for the first 15 cells and 14/57 for the
two prospectively wider references. Dynamic full/removal ceilings are
14/57 (except fixed no-structure); the separate structure-only ceiling is
9/37. Peaks include proposed transient growth even after rollback. The
next table publishes every cost/capacity cell. Prunes count actual proposed/
applied removals; all scheduled-prune lists are empty.

| Seed | Arm | Width final/peak | Parameters final/peak | Replay applied/rejected | Sleep commits/own attempts | Guard rows | Splits proposed/applied | Prunes proposed/applied |
|---|---|---|---|---|---|---|---|---|
| 263 | backprop_off | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 263 | pc_off | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 263 | neutral_off | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 263 | backprop_full_replay | 8/8 | 33/33 | 10/0 | 0/0 | 0 | 0/0 | 0/0 |
| 263 | pc_full_replay | 8/8 | 33/33 | 10/0 | 0/0 | 0 | 0/0 | 0/0 |
| 263 | neutral_full_replay | 8/8 | 33/33 | 10/0 | 5/0 | 0 | 0/0 | 0/0 |
| 263 | full | 6/8 | 25/33 | 10/2 | 5/6 | 216 | 0/0 | 3/2 |
| 263 | minus_replay | 5/8 | 21/33 | 0/0 | 5/6 | 216 | 0/0 | 4/3 |
| 263 | minus_gating | 6/8 | 25/33 | 10/2 | 5/6 | 216 | 0/0 | 3/2 |
| 263 | minus_structure | 8/8 | 33/33 | 12/0 | 6/6 | 216 | 0/0 | 0/0 |
| 263 | minus_schedule | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 263 | minus_difficulty | 6/8 | 25/33 | 10/2 | 5/6 | 216 | 0/0 | 3/2 |
| 263 | minus_homeostasis | 6/8 | 25/33 | 10/2 | 5/6 | 216 | 0/0 | 3/2 |
| 263 | minus_reset | 6/8 | 25/33 | 12/0 | 6/6 | 216 | 0/0 | 2/2 |
| 263 | periodic_structure_only | 7/9 | 29/37 | 0/0 | 6/6 | 216 | 3/3 | 4/4 |
| 263 | backprop_14_off | 14/14 | 57/57 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 263 | pc_14_off | 14/14 | 57/57 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 269 | backprop_off | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 269 | pc_off | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 269 | neutral_off | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 269 | backprop_full_replay | 8/8 | 33/33 | 8/0 | 0/0 | 0 | 0/0 | 0/0 |
| 269 | pc_full_replay | 8/8 | 33/33 | 8/0 | 0/0 | 0 | 0/0 | 0/0 |
| 269 | neutral_full_replay | 8/8 | 33/33 | 8/0 | 4/0 | 0 | 0/0 | 0/0 |
| 269 | full | 6/8 | 25/33 | 8/4 | 4/6 | 216 | 0/0 | 4/2 |
| 269 | minus_replay | 5/8 | 21/33 | 0/0 | 5/6 | 216 | 0/0 | 4/3 |
| 269 | minus_gating | 7/8 | 29/33 | 8/4 | 4/6 | 216 | 0/0 | 3/1 |
| 269 | minus_structure | 8/8 | 33/33 | 12/0 | 6/6 | 216 | 0/0 | 0/0 |
| 269 | minus_schedule | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 269 | minus_difficulty | 6/8 | 25/33 | 8/4 | 4/6 | 216 | 0/0 | 4/2 |
| 269 | minus_homeostasis | 6/8 | 25/33 | 8/4 | 4/6 | 216 | 0/0 | 4/2 |
| 269 | minus_reset | 7/8 | 29/33 | 12/0 | 6/6 | 216 | 0/0 | 1/1 |
| 269 | periodic_structure_only | 8/9 | 33/37 | 0/0 | 3/6 | 216 | 3/1 | 4/1 |
| 269 | backprop_14_off | 14/14 | 57/57 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 269 | pc_14_off | 14/14 | 57/57 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 271 | backprop_off | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 271 | pc_off | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 271 | neutral_off | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 271 | backprop_full_replay | 8/8 | 33/33 | 12/0 | 0/0 | 0 | 0/0 | 0/0 |
| 271 | pc_full_replay | 8/8 | 33/33 | 12/0 | 0/0 | 0 | 0/0 | 0/0 |
| 271 | neutral_full_replay | 8/8 | 33/33 | 12/0 | 6/0 | 0 | 0/0 | 0/0 |
| 271 | full | 6/8 | 25/33 | 12/0 | 6/6 | 216 | 0/0 | 2/2 |
| 271 | minus_replay | 6/8 | 25/33 | 0/0 | 6/6 | 216 | 0/0 | 2/2 |
| 271 | minus_gating | 6/8 | 25/33 | 12/0 | 6/6 | 216 | 0/0 | 2/2 |
| 271 | minus_structure | 8/8 | 33/33 | 12/0 | 6/6 | 216 | 0/0 | 0/0 |
| 271 | minus_schedule | 8/8 | 33/33 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 271 | minus_difficulty | 6/8 | 25/33 | 12/0 | 6/6 | 216 | 0/0 | 2/2 |
| 271 | minus_homeostasis | 6/8 | 25/33 | 10/2 | 5/6 | 216 | 0/0 | 3/2 |
| 271 | minus_reset | 7/8 | 29/33 | 12/0 | 6/6 | 216 | 0/0 | 1/1 |
| 271 | periodic_structure_only | 7/9 | 29/37 | 0/0 | 6/6 | 216 | 3/3 | 4/4 |
| 271 | backprop_14_off | 14/14 | 57/57 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |
| 271 | pc_14_off | 14/14 | 57/57 | 0/0 | 0/0 | 0 | 0/0 | 0/0 |

The frozen default full/removal cells proposed **no split** on any seed.
Their prunes and replay/guard work varied across component removals, so
these treatments do not have equal compute or capacity. The structure-only
row proposed nine splits and twelve removals, committing seven splits and
nine removals; its narrower bounds and previously declared 0/1 thresholds
differ from the full defaults. Across all controllers there were nine
proposed/seven applied splits and 62 proposed/44 applied removals. Keep the
inactive full split result; it does not justify threshold tuning.

No-schedule cells attempted zero sleep and applied zero replay or topology
change. No-replay cells retained the same available memory but executed zero
replay. No-structure cells stayed at width eight. Neutral, no-gating and
structure-only cells had plasticity one; difficulty-disabled cells had
scale one. Existing full difficulty modulation also weights importance
history, so the later removal contrast cannot be called a pure wake
learning-rate effect.

### Every rejected proposal

All 18 official rollbacks occurred during B. Every row restored identical
complete pre/post snapshot and parameter fingerprints and lineage, with
proposed IDs and executed replay retained in facts. Inner accuracy is rounded
here to six decimals; full precision remains in the result. Child ID 9 is
proposed again after rejection because rollback restores the ID counter and
RNG. These inner decision metrics are not outer performance claims.

| Seed | Phase:epoch | Arm | Inner guard pre/post | Rejected replay | Proposed splits (parent, child) | Proposed removed IDs |
|---|---|---|---|---|---|---|
| 263 | b:8 | full | 0.416667/0.333333 | 2 | [] | [4] |
| 263 | b:8 | minus_gating | 0.500000/0.416667 | 2 | [] | [4] |
| 263 | b:8 | minus_difficulty | 0.583333/0.416667 | 2 | [] | [4] |
| 263 | b:8 | minus_homeostasis | 0.416667/0.333333 | 2 | [] | [4] |
| 263 | b:12 | minus_replay | 1.000000/0.583333 | 0 | [] | [4] |
| 269 | b:4 | minus_gating | 0.916667/0.833333 | 2 | [] | [0] |
| 269 | b:4 | periodic_structure_only | 0.916667/0.833333 | 0 | [[5, 9]] | [0] |
| 269 | b:8 | full | 0.916667/0.833333 | 2 | [] | [3] |
| 269 | b:8 | minus_gating | 1.000000/0.833333 | 2 | [] | [3] |
| 269 | b:8 | minus_difficulty | 0.916667/0.833333 | 2 | [] | [3] |
| 269 | b:8 | minus_homeostasis | 0.916667/0.833333 | 2 | [] | [3] |
| 269 | b:8 | periodic_structure_only | 0.916667/0.833333 | 0 | [[5, 9]] | [0] |
| 269 | b:12 | full | 0.916667/0.833333 | 2 | [] | [3] |
| 269 | b:12 | minus_replay | 0.833333/0.750000 | 0 | [] | [4] |
| 269 | b:12 | minus_difficulty | 0.916667/0.833333 | 2 | [] | [3] |
| 269 | b:12 | minus_homeostasis | 0.916667/0.833333 | 2 | [] | [3] |
| 269 | b:12 | periodic_structure_only | 0.916667/0.833333 | 0 | [] | [3] |
| 271 | b:4 | minus_homeostasis | 1.000000/0.916667 | 2 | [] | [0] |

### Every circadian wake scale range

Each plasticity entry summarizes the observed per-epoch minimum across 24
wake batches. Difficulty scale entries summarize all recorded wake scales.
No observed range was used to change the frozen .75–1.5 clipping or settings.

| Seed | Arm | Minimum plasticity range | Difficulty scale range |
|---|---|---|---|
| 263 | neutral_off | 1.000000..1.000000 | 1.000000..1.000000 |
| 263 | neutral_full_replay | 1.000000..1.000000 | 1.000000..1.000000 |
| 263 | full | 0.878617..0.990238 | 0.834784..1.000000 |
| 263 | minus_replay | 0.929913..0.990238 | 0.836302..1.000000 |
| 263 | minus_gating | 1.000000..1.000000 | 0.830084..1.000000 |
| 263 | minus_structure | 0.911521..0.990238 | 0.834784..1.001711 |
| 263 | minus_schedule | 0.794619..0.990238 | 0.838771..1.000000 |
| 263 | minus_difficulty | 0.878099..0.990238 | 1.000000..1.000000 |
| 263 | minus_homeostasis | 0.877148..0.990238 | 0.834206..1.002706 |
| 263 | minus_reset | 0.717600..0.990238 | 0.838287..1.000310 |
| 263 | periodic_structure_only | 1.000000..1.000000 | 1.000000..1.000000 |
| 269 | neutral_off | 1.000000..1.000000 | 1.000000..1.000000 |
| 269 | neutral_full_replay | 1.000000..1.000000 | 1.000000..1.000000 |
| 269 | full | 0.890328..0.987166 | 0.750000..1.000000 |
| 269 | minus_replay | 0.919865..0.987166 | 0.750000..1.000000 |
| 269 | minus_gating | 1.000000..1.000000 | 0.750000..1.000000 |
| 269 | minus_structure | 0.903134..0.987166 | 0.750000..1.000000 |
| 269 | minus_schedule | 0.783422..0.987166 | 0.750000..1.000000 |
| 269 | minus_difficulty | 0.889909..0.987166 | 1.000000..1.000000 |
| 269 | minus_homeostasis | 0.888643..0.987166 | 0.750000..1.000000 |
| 269 | minus_reset | 0.711262..0.987166 | 0.750000..1.000000 |
| 269 | periodic_structure_only | 1.000000..1.000000 | 1.000000..1.000000 |
| 271 | neutral_off | 1.000000..1.000000 | 1.000000..1.000000 |
| 271 | neutral_full_replay | 1.000000..1.000000 | 1.000000..1.000000 |
| 271 | full | 0.916720..0.989345 | 0.813663..1.000000 |
| 271 | minus_replay | 0.928513..0.989345 | 0.824861..1.000000 |
| 271 | minus_gating | 1.000000..1.000000 | 0.805637..1.000000 |
| 271 | minus_structure | 0.916720..0.989345 | 0.787766..1.000000 |
| 271 | minus_schedule | 0.790518..0.989345 | 0.790855..1.000000 |
| 271 | minus_difficulty | 0.916507..0.989345 | 1.000000..1.000000 |
| 271 | minus_homeostasis | 0.879693..0.989345 | 0.798679..1.000000 |
| 271 | minus_reset | 0.720305..0.989345 | 0.817303..1.000000 |
| 271 | periodic_structure_only | 1.000000..1.000000 | 1.000000..1.000000 |

## Validation and reproduction

Read-only artifact checks parsed finite JSON; independently rederived every
model/decision/role/work/clock/memory/lineage fact; checked frozen
manifest/source/adapter and request/result/audit identities; checked costs,
wall/RSS limits and failure absence; and compared saved result bytes exactly.
Report tables are generated from those verified facts.

Five app tests verify surgical component switches, changed-manifest failure
before data, all-seed outer/final/arrival seals, exact repeats/parity and forged
work/role/ID/capacity/budget rejection. A forced all-rejected run instruments
the real core replay update path: **216 replay executions, 1,440 total
updates, 288 guard calls**, complete state restoration and no consumer replay.
Four public CLI cases verify all artifacts/source identities and duplicate
refusal, changed-source preflight, and timeout/nonfinite failure artifacts
without a false result/audit.

Commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_combined_factor_preflight --output-dir artifacts/runs/p63-combined-factor-preflight
.\.venv\Scripts\python.exe -m scripts.run_p63_combined_factor_preflight --output-dir artifacts/runs/p63-combined-factor-preflight-repeat
.\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_combined_factor_preflight.py tests/test_p63_combined_factor_preflight_cli.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
```

Both public runs exited 0. The nine new tests passed in **14.11 seconds**;
the final related regression command in the development log passed **76
tests in 30.65 seconds**, with no skips. Ruff, mypy (345 source files) and
the six new Python files' format checks passed. Full CPU suite, CUDA, large
sweeps, outer scoring, confirmation and final release were skipped; the
focused gate covers the changed train-only path and shared controls.

**Exact next action:** freeze c8's scored protocol against the saved c7
result hash above, all 51 cells, A/B primary metrics, component-removal and
matched-reference contrasts and unequal cost/capacity facts. Implement an
exact all-seed train-fact and checkpoint comparison before the first outer
input/label access, without changing these frozen sources or settings.
Keep confirmation seeds 277/281/283/293/307/311/313/317/331/337 and all earlier
reservations unused, final roles sealed, and c9 growth controls unchecked.
