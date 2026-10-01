# P6.3c9c paired parent-control development results

Date: 2026-09-30. Protocol: `continual_parent_factor_outer_development_v1`.
Checkout: `master` at `86cd5bff71b9c70da94ddcf69d8f62316f2d3382`, preserving
earlier dirty work. NumPy CPU, Windows 11, Intel Core i7-12700K,
Python 3.14.7 / NumPy 2.4.6. This is a three-seed outer-selection development
pilot; confirmation and final roles remain unopened. The
[prospective contract](p63-parent-factor-development.md) fixes every cell,
primary metric, pair, source/role and cost before scoring.

## Verified gates and saved identities

Both fresh bounded public runs completed all 24 cells/60 pairs with exactly
72 outer forward calls/1,440 examples. Before the first outer value, every
c9b train fact matched globally and every held after-A/after-B parameter,
width and complete circadian snapshot matched, including both RNGs and
selector/cursor/decision state. Copies remain unchanged after scoring.
No training setting, seed, metric or reference was selected from results.

Both ignored `artifacts/runs/p63-parent-factor-development{,-repeat}/`
contain `parent-factor-development.request.json`, `.result.json`,
`.audit.json` and no failure. Deterministic results are byte-identical:
`7f76e793eaf123b30116f57d68aa755962d63a56e4bba4b29703ae37b16e2040`.

| Identity | First | Repeat |
|---|---|---|
| Request SHA-256 | `c4654667f4f358cb36bed0a767232181c69f732729fd57a09773fad2f8421fc8` | `75631e6c286958ad43b576b09a1aa76d9e988f9d8b5949cb2d81fe9ba4d83009` |
| Audit SHA-256 | `c98f6cdd6fa5295b08f91dd9f5daeded9a74c393e2c6b1777182371bcaa7dc79` | `6fe0a947685c1157e4f1153c98f9b9286f2d930801a6f788534832cffdb4f1c4` |
| Whole-child elapsed seconds | 1.841334 | 1.844780 |
| Observed worker RSS peak bytes | 56,631,296 | 57,561,088 |
| RSS samples at 5 ms | 91 | 88 |

Manifest SHA-256: `a7938028ed3c9279ef74a5f9a2550012927e4bb626b72672861ae64aa71497c9`.
Selected 34-source map: `b0a86792167a9965e38007b0a0a3cd5a7ee3d229495df4950201198b545be55b`.
Scored adapter: `be57cf809149af570876d75d6f924ec073ee1a35a1d96b68ae4fc7b59129e9c0`.
Canonical c9b request/result/audit byte pins are in the contract. The complete
embedded c9b result is unchanged at SHA `555fc2fe...5a1874`. This map selects
sources; it does not cover a full transitive dependency tree.

## Every accuracy cell

All accuracies use outer selection. Primary mean is equal-task A/B after B;
positive signed forgetting means A deterioration. Optional retention is
null at zero A-after-A; values above one mean A improvement. Printed values
are rounded to six decimals; saved finite JSON retains full precision.

| Seed | Cell | A after A | A after B | B after B | Final mean | Signed forgetting A | Retention A |
|---|---|---:|---:|---:|---:|---:|---:|
| 347 | `backprop_off` | 0.958333 | 0.958333 | 1.000000 | 0.979167 | 0.000000 | 1.000000 |
| 347 | `pc_off` | 0.958333 | 0.958333 | 1.000000 | 0.979167 | 0.000000 | 1.000000 |
| 347 | `neutral_off` | 0.958333 | 0.958333 | 1.000000 | 0.979167 | 0.000000 | 1.000000 |
| 347 | `usage_growth` | 0.958333 | 0.958333 | 1.000000 | 0.979167 | 0.000000 | 1.000000 |
| 347 | `scheduled_growth` | 0.958333 | 0.958333 | 1.000000 | 0.979167 | 0.000000 | 1.000000 |
| 347 | `random_growth` | 0.958333 | 0.958333 | 1.000000 | 0.979167 | 0.000000 | 1.000000 |
| 347 | `backprop_13_off` | 0.958333 | 0.958333 | 1.000000 | 0.979167 | 0.000000 | 1.000000 |
| 347 | `pc_13_off` | 0.958333 | 0.958333 | 1.000000 | 0.979167 | 0.000000 | 1.000000 |
| 349 | `backprop_off` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 349 | `pc_off` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 349 | `neutral_off` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 349 | `usage_growth` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 349 | `scheduled_growth` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 349 | `random_growth` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 349 | `backprop_13_off` | 1.000000 | 1.000000 | 0.833333 | 0.916667 | 0.000000 | 1.000000 |
| 349 | `pc_13_off` | 0.875000 | 1.000000 | 0.750000 | 0.875000 | -0.125000 | 1.142857 |
| 353 | `backprop_off` | 0.958333 | 0.916667 | 0.833333 | 0.875000 | 0.041667 | 0.956522 |
| 353 | `pc_off` | 0.875000 | 0.916667 | 0.833333 | 0.875000 | -0.041667 | 1.047619 |
| 353 | `neutral_off` | 0.875000 | 0.916667 | 0.833333 | 0.875000 | -0.041667 | 1.047619 |
| 353 | `usage_growth` | 0.958333 | 0.875000 | 0.750000 | 0.812500 | 0.083333 | 0.913043 |
| 353 | `scheduled_growth` | 0.875000 | 0.916667 | 0.750000 | 0.833333 | -0.041667 | 1.047619 |
| 353 | `random_growth` | 0.875000 | 0.875000 | 0.750000 | 0.812500 | 0.000000 | 1.000000 |
| 353 | `backprop_13_off` | 0.958333 | 0.958333 | 0.916667 | 0.937500 | 0.000000 | 1.000000 |
| 353 | `pc_13_off` | 0.958333 | 0.958333 | 0.833333 | 0.895833 | 0.000000 | 1.000000 |

## All 60 ordered paired rows

Differences are left minus right. Positive final mean favors left; positive
forgetting means more A deterioration, subject to the two A accuracies.

| Seed | Left | Right | A after A delta | A after B delta | B after B delta | Final mean delta | Forgetting delta |
|---|---|---|---:|---:|---:|---:|---:|
| 347 | `usage_growth` | `scheduled_growth` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `usage_growth` | `random_growth` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `scheduled_growth` | `random_growth` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `usage_growth` | `backprop_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `usage_growth` | `pc_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `usage_growth` | `neutral_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `usage_growth` | `backprop_13_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `usage_growth` | `pc_13_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `scheduled_growth` | `backprop_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `scheduled_growth` | `pc_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `scheduled_growth` | `neutral_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `scheduled_growth` | `backprop_13_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `scheduled_growth` | `pc_13_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `random_growth` | `backprop_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `random_growth` | `pc_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `random_growth` | `neutral_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `random_growth` | `backprop_13_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `random_growth` | `pc_13_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `backprop_13_off` | `backprop_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 347 | `pc_13_off` | `pc_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `usage_growth` | `scheduled_growth` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `usage_growth` | `random_growth` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `scheduled_growth` | `random_growth` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `usage_growth` | `backprop_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `usage_growth` | `pc_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `usage_growth` | `neutral_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `usage_growth` | `backprop_13_off` | 0.000000 | 0.000000 | 0.166667 | 0.083333 | 0.000000 |
| 349 | `usage_growth` | `pc_13_off` | 0.125000 | 0.000000 | 0.250000 | 0.125000 | 0.125000 |
| 349 | `scheduled_growth` | `backprop_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `scheduled_growth` | `pc_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `scheduled_growth` | `neutral_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `scheduled_growth` | `backprop_13_off` | 0.000000 | 0.000000 | 0.166667 | 0.083333 | 0.000000 |
| 349 | `scheduled_growth` | `pc_13_off` | 0.125000 | 0.000000 | 0.250000 | 0.125000 | 0.125000 |
| 349 | `random_growth` | `backprop_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `random_growth` | `pc_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `random_growth` | `neutral_off` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 349 | `random_growth` | `backprop_13_off` | 0.000000 | 0.000000 | 0.166667 | 0.083333 | 0.000000 |
| 349 | `random_growth` | `pc_13_off` | 0.125000 | 0.000000 | 0.250000 | 0.125000 | 0.125000 |
| 349 | `backprop_13_off` | `backprop_off` | 0.000000 | 0.000000 | -0.166667 | -0.083333 | 0.000000 |
| 349 | `pc_13_off` | `pc_off` | -0.125000 | 0.000000 | -0.250000 | -0.125000 | -0.125000 |
| 353 | `usage_growth` | `scheduled_growth` | 0.083333 | -0.041667 | 0.000000 | -0.020833 | 0.125000 |
| 353 | `usage_growth` | `random_growth` | 0.083333 | 0.000000 | 0.000000 | 0.000000 | 0.083333 |
| 353 | `scheduled_growth` | `random_growth` | 0.000000 | 0.041667 | 0.000000 | 0.020833 | -0.041667 |
| 353 | `usage_growth` | `backprop_off` | 0.000000 | -0.041667 | -0.083333 | -0.062500 | 0.041667 |
| 353 | `usage_growth` | `pc_off` | 0.083333 | -0.041667 | -0.083333 | -0.062500 | 0.125000 |
| 353 | `usage_growth` | `neutral_off` | 0.083333 | -0.041667 | -0.083333 | -0.062500 | 0.125000 |
| 353 | `usage_growth` | `backprop_13_off` | 0.000000 | -0.083333 | -0.166667 | -0.125000 | 0.083333 |
| 353 | `usage_growth` | `pc_13_off` | 0.000000 | -0.083333 | -0.083333 | -0.083333 | 0.083333 |
| 353 | `scheduled_growth` | `backprop_off` | -0.083333 | 0.000000 | -0.083333 | -0.041667 | -0.083333 |
| 353 | `scheduled_growth` | `pc_off` | 0.000000 | 0.000000 | -0.083333 | -0.041667 | 0.000000 |
| 353 | `scheduled_growth` | `neutral_off` | 0.000000 | 0.000000 | -0.083333 | -0.041667 | 0.000000 |
| 353 | `scheduled_growth` | `backprop_13_off` | -0.083333 | -0.041667 | -0.166667 | -0.104167 | -0.041667 |
| 353 | `scheduled_growth` | `pc_13_off` | -0.083333 | -0.041667 | -0.083333 | -0.062500 | -0.041667 |
| 353 | `random_growth` | `backprop_off` | -0.083333 | -0.041667 | -0.083333 | -0.062500 | -0.041667 |
| 353 | `random_growth` | `pc_off` | 0.000000 | -0.041667 | -0.083333 | -0.062500 | 0.041667 |
| 353 | `random_growth` | `neutral_off` | 0.000000 | -0.041667 | -0.083333 | -0.062500 | 0.041667 |
| 353 | `random_growth` | `backprop_13_off` | -0.083333 | -0.083333 | -0.166667 | -0.125000 | 0.000000 |
| 353 | `random_growth` | `pc_13_off` | -0.083333 | -0.083333 | -0.083333 | -0.083333 | 0.000000 |
| 353 | `backprop_13_off` | `backprop_off` | 0.000000 | 0.041667 | 0.083333 | 0.062500 | -0.041667 |
| 353 | `pc_13_off` | `pc_off` | 0.083333 | 0.041667 | 0.000000 | 0.020833 | 0.041667 |

## Descriptive paired means and sample SD

Each pair uses all three seeds, the replication units. These are descriptive
development summaries, with no confirmatory interval or significance claim.

| Left | Right | Final mean delta: mean ± sample SD | Forgetting delta: mean ± sample SD |
|---|---|---:|---:|
| `usage_growth` | `scheduled_growth` | -0.006944 ± 0.012028 | 0.041667 ± 0.072169 |
| `usage_growth` | `random_growth` | 0.000000 ± 0.000000 | 0.027778 ± 0.048113 |
| `scheduled_growth` | `random_growth` | 0.006944 ± 0.012028 | -0.013889 ± 0.024056 |
| `usage_growth` | `backprop_off` | -0.020833 ± 0.036084 | 0.013889 ± 0.024056 |
| `usage_growth` | `pc_off` | -0.020833 ± 0.036084 | 0.041667 ± 0.072169 |
| `usage_growth` | `neutral_off` | -0.020833 ± 0.036084 | 0.041667 ± 0.072169 |
| `usage_growth` | `backprop_13_off` | -0.013889 ± 0.104859 | 0.027778 ± 0.048113 |
| `usage_growth` | `pc_13_off` | 0.013889 ± 0.104859 | 0.069444 ± 0.063647 |
| `scheduled_growth` | `backprop_off` | -0.013889 ± 0.024056 | -0.027778 ± 0.048113 |
| `scheduled_growth` | `pc_off` | -0.013889 ± 0.024056 | 0.000000 ± 0.000000 |
| `scheduled_growth` | `neutral_off` | -0.013889 ± 0.024056 | 0.000000 ± 0.000000 |
| `scheduled_growth` | `backprop_13_off` | -0.006944 ± 0.093943 | -0.013889 ± 0.024056 |
| `scheduled_growth` | `pc_13_off` | 0.020833 ± 0.095470 | 0.027778 ± 0.086736 |
| `random_growth` | `backprop_off` | -0.020833 ± 0.036084 | -0.013889 ± 0.024056 |
| `random_growth` | `pc_off` | -0.020833 ± 0.036084 | 0.013889 ± 0.024056 |
| `random_growth` | `neutral_off` | -0.020833 ± 0.036084 | 0.013889 ± 0.024056 |
| `random_growth` | `backprop_13_off` | -0.013889 ± 0.104859 | 0.000000 ± 0.000000 |
| `random_growth` | `pc_13_off` | 0.013889 ± 0.104859 | 0.041667 ± 0.072169 |
| `backprop_13_off` | `backprop_off` | -0.006944 ± 0.073164 | -0.013889 ± 0.024056 |
| `pc_13_off` | `pc_off` | -0.034722 ± 0.078874 | -0.027778 ± 0.086736 |

| Left | Right | A after A delta: mean ± sample SD | A after B delta: mean ± sample SD | B after B delta: mean ± sample SD |
|---|---|---:|---:|---:|
| `usage_growth` | `scheduled_growth` | 0.027778 ± 0.048113 | -0.013889 ± 0.024056 | 0.000000 ± 0.000000 |
| `usage_growth` | `random_growth` | 0.027778 ± 0.048113 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| `scheduled_growth` | `random_growth` | 0.000000 ± 0.000000 | 0.013889 ± 0.024056 | 0.000000 ± 0.000000 |
| `usage_growth` | `backprop_off` | 0.000000 ± 0.000000 | -0.013889 ± 0.024056 | -0.027778 ± 0.048113 |
| `usage_growth` | `pc_off` | 0.027778 ± 0.048113 | -0.013889 ± 0.024056 | -0.027778 ± 0.048113 |
| `usage_growth` | `neutral_off` | 0.027778 ± 0.048113 | -0.013889 ± 0.024056 | -0.027778 ± 0.048113 |
| `usage_growth` | `backprop_13_off` | 0.000000 ± 0.000000 | -0.027778 ± 0.048113 | 0.000000 ± 0.166667 |
| `usage_growth` | `pc_13_off` | 0.041667 ± 0.072169 | -0.027778 ± 0.048113 | 0.055556 ± 0.173472 |
| `scheduled_growth` | `backprop_off` | -0.027778 ± 0.048113 | 0.000000 ± 0.000000 | -0.027778 ± 0.048113 |
| `scheduled_growth` | `pc_off` | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | -0.027778 ± 0.048113 |
| `scheduled_growth` | `neutral_off` | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | -0.027778 ± 0.048113 |
| `scheduled_growth` | `backprop_13_off` | -0.027778 ± 0.048113 | -0.013889 ± 0.024056 | 0.000000 ± 0.166667 |
| `scheduled_growth` | `pc_13_off` | 0.013889 ± 0.104859 | -0.013889 ± 0.024056 | 0.055556 ± 0.173472 |
| `random_growth` | `backprop_off` | -0.027778 ± 0.048113 | -0.013889 ± 0.024056 | -0.027778 ± 0.048113 |
| `random_growth` | `pc_off` | 0.000000 ± 0.000000 | -0.013889 ± 0.024056 | -0.027778 ± 0.048113 |
| `random_growth` | `neutral_off` | 0.000000 ± 0.000000 | -0.013889 ± 0.024056 | -0.027778 ± 0.048113 |
| `random_growth` | `backprop_13_off` | -0.027778 ± 0.048113 | -0.027778 ± 0.048113 | 0.000000 ± 0.166667 |
| `random_growth` | `pc_13_off` | 0.013889 ± 0.104859 | -0.027778 ± 0.048113 | 0.055556 ± 0.173472 |
| `backprop_13_off` | `backprop_off` | 0.000000 ± 0.000000 | 0.013889 ± 0.024056 | -0.027778 ± 0.127294 |
| `pc_13_off` | `pc_off` | -0.013889 ± 0.104859 | 0.013889 ± 0.024056 | -0.083333 ± 0.144338 |

## Capacity and executed cost beside the outcomes

The complete [unscored cost/parent report](p63-parent-factor-preflight-results.md)
and embedded train facts retain every width/parameter/history, retained row,
guard proposal/commit and selector hash. No replay runs. All three growth
cells have five committed splits/six guarded sleeps per seed here, with
initial width eight, after-A eleven and final/peak thirteen.

| Seed | Cell | Width initial / after A / final / peak | Parameters initial / final / peak |
|---|---|---|---|
| 347 | `backprop_off` | 8 / 8 / 8 / 8 | 33 / 33 / 33 |
| 347 | `pc_off` | 8 / 8 / 8 / 8 | 33 / 33 / 33 |
| 347 | `neutral_off` | 8 / 8 / 8 / 8 | 33 / 33 / 33 |
| 347 | `usage_growth` | 8 / 11 / 13 / 13 | 33 / 53 / 53 |
| 347 | `scheduled_growth` | 8 / 11 / 13 / 13 | 33 / 53 / 53 |
| 347 | `random_growth` | 8 / 11 / 13 / 13 | 33 / 53 / 53 |
| 347 | `backprop_13_off` | 13 / 13 / 13 / 13 | 53 / 53 / 53 |
| 347 | `pc_13_off` | 13 / 13 / 13 / 13 | 53 / 53 / 53 |
| 349 | `backprop_off` | 8 / 8 / 8 / 8 | 33 / 33 / 33 |
| 349 | `pc_off` | 8 / 8 / 8 / 8 | 33 / 33 / 33 |
| 349 | `neutral_off` | 8 / 8 / 8 / 8 | 33 / 33 / 33 |
| 349 | `usage_growth` | 8 / 11 / 13 / 13 | 33 / 53 / 53 |
| 349 | `scheduled_growth` | 8 / 11 / 13 / 13 | 33 / 53 / 53 |
| 349 | `random_growth` | 8 / 11 / 13 / 13 | 33 / 53 / 53 |
| 349 | `backprop_13_off` | 13 / 13 / 13 / 13 | 53 / 53 / 53 |
| 349 | `pc_13_off` | 13 / 13 / 13 / 13 | 53 / 53 / 53 |
| 353 | `backprop_off` | 8 / 8 / 8 / 8 | 33 / 33 / 33 |
| 353 | `pc_off` | 8 / 8 / 8 / 8 | 33 / 33 / 33 |
| 353 | `neutral_off` | 8 / 8 / 8 / 8 | 33 / 33 / 33 |
| 353 | `usage_growth` | 8 / 11 / 13 / 13 | 33 / 53 / 53 |
| 353 | `scheduled_growth` | 8 / 11 / 13 / 13 | 33 / 53 / 53 |
| 353 | `random_growth` | 8 / 11 / 13 / 13 | 33 / 53 / 53 |
| 353 | `backprop_13_off` | 13 / 13 / 13 / 13 | 53 / 53 / 53 |
| 353 | `pc_13_off` | 13 / 13 / 13 / 13 | 53 / 53 / 53 |

| Seed | Cell | Wake updates / examples | Latent loops / example iterations | Guard attempts / calls / examples |
|---|---|---|---|---|
| 347 | `backprop_off` | 24 / 1296 | 0 / 0 | 0 / 0 / 0 |
| 347 | `pc_off` | 24 / 1296 | 48 / 2592 | 0 / 0 / 0 |
| 347 | `neutral_off` | 24 / 1296 | 48 / 2592 | 0 / 0 / 0 |
| 347 | `usage_growth` | 24 / 1296 | 48 / 2592 | 6 / 12 / 216 |
| 347 | `scheduled_growth` | 24 / 1296 | 48 / 2592 | 6 / 12 / 216 |
| 347 | `random_growth` | 24 / 1296 | 48 / 2592 | 6 / 12 / 216 |
| 347 | `backprop_13_off` | 24 / 1296 | 0 / 0 | 0 / 0 / 0 |
| 347 | `pc_13_off` | 24 / 1296 | 48 / 2592 | 0 / 0 / 0 |
| 349 | `backprop_off` | 24 / 1296 | 0 / 0 | 0 / 0 / 0 |
| 349 | `pc_off` | 24 / 1296 | 48 / 2592 | 0 / 0 / 0 |
| 349 | `neutral_off` | 24 / 1296 | 48 / 2592 | 0 / 0 / 0 |
| 349 | `usage_growth` | 24 / 1296 | 48 / 2592 | 6 / 12 / 216 |
| 349 | `scheduled_growth` | 24 / 1296 | 48 / 2592 | 6 / 12 / 216 |
| 349 | `random_growth` | 24 / 1296 | 48 / 2592 | 6 / 12 / 216 |
| 349 | `backprop_13_off` | 24 / 1296 | 0 / 0 | 0 / 0 / 0 |
| 349 | `pc_13_off` | 24 / 1296 | 48 / 2592 | 0 / 0 / 0 |
| 353 | `backprop_off` | 24 / 1296 | 0 / 0 | 0 / 0 / 0 |
| 353 | `pc_off` | 24 / 1296 | 48 / 2592 | 0 / 0 / 0 |
| 353 | `neutral_off` | 24 / 1296 | 48 / 2592 | 0 / 0 / 0 |
| 353 | `usage_growth` | 24 / 1296 | 48 / 2592 | 6 / 12 / 216 |
| 353 | `scheduled_growth` | 24 / 1296 | 48 / 2592 | 6 / 12 / 216 |
| 353 | `random_growth` | 24 / 1296 | 48 / 2592 | 6 / 12 / 216 |
| 353 | `backprop_13_off` | 24 / 1296 | 0 / 0 | 0 / 0 / 0 |
| 353 | `pc_13_off` | 24 / 1296 | 48 / 2592 | 0 / 0 / 0 |

Total training: 576 wake updates / 600 cap; zero applied or rejected replay.
All 54 guards commit, including nine final zero-add sleeps; 45 splits and
162 not-due decisions. Guards add 108 calls/1,944 examples. Outer scoring
adds 72 calls/1,440 examples. Equal wake updates do not equalize latent
work, width/FLOPs or guard overhead. Fixed-eight rows have fewer parameters;
planned-thirteen rows are wider throughout and initialize different tensors.
Growth uses neutral chemistry and the unchanged split noise.

Four circadian stores/shared FIFO retain 960 array bytes/seed before copies,
excluding source/role arrays, parameters, metadata and temporary copies.
RSS includes training, retained copies, validation, scoring and serialization;
stdout/parent writes are outside sampling. Peaks are observed at 5 ms and
may miss brief allocations. Both runs satisfy 120 s / 256 MiB. Per-arm
wall/RSS/FLOPs and complete resource attribution remain unmeasured.

## Interpretation, limits and next action

Usage minus scheduled final mean is **0, 0, -0.020833**; usage minus random
is **0, 0, 0**. There is no usage-ranking accuracy advantage in this pilot.
Scheduled exceeds random on the third seed only (+0.020833 final mean),
with higher A-after-B at equal A-after-A and B-after-B. All three growth
cells tie fixed-eight PC/backprop on the first two seeds and trail them on
the third: usage/random -0.0625, scheduled -0.041667. Growth adds capacity
and guard work without a measured advantage against fixed eight here.

Planned-thirteen comparisons are mixed: growth is above wide PC on seed
349 and below it on 353. Wide backprop/PC also have mixed seed differences
against their fixed-eight references. The full tables preserve these rows;
no wider or parent control is selected. Some forgetting differences reflect
different A-after-A learning, so always read both A values. This growth-only
neutral policy does not establish a full circadian mechanism result.

All 31 new tests and the 260-test related NumPy/Torch CPU regression gate
pass with zero skips; Ruff/mypy (361 files), four-file formatting and diff
checks pass. Tests use fresh isolated train-only fixtures, not ignored
goldens. Official runs use the canonical saved request/result/audit pins.
Full CPU suite, CUDA, broad sweeps, independent confirmation and final
release were not run. Same-environment repetition does not establish
cross-device/version portability. A clean clone can run fixture tests;
this exact canonical scored continuation requires the recorded local c9b
bundle, whose request/audit timing bytes cannot be regenerated verbatim.

C9c is complete for this frozen development contract. C9/P6.3c/P6.3 remain
open for original matrix/independent-confirmation criteria. Next inspect
the minimum matrix and every reserved confirmation protocol/seed set,
reconcile completed development scopes and freeze the smallest informative
confirmation train-only gate with explicit costs before opening final roles.
Choose no treatment, threshold, baseline, seed or metric from these outcomes.
