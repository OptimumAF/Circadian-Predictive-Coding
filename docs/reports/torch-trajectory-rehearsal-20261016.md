# Complete Torch demonstration rehearsal, 2026-10-16

## Frozen scope and interpretation

Eight fixed BP/PC/neutral/active × replay off/on arms, seed17, D2/H3, six complete
two-step A,A,B,B,A,A line-navigation demonstrations. Observations are position/2
and goal/2; goals±2 from start0 for training, goals±1 from opposite starts for
final-test action scoring. Native hard-binary objectives and old rates are
unchanged. Rewards/next-state/terminal are retained evidence, not RL targets.
Final data never select training, replay, promotion, probes or thresholds.

**Negative/limited outcome:** all arms start with A/B action accuracy1 and end
with B accuracy0. PC-family B4 accuracy1 falls to0 after returning A; replayed
PC-family A4 accuracy0 precedes recovery. BP new-B BCE gains are negative.
PC-family small positive B gains coexist with forgetting, which worsens with
replay. All return BCE gains are positive. This two-row/single-seed case
establishes no fresh acquisition, robust retention, useful policy, causal replay
benefit or circadian advantage. Scripted demonstrator return1 is not learner
return. No tuning, replacement seed or native experiment retry occurred.

## Every arm and declared metric

B gain=B0−B4 (higher), retention loss=A4−A2 (lower), return gain=A4−A6 (higher).
Full precision is retained in the JSON. Off/on exposures differ (12/18 native
examples per method); within-mode controls share all episode data and selections.

| Method | Replay | B gain | Retention loss | Return gain |
|---|---|---:|---:|---:|
| bp | False | -0.046707082444 | -0.003360136443 | 0.028828626465 |
| pc | False | 0.000945194581 | 0.030107140567 | 0.029335134559 |
| neutral_cpc | False | 0.000945194581 | 0.030107140567 | 0.029335134559 |
| gated_cpc | False | 0.000945135096 | 0.030107142338 | 0.029335078416 |
| bp | True | -0.090951851559 | -0.006027528708 | 0.036775436573 |
| pc | True | 0.002049318901 | 0.045141957336 | 0.043441039146 |
| neutral_cpc | True | 0.002049318901 | 0.045141957336 | 0.043441039146 |
| gated_cpc | True | 0.002049140642 | 0.045141989798 | 0.043440728807 |

## All 64 context score points

| Method | Replay | Wakes | A BCE | A accuracy | B BCE | B accuracy |
|---|---|---:|---:|---:|---:|---:|
| bp | False | 0 | 0.692090591373 | 1.000000 | 0.692090621144 | 1.000000 |
| bp | False | 2 | 0.649023738026 | 1.000000 | 0.736496023101 | 0.000000 |
| bp | False | 4 | 0.645663601583 | 1.000000 | 0.738797703588 | 0.000000 |
| bp | False | 6 | 0.616834975117 | 1.000000 | 0.769204687719 | 0.000000 |
| pc | False | 0 | 0.692090591373 | 1.000000 | 0.692090621144 | 1.000000 |
| pc | False | 2 | 0.662803437847 | 1.000000 | 0.722200125524 | 0.000000 |
| pc | False | 4 | 0.692910578414 | 1.000000 | 0.691145426563 | 1.000000 |
| pc | False | 6 | 0.663575443855 | 1.000000 | 0.721248918797 | 0.000000 |
| neutral_cpc | False | 0 | 0.692090591373 | 1.000000 | 0.692090621144 | 1.000000 |
| neutral_cpc | False | 2 | 0.662803437847 | 1.000000 | 0.722200125524 | 0.000000 |
| neutral_cpc | False | 4 | 0.692910578414 | 1.000000 | 0.691145426563 | 1.000000 |
| neutral_cpc | False | 6 | 0.663575443855 | 1.000000 | 0.721248918797 | 0.000000 |
| gated_cpc | False | 0 | 0.692090591373 | 1.000000 | 0.692090621144 | 1.000000 |
| gated_cpc | False | 2 | 0.662803495668 | 1.000000 | 0.722200125524 | 0.000000 |
| gated_cpc | False | 4 | 0.692910638006 | 1.000000 | 0.691145486048 | 1.000000 |
| gated_cpc | False | 6 | 0.663575559590 | 1.000000 | 0.721249041404 | 0.000000 |
| bp | True | 0 | 0.692090591373 | 1.000000 | 0.692090621144 | 1.000000 |
| bp | True | 2 | 0.611117780849 | 1.000000 | 0.778990037069 | 0.000000 |
| bp | True | 4 | 0.605090252142 | 1.000000 | 0.783042472704 | 0.000000 |
| bp | True | 6 | 0.568314815568 | 1.000000 | 0.823138811746 | 0.000000 |
| pc | True | 0 | 0.692090591373 | 1.000000 | 0.692090621144 | 1.000000 |
| pc | True | 2 | 0.648810746970 | 1.000000 | 0.737236891854 | 0.000000 |
| pc | True | 4 | 0.693952704306 | 0.000000 | 0.690041302243 | 1.000000 |
| pc | True | 6 | 0.650511665160 | 1.000000 | 0.735175801784 | 0.000000 |
| neutral_cpc | True | 0 | 0.692090591373 | 1.000000 | 0.692090621144 | 1.000000 |
| neutral_cpc | True | 2 | 0.648810746970 | 1.000000 | 0.737236891854 | 0.000000 |
| neutral_cpc | True | 4 | 0.693952704306 | 0.000000 | 0.690041302243 | 1.000000 |
| neutral_cpc | True | 6 | 0.650511665160 | 1.000000 | 0.735175801784 | 0.000000 |
| gated_cpc | True | 0 | 0.692090591373 | 1.000000 | 0.692090621144 | 1.000000 |
| gated_cpc | True | 2 | 0.648810803988 | 1.000000 | 0.737236891854 | 0.000000 |
| gated_cpc | True | 4 | 0.693952793786 | 0.000000 | 0.690041480502 | 1.000000 |
| gated_cpc | True | 6 | 0.650512064979 | 1.000000 | 0.735176050439 | 0.000000 |

## Actual work and storage

All60 native entries returned:48 wake/12 complete-episode replay calls,96 wake/
24 replay transition exposures,45 PC-family native steps/15 BP optimizer steps,
64 final forwards and16 actual pause/resumes. One joined RSS sampler, no worker
subprocesses/downloads/sweeps. Pilot13.1651991s/worker13.2139268s/parent14.3242911s;
RSS313,786,368B and226,294B report are below60s/90s/768MiB/2MiB caps.

Each method selects complete train episodes1,2,5 at wakes2,4,6. Every packet
preserves both legal transitions, action/reward/next-state/terminal and ID/context.
Encoded retention is904/1808/1808 bytes at those events; final four transitions
occupy two complete packets. The first event's recent/reservoir overlap is valid.
All12 selected packets/24 transitions,64 scores/24 metric values and16 resumes
were independently checked. Native updates at checkpoints are0/2/4/6 off and
0/3/6/9 on; native example clocks are0/4/8/12 off and0/6/12/18 on. Replay charges
do not create wake receipts. Original actors/source learners remain unchanged.

The following are the largest *saved checkpoint* tensor payloads, not measured
native peak allocations or total process memory. Encoded replay is separate.

| Method | Replay | Head tensor bytes | Optimizer tensor bytes | Gradient bytes |
|---|---|---:|---:|---:|
| bp | False | 68 | 68 | 68 |
| pc | False | 80 | 0 | 0 |
| neutral_cpc | False | 5268 | 0 | 0 |
| gated_cpc | False | 5268 | 0 | 0 |
| bp | True | 68 | 68 | 68 |
| pc | True | 80 | 0 | 0 |
| neutral_cpc | True | 5268 | 0 | 0 |
| gated_cpc | True | 5268 | 0 | 0 |

Encoded bytes include padded UTF8 metadata and dummy targets; Python objects,
copies, checkpoints and transient gradients are not estimated from row counts.
Total runtime/RSS includes other work without causal cost attribution. Native
PC energy, BP CE and optimizer/adaptive state costs remain distinct. No equal-
compute, efficiency, interval, confirmation or release-readiness claim.

## Engineering acceptance and limitations

237 focused cases qualified via236 unchanged+one repaired identity fixture/711
PASSphases,238 executions; original failed invocation retained. The identity
test uses FIFO to isolate deduplication from valid partition overlap; the frozen
native rotating policy/seed53 is unchanged. Eight-path Ruff/format/Windows+Linux
mypy and independent SOURCE/GO/ACTUAL review pass. Original bounded R2.4 criteria
are independently accepted for complete binary-action demonstrations and owner-
local continuation. Broader actions, recurrent RL, durable recovery and useful
adaptation remain separate open goals. Runtime full_R24_complete=false is retained
as pre-review metadata. See [usage](../trajectory-rehearsal.md) and
[exact commands/failures/review](../../artifacts/runs/r24-trajectory-rehearsal-20261016/RESULTS.md).
