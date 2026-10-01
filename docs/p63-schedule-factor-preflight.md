# P6.3 matched schedule-factor train-only contract

## Question, roles and fixed settings

This gate isolates the timing and guard costs of the existing periodic/current-adaptive/no-sleep policies at fixed width, with replay as the only sleep effect. It produces no outer-selection or final score. The v14 full-component route changes replay, topology and chemistry together; this new protocol removes those treatment differences so the selected replay rows and applied optimizer work can be compared explicitly. Full-minus-one remains a separate required part of P6.3c.

Protocol ID: `continual_schedule_factor_train_only_v1`. Development seeds are **79, 83, 89**, all retained. Reserved independent confirmation seeds are **199, 211, 223, 227, 229, 233, 239, 241, 251, 257**. Source and role geometry reuse v14's arrived 160/160 rotated/translated A→B two-cluster source and 0.2 inner/outer fractions: A train/inner/outer 72/24/24, B 36/12/12. B source is constructed only after all eleven models finish A wake and scheduled decisions. Neither outer array nor any final-source input/label is read. No confirmation source is constructed.

For each of the three policies, independently initialize width-eight backprop, ordinary PC and **neutral circadian PC** at `source_seed+1001`. The neutral head has `min_plasticity=1`, no difficulty modulation, no structural changes, homeostasis or chemical reset, and replay enabled. It uses the existing c2 replay settings: rate .01, two single-row updates, two latent steps at .15, unprioritized newest-retained sampling. The chemical state is observable and supplies the existing adaptive variance signal but cannot change wake or replay plasticity. Exact ordinary-PC/neutral tensor parity is required after each wake and committed replay. Add planned width-12 backprop/PC no-sleep/no-replay references with their own shared initialization. These are eleven models per seed, 33 model cells.

Every model receives 12 full-batch A and 12 full-batch B wake updates with the same rows and rates (backprop .12; PC/neutral .05, two latent steps at .2). The three policies differ only in the scheduling switch:

| Policy | Decision after each completed wake epoch |
|---|---|
| `periodic` | Force a phase-local interval-four attempt (A/B epochs 4,8,12). Adaptive switch off. |
| `adaptive` | Interval zero, force off; keep minimum spacing ten wake batches, eight-energy window, improvement ≤.001 and chemical variance ≥.02. |
| `no_sleep` | Interval zero and adaptive switch off; retain identical available memory without applying it. |

The adaptive signal uses the current neutral head's actual wake-energy history and chemical variance. No new trigger, threshold or forced adaptive event is introduced. An inactive adaptive rule is a valid result. Test fixtures may force readiness or rejection solely to verify wiring; official runs retain all defaults. Component mode resets the spacing clock only when a sleep commits. A guard-rejected attempt restores it and can retry after subsequent wake work, following the existing v14 helper's no-cooldown path; this cost is recorded.

## Matched replay, guard and work

One prediction-independent FIFO supply observes each arrived train batch after wake, retains eight distinct float64 labeled rows/192 array bytes, and offers the same newest two IDs at **every** epoch to all three policies. Each neutral model independently retains identical rows/order and previews the same IDs before its decision. Inputs passed to baseline replay are detached copies. No future rows are retained.

A due neutral sleep uses only the current phase's inner guard, twice, at tolerance zero. Only an accepted proposal commits replay; then both corresponding baseline models receive exactly the same two rows, rate and declared latent-work settings. A rejection restores the neutral state and applies no baseline replay. The guard is owned by the common neutral/ordinary-PC controller; it is not a separate backprop guard. Every opportunity records periodic/adaptive readiness, energy-window values, chemical variance, spacing before/after, guard role/accuracies, proposed and applied replay IDs/updates, rejected executed work, exact parameter hashes, and fixed capacity. Attempted circadian replay that rolls back is **executed cost** even though restored model clocks show zero applied replay.

Width eight has 33 trainable parameters; width 12 has 49. There is no transient growth. Per-model wake work is 24 updates/1,296 row presentations, with 48 latent loops/2,592 example-iterations for PC/neutral. Shared supply memory is 192 array bytes, plus 192 in each of the three neutral model buffers; baseline consumers use that shared supply rather than independent persistent arrays. This is matched memory access, with the actual retained-array scope labeled. Wider references have no replay supply and different per-update compute.

Total wake work is **792 optimizer updates**. Each seed has six periodic attempts, at most fifteen adaptive attempts and at most two accepted adaptive events under the unchanged minimum spacing. Including rejected neutral replay executions and accepted baseline replay, the prospective maximum is **1,014 executed optimizer updates** (792 wake + 222 replay), under a hard prelaunch **1,100-update cap**. At most 63 guarded attempts/126 guard accuracy evaluations occur across the study. The adapter enforces a **120-second** worker wall limit and an observed whole-worker RSS cap of **256 MiB**, sampled every 5 ms. Sampling may miss a shorter peak; RSS is not per-arm.

## Completion and next scored gate

Before the public run, pin the manifest, selected source hashes and adapter bytes. Write an exclusive request, deterministic result and audit or failure sidecar to an ignored local directory. Train all 33 cells, independently rederive decision/clock and work facts, verify memory/IDs, exact neutral parity, capacity, all role hashes/counts and budgets, and keep final/outer sentinels sealed. Independently repeat in a second process and require exact result bytes. Report every cell's actual work and every inactive/rejected event. No partial preflight permits scoring.

**Why this design:** matching ordinary PC to a neutral circadian head exposes replay or rollback discrepancies immediately. Supplying the same rows at every epoch makes available memory independent of each trigger, while conditional application preserves the policy's real treatment cost. Holding topology and other sleep effects fixed makes a later schedule contrast interpretable; the combined full model and its minus-one controls remain required.

A separate scored development task must bind this saved train-only result and freeze the outer accuracy matrix, two primary metrics, all policy contrasts, confirmation decision and global all-seed comparison before reading any score. A null/inactive preflight will not be tuned to manufacture a trigger benefit.

## Implementation identities

Before the first public train-only process, the manifest digest is `f8b6d60209516c58bc54e729f659c9fdd37ea1b869652b1cac548a71270ea42a`, the sorted selected 20-source SHA-256 map digest is `2e45a6200ef7629bdc0b27e86386ad02e8064ac144fe049b06d014541aa1dcfd`, and the adapter byte SHA-256 is `4490f9201b565d3283434da786799cc3d9ac8242218cfc8f9b881de9011b7fbe`. The adapter pins each source, including the reused c2 model/replay helpers, existing guard/scheduler and c3 artifact helpers. This is a selected map, not a full dependency-tree hash. Five app tests passed with outer/final sentinels, guard rejection, forced adaptive fixture and forged facts; static gates passed before these identities were frozen. No scored process is part of this contract.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_schedule_factor_preflight --output-dir artifacts/runs/p63-schedule-factor-preflight
.\.venv\Scripts\python.exe -m scripts.run_p63_schedule_factor_preflight --output-dir artifacts/runs/p63-schedule-factor-preflight-repeat
```
