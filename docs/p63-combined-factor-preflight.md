# P6.3 combined/full-minus-one train-only contract

## Source, cells and settings fixed before training

Protocol: `continual_combined_factor_train_only_v1`. This fills the combined
and component-removal rows after the isolated factors; no outer/final score
is produced. Reuse the arrived v14 source geometry and c2's wake/replay
settings: A/B source 160/160, A train/inner/outer 72/24/24, B 36/12/12,
12+12 full-batch wake updates, backprop rate .12, PC/circadian rate .05,
two latent steps at .2. B arrives only after all seventeen A models finish.
Final inputs/labels and outer arrays remain sealed to this trainer.

Development seeds **263/269/271** are the next three primes above the
previously reserved maximum 257. Independent confirmation seeds are the next
ten primes **277/281/283/293/307/311/313/317/331/337**, unused. Earlier
confirmation reservations remain unused. All seeds are retained; none is
selected from scores. Shared initialization within each width uses seed+1001.

| Cell | Treatment |
|---|---|
| `backprop_off`, `pc_off`, `neutral_off` | Width-eight no-sleep/no-replay references; exact PC/neutral parity. |
| `backprop_full_replay`, `pc_full_replay`, `neutral_full_replay` | Width-eight consumers of exactly the full controller's accepted replay IDs/work; exact PC/neutral parity. |
| `full` | Combined periodic sleep with gating, difficulty modulation, structure, replay, homeostasis and reset. |
| `minus_replay` | Full with replay disabled; keep identical available memory. |
| `minus_gating` | Full with min plasticity 1. |
| `minus_structure` | Full with split/prune disabled and width fixed at eight. |
| `minus_schedule` | Full with sleep disabled; retains wake gating/difficulty and memory. |
| `minus_difficulty` | Full with reward-modulated learning disabled. |
| `minus_homeostasis` | Full with homeostasis disabled. |
| `minus_reset` | Full with chemical reset disabled. |
| `periodic_structure_only` | C3 neutral structure-only config, now periodic; no replay/gating/difficulty/homeostasis/reset. |
| `backprop_14_off`, `pc_14_off` | Prospectively wider fixed references at the full arm's conservative peak ceiling. |

These are **17 cells per seed, 51 total**. The full config reuses v14's
component config (one split/prune per event, homeostatic factor .99, replay
two single-row updates at .01 with two latent steps at .15, newest FIFO,
reset .45, gating minimum .2), enabling its existing reward-modulation
switch with **unchanged default** EMA/exponent/.75–1.5 clipping so the
required difficulty removal has a defined counterpart. Its effect includes
reward-weighted importance history as well as wake scale; no pure
learning-rate-only claim is made. Keep v14's .8/.08 structural thresholds,
chemical settings, phase budgets and all other defaults. There is no adaptive
trigger/budget, saturation, dual chemistry or new algorithm.

Own scheduled cells use phase-local forced interval four: A/B epochs 4,8,12.
The no-schedule removal disables all sleep and consequently its downstream
effects; this is intentionally a different work treatment. All component
removals otherwise differ only in their named existing switch. Periodic
structure-only reuses c3's declared 0/1 thresholds, one split/prune, noise .02,
immediate pruning and min/max 7/9; its replay-memory fields are inert while
replay is disabled. All circadian cells retain the same available FIFO supply.

Dynamic full/removal cells have minimum width four and maximum/temporary
ceiling **14** (initial eight plus a conservative six one-split attempts).
No-structure and neutral controls stay at eight. Wider width 14 is planned
before observing any final width, not a retrospective oracle. Each shallow
head has 4×width+1 parameters: initial 33 or 57, dynamic minimum 17 and
peak ceiling 57. Record proposals, committed and rejected transient peaks.

## Replay, guard and state evidence

A prediction-independent shared FIFO observes each arrived train batch after
wake, retains eight distinct float64 labeled rows/192 array bytes and offers
the newest two IDs at every epoch. Each of eleven circadian buffers has the
same retention/order/content. Baseline replay receives detached row copies.
Per-seed persistent labeled-array scope is 192 shared + 11×192 = 2,304 bytes,
excluding metadata, temporary copies and parameters; whole-worker RSS includes
them. No future rows enter a buffer.

The existing inner guard at tolerance zero owns each of eight scheduled
controllers. Their proposals, guard results and rollback costs are independent.
Only the **full** controller's commit supplies the three matched replay
references. Backprop/PC/neutral replay use identical offered IDs, rate and
declared latent work. These consumers have no independent guard and do not
receive full topology, chemistry or difficulty settings. Neutral consumers
retain exact PC parameters after every wake/replay. Full/minus-one contrasts
have equal wake/source exposure but intentionally unequal replay/capacity and
guard costs; record the differences instead of calling them equal compute.

Every epoch records all model widths, parameter hashes, plasticity minima and
observed reward scales. Every own sleep opportunity records existing complete
telemetry except measured durations, stable IDs before/after, complete snapshot
state fingerprints, parameter hashes and clocks. Snapshot fingerprints bind
all snapshot fields, including chemistry, lineage, memory, RNG and clocks;
array shape/dtype/content is canonicalized. A rejection must restore the entire
fingerprint while proposed IDs and **executed** replay cost survive in facts.
The independent JSON validator rederives all roles, clocks, guard decisions,
width/ID histories, work totals and component-disable effects.

## Budget and completion

Wake work is **1,224 updates**. At most six scheduled replay-enabled
full/removal cells × six attempts × two updates plus three consumers × six
full commits × two updates adds 108 per seed: **maximum 1,548 executed
updates**, including rejected replay, under hard cap **1,600**. There are
648 own decision records, **144 guarded attempts/288 inner evaluations**
(5,184 evaluated guard examples), at most 18 full-controlled replay events.
No guard score changes settings. Worker limits: **120 seconds**, observed
whole-worker RSS **256 MiB**, sampled every 5 ms; short peaks can be missed.

Pin manifest, selected source and adapter identities before public training.
Use exclusive request/result/audit or failure artifacts. Validate all 51 cells,
72 opportunities, 648 decisions, six role hashes per seed, all capacities,
actual/rejected work, memory/IDs and exact neutral parity. Outer/final/arrival
and changed-manifest sentinels must pass. Exercise forced guard rejection with
observed core replay executions and full rollback. Repeat in a second fresh
bounded process and require identical deterministic bytes. Publish every
inactive component, rejected proposal and work/capacity cell without tuning.

**Why this scope:** compose existing component switches and controls before
introducing a new algorithm. Current `NeuronAdaptationPolicy` supplies add
counts but the core still ranks split parents by usage; it does not implement
scheduled/random parent selection. Those required growth controls remain
explicitly unfinished under c9. C8 separately freezes development scoring
against this saved all-seed gate. Neither this preflight nor the isolated c6
scores close the original matrix, confirmation or final release.

## Implementation identities

Before the first public process, manifest SHA-256 is
`729bf9df8472752f51696299373555339208af7b5add1892d14154d4f21ad04a`;
the selected combined 26-source map digest is
`457df802859b68e42b2a54a31e7a9a49f4e893d07062074bb22d3069a96c737f`;
the adapter byte digest is
`9c3d640638b48d7a2edc9cc4585e205cdb3f2d86bd1c72a6e7a8438bd0f3d404`.
The map includes c5's frozen 20 sources, the three new combined app modules,
c5's adapter, sleep-clock and neuron-lineage interfaces. It is selected,
not a full dependency-tree hash. Five app tests, Ruff and mypy (345 source
files) passed before freezing these identities. RSS sampling includes
training, independent validation and result serialization; final stdout
transport and parent artifact writing are outside that interval.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_combined_factor_preflight --output-dir artifacts/runs/p63-combined-factor-preflight
.\.venv\Scripts\python.exe -m scripts.run_p63_combined_factor_preflight --output-dir artifacts/runs/p63-combined-factor-preflight-repeat
```

Execution evidence is in [the complete result report](p63-combined-factor-preflight-results.md)
and the c7 development-log entry. Both 51-cell public results repeat at
SHA-256 `79a7d7e09f0ada01ff72d6e266ddee7576aca52316a0f9a8e2ab3dd402251d13`.
The frozen settings above remain unchanged, including inactive full split
proposals. C8 scoring and c9 growth controls remain separate open tasks.
