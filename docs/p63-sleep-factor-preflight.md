# P6.3 sleep-factor train-only preflight contract

## Question and evaluation seal

This preflight checks whether a small no-replay sleep-factor matrix can be
trained under matched A→B exposure, fixed role timing, a common A-boundary
guard, and explicit capacity costs. It produces **no outer-selection or
final-test accuracy**. A separate scored protocol may follow only after
all train-only gates pass. The source geometry and arrived role fractions
are the fixed v14 160/160 two-cluster A/B settings: A train/inner/outer
72/24/24; B train/inner/outer 36/12/12. B source is constructed only after
all A wake and sleep work. Inner guard accuracy is read immediately before
and after each proposed sleep; outer-selection and final arrays are not
read. The guard accepts only non-decreasing A-inner accuracy (tolerance
zero). A rejected proposal is rolled back; proposed and applied structure
and capacity are recorded separately.

The arrived-role validator's v5 source config requires a replay-enabled
`circadian_config`. This is only a source/role geometry carrier. The nine
models are constructed separately with the arm configs below, each with
`replay_steps=0`, `replay_memory_size=0`, and replay disabled. The source
config is never used to train a model in this preflight.

Development seeds are **67, 71, 73**. Independent reserved confirmation
seeds are **151, 157, 163, 167, 173, 179, 181, 191, 193, 197**. No
confirmation seed or final role is opened in this preflight. The distinct
protocol ID is `continual_sleep_factor_train_only_v1`.

## Fixed arms and controls

All shallow width-eight arms start from the same parameters using
`source_seed + 1001`; width-12 backprop and ordinary PC controls share
their own width-12 initialization. Every arm receives 12 full-batch A
and 12 full-batch B wake updates. Backprop uses rate 0.12; PC/circadian
use rate 0.05, two latent steps and latent rate 0.2. No arm stores or
applies replay. The nine arms are:

| Arm | A-boundary action and purpose |
|---|---|
| `backprop_8` | No sleep; matched-width learning-rule reference. |
| `pc_8` | No sleep; ordinary PC reference. |
| `neutral_sham` | Neutral circadian component sleep with all effects disabled; guard/schedule control, exact PC parity required. |
| `structure_only` | Neutral circadian, one split and one prune slot; all other sleep effects disabled. |
| `homeostasis_only` | Neutral circadian, global parameter downscale 0.99; fixed width and no other sleep effect. |
| `gating_sham` | Chemical plasticity gate (`min_plasticity=0.2`), sham component sleep; comparator for chemical reset. |
| `gating_reset` | Same gate, one chemical reset with factor 0.45; no other sleep effect. |
| `backprop_12` | Planned 1.5x-width no-sleep capacity reference. |
| `pc_12` | Planned 1.5x-width no-sleep capacity reference. |

The structural arm uses the existing v12 exercise settings: static split
threshold 0, prune threshold 1, caps one each, immediate prune,
`split_noise_scale=0.02`, minimum width 7 and maximum width 9. One split
can transiently increase the width to 9/37 parameters even if one prune
returns it to width 8/33. The choice reuses the existing mechanism
exercise contract, not any v12 final outcome. Accepted, rejected, or
inactive structural proposals are all valid observations; thresholds,
seeds, and the guard rule will not be changed to force activity or a
favorable score. The homeostasis and reset settings reuse the existing
core/v14 values and are isolated in different neutral or gated pairs.

Each of the five circadian arms (`neutral_sham`, `structure_only`,
`homeostasis_only`, `gating_sham`, `gating_reset`) attempts exactly one
forced component sleep after A epoch 12. Each uses the **same A inner
guard** twice. No scheduled sleep occurs during B. No outer-selection
role is used for sleep acceptance. The ordinary PC and neutral sham
parameters must remain exactly equal after every wake and sham sleep.
The gated pair must remain parameter-equal until the reset event; reset
changes chemistry, not parameters, at that boundary. Later B wake updates
may diverge.

Planned wake work is **648 optimizer updates**: 3 seeds × 9 arms × 24.
The hard prelaunch cap is **700 updates**, with zero replay updates and
exactly 15 guarded sleep attempts/30 inner-guard evaluations across the
study. Per arm/seed wake exposure is 1,296 rows. Width-eight arms have
33 trainable parameters initially, the planned width-12 references 49;
the structural arm's transient peak is bounded at width 9/37.
The public local preflight has a **120-second** wall limit and an observed
current-process RSS ceiling of **256 MiB** (absolute worker scope).
RSS sampling observes a high-water value; it cannot prevent a brief peak
between samples. Failure of any role/work/capacity/parity/budget check
leaves the task open and records the precise failure. No scored factor run
starts from a partial preflight.

**Why this order:** the earlier replay factor was fixed width and showed
that neutral circadian equals ordinary PC when replay is the only active
sleep component. A single A-boundary proposal allows structure, global
homeostasis, and reset conditional on gating to be examined with the same
wake exposure and guard semantics. Reusing the existing guard helper keeps
rollback and proposed/applied cost accounting consistent. Planned width
references are fixed before any structural outcome; they are not equal
compute controls or retrospective final-width oracles. This preflight
cannot establish an accuracy gain or close the P6.3 matrix.

## Artifact and next gate

The app returns deterministic train-only role, parameter, guard,
structural, and work facts. The public adapter will save an exclusive
prelaunch request, bounded worker result and audit or failure sidecar in
an ignored directory. It will exclude observed durations/RSS from the
deterministic result and record them only in the audit. Source, manifest,
and adapter SHA-256 identities will be frozen before the public preflight.
After the train-only gate, separately freeze the scored outer-selection
rule and independent confirmation decision before reading a score.
Before the public preflight, the exact manifest digest is
`97c576e16b8e2307d89952b28db7948b1feb433b87d3407f9f33e451472617d1`,
the sorted 13-source SHA-256 map digest is
`80c16f348f040da5c4e42e89e2787e1abf47ef7cdc7612fdd344e9dde76339fa`,
and the adapter byte SHA-256 is
`dae68cbdb8ef0464ff9a4971bfaedc2e50dc4d75980693a236a5a92eb38d0d3c`.
The adapter pins the individual source hashes. This is a selected source
map for the preflight and RSS reader, not a full dependency-tree hash.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_sleep_factor_preflight --output-dir artifacts/runs/p63-sleep-factor-preflight
.\.venv\Scripts\python.exe -m scripts.run_p63_sleep_factor_preflight --output-dir artifacts/runs/p63-sleep-factor-preflight-repeat
```
