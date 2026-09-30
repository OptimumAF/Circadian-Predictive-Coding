# P6.3 development-only chemical-gating pilot

## Prospective contract

This is the first small factor in the staged mechanism matrix. Its question
is whether the existing **wake chemical-plasticity gate**, by itself, changes
development-role adaptation relative to an exact neutral circadian/ordinary
PC pair. It is not a full circadian test or a basis for a final ranking.

Protocol `continual_mechanism_gating_dev_v1` uses the v14 source and arrived
role settings: independent 160-row A/B two-cluster sources; A noise 0.8; B
noise 1.0, rotation 40 degrees, translation `(0.9, -0.7)`; 0.5 B development
exposure; 25% source final reservation; then 20% inner guard and 20% outer
selection of each exposed development role. B is generated only after every
A wake update. The final roles remain unopened throughout this pilot.

Predeclared development seeds: **41, 43, 59**. Reserved independent
confirmation seeds: **101, 103, 107, 109, 113, 127, 131, 137, 139, 149**.
No confirmation seed or final role may be scored in this pilot. The fixed
shallow width is eight, with identical seeded initial PC/circadian tensors.
Each arm receives 12 full-batch wake updates on A followed by 12 on B at
learning rate 0.05, two latent iterations per update, and latent rate 0.2.
The three arms are ordinary PC, `CircadianConfig.matched_pc_control()`
(neutral), and that exact control with `min_plasticity=0.2` (gating). No arm
attempts sleep, replays rows, changes structure, or changes capacity. The
neutral model must remain bitwise equal to ordinary PC after every update;
otherwise the pilot stops before publication. The gating arm must show a
plasticity factor below one to establish treatment activity.

The prelaunch cap is **240 wake updates**. Planned work is exactly
3 seeds × 3 arms × 24 updates = **216 updates**, with 72 A and 36 B train
rows per update (1,296 presentations per arm and seed), 48 latent loops,
2,592 example-inference iterations, zero replay updates, zero sleep
attempts, and width eight/33 trainable parameters throughout. Run each
complete local process under a **120-second wall limit**. Stop and record a
failure if the declared work, source roles, parity, finite metrics, or wall
limit fails; do not change settings or drop a seed to get a favorable score.

Primary development diagnostics are final mean task accuracy
`(A_after_B + B_after_B)/2` and signed forgetting
`A_after_A - A_after_B`, both derived from the **outer-selection** roles.
`A_after_A`, `A_after_B`, and `B_after_B` stay explicit so negative
forgetting (positive transfer) and weak initial learning cannot be hidden.
The existing two-task balanced score is numerically the same mean and is
only a consistency label here; it is not an additional independent metric.
No test-informed selection or superiority claim follows from these
development values. Report every seed and paired gating-minus-neutral
contrast, including null/negative results. Later confirmation must freeze
its setting and sample size before opening independent final roles.

**Why this first factor:** the original minimum matrix names chemical
gating as a component, and the existing neutral NumPy parity test proves
that a no-effect shallow circadian model can match ordinary PC exactly.
The verified fixed v14 bundle supplies source/role and accounting patterns,
but its periodic arm adds replay and prunes capacity, so a periodic-minus-
no-sleep score is not an isolated chemical-gating effect. The v10 and v12
historical factor results remain contextual evidence, not selection data
for this new pilot. This new protocol does not change v9–v14 manifests or
their saved results ([ADR-0141](adr/ADR-0141-stage-matched-gating-pilot-before-full-mechanism-matrix.md)).

## Frozen execution and evidence

The app study is `src/app/continual_gating_pilot.py`; the local adapter is
`scripts/run_p63_gating_pilot.py`. The adapter saves an exclusive prelaunch
request, invokes one worker with the 120-second limit, verifies finite
result/config/role/work facts, and writes an audit or failure sidecar.
Use a new ignored directory for a repeat. Source SHA-256 values and exact
result digests are recorded in the session log before/after execution.
Before any published run, the frozen manifest digest is
`3aec33aba455ff3fa04b0d20993ed97f6e0aedc8e64f150fbe0040bf0a44dfe9`,
the sorted eight-source SHA-256 map digest is
`797183a7da84bab8c1cb2825dda06c5fdcf3db1fea051564f11122fd88979167`,
and the adapter byte SHA-256 is
`919347ba9a9a38d88a83c97b5096898e13f9807f53225a2834b9927cba967c07`.
The eight individual source hashes are pinned by the adapter and saved in
each request. This exact source selection covers the pilot app, v14 source
manifest, arrived/data splitter, and two PC cores; it does not claim a
complete environment or dependency-tree hash.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_gating_pilot --output-dir artifacts/runs/p63-gating-pilot
.\.venv\Scripts\python.exe -m scripts.run_p63_gating_pilot --output-dir artifacts/runs/p63-gating-pilot-repeat
```

This pilot leaves the rest of P6.3 open: replay, structural, homeostasis,
schedule, full-minus-one, matched backprop, fixed-width, and independent
confirmation cells still require separately frozen work and capacity
controls. P6.8's primary metric contract applies to those later studies.
