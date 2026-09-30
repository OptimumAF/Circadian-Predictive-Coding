# P6.2 corrected continual toy-profile reproduction

## Fixed request before training

Run the existing `baseline`, `strength-case`, and `hardest-case` profiles in
that order with `scripts.run_continual_shift_benchmark`. Keep each profile's
current preset values, the original ordered seeds `3,7,11,19,23,31,37`,
all three model learning rates, model order, and the runner's existing
retention/balanced-score definitions. Do not select a seed, retune a baseline,
or change the metric after seeing a result. Use the explicit
`continual_validation_v1` protocol with its 20% development validation
reservation for every profile. The older text artifacts use the historical
train/test route; they are references, not independent confirmation data.

**Why this protocol:** it is the corrected validation protocol supported by
all three unchanged profiles, including baseline's `replay_steps=0` and
legacy sleep mode. The opt-in v5 global final seal requires bounded replay
and component sleep, which would alter the baseline and historical policy
settings. The v1 runner completes both phases for a seed before scoring that
seed, but constructs final-source objects earlier and scores one seed before
later seeds finish. Results are descriptive, not strict-online or globally
sealed evidence. Its one-hidden NumPy model initializations and learning
rules are also not the matched-head reference; no algorithm superiority
claim follows from a profile score.

| Profile | Source rows A/B | A/B epochs | Width | Phase B fraction | Noise A/B | B transform | Sleep intervals A/B | Circadian policy digest |
|---|---:|---:|---|---:|---|---|---|---|
| baseline | 500/500 | 110/80 | 12 | 0.14 | 0.8/1.0 | 40°, (+0.9, −0.7) | 40/8 | `f3f6801e915a409cb2d2ae50876bcb0e27cc9f1f1c044cba0b16e783499b645c` |
| strength-case | 500/500 | 110/80 | 12 | 0.14 | 0.8/1.0 | 40°, (+0.9, −0.7) | 40/8 | `7c51a9c764e6fc95ecb2f2bb94855e6a94bd11c6baf6acf86b743e6540c4732a` |
| hardest-case | 700/700 | 120/180 | 24/24/24 | 0.05 | 0.8/1.45 | 68°, (+1.6, −1.3) | 40/6 | `91974bbe5f4aadf75a1bc650239284360ecdffae05d219186d94b98f5f03e0e3` |

Policy digests are SHA-256 of each current `CircadianConfig` as sorted,
compact JSON from `dataclasses.asdict`. The baseline has zero replay steps;
strength/hardest retain their previously tuned two/three replay steps.
Source SHA-256 before running: CLI
`a605b8ff4d725299b90fb3365de377a40b279974bb3193d427878ebe7b032044`,
app runner
`53037d6cfbf4a24a807eb5a4b67d5525cc422e1d209e39ab5c54c1ec0018f5fa`,
and circadian core
`08f36db0a5c1f71d5de198fe0cf1c1be0e6a37eaf5855dec53bbb4a299ec27aa`.
The checkout is `master` at
`c17a37d792a6f26728a557c5ecc613d528d0f9ad` with unrelated and prior
Phase 6 working changes preserved; these hashes pin the current bytes,
including earlier working-tree edits to the app runner.

Use an ignored fresh directory `artifacts/runs/p62-profiles/`. The
`scripts.run_p62_profile_reproduction` adapter checks the frozen source and
policy hashes, writes an exclusive prelaunch `*.request.json`, runs the
unchanged public CLI, then reads `*.txt`, `*.json`, and `*-config.json` and
writes an exclusive `*.audit.json` or `*.failure.json`. The request saves
exact argv, source/policy hashes, UTC start, and declared timeout.
Stop a whole profile at 240 seconds for baseline or strength and 600 seconds
for hardest; these are wall-clock safety limits, not training budgets or a
reason to change settings. Do not automatically retry a failed/timeout run
under the same request or publish a partial result as completed. Run one
profile at a time. The planned seven seeds imply 3,990 model-epochs for
baseline/strength and 6,300 for hardest; sleep/replay work is additional.

Run one fixed profile at a time from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p62_profile_reproduction --profile baseline
.\.venv\Scripts\python.exe -m scripts.run_p62_profile_reproduction --profile strength-case
.\.venv\Scripts\python.exe -m scripts.run_p62_profile_reproduction --profile hardest-case
```

Before treating a profile as complete, read strict finite JSON and verify
`resolved_config.config == result.config`, exact profile/seeds/source
settings, seven seed rows and six role hashes per seed, finite
per-seed and aggregate values, and text/JSON consistency. Record all output
SHA-256 values. A new profile's aggregate may differ from the historical
strength/hardest text because source splitting, mathematics, and later
correctness fixes changed; the historical exact execution config and seed
rows are absent. Report the difference without assigning it to one fix.
After all three complete, compare corrected profiles descriptively and
explicitly state that their circadian policies were historically tuned.
For a same-environment repeat, use a fresh directory and the same commands
with `--output-dir artifacts/runs/p62-profiles-repeat`. Then run
`scripts.verify_p62_profile_repeats` with both directories. It checks saved
artifact hashes and requires identical text/config bytes and exact result
content after removing only observed `sleep_events[].durations` values.
This timing exclusion was already part of the project reproducibility scope;
it does not relax a score, role, or sleep-decision comparison.

Historical sources are
[`historical-benchmark-provenance.md`](historical-benchmark-provenance.md),
[`benchmark_continual_shift_strength_case_2026-02-28.txt`](benchmarks/benchmark_continual_shift_strength_case_2026-02-28.txt),
and
[`benchmark_continual_shift_hardest_case_2026-02-28.txt`](benchmarks/benchmark_continual_shift_hardest_case_2026-02-28.txt).
