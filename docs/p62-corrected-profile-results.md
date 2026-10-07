# P6.2 corrected continual toy profiles — 2026-09-30

All three existing profiles ran under the frozen
[`P6.2 request`](p62-profile-reproduction.md):
`continual_validation_v1`, original ordered seeds `3,7,11,19,23,31,37`,
unchanged profile policies, current NumPy CPU runner, and no new tuning.
Each public CLI run produced text, complete result JSON, and resolved config
under ignored `artifacts/runs/p62-profiles/`; the prelaunch request and
postrun audit are alongside them. Each completed well below its declared
240/240/600-second wall limit (about 2.54/2.60/9.48 seconds). The audit
checked strict finite JSON, exact settings, six role hashes and all three
methods for every seed, independently recomputed aggregate means/spreads,
text/stdout agreement, numeric and role-hash text/JSON agreement, and SHA-256
of every output. The stronger text cross-check was also applied read-only
to all six saved primary/repeat runs after completion. Python 3.14.7, NumPy
2.4.6, Windows 11, Intel x64 CPU. This is a descriptive reproduction of
previously tuned profiles, not a new selection or confirmation study.

## Complete observed aggregates

Values are mean +/- population standard deviation across the seven fixed
seeds. The metric definitions are those of the current unchanged runner.

| Profile | Method | A pre | A post | B post | Retention | Balanced |
|---|---|---:|---:|---:|---:|---:|
| baseline | Backprop | 0.974857 +/- 0.011655 | 0.956571 +/- 0.020886 | 0.933714 +/- 0.022512 | 0.981377 +/- 0.024244 | 0.945143 +/- 0.015815 |
| baseline | PC | 0.969143 +/- 0.017852 | 0.971429 +/- 0.010349 | 0.920000 +/- 0.023028 | 1.002641 +/- 0.018419 | 0.945714 +/- 0.010443 |
| baseline | Circadian | 0.976000 +/- 0.016000 | 0.969143 +/- 0.011655 | 0.917714 +/- 0.032203 | 0.993068 +/- 0.008045 | 0.943429 +/- 0.013081 |
| strength-case | Backprop | 0.974857 +/- 0.011655 | 0.956571 +/- 0.020886 | 0.933714 +/- 0.022512 | 0.981377 +/- 0.024244 | 0.945143 +/- 0.015815 |
| strength-case | PC | 0.969143 +/- 0.017852 | 0.971429 +/- 0.010349 | 0.920000 +/- 0.023028 | 1.002641 +/- 0.018419 | 0.945714 +/- 0.010443 |
| strength-case | Circadian | 0.976000 +/- 0.016000 | 0.968000 +/- 0.012095 | 0.918857 +/- 0.032406 | 0.991877 +/- 0.006073 | 0.943429 +/- 0.014411 |
| hardest-case | Backprop | 0.974694 +/- 0.006732 | 0.657143 +/- 0.188435 | 0.829388 +/- 0.035807 | 0.673830 +/- 0.191761 | 0.743265 +/- 0.104158 |
| hardest-case | PC | 0.973878 +/- 0.010064 | 0.744490 +/- 0.173987 | 0.827755 +/- 0.032387 | 0.764980 +/- 0.180448 | 0.786122 +/- 0.076137 |
| hardest-case | Circadian | 0.974694 +/- 0.007998 | 0.707755 +/- 0.185653 | 0.828571 +/- 0.032325 | 0.726084 +/- 0.190246 | 0.768163 +/- 0.098968 |

Every seed and method's balanced score is retained below; the full local
result JSON also contains each seed's A/B accuracies, retention, sleep
events, and role hashes.

| Profile | Seed | Backprop | PC | Circadian |
|---|---:|---:|---:|---:|
| baseline | 3 | 0.940000 | 0.936000 | 0.948000 |
| baseline | 7 | 0.924000 | 0.932000 | 0.928000 |
| baseline | 11 | 0.964000 | 0.956000 | 0.948000 |
| baseline | 19 | 0.956000 | 0.944000 | 0.920000 |
| baseline | 23 | 0.924000 | 0.940000 | 0.948000 |
| baseline | 31 | 0.944000 | 0.948000 | 0.952000 |
| baseline | 37 | 0.964000 | 0.964000 | 0.960000 |
| strength-case | 3 | 0.940000 | 0.936000 | 0.948000 |
| strength-case | 7 | 0.924000 | 0.932000 | 0.928000 |
| strength-case | 11 | 0.964000 | 0.956000 | 0.952000 |
| strength-case | 19 | 0.956000 | 0.944000 | 0.916000 |
| strength-case | 23 | 0.924000 | 0.940000 | 0.948000 |
| strength-case | 31 | 0.944000 | 0.948000 | 0.952000 |
| strength-case | 37 | 0.964000 | 0.964000 | 0.960000 |
| hardest-case | 3 | 0.897143 | 0.880000 | 0.908571 |
| hardest-case | 7 | 0.800000 | 0.860000 | 0.828571 |
| hardest-case | 11 | 0.628571 | 0.680000 | 0.628571 |
| hardest-case | 19 | 0.671429 | 0.740000 | 0.671429 |
| hardest-case | 23 | 0.651429 | 0.711429 | 0.734286 |
| hardest-case | 31 | 0.877143 | 0.868571 | 0.882857 |
| hardest-case | 37 | 0.677143 | 0.762857 | 0.722857 |

Circadian mean sleep events/splits/prunes/final hidden width were
`1.86/1.57/0.71/12.86` for baseline,
`5.29/5.29/0/17.29` for strength-case, and
`25.00/49.86/0/73.86` for hardest-case. These record added mechanism
and capacity work, not matched work for a learning-rule ranking.

## Historical change versus current policy contrast

Historical tracked [strength-case](benchmarks/benchmark_continual_shift_strength_case_2026-02-28.txt)
and [hardest-case](benchmarks/benchmark_continual_shift_hardest_case_2026-02-28.txt)
text reported rounded balanced means below. This comparison spans the
corrected validation split and intervening mathematical/code changes;
historical exact execution commits, resolved configs, and per-seed rows
are unavailable. The observed shifts cannot be attributed to one fix or
used as independent confirmation. Baseline has no tracked historical text.

| Profile | Method | Historical rounded mean | Current mean | Approximate difference |
|---|---|---:|---:|---:|
| strength-case | Backprop | 0.946 | 0.945143 | -0.000857 |
| strength-case | PC | 0.947 | 0.945714 | -0.001286 |
| strength-case | Circadian | 0.949 | 0.943429 | -0.005571 |
| hardest-case | Backprop | 0.753 | 0.743265 | -0.009735 |
| hardest-case | PC | 0.808 | 0.786122 | -0.021878 |
| hardest-case | Circadian | 0.812 | 0.768163 | -0.043837 |

The *current-protocol policy contrast* is separable for baseline versus
strength-case: their source, seeds, model order, widths, epochs, metrics,
and baseline/PC settings are identical; only the bundled circadian policy
differs. Backprop and PC rows are byte-equivalent at the metric level.
Circadian's balanced mean is **0.943429 under both**. Paired
strength-minus-baseline circadian balanced differences by ordered seed are
`0, 0, +0.004, -0.004, 0, 0, 0`; there is no mean gain in this fixed set,
despite more sleep/splits. This is not a new per-knob ablation. The hardest
profile also changes data difficulty, architecture, epochs, and policy, so
its score cannot isolate a policy effect. Under its current corrected
profile, circadian's mean `0.768163` is below PC's `0.786122`; the negative
outcome is retained without seed or metric adjustment.

## Repeat and limits

All three profiles were repeated in a second fresh local process under
`artifacts/runs/p62-profiles-repeat/`, with the same frozen source, policy,
seed, and protocol identities. `scripts.verify_p62_profile_repeats` checked
both audits against each file's SHA-256 and compared the two results.
Text and resolved-config files are byte-identical for every profile. Raw
result JSON bytes differ only in observed sleep `attempt_seconds` and
`core_seconds` (168, 168, and 461 scalar fields respectively); after
excluding precisely those measured duration fields, all deterministic
JSON content matches exactly. This exclusion follows the pre-existing
[reproducibility scope](reproducibility-scope.md); it does not discard
sleep decisions, work, seeds, role identities, or scores.

| Profile | First result SHA-256 | Repeat result SHA-256 | Deterministic normalized SHA-256 |
|---|---|---|---|
| baseline | `33afd017769c1ec112ac2204db07c923455316adbdd0f92e470a6aaf118e31de` | `ca9bfe5ee618e5fdbe946b810e07ffe1ea828c97de875f073bea4a9407dab73a` | `05adf9998e3dce485752408b041d63c6797a8293d394436139de63ce23803e5a` |
| strength-case | `1198c5e35cc8959a677a0840fecb8d99ea83bd4fcf94d0a8735d06d61f6d1827` | `d302435a352217e6691dfa52f996611f3bd35e404484f8ae5d84909dd75b51ff` | `181361c534a2742d0d1259af3a8c4af52b3e094590482b02d1678bc4065bb2d9` |
| hardest-case | `7269356bdc546ebdd138547e4b3ce43caa66f0db390e0ddf09bf4742dd692116` | `015726a647ed4bf6afa0501765f55a1697bf27eb4abe1e831e68df6a4ed9ae55` | `dc6f547a86bd165891ba3dd4b5d628e29f0019bf51b3853e75bd332545bc24ea` |

These repeats establish only same-environment deterministic behavior with
the declared timing exclusion. The v1 profile constructs final-source
objects before training and scores each seed before later seeds train; it
does not meet the later v5 global-final-seal contract. The NumPy reference
heads are not fully matched for initialization, learning updates, replay,
or capacity. No cross-device/version tolerance or general superiority
claim follows from these runs.
