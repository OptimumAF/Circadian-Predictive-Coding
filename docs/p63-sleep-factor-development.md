# P6.3 guarded sleep-factor development scoring contract

## Frozen question and source

This is the scored continuation of the [train-only c3 preflight](p63-sleep-factor-preflight.md), not a new search over settings. Keep its nine arms, development seeds 67/71/73, A/B arrived source and disjoint roles, initial seeds, 12+12 wake epochs, one A-boundary guarded sleep for each circadian arm, zero model replay, and fixed width-eight/12 controls. The reference is the complete saved c3 result at `artifacts/runs/p63-sleep-factor-preflight/sleep-factor-preflight.result.json`, byte SHA-256 `e3140d5a0529dc3023b90b60801d8f9caf2e15dd8edb4a85fa875f76f60d624d`. Its independent repeat has the same result bytes. Require the c3 public verifier, its request/result audit hashes, and this exact byte hash before source construction.

For each seed, train all arms through A and the guarded boundary, then retain after-A model snapshots without evaluating the outer role. Construct B only after A work, train all arms through B, and derive the complete c3 `SleepFactorPreflight` train-only fact object. Require exact equality with the saved c3 fact object **across all 27 cells before the first outer-selection input or target access**. A mismatch or incomplete seed fails the entire run with zero outer scores. Final-test source inputs/labels are never released. The ten reserved confirmation seeds remain unopened.

Only after this global gate, score each arm on A outer selection using its after-A snapshot (`A_after_A`), and on A and B outer selection using its after-B model (`A_after_B`, `B_after_B`). These are development scores, never final or confirmation scores. Each arm receives exactly three outer evaluations (24+24+12 rows), for 81 evaluations and 1,620 evaluated examples total. No score affects the guard, training, settings, or arm inclusion.

## Outcomes and contrasts

Use `phase6_two_task_accuracy_v1`: `final_mean_task_accuracy = (A_after_B+B_after_B)/2` and `signed_forgetting_A = A_after_A-A_after_B`. Report all three accuracies, both primary measures, and the optional retention ratio (`A_after_B/A_after_A` if the denominator is positive, otherwise null). All inputs and derived values must be finite, and output verification recomputes them. No post-hoc checkpoint, seed, or metric selection is allowed.

Prespecify exactly three paired comparisons per seed, with **left minus right** for all three accuracies and both primary measures: `structure_only-neutral_sham`, `homeostasis_only-neutral_sham`, and `gating_reset-gating_sham`. Report every seed's paired values and the three-seed mean and sample standard deviation descriptively; three seeds are insufficient for a confirmatory significance claim. Also report ordinary `pc_8`/`neutral_sham` outcome parity, all nine arms, and the c3 guard, wake, replay, width, and parameter facts beside outcomes. The structural arm's transient width-nine/37-parameter cost remains visible even if its final width is eight. Width-12 arms are prospective capacity references with more parameters and different per-update compute. No cost-adjusted winner is defined.

The c3 preflight's 648 planned wake optimizer updates remain the only training updates. The scored route adds forward evaluation, with a 700-update hard prelaunch cap, 120-second local child wall limit, and observed whole-worker RSS ceiling 256 MiB. It writes exclusive request/result/audit or failure files to an ignored local directory and independently repeats the deterministic result in another directory. The result excludes wall/RSS measurements; the audit records them with process scope and sampling caveat. Require exact repeat result bytes, read back all IDs, role hashes, facts, metrics, paired contrasts, and artifact hashes. Null or negative contrasts remain published.

**Why this order:** the unscored c3 artifact is an independent, previously completed witness for the precise train path. Checking the complete object globally prevents partial outer release when a later seed fails. Delaying even `A_after_A` outer evaluation until the gate prevents the first seed's score from influencing completion or repair of later cells. Deep copies of the small after-A models preserve the checkpoint without a second training pass.

Protocol ID: `continual_sleep_factor_outer_development_v1`. This exploratory development factor does not close P6.3c's schedule/full-minus-one or independent confirmation work.

## Frozen implementation identities before public scoring

The c3 13-source map remains frozen at sorted-map SHA-256 `80c16f348f040da5c4e42e89e2787e1abf47ef7cdc7612fdd344e9dde76339fa`. Before the first outer score, the scored adapter also pins:

| Additional source | SHA-256 |
|---|---|
| `src/app/continual_sleep_factor_development.py` | `e6a498c107d0071b822309907691e225b1a1ae828c93f8435c94ba0cfa94bf88` |
| `src/core/continual_metrics.py` | `8eb491da4221a232a32a7cc1eb7b08a3c36a96fb748543450ac068dac3ae7f78` |
| `scripts/run_p63_sleep_factor_preflight.py` | `dae68cbdb8ef0464ff9a4971bfaedc2e50dc4d75980693a236a5a92eb38d0d3c` |

The sorted combined 16-source map SHA-256 is `f618900fde912e7b7fdfecf0ebe329266e6c69604133bbd425560384235daa77`; the scored adapter byte SHA-256 is `08dedd66328f41ab24403e29ca0f8bc8c4b10030528f9e4a2c745b4f0d330634`. These identities were recorded after implementation/static checks and before any scored test or public process. The source map is selected, not a full dependency-tree hash.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p63_sleep_factor_development --output-dir artifacts/runs/p63-sleep-factor-development
.\.venv\Scripts\python.exe -m scripts.run_p63_sleep_factor_development --output-dir artifacts/runs/p63-sleep-factor-development-repeat
```
