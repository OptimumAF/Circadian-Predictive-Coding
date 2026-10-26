# Matched native exposure: four-method observations

One fixed seed17/H32 known-task study. Exactly six new PC-family cases; the accepted BP pair is reused. All eight cases qualify, both PC-neutral parity records qualify, and all14signed pairs/448points/896cue contrasts are retained. No production code changed.

## Fit and adaptation

Each entry is before F / after48total native adaptation calls / after16A-return updates. F48 checkpoints0/12/24/48 align with replay logicalF0/4/8/16; Areturn0/4/8/16 stays fixed. Correct actions are teacher-forced rows out of18, not live success.

| Source | Method | Allocation | F correct/18 | A correct/18 |
|---:|---|---|---|---|
|0|bp|f48|1 / 16 / 16|2 / 14 / 16|
|0|bp|replay|1 / 15 / 15|2 / 14 / 16|
|0|pc|f48|1 / 10 / 10|2 / 10 / 10|
|0|pc|replay|1 / 10 / 10|2 / 10 / 10|
|0|neutral_cpc|f48|1 / 10 / 10|2 / 10 / 10|
|0|neutral_cpc|replay|1 / 10 / 10|2 / 10 / 10|
|0|gated_cpc|f48|1 / 10 / 10|2 / 10 / 10|
|0|gated_cpc|replay|1 / 10 / 10|2 / 10 / 10|
|128|bp|f48|12 / 17 / 16|17 / 16 / 16|
|128|bp|replay|12 / 14 / 15|17 / 15 / 16|
|128|pc|f48|10 / 10 / 10|10 / 10 / 10|
|128|pc|replay|10 / 10 / 10|10 / 10 / 10|
|128|neutral_cpc|f48|10 / 10 / 10|10 / 10 / 10|
|128|neutral_cpc|replay|10 / 10 / 10|10 / 10 / 10|
|128|gated_cpc|f48|10 / 10 / 10|10 / 10 / 10|
|128|gated_cpc|replay|10 / 10 / 10|10 / 10 / 10|

Source128 PC, neutral and active remain5/9 on each cue at every recorded point. Stable majority-action accuracy is not retention of an acquired task. Source0 starts F0/9blue+1/9red and A1/9each, improving to5/9 per cue after training; all intermediate curves remain in raw reports and analysis.json. BP source128 A-red9/9→8/9→8/9 for both allocations; its initial exact trace is not fully retained/recovered. At source0 afterreturn BP F48 Ablue8/red8 versus replayblue9/red7: equal means hide opposite cue signs, which are preserved.

## CE endpoints

Mean across both cues. Higher CE means worse fit. These raw method gaps include prior benchmark differences, not an isolated causal estimate of a gate or plasticity loss.

| Source | Method | Allocation | F CE afteradaptation / return | A CE afteradaptation / return |
|---:|---|---|---|---|
|0|bp|f48|0.413788335 / 0.602306287|1.238179934 / 0.624667815|
|0|bp|replay|0.532925213 / 0.615345148|0.459278134 / 0.345347418|
|0|pc|f48|1.245097437 / 1.214914500|1.315498203 / 1.273494757|
|0|pc|replay|1.279938447 / 1.249337138|1.298342483 / 1.258575819|
|0|neutral_cpc|f48|1.245097437 / 1.214914500|1.315498203 / 1.273494757|
|0|neutral_cpc|replay|1.279938447 / 1.249337138|1.298342483 / 1.258575819|
|0|gated_cpc|f48|1.266776076 / 1.242099223|1.330972426 / 1.297566494|
|0|gated_cpc|replay|1.297898466 / 1.273096956|1.316248621 / 1.284472789|
|128|bp|f48|0.199679002 / 0.380381247|0.871204338 / 0.376369456|
|128|bp|replay|0.490402547 / 0.518670167|0.449583141 / 0.253679101|
|128|pc|f48|1.032142524 / 1.015523879|1.057100598 / 1.014712268|
|128|pc|replay|1.079740347 / 1.063278656|1.023439040 / 0.985305924|
|128|neutral_cpc|f48|1.032142524 / 1.015523879|1.057100598 / 1.014712268|
|128|neutral_cpc|replay|1.079740347 / 1.063278656|1.023439040 / 0.985305924|
|128|gated_cpc|f48|1.142706788 / 1.130664068|1.149949079 / 1.128719870|
|128|gated_cpc|replay|1.168966989 / 1.157049035|1.133693033 / 1.112797985|

F48 has lower observed F CE and higher A CE than replay at these endpoints for all four methods. Active CPC has higher CE than neutral at the shown post-adaptation/return endpoints, with equal accuracy; no circadian advantage is established. PC and neutral match exactly in270TRAIN probability rows and32complete F/A score+decision-diagnostic records per allocation. This does not assert unavailable raw probe-probability equality. All native policies remain method-specific.

## Frozen signed comparisons

Four F48-minus-replay pairs, and active-minus-neutral, neutral-minus-PC, PC/neutral/active-minus-BP within each allocation:14pairs×32points×2cues. Positive CE means higher first-named loss; positive accuracy means higher first-named accuracy. No endpoint, seed, metric or baseline was selected after results. Unknowns are never coerced to zero; the raw tiny floating mean for BP source0 afterreturn remains unchanged.

## Work and resources

Each condition:256native calls/8064rows/P94/846rows. F48 uses48 F updates/432rows per probe with zero replay. Replay uses16 F updates+32old-A calls/432rows, separately paid288replay rows per probe. Both keep the same original16A return/144rows. Final evaluation and games are unavailable; source/actor/input/RNG/original-budget proofs pass. The original128receipts270TRAIN probabilities match prior method-specific controls.

| Method | Allocation | New or reused | Body seconds | Supervised seconds | Peak RSS bytes |
|---|---|---|---:|---:|---:|
|bp|f48|reused|39.863621|44.474149|435785728|
|bp|replay|reused|51.047682|55.404701|517976064|
|pc|f48|new|30.037970|32.865700|360833024|
|pc|replay|new|42.812077|46.018249|430305280|
|neutral_cpc|f48|new|57.678255|64.551723|357421056|
|neutral_cpc|replay|new|41.147842|44.703007|429932544|
|gated_cpc|f48|new|48.852603|52.542201|357236736|
|gated_cpc|replay|new|40.717429|43.681640|421085184|

New six cases:1536native/48384rows/P564/5076rows/SGD0/60teacherresets540physics; nominal96768latent rowsteps derive from unchanged PC S2. Full eight including reused BP:2048native/64512rows/P752/6768rows/SGD512/80resets720physics. New replay banks523680logicalB/216validation projection rows511488arrayB; full replay banks698240B/288rows681984B. Matching native training/scoring counts does not equalize bank, validation, projection, copy, native state, Python, RSS or wall costs. Single-run timings do not establish causal efficiency.

CPU1/GPU0/Torch intra+interop1/RSS768MiB, original180s bodies and210/220/225 supervision preserved. All six authorities SPENT/no retries. Eight endpoint snapshot reads per new case retain only four scalar proofs; full native clocks48/64/176/192 and rows432/576/7344/7488 pass. No unlimited checkpoint/state history.

## Limits and next action

Single fixed seed/input and known teacher-forced traces; no heldout/live-gameplay, useful acquired recovery, robustness, generalization, release or repository-wide readiness claim. Simpler BP has stronger observed fit here; this does not rank unseen tasks. All negative and inconclusive historical results remain intact.

Next implementation: enable the existing TRAIN-only realized-PC-update diagnostic for explicit H32 PC/neutral checkpoints0/128. Its numerical reader already supports H32; the public/CLI/checkpoint wide guards currently refuse it. Keep native objectives, doses and seeds fixed, budget diagnostic/counterfactual work, and prove source/optimizer/RNG isolation. Diagnose the common fit plateau before new algorithm changes; no new capture/runner/horizon tuning.

[Exact commands, gates, failures, costs and handoff](../../artifacts/runs/matched-exposure-controls-20261018/RESULTS.md) · [Independent review](../../artifacts/runs/matched-exposure-controls-20261018/REVIEW.md).
