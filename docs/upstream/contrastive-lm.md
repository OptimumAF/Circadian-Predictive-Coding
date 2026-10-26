# Contrastive LM: pinned source contract

Initially inspected on 2026-10-22 without tensor execution. The separately
qualified released CPU bridge below now verifies the selected assets and plain
TRAIN recipe. Full R5.1 and comparative performance remain open.

## Revisions and release identity

Official code is pinned at
[`d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6`](https://github.com/Contrastive-LM/CLM/tree/d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6).
The published [head configuration](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/blob/e939398d4556fcd9400c76fa8c5a513202f42b0a/config.json)
requires Qwen3-8B, last-token pooling and 4096-dimensional embeddings. Head revision
is `e939398d4556fcd9400c76fa8c5a513202f42b0a`; `CLM_v0.1-8B.pt` has published
size 75,557,149 bytes and LFS SHA256
`b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`.
These initial metadata identities were subsequently verified against the local
selected payloads by the CPU bridge. Qwen was separately pinned at `b968826d9c46dd6066d109eabc6255188de91218`.

## Verified serving semantics

[`POST /v1/systemone`](https://github.com/Contrastive-LM/CLM/blob/d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6/src/clm/server.py)
accepts `state`, `model`, `questions`, and optional `temperature`. A `choice`
question contains instructions and an ordered `criteria` map of ID to description.
The [schema](https://github.com/Contrastive-LM/CLM/blob/d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6/src/clm/schema.py)
renders state as prose and appends instructions; candidates are description text,
falling back to the ID only for empty/None descriptions. Choice replies contain
`type`, `choice`, `confidence`, and `probabilities`. Confidence is top probability
minus the mean of the others; ties use insertion order. Probabilities already
contain softmax. Raw logits are absent. Input-token usage counts embedding-cache
misses, not all logical query tokens; output tokens are zero.

The [engine](https://github.com/Contrastive-LM/CLM/blob/d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6/src/clm/engine.py)
uses separate normalized state/action projections and scaled cosine, divided by
request temperature (default one), before softmax. The
[heads loader](https://github.com/Contrastive-LM/CLM/blob/d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6/src/clm/heads.py)
exponentiates checkpoint `logit_scale`, capped at 100. No calibration fit was
verified. The model card describes bidirectional InfoNCE and frozen encoders;
its benchmark claims remain unexecuted here.

CLM delegates tokenization to the embedding server. Metadata names
`Qwen2Tokenizer`, no added BOS, and last-token pooling. Serving sends text with
2048-token truncation; [training recipes](https://github.com/Contrastive-LM/CLM/blob/d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6/train/embed_utils.py)
distinguish plain-text head/tail truncation from structured-state chat templates.
The selected plain-text TRAIN recipe now has executed tokenization/pooling
evidence below; vLLM serving parity remains unverified.

## Execution and terms limits

Default downloads omit revisions/digest checks, accept existing local files, and
hot-reload changed heads. Encoder and minimum-version dependencies are unpinned.
**Why this:** execution needs explicit encoder/head/dependency identities before
reproducibility or calibration claims.

The [encoder metadata](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)
lists 8,190,735,360 BF16 parameters: approximately 15.26 GiB of full weights before
runtime/head/cache overhead, exceeding the observed 10 GiB GPU. CPU loading in
64 GiB RAM was initially a capacity estimate; the selected CPU execution is now
measured below, without a general speed or compatibility claim.
Quantization/offload requires separate parity checks.

Repository and head LICENSE texts are Apache-2.0; encoder card metadata declares
Apache-2.0. Announced training-data grants, overlap and provenance were not audited.
Code/model licensing does not settle those questions.

Evidence: [`clm/inspection.json`](../../artifacts/runs/native-decision-contracts-20261022/clm/inspection.json)
and URL/hash ledger: 20 bounded raw requests, 145,802 bytes, zero failures; one
separate web repository open has unmetered transport. No model execution occurred.

## Conventional learner component

The optional [frozen-feature learner](../clm-head-baseline.md) ports the selected
paired group-masked objective from the pinned
[trainer](https://github.com/Contrastive-LM/CLM/blob/d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6/train/finetune.py),
with AdamW, clipping at1 and cosine OneCycle. The separately recorded trainer
fetch was40,079 bytes/SHA256
`4917046b1b38f4ce3602b039b85dc7f238263b87f040793f6b12446a271bb9d6`.
This is the `clm` paired objective, not the trainer's separate typed-choice loss
variants. Native head/gradient/optimizer parity on tiny frozen tensors does not
qualify the released encoder, trained checkpoint, tokenization/pooling, serving
schema or dataset claims. Those R5.1 and full R5.3 requirements remain open.

## Released CPU bridge: scoped execution and negative task result

[Usage](../clm-frozen-encoder.md) and
[executed results](../../artifacts/runs/clm-frozen-encoder-20261022/RESULTS.md)
record pinned local payload checks and one HF CPU/plain TRAIN-recipe request.
The released head has input4096,width1536,depth3,projection512,GELU,LayerNorm and
no residual. The embedding backbone omits the unused causal lm_head. Exact
no-special-token head/tail truncation, last nonpadding pooling and normalized
features agree with the pinned source reference on the single paid output.
The password request nonetheless selects **refund**, probability0.803503, in both
production and reference. This negative result is retained without tuning/retry.
One encoder forward/80padded tokens took65.158632s including demand paging;
sampled process RSS14,557,581,312B is not a true peak or model-only footprint.
Full vLLM/serving/cache/training-data/other-family qualification remains open.
