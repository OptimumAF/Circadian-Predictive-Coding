# Open-Jev — selected native reference

Initial source/model inspection2026-10-22 executed no weights, tokenizer or
inference. The subsequent [bounded native CPU smoke](../../artifacts/runs/open-jev-native-smoke-20261022/RESULTS.md)
loaded these exact weights/source and executed two fixed requests, independently
accepted. CPU float32 changes released BF16 arithmetic; calibration and task
quality remain open. This is distinct from Contrastive LM, Context LM and causal
Qwen. No source-reported benchmark result is ours.

## Exact selected identities

| Item | Identity |
|---|---|
| Inspected source | [`bd4118882f733574a3250a4b65fe4d884130c08b`](https://github.com/Zefan-Cai/Open-Jev/tree/bd4118882f733574a3250a4b65fe4d884130c08b) |
| Released adapter/head | [`ZefanCai/Open-Jev-2B@0c7aa498b1627be8da4acf34c863ff0ee0a92785`](https://huggingface.co/ZefanCai/Open-Jev-2B/tree/0c7aa498b1627be8da4acf34c863ff0ee0a92785) |
| Required base/tokenizer | [`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) |
| Upstream deployment recipe source | `80ca8e81d08992cb2a4cbb6a0caa1355ad3e5aee`; separate from inspected source |
| Package-declared training source | `99e881108c6cacadafd364088505e84975ca43fc`; not a locally reproduced training run |

[Pinned package model configuration](https://huggingface.co/ZefanCai/Open-Jev-2B/blob/0c7aa498b1627be8da4acf34c863ff0ee0a92785/package/checkpoint/model.json)
requires base revision above, max_length4096/rank8. The package omits base weights.
Loader `jev.serving.load_predictor(checkpoint=...)` loads native adapter/head and
saved temperature1.518796342858676; generic text-generation PEFT loading is not
equivalent. [Loader source](https://github.com/Zefan-Cai/Open-Jev/blob/bd4118882f733574a3250a4b65fe4d884130c08b/jev/serving.py).

## Native semantics

`POST /v1/systemone` accepts state/questions. Choice criteria map ID→description;
both names and descriptions enter independently scored candidate prompts.
Requests may also contain native Noul/Score, which our first bridge does not
implement. Choice replies provide choice/probabilities/confidence; raw logits
are absent. First supplied candidate wins exact ties. Confidence is a
chance-relative margin, `(max(p)-1/K)/(1-1/K)`, singleton1, not correctness probability.
The local bridge preserves request order/probabilities and raw body; it does not
softmax or recalibrate replies. [Request/response source](https://github.com/Zefan-Cai/Open-Jev/blob/bd4118882f733574a3250a4b65fe4d884130c08b/jev/api.py),
[confidence source](https://github.com/Zefan-Cai/Open-Jev/blob/bd4118882f733574a3250a4b65fe4d884130c08b/jev/metrics.py).

Native tokenizer uses the exact base revision, right padding, user chat template,
generation prompt and thinking disabled; last nonpadding token supplies the
scalar head. Overlength input refuses instead of truncating. LoRA/scalar training
uses declared NLL plus Brier weighting; calibration is separate. Preserve these
semantics for future adapters. [Model source](https://github.com/Zefan-Cai/Open-Jev/blob/bd4118882f733574a3250a4b65fe4d884130c08b/jev/model.py),
[training provenance](https://huggingface.co/ZefanCai/Open-Jev-2B/blob/0c7aa498b1627be8da4acf34c863ff0ee0a92785/package/provenance.json).

## Terms, resources and remaining gates

Code MIT; package/base weights Apache2 are separate. Declared training consumed
80,816 rows; the redistributable split omits1,700 Wikispeedia records whose
redistribution permission is unresolved. No dataset rows were fetched or adopted.
[Model card](https://huggingface.co/ZefanCai/Open-Jev-2B/blob/0c7aa498b1627be8da4acf34c863ff0ee0a92785/README.md),
[notices](https://github.com/Zefan-Cai/Open-Jev/blob/bd4118882f733574a3250a4b65fe4d884130c08b/THIRD_PARTY_NOTICES.md).

Observed host RTX3080/10GiB and68,399,599,616B physical RAM. Upstream reports
6.917GiB allocator peak on particular H200 workloads and a CPU deployment; neither
establishes local fit, latency, dtype parity or clean environment compatibility.
Two-billion-scale BF16 is a plausible local candidate; run no automatic fallback
or quantization. Inference extras require Transformers5.10.2/PEFT0.19.1/Accelerate
1.13.0 and Torch>=2.8. [Pinned deployment guide](https://github.com/Zefan-Cai/Open-Jev/blob/bd4118882f733574a3250a4b65fe4d884130c08b/docs/deployment.md),
[dependency source](https://github.com/Zefan-Cai/Open-Jev/blob/bd4118882f733574a3250a4b65fe4d884130c08b/pyproject.toml).

Local native execution, exact loaded tensor bytes, dependency installation,
tokenizer/candidate-order controls, calibration and task retention remain open.
The codec does not authenticate a loaded checkpoint or correlate an echoed nonce;
the upstream reply lacks one. Source inspection does not establish task success.
[Evidence and resource failure](../../artifacts/runs/native-decision-contracts-20261022/RESULTS.md).
