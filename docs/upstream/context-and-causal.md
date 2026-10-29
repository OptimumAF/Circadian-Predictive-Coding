# Context LM and causal LM: pinned native contracts

This is source/metadata inspection, not model execution or task accuracy evidence.
Context Language Models and Contrastive Language Models are separate families.
No model weights, tokenizer vocabulary, dataset examples, packages or device state
were loaded. Full R5.1 remains open. The original bounded R5.2 typed-decision
baseline was accepted separately; that does not qualify these unexecuted native
Context LM or causal interfaces.

## Official release metadata recheck — 2026-10-26

[Issue189](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/189)
rechecked the public source/release metadata without model execution. The
[current-commit response](https://api.github.com/repos/facebookresearch/context-language-models/commits?per_page=1)
still returned `483198229ab7b90c0eed4651306ab3b3505aaf50` (2026-10-14), matching
the original pin below. The inspected
[release response](https://api.github.com/repos/facebookresearch/context-language-models/releases?per_page=5)
was empty. Neither a GitHub source pin nor an empty releases list establishes
whether model weights exist elsewhere.

The exact pinned [README](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/README.md)
and [RL instructions](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_rl/README.md)
still did not identify a released trained RL model handle. The advertised
zero-shot harness and community integration packages are code/interfaces; the
linked ContextBench artifact is a dataset. The named Qwen3.5-9B base and external
training instructions do not supply an immutable trained Context LM checkpoint.
No weight/tokenizer revision or applicable trained-weight terms can be pinned
from this bounded evidence. They remain **unresolved external prerequisites**,
not permission to substitute random/unofficial weights or label a causal base
as the trained RL result. Existing source and data terms remain separate.

Five bounded direct requests returned32,347raw bytes, allHTTP200; request time
2.568708s/network phase18.547254s. One ancillary official web view verified the
public repository; its transport bytes were not metered. Three unused requests
were not repurposed into speculative model searches. No weights/vocabulary/
dataset examples/upstream imports/model/RNG/device work or installs. This is a
dated bounded finding, not a global absence claim or legal interpretation.
[Protocol, request ledger, captured permitted metadata and self-review](../../artifacts/runs/contextlm-release-metadata-20261026/RESULTS.md).

## Pinned records

| Entry | Exact inspected revision | Meaning |
| --- | --- | --- |
| Context LM | [`facebookresearch/context-language-models@483198229ab7b90c0eed4651306ab3b3505aaf50`](https://github.com/facebookresearch/context-language-models/tree/483198229ab7b90c0eed4651306ab3b3505aaf50), commit date 2026-10-14 | Agent harness/source pin, not a trained checkpoint. |
| Context paper | [arXiv 2609.37725v1](https://arxiv.org/html/2609.37725v1), 2026-09-29 | Context-editing mechanism and reported research methods; no local reproduction. |
| Causal reference | [`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) | Official post-trained causal model with a vision encoder; separate from an Open-Jev scalar task head or Context LM RL checkpoint. |

The Qwen revision was independently returned by the official model metadata API,
not inferred from a moving main branch. Its card identifies the upstream base as
`Qwen/Qwen3.5-2B-Base`; the inspected artifact is the post-trained `-2B` release.
Do not equate these names or substitute the base for a released decision head.

## Context LM native interface

The [harness documentation](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_harness/README.md)
and [agent source](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_harness/clm_agent/harness.py)
implement a Harbor agent using LiteLLM chat messages and tool replies. The model
can issue a `bash` call with a `command` argument. This is an external-action
interface, not an allowed-candidate probability vector, embedding pooler or
classification head. Merely parsing this contract does not authorize commands.

The [context environment](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_harness/context_env/env.py)
mirrors editable conversation turns in `/tmp/.live_ctx/LIVE_CTX_MAIN.txt` before a
command. Default protection retains the first two original messages (system and
task). After an edit, the harness parses the file, checks the edit/budget policy,
and replaces the live message list. Fit/shrink policies, context/output reserves,
rollback attempts and LM-call limits remain part of the upstream execution cost.
They are not equivalent to a single stateless prediction.

The [transcript parser](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_harness/context_utils/context_string.py)
uses `[[CTX_TURN ... role=...]]` boundaries. It reattaches the protected prefix,
keeps assistant body text as assistant, converts other edited body roles to user,
drops empty text and merges adjacent compatible roles. It does not reconstruct
native tool-call structure from the edited transcript. A local codec must preserve
this distinction rather than claim arbitrary lossless message round trips.

The [token accounting source](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_harness/utils/tokens.py)
uses tiktoken (`o200k_base` by default, then `cl100k_base`) and can fall back to a
character estimate. It includes reasoning/content/tool names and arguments.
Those are accounting methods, not proof of exact Qwen tokenizer counts. The
harness can calibrate its gate against server usage; actual usage is unobserved
here. Optional tokenizer-aware FLOPs paths exist but were not executed or fully
qualified in this inspection.

### Objective, release and terms

The [RL description](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_rl/README.md)
uses Qwen3.5-9B with task-reward GRPO plus a success-gated efficiency term on
context-edit tokens. It documents OpenResearcher `rl` training, 3,040 prompts and
one epoch. Training infrastructure is external (ProRL-Agent-Server/Slime); the
repository supplies patches and describes additional metadata that those patches
do not produce. This is neither a contrastive sentence objective nor next-token
classification of the project's allowed IDs.

No trained Context LM checkpoint ID/revision was identified in the inspected
README/source; the observed GitHub releases response is empty. This is a bounded
finding, not proof that no checkpoint exists anywhere. A released RL weight pin,
its tokenizer and applicable model terms remain unresolved prerequisites for
claims about that trained model. Zero-shot harness use of a separately pinned
causal model is a different condition.

[Code LICENSE](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/LICENSE)
is CC BY-NC 4.0; its [NOTICE](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/NOTICE)
also identifies dependency/template attributions. The paper has its own CC BY 4.0
notice. Neither becomes project MIT. [ContextBench metadata](https://huggingface.co/datasets/context-language-model/ContextBench/tree/52bd961207ae9a0a79631c6fb65b162af9272983)
declares Apache-2.0 and test splits; only metadata was read, with no TRAIN
relabeling or benchmark access. Dataset/component rights require their own review.

## Causal control: loader, tokens and output

The [pinned config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json)
identifies `Qwen3_5ForConditionalGeneration`, with a 24-layer hybrid text decoder,
248,320 vocabulary entries, tied embeddings and a vision encoder. Its [card](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/README.md)
describes causal pre/post-training and text-only chat use. It recommends compatible
recent Transformers/vLLM/SGLang serving; its moving-latest install suggestion is
not an exact local dependency pin. No framework version was installed or tested.

The [tokenizer configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer_config.json)
identifies Qwen2Tokenizer and special-token/chat settings; the [chat template](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/chat_template.jinja)
handles role/tool/multimodal formatting and thinking configuration. Metadata and
the template were read; full tokenizer JSON, merges and vocabulary were not.
A future load must pin compatible code plus all required tokenizer artifacts and
record the actual thinking/template/generation policy.

Native outputs are generated tokens/text (and model-specific token logits), not
project candidate probabilities. A task prompt, label-token mapping, sequence
scoring/termination policy and calibration contract would need separate validation;
no class softmax or raw candidate logits may be invented from a text reply.
The [pure generated-ID contract](../causal-generated-decisions.md) now validates
exact request/model/allowedID/stop/reported-usage bindings on zero-model fixtures,
without probabilities. This is not tokenizer/template/provider/model execution
or authentication of physical weights/work; those gates remain separate.
[The model LICENSE](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE)
is Apache-2.0; that does not license unrelated Context LM code or benchmark data.

## Hardware and remaining qualification

Official safetensors metadata totals 2,274,069,824 parameters. A simple two-byte
weight-only estimate is 4,548,139,648 bytes; runtime/vision buffers, cache, loading
copies and output memory are additional. Against the stage's observed RTX3080
10,240 MiB / 68,399,599,616-byte RAM, small text-only inference is a candidate,
not a demonstrated fit. The advertised 262,144-token limit is not a claim that
this context fits that GPU. BF16/kernel/framework compatibility is untested.
The Context LM paper's 9B RL model or 27B serving examples exceed 10 GiB in
unquantized two-byte weights alone (rough estimates); no offload/quantization or
paid endpoint is authorized by this record.

Evidence: [request ledger](../../artifacts/runs/native-decision-contracts-20261022/context/fetch-ledger.json)
and [contract summary](../../artifacts/runs/native-decision-contracts-20261022/context/contract-summary.json).
Exactly 20 bounded metadata requests returned 761,485 raw payload bytes; largest
457,400 bytes, all within 20-second/512-KiB response and 1-MiB total limits. Web
snippets/transport overhead were not metered. One local regex inspection emitted
invalid-escape SyntaxWarnings but completed; no request failed or retried. No
weights/vocabulary downloads, upstream imports, model/RNG/device work or execution
accuracy are claimed. Next work is an explicitly reviewed native wire bridge or
separately pinned execution protocol, not an inference claim from source alone.
