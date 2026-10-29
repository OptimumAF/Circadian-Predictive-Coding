"""One local causal text observation using the existing immutable decision input.

The caller supplies an eval model/tokenizer, inference context, pinned identity
and native resource admission. This module builds a bounded text-only prompt and
preserves observed token counts/raw output; it performs no loading, IO, training,
effects or probability fabrication. Strict selection remains a separate core
decode_causal_decision call so rejected raw observations remain available.
"""

from dataclasses import replace
import json
from typing import Any, Callable

from src.core.causal_decision import CausalGeneration, CausalModelReference
from src.core.typed_decision import DecisionRequest, _count

SYSTEM_TEXT = (
    "Select the support intent. Reply with exactly one allowed candidate_id and no other text."
)


def causal_messages(request: DecisionRequest) -> list[dict[str, str]]:
    """Serialize original candidate order without adding targets or permissions."""
    if type(request) is not DecisionRequest:
        raise ValueError("expected exact DecisionRequest")
    request.__post_init__()
    payload = dict(
        request_id=request.request_id,
        query=request.query,
        allowed_candidates=[
            dict(candidate_id=c.candidate_id, description=c.description)
            for c in request.allowed_candidates
        ],
    )
    return [
        dict(role="system", content=SYSTEM_TEXT),
        dict(role="user", content=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
    ]


def _token_row(tensor: Any, maximum: int, vocab_size: int) -> list[int]:
    # Why: inspect dimensions before tolist, bounding the detached Python copy.
    if len(tensor.shape) != 2 or tensor.shape[0] != 1 or not 1 <= tensor.shape[1] <= maximum:
        raise ValueError("expected one bounded nonempty token sequence")
    row = tensor.tolist()[0]
    if (
        type(row) is not list
        or len(row) != tensor.shape[1]
        or any(type(token) is not int or not 0 <= token < vocab_size for token in row)
    ):
        raise ValueError("expected bounded native integer token IDs")
    return row


def observe_causal_choice(
    request: DecisionRequest,
    *,
    model: Any,
    tokenizer: Any,
    reference: CausalModelReference,
    max_input_tokens: int,
    max_output_tokens: int,
    eos_token_id: int,
    vocab_size: int,
    before_generate: Callable[[], None] = lambda: None,
) -> CausalGeneration:
    """Run one greedy text generation; retain unqualified raw text and usage.

    Native caller owns inference_mode, a pristine generation configuration and
    clock/memory/forward guards. Token counts include EOS. Early non-EOS stopping
    is unobserved, never falsely upgraded to successful completion. Tokenizer
    decodes all remaining special tokens verbatim; only final EOS is structural.
    """
    messages = causal_messages(request)
    # Freeze scalar bindings before calling mutable third-party objects.
    request_id = request.request_id
    if type(reference) is not CausalModelReference:
        raise ValueError("expected exact causal model reference")
    reference.__post_init__()
    reference = replace(reference)
    for name, value in (
        ("max_input_tokens", max_input_tokens),
        ("max_output_tokens", max_output_tokens),
        ("vocab_size", vocab_size),
    ):
        _count(value, name)
        if not 1 <= value <= 262144:
            raise ValueError("token dimensions must be positive and bounded by262144")
    _count(eos_token_id, "eos_token_id")
    if eos_token_id >= vocab_size or model.training is not False:
        raise ValueError("expected valid EOS and eval model")
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
    )
    if set(inputs) != {"input_ids", "attention_mask"}:
        raise ValueError("expected text-only token IDs and attention mask")
    prefix = _token_row(inputs["input_ids"], max_input_tokens, vocab_size)
    mask = _token_row(inputs["attention_mask"], max_input_tokens, 2)
    if len(mask) != len(prefix) or any(value != 1 for value in mask):
        raise ValueError("expected a complete unpadded input sequence")
    before_generate()
    output = model.generate(
        **inputs,
        max_new_tokens=max_output_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        logits_to_keep=1,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        return_dict_in_generate=False,
        output_scores=False,
        output_logits=False,
    )
    tokens = _token_row(output, len(prefix) + max_output_tokens, vocab_size)
    if tokens[: len(prefix)] != prefix or len(tokens) <= len(prefix):
        raise ValueError("generation must extend the unchanged observed input")
    generated = tokens[len(prefix) :]
    stopped = generated[-1] == eos_token_id
    content = generated[:-1] if stopped else generated
    if eos_token_id in content:
        raise ValueError("generation continued after EOS")
    text = tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    reason = (
        "stop" if stopped else "length" if len(generated) == max_output_tokens else "unobserved"
    )
    return CausalGeneration(request_id, reference, text, reason, len(prefix), len(generated))
