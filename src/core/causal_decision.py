"""Probability-free selection from one bounded causal generation.

Inputs are existing allowed-candidate requests, raw provider output and explicit
model/token limits. Outputs preserve raw evidence and an exact selected ID.
No model, tokenizer, IO, execution permissions, calibration or physical-work
authentication lives here. Source identities and usage are provider assertions;
native loading and measured computation require a separate outer contract.
"""

from dataclasses import dataclass, replace
from string import hexdigits

from src.core.typed_decision import DecisionRequest, _count, _text

MAX_RESPONSE_BYTES = 4096


@dataclass(frozen=True, slots=True)
class CausalModelReference:
    model_id: str
    revision: str

    def __post_init__(self) -> None:
        if type(self) is not CausalModelReference:
            raise ValueError("expected exact CausalModelReference")
        _text(self.model_id, "model_id", 256)
        if (
            type(self.revision) is not str
            or len(self.revision) != 40
            or any(char not in hexdigits[:16] for char in self.revision)
        ):
            raise ValueError("model revision must be an immutable40-character lowercase hex pin")


@dataclass(frozen=True, slots=True)
class CausalGeneration:
    request_id: str
    model: CausalModelReference
    text: str
    finish_reason: str | None
    reported_input_tokens: int | None = None
    reported_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if type(self) is not CausalGeneration or type(self.model) is not CausalModelReference:
            raise ValueError("expected exact CausalGeneration and model reference")
        self.model.__post_init__()
        _text(self.request_id, "request_id", 64)
        if type(self.text) is not str:
            raise ValueError("raw generation text must be an exact string")
        # Bound the encoding allocation even when the caller bypasses transport.
        if len(self.text) > MAX_RESPONSE_BYTES:
            raise ValueError("raw generation text exceeds4096UTF-8 bytes")
        try:
            size = len(self.text.encode("utf8"))
        except UnicodeError as error:
            raise ValueError("raw generation text must be valid UTF-8") from error
        if size > MAX_RESPONSE_BYTES:
            raise ValueError("raw generation text exceeds4096UTF-8 bytes")
        # Why: empty, prose or truncated outputs are still bounded raw evidence.
        # Only the decoder can qualify a successful allowed-ID selection.
        if self.finish_reason is not None:
            _text(self.finish_reason, "finish_reason", 64)
        for name in ("reported_input_tokens", "reported_output_tokens"):
            value = getattr(self, name)
            if value is not None:
                _count(value, name)


@dataclass(frozen=True, slots=True)
class GeneratedDecisionResult:
    request_id: str
    selected_id: str
    generation: CausalGeneration

    def __post_init__(self) -> None:
        if (
            type(self) is not GeneratedDecisionResult
            or type(self.generation) is not CausalGeneration
        ):
            raise ValueError("expected exact generated decision and raw evidence")
        self.generation.__post_init__()
        _text(self.request_id, "request_id", 64)
        _text(self.selected_id, "selected_id", 64)
        if (
            self.request_id != self.generation.request_id
            or self.selected_id != self.generation.text
            or self.generation.finish_reason != "stop"
            or any(
                value is None or value == 0
                for value in (
                    self.generation.reported_input_tokens,
                    self.generation.reported_output_tokens,
                )
            )
        ):
            raise ValueError(
                "generated decision requires matching completed evidence and known positive token counts"
            )


def decode_causal_decision(
    request: DecisionRequest,
    generation: CausalGeneration,
    *,
    expected_model: CausalModelReference,
    max_input_tokens: int,
    max_output_tokens: int,
) -> GeneratedDecisionResult:
    """Qualify exact text/support/binding/usage, without invented probabilities.

    Counts remain provider-reported token totals, not FLOPs or authenticated
    physical work. A missing count cannot establish this bounded contract.
    """
    if type(request) is not DecisionRequest or type(generation) is not CausalGeneration:
        raise ValueError("expected exact request and raw causal generation")
    if type(expected_model) is not CausalModelReference:
        raise ValueError("expected exact pinned causal model reference")
    request.__post_init__()
    generation.__post_init__()
    expected_model.__post_init__()
    for name, limit in (
        ("max_input_tokens", max_input_tokens),
        ("max_output_tokens", max_output_tokens),
    ):
        _count(limit, name)
        if limit == 0:
            raise ValueError("causal token limits must be positive")
    if generation.request_id != request.request_id or generation.model != expected_model:
        raise ValueError("causal output request or pinned model differs")
    if generation.finish_reason != "stop" or generation.text not in {
        candidate.candidate_id for candidate in request.allowed_candidates
    }:
        raise ValueError("completed causal output must be exactly one allowed candidate ID")
    for count, limit in (
        (generation.reported_input_tokens, max_input_tokens),
        (generation.reported_output_tokens, max_output_tokens),
    ):
        if count is None or not 1 <= count <= limit:
            raise ValueError("known positive causal token counts must fit the explicit limits")
    # Detach mutable record shells; exact str/int leaves are already immutable.
    owned = replace(generation, model=replace(generation.model))
    return GeneratedDecisionResult(request.request_id, generation.text, owned)


def validate_generated_decision_result(
    request: DecisionRequest,
    result: GeneratedDecisionResult,
    *,
    expected_model: CausalModelReference,
    max_input_tokens: int,
    max_output_tokens: int,
) -> None:
    """Revalidate at a receiver boundary, including bypassed frozen fields."""
    if type(result) is not GeneratedDecisionResult:
        raise ValueError("expected exact GeneratedDecisionResult")
    result.__post_init__()
    decode_causal_decision(
        request,
        result.generation,
        expected_model=expected_model,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
    )
