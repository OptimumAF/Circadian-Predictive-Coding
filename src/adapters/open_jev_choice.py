"""Pure Choice wire bridge for the inspected Open-Jev API, without inference.

Owns bounded request encoding and native reply validation. Does not transport
requests, load models, authenticate provenance, train, calibrate, execute actions
or invent raw logits/token counts absent from the wire response.
"""

from dataclasses import dataclass
import json
from math import isclose, isfinite
from typing import Any

from src.core.typed_decision import DecisionRequest, ProbabilityDecisionResult, _text

MAX_REPLY_BYTES = 65536
MAX_JSON_DEPTH = 64
NATIVE_MASS_TOLERANCE = 1e-6


@dataclass(frozen=True)
class OpenJevChoiceReply:
    request_id: str
    question_id: str
    choice: str
    probabilities: tuple[tuple[str, float], ...]
    confidence: float
    raw_response: bytes


def _request(request: DecisionRequest, question_id: str) -> None:
    if type(request) is not DecisionRequest:
        raise ValueError("expected exact DecisionRequest")
    request.__post_init__()
    _text(question_id, "question_id", 64)


def build_open_jev_choice(
    request: DecisionRequest, instructions: str, *, question_id: str = "decision"
) -> dict[str, Any]:
    """Preserve native candidate order; caller supplies instructions without labels."""
    _request(request, question_id)
    _text(instructions, "instructions", 2048)
    return dict(
        model="open-jev",
        state=request.query,
        questions={
            question_id: dict(
                type="choice",
                instructions=instructions,
                criteria={
                    item.candidate_id: item.description for item in request.allowed_candidates
                },
            )
        },
    )


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate response JSON key")
        result[key] = value
    return result


def _float(token: str) -> float:
    value = float(token)
    if not isfinite(value):
        raise ValueError("nonfinite or overflow response JSON number")
    return value


def _integer(token: str) -> int:
    value = int(token)
    if abs(value) > 2**53 - 1:
        raise ValueError("response JSON integer exceeds exact scalar bound")
    return value


def _constant(token: str):
    raise ValueError("nonfinite response JSON constant")


def _probability(value: Any) -> float:
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("native probability/confidence must be finite in [0,1]")
    return float(value)


def _check_depth(body: bytes) -> None:
    # Why this: JSON parser recursion limits vary by host/test runner. Inspect
    # structure before parsing, ignoring brackets inside quoted/escaped strings.
    depth, quoted, escaped = 0, False, False
    for value in body:
        if quoted:
            if escaped:
                escaped = False
            elif value == 92:
                escaped = True
            elif value == 34:
                quoted = False
        elif value == 34:
            quoted = True
        elif value in (123, 91):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError("response JSON exceeds 64 container levels")
        elif value in (125, 93):
            depth -= 1
            if depth < 0:
                raise ValueError("response JSON contains unmatched closing brackets")


def _decode(body: bytes) -> dict[str, Any]:
    if type(body) is not bytes or not 1 <= len(body) <= MAX_REPLY_BYTES:
        raise ValueError("response body must be exact bytes of length 1..65536")
    _check_depth(body)
    try:
        result = json.loads(
            body.decode("utf8"),
            object_pairs_hook=_pairs,
            parse_float=_float,
            parse_int=_integer,
            parse_constant=_constant,
        )
    except (UnicodeError, RecursionError, ValueError) as error:
        raise ValueError("invalid bounded UTF-8 response JSON") from error
    if type(result) is not dict:
        raise ValueError("native response must be a JSON object")
    return result


def decode_open_jev_choice(
    request: DecisionRequest, body: bytes, *, question_id: str = "decision"
) -> OpenJevChoiceReply:
    """Validate native observations without renormalizing or authenticating them."""
    _request(request, question_id)
    document = _decode(body)
    answers = document.get("answers")
    if type(answers) is not dict or set(answers) != {question_id}:
        raise ValueError("response answer IDs differ from requested question")
    answer = answers[question_id]
    if type(answer) is not dict or set(answer) != {"type", "choice", "probabilities", "confidence"}:
        raise ValueError("native Choice answer fields differ from inspected schema")
    if answer["type"] != "choice" or type(answer["type"]) is not str:
        raise ValueError("expected a native Choice answer")
    ids = tuple(item.candidate_id for item in request.allowed_candidates)
    values = answer["probabilities"]
    if type(values) is not dict or set(values) != set(ids):
        raise ValueError("native probabilities must cover all and only requested candidates")
    probabilities = tuple((identity, _probability(values[identity])) for identity in ids)
    # Why this: upstream uses supplied-order sum and first maximum, not ID ties.
    mass = sum(value for _, value in probabilities)
    if not isclose(mass, 1.0, rel_tol=NATIVE_MASS_TOLERANCE, abs_tol=NATIVE_MASS_TOLERANCE):
        raise ValueError("native probability mass must be one within 1e-6")
    selected = max(probabilities, key=lambda item: item[1])[0]
    if type(answer["choice"]) is not str or answer["choice"] != selected:
        raise ValueError("native choice must maximize probability, ties in request order")
    confidence = _probability(answer["confidence"])
    count = len(ids)
    expected = (
        1.0
        if count == 1
        else (max(value for _, value in probabilities) / mass - 1 / count) / (1 - 1 / count)
    )
    if not isclose(
        confidence, expected, rel_tol=NATIVE_MASS_TOLERANCE, abs_tol=NATIVE_MASS_TOLERANCE
    ):
        raise ValueError("native confidence disagrees with the inspected margin formula")
    return OpenJevChoiceReply(
        request.request_id, question_id, selected, probabilities, confidence, body
    )


def decode_open_jev_probabilities(
    request: DecisionRequest, body: bytes, *, question_id: str = "decision"
) -> ProbabilityDecisionResult:
    """Project validated Choice observations; retain the original body at the caller.

    Reported tokens are optional provider claims, never inferred from text or
    candidate count. No softmax, normalization, raw-score invention or inference.
    """
    reply = decode_open_jev_choice(request, body, question_id=question_id)
    document = _decode(body)
    usage = document.get("usage", {})
    if type(usage) is not dict:
        raise ValueError("native usage must be an object if reported")
    return ProbabilityDecisionResult(
        reply.request_id,
        reply.choice,
        reply.probabilities,
        reply.confidence,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
    )
