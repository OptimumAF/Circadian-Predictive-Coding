"""Immutable local decision inputs/results and allowed-candidate validation.

Owns native scalar schema and probability/selection rules. No labels, model,
execution permissions, IO, sampling or external tool calls are defined here.
"""

from dataclasses import dataclass
from math import fsum, isclose, isfinite


def _text(value: str, name: str, limit: int) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty trimmed text")
    try:
        size = len(value.encode("utf8"))
    except UnicodeError as error:
        raise ValueError(f"{name} must be UTF-8 text") from error
    if size > limit or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} exceeds text bounds or contains control characters")


def _finite(value: float, name: str) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite native number")
    try:
        valid = isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite")


def _count(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError(f"{name} must be a nonnegative safe native integer")


@dataclass(frozen=True)
class DecisionCandidate:
    candidate_id: str
    description: str

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id", 64)
        _text(self.description, "description", 1024)


@dataclass(frozen=True)
class DecisionRequest:
    request_id: str
    query: str
    allowed_candidates: tuple[DecisionCandidate, ...]

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id", 64)
        _text(self.query, "query", 2048)
        if type(self.allowed_candidates) is not tuple or not 1 <= len(self.allowed_candidates) <= 8:
            raise ValueError("allowed_candidates must be an immutable tuple of1..8 candidates")
        for candidate in self.allowed_candidates:
            if type(candidate) is not DecisionCandidate:
                raise ValueError("expected exact DecisionCandidate")
            candidate.__post_init__()
        ids = [item.candidate_id for item in self.allowed_candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate allowed candidate ID")


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    raw_score: float
    probability: float

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id", 64)
        _finite(self.raw_score, "raw_score")
        _finite(self.probability, "probability")
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be in[0,1]")


@dataclass(frozen=True)
class DecisionResult:
    request_id: str
    selected_id: str
    scores: tuple[CandidateScore, ...]
    token_visits: int
    candidate_comparisons: int

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id", 64)
        _text(self.selected_id, "selected_id", 64)
        if type(self.scores) is not tuple or not 1 <= len(self.scores) <= 8:
            raise ValueError("scores must be an immutable tuple of1..8 scores")
        for score in self.scores:
            if type(score) is not CandidateScore:
                raise ValueError("expected exact CandidateScore")
            score.__post_init__()
        if len({item.candidate_id for item in self.scores}) != len(self.scores):
            raise ValueError("duplicate scored candidate ID")
        if abs(fsum(item.probability for item in self.scores) - 1.0) > 1e-12:
            raise ValueError("probability mass must sum to1 within1e-12")
        winner = min(self.scores, key=lambda item: (-item.probability, item.candidate_id))
        if self.selected_id != winner.candidate_id:
            raise ValueError("selection must maximize probability with candidate-ID tie breaking")
        _count(self.token_visits, "token_visits")
        _count(self.candidate_comparisons, "candidate_comparisons")
        if self.candidate_comparisons != len(self.scores):
            raise ValueError("candidate_comparisons must count all scored candidates")


def validate_decision_result(request: DecisionRequest, result: DecisionResult) -> None:
    """Revalidate exact records, including bypassed frozen constructors."""
    if type(request) is not DecisionRequest or type(result) is not DecisionResult:
        raise ValueError("expected exact DecisionRequest and DecisionResult")
    request.__post_init__()
    result.__post_init__()
    if result.request_id != request.request_id:
        raise ValueError("result request identity differs")
    if {item.candidate_id for item in result.scores} != {
        item.candidate_id for item in request.allowed_candidates
    }:
        raise ValueError("results must score all and only allowed candidates")


@dataclass(frozen=True)
class ProbabilityDecisionResult:
    """Observed probabilities without inventing unexposed logits or work.

    The provider owns tie policy; any exact maximum is admissible. Native margin
    confidence is retained separately from max probability used by calibration.
    None counters mean unreported, not zero; counts remain provider assertions.
    """

    request_id: str
    selected_id: str
    probabilities: tuple[tuple[str, float], ...]
    reported_confidence: float | None = None
    reported_input_tokens: int | None = None
    reported_output_tokens: int | None = None

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id", 64)
        _text(self.selected_id, "selected_id", 64)
        if type(self.probabilities) is not tuple or not 1 <= len(self.probabilities) <= 8:
            raise ValueError("probabilities must be an immutable tuple of1..8 pairs")
        for item in self.probabilities:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("probability entries must be exact (ID, probability) tuples")
            _text(item[0], "candidate_id", 64)
            _finite(item[1], "probability")
            if not 0 <= item[1] <= 1:
                raise ValueError("probability must be in[0,1]")
        values = dict(self.probabilities)
        if len(values) != len(self.probabilities):
            raise ValueError("duplicate probability candidate ID")
        if not isclose(fsum(values.values()), 1.0, abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError("probability mass must sum to1 within1e-6")
        if self.selected_id not in values or values[self.selected_id] != max(values.values()):
            raise ValueError("selection must be an allowed probability maximum")
        if self.reported_confidence is not None:
            _finite(self.reported_confidence, "reported_confidence")
            if not 0 <= self.reported_confidence <= 1:
                raise ValueError("reported confidence must be in[0,1]")
        for name in ("reported_input_tokens", "reported_output_tokens"):
            value = getattr(self, name)
            if value is not None:
                _count(value, name)


def validate_probability_decision_result(
    request: DecisionRequest, result: ProbabilityDecisionResult
) -> None:
    """Revalidate identity/support without changing probabilities or tie policy."""
    if type(request) is not DecisionRequest or type(result) is not ProbabilityDecisionResult:
        raise ValueError("expected exact DecisionRequest and ProbabilityDecisionResult")
    request.__post_init__()
    result.__post_init__()
    if result.request_id != request.request_id:
        raise ValueError("result request identity differs")
    if {key for key, _ in result.probabilities} != {
        item.candidate_id for item in request.allowed_candidates
    }:
        raise ValueError("results must score all and only allowed candidates")
