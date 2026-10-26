"""Stateless lexical reference for typed candidate selection.

Uses Unicode word-token sets, Jaccard similarity and fixed-temperature softmax.
No fitted statistics, labels, modern-model substitution, IO or external actions.
"""

from math import exp, fsum
import re

from src.core.typed_decision import (
    CandidateScore,
    DecisionRequest,
    DecisionResult,
    validate_decision_result,
)


def retrieve_decision(request: DecisionRequest) -> DecisionResult:
    if type(request) is not DecisionRequest:
        raise ValueError("expected exact DecisionRequest")
    request.__post_init__()
    query_words = re.findall(r"\w+", request.query.casefold())
    query = set(query_words)
    token_visits = len(query_words)
    similarities = []
    # Why this: canonical arithmetic and tie ordering do not depend on input order.
    for candidate in sorted(request.allowed_candidates, key=lambda item: item.candidate_id):
        words = re.findall(r"\w+", candidate.description.casefold())
        token_visits += len(words)
        terms = set(words)
        union = query | terms
        value = len(query & terms) / len(union) if union else 0.0
        similarities.append((candidate.candidate_id, value))
    weights = [exp(value) for _, value in similarities]
    total = fsum(weights)
    scores = tuple(
        CandidateScore(identity, value, weight / total)
        for (identity, value), weight in zip(similarities, weights, strict=True)
    )
    winner = min(scores, key=lambda item: (-item.probability, item.candidate_id))
    result = DecisionResult(
        request.request_id, winner.candidate_id, scores, token_visits, len(scores)
    )
    validate_decision_result(request, result)
    return result
