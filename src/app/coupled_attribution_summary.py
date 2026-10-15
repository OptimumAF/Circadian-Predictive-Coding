"""Complete ordered seed summaries for the fixed exploratory attribution pilot.

Inputs are all declared run records, including failures. Outputs are conditional
fixed-data summaries and paired differences. No fitting, seed selection or IO.
"""

from dataclasses import asdict
from itertools import product
from math import isfinite
from typing import Any

from src.core.seed_statistics import IntervalSpec, SeedObservation, summarize_seeds

PLANNED_SEEDS = (23, 41, 59)
STREAMS = ("stationary", "recurring")
METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
CHECKPOINTS = (0, 6, 12, 18)
CONTEXTS = ("A", "B")
PAIRS = (("pc", "bp"), ("neutral_cpc", "pc"), ("gated_cpc", "neutral_cpc"))
# NIST t table: df=2, 0.975 column. Marginal, exploratory, conditional on fixed
# data; n=3 cannot establish the normal-seed-effect assumption or family coverage.
INTERVALS = IntervalSpec(3, 0.95, 4.303)


def _index_rows(rows: tuple[dict[str, Any], ...]) -> dict[tuple[int, str, str], dict[str, Any]]:
    expected = set(product(PLANNED_SEEDS, STREAMS, METHODS))
    if type(rows) is not tuple or len(rows) != len(expected):
        raise ValueError("attribution summaries require all 24 declared run records")
    indexed = {}
    for row in rows:
        if type(row) is not dict or type(row.get("seed")) is not int:
            raise ValueError("run record must contain an exact integer seed")
        if type(row.get("stream")) is not str or type(row.get("method")) is not str:
            raise ValueError("run stream and method must be native strings")
        key = (row["seed"], row["stream"], row["method"])
        if key not in expected or key in indexed:
            raise ValueError("run records must match unique declared seed/stream/method keys")
        if row.get("status") not in ("complete", "failed", "not_run"):
            raise ValueError("run record must declare complete, failed or not_run")
        if row["status"] != "complete" and not (
            type(row.get("reason")) is str and row["reason"].strip()
        ):
            raise ValueError("incomplete run records require a reason")
        if row["status"] == "complete":
            result = row.get("stream_result")
            points = result.get("checkpoints") if type(result) is dict else None
            if (
                type(points) is not list
                or len(points) != len(CHECKPOINTS)
                or any(type(p) is not dict or type(p.get("updates")) is not int for p in points)
                or [p["updates"] for p in points] != list(CHECKPOINTS)
            ):
                raise ValueError("complete run must contain the exact fixed checkpoint order")
        indexed[key] = row
    return indexed


def _observation(row: dict[str, Any], point: int, context: str, metric: str) -> SeedObservation:
    seed = row["seed"]
    if row["status"] != "complete":
        return SeedObservation(seed, None, row["reason"])
    try:
        value = row["stream_result"]["checkpoints"][point]["scores"][context][metric]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("complete run has missing score fields") from error
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("complete run scores must be finite native numbers")
    if value < 0 or (metric == "accuracy" and value > 1):
        raise ValueError("complete run score is outside its metric domain")
    return SeedObservation(seed, float(value))


def _difference(left: SeedObservation, right: SeedObservation) -> SeedObservation:
    if left.value is None or right.value is None:
        return SeedObservation(left.seed, None, left.reason or right.reason)
    return SeedObservation(left.seed, left.value - right.value)


def summarize_attribution(rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Refuse lost/duplicate runs; never remove failed seeds from intervals."""
    indexed = _index_rows(rows)
    summaries = []
    for stream, (point, updates), context, metric in product(
        STREAMS, enumerate(CHECKPOINTS), CONTEXTS, ("bce", "accuracy")
    ):
        vectors = {
            method: tuple(
                _observation(indexed[seed, stream, method], point, context, metric)
                for seed in PLANNED_SEEDS
            )
            for method in METHODS
        }
        comparisons = {
            f"{left}_minus_{right}": tuple(map(_difference, vectors[left], vectors[right]))
            for left, right in PAIRS
        }
        for name, vector in {**vectors, **comparisons}.items():
            summaries.append(
                {
                    "stream": stream,
                    "updates": updates,
                    "context": context,
                    "metric": metric,
                    "comparison": name,
                    "difference_direction": "left_minus_right" if "_minus_" in name else None,
                    "summary": asdict(summarize_seeds(vector, PLANNED_SEEDS, INTERVALS)),
                }
            )
    return summaries
