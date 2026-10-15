"""Complete prespecified seed groups for the fixed nonlinear scaling pilot.

Summaries retain every failure and use complete n=3 groups/pairs only. They are
exploratory fixed-data intervals, not population coverage or model selection.
"""

from dataclasses import asdict
from itertools import product
from math import isfinite
from typing import Any

from src.core.seed_statistics import IntervalSpec, SeedObservation, summarize_seeds

SEEDS = (23, 41, 59)
WIDTHS = (2, 8)
DEPTHS = (1, 3)
SCHEMES = ("fan_in", "mean_field")
LEGACY_SCHEMES = ("legacy_gaussian",)
ALL_SCHEMES = ("legacy_gaussian", "fan_in", "mean_field")
METHODS = ("bp", "pc", "neutral_cpc", "active_cpc")
CHECKPOINTS = (0, 1, 2, 3)
PAIRS = (("pc", "bp"), ("neutral_cpc", "pc"), ("active_cpc", "neutral_cpc"))
INTERVALS = IntervalSpec(3, 0.95, 4.303)


def _validate_schemes(schemes: tuple[str, ...]) -> tuple[str, ...]:
    if (
        type(schemes) is not tuple
        or any(type(scheme) is not str for scheme in schemes)
        or schemes not in (SCHEMES, LEGACY_SCHEMES, ALL_SCHEMES)
    ):
        raise ValueError(
            "schemes must name the original pair, legacy-only, or ordered all-three group"
        )
    return schemes


def planned_keys(
    *, schemes: tuple[str, ...] = SCHEMES
) -> tuple[tuple[int, int, str, int, str], ...]:
    return tuple(product(WIDTHS, DEPTHS, _validate_schemes(schemes), SEEDS, METHODS))


def _index(
    rows: tuple[dict[str, Any], ...],
    *,
    schemes: tuple[str, ...] = SCHEMES,
) -> dict[tuple[int, int, str, int, str], dict[str, Any]]:
    expected = set(planned_keys(schemes=schemes))
    if type(rows) is not tuple or len(rows) != len(expected):
        raise ValueError(f"summary requires all {len(expected)} prespecified run records")
    indexed = {}
    for row in rows:
        if type(row) is not dict or any(
            type(row.get(name)) is not int for name in ("width", "depth", "seed")
        ):
            raise ValueError("run width/depth/seed must be native integers")
        if any(type(row.get(name)) is not str for name in ("scheme", "method")):
            raise ValueError("run scheme/method must be native strings")
        key = (row["width"], row["depth"], row["scheme"], row["seed"], row["method"])
        if key not in expected or key in indexed:
            raise ValueError("run keys must match unique prespecified cases")
        if row.get("status") not in ("complete", "failed", "not_run"):
            raise ValueError("unknown run status")
        result = row.get("case_result")
        if row["status"] == "complete":
            stream = result.get("stream_result") if type(result) is dict else None
            points = stream.get("checkpoints") if type(stream) is dict else None
            if (
                type(points) is not list
                or any(type(p) is not dict or type(p.get("updates")) is not int for p in points)
                or [p["updates"] for p in points] != list(CHECKPOINTS)
            ):
                raise ValueError("complete run must contain exactly checkpoints0,1,2,3")
        else:
            if type(row.get("reason")) is not str or not row["reason"].strip():
                raise ValueError("incomplete run must retain a reason")
            if type(result) is dict and result.get("stream_result") is not None:
                raise ValueError("failed run must not expose partial final scores")
        indexed[key] = row
    return indexed


def _observation(row: dict[str, Any], point: int, metric: str) -> SeedObservation:
    if row["status"] != "complete":
        return SeedObservation(row["seed"], None, row["reason"])
    try:
        value = row["case_result"]["stream_result"]["checkpoints"][point]["scores"]["stationary"][
            metric
        ]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("complete run has missing metric fields") from error
    if (
        type(value) not in (int, float)
        or not isfinite(value)
        or value < 0
        or (metric == "accuracy" and value > 1)
    ):
        raise ValueError("run metric is outside its finite domain")
    return SeedObservation(row["seed"], float(value))


def summarize_nonlinear_scaling(
    rows: tuple[dict[str, Any], ...], *, schemes: tuple[str, ...] = SCHEMES
) -> list[dict[str, Any]]:
    schemes = _validate_schemes(schemes)
    indexed = _index(rows, schemes=schemes)
    summaries = []
    for width, depth, scheme, (point, updates), metric in product(
        WIDTHS, DEPTHS, schemes, enumerate(CHECKPOINTS), ("bce", "accuracy")
    ):
        vectors = {
            method: tuple(
                _observation(indexed[width, depth, scheme, seed, method], point, metric)
                for seed in SEEDS
            )
            for method in METHODS
        }
        for left, right in PAIRS:
            vectors[f"{left}_minus_{right}"] = tuple(
                SeedObservation(a.seed, None, a.reason or b.reason)
                if a.value is None or b.value is None
                else SeedObservation(a.seed, a.value - b.value)
                for a, b in zip(vectors[left], vectors[right])
            )
        for comparison, vector in vectors.items():
            summaries.append(
                {
                    "width": width,
                    "depth": depth,
                    "scheme": scheme,
                    "updates": updates,
                    "metric": metric,
                    "comparison": comparison,
                    "difference_direction": "left_minus_right" if "_minus_" in comparison else None,
                    "summary": asdict(summarize_seeds(vector, SEEDS, INTERVALS)),
                }
            )
    return summaries
