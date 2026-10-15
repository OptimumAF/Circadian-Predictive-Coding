"""Pure complete-seed summaries for the frozen allocated-time PC comparison.

Inputs include every planned case, including failures. Outputs retain null
observations and paired differences. No fitting, timing, model execution or IO.
"""

from dataclasses import asdict
from math import isfinite
from typing import Any

from src.core.seed_statistics import IntervalSpec, SeedObservation, summarize_seeds

PLANNED_SEEDS = (23, 41, 59)
METHODS = ("fixed", "adaptive")
CASE_ORDER = (
    (23, "fixed"),
    (23, "adaptive"),
    (41, "adaptive"),
    (41, "fixed"),
    (59, "fixed"),
    (59, "adaptive"),
)
WINDOW_SECONDS = 0.01
MAX_UPDATES = 80
INTERVALS = IntervalSpec(3, 0.95, 4.303)
WORK_METRICS = (
    "latent_steps",
    "inference_energy_evaluations",
    "parameter_energy_evaluations",
    "gradient_norm_evaluations",
    "energy_comparisons",
    "fallback_updates",
    "training_row_exposures",
)
METRICS = (
    "initial_bce",
    "initial_accuracy",
    "final_bce",
    "final_accuracy",
    "committed_updates",
    "training_window_seconds",
    "train_call_seconds",
    "overrun_seconds",
    *WORK_METRICS,
)


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite nonnegative native number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative native number")
    return number


def _timing_controls(
    window_seconds: float, max_iteration_seconds: float | None
) -> tuple[float, float | None]:
    window = _number(window_seconds, "window_seconds")
    if not 0 < window <= 60:
        raise ValueError("window_seconds must be positive and at most 60 seconds")
    limit = None
    if max_iteration_seconds is not None:
        limit = _number(max_iteration_seconds, "max_iteration_seconds")
        if not 0 < limit <= 0.1 * window:
            raise ValueError(
                "max_iteration_seconds must be positive and at most 10% of window_seconds"
            )
    return window, limit


def _value(result: dict[str, Any], metric: str) -> float:
    try:
        if metric.startswith(("initial_", "final_")):
            stage, score = metric.split("_", 1)
            value = _number(result[f"{stage}_scores"][score], metric)
            if score == "accuracy" and value > 1:
                raise ValueError("accuracy must be <=1")
            return value
        if metric in WORK_METRICS:
            return _number(result["work"][metric], metric)
        return _number(result[metric], metric)
    except (KeyError, TypeError) as error:
        raise ValueError(f"complete case is missing {metric}") from error


def _index(
    rows: tuple[dict[str, Any], ...], window_seconds: float, iteration_limit: float | None
) -> dict[tuple[int, str], dict[str, Any]]:
    if type(rows) is not tuple or len(rows) != len(CASE_ORDER):
        raise ValueError("comparison requires all six cases in declared order")
    indexed = {}
    for row, key in zip(rows, CASE_ORDER, strict=True):
        if (
            type(row) is not dict
            or type(row.get("seed")) is not int
            or type(row.get("method")) is not str
            or (row["seed"], row["method"]) != key
            or row.get("status") not in ("complete", "failed", "not_run")
        ):
            raise ValueError("cases differ from the declared unique ordered keys/statuses")
        if row["status"] != "complete":
            if type(row.get("reason")) is not str or not row["reason"].strip():
                raise ValueError("incomplete cases require a reason")
        else:
            result = row.get("result")
            if type(result) is not dict or result.get("status") != "complete":
                raise ValueError("complete case requires a complete result")
            updates = result.get("committed_updates")
            if type(updates) is not int or not 1 <= updates <= MAX_UPDATES:
                raise ValueError("complete case requires 1..80 committed updates")
            if result.get("window_budget_seconds") != window_seconds:
                raise ValueError("complete case differs from fixed window allocation")
            window = _value(result, "training_window_seconds")
            if not window_seconds <= window <= 1.1 * window_seconds:
                raise ValueError("complete case exceeds the frozen observed window bounds")
            if iteration_limit is not None:
                declared = _number(result.get("max_iteration_seconds"), "max_iteration_seconds")
                if declared != iteration_limit:
                    raise ValueError("complete case differs from the declared iteration ceiling")
                observed = _number(
                    result.get("observed_max_iteration_seconds"), "observed_max_iteration_seconds"
                )
                if observed > iteration_limit:
                    raise ValueError("complete case exceeds the complete-iteration ceiling")
            for metric in METRICS:
                _value(result, metric)
        indexed[key] = row
    return indexed


def summarize_inference_comparison(
    rows: tuple[dict[str, Any], ...],
    *,
    window_seconds: float = WINDOW_SECONDS,
    max_iteration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Keep every seed and qualify the selected window/optional iteration limit.

    Defaults retain the original strict allocation. Incomplete vectors have no
    complete mean/CI; opt-in complete rows require a matching known ceiling.
    """
    window, iteration_limit = _timing_controls(window_seconds, max_iteration_seconds)
    indexed = _index(rows, window, iteration_limit)
    summaries = []
    for metric in METRICS:
        vectors = {}
        for method in METHODS:
            vectors[method] = tuple(
                SeedObservation(seed, _value(indexed[seed, method]["result"], metric))
                if indexed[seed, method]["status"] == "complete"
                else SeedObservation(seed, None, indexed[seed, method]["reason"])
                for seed in PLANNED_SEEDS
            )
        vectors["adaptive_minus_fixed"] = tuple(
            SeedObservation(left.seed, left.value - right.value)
            if left.value is not None and right.value is not None
            else SeedObservation(left.seed, None, left.reason or right.reason)
            for left, right in zip(vectors["adaptive"], vectors["fixed"], strict=True)
        )
        for name, vector in vectors.items():
            summaries.append(
                {
                    "metric": metric,
                    "comparison": name,
                    "difference_direction": "adaptive_minus_fixed"
                    if name == "adaptive_minus_fixed"
                    else None,
                    "summary": asdict(summarize_seeds(vector, PLANNED_SEEDS, INTERVALS)),
                }
            )
    return summaries
