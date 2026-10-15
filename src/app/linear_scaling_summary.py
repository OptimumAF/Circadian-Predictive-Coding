"""Summarize every declared scaling seed; missing runs remain explicit nulls."""

from dataclasses import asdict
from itertools import product
from math import isfinite
from typing import Any

from src.core.seed_statistics import IntervalSpec, SeedObservation, summarize_seeds

SEEDS = (23, 41, 59)
WIDTHS = (2, 8, 32)
DEPTHS = (1, 2, 4)
SCHEMES = ("legacy_gaussian", "fan_in", "mean_field")
UPDATES = 6
KEYS = tuple(product(SEEDS, WIDTHS, DEPTHS, SCHEMES))
# Same predeclared marginal policy as the attribution pilot; no family coverage.
INTERVALS = IntervalSpec(3, 0.95, 4.303)


def summarize_scaling(rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Require all 81 unique records and the full checkpoint sequence."""
    if type(rows) is not tuple or len(rows) != len(KEYS):
        raise ValueError("scaling summary requires all 81 declared records")
    indexed = {}
    for row in rows:
        if (
            type(row) is not dict
            or any(type(row.get(field)) is not int for field in ("seed", "width", "hidden_layers"))
            or type(row.get("scheme")) is not str
        ):
            raise ValueError("scaling record keys must have native declared types")
        key = (row["seed"], row["width"], row["hidden_layers"], row["scheme"])
        if key not in KEYS or key in indexed:
            raise ValueError("scaling records must have unique declared keys")
        status = row.get("status")
        if type(status) is not str or status not in ("complete", "failed", "not_run"):
            raise ValueError("scaling record status is invalid")
        if status == "complete":
            points, scores = row.get("checkpoints"), row.get("heldout_half_mse")
            if (
                type(row.get("committed_updates")) is not int
                or row["committed_updates"] != UPDATES
                or type(points) is not list
                or len(points) != UPDATES + 1
                or any(type(p) is not dict or type(p.get("updates")) is not int for p in points)
                or [p["updates"] for p in points] != list(range(UPDATES + 1))
                or type(scores) is not list
                or len(scores) != UPDATES + 1
                or any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in scores)
            ):
                raise ValueError("complete scaling record requires all finite checkpoint scores")
        elif type(row.get("reason")) is not str or not row["reason"].strip():
            raise ValueError("incomplete scaling record requires a reason")
        indexed[key] = row
    summaries = []
    for width, depth, scheme in product(WIDTHS, DEPTHS, SCHEMES):
        result: dict[str, Any] = {"width": width, "hidden_layers": depth, "scheme": scheme}
        for label, point in (("initial", 0), ("final", UPDATES)):
            vector = tuple(
                SeedObservation(seed, float(row["heldout_half_mse"][point]))
                if (row := indexed[seed, width, depth, scheme])["status"] == "complete"
                else SeedObservation(seed, None, row["reason"])
                for seed in SEEDS
            )
            result[label] = asdict(summarize_seeds(vector, SEEDS, INTERVALS))
        summaries.append(result)
    return summaries
