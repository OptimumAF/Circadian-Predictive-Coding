"""Fixed final sensor data and pure signed retention measurements.

Inputs to the summary are all thirteen completed A/B checkpoint score rows;
outputs retain fixed phase endpoints and direction-aware gains, losses and gaps.
The fixture returns owned arrays without model access. Callers assign final-test
roles and admit scoring only after training and probes finish. This module does
not expose final data to learning, select checkpoints, run inference or claim a
causal recovery or comparative learning benefit.
"""

from math import isfinite
from typing import Any

import numpy as np

from src.core.sensor_stream import SensorBatch


def make_sensor_heldout(*, contrast_inversion: bool = False) -> tuple[SensorBatch, SensorBatch]:
    """Build frozen noisy A frames and paired brightness-shifted B frames.

    Why: fresh pixel noise prevents exact training-frame reuse while retaining
    the same left/right label rule. The paired shift isolates the sensor change;
    neither noise nor endpoints depend on observed model results.
    An optional camera negative reverses B's contrast while preserving the
    original semantic labels, giving a distinct label-preserving sensor change.
    """
    if type(contrast_inversion) is not bool:
        raise ValueError("contrast_inversion must be an exact boolean")
    frames = np.full((32, 4, 4), 0.1, dtype=np.float64)
    frames[0::2, :, :2] = 0.6
    frames[1::2, :, 2:] = 0.6
    features = frames.reshape(32, 16) + np.random.default_rng(27031).uniform(
        -0.025, 0.025, (32, 16)
    )
    noisy_frames = features.reshape(32, 4, 4)
    targets = (
        (noisy_frames[:, :, 2:].mean(axis=(1, 2)) > noisy_frames[:, :, :2].mean(axis=(1, 2)))
        .astype(np.float64)
        .reshape(32, 1)
    )
    return (
        SensorBatch(features.copy(), targets.copy()),
        SensorBatch(
            (1.0 - features).copy() if contrast_inversion else (features + 0.25).copy(),
            targets.copy(),
        ),
    )


def describe_sensor_heldout(*, contrast_inversion: bool = False) -> dict[str, Any]:
    """Describe frozen fixture, complete readout costs and comparison signs."""
    if type(contrast_inversion) is not bool:
        raise ValueError("contrast_inversion must be an exact boolean")
    result = {
        "seed": 27031,
        "noise_uniform": (-0.025, 0.025),
        "contexts": ("A", "B"),
        "rows_per_context": 32,
        "feature_width": 16,
        "brightness_shift": 0.25,
        "numeric_bytes": 8704,
        "checkpoint_updates": tuple(range(13)),
        "summary_updates": (0, 4, 8, 12),
        "readout_forward_calls": 26,
        "readout_forward_rows": 832,
        "metric_directions": {"bce": "lower", "accuracy": "higher"},
        "gain_direction": "positive_is_better",
        "loss_and_gap_direction": "positive_is_worse",
    }
    if contrast_inversion:
        del result["brightness_shift"]
        result.update(
            contrast_inversion=True,
            transform="1 - pixels; semantic labels unchanged",
        )
    return result


def _validated_scores(scores: Any) -> dict[str, dict[str, float]]:
    if type(scores) is not dict or set(scores) != {"A", "B"}:
        raise ValueError("each checkpoint must score exactly contexts A and B")
    result = {}
    for context in ("A", "B"):
        metrics = scores[context]
        if type(metrics) is not dict or set(metrics) != {"bce", "accuracy"}:
            raise ValueError("each context must contain exactly bce and accuracy")
        for metric, value in metrics.items():
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{context} {metric} must be a finite nonnegative native number")
            if metric == "accuracy" and value > 1:
                raise ValueError(f"{context} accuracy must be in [0, 1]")
        result[context] = {metric: float(value) for metric, value in metrics.items()}
    return result


def _validated_points(points: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    if type(points) is not tuple or len(points) != 13:
        raise ValueError("retention requires the complete tuple of thirteen checkpoints")
    validated = []
    for updates, point in enumerate(points):
        if type(point) is not dict or set(point) != {"updates", "training_context", "scores"}:
            raise ValueError("each checkpoint requires updates, training_context and scores")
        if type(point["updates"]) is not int or point["updates"] != updates:
            raise ValueError("checkpoints must be ordered updates 0 through 12 without duplicates")
        context = None if updates == 0 else "A" if updates <= 4 or updates > 8 else "B"
        if point["training_context"] != context or type(point["training_context"]) is not type(
            context
        ):
            raise ValueError(
                "checkpoint training contexts must follow None, four A, four B, four A"
            )
        validated.append(
            {
                "updates": updates,
                "training_context": context,
                "scores": _validated_scores(point["scores"]),
            }
        )
    return tuple(validated)


def summarize_sensor_retention(points: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Retain fixed endpoints and signed phase changes without clipping losses.

    All checkpoints are validated, including non-endpoints. Positive gains are
    desirable; positive old-task loss or return gap is worse. A negative loss
    therefore records improvement rather than silently becoming zero.
    """
    checked = _validated_points(points)
    metrics = {}
    for metric, direction in (("bce", 1.0), ("accuracy", -1.0)):
        a0, a4, a8, a12 = (checked[index]["scores"]["A"][metric] for index in (0, 4, 8, 12))
        b4, b8 = (checked[index]["scores"]["B"][metric] for index in (4, 8))
        metrics[metric] = {
            "initial_A_gain": direction * (a0 - a4),
            "new_B_gain": direction * (b4 - b8),
            "old_A_loss": direction * (a8 - a4),
            "return_A_gain": direction * (a8 - a12),
            "return_gap": direction * (a12 - a4),
        }
    return {"endpoints": tuple(checked[index] for index in (0, 4, 8, 12)), "metrics": metrics}
