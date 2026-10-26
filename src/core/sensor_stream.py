"""Deterministic public pixels and labels for recurring or stationary streams.

``make_sensor_stream()`` returns twelve owned batches of four 4x4 flattened
float64 frames and binary column targets. Four baseline batches precede four
brightness-shifted batches and four exact baseline returns. An optional camera
negative inverts only the middle phase. The stationary control uses twelve
exact baseline batches. Pixels contain no
context IDs. This module does not release labels, run a learner, calibrate a
detector, select a probe, or define comparative scientific acceptance criteria.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SensorBatch:
    """A fixture's public flattened frames and their separate binary targets."""

    features: NDArray[np.float64]
    targets: NDArray[np.float64]


def make_sensor_stream(
    *, stationary: bool = False, contrast_inversion: bool = False
) -> tuple[SensorBatch, ...]:
    """Build a fixed 48-frame, balanced, label-preserving A-to-B-to-A stream.

    A class-zero frame has a brighter left half; class one reverses the halves.
    Why: adding the same brightness to all pixels changes the sensor condition
    while preserving the underlying left-versus-right classification rule.
    With stationary=True, all twelve batches use the exact baseline pixels.
    With contrast_inversion=True, B uses a camera negative (1 - pixels), keeping
    object-identity labels. The contrast rule therefore reverses in that phase.
    Every returned array is independent, including repeated contexts and calls.
    """
    if type(stationary) is not bool:
        raise ValueError("stationary must be an exact boolean")
    if type(contrast_inversion) is not bool:
        raise ValueError("contrast_inversion must be an exact boolean")
    if stationary and contrast_inversion:
        raise ValueError("stationary and contrast_inversion cannot both be enabled")
    targets = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float64)
    frames = np.full((4, 4, 4), 0.1, dtype=np.float64)
    frames[0::2, :, :2] = 0.6
    frames[1::2, :, 2:] = 0.6
    baseline = frames.reshape(4, 16)
    if contrast_inversion:
        return tuple(
            SensorBatch(
                features=(1.0 - baseline).copy() if 4 <= tick < 8 else baseline.copy(),
                targets=targets.copy(),
            )
            for tick in range(12)
        )
    # No clipping: even the shifted high level (0.85) remains a valid pixel.
    brightness_offsets = (0.0,) * 12 if stationary else (0.0,) * 4 + (0.25,) * 4 + (0.0,) * 4
    return tuple(
        SensorBatch(features=(baseline + offset).copy(), targets=targets.copy())
        for offset in brightness_offsets
    )
