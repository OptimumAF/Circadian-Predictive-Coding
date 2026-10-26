"""Immutable scalar observations from a committed CLM TRAIN update.

No tensors, optimizer state, callbacks, IO or model authority. The observation
describes the existing clipping operation; it is not a loss or task metric.
"""

from dataclasses import dataclass
from math import isfinite

from src.core.clm_head import _exact
from src.core.learner_ports import TrainingDiagnostic

GRADIENT_DEFINITION = "clm_optimizer_pre_clip_global_l2_v1"


@dataclass(frozen=True)
class CLMGradientObservation:
    committed_update: int
    training_rows: int
    gradient_tensor_count: int
    pre_clip_l2_norm: float
    clip_limit: float = 1.0
    definition: str = GRADIENT_DEFINITION

    def __post_init__(self):
        for value in (self.committed_update, self.training_rows, self.gradient_tensor_count):
            if type(value) is not int or not 1 <= value <= 2**31 - 1:
                raise ValueError("gradient observation requires positive native counts")
        if (
            type(self.pre_clip_l2_norm) is not float
            or not isfinite(self.pre_clip_l2_norm)
            or self.pre_clip_l2_norm < 0
        ):
            raise ValueError("gradient observation requires a finite nonnegative native float")
        if type(self.clip_limit) is not float or self.clip_limit != 1.0:
            raise ValueError("gradient observation describes the original unit clipping limit")
        if type(self.definition) is not str or self.definition != GRADIENT_DEFINITION:
            raise ValueError("gradient observation definition changed")


@dataclass(frozen=True)
class CLMObservedUpdate:
    diagnostic: TrainingDiagnostic
    observation: CLMGradientObservation

    def __post_init__(self):
        _exact(self.diagnostic, TrainingDiagnostic)
        _exact(self.observation, CLMGradientObservation)
