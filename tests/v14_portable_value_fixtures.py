"""Exact local v14 payload controls; no historical reproduction or admission.

Inputs are the unchanged fixed manifest and current interpreter environment.
Outputs are immutable in-memory bytes. This helper never writes saved evidence.
"""

from functools import lru_cache
import sys

import numpy as np
import pytest

from scripts.run_continual_trigger_replay_outcomes import serialize_outcome_comparison
from scripts.run_continual_trigger_replay_training import serialize_training_study
from src.app.continual_trigger_replay_outcomes import score_trigger_replay_training_study
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.app.continual_trigger_replay_training_study import run_trigger_replay_training_study
from src.app.v14_measured_observations import serialize_wake_diagnostic_study

recorded_v14_environment = pytest.mark.skipif(
    sys.platform != "win32" or sys.version_info[:3] != (3, 14, 7) or np.__version__ != "2.4.6",
    reason="historical v14 byte reproduction requires Windows CPython 3.14.7 / NumPy 2.4.6",
)


@lru_cache(maxsize=1)
def fixed_v14_payloads() -> tuple[bytes, bytes, bytes]:
    # Why this: resumed/derived bytes must match a fresh run in the same environment.
    study = run_trigger_replay_training_study(
        fixed_trigger_replay_manifest(), capture_wake_diagnostics=True
    )
    training = serialize_training_study(study).encode("utf-8")
    diagnostics = serialize_wake_diagnostic_study(study)
    outcomes = serialize_outcome_comparison(score_trigger_replay_training_study(study)).encode(
        "utf-8"
    )
    return training, outcomes, diagnostics
