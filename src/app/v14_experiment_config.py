"""Resolve the single fixed v14 experiment preset.

Inputs are a preset ID. Output is the typed v14 manifest. This module does
not open sources, train models, score outcomes, or write artifacts.
"""

from __future__ import annotations

from typing import Literal

from src.app.continual_trigger_replay_schedule import (
    TriggerReplayOpportunityManifest,
    fixed_trigger_replay_manifest,
)


V14PresetId = Literal["fixed-v14"]
FIXED_V14_PRESET_ID: V14PresetId = "fixed-v14"


def resolve_v14_preset(preset_id: str) -> TriggerReplayOpportunityManifest:
    """Reject unknown settings before starting the frozen v14 study."""
    if type(preset_id) is not str or preset_id != FIXED_V14_PRESET_ID:
        raise ValueError(f"unknown v14 preset {preset_id!r}; expected {FIXED_V14_PRESET_ID!r}")
    # Why this: the historical v14 ID binds one matched protocol and its raw
    # byte hashes. New settings require a new versioned study, not an override.
    return fixed_trigger_replay_manifest()
