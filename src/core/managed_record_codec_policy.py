"""Independent full component policies for paired record bytes; no authority.

Inputs are original typed policies and aggregate/string limits. Reject policy
disagreement without wire construction, ports or IO. Restore is not authorized.
"""

from dataclasses import dataclass

from src.core.consolidation_codec_policy import ConsolidationCodecPolicy
from src.core.lifecycle_codec_policy import LifecycleCodecPolicy
from src.core.managed_lifecycle_validation import require_state_record


@dataclass(frozen=True)
class ManagedRecordCodecPolicy:
    lifecycle: LifecycleCodecPolicy
    consolidation: ConsolidationCodecPolicy

    def __post_init__(self) -> None:
        require_state_record(self, ManagedRecordCodecPolicy)
        require_state_record(self.lifecycle, LifecycleCodecPolicy)
        require_state_record(self.consolidation, ConsolidationCodecPolicy)
        LifecycleCodecPolicy.__post_init__(self.lifecycle)
        ConsolidationCodecPolicy.__post_init__(self.consolidation)
        if (
            self.consolidation.max_identifier_bytes
            != self.lifecycle.capture_limits.max_identifier_bytes
        ):
            raise ValueError("paired original component string capacities disagree")
