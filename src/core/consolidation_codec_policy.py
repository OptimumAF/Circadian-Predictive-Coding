"""Original bounded consolidation schema policy; no encoded-input authority.

The original owner supplies the exact lifetime attempt limit and string bound.
Wire metadata cannot enlarge either. No live owner, clock, resource or IO here.
"""

from dataclasses import dataclass

from src.core.consolidation_cursor import require_consolidation_counter

POLICY_FIELDS = frozenset({"consolidation_limit", "max_identifier_bytes"})


@dataclass(frozen=True)
class ConsolidationCodecPolicy:
    consolidation_limit: int
    max_identifier_bytes: int

    def __post_init__(self) -> None:
        if vars(self).keys() != POLICY_FIELDS:
            raise ValueError("consolidation policy differs from complete supported schema")
        require_consolidation_counter(self.consolidation_limit)
        require_consolidation_counter(self.max_identifier_bytes)
        if self.max_identifier_bytes == 0:
            raise ValueError("consolidation string bound must be positive")
