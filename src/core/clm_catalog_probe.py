"""Explicit catalog-objective probe inputs, without TRAIN conversion.

Features are opaque. These records bind catalog IDs and a declared encoder;
they neither grant data access nor make a disposable probe deployable.
"""

from dataclasses import dataclass
from typing import Any

from src.core.clm_catalog import _candidate_ids
from src.core.clm_head import FrozenEncoderIdentity, _exact


@dataclass(frozen=True)
class FrozenProbeCatalogBatch:
    identity: FrozenEncoderIdentity
    states: Any
    actions: Any
    candidate_ids: tuple[str, ...]
    role: str = "probe_train"

    def __post_init__(self):
        _exact(self.identity, FrozenEncoderIdentity)
        _candidate_ids(self.candidate_ids)
        if type(self.role) is not str or self.role != "probe_train":
            raise ValueError("catalog probe updates require explicit probe_train features")


@dataclass(frozen=True)
class CatalogProbeRehearsal:
    """Declared arrived TRAIN history and six bounded mixed probe packets.

    Only the disposable probe may consume these packets. The app checks tensor
    equality, source checkpoint, consent and paid work; this record grants none.
    """

    phase: str
    training: tuple[Any, Any]
    packets: tuple[tuple[Any, Any], ...]
    permissions: Any

    def __post_init__(self):
        from src.core.clm_catalog import CLMCatalogTargets, FrozenCatalogTrainingBatch
        from src.core.experience import ExperiencePermissions
        from src.core.temporal_clm_probe import PROBE_UPDATES
        from src.core.temporal_decision_sequence import PHASES, temporal_examples

        if type(self.phase) is not str or self.phase not in PHASES:
            raise ValueError("rehearsal requires an arrived non-baseline TRAIN phase")
        if type(self.training) is not tuple or len(self.training) != 2:
            raise ValueError("rehearsal requires an original TRAIN feature/target tuple")
        batch, targets = self.training
        _exact(batch, FrozenCatalogTrainingBatch)
        _exact(targets, CLMCatalogTargets)
        _exact(self.permissions, ExperiencePermissions)
        if self.permissions != ExperiencePermissions(training=True, replay=True):
            raise ValueError("TRAIN rehearsal requires explicit training and replay consent")
        examples = temporal_examples(self.phase, "train")
        ids = tuple(row.candidate_id for row in examples[0].request.allowed_candidates)
        if (
            batch.candidate_ids != ids
            or targets.candidate_ids != ids
            or targets.candidate_indices != tuple(ids.index(row.target_id) for row in examples)
        ):
            raise ValueError("rehearsal TRAIN labels differ from canonical arrived phase")
        if type(self.packets) is not tuple or len(self.packets) != PROBE_UPDATES:
            raise ValueError("rehearsal requires exactly six fixed mixed packets")
        for pair in self.packets:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("mixed rehearsal packets require feature/target tuples")
            _exact(pair[0], FrozenProbeCatalogBatch)
            _exact(pair[1], CLMCatalogTargets)


@dataclass(frozen=True)
class CatalogProbeMarkerControl:
    """Six extra-marker TRAIN packets; no replay permission or history."""

    phase: str
    packets: tuple[tuple[Any, Any], ...]

    def __post_init__(self):
        from src.core.clm_catalog import CLMCatalogTargets
        from src.core.temporal_clm_probe import PROBE_UPDATES
        from src.core.temporal_decision_sequence import PHASES

        if type(self.phase) is not str or self.phase not in PHASES:
            raise ValueError("marker control requires a non-baseline checkpoint")
        if type(self.packets) is not tuple or len(self.packets) != PROBE_UPDATES:
            raise ValueError("marker control requires six fixed extra-TRAIN packets")
        for pair in self.packets:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("marker control requires feature/target tuples")
            _exact(pair[0], FrozenProbeCatalogBatch)
            _exact(pair[1], CLMCatalogTargets)
