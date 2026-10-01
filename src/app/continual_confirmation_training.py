"""Hold every frozen family checkpoint behind a joint arrived-data barrier.

Inputs are the exact P6.7a confirmation manifest. Outputs are unscored live
checkpoints/roles and finite family facts. This is the training composition
component; independent saved JSON/resource/artifact gates belong to P6.7b2.
It performs no evaluation, source selection, publication or artifact IO.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import Any, Callable, Protocol

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_confirmation_manifest import (
    ConfirmationFamily,
    ConfirmationManifest,
    validate_confirmation_manifest,
)
from src.app.continual_confirmation_periodic import (
    CombinedTrajectory,
    ParentTrajectory,
    ScheduleTrajectory,
)
from src.app.continual_confirmation_simple import (
    GatingTrajectory,
    ReplayTrajectory,
    SleepTrajectory,
)
from src.app.continual_confirmation_state import (
    FamilySeedFacts,
    GuardWitness,
    HeldSeed,
    CheckpointFacts,
    RoleFacts,
    capture_models,
    capture_roles,
    require_checkpoints,
    require_held_seed,
)
from src.app.continual_replay_factor_pilot import Model
from src.infra.continual_roles import PhaseDecisionRoles


PROTOCOL_ID = "continual_mechanism_confirmation_train_only_v1"


class _Trajectory(Protocol):
    manifest: Any
    arms: tuple[str, ...]
    models: dict[str, Model]
    guards: list[GuardWitness]

    def train_phase(self, roles: PhaseDecisionRoles) -> None: ...

    def finish(self, seed: int, a: PhaseDecisionRoles, b: PhaseDecisionRoles) -> dict[str, Any]: ...


_TRAJECTORIES: dict[str, Callable[[int], _Trajectory]] = {
    "gating": GatingTrajectory,
    "replay": ReplayTrajectory,
    "sleep": SleepTrajectory,
    "schedule": ScheduleTrajectory,
    "combined": CombinedTrajectory,
    "parent": ParentTrajectory,
}


@dataclass(frozen=True)
class ConfirmationTrainingFacts:
    manifest: ConfirmationManifest
    seed_results: tuple[FamilySeedFacts, ...]
    protocol_id: str = PROTOCOL_ID
    all_a_completed_before_first_b: bool = True
    outer_selection_scored: bool = False
    final_released: bool = False


@dataclass
class TrainedConfirmation:
    facts: ConfirmationTrainingFacts
    held: tuple[HeldSeed, ...]


@dataclass
class _AfterA:
    family: str
    seed: int
    trajectory: _Trajectory
    roles: PhaseDecisionRoles
    role_facts: RoleFacts
    initial: dict[str, CheckpointFacts]
    checkpoint: dict[str, CheckpointFacts]
    models: dict[str, Model]


def _train_a(family: ConfirmationFamily, seed: int) -> _AfterA:
    # The original source is obtained from the fixed family configuration;
    # this private seam also permits existing development-source fixtures.
    if type(seed) is not int or seed not in family.seeds:
        raise ValueError("confirmation trajectory seed differs from its declared scope")
    trajectory = _TRAJECTORIES[family.name](seed)
    resolved = json.dumps(
        asdict(trajectory.manifest), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if trajectory.arms != family.arms or resolved != family.development_manifest_json:
        raise ValueError("confirmation trajectory settings/arm inventory differs")
    roles = arrived._build_phase_a_roles(trajectory.manifest.source, seed)
    role_facts = capture_roles(roles, "a", seed)
    initial = capture_models(trajectory.models, family.arms)
    trajectory.train_phase(roles)
    checkpoint = capture_models(trajectory.models, family.arms)
    models = deepcopy(trajectory.models)
    require_checkpoints(models, checkpoint, f"{family.name}/{seed}/a copy")
    return _AfterA(family.name, seed, trajectory, roles, role_facts, initial, checkpoint, models)


def _require_a(item: _AfterA) -> None:
    if capture_roles(item.roles, "a", item.seed) != item.role_facts:
        raise ValueError("confirmation A role changed before B arrival")
    require_checkpoints(item.models, item.checkpoint, f"{item.family}/{item.seed}/a held")
    require_checkpoints(
        item.trajectory.models, item.checkpoint, f"{item.family}/{item.seed}/a live"
    )


def _train_b(item: _AfterA) -> HeldSeed:
    trajectory = item.trajectory
    roles_b = arrived._build_phase_b_roles(trajectory.manifest.source, item.seed)
    role_b = capture_roles(roles_b, "b", item.seed)
    trajectory.train_phase(roles_b)
    legacy = trajectory.finish(item.seed, item.roles, roles_b)
    facts = FamilySeedFacts(
        item.family,
        item.seed,
        (item.role_facts, role_b),
        item.initial,
        item.checkpoint,
        capture_models(trajectory.models, trajectory.arms),
        legacy,
        tuple(trajectory.guards),
    )
    held = HeldSeed(facts, item.roles, roles_b, item.models, trajectory.models)
    require_held_seed(held)
    return held


def _train_families(families: tuple[ConfirmationFamily, ...]) -> tuple[HeldSeed, ...]:
    after_a = tuple(_train_a(family, seed) for family in families for seed in family.seeds)
    # Why this: a late A checkpoint failure must prevent the first B source,
    # including when the failed cell is in the last family or seed.
    for item in after_a:
        _require_a(item)
    held = tuple(_train_b(item) for item in after_a)
    for held_item in held:
        require_held_seed(held_item)
    return held


def train_confirmation(manifest: ConfirmationManifest) -> TrainedConfirmation:
    """Validate the entire reserved scope before constructing any source."""
    validate_confirmation_manifest(manifest)
    held = _train_families(manifest.families)
    return TrainedConfirmation(
        ConfirmationTrainingFacts(manifest, tuple(item.facts for item in held)), held
    )
