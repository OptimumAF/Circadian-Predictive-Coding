"""Causal state42 plus previous actually executed robot command history.

Recipes select executed or fixed-zero history: the original47D state prefix
or an explicit fixed-physical taskspace12 view. Only the original raw prefix can
be quaternion-canonical. State advances only from exact native
float32 actions; no guessed rounding, teacher target, clock, learner or IO occurs.
The closing count is a consecutive-command proxy, not grasp/contact evidence.
"""

from dataclasses import dataclass
import struct

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
    _copy_spec,
    _vector,
    copy_continuous_episode,
)
from src.core.pick_cube_corrections import _UNITS
from src.core.pick_cube_state_quaternions import (
    state_quaternion_observation,
    state_quaternion_projection,
)
from src.core.pick_cube_train_context import TRAIN_SEEDS, pick_cube_training_context
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from src.core.pick_cube_mass_context import (
    PickCubeMassContext,
    require_mass_binding,
)
from src.core.pick_cube_readiness_corrections import _observed_grasp_flags
from src.core.pick_cube_taskspace_history import (
    PickCubeTaskSpaceHistoryRecipe,
    taskspace_observation,
    validate_taskspace_history_recipe,
)


DEFINITION = "pick_cube_state42_previous_executed_action4_closing_count_v1"


@dataclass(frozen=True)
class PickCubeActionHistoryRecipe:
    mode: str = "executed"
    canonical_quaternions: bool = False
    definition: str = DEFINITION
    source_dim: int = 42
    feature_dim: int = 47
    closing_limit: int = 6

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in ("executed", "zero"):
            raise ValueError("action history mode must be executed or zero")
        if type(self.canonical_quaternions) is not bool:
            raise ValueError("action history canonical_quaternions must be an exact bool")
        if (
            type(self.definition) is not str
            or self.definition != DEFINITION
            or any(
                type(value) is not int or value != expected
                for value, expected in (
                    (self.source_dim, 42),
                    (self.feature_dim, 47),
                    (self.closing_limit, 6),
                )
            )
        ):
            raise ValueError("fixed action history recipe metadata changed")


ActionHistoryRecipe = PickCubeActionHistoryRecipe | PickCubeTaskSpaceHistoryRecipe


def is_taskspace_recipe(value: object) -> bool:
    return type(value) is PickCubeTaskSpaceHistoryRecipe


def validate_action_history_recipe(value: ActionHistoryRecipe) -> ActionHistoryRecipe:
    if type(value) is PickCubeTaskSpaceHistoryRecipe:
        return validate_taskspace_history_recipe(value)
    if type(value) is not PickCubeActionHistoryRecipe:
        raise ValueError("expected an exact supported action history recipe")
    return PickCubeActionHistoryRecipe(
        value.mode,
        value.canonical_quaternions,
        value.definition,
        value.source_dim,
        value.feature_dim,
        value.closing_limit,
    )


def _native_action(action: tuple[float, ...]) -> tuple[float, ...]:
    owned = _vector(action, "executed nativefloat32 action", 4)
    if len(owned) != 4 or any(not -1.0 <= value <= 1.0 for value in owned):
        raise ValueError("action history requires four normalized executed axes")
    if any(struct.unpack("=f", struct.pack("=f", value))[0] != value for value in owned):
        raise ValueError("executed action must already be exactly nativefloat32; no rounding")
    return owned


@dataclass(frozen=True)
class ActionHistoryState:
    previous_action: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    closing_count: int = 0

    def __post_init__(self) -> None:
        action = _native_action(self.previous_action)
        if type(self.closing_count) is not int or not 0 <= self.closing_count <= 6:
            raise ValueError("action history closing_count must be a native integer in0..6")
        if (action[3] < 0.0) != (self.closing_count > 0):
            raise ValueError("action history closing_count differs from the previous gripper sign")
        object.__setattr__(self, "previous_action", action)


def validate_action_history_state(value: ActionHistoryState) -> ActionHistoryState:
    if type(value) is not ActionHistoryState:
        raise ValueError("expected an exact ActionHistoryState")
    return ActionHistoryState(value.previous_action, value.closing_count)


def advance_action_history(
    state: ActionHistoryState, executed_action: tuple[float, ...]
) -> ActionHistoryState:
    owned = validate_action_history_state(state)
    action = _native_action(executed_action)
    count = min(6, owned.closing_count + 1) if action[3] < 0.0 else 0
    return ActionHistoryState(action, count)


def action_history_spec(
    spec: ContinuousActionSpec, recipe: ActionHistoryRecipe
) -> ContinuousActionSpec:
    config = validate_action_history_recipe(recipe)
    owned = _copy_spec(spec)
    if (
        owned.observation_dim != config.source_dim
        or owned.action_units != _UNITS
        or owned.lower_bounds != (-1.0,) * 4
        or owned.upper_bounds != (1.0,) * 4
    ):
        raise ValueError("action history requires the complete normalized root-axis raw42 spec")
    return ContinuousActionSpec(
        config.feature_dim, owned.action_units, owned.lower_bounds, owned.upper_bounds
    )


def action_history_observation(
    raw42: tuple[float, ...], state: ActionHistoryState, recipe: ActionHistoryRecipe
) -> tuple[float, ...]:
    config = validate_action_history_recipe(recipe)
    owned = validate_action_history_state(state)
    if type(raw42) is not tuple or len(raw42) != config.source_dim:
        raise ValueError("action history requires a complete raw42 observation tuple")
    raw = _vector(raw42, "raw action-history state42", config.source_dim)
    if type(config) is PickCubeTaskSpaceHistoryRecipe:
        prefix = taskspace_observation(raw, config)
    else:
        prefix = (
            state_quaternion_observation(raw, state_quaternion_projection())
            if config.canonical_quaternions
            else raw
        )
    tail = (
        (*owned.previous_action, owned.closing_count / config.closing_limit)
        if config.mode == "executed"
        else (0.0,) * 5
    )
    return prefix + tail


def _training_source(
    episode: ContinuousEpisode, *, mass_context: PickCubeMassContext | None = None
) -> ContinuousEpisode:
    owned = copy_continuous_episode(episode)
    mass = require_mass_binding(mass_context, context=owned.context, seed=17, role=owned.role)
    if (
        owned.role != "train"
        or (
            mass is None
            and owned.context not in tuple(pick_cube_training_context(seed) for seed in TRAIN_SEEDS)
        )
        or len(owned.transitions) > 50
    ):
        raise ValueError("action history requires a complete declared TRAIN episode")
    return owned


def _mass_recipe(config: ActionHistoryRecipe, mass_context: PickCubeMassContext | None) -> None:
    # Why: mass metadata binds the existing physical recipe, not a new feature or
    # authority to qualify a teacher or convert a reserved role into TRAIN.
    if mass_context is not None and (
        type(config) is not PickCubeTaskSpaceHistoryRecipe or config.mode != "executed"
    ):
        raise ValueError("mass action history requires fixed executed taskspace12")


def action_history_training_episode(
    episode: ContinuousEpisode,
    recipe: ActionHistoryRecipe,
    *,
    mass_context: PickCubeMassContext | None = None,
) -> ContinuousEpisode:
    """Derive before/after execution features, resetting history per episode."""
    config = validate_action_history_recipe(recipe)
    _mass_recipe(config, mass_context)
    owned = _training_source(episode, mass_context=mass_context)
    return _action_history_episode(owned, config)


def _probe_source(episode: ContinuousEpisode) -> ContinuousEpisode:
    # Check original native flag types before episode ownership can normalize
    # forged integer flags. This does not qualify the collection or release data.
    _observed_grasp_flags(episode)
    owned = copy_continuous_episode(episode)
    binding = pick_cube_probe_binding(owned.role)
    if owned.context != binding.context or owned.episode_id != binding.episode_id:
        raise ValueError("action history requires the original reserved23/24 probe identity")
    return owned


def action_history_probe_episode(
    episode: ContinuousEpisode, recipe: ActionHistoryRecipe
) -> ContinuousEpisode:
    """Encode a fixed reserved probe without changing its role or reset identity."""
    config = validate_action_history_recipe(recipe)
    owned = _probe_source(episode)
    return _action_history_episode(owned, config)


def _action_history_episode(
    owned: ContinuousEpisode, config: ActionHistoryRecipe
) -> ContinuousEpisode:
    spec = action_history_spec(owned.spec, config)
    state = ActionHistoryState()
    transitions = []
    for row in owned.transitions:
        before = action_history_observation(row.observation, state, config)
        state = advance_action_history(state, row.action)
        after = action_history_observation(row.next_observation, state, config)
        transitions.append(
            ContinuousTransition(
                row.step,
                before,
                row.action,
                row.reward,
                after,
                row.terminated,
                row.truncated,
            )
        )
    return ContinuousEpisode(owned.episode_id, owned.context, owned.role, spec, tuple(transitions))


def validate_action_history_training_episode(
    episode: ContinuousEpisode,
    recipe: ActionHistoryRecipe,
    *,
    mass_context: PickCubeMassContext | None = None,
) -> ContinuousEpisode:
    """Own an encoded TRAIN episode only when every tail follows actual actions."""
    config = validate_action_history_recipe(recipe)
    _mass_recipe(config, mass_context)
    owned = _training_source(episode, mass_context=mass_context)
    return _validate_encoded_history(owned, config)


def validate_action_history_probe_episode(
    episode: ContinuousEpisode, recipe: ActionHistoryRecipe
) -> ContinuousEpisode:
    """Own reserved encoded data only when its tails follow original executions.

    A reduced12D prefix cannot reconstruct raw42; the collecting boundary keeps
    that raw/model binding. This validator grants neither training nor test use.
    """
    config = validate_action_history_recipe(recipe)
    owned = _probe_source(episode)
    return _validate_encoded_history(owned, config)


def _validate_encoded_history(
    owned: ContinuousEpisode, config: ActionHistoryRecipe
) -> ContinuousEpisode:
    raw_spec = ContinuousActionSpec(
        config.source_dim, owned.spec.action_units, owned.spec.lower_bounds, owned.spec.upper_bounds
    )
    if owned.spec != action_history_spec(raw_spec, config):
        raise ValueError("encoded action history requires the declared model spec")
    state = ActionHistoryState()
    for row in owned.transitions:
        if type(config) is PickCubeTaskSpaceHistoryRecipe:
            before = _taskspace_encoded_observation(row.observation, state, config)
            state = advance_action_history(state, row.action)
            after = _taskspace_encoded_observation(row.next_observation, state, config)
        else:
            before = action_history_observation(row.observation[:42], state, config)
            state = advance_action_history(state, row.action)
            after = action_history_observation(row.next_observation[:42], state, config)
        if row.observation != before or row.next_observation != after:
            raise ValueError("encoded history differs from preceding actually executed commands")
    return owned


def _taskspace_encoded_observation(observation, state, config):
    # Raw42 is deliberately not reconstructed from the reduced feature view.
    # Outer raw/model binding supplies that authority; this checks causal tails.
    if type(observation) is not tuple or len(observation) != config.feature_dim:
        raise ValueError("encoded taskspace history requires a complete12D observation")
    values = _vector(observation, "encoded taskspace history", config.feature_dim)
    if values[0] not in (0.0, 1.0):
        raise ValueError("encoded taskspace is_grasped must be exactly zero or one")
    tail = (
        (*state.previous_action, state.closing_count / config.closing_limit)
        if config.mode == "executed"
        else (0.0,) * 5
    )
    return values[:7] + tail
