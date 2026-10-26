"""Build a finite catalogue of key/door action-remapping instances.

Inputs are two distinct native catalogue indices in 0..23, called environment
seeds for experiment configuration. They select all combinations of four
rotations, three branch rows and two key assignments; they are not arbitrary
random draws. Outputs are fresh frozen A/B/C/D layouts. A/B/C share geometry,
keys and appearance; only B changes controls. D uses the heldout index.

No RNG, model, observations, training, evaluation, IO, instance selection from
results or claim of statistical independence belongs here. Callers own train/
heldout roles and must freeze catalogue indices before comparative execution.
"""

from src.core.key_door_navigation import KeyDoorLayout


def _layout(
    seed: int, identity: str, movement_action_map: tuple[int, int, int, int]
) -> KeyDoorLayout:
    return KeyDoorLayout(
        layout_id=identity,
        rotation=seed % 4,
        branch_row=1 + (seed // 4) % 3,
        swap_keys=bool(seed // 12),
        appearance=0.0,
        movement_action_map=movement_action_map,
    )


def seeded_action_remapping_layouts(
    environment_seed: int, heldout_environment_seed: int
) -> tuple[KeyDoorLayout, ...]:
    """Return A/B/C training contexts and a physically distinct heldout D.

    Why catalogue indices: deterministic finite coverage keeps changed controls
    separate from changed geometry, without introducing another random stream.
    A and C use identity controls; returning to A restores the original instance.
    """
    for name, seed in (
        ("environment_seed", environment_seed),
        ("heldout_environment_seed", heldout_environment_seed),
    ):
        if type(seed) is not int or not 0 <= seed <= 23:
            raise ValueError(f"{name} must be a native integer catalogue index in 0..23")
    if environment_seed == heldout_environment_seed:
        raise ValueError("environment and heldout environment seeds must be distinct")
    identity = (0, 1, 2, 3)
    return (
        _layout(environment_seed, "A", identity),
        _layout(environment_seed, "B", (1, 2, 3, 0)),
        _layout(environment_seed, "C", identity),
        _layout(heldout_environment_seed, "D", identity),
    )
