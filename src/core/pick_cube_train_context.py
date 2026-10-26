"""Declared same-task TRAIN reset contexts; no simulator or role conversion.

Why: expose fixed TRAIN coverage without admitting reserved probe/final seeds
or selecting a successful seed after observation. Context names are metadata.
"""

TRAIN_SEEDS = (17, 19, 20)


def pick_cube_training_context(seed: int) -> str:
    if type(seed) is not int or seed not in TRAIN_SEEDS:
        raise ValueError(
            "TRAIN PickCube seed must be exactly17,19 or20; probe/final seeds are reserved"
        )
    return "pick-cube" if seed == 17 else f"pick-cube-train-{seed}"
