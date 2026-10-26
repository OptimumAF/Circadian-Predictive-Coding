"""Fixed guard-only reset allocation; no source relabeling or evaluation release.

Inputs: a declared native reset seed. Output: its inner_guard context. This
helper never collects data, selects successful outcomes or authorizes promotion.
Why: separate guard acquisition from TRAIN17/19 and reserved probe23/final29.
"""


def pick_cube_guard_context(seed: int) -> str:
    if type(seed) is not int or seed not in (31, 37):
        raise ValueError("inner_guard PickCube seed must be exactly31 or37")
    return f"pick-cube-inner-guard-{seed}"
