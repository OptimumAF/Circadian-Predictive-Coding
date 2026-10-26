"""Fixed recorded-stage index sampling for joint TRAIN corrections.

Input is the original ordered stage tuple. Output is eight bounded index tuples;
callers gather existing observations/targets without rebuilding action histories.
No rows, labels, teacher phases, weights, randomness, learner, IO or mutable
schedule state are created. This changes training exposure, not source authority.
"""

STAGES = ("align", "descend", "grasp", "lift", "settle")
_UPDATES = 8


def plan_joint_stage_sampling(stages: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    """Give each recorded stage N/5 rows per update, cycling original ordinals.

    Stage blocks always follow STAGES. A stage's cursor continues across all
    eight batches, preserving original ascending row order within each cycle.
    Every original row must be represented somewhere in the complete plan.
    """
    if type(stages) is not tuple:
        raise ValueError("joint stage sampling requires an exact stage tuple")
    count = len(stages)
    if not 5 <= count <= 128 or count % len(STAGES) != 0:
        raise ValueError("joint stage sampling requires5..128 rows divisible by five")
    indices: dict[str, list[int]] = {stage: [] for stage in STAGES}
    for index, stage in enumerate(stages):
        if type(stage) is not str or stage not in indices:
            raise ValueError("joint stage sampling requires exact known recorded stage strings")
        indices[stage].append(index)
    if any(not indices[stage] for stage in STAGES):
        raise ValueError("joint stage sampling requires all five recorded stages")
    quota = count // len(STAGES)
    batches = []
    for update in range(_UPDATES):
        batch: list[int] = []
        for stage in STAGES:
            source = indices[stage]
            start = update * quota
            batch.extend(source[(start + offset) % len(source)] for offset in range(quota))
        batches.append(tuple(batch))
    plan = tuple(batches)
    # Why: balancing must not silently discard a source row from the fixed dose.
    if {index for batch in plan for index in batch} != set(range(count)):
        raise ValueError("joint stage sampling cannot cover every original row in eight updates")
    return plan
