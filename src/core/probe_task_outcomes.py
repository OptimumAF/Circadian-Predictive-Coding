"""Descriptive changes for one already-observed task; no learner or IO.

None means an endpoint is unknown or a conditional measure does not apply.
The caller supplies fixed cohorts and independently qualified observations.
"""


def task_outcome_changes(before: bool | None, after: bool | None, returned: bool | None) -> dict:
    if any(value is not None and type(value) is not bool for value in (before, after, returned)):
        raise ValueError("task success must be an exact boolean or unknown")
    change: int | None
    acquired: bool | None
    lost: bool | None
    if before is not None and after is not None:
        change = int(after) - int(before)
        acquired = not before and after
        lost = before and not after
    else:
        change = acquired = lost = None
    return dict(
        probe_success_change=change,
        acquired_during_probe=acquired,
        retained_through_probe=after if before is True else None,
        lost_during_probe=lost,
        return_success_change=int(returned) - int(after)
        if after is not None and returned is not None
        else None,
        # Why: better performance on return is recovery only after observed loss.
        recovered_on_return=returned if lost is True else None,
        retained_after_return=returned if after is True else None,
    )
