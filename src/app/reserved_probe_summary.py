"""Summarize existing ordinary reserved-probe records, without new model work.

Owns fixed cohort/role/bounds admission and a small detached projection. It does
not authenticate source preservation, re-score actions, select probes or train.
"""

from src.core.probe_task_outcomes import task_outcome_changes

OLD_TASKS = (("v1", "invoice"), ("v1", "password"), ("v2", "invoice"), ("v2", "parcel"))
FRESH_TASKS = (("v1", "parcel"), ("v2", "password"))
TASKS = OLD_TASKS + FRESH_TASKS
PHASES = ("before_probe", "after_probe", "after_return")


def _dictionary(value, name):
    if value is None:
        return {}
    if type(value) is not dict:
        raise ValueError(f"{name} must be a plain record")
    return value


def _admit(episodes):
    if type(episodes) is not dict or not set(episodes) <= set(PHASES):
        raise ValueError("reserved phases must be the original three phases")
    # All headers precede outcome access, including headers in later phases.
    for phase, rows in episodes.items():
        if type(rows) is not list or len(rows) > 6:
            raise ValueError("reserved phase requires at most six fixed task rows")
        for index, row in enumerate(rows):
            if (
                type(row) is not dict
                or type(row.get("role")) is not str
                or row["role"] != "controller_train"
                or row.get("phase") != phase
                or type(row.get("policy_version")) is not str
                or type(row.get("concept")) is not str
                or (row["policy_version"], row["concept"]) != TASKS[index]
                or type(row.get("complete")) is not bool
            ):
                raise ValueError("reserved task header/role/order differs from fixed TRAIN readout")


def _phase(rows):
    observed: dict[str, bool | None] = {f"{v}:{c}": None for v, c in TASKS}
    for row in rows:
        if row["complete"] is not True:
            continue
        episode = _dictionary(row.get("episode"), "episode")
        if (
            episode.get("complete") is not True
            or type(episode.get("success")) is not bool
            or episode.get("scenario") != "ordinary"
            or episode.get("training_updates") != 0
            or type(episode.get("training_updates")) is not int
        ):
            raise ValueError("complete task requires original ordinary zero-training outcome")
        observed[f"{row['policy_version']}:{row['concept']}"] = episode["success"]
    complete = len(rows) == 6 and all(value is not None for value in observed.values())
    return dict(
        status="complete" if complete else "partial" if rows else "not_run",
        observed_success=observed,
    )


def _counts(record, names):
    values = _dictionary(record, "reported work")
    result = {}
    for name in names:
        value = values.get(name)
        if value is not None and (type(value) is not int or not 0 <= value <= 2**53 - 1):
            raise ValueError("reported work must be bounded nonnegative counts or unknown")
        result[name] = value
    return result


def _ce_rehearsal_costs(private):
    record = _dictionary(private.get("preserved_ce_rehearsal_work"), "CE rehearsal work")
    events = record.get("events", [])
    if type(events) is not list or len(events) > 32 or any(type(e) is not dict for e in events):
        raise ValueError("CE rehearsal requires bounded scalar events")
    keys = (
        "closure_attempts",
        "forward_rows",
        "backward_attempts",
        "backward_rows",
        "probe_gradient_rows",
        "current_train_gradient_rows",
        "retained_train_gradient_rows",
        "application_attempts",
        "application_returns",
        "application_verified",
    )
    attempted = _counts(record, ("attempted_updates",))["attempted_updates"]
    coverage_known = "events" in record and type(attempted) is int and attempted == len(events)
    # Empty observed work is zero only with explicit complete event coverage.
    totals: dict[str, int | None] = {k: 0 if coverage_known else None for k in keys}
    unknown = 0 if coverage_known else None
    for event in events:
        if event.get("work_known") is not True and unknown is not None:
            unknown += 1
        counts = _counts(event.get("work"), keys)
        for key, value in counts.items():
            totals[key] = None if totals[key] is None or value is None else totals[key] + value
    return dict(
        application_slots=_counts(
            record,
            (
                "attempted_updates",
                "attempted_rows",
                "returned_updates",
                "applied_updates",
                "native_sgd_updates",
            ),
        ),
        auxiliary_work=totals,
        unknown_work_events=unknown,
        counter_unit="callback applications and current rows; auxiliary gradients separate from native SGD",
    )


def _costs(report, private):
    auxiliary = _dictionary(report.get("probe_initialization"), "auxiliary initialization")
    probe = _dictionary(private.get("probe"), "probe")
    returned = _dictionary(probe.get("return_phase"), "return phase")
    native_keys = ("attempted_updates", "attempted_rows", "returned_updates")
    phase_keys = (
        "attempted_updates",
        "completed_updates",
        "attempted_training_examples",
        "completed_training_examples",
        "score_forward_calls_attempted",
        "score_forward_calls_completed",
    )
    readouts = _dictionary(private.get("task_readout_work"), "task readout work")
    linear = _dictionary(private.get("preserved_linear_rehearsal_work"), "shared linear rehearsal")
    return dict(
        **(
            {
                "linear_rehearsal": _counts(
                    linear,
                    (
                        "attempted_updates",
                        "attempted_rows",
                        "returned_updates",
                        "applied_updates",
                        "application_unknown_updates",
                        "additional_rows_charged",
                        "native_sgd_updates",
                    ),
                )
            }
            if report.get("reserved_probe_rehearsal") in ("joint", "current_only")
            else {}
        ),
        **(
            {"ce_rehearsal": _ce_rehearsal_costs(private)}
            if report.get("reserved_probe_update") == "ce_rehearsal"
            else {}
        ),
        source_training=_counts(report.get("training_work"), native_keys),
        auxiliary_fit=_counts(
            auxiliary.get("preserved_fit_work", auxiliary.get("fit_work")),
            (
                "optimizer_constructions",
                "closure_attempts",
                "forward_attempts",
                "forward_rows",
                "backward_attempts",
                "backward_rows",
                "step_attempts",
                "step_returns",
                "final_read_rows",
            ),
        ),
        private_training=_counts(private.get("preserved_training_work"), native_keys),
        fresh_training=_counts(probe, phase_keys),
        return_training=_counts(returned, phase_keys),
        task_readouts={
            p: _counts(
                readouts.get(p),
                (
                    "decision_attempts",
                    "decision_returns",
                    "native_forward_attempts",
                    "native_forward_returns",
                ),
            )
            for p in PHASES
        },
        scope="reported counts only; missing is unknown, auxiliary old-TRAIN reuse is paid rehearsal; see raw timing/memory/copy costs; not matched total computation",
    )


def summarize_reserved_probe(report: dict) -> dict:
    report = _dictionary(report, "source report")
    if report.get("reserved_probe_requested") is not True:
        raise ValueError("summary requires an explicitly requested reserved probe")
    private = _dictionary(report.get("reserved_probe"), "reserved probe")
    if any(
        type(r.get("final_examples", 0)) is not int or r.get("final_examples", 0) != 0
        for r in (report, private)
    ):
        raise ValueError("final evaluation is not an input to the reserved summary")
    episodes = _dictionary(private.get("episodes"), "phase episodes")
    _admit(episodes)
    phases = {p: _phase(episodes.get(p, [])) for p in PHASES}
    cohorts = {}
    for name, tasks in (("source_trained", OLD_TASKS), ("fresh", FRESH_TASKS)):
        labels = [f"{v}:{c}" for v, c in tasks]
        qualified = {p: phases[p]["status"] == "complete" for p in PHASES}
        cohorts[name] = dict(
            tasks=labels,
            successes_by_phase={
                p: sum(phases[p]["observed_success"][t] is True for t in labels)
                if qualified[p]
                else None
                for p in PHASES
            },
            task_changes={
                t: task_outcome_changes(
                    *(phases[p]["observed_success"][t] if qualified[p] else None for p in PHASES)
                )
                for t in labels
            },
        )
    statuses = [phases[p]["status"] for p in PHASES]
    return dict(
        definition="ordinary_reserved_task_outcomes_v1",
        status="complete"
        if all(s == "complete" for s in statuses)
        else "not_run"
        if all(s == "not_run" for s in statuses)
        else "partial",
        phases=phases,
        cohorts=cohorts,
        reported_costs=_costs(report, private),
        qualification="known-template recorded outcomes only, not source-preservation authentication or statistical/generalization/CPC benefit; null recovery means no observed loss or unknown return",
    )


def add_reserved_probe_summary(report: dict) -> None:
    """Append metadata after all original bookkeeping; preserve original failure."""
    try:
        summary = summarize_reserved_probe(report)
    except ValueError as error:
        summary = dict(
            definition="ordinary_reserved_task_outcomes_v1",
            status="invalid",
            reason=str(error)[:256],
        )
    report["reserved_probe_summary"] = summary
