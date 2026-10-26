"""Present recorded private affine learning, retention and partial native games.

Consumes bounded JSON, returns detached presentation data with original pointers.
No parameter reconstruction, authenticity claim, learner, IO or experiment selection.
Why a separate projection: the private phase clocks/roles differ from normal TRAIN;
sharing the existing viewer does not require relabeling its evidence.
"""

from copy import deepcopy
from typing import Any

from .minigrid_affine_inspector import METHOD, SOURCES, _fixed, _native_costs, _object, _training
from .minigrid_inspector_projection import (
    _close,
    _envelope,
    _episode,
    _integer,
    _number,
    _records,
    _require,
    _status,
)
from .minigrid_learnability_inspector import _flatten, _missing, _proofs

DEFINITION = "portable_affine_private_adaptation_v1"
ARMS = ("current_only", "joint_ABC_F")
LABELS = ("F-only control", "Joint ABC + F")
SLOTS = (
    ("F-blue", "blue", "probe_test"),
    ("F-red", "red", "probe_test"),
    ("A", "blue", "train"),
    ("A", "red", "train"),
)


def _costs(record, pointer, keys, label):
    return _flatten({key: record[key] for key in keys if key in record}, pointer, label)


def _budget(record, *, maximum, replay, wall, done=False):
    _object(record, "private budget")
    limits = _object(record.get("limits"), "private budget limits")
    _require(record.get("initial_limits") == limits, "original private budget limits changed")
    for key, value in dict(
        max_training_updates=maximum,
        max_replay_examples=replay,
        max_wall_seconds=wall,
        max_process_rss_bytes=512 * 1024**2,
    ).items():
        _fixed(limits, key, value)
    charged = _integer(record.get("updates_completed"), 0, maximum, "paid updates")
    reused = _integer(record.get("replay_examples_completed"), 0, replay, "paid replay")
    _require(
        not done or (charged == maximum and reused == replay), "complete private charges missing"
    )
    start = _number(record.get("started_at"), "original start", 0)
    _require(record.get("observed_budget_started_at") == start, "private clock was renewed")
    _number(record.get("ended_at"), "original end", start)


def _points(records, name):
    result = _records(records, 3, name)
    _require(
        [p.get("updates") for p in result] == list(range(len(result))), "private checkpoint order"
    )
    for p in result:
        _integer(p["updates"], 0, 2, "private update")
        _require(_status(p.get("status", "complete")), "invalid private checkpoint status")
    _require(
        all(p.get("status", "complete") == "complete" for p in result[:-1]),
        "private readouts continue after a failed point",
    )
    return result


def _private_curves(arm, path):
    curves = []
    probe = _object(arm.get("probe") or {}, "private probe")
    groups = (
        (
            _points(probe.get("checkpoints", []), "F readouts"),
            "scores",
            "Private F",
            "probe_test",
            tuple((f"F-{cue}", f"F-{cue}", cue) for cue in ("blue", "red")),
            f"{path}/probe/checkpoints",
        ),
        (
            _points(arm.get("old_task_readouts", []), "old TRAIN readouts"),
            "contexts",
            "Old TRAIN fit",
            "train",
            SOURCES,
            f"{path}/old_task_readouts",
        ),
    )
    for points, field, phase, role, identities, prefix in groups:
        if arm["status"] == "complete":
            _require(len(points) == 3, "complete arm lacks private readouts")
        for update in range(3):
            point = points[update] if update < len(points) else {}
            scores = _object(point.get(field, {}), "private scores")
            _require(set(scores) <= {i[0] for i in identities}, "unknown private score source")
            for identity, context, cue in identities:
                score = scores.get(identity)
                pointer = None if score is None else f"{prefix}/{update}/{field}/{identity}"
                ce = accuracy = None
                if score is not None:
                    _object(score, "private score")
                    ce = _number(score.get("ce"), "recorded private CE", 0)
                    if role == "train":
                        _fixed(score, "rows", 9)
                        correct = _integer(score.get("correct"), 0, 9, "old correct actions")
                        for key in ("actions", "expected"):
                            _require(
                                type(score.get(key)) is list and len(score[key]) == 9,
                                "old action rows",
                            )
                            for value in score[key]:
                                _integer(value, 0, 6, "old action")
                        _close(
                            correct,
                            sum(a == b for a, b in zip(score["actions"], score["expected"])),
                            "old correct actions",
                        )
                        accuracy = correct / 9
                    else:
                        accuracy = _number(score.get("accuracy"), "recorded F accuracy", 0, 1)
                _require(
                    arm["status"] != "complete" or score is not None,
                    "complete private score missing",
                )
                curves.append(
                    dict(
                        source_updates=6,
                        phase=phase,
                        context=context,
                        cue=cue,
                        role=role,
                        updates=update,
                        status=point.get("status", "complete") if score is not None else "not_run",
                        ce=ce,
                        accuracy=accuracy,
                        pointer=pointer,
                        derived_accuracy=role == "train",
                    )
                )
    return curves


def _arm(arm, index, path):
    _fixed(arm, "arm", ARMS[index])
    _require(_status(arm.get("status")), "invalid private arm status")
    done = arm["status"] == "complete"
    replay = 54 if index else 0
    if arm.get("budget") is not None:
        _budget(arm["budget"], maximum=2, replay=replay, wall=90, done=done)
    _require(not done or arm.get("budget") is not None, "complete private budget missing")
    attempts = _records(arm.get("attempts", []), 2, "permanent private attempts")
    for i, attempt in enumerate(attempts):
        for key, value in dict(
            update=i + 1,
            fit_rows_attempted=36,
            old_TRAIN_reuse_charged=27 if index else 0,
            extra_F_exposures_charged=0 if index else 27,
        ).items():
            _fixed(attempt, key, value)
    if done:
        _require(len(attempts) == 2, "complete private attempts missing")
        _fixed(arm, "source_fork_calls", 1)
        _fixed(arm, "source_native_state_unchanged", True)
        _fixed(arm, "committed_old_TRAIN_reuse_rows", replay)
        _fixed(arm, "committed_extra_F_rows", 54 - replay)
        probe = _object(arm.get("probe"), "complete probe")
        _fixed(probe, "completed_updates", 2)
        _fixed(probe, "completed_training_examples", 72)
        state = _object(arm.get("native_endpoint"), "private native endpoint")
        _fixed(state, "completed_updates", 8)
        _fixed(state, "training_examples", 180)
        _require(state.get("failed_work") is None, "complete private endpoint contains failed work")
        heads = _records(state.get("heads"), 2, "private heads")
        _require(len(heads) == 2, "private endpoint requires two heads")
        for head in heads:
            _fixed(head, "fit_rows", 36)
    costs = _costs(
        arm,
        path,
        (
            "budget",
            "source_fork_calls",
            "updates_accounted_by_study",
            "committed_old_TRAIN_reuse_rows",
            "committed_extra_F_rows",
            "elapsed_seconds",
        ),
        "Private arm",
    )
    costs += _costs(
        arm.get("plan") or {},
        f"{path}/plan",
        ("input_array_bytes", "old_TRAIN_reuse_limit", "extra_F_exposure_limit", "storage_scope"),
        "Owned fit matrices",
    )
    for i, attempt in enumerate(attempts):
        costs += _flatten(attempt, f"{path}/attempts/{i}", "Permanent admission attempt")
    if type(arm.get("probe")) is dict:
        costs += _costs(
            arm["probe"],
            f"{path}/probe",
            (
                "attempted_updates",
                "completed_updates",
                "attempted_training_examples",
                "completed_training_examples",
                "elapsed_seconds",
                "reserved_data_bytes",
                "timing_status",
                "score_forward_calls_attempted",
                "score_forward_calls_completed",
                "training_readouts",
            ),
            "Private diagnostics",
        )
    return _private_curves(arm, path), costs, _proofs(arm, path)


def _games(phase, index, complete):
    pointer = f"/games/{index}"
    records, proofs, costs = [], [], []
    if phase is not None:
        _object(phase, "private native phase")
        for key, value in dict(
            method=METHOD,
            role="private_F_then_returned_A",
            rehearsal_arm=ARMS[index],
            updates=8,
            source_fit_calls=8,
            source_fit_rows=180,
            new_fit_calls=0,
            new_optimizer_updates=0,
        ).items():
            _fixed(phase, key, value)
        _require(_status(phase.get("status")), "invalid private native status")
        done = phase["status"] == "complete"
        _require(not complete or done, "complete workflow contains partial native phase")
        _require(
            not done or phase.get("state_unchanged") is True, "complete native preservation missing"
        )
        _require(phase.get("scores") == [], "native phase contains labelled scores")
        records = _records(phase.get("episodes", []), 4, "private native episodes")
        _require(
            not done or len(records) == 4 and all(e.get("status") == "complete" for e in records),
            "complete private games missing",
        )
        proofs += _proofs(phase, pointer)
        costs += _native_costs(phase, records, pointer=pointer, label="Private native readout")
    else:
        _require(not complete, "complete workflow lacks native phase")
    episodes, failed = [], False
    for i, (context, cue, role) in enumerate(SLOTS):
        label = "Fresh F" if i < 2 else "Returning A"
        if i >= len(records):
            game = _missing(6, label, context, cue, role, 8)
        else:
            row, path = records[i], f"{pointer}/episodes/{i}"
            _require(
                not failed or row.get("status") == "not_run", "native work follows failed episode"
            )
            if row.get("status") == "not_run":
                for key, value in dict(
                    context=context,
                    cue=cue,
                    source_role=role,
                    prediction_calls_attempted=0,
                    prediction_calls_returned=0,
                ).items():
                    _fixed(row, key, value)
                _require(
                    not row.get("predictions")
                    and row.get("environment") is None
                    and row.get("success") is None
                    and row.get("return") is None
                    and row.get("state_unchanged") is None,
                    "unrun private game contains observations",
                )
                game = _missing(6, label, context, cue, role, 8)
                game.update(pointer=path, record=deepcopy(row), derived=False)
            else:
                game = _episode(
                    row,
                    path,
                    METHOD,
                    8,
                    cue,
                    context=context,
                    role=role,
                    branch_row=2 if i < 2 else 3,
                    allow_observed_failure_tail=True,
                )
                game.update(source_updates=6, phase=label, context=context, derived=False)
            failed = failed or row.get("status") != "complete"
        episodes.append(game)
    return episodes, proofs, costs


def project_private_affine_report(report: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    _envelope(report, source_label)
    _fixed(report, "definition", DEFINITION)
    _require(_status(report.get("status")), "invalid private workflow status")
    source = deepcopy(report)
    done = source["status"] == "complete"
    _require(
        not done or source.get("TRAIN_reference_unchanged") is True,
        "complete TRAIN reference preservation missing",
    )
    training = source.get("training")
    curves, receipts, wakes, train_costs, train_proofs = _training(training, done)
    conditions = [
        dict(
            key="TRAIN reference",
            pointer="/training" if training is not None else None,
            status=(training or {}).get("status", "not_run"),
            source_role="train",
            episodes=[],
            curves=curves,
            receipts=receipts,
            wakes=wakes,
            costs=train_costs,
            proofs=train_proofs,
            native_status=None,
        )
    ]
    comparison = source.get("private_comparison")
    arms = []
    if comparison is not None:
        _object(comparison, "private comparison")
        _fixed(comparison, "definition", "minigrid_private_affine_rehearsal_comparison_v1")
        _require(
            type(training) is dict and training.get("status") == "complete",
            "private work precedes complete TRAIN",
        )
        _require(_status(comparison.get("status")), "invalid comparison status")
        arms = _records(comparison.get("arms", []), 2, "private arms")
        complete = comparison["status"] == "complete"
        protocol = comparison.get("protocol")
        if protocol is not None:
            _object(protocol, "private lineage protocol")
            for key, value in dict(
                upstream_paid_fits=6,
                upstream_paid_rows=108,
                source_native_fits=6,
                source_native_rows=108,
            ).items():
                _fixed(protocol, key, value)
        _require(not complete or protocol is not None, "complete private lineage missing")
        _require(not done or complete, "complete workflow contains incomplete comparison")
        _require(not complete or len(arms) == 2, "complete comparison lacks both arms")
        if comparison.get("budget") is not None:
            _budget(comparison["budget"], maximum=4, replay=54, wall=90, done=complete)
        _require(
            not complete or comparison.get("budget") is not None,
            "complete comparison budget missing",
        )
    else:
        _require(not done, "complete workflow lacks private comparison")
    phases = _records(source.get("games") or [], 2, "private native phases")
    _require(
        not phases or comparison is not None and comparison["status"] == "complete",
        "games precede complete private comparison",
    )
    _require(not done or len(phases) == 2, "complete workflow lacks both native phases")
    for index, label in enumerate(LABELS):
        path = f"/private_comparison/arms/{index}"
        arm = arms[index] if index < len(arms) else None
        if index and index < len(arms):
            _require(arms[0]["status"] == "complete", "private work follows failed arm")
        private_curves, costs, proofs = (
            _arm(arm, index, path)
            if arm is not None
            else (_private_curves(dict(status="not_run"), path), [], [])
        )
        if (
            arm is not None
            and arm.get("budget") is not None
            and comparison is not None
            and comparison.get("budget") is not None
        ):
            _require(
                arm["budget"]["started_at"] == comparison["budget"]["started_at"]
                and arm["budget"]["ended_at"] <= comparison["budget"]["ended_at"],
                "private arm budget is outside the original study lifetime",
            )
        phase = phases[index] if index < len(phases) else None
        if index and phase is not None:
            _require(phases[0]["status"] == "complete", "native phase follows incomplete readout")
        episodes, game_proofs, game_costs = _games(phase, index, done)
        conditions.append(
            dict(
                key=label,
                pointer=path if arm is not None else None,
                status=(arm or {}).get("status", "not_run"),
                source_role="private probe_train" + (" + TRAIN reuse" if index else ""),
                native_status=(phase or {}).get("status"),
                episodes=episodes,
                curves=private_curves,
                costs=costs + game_costs,
                proofs=proofs + game_proofs,
                receipts=receipts,
                wakes=wakes,
            )
        )
    global_costs = _costs(
        source,
        "",
        ("workflow_budget", "readout_budget", "attempted_phases"),
        "Workflow recorded scope",
    )
    if source.get("readout_budget") is not None:
        _budget(source["readout_budget"], maximum=0, replay=0, wall=150)
    _require(
        not done or source.get("readout_budget") is not None,
        "complete workflow readout budget missing",
    )
    if comparison is not None:
        global_costs += _costs(
            comparison, "/private_comparison", ("budget", "elapsed_seconds"), "Private study"
        )
    collection = _records(source.get("collection", []), 4, "teacher collection")
    if comparison is not None:
        _require(
            len(collection) == 4
            and all(r.get("closed") is True and r.get("status") == "complete" for r in collection),
            "private fitting precedes complete teacher collection",
        )
    for i, record in enumerate(collection):
        global_costs += _costs(
            record,
            f"/collection/{i}",
            (
                "role",
                "context",
                "cue",
                "closed",
                "status",
                "environment_steps",
                "source_array_bytes",
            ),
            "Teacher collection",
        )
    if type(source.get("protocol")) is dict:
        global_costs += _costs(
            source["protocol"],
            "/protocol",
            ("new_fit_calls", "new_fit_rows", "historical_source_reads"),
            "Protocol plan (not a paid-work receipt)",
        )
    for row in conditions:
        row["proofs"] += _proofs(source, "")
    return dict(
        schema_id="minigrid_private_affine_inspector_v1",
        source_label=source_label,
        origin="recorded_artifact",
        source=source,
        conditions=conditions,
        global_costs=global_costs,
    )
