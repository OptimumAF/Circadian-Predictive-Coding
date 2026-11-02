"""Pure recorded TRAIN readout playback; no learner, IO or scientific scoring.

Checks saved consistency, not authenticity. Missing/failed evidence stays visible;
all frame and outcome links address the copied original JSON.
"""

from copy import deepcopy
import json
from typing import Any

from .minigrid_inspector_projection import (
    _envelope,
    _episode,
    _integer,
    _number,
    _records,
    _require,
    _status,
    _counts,
)
from .minigrid_learnability_inspector import _flatten

DEFINITION = "recurrent_readout_train_ABCA_native_v1"
METHOD = "recurrent_readout_fit"
CONTEXTS = ("A", "B", "C_train")
POINTS = (2, 4, 6, 8)
LAYOUTS = {
    "A": dict(context="A", rotation=0, branch_row=3, swap_keys=False, wall_color="grey"),
    "B": dict(context="B", rotation=1, branch_row=4, swap_keys=True, wall_color="blue"),
    "C_train": dict(context="C_train", rotation=2, branch_row=3, swap_keys=True, wall_color="red"),
}


def _object(value, name):
    _require(type(value) is dict, f"invalid {name}")
    return value


def _flag(value, name):
    _require(value is None or type(value) is bool, f"invalid {name}")


def _budget(record, name, complete):
    if record is None:
        _require(not complete, f"complete report lacks {name}")
        return
    _object(record, name)
    limits = _object(record.get("limits"), "budget limits")
    readout = name == "readout_budget"
    for key, expected in dict(
        max_training_updates=0 if readout else 8,
        max_wall_seconds=90 if readout else 60,
        max_process_rss_bytes=512 * 1024**2,
    ).items():
        _require(type(limits.get(key)) is int and limits[key] == expected, "budget limits differ")
    _require(
        limits.get("max_replay_examples") == (0 if readout else None), "replay authority differs"
    )
    if readout:
        _require(type(limits["max_replay_examples"]) is int, "invalid replay authority")
    birth = _number(record.get("birth"), "budget birth", 0)
    last = _number(record.get("last_clock"), "budget last clock", birth)
    updates = _integer(record.get("updates"), 0, 0 if readout else 8, "charged fits")
    _integer(record.get("replay"), 0, 0, "replay charges")
    if complete:
        _require(last - birth <= limits["max_wall_seconds"], "complete budget exceeded")
        _require(updates == (0 if readout else 8), "completed charges differ")


def _summary(report, episodes):
    summary = report.get("outcomes")
    if summary is None:
        _require(report["status"] != "complete", "complete report lacks outcomes")
        return
    _object(summary, "outcomes")
    observed = {
        (e["updates"], e["context"], e["cue"]): e["record"]["success"]
        for e in episodes
        if e["record"]["status"] == "complete"
    }
    _require(len(observed) == 18, "outcome summary requires all complete observations")
    for key, roster in (
        ("acquisition", [(p, c, cue) for p, c in zip(POINTS, CONTEXTS) for cue in ("blue", "red")]),
        (
            "retention",
            [
                (p, c, cue)
                for p, current in ((4, "B"), (6, "C_train"), (8, "A"))
                for c in CONTEXTS[: min(p, 6) // 2]
                if c != current
                for cue in ("blue", "red")
            ],
        ),
    ):
        rows = _records(summary.get(key), len(roster), key)
        _require(len(rows) == len(roster), "incomplete outcome roster")
        for row, (p, c, cue) in zip(rows, roster, strict=True):
            _require(
                type(row.get("updates")) is int
                and row["updates"] == p
                and row.get("context") == c
                and row.get("cue") == cue
                and row.get("success") is observed[p, c, cue],
                "outcome differs from episode",
            )
    rows = _records(summary.get("returning_A"), 2, "returning A")
    _require(len(rows) == 2, "missing returning cues")
    for row, cue in zip(rows, ("blue", "red"), strict=True):
        acquired, returned = observed[2, "A", cue], observed[8, "A", cue]
        lost = any(observed[p, "A", cue] is False for p in (4, 6))
        last_lost = observed[6, "A", cue] is False
        expected = dict(
            cue=cue,
            initially_acquired=acquired,
            failed_while_absent=lost,
            failed_before_return=last_lost,
            return_success=returned,
            observed_recovery_by_return=acquired and lost and returned,
            observed_recovery=acquired and last_lost and returned,
        )
        _require(
            row.keys() == expected.keys()
            and all(type(row[k]) is type(v) and row[k] == v for k, v in expected.items()),
            "return recovery contradicts observed history",
        )


def _phase_identity(phase, games, point, training):
    """Why: a protocol label cannot override conflicting paid phase evidence."""
    for key, expected in dict(
        method=METHOD,
        role="train",
        updates=point,
        readout_fit_updates=point,
        readout_fit_rows=9 * point,
        native_recurrent_updates=0,
        native_recurrent_training_rows=0,
        arrived_contexts=list(CONTEXTS[: min(point, 6) // 2]),
    ).items():
        _require(
            type(games.get(key)) is type(expected) and games[key] == expected,
            f"phase {key} differs",
        )
    # The actual failure wrapper lacks this successful-phase field. Its missing
    # prefix is unknown; top-level charges cannot manufacture a phase receipt.
    if "training_receipts" not in phase and games["status"] != "complete":
        return
    receipts = _records(phase.get("training_receipts"), 8, "phase receipts")
    _require(len(receipts) == point, "phase receipt count differs")
    for i, receipt in enumerate(receipts):
        for key, expected in dict(
            update_number=i + 1,
            observed_at=i,
            arrived_at=i + 1,
            applied_at=i + 1,
            actor_version="a0",
            learner_version="c0",
        ).items():
            _require(
                type(receipt.get(key)) is type(expected) and receipt[key] == expected,
                "phase receipt chronology differs",
            )
    if training is not None:
        # JSON encoding preserves bool/int/float distinctions that Python == loses.
        _require(
            json.dumps(receipts, sort_keys=True)
            == json.dumps(training.get("applied_updates", [])[:point], sort_keys=True),
            "phase paid receipt prefix differs",
        )


def project_recurrent_readout_report(
    report: dict[str, Any], *, source_label: str
) -> dict[str, Any]:
    """Validate fixed recipe and saved physics, returning detached source links."""
    _envelope(report, source_label)
    _require(_status(report.get("status")), "invalid run status")
    complete = report["status"] == "complete"
    protocol = _object(report.get("protocol"), "protocol")
    for key, expected in dict(
        definition=DEFINITION,
        context_sequence=["A", "B", "C_train", "A"],
        fit_points=list(POINTS),
        readout_contexts=[["A"], ["A", "B"], list(CONTEXTS), list(CONTEXTS)],
        final_available=False,
        probe_available=False,
    ).items():
        _require(
            type(protocol.get(key)) is type(expected) and protocol[key] == expected,
            f"unsupported {key}",
        )
    limits = _object(protocol.get("limits"), "protocol limits")
    _require(all(type(point) is int for point in protocol["fit_points"]), "invalid fit point types")
    for key, expected in dict(
        body_seconds=120, training_seconds=60, readout_seconds=90, rss_bytes=512 * 1024**2
    ).items():
        _require(type(limits.get(key)) is int and limits[key] == expected, "protocol limits differ")
    _require(report.get("reason") is None or type(report["reason"]) is str, "invalid reason")
    for key in ("seconds", "peak_bytes"):
        if report.get(key) is not None:
            if key == "peak_bytes":
                _integer(report[key], 0, 2**53 - 1, key)
            else:
                _number(report[key], key, 0)
    if complete:
        _number(report.get("seconds"), "complete seconds", 0, 120)
        _integer(report.get("peak_bytes"), 1, 512 * 1024**2, "complete RSS")
    preservation = report.get("preservation")
    _require(not complete or type(preservation) is dict, "complete report lacks preservation")
    if preservation is not None:
        _object(preservation, "preservation")
        _require(
            set(preservation) == {"source", "actor", "torch_rng", "numpy_rng", "python_rng"},
            "preservation identities differ",
        )
        for value in preservation.values():
            _flag(value, "preservation")
            _require(not complete or value is True, "complete conservation proof missing")
    for key in ("training_budget", "readout_budget"):
        _budget(report.get(key), key, complete)
    training = report.get("training")
    _require(not complete or type(training) is dict, "complete report lacks training")
    if training is not None:
        _object(training, "training")
        receipts = _records(training.get("applied_updates", []), 8, "receipts")
        for index, row in enumerate(receipts):
            for key, expected in dict(
                update_number=index + 1,
                observed_at=index,
                arrived_at=index + 1,
                applied_at=index + 1,
                actor_version="a0",
                learner_version="c0",
            ).items():
                _require(
                    type(row.get(key)) is type(expected) and row[key] == expected,
                    "receipt chronology/version differs",
                )
        _require(not complete or len(receipts) == 8, "complete report lacks all receipts")
        if complete:
            for key, expected in dict(completed_updates=8, training_examples=72).items():
                _require(
                    type(training.get(key)) is int and training[key] == expected,
                    "fit totals differ",
                )
    records = _records(report.get("readouts", []), 4, "readouts")
    _require(not complete or len(records) == 4, "complete report lacks phases")
    episodes, scores, costs = [], [], []
    for i, point in enumerate(POINTS):
        phase = records[i] if i < len(records) else None
        roster = [(c, cue) for c in CONTEXTS[: min(point, 6) // 2] for cue in ("blue", "red")]
        games = None
        rows = []
        if phase is not None:
            _require(
                type(phase.get("updates")) is int and phase["updates"] == point,
                "phase order differs",
            )
            games = _object(phase.get("games"), "games")
            _phase_identity(phase, games, point, training)
            _require(_status(games.get("status")), "invalid game phase status")
            _flag(games.get("state_unchanged"), "phase preservation")
            _require(
                not complete or games["status"] == "complete", "incomplete phase in complete run"
            )
            _require(
                games["status"] != "complete" or games.get("state_unchanged") is True,
                "complete phase lacks preservation",
            )
            rows = _records(games.get("episodes", []), len(roster), "game slots")
            _counts(games.get("work"))
            _require(
                games["status"] != "complete"
                or (len(rows) == len(roster) and all(r.get("status") == "complete" for r in rows)),
                "missing complete games",
            )
            score_rows = _records(games.get("scores", []), len(roster), "scores")
            _require(
                games["status"] != "complete" or len(score_rows) == len(roster),
                "missing complete scores",
            )
            for j, score in enumerate(score_rows):
                c, cue = roster[j]
                _require(
                    score.get("context") == c and score.get("cue") == cue, "score order differs"
                )
                for key, high in (("ce", None), ("accuracy", 1), ("exact_sequence_accuracy", 1)):
                    if score.get(key) is not None:
                        _number(score[key], key, 0, high)
                scores.append(
                    dict(
                        pointer=f"/readouts/{i}/games/scores/{j}",
                        updates=point,
                        record=deepcopy(score),
                    )
                )
        for j, (context, cue) in enumerate(roster):
            episode_record = rows[j] if j < len(rows) else None
            pointer = f"/readouts/{i}/games/episodes/{j}" if episode_record is not None else None
            entry: dict[str, Any]
            if episode_record is None:
                entry = dict(
                    pointer=None,
                    record=dict(
                        status="not_run", context=context, cue=cue, success=None, **{"return": None}
                    ),
                    frames=[],
                    initial_image=None,
                    updates=point,
                )
            else:
                assert pointer is not None
                entry = _episode(
                    episode_record,
                    pointer,
                    METHOD,
                    point,
                    cue,
                    context=context,
                    expected_layout=LAYOUTS[context],
                    allow_observed_failure_tail=True,
                )
            entry.update(
                context=context,
                phase=("A", "B", "C_train", "return A")[i],
                derived=episode_record is None,
            )
            episodes.append(entry)
        if games is not None:
            costs += _flatten(
                {k: v for k, v in games.items() if k in ("work", "state_unchanged")},
                f"/readouts/{i}/games",
                f"Fit {point} readout",
            )
    _summary(report, episodes)
    costs += _flatten(
        {
            k: report[k]
            for k in (
                "training_budget",
                "readout_budget",
                "seconds",
                "peak_bytes",
                "preservation",
                "failure_work",
                "readback_error",
            )
            if k in report
        },
        "",
        "Recorded app scope",
    )
    return dict(
        schema_id="recurrent_readout_inspector_v1",
        origin="recorded_artifact",
        source_label=source_label,
        source=deepcopy(report),
        conditions=[
            dict(
                key=METHOD,
                pointer="",
                status=report["status"],
                episodes=episodes,
                scores=scores,
                costs=costs,
            )
        ],
    )
