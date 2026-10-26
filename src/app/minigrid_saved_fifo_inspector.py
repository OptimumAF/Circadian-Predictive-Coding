"""Project saved FIFO36 F16/A16 evidence without running a learner or environment.

Inputs are bounded recorded JSON. Outputs own the original source and expose exact
JSON pointers, planned unknowns, charges and signed changes. Validation establishes
internal consistency, not authentication of external history or scientific benefit.
"""

from copy import deepcopy
from typing import Any, cast

from .minigrid_inspector_projection import (
    _boolean,
    _close,
    _envelope,
    _episode,
    _integer,
    _number,
    _records,
    _require,
    _status,
    decode_public_observation,
)
from .minigrid_learnability_inspector import _flatten, _proofs, _score

PROTOCOL_ID = "minigrid_saved_invariant_fifo36_F16_A16_v1"
METHOD = "public_invariant_observation_memory_fifo"
POINTS = (0, 4, 8, 16)
POLICY = dict(
    method="observation_memory_fifo",
    feature_dim=296,
    num_classes=7,
    max_entries=36,
    max_payload_bytes=87264,
)
ENCODING = "minigrid_public_wall_direction_invariant_history_296_v1"
METRICS = (
    "new_F_gain",
    "old_A_retention_loss",
    "old_A_return_gain",
    "new_F_retention_loss",
)

FINAL_SOURCE_SHA = "4add6c431b7911ce2c516bfbbe3667bb6bee4331e13255fd6065d159cf2b98b7"
FINAL_LAYOUT = dict(context="D", rotation=3, branch_row=4, swap_keys=False, wall_color="yellow")
FINAL_ZERO_WORK = (
    "new_training_updates",
    "new_training_rows",
    "replay_examples",
    "probe_updates",
    "teacher_episodes",
    "promotions",
)
FINAL_PROOFS = (
    "python_rng_unchanged",
    "numpy_rng_unchanged",
    "source_unchanged",
    "input_bytes_unchanged",
)


def _final_identity(report: dict[str, Any]) -> bool:
    complete = _boolean(report.get("complete"), "final completion")
    for key, value in (
        ("role", "final_test"),
        ("source_sha256", FINAL_SOURCE_SHA),
        ("benchmark_updates", 128),
        ("benchmark_rows", 1152),
    ):
        _fixed(report, key, value)
    for key in FINAL_ZERO_WORK:
        _fixed(report, key, 0)
    for key in FINAL_PROOFS:
        if report.get(key) is not None:
            _boolean(report[key], key)
        _require(not complete or report.get(key) is True, "complete final lacks preservation proof")
    return complete


def _final_cost_identity(report: dict[str, Any], complete: bool) -> None:
    for key in ("origin", "memory", "readout_budget", "sampled_process_rss", "elapsed_seconds"):
        _require(not complete or key in report, f"complete final lacks {key}")
    if "origin" in report:
        origin = _object(report["origin"], "original history")
        _fixed(origin, "training_updates", 128)
        _fixed(origin, "training_rows", 1152)
        _fixed(origin, "status", "incomplete")
        _number(origin.get("elapsed_seconds"), "original elapsed seconds", 0)
    if "memory" in report:
        memory = _object(report["memory"], "final memory")
        policy = _object(memory.get("policy"), "encoded final policy")
        _policy(policy.get("native_policy"))
        _fixed(policy, "encoding_id", ENCODING)
        for key, value in (
            ("entries", 36),
            ("numeric_table_bytes", 87264),
            ("retirement_metadata_bytes", 16),
            ("evictions", 36),
            ("retired_examples", 576),
        ):
            _fixed(memory, key, value)
    _final_budget(report, complete)


def _final_budget(report: dict[str, Any], complete: bool) -> None:
    elapsed = None
    if "elapsed_seconds" in report:
        elapsed = _number(report["elapsed_seconds"], "final elapsed seconds", 0)
    budget = report.get("readout_budget")
    if budget is not None:
        budget = _object(budget, "final readout budget")
        limits = _object(budget.get("limits"), "final budget limits")
        for key, value in (
            ("max_training_updates", 0),
            ("max_replay_examples", 0),
            ("max_wall_seconds", 120),
            ("max_process_rss_bytes", 805306368),
            ("max_hidden_width", None),
        ):
            _fixed(limits, key, value)
        _fixed(budget, "updates_completed", 0)
        _fixed(budget, "replay_examples_completed", 0)
        start = _number(budget.get("started_at"), "budget start", 0)
        end = _number(budget.get("last_clock"), "budget clock", start)
        _require(not complete or end - start <= 120, "complete final exceeded clock allowance")
        _require(elapsed is None or end - start <= elapsed + 1e-6, "budget clock exceeds body time")
    if "sampled_process_rss" in report:
        rss = _object(report["sampled_process_rss"], "sampled final RSS")
        start_bytes = _integer(rss.get("start_bytes"), 0, 2**63, "starting RSS")
        peak = _integer(rss.get("peak_bytes"), start_bytes, 2**63, "peak RSS")
        _integer(rss.get("pid"), 1, 2**63, "RSS process")
        _integer(rss.get("sample_count"), 1, 2**63, "RSS samples")
        _number(rss.get("interval_seconds"), "RSS sample interval", 0)
        _require(not complete or peak <= 805306368, "complete final exceeded RSS allowance")
    _require(
        not complete or elapsed is not None and elapsed <= 120,
        "complete final exceeded wall allowance",
    )


def _final_work(game: dict[str, Any]) -> None:
    work = _object(game.get("work"), "final game work")
    complete = game["status"] == "complete"
    unknown = _boolean(work.get("failure_work_may_be_unobserved"), "unobserved final work")
    _require(not complete or not unknown, "complete game has unobserved work")
    episodes = game["episodes"]
    environments = [e["environment"] for e in episodes if e.get("environment") is not None]
    attempts = sum(e["prediction_calls_attempted"] for e in episodes)
    returned = sum(e["prediction_calls_returned"] for e in episodes)
    expected = dict(
        live_prediction_calls_attempted=attempts,
        live_prediction_calls_returned=returned,
        environment_resets_attempted=len(environments),
        environment_resets_completed=len(environments),
        environment_steps_attempted=sum(e["environment_steps_attempted"] for e in environments),
        environment_steps_returned=sum(e["environment_steps"] for e in environments),
        internal_projection_rows_returned=returned,
        internal_projection_array_bytes_returned=returned * 296 * 8,
        history_episodes_started=len(environments),
        history_live_feature_rows=attempts,
        history_live_feature_array_bytes=attempts * 296 * 8,
    )
    for key in (
        "raw_score_forwards_attempted",
        "raw_score_forwards_completed",
        "raw_score_rows_attempted",
        "raw_score_rows_completed",
        "live_sessions_started",
        "history_raw_episodes_encoded",
        "history_raw_feature_array_bytes",
        "history_raw_target_array_bytes",
    ):
        _fixed(work, key, 0)
    for key, observed in expected.items():
        lower = observed
        upper = observed
        if unknown and not complete:
            if key.startswith("environment_resets"):
                upper = 2
            elif key == "history_episodes_started":
                lower, upper = max(0, observed - 1), 2
            elif key.startswith("history_live_feature"):
                unit = 296 * 8 if key.endswith("bytes") else 1
                lower = max(0, observed - unit)
            elif not environments and key.startswith("environment_steps"):
                upper = 32
        _integer(work.get(key), lower, upper, key)
    for rows, size in (
        ("history_live_feature_rows", "history_live_feature_array_bytes"),
        ("internal_projection_rows_returned", "internal_projection_array_bytes_returned"),
    ):
        _require(work[size] == work[rows] * 296 * 8, "final feature row/byte charges disagree")
    _require(
        work["environment_resets_completed"] <= work["environment_resets_attempted"],
        "final returned resets exceed attempts",
    )


def _final_game(game: dict[str, Any], pointer: str, complete: bool) -> list[dict[str, Any]]:
    for key, value in (
        ("updates", 128),
        ("role", "final_test"),
        ("method", METHOD),
        ("scores", []),
        ("native_training_examples", 1152),
        ("model_encoding", ENCODING),
        ("native_evictions", 36),
        ("native_retired_examples", 576),
        ("native_retained_entries", 36),
        ("native_retained_logical_payload_bytes", 87264),
        ("native_retirement_metadata_bytes", 16),
    ):
        _fixed(game, key, value)
    _require(_status(game.get("status")), "invalid final game status")
    _require(not complete or game["status"] == "complete", "complete final has partial games")
    if game.get("state_unchanged") is not None:
        _boolean(game["state_unchanged"], "final game state unchanged")
    _require(
        not complete or game.get("state_unchanged") is True, "complete final has changed game state"
    )
    history = _object(game.get("public_history"), "final public history")
    for key, value in (
        ("enabled", True),
        ("applied", True),
        ("observation_dim", 148),
        ("prediction_feature_dim", 296),
        ("initial_numeric_bytes", 1184),
    ):
        _fixed(history, key, value)
    rows = _records(game.get("episodes"), 2, "final episode slots")
    _require(len(rows) == 2, "final requires both planned cue slots")
    entries = []
    for i, cue in enumerate(("blue", "red")):
        entry = _episode(
            rows[i],
            f"{pointer}/episodes/{i}",
            METHOD,
            128,
            cue,
            context="D",
            role="final_test",
            expected_layout=FINAL_LAYOUT,
            allow_observed_failure_tail=True,
        )
        _require(
            game["status"] != "complete" or rows[i]["status"] == "complete",
            "complete game lacks cue",
        )
        entry.update(context="D", source_role="final_test")
        entries.append(entry)
    _final_work(game)
    return entries


def _final_queries(report: dict[str, Any], entries: list[dict[str, Any]], complete: bool) -> None:
    if "query_matches" not in report:
        _require(
            not complete and "matched_queries" not in report and "total_queries" not in report,
            "missing final query diagnostics",
        )
        return
    rows = _records(report["query_matches"], 32, "recorded query matches")
    predictions = [
        (e["cue"], i, p)
        for e in entries
        for i, p in enumerate(e["record"].get("predictions", []))
        if p.get("probabilities") is not None
    ]
    _require(len(rows) == len(predictions), "query match count contradicts decisions")
    matched = 0
    for row, (cue, step, prediction) in zip(rows, predictions, strict=True):
        for key, value in (("cue", cue), ("step", step), ("action", prediction["action"])):
            _fixed(row, key, value)
        flag = _boolean(row.get("matched"), "stored key match")
        matched += flag
        _require(
            flag or all(abs(p - 1 / 7) <= 1e-12 for p in prediction["probabilities"]),
            "unmatched key must retain uniform fallback",
        )
    _fixed(report, "total_queries", len(rows))
    _fixed(report, "matched_queries", matched)


def project_minigrid_heldout_fifo_report(report: Any, *, source_label: str) -> dict[str, Any]:
    """Read final D records only. Never constructs a learner, fits or selects a probe.

    Why separate: the existing F/A schema owns disposable probe learning, whereas
    this fixed benchmark128 record contains final observations and no fitting.
    Consistency checks do not authenticate a supplied record or imply benefit.
    """
    _envelope(report, source_label)
    # Keep this new source smaller than the shared legacy envelope.
    import json

    _require(
        len(json.dumps(report, ensure_ascii=False).encode("utf8")) <= 4 * 1024 * 1024,
        "final FIFO source exceeds four MiB",
    )
    complete = _final_identity(report)
    _final_cost_identity(report, complete)
    _require(not ("games" in report and "failure_report" in report), "ambiguous final game source")
    key = "games" if "games" in report else "failure_report"
    game = report.get(key)
    _require(not complete or game is not None, "complete final lacks games")
    entries = (
        _final_game(_object(game, "final games"), f"/{key}", complete) if game is not None else []
    )
    if not entries:
        for cue in ("blue", "red"):
            entries.append(
                dict(
                    pointer=None,
                    updates=128,
                    cue=cue,
                    context="D",
                    derived=True,
                    source_role="final_test",
                    frames=[],
                    initial_image=None,
                    record=dict(
                        cue=cue,
                        context="D",
                        source_role="final_test",
                        status="not_run",
                        success=None,
                        **{"return": None},
                    ),
                )
            )
    _final_queries(report, entries, complete)
    proofs = [
        dict(label=name, value=report.get(name), pointer=f"/{name}" if name in report else None)
        for name in FINAL_PROOFS
    ]
    if game is not None:
        proofs.append(
            dict(
                label="game state_unchanged",
                value=game.get("state_unchanged"),
                pointer=f"/{key}/state_unchanged" if "state_unchanged" in game else None,
            )
        )
    costs = []
    for name in (
        *FINAL_ZERO_WORK,
        "memory",
        "readout_budget",
        "sampled_process_rss",
        "elapsed_seconds",
    ):
        if name in report:
            costs.extend(_flatten({name: report[name]}, ""))
    if game is not None:
        costs.extend(_flatten(game["work"], f"/{key}/work", "final work"))
        costs.extend(_flatten({"public_history": game["public_history"]}, f"/{key}"))
        if "readout_scope" in game:
            costs.extend(_flatten({"readout_scope": game["readout_scope"]}, f"/{key}"))
    return dict(
        schema_id="minigrid_heldout_fifo_inspector_v1",
        source_label=source_label,
        origin="recorded_artifact",
        source=deepcopy(report),
        conditions=[
            dict(
                key=METHOD,
                pointer=f"/{key}" if game is not None else "",
                status="complete" if complete else "incomplete",
                source_role="final_test",
                episodes=entries,
                curves=[],
                proofs=proofs,
                costs=costs,
                unknown="No TRAIN or probe outcome scores are recorded in this artifact.",
            )
        ],
    )


def _object(value: Any, name: str) -> dict[str, Any]:
    _require(type(value) is dict, f"invalid {name} object")
    return value


def _fixed(record: dict[str, Any], key: str, expected: Any) -> None:
    value = record.get(key)
    _require(type(value) is type(expected) and value == expected, f"unsupported fixed {key}")


def _points(value: Any, expected: tuple[int, ...], name: str) -> None:
    _require(
        type(value) is list
        and value == list(expected)
        and all(type(point) is int for point in value),
        f"invalid {name}",
    )


def _policy(value: Any) -> None:
    policy = _object(value, "FIFO policy")
    _require(set(policy) == set(POLICY), "unsupported FIFO policy fields")
    for key, expected in POLICY.items():
        _fixed(policy, key, expected)


def _protocol(report: dict[str, Any]) -> None:
    protocol = _object(report.get("protocol"), "protocol")
    for key, expected in (
        ("protocol_id", PROTOCOL_ID),
        ("method", METHOD),
        ("seed", 17),
        ("benchmark_updates", 0),
        ("retirement_metadata_bytes", 16),
    ):
        _fixed(protocol, key, expected)
    _points(protocol.get("source_updates"), (0, 128), "source checkpoints")
    _points(protocol.get("native_training_examples_before"), (0, 1152), "source rows")
    _points(protocol.get("checkpoint_updates"), POINTS, "relative checkpoints")
    _policy(protocol.get("policy"))
    _require(
        protocol.get("phases")
        == [
            dict(name="F", updates=16, role="probe_train"),
            dict(name="A_return", updates=16, role="train"),
        ],
        "unsupported phase roles or updates",
    )
    for phase in protocol["phases"]:
        _fixed(phase, "updates", 16)
    expected_layouts = [
        dict(context=f"F-{cue}", rotation=0, branch_row=2, swap_keys=False, wall_color="grey")
        for cue in ("blue", "red")
    ]
    _require(protocol.get("probe_layouts") == expected_layouts, "unsupported reserved F layouts")
    _require(report.get("scientific_advantage_verified") is False, "unverified scientific claim")
    planned = _object(protocol.get("planned_work"), "planned work")
    for key, expected in (
        ("maximum_storage_updates", 64),
        ("maximum_training_rows", 576),
        ("maximum_score_calls", 64),
        ("maximum_score_rows", 576),
        ("teacher_episodes", 4),
        ("expected_teacher_steps", 36),
        ("maximum_teacher_steps", 64),
        ("benchmark_updates", 0),
        ("replay_examples", 0),
    ):
        _fixed(planned, key, expected)
    final = report.get("final_evaluation")
    _require(
        final is None or final == {"status": "not_run", "role": "final_test"},
        "final evaluation must remain unavailable",
    )


def _episode_data(episode: dict[str, Any], identity: str, context: str, role: str) -> list[int]:
    _fixed(episode, "episode_id", identity)
    _fixed(episode, "context", context)
    _fixed(episode, "role", role)
    _fixed(episode, "num_classes", 7)
    steps = _records(episode.get("transitions"), 9, "episode transitions")
    _require(len(steps) == 9, "source DTO requires nine transitions")
    actions, prior = [], None
    for index, step in enumerate(steps):
        _fixed(step, "step", index)
        _fixed(step, "num_classes", 7)
        decode_public_observation(step.get("observation"))
        decode_public_observation(step.get("next_observation"))
        _require(prior is None or prior == step["observation"], "source continuity mismatch")
        _require(
            _boolean(step.get("terminal"), "source terminal") == (index == 8),
            "source terminal mismatch",
        )
        _number(step.get("reward"), "source reward", -1, 1)
        actions.append(_integer(step.get("action"), 0, 6, "source action"))
        prior = step["next_observation"]
    return actions


def _data(report: dict[str, Any], complete: bool) -> dict[str, list[int]]:
    sources = _records(report.get("return_sources", []), 2, "original A sources")
    _require(len(sources) == 2, "original A TRAIN sources are required")
    targets = {}
    for source, cue in zip(sources, ("blue", "red"), strict=True):
        targets[f"minigrid-source-A-{cue}:saved-monitor"] = _episode_data(
            source, f"minigrid-source-A-{cue}", "A", "train"
        )
    data = _records(report.get("probe_data", []), 22, "reserved source DTOs")
    _require(not complete or len(data) == 22, "complete report requires all reserved source DTOs")
    identities = (
        [
            (
                f"minigrid-probe-source-probe_train-{('blue', 'red')[i % 2]}:copy:{i}",
                f"F-{('blue', 'red')[i % 2]}",
                "probe_train",
            )
            for i in range(16)
        ]
        + [
            (f"minigrid-probe-source-probe_test-{cue}", f"F-{cue}", "probe_test")
            for cue in ("blue", "red")
        ]
        + [
            (f"minigrid-source-A-{cue}:saved-{label}", f"A-{cue}", "train")
            for label in ("monitor", "return")
            for cue in ("blue", "red")
        ]
    )
    for index, episode in enumerate(data):
        identity, context, role = identities[index]
        actions = _episode_data(episode, identity, context, role)
        if index >= 18:
            original = sources[(index - 18) % 2]
            _require(
                episode["transitions"] == original["transitions"],
                "A DTO differs from original TRAIN",
            )
        elif index < 16 and index >= 2:
            _require(
                episode["transitions"] == data[index % 2]["transitions"],
                "storage copy changed its labels",
            )
        if role == "probe_test":
            targets[context] = actions
    teachers = _records(report.get("source_reports", []), 4, "teacher reports")
    _require(not complete or len(teachers) == 4, "complete report requires four teacher reports")
    for index, teacher in enumerate(teachers):
        cue = ("blue", "red")[index % 2]
        role = "probe_train" if index < 2 else "probe_test"
        for key, expected in (
            ("seed", 17),
            ("max_steps", 16),
            ("role", role),
            ("context", f"F-{cue}"),
            ("cue", cue),
            ("num_classes", 7),
            ("observation_dim", 148),
        ):
            _fixed(teacher, key, expected)
        _require(_status(teacher.get("status")), "invalid teacher status")
        if complete:
            _require(
                teacher.get("status") == "complete" and teacher.get("closed") is True,
                "unfinished complete teacher",
            )
    return targets


def _origin(report: dict[str, Any]) -> None:
    origin = _object(report.get("origin"), "original failed run")
    for key, expected in (
        ("status", "incomplete"),
        ("training_updates", 128),
        ("training_rows", 1152),
        ("budget_status", "spent; not restored"),
    ):
        _fixed(origin, key, expected)
    _require(
        type(origin.get("reason")) is str and bool(origin["reason"]),
        "original failure reason missing",
    )
    _number(origin.get("elapsed_seconds"), "original elapsed seconds", 0)
    censored = origin.get("censored_games")
    _require(
        censored == [[64, "B", "blue"]] and type(censored[0][0]) is int,
        "original censored attempt changed",
    )


def _snapshot(value: Any, source_updates: int, charged: int | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    snapshot = _object(value, "end snapshot")
    policy = _object(snapshot.get("policy"), "snapshot policy")
    _policy(policy.get("native_policy"))
    _fixed(policy, "encoding_id", ENCODING)
    native = _object(snapshot.get("native_state"), "native snapshot")
    _policy(native.get("policy"))
    updates = _integer(
        native.get("completed_updates"), source_updates, source_updates + 32, "native clock"
    )
    rows = _integer(native.get("training_examples"), 0, 1440, "native rows")
    _require(
        rows == 9 * updates and (charged is None or updates == source_updates + charged),
        "native clock/charge mismatch",
    )
    entries = _records(native.get("entries"), 36, "FIFO entries")
    retained = 0
    seen = set()
    for entry in entries:
        features, counts = entry.get("features"), entry.get("counts")
        _require(type(features) is list and len(features) == 296, "invalid FIFO feature size")
        key = tuple(_number(x, "FIFO feature", -1, 1) for x in cast(list[Any], features))
        _require(key not in seen, "duplicate FIFO feature")
        seen.add(key)
        _require(type(counts) is list and len(counts) == 7, "invalid FIFO action counts")
        total = sum(_integer(x, 0, 1440, "FIFO count") for x in cast(list[Any], counts))
        _require(total > 0, "empty FIFO entry")
        retained += total
    retired = _integer(native.get("retired_examples"), 0, 1440, "retired rows")
    _integer(native.get("evictions"), 0, 1440, "evictions")
    _require(retained + retired == rows, "FIFO retained/retired rows mismatch")
    return _flatten(
        {
            key: native[key]
            for key in ("completed_updates", "training_examples", "evictions", "retired_examples")
        },
        "",
    )


def _counters(record: dict[str, Any], pairs: tuple[tuple[str, str, int], ...]) -> None:
    for attempted, completed, maximum in pairs:
        if record.get(attempted) is None and record.get(completed) is None:
            continue
        attempt = _integer(record.get(attempted), 0, maximum, attempted)
        done = _integer(record.get(completed), 0, maximum, completed)
        _require(done <= attempt, "returned work exceeds attempted work")


def _record_proofs(record: dict[str, Any], pointer: str) -> list[dict[str, Any]]:
    proofs = _proofs(record, pointer)
    covered = {proof["label"] for proof in proofs}
    for key, value in record.items():
        if key not in covered and key.endswith(("unknown", "may_be_unobserved")):
            if value is not None:
                _boolean(value, key)
            proofs.append(dict(label=key, value=value, pointer=f"{pointer}/{key}"))
    return proofs


def _curves(
    container: dict[str, Any],
    pointer: str,
    source: int,
    phase: str,
    context: str,
    targets: dict[str, list[int]],
) -> list[dict[str, Any]]:
    checkpoints = _records(container.get("checkpoints", []), 4, "curve checkpoints")
    _require(
        [p.get("updates") for p in checkpoints] == list(POINTS[: len(checkpoints)]),
        "curve checkpoint order mismatch",
    )
    rows = []
    for index, point in enumerate(checkpoints):
        _fixed(point, "updates", POINTS[index])
        _require(_status(point.get("status", "complete")), "invalid checkpoint status")
        scores = _object(point.get("scores"), "checkpoint scores")
        details = _object(point.get("decision_diagnostics", {}), "checkpoint diagnostics")
        names = [
            f"F-{cue}" if context == "F" else f"minigrid-source-A-{cue}:saved-monitor"
            for cue in ("blue", "red")
        ]
        _require(
            set(scores) <= set(names) and set(details) <= set(scores),
            "unsupported score context or role",
        )
        if container.get("status") == "complete":
            _require(set(scores) == set(names), "complete checkpoint requires both cue scores")
        for cue, name in zip(("blue", "red"), names, strict=True):
            if name not in scores:
                continue
            score = _object(scores[name], "score")
            ce, accuracy, exact = _score(score, details.get(name))
            if container.get("status") == "complete":
                _require(
                    ce is not None and accuracy is not None, "complete score missing measurement"
                )
            diagnostic = details.get(name)
            if diagnostic is not None:
                identity = f"minigrid-probe-source-probe_test-{cue}" if context == "F" else name
                _fixed(diagnostic, "episode_id", identity)
                _fixed(diagnostic, "sequence_kind", "complete_episode")
                _require(
                    diagnostic.get("row_ids") == [f"{identity}:step:{i}" for i in range(9)],
                    "diagnostic row identity mismatch",
                )
                if context == "A":
                    _fixed(diagnostic, "role", "train")
                    _fixed(diagnostic, "context", f"A-{cue}")
                else:
                    if "role" in diagnostic:
                        _fixed(diagnostic, "role", "probe_test")
                    if "context" in diagnostic:
                        _fixed(diagnostic, "context", f"F-{cue}")
            if diagnostic is not None and name in targets:
                _require(
                    [d["target_class"] for d in diagnostic["decisions"]] == targets[name],
                    "diagnostic target/source mismatch",
                )
            rows.append(
                dict(
                    source_updates=source,
                    phase=phase,
                    updates=point["updates"],
                    context=f"F-{cue}" if context == "F" else "A",
                    cue=cue,
                    role="probe_test" if context == "F" else "train",
                    status=point.get("status", "complete"),
                    ce=ce,
                    accuracy=accuracy,
                    exact_sequence_accuracy=exact,
                    pointer=f"{pointer}/checkpoints/{index}/scores/{name}",
                )
            )
    if container.get("status") == "complete":
        _require(len(checkpoints) == 4, "complete phase requires four checkpoints")
    observed = {(row["updates"], row["cue"]) for row in rows}
    for updates in POINTS:
        for cue in ("blue", "red"):
            if (updates, cue) not in observed:
                rows.append(
                    dict(
                        source_updates=source,
                        phase=phase,
                        updates=updates,
                        context=f"F-{cue}" if context == "F" else "A",
                        cue=cue,
                        role="probe_test" if context == "F" else "train",
                        status="not_run",
                        ce=None,
                        accuracy=None,
                        exact_sequence_accuracy=None,
                        pointer=None,
                        derived=True,
                    )
                )
    return rows


def _phase(record: dict[str, Any], charge: int | None, returning: bool) -> None:
    if not record:
        return
    _require(_status(record.get("status")), "invalid phase status")
    _fixed(record, "num_classes", 7)
    _counters(
        record,
        (
            ("attempted_updates", "completed_updates", 16),
            ("attempted_training_examples", "completed_training_examples", 144),
            ("score_forward_calls_attempted", "score_forward_calls_completed", 8),
        ),
    )
    completed = record.get("completed_updates")
    if completed is not None:
        _require(
            record.get("completed_training_examples") == 9 * completed, "phase native rows mismatch"
        )
        _require(
            charge is None or completed <= charge, "phase returned work exceeds permanent charge"
        )
    _checkpoint_work(record, completed)
    if record["status"] == "complete":
        _require(
            completed == 16 and charge == 16 and record.get("score_forward_calls_completed") == 8,
            "complete phase clocks or charge mismatch",
        )
    if returning:
        _points(record.get("checkpoint_updates"), POINTS, "return checkpoints")
        _require(
            record.get("schedule") == [0, 1] * 8
            and all(type(x) is int for x in record["schedule"]),
            "return schedule mismatch",
        )
    diagnostics = _records(record.get("diagnostics", []), 16, "update diagnostics")
    _require(
        completed is None or len(diagnostics) == completed,
        "update diagnostics/completed count mismatch",
    )
    for diagnostic in diagnostics:
        _fixed(diagnostic, "definition", "fifo_observation_action_count_increment_rows_v1")
        _close(diagnostic.get("value"), 9, "count increment rows")


def _checkpoint_work(record: dict[str, Any], completed: int | None) -> None:
    points = _records(record.get("checkpoints", []), 4, "paid checkpoint prefix")
    if points:
        _require(completed is not None, "checkpoint lacks a completed native clock")
    observed_calls = 0
    for point in points:
        updates = _integer(point.get("updates"), 0, 16, "checkpoint native progress")
        _require(
            completed is not None and updates <= completed,
            "checkpoint exceeds completed native updates",
        )
        observed_calls += len(_object(point.get("scores"), "paid scores"))
    returned = record.get("score_forward_calls_completed")
    if returned is not None:
        _require(returned >= observed_calls, "published scores exceed returned forwards")
        if record.get("status") == "complete":
            _require(
                returned == observed_calls, "complete forwards differ from published checkpoints"
            )


def _monitor(record: dict[str, Any], completed: int | None) -> None:
    if not record:
        return
    _require(_status(record.get("status")), "invalid A monitor status")
    _counters(
        record,
        (
            ("score_forward_calls_attempted", "score_forward_calls_completed", 8),
            ("attempted_rows", "completed_rows", 72),
        ),
    )
    calls, rows = record.get("score_forward_calls_completed"), record.get("completed_rows")
    _require(calls is None or rows == 9 * calls, "monitor score rows mismatch")
    if record["status"] == "complete":
        _require(calls == 8, "complete monitor requires eight score calls")
    _checkpoint_work(record, completed)


def _mean(curves: list[dict[str, Any]], phase: str, context: str, updates: int) -> float | None:
    values = [
        r["ce"]
        for r in curves
        if r["phase"] == phase
        and r["context"] in (("F-blue", "F-red") if context == "F" else ("A",))
        and r["updates"] == updates
    ]
    return sum(values) / 2 if len(values) == 2 and all(v is not None for v in values) else None


def _metrics(
    row: dict[str, Any], curves: list[dict[str, Any]], pointer: str, source: int
) -> list[dict[str, Any]]:
    outcomes = row.get("outcomes")
    outcomes = {} if outcomes is None else _object(outcomes, "signed outcomes")
    _require(set(outcomes) <= set(METRICS), "unsupported signed outcome")
    endpoints = (
        (("F learning", "F", 0), ("F learning", "F", 16)),
        (("F learning", "A", 16), ("F learning", "A", 0)),
        (("A return", "A", 0), ("A return", "A", 16)),
        (("A return", "F", 16), ("F learning", "F", 16)),
    )
    result = []
    for name, endpoint in zip(METRICS, endpoints, strict=True):
        left, right = _mean(curves, *endpoint[0]), _mean(curves, *endpoint[1])
        value = outcomes.get(name)
        if value is not None:
            _require(
                left is not None and right is not None, "signed metric lacks endpoint evidence"
            )
            _close(value, cast(float, left) - cast(float, right), name)
        if row.get("status") == "complete":
            _require(value is not None, "complete cohort lacks signed metric")
        result.append(
            dict(
                source_updates=source,
                label=name,
                value=value,
                pointer=f"{pointer}/outcomes/{name}" if name in outcomes else None,
                direction="positive_loss_worse"
                if name.endswith("loss")
                else "positive_gain_better",
            )
        )
    return result


def project_minigrid_saved_fifo_report(
    report: dict[str, Any], *, source_label: str
) -> dict[str, Any]:
    """Validate source consistency and project all64 planned read-only curve slots."""
    _envelope(report, source_label)
    _protocol(report)
    _origin(report)
    _require(_status(report.get("status")), "invalid saved FIFO status")
    complete = report["status"] == "complete"
    targets = _data(report, complete)
    cohorts = _records(report.get("cohorts", []), 2, "cohorts")
    _require(not complete or len(cohorts) == 2, "complete report requires both cohorts")
    curves, proofs, costs, metrics = [], [], [], []
    known_work = dict(
        attempted_storage_updates=0,
        completed_storage_updates=0,
        attempted_training_rows=0,
        completed_training_rows=0,
        score_calls_attempted=0,
        score_calls_completed=0,
        score_rows_completed=0,
    )
    known_charges = 0
    work_known = len(cohorts) == 2
    for index, source in enumerate((0, 128)):
        row = cohorts[index] if index < len(cohorts) else {}
        pointer = f"/cohorts/{index}"
        if row:
            _fixed(row, "source_updates", source)
            _fixed(row, "native_training_examples_before", 9 * source)
            _fixed(row, "method", METHOD)
            _fixed(row, "fifo_retirement_metadata_bytes", 16)
            _require(_status(row.get("status")), "invalid cohort status")
            if complete:
                _require(row["status"] == "complete", "complete report contains unfinished cohort")
        raw = row.get("report")
        raw = {} if raw is None else _object(raw, "helper report")
        work_known = work_known and bool(raw)
        returned = _object(raw.get("return_phase", {}), "return phase")
        charges = row.get("phase_charges")
        charges = {} if charges is None else _object(charges, "phase charges")
        _require(set(charges) <= {"F", "A_return"}, "unsupported phase charge")
        for value in charges.values():
            _integer(value, 0, 16, "phase charge")
        charged = row.get("charged_updates")
        if charged is not None:
            _integer(charged, 0, 32, "permanent charged updates")
            _require(
                len(charges) == 2 and charged == sum(charges.values()),
                "permanent charge sum mismatch",
            )
            known_charges += charged
        observed = row.get("budget_updates_observed")
        if observed is not None:
            _integer(observed, 0, 32, "observed budget updates")
            _require(
                row.get("charge_readback_unknown") is True or observed == charged,
                "observed/original charge mismatch",
            )
        _phase(raw, charges.get("F"), False)
        _phase(returned, charges.get("A_return"), True)
        if returned:
            _require(
                raw.get("completed_updates") == 16 and charges.get("F") == 16,
                "A return lacks completed F prerequisite",
            )
        local_curves = []
        for phase, container, base in (
            ("F learning", raw, f"{pointer}/report"),
            ("A return", returned, f"{pointer}/report/return_phase"),
        ):
            monitor = _object(container.get("training_readouts", {}), "A monitor")
            _monitor(monitor, container.get("completed_updates"))
            if row.get("status") == "complete":
                _require(
                    monitor.get("status") == "complete",
                    "complete cohort contains unfinished A monitor",
                )
            work_known = work_known and bool(container) and bool(monitor)
            for key, source_key in (
                ("attempted_storage_updates", "attempted_updates"),
                ("completed_storage_updates", "completed_updates"),
                ("attempted_training_rows", "attempted_training_examples"),
                ("completed_training_rows", "completed_training_examples"),
            ):
                value = container.get(source_key)
                if value is None:
                    work_known = False
                else:
                    known_work[key] += value
            for record in (container, monitor):
                for key, source_key in (
                    ("score_calls_attempted", "score_forward_calls_attempted"),
                    ("score_calls_completed", "score_forward_calls_completed"),
                ):
                    value = record.get(source_key)
                    if value is None:
                        work_known = False
                    else:
                        known_work[key] += value
                value = record.get("score_forward_calls_completed")
                if value is not None:
                    known_work["score_rows_completed"] += 9 * value
            local_curves.extend(_curves(container, base, source, phase, "F", targets))
            local_curves.extend(
                _curves(monitor, f"{base}/training_readouts", source, phase, "A", targets)
            )
            proofs.extend(_record_proofs(container, base))
            proofs.extend(_record_proofs(monitor, f"{base}/training_readouts"))
            costs.extend(
                _flatten(
                    {
                        k: v
                        for k, v in container.items()
                        if k
                        not in (
                            "checkpoints",
                            "decision_diagnostics",
                            "diagnostics",
                            "training_readouts",
                            "return_phase",
                            "episode_scope",
                        )
                    },
                    base,
                )
            )
            costs.extend(
                _flatten(
                    {k: v for k, v in monitor.items() if k != "checkpoints"},
                    f"{base}/training_readouts",
                )
            )
        if row.get("status") == "complete":
            _require(
                raw.get("status") == "complete"
                and returned.get("status") == "complete"
                and all(
                    r["pointer"] is not None and r["ce"] is not None and r["status"] == "complete"
                    for r in local_curves
                ),
                "complete cohort contains unknown phase evidence",
            )
            _require(
                charged == 32 and row.get("end_snapshot") is not None,
                "complete cohort lacks final state or charges",
            )
        _snapshot(row.get("end_snapshot"), source, charged)
        if row.get("end_snapshot") is not None:
            native = row["end_snapshot"]["native_state"]
            costs.extend(
                _flatten(
                    {
                        k: native[k]
                        for k in (
                            "completed_updates",
                            "training_examples",
                            "evictions",
                            "retired_examples",
                        )
                    },
                    f"{pointer}/end_snapshot/native_state",
                )
            )
        proofs.extend(_record_proofs(row, pointer))
        costs.extend(
            _flatten(
                {
                    k: row[k]
                    for k in (
                        "charged_updates",
                        "budget_updates_observed",
                        "phase_charges",
                        "fifo_retirement_metadata_bytes",
                    )
                    if k in row
                },
                pointer,
            )
        )
        metrics.extend(_metrics(row, local_curves, pointer, source))
        curves.extend(local_curves)
    for key in ("work", "observed_process_rss", "origin"):
        value = report.get(key)
        if value is not None:
            costs.extend(_flatten(_object(value, key), f"/{key}"))
    costs.extend(
        _flatten(
            {
                key: report[key]
                for key in ("total_seconds", "retained_input_array_bytes", "source_load_seconds")
                if key in report
            },
            "",
        )
    )
    work = report.get("work")
    if work is not None:
        _counters(
            work,
            (
                ("attempted_storage_updates", "completed_storage_updates", 64),
                ("attempted_training_rows", "completed_training_rows", 576),
                ("score_calls_attempted", "score_calls_completed", 64),
            ),
        )
        for key in ("benchmark_updates", "replay_examples", "reused_TRAIN_A_new_physics_steps"):
            _fixed(work, key, 0)
        lower_bound = work.get("observed_charged_updates_lower_bound")
        if lower_bound is not None:
            _integer(lower_bound, 0, 64, "charged update lower bound")
            _require(
                lower_bound >= known_charges,
                "aggregate charges below preserved permanent charges",
            )
            if work_known and work.get("probe_totals_are_unknown") is False:
                _require(
                    lower_bound == known_charges,
                    "aggregate charge lower bound contradicts observed charges",
                )
        for rows_key, count_key in (
            ("completed_training_rows", "completed_storage_updates"),
            ("score_rows_completed", "score_calls_completed"),
        ):
            rows, count = work.get(rows_key), work.get(count_key)
            if rows is not None and count is not None:
                _require(rows == 9 * count, "aggregate row count mismatch")
            else:
                _require(
                    not complete and work.get("probe_totals_are_unknown") is True,
                    "unknown aggregate work requires an explicit partial status",
                )
        for key, known in known_work.items():
            value = work.get(key)
            if value is not None:
                _integer(
                    value,
                    0,
                    576 if key.endswith("rows") or key == "score_rows_completed" else 64,
                    key,
                )
                _require(value >= known, "aggregate work below returned helper prefix")
                if work_known and work.get("probe_totals_are_unknown") is False:
                    _require(value == known, "aggregate work contradicts known helper counters")
        if complete:
            _fixed(work, "completed_storage_updates", 64)
            _fixed(work, "score_calls_completed", 64)
    return dict(
        schema_id="minigrid_saved_fifo_inspector_v1",
        source_label=source_label,
        origin="recorded_artifact",
        source=deepcopy(report),
        conditions=[
            dict(
                key=METHOD,
                pointer="",
                status=report["status"],
                source_role="train_and_probe_test",
                episodes=[],
                curves=curves,
                proofs=proofs,
                costs=costs,
                signed_metrics=metrics,
                unknown="No autonomous games planned; gameplay outcomes unavailable. Partial observations and costs stay unknown. Return is cold acquisition or retained TRAIN fit, not evidence of recovery from degradation or a circadian advantage.",
            )
        ],
    )
