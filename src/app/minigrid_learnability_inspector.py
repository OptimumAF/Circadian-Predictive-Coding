"""Project bounded, saved H32 F-learning/A-return reports for offline inspection.

This is a source-faithful viewer, not scientific qualification or a learner.
Inputs are JSON; outputs retain the original source and exact JSON pointers.
Missing planned slots are explicitly derived placeholders with no source pointer.
No model, environment, filesystem, training or final evaluation is invoked.
"""

from copy import deepcopy
from math import fsum, log
from typing import Any

from .minigrid_inspector_projection import (
    _boolean,
    _close,
    _envelope,
    _episode,
    _integer,
    _number,
    _probabilities,
    _records,
    _require,
    _sources,
    _status,
)

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
PROTOCOL_IDS = tuple(
    f"minigrid_invariant_{method}_learnability_qualification_distinct_probe_"
    "old_task_readouts_a_return_joint_catalogue_h32_probe_rollout_v1"
    for method in METHODS
)
POINTS = (0, 4, 8, 16)
SLOTS = (
    ("A", "blue", "train"),
    ("A", "red", "train"),
    ("F-blue", "blue", "probe_test"),
    ("F-red", "red", "probe_test"),
)


def _object(value: Any, name: str) -> dict[str, Any]:
    _require(type(value) is dict, f"invalid {name} object")
    return value


def _fixed_int(record: dict[str, Any], key: str, expected: int) -> None:
    _require(type(record.get(key)) is int and record[key] == expected, f"unsupported fixed {key}")


def _int_points(value: Any, expected: list[int], name: str) -> None:
    _require(
        type(value) is list and value == expected and all(type(point) is int for point in value),
        f"unsupported {name}",
    )


def _protocol(report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    protocol = _object(report.get("protocol"), "protocol")
    _require(protocol.get("protocol_id") in PROTOCOL_IDS, "unsupported H32 learnability protocol")
    native = METHODS[PROTOCOL_IDS.index(protocol["protocol_id"])]
    method = f"public_invariant_{native}"
    _require(protocol.get("methods") == [method], "H32 method selector mismatch")
    for key, value in (
        ("seed", 17),
        ("feature_dim", 296),
        ("hidden_dim", 32),
        ("num_classes", 7),
        ("phase_updates", 32),
    ):
        _fixed_int(protocol, key, value)
    _int_points(protocol.get("checkpoint_updates"), [0, 32, 64, 96, 128], "benchmark checkpoints")
    _require(protocol.get("cues") == ["blue", "red"], "unsupported cue order")
    stream = _object(protocol.get("stream"), "training stream")
    _require(
        stream.get("training_only") is True
        and stream.get("heldout") == []
        and type(stream.get("label_delay")) is int
        and stream["label_delay"] == 1,
        "benchmark must retain training-only isolation",
    )
    probe = _object(protocol.get("probe"), "probe recipe")
    _int_points(probe.get("source_updates"), [0, 128], "probe sources")
    _int_points(probe.get("checkpoint_updates"), list(POINTS), "probe checkpoints")
    _fixed_int(probe, "updates", 16)
    returned = _object(protocol.get("return_phase"), "return recipe")
    _require(
        returned.get("context") == "A"
        and returned.get("role") == "train"
        and returned.get("checkpoint_updates") == list(POINTS)
        and returned.get("schedule") == [0, 1] * 8,
        "unsupported A return recipe",
    )
    _fixed_int(returned, "updates", 16)
    _int_points(returned.get("checkpoint_updates"), list(POINTS), "return checkpoints")
    _int_points(returned.get("schedule"), [0, 1] * 8, "return schedule")
    games = _object(protocol.get("probe_rollout"), "game recipe")
    _require(
        games.get("phases") == ["after_probe", "after_return"]
        and games.get("source_updates") == [0, 128]
        and games.get("native_endpoint_updates") == [16, 32, 144, 160],
        "unsupported known-task game endpoints",
    )
    _int_points(games.get("source_updates"), [0, 128], "game sources")
    _int_points(games.get("native_endpoint_updates"), [16, 32, 144, 160], "game endpoints")
    _require(
        report.get("scientific_advantage_verified") is False,
        "inspector does not accept a scientific advantage claim",
    )
    return protocol, method


def _flatten(record: dict[str, Any], pointer: str, prefix: str = "") -> list[dict[str, Any]]:
    """Keep nulls, lower-bound names and scope text beside recorded scalar costs."""
    result = []
    for key, value in record.items():
        label = f"{prefix}.{key}" if prefix else key
        path = f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}"
        if type(value) is dict:
            result.extend(_flatten(value, path, label))
        elif value is None or type(value) in (str, bool, int, float):
            if type(value) in (int, float):
                _number(value, label, 0)
            result.append(dict(label=label, value=value, pointer=path))
    return result


def _proofs(record: dict[str, Any], pointer: str) -> list[dict[str, Any]]:
    result = []
    for key, value in record.items():
        if key.endswith(("unchanged", "binding_preserved", "work_unknown")):
            if value is not None:
                _boolean(value, key)
            result.append(dict(label=key, value=value, pointer=f"{pointer}/{key}"))
    return result


def _score(
    score: dict[str, Any], diagnostic: Any
) -> tuple[float | None, float | None, float | None]:
    ce, accuracy, exact = (
        score.get("ce"),
        score.get("accuracy"),
        score.get("exact_sequence_accuracy"),
    )
    if ce is not None:
        _number(ce, "CE", 0)
    if accuracy is not None:
        _number(accuracy, "accuracy", 0, 1)
    if exact is not None:
        _number(exact, "exact sequence accuracy", 0, 1)
        _require(exact in (0, 1), "complete episode exact score must be binary")
    # Nine rows are a complete fixed episode, so exactness follows from all correct.
    if accuracy is not None:
        inferred = float(accuracy == 1)
        _require(exact is None or exact == inferred, "exact score contradicts accuracy")
        exact = inferred
    if diagnostic is not None:
        diagnostic = _object(diagnostic, "decision diagnostics")
        _fixed_int(diagnostic, "num_classes", 7)
        _fixed_int(diagnostic, "rows", 9)
        confusion = diagnostic.get("confusion")
        _require(
            type(confusion) is list
            and len(confusion) == 7
            and all(type(row) is list and len(row) == 7 for row in confusion),
            "invalid seven-action confusion",
        )
        for row in confusion:
            for count in row:
                _integer(count, 0, 9, "confusion count")
        _require(sum(map(sum, confusion)) == 9, "confusion must cover nine rows")
        decisions = _records(diagnostic.get("decisions"), 9, "diagnostic decisions")
        _require(len(decisions) == 9, "diagnostics require nine decisions")
        observed = [[0] * 7 for _ in range(7)]
        losses = []
        for index, decision in enumerate(decisions):
            _fixed_int(decision, "index", index)
            target = _integer(decision.get("target_class"), 0, 6, "diagnostic target")
            predicted = _integer(decision.get("predicted_class"), 0, 6, "diagnostic prediction")
            _require(
                decision.get("correct") is (target == predicted), "diagnostic correctness mismatch"
            )
            observed[target][predicted] += 1
            losses.append(_number(decision.get("loss"), "diagnostic loss", 0))
        _require(observed == confusion, "confusion contradicts recorded decisions")
        if ce is not None:
            _close(ce, fsum(losses) / 9, "diagnostic CE")
        correct = sum(confusion[i][i] for i in range(7)) / 9
        _require(
            accuracy is None or abs(accuracy - correct) <= 1e-10, "accuracy contradicts confusion"
        )
        if accuracy is not None:
            exact = float(correct == 1)
    probabilities = score.get("probabilities")
    if probabilities is not None:
        labels: Any = score.get("target_actions")
        _require(
            type(probabilities) is list
            and len(probabilities) == 9
            and type(labels) is list
            and len(labels) == 9,
            "invalid benchmark decisions",
        )
        labels = [_integer(x, 0, 6, "benchmark target") for x in labels]
        validated = [_probabilities(row) for row in probabilities]
        predictions = [action for _, action in validated]
        _require(score.get("predicted_actions") == predictions, "benchmark argmax mismatch")
        _close(
            ce,
            fsum(
                -log(max(row[target], 1e-12))
                for (row, _), target in zip(validated, labels, strict=True)
            )
            / 9,
            "CE",
        )
        _close(
            accuracy, sum(p == t for p, t in zip(predictions, labels, strict=True)) / 9, "accuracy"
        )
    return ce, accuracy, exact


def _curve_rows(
    container: dict[str, Any],
    pointer: str,
    source_updates: int,
    phase: str,
    context: str,
    role: str,
) -> list[dict[str, Any]]:
    rows = []
    checkpoints = _records(container.get("checkpoints", []), 4, "curve checkpoints")
    _require(
        [point.get("updates") for point in checkpoints] == list(POINTS[: len(checkpoints)]),
        "curve checkpoint order mismatch",
    )
    for index, point in enumerate(checkpoints):
        _integer(point["updates"], 0, 16, "curve updates")
        scores = _object(point.get("scores"), "curve scores")
        diagnostics = _object(point.get("decision_diagnostics", {}), "curve diagnostics")
        _require(
            set(scores)
            <= (
                {"F-blue", "F-red"}
                if context == "F"
                else {"minigrid-source-A-blue", "minigrid-source-A-red"}
            ),
            "curve contains unsupported role/context",
        )
        for cue in ("blue", "red"):
            name = f"F-{cue}" if context == "F" else f"minigrid-source-A-{cue}"
            if name not in scores:
                continue
            score = _object(scores[name], "curve score")
            ce, accuracy, exact = _score(score, diagnostics.get(name))
            rows.append(
                dict(
                    source_updates=source_updates,
                    phase=phase,
                    updates=point["updates"],
                    context=f"F-{cue}" if context == "F" else "A",
                    cue=cue,
                    role=role,
                    status=point.get("status", "complete"),
                    ce=ce,
                    accuracy=accuracy,
                    exact_sequence_accuracy=exact,
                    pointer=f"{pointer}/checkpoints/{index}/scores/{name}",
                )
            )
    # Planned missing scores remain visible without inventing a source JSON record.
    observed = {(row["updates"], row["cue"]) for row in rows}
    for updates in POINTS:
        for cue in ("blue", "red"):
            if (updates, cue) not in observed:
                rows.append(
                    dict(
                        source_updates=source_updates,
                        phase=phase,
                        updates=updates,
                        context=f"F-{cue}" if context == "F" else "A",
                        cue=cue,
                        role=role,
                        status="not_run",
                        ce=None,
                        accuracy=None,
                        exact_sequence_accuracy=None,
                        pointer=None,
                        derived=True,
                    )
                )
    return rows


def _curves(
    method: dict[str, Any], pointer: str, targets: dict[str, list[int]]
) -> list[dict[str, Any]]:
    rows = []
    readouts = _records(method.get("readouts", []), 5, "benchmark readouts")
    for index, readout in enumerate(readouts):
        updates = _integer(readout.get("updates"), 0, 128, "benchmark updates")
        _require(
            updates == (0, 32, 64, 96, 128)[index] and readout.get("role") == "train",
            "benchmark readout order/role mismatch",
        )
        for si, score in enumerate(_records(readout.get("scores", []), 6, "benchmark scores")):
            context, cue = score.get("context"), score.get("cue")
            _require(
                (context, cue)
                == [(c, q) for c in ("A", "B", "C_train") for q in ("blue", "red")][si],
                "benchmark context order",
            )
            episode_id = f"minigrid-source-{context}-{cue}"
            _require(
                score.get("episode_id") == episode_id
                and score.get("target_actions") == targets[episode_id],
                "benchmark targets differ from original TRAIN source",
            )
            ce, accuracy, exact = _score(score, None)
            rows.append(
                dict(
                    source_updates=updates,
                    phase="benchmark",
                    updates=updates,
                    context=context,
                    cue=cue,
                    role="train",
                    status=readout.get("status"),
                    ce=ce,
                    accuracy=accuracy,
                    exact_sequence_accuracy=exact,
                    pointer=f"{pointer}/readouts/{index}/scores/{si}",
                )
            )
    probes = method.get("probes", [])
    for pi, source_updates in enumerate((0, 128)):
        probe = probes[pi] if pi < len(probes) else {}
        report = probe.get("report")
        report = {} if report is None else _object(report, "probe report")
        base = f"{pointer}/probes/{pi}/report"
        rows.extend(_curve_rows(report, base, source_updates, "F learning", "F", "probe_test"))
        monitor = _object(report.get("training_readouts", {}), "old A readouts")
        rows.extend(
            _curve_rows(
                monitor, f"{base}/training_readouts", source_updates, "F learning", "A", "train"
            )
        )
        returned = _object(report.get("return_phase", {}), "A return")
        rows.extend(
            _curve_rows(
                returned, f"{base}/return_phase", source_updates, "A return", "F", "probe_test"
            )
        )
        monitor = _object(returned.get("training_readouts", {}), "return old A readouts")
        rows.extend(
            _curve_rows(
                monitor,
                f"{base}/return_phase/training_readouts",
                source_updates,
                "A return",
                "A",
                "train",
            )
        )
    return rows


def _missing(
    source_updates: int, phase: str, context: str, cue: str, role: str, updates: int
) -> dict[str, Any]:
    return dict(
        pointer=None,
        record=dict(
            context=context,
            cue=cue,
            source_role=role,
            status="not_run",
            reason="No recorded game for this planned slot",
            success=None,
            **{"return": None},
        ),
        updates=updates,
        cue=cue,
        frames=[],
        initial_image=None,
        source_updates=source_updates,
        phase=phase,
        context=context,
        derived=True,
    )


def _games(
    method: dict[str, Any], pointer: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes, proofs = [], []
    probes = _records(method.get("probes", []), 2, "source probes")
    _require(
        [p.get("checkpoint_updates") for p in probes] == [0, 128][: len(probes)],
        "source probe order mismatch",
    )
    for pi, source_updates in enumerate((0, 128)):
        probe = probes[pi] if pi < len(probes) else {}
        base = f"{pointer}/probes/{pi}"
        proofs.extend(_proofs(probe, base))
        endpoints = _records(probe.get("probe_rollouts", []), 2, "game endpoints")
        _require(
            [e.get("phase") for e in endpoints]
            == ["after_probe", "after_return"][: len(endpoints)],
            "game phase order mismatch",
        )
        for ei, phase in enumerate(("after_probe", "after_return")):
            relative = 16 if phase == "after_probe" else 32
            updates = source_updates + relative
            endpoint = endpoints[ei] if ei < len(endpoints) else {}
            path = f"{base}/probe_rollouts/{ei}"
            proofs.extend(_proofs(endpoint, path))
            report = endpoint.get("report")
            records = []
            if endpoint:
                _require(_status(endpoint.get("status")), "invalid endpoint status")
                _require(
                    endpoint.get("source_updates") == source_updates,
                    "endpoint source clock mismatch",
                )
                _fixed_int(endpoint, "source_updates", source_updates)
                _fixed_int(endpoint, "charged_updates", 0)
                _fixed_int(endpoint, "replay_examples", 0)
            if report is not None:
                report = _object(report, "game report")
                _require(_status(report.get("status")), "invalid game report status")
                _require(
                    report.get("method") == method["method"]
                    and report.get("role") == "same_fork_probe_exploratory"
                    and report.get("phase") == phase,
                    "game report role/method/phase mismatch",
                )
                for key, expected in (
                    ("updates", updates),
                    ("source_updates", source_updates),
                    ("relative_updates", relative),
                    ("hidden_dim", 32),
                    ("native_training_examples", 54 * source_updates + 9 * relative),
                ):
                    _fixed_int(report, key, expected)
                proofs.extend(_proofs(report, f"{path}/report"))
                records = _records(report.get("episodes", []), 4, "game records")
                _require(
                    report.get("status") != "complete" or len(records) == 4,
                    "complete endpoint requires four games",
                )
            if endpoint.get("status") == "complete" or (
                report is not None and report.get("status") == "complete"
            ):
                _require(
                    len(records) == 4 and all(r.get("status") == "complete" for r in records),
                    "complete endpoint contains unfinished games",
                )
            for ri, (context, cue, role) in enumerate(SLOTS):
                if ri >= len(records):
                    episodes.append(_missing(source_updates, phase, context, cue, role, updates))
                    continue
                record = records[ri]
                game_path = f"{path}/report/episodes/{ri}"
                if record.get("status") == "not_run":
                    _require(
                        record.get("context") == context
                        and record.get("cue") == cue
                        and record.get("source_role") == role
                        and not record.get("predictions")
                        and record.get("environment") is None
                        and record.get("success") is None
                        and record.get("return") is None
                        and type(record.get("prediction_calls_attempted")) is int
                        and record["prediction_calls_attempted"] == 0
                        and type(record.get("prediction_calls_returned")) is int
                        and record["prediction_calls_returned"] == 0
                        and record.get("state_unchanged") is None,
                        "unrun game contains observed work or incorrect identity",
                    )
                    game = _missing(source_updates, phase, context, cue, role, updates)
                    game.update(pointer=game_path, record=deepcopy(record), derived=False)
                else:
                    if record.get("status") != "complete" and record.get("success") is True:
                        native = _object(record.get("environment"), "partial success evidence")
                        _require(
                            native.get("terminated") is True
                            and native.get("truncated") is False
                            and _number(native.get("total_return"), "partial success return") > 0,
                            "partial success contradicts native outcome",
                        )
                    game = _episode(
                        record,
                        game_path,
                        method["method"],
                        updates,
                        cue,
                        context=context,
                        role=role,
                        branch_row=3 if context == "A" else 2,
                        allow_observed_failure_tail=True,
                    )
                    game.update(
                        source_updates=source_updates, phase=phase, context=context, derived=False
                    )
                episodes.append(game)
    return episodes, proofs


def project_minigrid_learnability_report(
    report: dict[str, Any], *, source_label: str
) -> dict[str, Any]:
    """Validate the fixed saved-report contract and derive an inert display payload."""
    _envelope(report, source_label)
    _protocol_value, method_name = _protocol(report)
    _require(_status(report.get("status")), "invalid H32 report status")
    _sources(report, True)
    _require(
        [s.get("episode_id") for s in report["training_sources"]]
        == [
            f"minigrid-source-{context}-{cue}"
            for context in ("A", "B", "C_train")
            for cue in ("blue", "red")
        ],
        "unsupported original TRAIN source identity",
    )
    targets = {
        source["episode_id"]: [step["action"] for step in source["transitions"]]
        for source in report["training_sources"]
    }
    methods = _records(report.get("methods"), 1, "H32 method")
    _require(
        len(methods) == 1 and methods[0].get("method") == method_name, "H32 report method mismatch"
    )
    method = methods[0]
    _require(_status(method.get("status")), "invalid H32 method status")
    final = _object(method.get("final_evaluation"), "final evaluation")
    _require(
        final.get("status") == "not_run"
        and final.get("role") == "final_test"
        and set(final) <= {"status", "role", "updates"},
        "final evaluation must remain unavailable",
    )
    episodes, proofs = _games(method, "/methods/0")
    if report["status"] == "complete":
        _require(
            all(e["record"]["status"] == "complete" for e in episodes)
            and len(method.get("probes", [])) == 2
            and all(p.get("status") == "complete" for p in method["probes"]),
            "complete report contains unfinished games or probes",
        )
    proofs = _proofs(method, "/methods/0") + proofs
    costs = _flatten(_object(method.get("work"), "method work"), "/methods/0/work")
    return dict(
        schema_id="minigrid_learnability_inspector_v1",
        source_label=source_label,
        origin="recorded_artifact",
        source=deepcopy(report),
        conditions=[
            dict(
                key=method_name,
                pointer="/methods/0",
                status=method["status"],
                source_role="train_and_probe_test",
                episodes=episodes,
                curves=_curves(method, "/methods/0", targets),
                proofs=proofs,
                costs=costs,
                unknown="Missing observations and costs remain unknown; no scientific qualification.",
            )
        ],
    )
