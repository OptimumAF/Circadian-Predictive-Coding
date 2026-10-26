"""Run the fixed isolated MiniGrid count-memory probe from saved native states.

The default routes read accepted memory-live reports; the saved FIFO route reads
the original recurring FIFO36 report, including its failed budget and real states.
This script owns bounded JSON conversion and publication; the existing adapter
owns probe budgets and execution. No original budget or serving runtime is restored.
"""

import argparse
import json
from math import isfinite
import os
from pathlib import Path
import sys
from time import monotonic
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from src.adapters.minigrid_saved_fifo_inputs import SavedFifoInputs
    from src.core.action_memory import ActionMemorySnapshot
    from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition

MAX_INPUT_BYTES = 32 * 1024**2
MAX_SAVED_FIFO_INPUT_BYTES = 16 * 1024**2
MAX_OUTPUT_BYTES = 32 * 1024**2
_POINTS = (0, 32, 64, 96, 128)


def describe_protocol() -> dict[str, Any]:
    from src.adapters.minigrid_action_memory_probes import describe_protocol as describe

    return describe()


def run_pilot(snapshots: tuple["ActionMemorySnapshot", "ActionMemorySnapshot"]) -> dict[str, Any]:
    from src.adapters.minigrid_action_memory_probes import run_pilot as run

    return run(snapshots)


def describe_return_protocol() -> dict[str, Any]:
    from src.adapters.minigrid_action_memory_probes import describe_return_protocol as describe

    return describe()


def run_return_pilot(
    snapshots: tuple["ActionMemorySnapshot", "ActionMemorySnapshot"],
    return_sources: tuple["DemonstrationEpisode", "DemonstrationEpisode"],
) -> dict[str, Any]:
    from src.adapters.minigrid_action_memory_probes import run_return_pilot as run

    return run(snapshots, return_sources)


def describe_saved_fifo_return_protocol() -> dict[str, Any]:
    from src.adapters.minigrid_action_memory_probes import (
        describe_saved_fifo_return_protocol as describe,
    )

    return describe()


def run_saved_fifo_return_pilot(
    inputs: "SavedFifoInputs", *, max_seconds: float = 120.0
) -> dict[str, Any]:
    from src.adapters.minigrid_action_memory_probes import run_saved_fifo_return_pilot as run

    return run(inputs, max_seconds=max_seconds)


def load_saved_fifo_inputs(path: Path) -> "SavedFifoInputs":
    """Decode original FIFO36 evidence without restoring its spent budget."""
    from src.adapters.minigrid_saved_fifo_inputs import decode_saved_fifo_inputs

    return decode_saved_fifo_inputs(_read_source(path, max_bytes=MAX_SAVED_FIFO_INPUT_BYTES))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == fields, f"invalid exact {name} fields")
    return cast(dict[str, Any], value)


def _fixed_count(value: Any, expected: int, name: str) -> None:
    _require(type(value) is int and value == expected, f"invalid original {name}")


def _finite_number(text: str) -> float:
    value = float(text)
    _require(isfinite(value), "source artifact contains a nonfinite JSON number")
    return value


def _reject_constant(text: str) -> Any:
    raise ValueError(f"source artifact contains nonfinite JSON {text}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _require(key not in result, "source artifact contains duplicate JSON fields")
        result[key] = value
    return result


def _read_source(path: Path, *, max_bytes: int | None = None) -> dict[str, Any]:
    limit = MAX_INPUT_BYTES if max_bytes is None else max_bytes
    with path.open("rb") as handle:
        payload = handle.read(limit + 1)
    _require(len(payload) <= limit, f"source artifact exceeds {limit // 1024**2} MiB")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_finite_number,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("source artifact must be bounded finite UTF-8 JSON") from error
    _require(type(value) is dict, "source artifact must be a JSON object")
    return cast(dict[str, Any], value)


def _snapshot(value: Any, point: int) -> "ActionMemorySnapshot":
    from src.core.action_memory import (
        ActionMemoryEntry,
        ActionMemoryPolicy,
        ActionMemorySnapshot,
        copy_action_memory_snapshot,
    )

    value = _object(
        value, {"policy", "entries", "completed_updates", "training_examples"}, "snapshot"
    )
    policy_data = _object(
        value["policy"],
        {"method", "feature_dim", "num_classes", "max_entries", "max_payload_bytes"},
        "snapshot policy",
    )
    policy = ActionMemoryPolicy(**policy_data)
    _require(
        policy
        == ActionMemoryPolicy(
            feature_dim=296, num_classes=7, max_entries=64, max_payload_bytes=155136
        ),
        "source snapshot geometry/capacity differs from the fixed protocol",
    )
    records = value["entries"]
    _require(type(records) is list and len(records) <= 64, "invalid bounded snapshot entries")
    entries = []
    for record in records:
        record = _object(record, {"features", "counts"}, "snapshot entry")
        _require(
            type(record["features"]) is list and len(record["features"]) == 296,
            "snapshot entry requires 296 recorded features",
        )
        _require(
            type(record["counts"]) is list and len(record["counts"]) == 7,
            "snapshot entry requires seven recorded counts",
        )
        entries.append(ActionMemoryEntry(tuple(record["features"]), tuple(record["counts"])))
    owned = copy_action_memory_snapshot(
        ActionMemorySnapshot(
            policy, tuple(entries), value["completed_updates"], value["training_examples"]
        )
    )
    _fixed_count(owned.completed_updates, point, "snapshot update clock")
    _fixed_count(owned.training_examples, 9 * point, "snapshot row clock")
    return owned


def _train_binding(report: dict[str, Any]) -> None:
    sources: Any = report.get("training_sources")
    _require(
        type(sources) is list and len(sources) == 6, "source requires six original TRAIN packets"
    )
    for record, (context, cue) in zip(
        sources,
        ((context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")),
        strict=True,
    ):
        _require(
            type(record) is dict
            and record.get("role") == "train"
            and type(record.get("role")) is str
            and record.get("context") == context
            and record.get("episode_id") == f"minigrid-source-{context}-{cue}",
            "source packet does not retain its original TRAIN role/identity",
        )
        _fixed_count(record.get("num_classes"), 7, "source class count")


def _snapshots_from_report(
    report: dict[str, Any],
) -> tuple["ActionMemorySnapshot", "ActionMemorySnapshot"]:
    """Share the unchanged prior snapshot admission between the two CLI routes."""
    from src.adapters.minigrid_action_memory_pilot import describe_protocol as prior_protocol
    from src.app.coupled_attribution import _same_value

    _require(
        _same_value(report.get("protocol"), prior_protocol()),
        "source protocol differs from prior memory-live",
    )
    _require(report.get("status") == "complete", "source artifact must be complete")
    methods: Any = report.get("methods")
    _require(
        type(methods) is list and len(methods) == 1 and type(methods[0]) is dict,
        "source artifact requires one completed observation_memory method",
    )
    row = methods[0]
    _require(
        row.get("method") == "observation_memory" and row.get("status") == "complete",
        "source memory method must be complete",
    )
    _require(
        row.get("source_unchanged") is True and row.get("actor_unchanged") is True,
        "source memory method lacks unchanged source/actor evidence",
    )
    _fixed_count(row.get("charged_updates"), 128, "charged storage updates")
    _fixed_count(row.get("native_training_examples"), 1152, "native training rows")
    _require(
        _same_value(row.get("checkpoint_updates"), list(_POINTS)),
        "source checkpoint schedule changed",
    )
    stream = row.get("stream_result")
    _require(
        type(stream) is dict and stream.get("evaluation_scope") == "training_only",
        "source stream must retain its TRAIN-only scope",
    )
    _fixed_count(stream.get("completed_updates"), 128, "stream updates")
    _fixed_count(stream.get("training_examples"), 1152, "stream rows")
    _fixed_count(stream.get("score_forward_calls"), 0, "stream held-out forwards")
    _fixed_count(stream.get("num_classes"), 7, "stream class count")
    _train_binding(report)
    observations: Any = report.get("observations")
    _require(
        type(observations) is list and len(observations) == 5,
        "source requires all five original native observations",
    )
    snapshots = []
    for observation, point in zip(observations, _POINTS, strict=True):
        observation = _object(
            observation,
            {"updates", "native_training_examples", "entries", "logical_payload_bytes", "snapshot"},
            "observation",
        )
        _fixed_count(observation["updates"], point, "observation update clock")
        _fixed_count(observation["native_training_examples"], 9 * point, "observation row clock")
        state = _snapshot(observation["snapshot"], point)
        _fixed_count(observation["entries"], len(state.entries), "observation entry count")
        _fixed_count(
            observation["logical_payload_bytes"],
            len(state.entries) * 2424,
            "observation logical payload",
        )
        snapshots.append(state)
    _require(
        not snapshots[0].entries and len(snapshots[-1].entries) == 54,
        "source must retain the accepted empty initial and 54-key mature checkpoints",
    )
    return snapshots[0], snapshots[-1]


def load_snapshots(path: Path) -> tuple["ActionMemorySnapshot", "ActionMemorySnapshot"]:
    """Admit the fixed completed prior report and detach its actual 0/128 states."""
    return _snapshots_from_report(_read_source(path))


def _return_transition(value: Any, index: int) -> "DemonstrationTransition":
    from src.core.demonstration_episode import DemonstrationTransition

    value = _object(
        value,
        {"step", "observation", "action", "reward", "next_observation", "terminal", "num_classes"},
        "original A transition",
    )
    _fixed_count(value["step"], index, "original A transition index")
    _fixed_count(value["num_classes"], 7, "original A transition class count")
    for key in ("observation", "next_observation"):
        _require(
            type(value[key]) is list and len(value[key]) == 148,
            "original A transition requires 148-coordinate public frames",
        )
    return DemonstrationTransition(
        value["step"],
        tuple(value["observation"]),
        value["action"],
        value["reward"],
        tuple(value["next_observation"]),
        value["terminal"],
        value["num_classes"],
    )


def _bind_return_transition(value: dict[str, Any], recorded: Any) -> None:
    from src.app.coupled_attribution import _same_value

    recorded = _object(
        recorded,
        {
            "observation",
            "action",
            "attempted",
            "applied",
            "next_observation",
            "reward",
            "terminated",
            "truncated",
            "raw_next_observation",
        },
        "original A recorded transition",
    )
    for flag in ("attempted", "applied", "terminated", "truncated"):
        _require(type(recorded[flag]) is bool, f"invalid original A recorded {flag}")
    _require(
        recorded["attempted"] is True and recorded["applied"] is True,
        "original A record requires actual applied TRAIN transitions",
    )
    for key in ("observation", "action", "reward", "next_observation"):
        _require(
            _same_value(value[key], recorded[key]),
            f"original A DTO {key} differs from its recorded TRAIN transition",
        )
    _require(
        value["terminal"] is (recorded["terminated"] or recorded["truncated"]),
        "original A DTO terminal differs from its recorded TRAIN transition",
    )


def _return_episodes(
    report: dict[str, Any],
) -> tuple["DemonstrationEpisode", "DemonstrationEpisode"]:
    from src.app.coupled_attribution import _same_value
    from src.core.demonstration_episode import DemonstrationEpisode

    records: Any = report.get("source_reports")
    _require(
        type(records) is list and len(records) == 6,
        "return diagnostics require the six original TRAIN source reports",
    )
    originals: Any = report["training_sources"]
    episodes = []
    for index, cue in enumerate(("blue", "red")):
        original = _object(
            originals[index],
            {"episode_id", "context", "role", "transitions", "num_classes"},
            "original A episode",
        )
        _require(
            original["episode_id"] == f"minigrid-source-A-{cue}"
            and original["context"] == "A"
            and original["role"] == "train",
            "return diagnostics require the original TRAIN A cue pair",
        )
        _fixed_count(original["num_classes"], 7, "original A class count")
        transitions = original["transitions"]
        _require(
            type(transitions) is list and len(transitions) == 9,
            "original A episode requires nine complete transitions",
        )
        recorded = records[index]
        _require(
            type(recorded) is dict
            and recorded.get("episode_id") == original["episode_id"]
            and recorded.get("context") == "A"
            and recorded.get("cue") == cue
            and recorded.get("role") == "train"
            and recorded.get("status") == "complete",
            "original A source report role/identity/status differs from its DTO",
        )
        for key, expected in (
            ("observation_dim", 148),
            ("num_classes", 7),
            ("seed", 17),
            ("max_steps", 16),
            ("environment_steps_attempted", 9),
            ("environment_steps", 9),
            ("reset_calls", 1),
        ):
            _fixed_count(recorded.get(key), expected, f"original A source {key}")
        _require(
            recorded.get("closed") is True
            and recorded.get("terminated") is True
            and recorded.get("truncated") is False,
            "original A source report must retain its completed TRAIN ending",
        )
        _require(
            _same_value(recorded.get("layout"), report["protocol"]["layouts"][0]),
            "original A source report layout differs from the original TRAIN layout",
        )
        actual = recorded.get("transitions")
        _require(
            type(actual) is list and len(actual) == 9,
            "original A source report requires nine actual transitions",
        )
        owned = []
        for step, (transition, observed) in enumerate(zip(transitions, actual, strict=True)):
            native = _return_transition(transition, step)
            _bind_return_transition(transition, observed)
            _require(
                observed["truncated"] is False and observed["terminated"] is (step == 8),
                "original A source transition ending differs from its complete report",
            )
            owned.append(native)
        episodes.append(
            DemonstrationEpisode(
                original["episode_id"],
                original["context"],
                original["role"],
                tuple(owned),
                original["num_classes"],
            )
        )
    return episodes[0], episodes[1]


def load_return_inputs(
    path: Path,
) -> tuple[
    tuple["ActionMemorySnapshot", "ActionMemorySnapshot"],
    tuple["DemonstrationEpisode", "DemonstrationEpisode"],
]:
    """Read once; bind old checkpoints and original A TRAIN DTOs to that report."""
    report = _read_source(path)
    snapshots = _snapshots_from_report(report)
    return snapshots, _return_episodes(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--describe", action="store_true", help="print the resource-free fixed protocol"
    )
    parser.add_argument(
        "--source-artifact", type=Path, help="original JSON artifact required by the selected route"
    )
    parser.add_argument("--output", type=Path, help="publish a new JSON artifact; never overwrite")
    routes = parser.add_mutually_exclusive_group()
    routes.add_argument(
        "--return-comparison",
        action="store_true",
        help="compare fixed refusal/FIFO new-task and known TRAIN A return learning",
    )
    routes.add_argument(
        "--saved-fifo-return",
        action="store_true",
        help="probe new-task and original TRAIN A return on genuine saved FIFO36 states",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="saved FIFO return body allowance in (0, 120] seconds; default 120",
    )
    args = parser.parse_args(argv)
    if args.max_seconds is not None:
        if not args.saved_fifo_return:
            parser.error("--max-seconds requires --saved-fifo-return")
        if not isfinite(args.max_seconds) or not 0 < args.max_seconds <= 120:
            parser.error("--max-seconds must be finite and in (0, 120]")
    if args.describe:
        if args.output is not None or args.source_artifact is not None:
            parser.error("--describe does not read source artifacts or publish run outputs")
        if args.saved_fifo_return:
            protocol = describe_saved_fifo_return_protocol()
        else:
            protocol = describe_return_protocol() if args.return_comparison else describe_protocol()
        print(json.dumps(protocol, indent=2, allow_nan=False))
        return 0
    if args.source_artifact is None:
        parser.error("running the fixed probe requires --source-artifact")
    if args.saved_fifo_return and args.output is None:
        parser.error("--saved-fifo-return requires --output to preserve its attempt")
    if args.output is not None:
        if args.output.exists():
            parser.error("output already exists; preserve the original attempt")
        if not args.output.parent.is_dir():
            parser.error("output parent directory must exist")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = monotonic()
    if args.saved_fifo_return:
        inputs = load_saved_fifo_inputs(args.source_artifact)
        source_load_seconds = monotonic() - started
        report = run_saved_fifo_return_pilot(
            inputs, max_seconds=120.0 if args.max_seconds is None else args.max_seconds
        )
    elif args.return_comparison:
        snapshots, return_sources = load_return_inputs(args.source_artifact)
        source_load_seconds = monotonic() - started
        report = run_return_pilot(snapshots, return_sources)
    else:
        snapshots = load_snapshots(args.source_artifact)
        source_load_seconds = monotonic() - started
        report = run_pilot(snapshots)
    report.update(
        source_artifact=str(args.source_artifact), source_load_seconds=source_load_seconds
    )
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError("memory probe output exceeds 32 MiB; no result published")
    if args.output is not None:
        with args.output.open("xb") as handle:
            handle.write(payload)
    sys.stdout.write(payload.decode("utf-8"))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
