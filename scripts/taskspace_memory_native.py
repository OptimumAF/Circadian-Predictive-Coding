"""Optional frozen native demo composed from the bundled TRAIN stream.

Owns at most four bounded scene receipts and exclusive local capture. No collection,
teacher, fitting, promotion, probe selection or final evaluation is provided.
The default portable replay does not import this optional module.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import monotonic
from typing import Any

from scripts.maniskill_continuous_smoke import run_frozen_rollout
from scripts.push_cube_continuation_study import roll_push
from scripts.push_cube_memory_control import _encoded, _rejected, _require_native, _summary
from src.adapters.continuous_stream_sequence import run_sequence
from src.adapters.taskspace_memory_demo import (
    REPLAY_SCENES,
    _admit,
    _pin_readout,
    _require_readout,
    _readout_state,
    _retention_options,
    _RetentionOptions,
    run_demo,
    training_probe_outputs,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_train_context import pick_cube_training_context
from src.core.nearest_retention import NearestRetentionPolicy
from src.adapters.numpy_retained_nearest_bc import retained_costs

MAX_REPORT_BYTES = 3 * 1024**2


def run_native_training_probe(
    payload: dict, training, output: Path, *, training_input_receipt=None
) -> dict[str, Any]:
    """Observe original23 before/after the existing private probe; never promote.

    Why: NativeScene continues to allocate only ordinary TRAIN resets. Reserved
    observations use their explicit existing frozen port and probe_train role.
    At most two paid raw returns and file identities are kept, with no retry.
    """
    from src.app.probe_return_phase import paid_report_guard
    from src.core.pick_cube_probe_context import pick_cube_probe_binding

    _, failure, before, after = training_probe_outputs(output)
    phases = ("before_probe", "after_probe")
    paths = (before, after)
    report: dict[str, Any] = dict(
        definition="taskspace_memory_native_training_probe_v1",
        complete=False,
        error=None,
        workflow=None,
        native=[],
        attempted=[],
        returned=[],
        outcomes={},
        unreturned_native_work_unknown=False,
        training_input_receipt=deepcopy(training_input_receipt),
    )
    native = report["native"]
    attempted, returned_phases = report["attempted"], report["returned"]
    expected_attempted: list[str] = []
    expected_returned: list[str] = []
    outcomes_seal = _encoded(report["outcomes"])
    identity_seal = _encoded({key: report[key] for key in ("definition", "training_input_receipt")})
    proof: list[Any] = []
    workflow_guard = None
    saved_main = None
    rows = None
    binding = pick_cube_probe_binding("probe_train")

    def pure():
        if report["native"] is not native or len(native) != len(proof):
            raise RuntimeError("paid original23 native holder changed")
        if (
            report["attempted"] is not attempted
            or attempted != expected_attempted
            or report["returned"] is not returned_phases
            or returned_phases != expected_returned
            or _encoded(report["outcomes"]) != outcomes_seal
            or _encoded({key: report[key] for key in ("definition", "training_input_receipt")})
            != identity_seal
        ):
            raise RuntimeError("original23 paid phase, outcome or input receipt changed")
        if workflow_guard is not None:
            workflow_guard()
        for index, (entry, holder, raw, seal, _, _) in enumerate(proof):
            if (
                native[index] is not entry
                or entry.get("report") is not raw
                or holder.report is not raw
                or _encoded(entry) != seal
            ):
                raise RuntimeError("preceding original23 paid evidence changed or was replaced")

    def guard():
        pure()
        for _, _, _, _, path, saved in proof:
            if saved is not None and _read_capture(path, 1024**2) != saved:
                raise RuntimeError("preceding original23 capture changed or was replaced")
        if saved_main is not None and _read_capture(output, MAX_REPORT_BYTES) != saved_main:
            raise RuntimeError("original23 main capture changed or was replaced")
        pure()

    def save():
        nonlocal saved_main
        guard()
        raw = _encoded(report) + b"\n"
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError("original23 workflow report exceeds3MiB")
        with output.open("xb" if saved_main is None else "wb") as handle:
            handle.write(raw)
        saved_main = _read_capture(output, MAX_REPORT_BYTES)
        if saved_main[0] != raw:
            raise ValueError("original23 main capture differs from paid report")
        guard()

    def endpoint(readout, phase, check):
        nonlocal rows, outcomes_seal
        guard()
        index = len(proof)
        if index >= 2 or phase != phases[index]:
            raise ValueError("original23 probe endpoints repeated or reordered")

        def observe():
            nonlocal rows, outcomes_seal
            guard()
            current = readout._state.training_examples
            if index == 0:
                rows = current
            elif rows is None or current != rows + 18:
                raise ValueError("original23 private bank differs from one18-row append")
            report["attempted"].append(phase)
            expected_attempted.append(phase)
            save()
            check()
            holder = run_frozen_rollout(
                paths[index],
                learner=readout,
                probe_train_seed=23,
                action_history_recipe=PickCubeTaskSpaceHistoryRecipe(mode="executed"),
            )
            raw = holder.report
            report["returned"].append(phase)
            expected_returned.append(phase)
            entry = dict(phase=phase, report=raw)
            # Birth values are retained before file access, validation or cleanup.
            seal = _encoded(_rejected(entry))
            native.append(entry)
            proof.append((entry, holder, raw, seal, paths[index], None))
            save()
            saved = _read_capture(paths[index], 1024**2)
            guard()
            proof[index] = (entry, holder, raw, seal, paths[index], saved)
            if holder.corrections is not None or json.loads(saved[0]) != json.loads(_encoded(raw)):
                raise ValueError(
                    "original23 saved/returned evidence or correction authority differs"
                )
            _require_native(raw, push=False, updates=7 + index, rows=current)
            if (
                raw.get("role") != binding.role
                or type(raw.get("seed")) is not int
                or raw["seed"] != 23
                or raw["episode"].get("context") != binding.context
                or raw["episode"].get("episode_id") != "pick-cube-probe-train-smoke-23"
                or raw.get("scoring_only") is not True
                or raw.get("nearest_work_scope") != "validated_native_work"
            ):
                raise ValueError("original23 native role, inference identity or work scope differs")
            report["outcomes"][phase] = _summary(raw, push=False)
            outcomes_seal = _encoded(report["outcomes"])
            guard()

        check(observe)
        guard()

    save()
    try:
        returned = run_demo(
            payload, training_probe=training, endpoint_observer=endpoint, check_boundary=guard
        )
        report["workflow"] = returned
        paid_guard, trusted = paid_report_guard({"workflow": returned})

        def require_workflow():
            if report["workflow"] is not returned:
                raise RuntimeError("paid original23 workflow replaced")
            paid_guard()

        workflow_guard = require_workflow
        guard()
        if returned.get("complete") is not True or len(proof) != 2 or len(report["outcomes"]) != 2:
            raise RuntimeError(f"original23 workflow incomplete: {returned.get('error')}")
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["unreturned_native_work_unknown"] = expected_attempted != expected_returned
    try:
        save()
    except Exception as error:
        report["complete"] = False
        report["capture_error"] = f"{type(error).__name__}: {error}"
        report["native"] = [json.loads(row[3]) for row in proof]
        report["attempted"] = expected_attempted.copy()
        report["returned"] = expected_returned.copy()
        report["outcomes"] = json.loads(outcomes_seal)
        report.update(json.loads(identity_seal))
        if workflow_guard is not None:
            report["workflow"] = trusted["workflow"]
        raw = _encoded(report) + b"\n"
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError("original23 failure report exceeds3MiB") from error
        with failure.open("xb") as handle:
            handle.write(raw)
    return deepcopy(report)


@dataclass(frozen=True)
class NativeScene:
    """An existing TRAIN reset allocation, never an allocation of probe/final seeds."""

    task: str
    seed: int

    def __post_init__(self):
        if type(self.task) is not str or self.task not in ("pick", "push"):
            raise ValueError("native scene task must be exactly pick or push")
        if type(self.seed) is not int or self.seed not in (17, 19, 20):
            raise ValueError("native scene seed must be exactly TRAIN integer17,19,20")


def _scene_plan(development_scenes):
    if development_scenes is None:
        descriptors = tuple(
            NativeScene("push" if name.startswith("push") else "pick", seed)
            for name, seed in REPLAY_SCENES
        )
    else:
        if (
            type(development_scenes) is not tuple
            or not 1 <= len(development_scenes) <= 2
            or any(type(scene) is not NativeScene for scene in development_scenes)
        ):
            raise ValueError("development scenes require one or two exact NativeScene descriptors")
        descriptors = development_scenes
    plan = []
    for scene in descriptors:
        # Why: snapshot validated primitive values, not caller-owned descriptor authority.
        owned = NativeScene(scene.task, scene.seed)
        context = (
            pick_cube_training_context(owned.seed)
            if owned.task == "pick"
            else f"push-cube-train-{owned.seed}"
        )
        plan.append((f"{owned.task}{owned.seed}", owned.seed, owned.task, context))
    if len({row[0] for row in plan}) != len(plan):
        raise ValueError("development scenes must be distinct")
    return tuple(plan)


def _read_capture(path, maximum):
    if path.is_symlink():
        raise ValueError("native capture cannot be a symbolic link")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("native capture exceeds its byte allowance")
    info = path.stat()
    return raw, (info.st_dev, info.st_ino)


def run_native_demo(
    payload: dict,
    output: Path,
    *,
    development_scenes: tuple[NativeScene, ...] | None = None,
    retention: NearestRetentionPolicy | None = None,
    max_prototypes: int = 256,
) -> dict[str, Any]:
    """Rebuild TRAIN memory and run bounded fresh, read-only native endpoints.

    A complete negative task score is valid. Execution/capture failures stop the
    remaining scenes and retain all observed paid returns. Outputs must be fresh.
    """
    retention = _retention_options(retention, max_prototypes)
    if retention is not None and development_scenes is not None:
        raise ValueError("retained TRAIN mode and development probe are incompatible")
    plan = _scene_plan(development_scenes)
    development = development_scenes is not None
    paths = {
        scene: output.parent / f"{output.stem}-{scene}.json"
        for scene, _, task, _ in plan
        if task == "pick"
    }
    failure = output.with_suffix(".failure.json")
    destinations = (output, failure, *paths.values())
    if (
        len(set(destinations)) != 2 + len(paths)
        or not output.parent.is_dir()
        or any(path.exists() or path.is_symlink() for path in destinations)
    ):
        raise ValueError("native demo requires distinct fresh outputs in an existing directory")
    # Admit the entire bounded bundle before constructing a learner or simulator.
    # The recordings authenticate their own schema, but never select native work.
    episodes, _, provenance = _admit(payload)
    if development:
        represented = {episode.context for episode in episodes}
        if any(context in represented for _, _, _, context in plan):
            raise ValueError("development context is already represented in the TRAIN bank")
    rows = sum(len(episode.transitions) for episode in episodes)
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="taskspace_memory_native_v1",
        mode="native_fresh_rollout",
        complete=False,
        error=None,
        provenance=provenance,
        attempted=[],
        returned=[],
        native=[],
        stream=None,
        outcomes={},
        unreturned_stage_work_unknown=False,
        work=dict(
            neural_initializations=0,
            optimizer_updates=0,
            teacher_queries=0,
            probe_updates=0,
            final_reads=0,
        ),
    )
    if development:
        report.update(
            definition="taskspace_memory_development_v1",
            mode="unrepresented_train_contexts",
            development_scenes=[
                dict(scene=name, seed=seed, task=task, context=context, role="train")
                for name, seed, task, context in plan
            ],
            classification="fixed TRAIN resets unrepresented in admitted bank; no global unseen-data claim",
        )
    if retention is not None:
        report.update(
            definition="taskspace_retained_memory_native_v1",
            mode="retained_native_training_rollout",
            classification="four fixed known TRAIN scenes; no generalization claim",
            retention_options=dict(rule=asdict(retention), max_prototypes=max_prototypes),
        )
    scene_seal = _encoded(report.get("development_scenes"))
    identity_keys = ("definition", "mode", "classification", "retention_options")
    identity_seal = _encoded({key: report.get(key) for key in identity_keys})
    native = report["native"]
    proof: list[Any] = []
    saved_main = None
    readout = pin = metadata_owner = metadata_seal = None
    stream_owner = stream_seal = None
    outcomes_seal = _encoded({})
    retention_seal = None
    final_seen = False

    def pure():
        if (development or retention is not None) and _encoded(
            {key: report.get(key) for key in identity_keys}
        ) != identity_seal:
            raise ValueError("declared demo classification or retention changed")
        if _encoded(report.get("development_scenes")) != scene_seal:
            raise ValueError("declared development scene evidence changed")
        if report["native"] is not native or len(native) != len(proof):
            raise ValueError("preceding native receipt list changed or was replaced")
        if readout is not None:
            _require_readout(readout, pin)
            if _encoded(metadata_owner) != metadata_seal:
                raise ValueError("native checkpoint metadata changed")
        for index, (owner, raw, seal, path, saved, holder) in enumerate(proof):
            if (
                native[index] is not owner
                or owner.get("scene") != plan[index][0]
                or owner.get("report") is not raw
                or _encoded(raw) != seal
                or holder is not None
                and holder.report is not raw
            ):
                raise ValueError("preceding paid native report changed or was replaced")
        if stream_seal is not None and (
            report["stream"] is not stream_owner or _encoded(report["stream"]) != stream_seal
        ):
            raise ValueError("paid stream report changed")
        if _encoded(report["outcomes"]) != outcomes_seal:
            raise ValueError("derived native outcomes changed")
        if retention_seal is not None and _encoded(report.get("retention")) != retention_seal:
            raise ValueError("retained memory cost evidence changed")

    def guard():
        pure()
        if saved_main is not None and _read_capture(output, MAX_REPORT_BYTES) != saved_main:
            raise ValueError("preceding main capture changed or was replaced")
        for _, _, _, path, saved, _ in proof:
            if path is not None and _read_capture(path, 1024**2) != saved:
                raise ValueError("preceding Pick capture changed or was replaced")
        pure()

    def save(*, verify=True):
        nonlocal saved_main
        if verify:
            guard()
        raw = _encoded(report) + b"\n"
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError("native main report exceeds3MiB")
        with output.open("xb" if saved_main is None else "wb") as handle:
            handle.write(raw)
        saved_main = _read_capture(output, MAX_REPORT_BYTES)
        if saved_main[0] != raw:
            raise ValueError("native capture did not preserve the current report")
        pure()

    def restore_paid():
        # These seals were captured at return, before later filesystem access.
        report["native"] = [
            dict(scene=plan[i][0], report=json.loads(record[2])) for i, record in enumerate(proof)
        ]
        if stream_seal is not None:
            report["stream"] = json.loads(stream_seal)
        report["outcomes"] = json.loads(outcomes_seal)
        if retention_seal is not None:
            report["retention"] = json.loads(retention_seal)
        if development:
            report["development_scenes"] = json.loads(scene_seal)
        if development or retention is not None:
            report.update(json.loads(identity_seal))

    def observe(learner, metadata):
        nonlocal \
            final_seen, \
            readout, \
            pin, \
            metadata_owner, \
            metadata_seal, \
            outcomes_seal, \
            retention_seal
        guard()
        if metadata["phase"] != "after_release" or metadata["packet_index"] != 5:
            return {}
        if final_seen:
            raise ValueError("native final checkpoint repeated")
        final_seen = True
        readout, pin = learner, _pin_readout(learner)
        metadata_owner, metadata_seal = metadata, _encoded(metadata)
        state = _readout_state(pin)
        stored_rows = state.retained_rows if retention is not None else rows
        if (
            state.completed_updates != 7
            or state.training_examples != rows
            or metadata["completed_updates"] != 7
            or metadata["prototype_rows"] != stored_rows
            or metadata["known_episode_ids"] != [episode.episode_id for episode in episodes]
        ):
            raise ValueError("native final checkpoint differs from the admitted TRAIN bank")
        if retention is not None:
            report["retention"] = retained_costs(state)
            retention_seal = _encoded(report["retention"])
        for scene, seed, task, context in plan:
            guard()
            report["attempted"].append(scene)
            save()
            path = paths.get(scene)
            holder = None
            if path is None:
                raw = roll_push(learner, label="portable-memory-demo", seed=seed)
            else:
                holder = run_frozen_rollout(
                    path,
                    learner=learner,
                    train_seed=seed,
                    action_history_recipe=PickCubeTaskSpaceHistoryRecipe(mode="executed"),
                )
                raw = holder.report
            report["returned"].append(scene)
            # Why this: keep the birth return even when later IO or validation
            # detects changed evidence. Four bounded records, no success chain.
            seal = _encoded(_rejected(raw))
            report["pending_paid_return"] = dict(scene=scene, report=json.loads(seal))
            save()
            if _encoded(raw) != seal or holder is not None and holder.report is not raw:
                raise ValueError("current paid native return changed during capture")
            saved = None if path is None else _read_capture(path, 1024**2)
            guard()
            if _encoded(raw) != seal or holder is not None and holder.report is not raw:
                raise ValueError("current paid native return changed during file readback")
            entry = dict(scene=scene, report=raw)
            native.append(entry)
            proof.append((entry, raw, seal, path, saved, holder))
            del report["pending_paid_return"]
            save()
            # Native dataclass tuples become JSON arrays. Compare stored content
            # in that representation; raw report/holder owners stay pinned above.
            if holder is not None and (
                holder.corrections is not None
                or saved is None
                or _encoded(json.loads(saved[0])) != _encoded(json.loads(seal))
            ):
                raise ValueError(
                    "frozen Pick saved/returned result or correction authority differs"
                )
            _require_native(
                raw,
                push=path is None,
                updates=7,
                rows=rows,
                **(dict(prototype_rows=stored_rows) if retention is not None else {}),
            )
            if (
                raw["role"] != "train"
                or type(raw["seed"]) is not int
                or raw["seed"] != seed
                or (raw["episode"]["context"] != context)
            ):
                raise ValueError("native return differs from the declared TRAIN scene")
            report["outcomes"][scene] = _summary(raw, push=path is None)
            outcomes_seal = _encoded(report["outcomes"])
            guard()
        return dict(native_scenes=len(plan), final_checkpoint_only=True)

    options: _RetentionOptions = (
        dict(retention=retention, max_prototypes=max_prototypes) if retention is not None else {}
    )
    save()
    try:
        report["attempted"].append("stream")
        save()
        report["stream"] = run_sequence(
            episodes[0],
            episodes[1:],
            observe,
            fixed_input=True,
            maximum_releases=6,
            **options,
        )
        stream_owner, stream_seal = report["stream"], _encoded(report["stream"])
        report["returned"].append("stream")
        guard()
        if (
            report["stream"].get("complete") is not True
            or not final_seen
            or len(proof) != len(plan)
            or len(report["outcomes"]) != len(plan)
        ):
            raise RuntimeError(
                f"native stream or endpoints incomplete: {report['stream'].get('error')}"
            )
        report["complete"] = True
    except Exception as error:
        report["complete"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["unreturned_stage_work_unknown"] = any(
            name not in report["returned"] for name in report["attempted"]
        )
        try:
            guard()
        except Exception as integrity_error:
            # Keep trusted paid bytes on corruption; never bless changed owners.
            # Ordinary execution failures can still finish the original capture.
            report["integrity_error"] = f"{type(integrity_error).__name__}: {integrity_error}"
            restore_paid()
    report["elapsed_seconds"] = monotonic() - started
    try:
        save()
    except Exception as error:
        report["complete"] = False
        report["capture_error"] = f"{type(error).__name__}: {error}"
        restore_paid()
        raw = _encoded(report) + b"\n"
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError("native failure report exceeds3MiB") from error
        with failure.open("xb") as handle:
            handle.write(raw)
    return deepcopy(report)
