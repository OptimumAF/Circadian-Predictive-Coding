"""Fixed TRAIN memory control composed from the existing stream and native ports.

Owns source binding, two predefined readouts and bounded local evidence files.
Does not fit features, select probes, query teachers, promote, or access final.
The BP comparison shares information, not objective, exposure or computation.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from typing import Any

from scripts.diagnose_aggregated_supervision import (
    SOURCE,
    _bound_bytes,
    _require_original_unchanged,
    read_aggregated_inputs,
)
from scripts.maniskill_continuous_smoke import run_frozen_rollout
from scripts.push_cube_continuation_study import _same_push_reset, roll_push
from src.adapters.continuous_stream_sequence import _same, run_sequence
from src.adapters.numpy_nearest_continuous_bc import _snapshot
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe

READOUTS = {1: ("before", 3, 80), 4: ("after", 6, 198)}
SCENES = ("push17", "pick17", "pick19", "pick20")
SCORES = ("final_success", "any_success", "sparse_return")


def _encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf8")


def _rejected(value):
    """Preserve malformed evidence without letting refusal break capture."""
    try:
        return json.loads(_encoded(value))
    except Exception:
        return dict(type=type(value).__name__, malformed_repr=repr(value)[:2048])


def _read_saved(path: Path, work) -> bytes:
    work["attempts"] += 1
    with path.open("rb") as handle:
        raw = handle.read(1024**2 + 1)
    work["returns"] += 1
    work["bytes"] += len(raw)
    if len(raw) > 1024**2:
        raise ValueError("saved Pick evidence exceeds1MiB")
    return raw


def _pick_reset(value):
    return dict(
        observation=value["episode"]["transitions"][0]["observation"],
        metadata={
            key: value[key]
            for key in ("seed", "role", "task", "context", "controller", "observation_recipe")
        },
        history_recipe=value["action_history"]["recipe"],
    )


def _summary(raw, *, push):
    rows = raw["transitions"] if push else raw["episode"]["transitions"]
    work = raw["work"]
    physics = work["completed_control_physics_steps"]
    return dict(
        **{key: raw[key] for key in SCORES},
        controls=len(rows),
        control_physics_steps=physics,
        distance_candidates=raw.get("nearest_distance_candidates_completed", 0),
        distance_feature_values=raw.get("nearest_distance_feature_values_completed", 0),
    )


def _require_native(raw, *, push, updates, rows):
    transitions = raw["transitions"] if push else raw["episode"]["transitions"]
    count = len(transitions)
    if (
        raw["complete"] is not True
        or raw["source_updates"] != updates
        or type(raw["source_updates"]) is not int
        or raw["source_training_rows"] != rows
        or type(raw["source_training_rows"]) is not int
        or not 1 <= count <= 50
        or raw["prediction_attempts"] != count
        or raw["prediction_returns"] != count
        or raw["nearest_prediction_attempts"] != count
        or raw["nearest_prediction_returns"] != count
        or raw["nearest_distance_candidate_upper_bound"] != count * rows
        or raw["nearest_distance_candidates_completed"] != count * rows
        or raw["nearest_distance_feature_upper_bound"] != count * rows * 12
        or raw["nearest_distance_feature_values_completed"] != count * rows * 12
        or len(raw["nearest_selected_indices"]) != count
    ):
        raise ValueError("frozen memory endpoint or paid retrieval work differs")
    if push:
        if (
            raw["source_unchanged"] is not True
            or raw["unreturned_prediction_possible"] is not False
            or raw["unreturned_native_work_unknown"] is not False
            or raw["unvalidated_nearest_prediction_work_unknown"] is not False
            or raw["control_attempts"] != count
            or raw["control_returns"] != count
        ):
            raise ValueError("Push endpoint has incomplete or unknown work")
    elif (
        raw["frozen_state_exact"] is not True
        or raw["original_owner_exact"] is not True
        or raw["action_history_exact"] is not True
        or raw["optimizer_updates"] != 0
        or raw["work"]["possible_failed_physics_steps"] != 0
    ):
        raise ValueError("Pick endpoint lost its frozen owner or work binding")


def _require_stream(stream):
    if (
        stream["complete"] is not True
        or stream["fixed_input"] is not True
        or stream["release_scope"]["maximum_releases"] != 5
        or stream["experiment_state"] != dict(completed_updates=5, next_tick=15)
        or stream["model_state"]
        != dict(
            source_publications=1,
            candidate_publications=6,
            candidate_prototype_rows=198,
            durable_checkpoint=False,
        )
        or len(stream["checkpoints"]) != 11
        or len(stream["applied"]) != 5
        or stream["work"]["validated_updates"] != 5
        or stream["work"]["fit_attempts"] != 0
    ):
        raise ValueError("fixed five-release memory stream incomplete")
    for index, receipt in enumerate(stream["applied"]):
        if (
            receipt["observed_at"] != 3 * index
            or receipt["arrived_at"] != 3 * index + 2
            or receipt["applied_at"] != 3 * index + 2
            or receipt["update_number"] != index + 1
        ):
            raise ValueError("paid memory receipt clock or update changed")


def run_control(output: Path, *, source: Path = SOURCE) -> dict[str, Any]:
    """Run the frozen two-checkpoint control once, retaining partial paid work."""
    paths = {
        (label, seed): output.parent / f"{output.stem}-{label}-pick-{seed}.json"
        for label in ("before", "after")
        for seed in (17, 19, 20)
    }
    if not output.parent.is_dir() or any(
        path.exists() or path.is_symlink() for path in (output, *paths.values())
    ):
        raise ValueError("memory control requires fresh outputs in an existing directory")
    started = monotonic()
    report: dict[str, Any] = dict(
        protocol="push_cube_fixed_input_memory_control_v1",
        complete=False,
        error=None,
        source_bytes_unchanged=False,
        attempted=[],
        observed_stage_returns=[],
        readouts=[],
        unreturned_stage_work_unknown=False,
        work=dict(
            accepted_study_reads=dict(attempts=0, returns=0, bytes=0),
            original_correction_reads=dict(attempts=0, returns=0, bytes=0),
            saved_evidence_reads=dict(attempts=0, returns=0, bytes=0),
            model_fits=0,
            neural_initializations=0,
            random_draws=0,
            optimizer_updates=0,
            teacher_queries=0,
            probe_updates=0,
        ),
    )
    last_saved: bytes | None = None
    proof: list[Any] = []
    tail_guard = lambda: None
    readout_owners: list[dict[str, Any]] = []
    readout_metadata: list[bytes] = []
    published_readouts = report["readouts"]
    stream_pin: tuple[Any, bytes] | None = None

    def save():
        nonlocal last_saved
        if last_saved is not None and output.read_bytes() != last_saved:
            raise ValueError("preceding paid main evidence changed on disk")
        tail_guard()
        raw = _encoded(report) + b"\n"
        if len(raw) > 6 * 1024**2:
            raise ValueError("memory control report exceeds6MiB")
        with output.open("xb" if last_saved is None else "wb") as handle:
            handle.write(raw)
        last_saved = raw

    def restore_rejected():
        rejected = []
        for owner, scene, raw, encoded, path, saved, holder in proof:
            current = owner["scenes"].get(scene)
            try:
                changed = current is not raw or _encoded(current) != encoded
            except Exception:
                changed = True
            if changed:
                rejected.append(dict(scene=scene, changed_report=_rejected(current)))
            owner["scenes"][scene] = json.loads(encoded)
        if rejected:
            report["rejected_prior_evidence"] = rejected
        for checkpoint, expected in zip(readout_owners, readout_metadata, strict=True):
            label, metadata = json.loads(expected)
            if _rejected((checkpoint.get("label"), checkpoint.get("metadata"))) != [
                label,
                metadata,
            ]:
                report.setdefault("rejected_checkpoint_metadata", []).append(_rejected(checkpoint))
            checkpoint.update(label=label, metadata=metadata)
        if stream_pin is not None:
            owner, expected = stream_pin
            if report.get("stream") is not owner or _rejected(owner) != json.loads(expected):
                report["rejected_stream_evidence"] = _rejected(report.get("stream"))
            owner.clear()
            owner.update(json.loads(expected))
            report["stream"] = owner

    save()
    try:
        report["attempted"].append("input-admission")
        groups, receipt = read_aggregated_inputs(source)
        report["sources"] = receipt
        report["inputs"] = [dict(group=name, episode=asdict(e)) for name, e in groups]
        report["observed_stage_returns"].append("input-admission")
        report["work"]["source_derived_original_correction_reads"] = 2
        report["work"]["source_derived_original_correction_bytes"] = (
            2 * receipt["original_correction"]["bytes"]
        )
        baseline = json.loads(_bound_bytes(source, work=report["work"]["accepted_study_reads"]))
        source_pin = _encoded((report["sources"], report["inputs"], baseline))

        def pure_guard():
            if source_pin != _encoded((report["sources"], report["inputs"], baseline)):
                raise ValueError("admitted input or baseline evidence changed")
            if (
                report["readouts"] is not published_readouts
                or report["readouts"] != readout_owners
                or any(a is not b for a, b in zip(report["readouts"], readout_owners, strict=True))
            ):
                raise ValueError("preceding paid checkpoint owner was replaced")
            for checkpoint, expected in zip(readout_owners, readout_metadata, strict=True):
                if _encoded((checkpoint["label"], checkpoint["metadata"])) != expected:
                    raise ValueError("preceding memory checkpoint metadata changed")
            if stream_pin is not None and (
                report.get("stream") is not stream_pin[0]
                or _encoded(report["stream"]) != stream_pin[1]
            ):
                raise ValueError("preceding paid stream evidence changed or was replaced")
            for owner, scene, raw, encoded, path, saved, holder in proof:
                if owner["scenes"].get(scene) is not raw or _encoded(raw) != encoded:
                    raise ValueError("preceding paid native evidence changed or was replaced")
                if holder is not None and holder.report is not raw:
                    raise ValueError("preceding Pick return holder was replaced")

        tail_guard = pure_guard

        def guard():
            pure_guard()
            _bound_bytes(source, work=report["work"]["accepted_study_reads"])
            _require_original_unchanged(
                receipt["original_correction"], report["work"]["original_correction_reads"]
            )
            for owner, scene, raw, encoded, path, saved, holder in proof:
                if (
                    path is not None
                    and _read_saved(path, report["work"]["saved_evidence_reads"]) != saved
                ):
                    raise ValueError("preceding saved Pick evidence changed")
            pure_guard()

        def observe(learner, metadata):
            guard()
            if metadata["phase"] != "after_release" or metadata["packet_index"] not in READOUTS:
                return dict(native_readout=False)
            label, updates, rows = READOUTS[metadata["packet_index"]]
            if metadata["prototype_rows"] != rows or metadata["completed_updates"] != updates:
                raise ValueError("predefined memory checkpoint differs")
            if len(readout_owners) != (0 if label == "before" else 1):
                raise ValueError("memory readout order or multiplicity differs")
            pin = _snapshot(learner._state, learner._policy)
            owners = (
                learner._state,
                learner._policy,
                learner._state.prototype_features,
                learner._state.prototype_actions,
            )

            def frozen():
                current = (
                    learner._state,
                    learner._policy,
                    learner._state.prototype_features,
                    learner._state.prototype_actions,
                )
                if any(a is not b for a, b in zip(current, owners, strict=True)) or not _same(
                    pin, _snapshot(learner._state, learner._policy)
                ):
                    raise ValueError("detached checkpoint state or owner changed")

            checkpoint = dict(label=label, metadata=deepcopy(metadata), scenes={})
            report["readouts"].append(checkpoint)
            readout_owners.append(checkpoint)
            readout_metadata.append(_encoded((label, checkpoint["metadata"])))
            for scene in SCENES:
                guard()
                frozen()
                stage = f"{label}-{scene}"
                report["attempted"].append(stage)
                save()
                push = scene == "push17"
                path = None if push else paths[label, int(scene[4:])]
                if push:
                    raw = roll_push(learner, label=f"memory-{label}")
                    corrections = None
                    rolled = None
                else:
                    assert path is not None
                    rolled = run_frozen_rollout(
                        path,
                        learner=learner,
                        train_seed=int(scene[4:]),
                        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(mode="executed"),
                    )
                    raw, corrections = rolled.report, rolled.corrections
                report["observed_stage_returns"].append(stage)
                report["pending_paid_return"] = dict(stage=stage, report=_rejected(raw))
                # An opaque later call must not replace the trusted preceding
                # capture with changed history. Keep its paid return on refusal.
                guard()
                frozen()
                checkpoint["scenes"][scene] = raw
                encoded = _encoded(raw)
                saved = (
                    None
                    if path is None
                    else _read_saved(path, report["work"]["saved_evidence_reads"])
                )
                proof.append((checkpoint, scene, raw, encoded, path, saved, rolled))
                # Why: save paid returns before inspecting success, provenance or costs.
                save()
                del report["pending_paid_return"]
                if path is not None and (saved is None or json.loads(saved) != json.loads(encoded)):
                    raise ValueError("saved and returned Pick evidence differ")
                frozen()
                guard()
                if corrections is not None:
                    raise ValueError("frozen memory rollout returned correction authority")
                _require_native(raw, push=push, updates=updates, rows=rows)
                old = baseline["native_push"] if push else baseline["pick"][SCENES.index(scene) - 1]
                same = (
                    _same_push_reset(raw, old)
                    if push
                    else _encoded(_pick_reset(raw)) == _encoded(_pick_reset(old))
                )
                if label == "after":
                    before = readout_owners[0]["scenes"][scene]
                    same = same and (
                        _same_push_reset(raw, before)
                        if push
                        else _encoded(_pick_reset(raw)) == _encoded(_pick_reset(before))
                    )
                if not same:
                    raise ValueError("native reset differs from accepted BP or earlier checkpoint")
            return dict(
                native_readout=True,
                label=label,
                summaries={
                    scene: _summary(raw, push=scene == "push17")
                    for scene, raw in checkpoint["scenes"].items()
                },
                full_reports="retained by fixed outer control; no runtime authority returned",
            )

        packets = dict(groups)
        report["attempted"].append("stream")
        report["stream"] = run_sequence(
            packets["a"],
            tuple(
                packets[name]
                for name in ("b", "train20", "push", "corrections", "current_corrections")
            ),
            observer=observe,
            fixed_input=True,
            maximum_releases=5,
        )
        stream_pin = (report["stream"], _encoded(report["stream"]))
        report["observed_stage_returns"].append("stream")
        save()
        guard()
        _require_stream(report["stream"])
        if len(readout_owners) != 2 or any(
            tuple(item["scenes"]) != SCENES for item in readout_owners
        ):
            raise ValueError("both complete predefined native checkpoints are required")
        before, after = (item["scenes"] for item in readout_owners)
        report["comparisons"] = dict(
            direction="after-minus-before; higher success/return is better; costs descriptive",
            scope="known TRAIN resets, same198 information; unequal objectives/exposure/storage/compute, no CPC or generalization claim",
            acquisition_retention={},
            final_memory_minus_accepted_bp128={},
        )
        for scene in SCENES:
            push = scene == "push17"
            old = baseline["native_push"] if push else baseline["pick"][SCENES.index(scene) - 1]
            current = _summary(after[scene], push=push)
            for key, previous in (
                ("acquisition_retention", before[scene]),
                ("final_memory_minus_accepted_bp128", old),
            ):
                prior = _summary(previous, push=push)
                report["comparisons"][key][scene] = {
                    name: current[name] - prior[name] for name in current
                }
        report["elapsed_seconds"] = monotonic() - started
        guard()
        report["source_bytes_unchanged"] = True
        report["complete"] = True
    except Exception as error:
        report["complete"] = False
        report["error"] = dict(type=type(error).__name__, message=str(error)[:1024])
        restore_rejected()
        report["unreturned_stage_work_unknown"] = any(
            stage not in report["observed_stage_returns"] for stage in report["attempted"]
        )
        # No more work is admitted after failure; preserve trusted sealed JSON
        # and the refused return without invoking the invalid live owners again.
        tail_guard = lambda: None
    if "elapsed_seconds" not in report:
        report["elapsed_seconds"] = monotonic() - started
    try:
        save()
    except Exception as error:
        report["complete"] = False
        report["save_error"] = dict(type=type(error).__name__, message=str(error)[:1024])
        restore_rejected()
        # Preserve replaced evidence; a fresh failure file never overwrites it.
        failure = output.with_suffix(".failure.json")
        with failure.open("xb") as handle:
            handle.write(_encoded(report) + b"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_control(args.output)
    print(
        json.dumps(
            dict(
                complete=report["complete"],
                error=report["error"],
                elapsed_seconds=report["elapsed_seconds"],
            )
        )
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
