"""Compare saved Pick19 supervision and replay a failed trace without updates.

Composes existing qualification, history, consistency and nearest kernels.
Outputs are TRAIN data diagnostics, not counterfactual rewards or new labels.
No model, fit, teacher, simulator, probe, promotion or final access occurs.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
from time import monotonic
from typing import Any

import numpy as np

from scripts.diagnose_learned_robot_control import _episode
from scripts.run_native_stream_demo import _read_readiness_teacher
from src.core.continuous_bc_diagnostics import (
    _InputTargetConsistencyWork,
    _native_consistency_targets,
    continuous_input_target_consistency,
)
from src.core.continuous_demonstration import (
    ContinuousEpisode,
    continuous_training_pairs,
    copy_continuous_episode,
)
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.nearest_continuous_bc import nearest_continuous_actions
from src.core.pick_cube_action_history import (
    action_history_training_episode,
    validate_action_history_training_episode,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe

MEMORY = Path("artifacts/runs/push-memory-control-20261021/run.json")
TEACHER = Path("artifacts/runs/state-readiness-acquisition-20261020/teacher-19.json")
MEMORY_SHA256 = "0ee63292661295619d485b074d7d617619c136ae026bcdb4294712c8f34e5dae"
TEACHER_SHA256 = "36addcef50a41a7ad1acf92d401645d89b2cdb8f53a88fba7cb421d89c9edc4a"
BANK_ORDER = ("a", "b", "train20", "push", "corrections", "current_corrections")


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _read(path, sha256, work):
    work["attempts"] += 1
    with path.open("rb") as handle:
        raw = handle.read(1024**2 + 1)
    work["returns"] += 1
    work["bytes"] += len(raw)
    if len(raw) > 1024**2 or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("accepted Pick19 source bytes differ")
    return raw


def _require_sources(receipt, work):
    for source in receipt["sources"]:
        raw = _read(Path(source["path"]), source["sha256"], work)
        if len(raw) != source["bytes"]:
            raise ValueError("accepted Pick19 source length changed")


def read_pick19_inputs(*, work=None):
    """Own two accepted byte-bound sources; reuse the teacher's real qualifier."""
    work = dict(attempts=0, returns=0, bytes=0) if work is None else work
    memory_bytes = _read(MEMORY, MEMORY_SHA256, work)
    teacher_bytes = _read(TEACHER, TEACHER_SHA256, work)
    memory = json.loads(memory_bytes)
    if (
        memory["protocol"] != "push_cube_fixed_input_memory_control_v1"
        or memory["complete"] is not True
        or memory["error"] is not None
        or memory["source_bytes_unchanged"] is not True
        or memory["unreturned_stage_work_unknown"] is not False
        or memory["stream"]["complete"] is not True
        or memory["stream"]["model_state"]["candidate_prototype_rows"] != 198
    ):
        raise ValueError("accepted198 memory scope is incomplete")
    groups = {item["group"]: _episode(item["episode"]) for item in memory["inputs"]}
    if tuple(groups) != ("current_corrections", "corrections", "push", "a", "b", "train20"):
        raise ValueError("accepted memory input groups differ")
    bank = tuple((name, groups[name]) for name in BANK_ORDER)
    if tuple(len(episode.transitions) for _, episode in bank) != (15, 50, 15, 18, 50, 50):
        raise ValueError("accepted198 bank group sizes differ")
    recipe = PickCubeTaskSpaceHistoryRecipe(mode="executed")
    teacher, teacher_receipt = _read_readiness_teacher(TEACHER, 19, _payload=teacher_bytes)
    b22 = action_history_training_episode(teacher, recipe)
    failed = memory["readouts"][1]["scenes"]["pick19"]
    if (
        memory["readouts"][1]["label"] != "after"
        or failed["seed"] != 19
        or failed["role"] != "train"
        or failed["complete"] is not True
        or failed["source_updates"] != 6
        or failed["source_training_rows"] != 198
        or failed["frozen_state_exact"] is not True
        or failed["original_owner_exact"] is not True
        or failed["action_history_exact"] is not True
        or failed["final_success"] is not False
        or failed["any_success"] is not False
        or failed["sparse_return"] != 0
        or failed["action_history"]["recipe"] != asdict(recipe)
    ):
        raise ValueError("accepted failed Pick19 readout differs")
    trace = action_history_training_episode(_episode(failed["episode"]), recipe)
    if (len(b22.transitions), len(trace.transitions)) != (22, 50):
        raise ValueError("complete Pick19 demonstration/trace sizes differ")
    receipt = dict(
        sources=[
            dict(name=name, path=str(path), sha256=sha, bytes=len(raw))
            for name, path, sha, raw in (
                ("memory", MEMORY, MEMORY_SHA256, memory_bytes),
                ("teacher", TEACHER, TEACHER_SHA256, teacher_bytes),
            )
        ],
        teacher_qualification=teacher_receipt,
        recipe=asdict(recipe),
        bank_order=list(BANK_ORDER),
        trace_scope="observed executed commands, not new supervision",
    )
    return b22, groups["b"], trace, bank, tuple(failed["nearest_selected_indices"]), receipt


def _admit(b22, b50, trace, bank, selected_indices):
    originals = (b22, b50, trace)
    if any(type(e) is not ContinuousEpisode or e.role != "train" for e in originals):
        raise ValueError("Pick19 diagnosis requires exact released TRAIN episodes")
    owned = tuple(copy_continuous_episode(e) for e in originals)
    if (
        any(e.spec != cube_taskspace_spec() or not 1 <= len(e.transitions) <= 50 for e in owned)
        or b22.episode_id == b50.episode_id
        or len(b22.transitions) + len(b50.transitions) > 100
        or type(bank) is not tuple
        or not 2 <= len(bank) <= 6
    ):
        raise ValueError("Pick19 packets require bounded D12 TRAIN groups")
    checked = []
    for item in bank:
        if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or not item[0]:
            raise ValueError("bank requires named complete TRAIN episodes")
        name, e = item
        if (
            type(e) is not ContinuousEpisode
            or e.role != "train"
            or e.spec != b50.spec
            or not 1 <= len(e.transitions) <= 50
        ):
            raise ValueError("bank requires bounded same-spec TRAIN episodes")
        checked.append((name, copy_continuous_episode(e)))
    if (
        len({name for name, _ in checked}) != len(checked)
        or len({e.episode_id for _, e in checked}) != len(checked)
        or sum(len(e.transitions) for _, e in checked) > 198
        or sum(e == owned[1] for _, e in checked) != 1
    ):
        raise ValueError("bank identities or B50 membership differ")
    count = sum(len(e.transitions) for _, e in checked)
    if (
        type(selected_indices) is not tuple
        or len(selected_indices) != len(trace.transitions)
        or any(type(i) is not int or not 0 <= i < count for i in selected_indices)
    ):
        raise ValueError("recorded indices must be native counts within the full bank")
    validate_action_history_training_episode(
        owned[2], PickCubeTaskSpaceHistoryRecipe(mode="executed")
    )
    # B50 has desired corrective targets at original executed inputs. Its history
    # must never be regenerated from those hypothetical target actions.
    for row in owned[2].transitions:
        if tuple(struct.unpack("!4f", struct.pack("!4f", *row.action))) != row.action:
            raise ValueError("trace actions must be original native float32 commands")
    return (*owned, tuple(checked))


def _prediction(value):
    return dict(
        actions=value.actions.tolist(),
        selected_indices=list(value.selected_indices),
        distance_candidates=value.distance_candidates,
        distance_feature_values=value.distance_feature_values,
    )


def analyze_pick19(b22, b50, trace, bank, selected_indices):
    """Retain all pairs and trace steps; offline actions imply no rollout outcome."""
    b22, b50, trace, bank = _admit(b22, b50, trace, bank, selected_indices)
    pair_work = _InputTargetConsistencyWork()
    cast_work = _InputTargetConsistencyWork()
    work = dict(
        kernel_calls_attempted=0,
        kernel_calls_returned=0,
        nearest_calls_attempted=0,
        nearest_calls_returned=0,
        nearest_query_rows=0,
        nearest_candidate_upper_bound=0,
        nearest_feature_upper_bound=0,
        nearest_candidates_completed=0,
        nearest_features_completed=0,
    )
    report: dict[str, Any] = dict(
        complete=False,
        error=None,
        consistency=None,
        replay=None,
        b22_readout=None,
        trace_rows=[],
        work=work,
        unknown_unreturned_kernel_work=False,
        unknown_unreturned_nearest_work=False,
    )
    spec = b22.spec
    try:
        x22, y22 = continuous_training_pairs(b22, role="train")
        x50, y50 = continuous_training_pairs(b50, role="train")
        work["kernel_calls_attempted"] += 1
        consistency = continuous_input_target_consistency(
            np.asarray((*x22, *x50), np.float64),
            np.asarray((*y22, *y50), np.float64),
            spec,
            _work=pair_work,
        )
        work["kernel_calls_returned"] += 1
        report["consistency"] = asdict(consistency)
        nearest_cross: dict[int, tuple[float, int]] = {}
        cross_pairs = 0
        for pair in consistency.pairs:
            if pair.first_row < len(x22) <= pair.second_row:
                cross_pairs += 1
                for query, other in (
                    (pair.first_row, pair.second_row),
                    (pair.second_row, pair.first_row),
                ):
                    choice = (pair.input_l2, other)
                    if query not in nearest_cross or choice < nearest_cross[query]:
                        nearest_cross[query] = choice
        work.update(cross_group_pairs=cross_pairs, cross_group_nearest_comparisons=2 * cross_pairs)
        report["cross_group_nearest"] = [
            dict(
                global_row=row,
                other_global_row=nearest_cross[row][1],
                input_l2=nearest_cross[row][0],
            )
            for row in sorted(nearest_cross)
        ]
        queries = np.asarray([row.observation for row in trace.transitions], np.float64)
        bindings = [
            (name, index) for name, episode in bank for index in range(len(episode.transitions))
        ]
        bx = np.asarray([row.observation for _, e in bank for row in e.transitions], np.float64)
        by = np.asarray([row.action for _, e in bank for row in e.transitions], np.float64)

        def retrieve(name, features, targets):
            work["nearest_calls_attempted"] += 1
            work["nearest_query_rows"] += len(queries)
            work["nearest_candidate_upper_bound"] += len(queries) * len(features)
            work["nearest_feature_upper_bound"] += len(queries) * len(features) * 12
            raw = nearest_continuous_actions(queries, features, targets, spec)
            work["nearest_calls_returned"] += 1
            report[name] = _prediction(raw)  # retain paid result before replay validation
            work["nearest_candidates_completed"] += raw.distance_candidates
            work["nearest_features_completed"] += raw.distance_feature_values
            return raw

        replay = retrieve("replay", bx, by)
        native = _native_consistency_targets(replay.actions, cast_work)
        if replay.selected_indices != selected_indices or any(
            struct.pack("!4f", *actual.action) != struct.pack("!4f", *prediction)
            for actual, prediction in zip(trace.transitions, native, strict=True)
        ):
            raise ValueError("offline full-bank replay differs from recorded indices or commands")
        alternative = retrieve(
            "b22_readout", np.asarray(x22, np.float64), np.asarray(y22, np.float64)
        )
        alternative_native = _native_consistency_targets(alternative.actions, cast_work)
        for index, row in enumerate(trace.transitions):
            selected = selected_indices[index]
            group, local = bindings[selected]
            action = alternative_native[index]
            report["trace_rows"].append(
                dict(
                    step=row.step,
                    selected_global_row=selected,
                    selected_group=group,
                    selected_group_row=local,
                    executed_action=list(row.action),
                    b22_row=alternative.selected_indices[index],
                    b22_native_action=list(action),
                    b22_minus_executed=[a - b for a, b in zip(action, row.action, strict=True)],
                )
            )
        report["complete"] = True
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error)[:1024])
    work.update(asdict(pair_work))
    work.update(
        additional_native_conversion_attempts=cast_work.native_conversion_attempts,
        additional_native_conversion_returns=cast_work.native_conversion_returns,
    )
    report["unknown_unreturned_kernel_work"] = (
        work["kernel_calls_attempted"] != work["kernel_calls_returned"]
    )
    report["unknown_unreturned_nearest_work"] = (
        work["nearest_calls_attempted"] != work["nearest_calls_returned"]
    )
    return report


def run_diagnosis(output: Path):
    if not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("Pick19 diagnosis requires a fresh output in an existing directory")
    started = monotonic()
    report: dict[str, Any] = dict(
        complete=False,
        error=None,
        sources=None,
        analysis=None,
        source_bytes_unchanged=False,
        work=dict(
            input_reads=dict(attempts=0, returns=0, bytes=0),
            final_reads=dict(attempts=0, returns=0, bytes=0),
            model_initializations=0,
            model_random_draws=0,
            optimizer_updates=0,
            empirical_fits=0,
            teacher_queries=0,
            simulator_controls=0,
            probe_updates=0,
            final_reads_for_selection=0,
        ),
    )
    analysis_seal = None
    input_seal = None
    try:
        b22, b50, trace, bank, indices, receipt = read_pick19_inputs(
            work=report["work"]["input_reads"]
        )
        report["sources"] = receipt
        report["inputs"] = dict(
            b22=asdict(b22),
            b50=asdict(b50),
            trace=asdict(trace),
            bank=[dict(group=name, episode=asdict(e)) for name, e in bank],
            selected_indices=list(indices),
        )
        input_seal = _json((report["inputs"], report["sources"]))
        report["analysis"] = analyze_pick19(b22, b50, trace, bank, indices)
        analysis_seal = _json(report["analysis"])
        report["elapsed_seconds"] = monotonic() - started
        _require_sources(receipt, report["work"]["final_reads"])
        if (
            _json((report["inputs"], report["sources"])) != input_seal
            or _json(report["analysis"]) != analysis_seal
        ):
            raise ValueError("paid Pick19 inputs or analysis changed before publication")
        report["source_bytes_unchanged"] = True
        if report["analysis"]["complete"] is not True:
            raise ValueError("saved Pick19 analysis is incomplete")
        report["complete"] = True
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error)[:1024])
        if analysis_seal is not None:
            try:
                rejected = _json(report["analysis"])
            except Exception:
                rejected = repr(report["analysis"])[:2048]
            if rejected != analysis_seal:
                report["rejected_analysis"] = rejected
            report["analysis"] = json.loads(analysis_seal)
        if input_seal is not None:
            report["inputs"], report["sources"] = json.loads(input_seal)
    if "elapsed_seconds" not in report:
        report["elapsed_seconds"] = monotonic() - started
    raw = (_json(report) + "\n").encode("utf8")
    if len(raw) > 3 * 1024**2:
        raise ValueError("Pick19 diagnosis exceeds3MiB")
    with output.open("xb") as handle:
        handle.write(raw)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_diagnosis(args.output)
    print(_json({key: report[key] for key in ("complete", "error", "elapsed_seconds")}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
