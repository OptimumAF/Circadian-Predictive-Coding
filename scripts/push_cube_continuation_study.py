"""Fixed local TRAIN study using existing source, continuation and native ports.

No tuning, final access, promotion or generic orchestration. Reserved probes are
explicit, disposable and never update the source. This known-reset
study charges source reconstruction and references; native failure is an outcome.
Run only under the separately frozen PLC or mixed protocol/resource limits.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts.joint_acquisition_inputs import _episode, read_joint_acquisition_inputs
from scripts.maniskill_continuous_smoke import run_frozen_rollout, _same_state as same_nearest_state
from scripts.push_cube_training_readouts import run_training_readout
from scripts.push_cube_reserved_probe_inputs import load_reserved_probe_inputs
from scripts.push_cube_correction_inputs import (
    load_push_correction_inputs,
    load_balanced_push_correction_inputs,
)
from scripts.push_cube_correction_weights import (
    build_balanced_correction_weights,
    build_aggregated_correction_weights,
)
from src.adapters.push_cube_correction_collection import collect_push_cube_corrections
from src.adapters.continuous_probe import copy_probe_plan, run_continuous_probe
from src.adapters.continuous_probe_control import (
    _require_completed as require_completed_probe,
    prepare_zero_velocity_bc,
)
from scripts.diagnose_learned_robot_control import _read_push_study, PUSH_STUDY_SHA256
from src.adapters.continuous_train_continuation import run_continuous_train_continuation
from src.adapters.joint_readiness_corrections import run_joint_readiness_corrections
from src.adapters.joint_readiness_stream import run_joint_readiness_stream
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner, _snapshot
from src.adapters.numpy_nearest_continuous_bc import (
    FixedInputNearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
    _snapshot as validate_push_nearest_snapshot,
)
from src.core.nearest_continuous_bc import NearestPrediction
from src.adapters.continuous_learnability_modes import same_continuous_state
from src.core.continuous_demonstration import ContinuousEpisode
from src.core.push_cube_corrections import PushCubeCorrectionBatch
from src.core.continuous_bc import MIXED_OBJECTIVE, OBJECTIVE, WEIGHTED_MIXED_OBJECTIVE, _row_masses
from src.core.cube_taskspace_state import CubeTaskSpaceState, cube_taskspace_spec
from src.core.pick_cube_action_history import action_history_training_episode
from src.core.pick_cube_action_history_corrections import action_history_correction_view
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.push_cube_teacher import PushCubeDecision
from src.infra.maniskill_push_cube import ManiSkillPushCubeEpisode, PushCubeWork
from src.infra.maniskill_push_teacher import PushCubeTeacherResult

PUSH_SHA256 = "71de05acb941da84ea6778719a25cec954d34ba5fe57156c53da962eb00ad6cf"
ROOT = Path("artifacts/runs")
INPUTS = (
    ROOT / "observed-grasp-teacher-native-20261020/teacher-17.json",
    ROOT / "state-readiness-acquisition-20261020/teacher-19.json",
    ROOT / "gn-arrived-corrections-20261021/correction/collection.json",
    ROOT / "joint-supervised-corrections-20261021/updated-17.json",
    ROOT / "saved-readiness-curation-20261021/case.json",
    ROOT / "train-context-diversity-20261021/teacher-20.json",
)
PUSH = ROOT / "push-cube-supervision-20261021/native-report.json"
REUSED_CONTROL = ROOT / "push-cube-learned-continuation-20261021/run.json"
REUSED_CONTROL_SHA256 = "d6ff9ddbcce6f4e004841aae9e4f9464b8753198847e5060e831bf8ced617935"
REUSED_MIXED = ROOT / "mixed-push-study-20261021/run.json"
REUSED_MIXED_SHA256 = "915b6667a052d9b1c96236adc21e869bc9aeb84620e2f3c95869ea76bbc17f2b"
REUSED_LONGER = ROOT / "mixed-longer-train-fit-20261021/run.json"
REUSED_LONGER_SHA256 = "f9006ccd2e40aa1b087b8a233cb114f273efcaa763d2df8293b461fb97ccc101"
REUSED_RETENTION = ROOT / "mixed-pick-retention-20261021/run.json"
REUSED_RETENTION_SHA256 = "787eee28b99fa4940bfa9558800979309fd4edd05647498979b4bf8312c278d6"
REUSED_NATIVE = ROOT / "mixed-longer-native-endpoint-20261021/run.json"
REUSED_NATIVE_SHA256 = "4a610605d814f5325873eb8e01c4c2b526df5ed1af89675065303c1908269405"
REUSED_CORRECTIVE = ROOT / "mixed-correction-fit-20261021/run.json"
REUSED_CORRECTIVE_SHA256 = "9455a3100f44f25ee9d136b7bb9f92fde5e85ecb74f23ad2d435c74680606021"
REUSED_CORRECTIVE_READOUTS = ROOT / "corrective-learnability-readouts-20261021/run.json"
REUSED_BALANCED = ROOT / "balanced-correction-study-20261021/run.json"
REUSED_CORRECTIVE_READOUTS_SHA256 = (
    "e5ef331e051b6c18bdd623be88aee0b7c94dd4cacdb22c071d0911a5036dfb52"
)


def read_reused_native(path: Path) -> dict[str, Any]:
    """Bind the original failed112 trajectory before collecting correction labels."""
    with path.open("rb") as handle:
        payload = handle.read(4 * 1024**2 + 1)
    if len(payload) > 4 * 1024**2 or hashlib.sha256(payload).hexdigest() != REUSED_NATIVE_SHA256:
        raise ValueError("accepted112 native binding differs")
    control = json.loads(payload)
    if (
        control["complete"] is not True
        or control["source_unchanged"] is not True
        or control["unreturned_stage_work_unknown"] is not False
        or control["native_push"]["complete"] is not True
    ):
        raise ValueError("accepted112 native endpoint is incomplete")
    return control


def read_reused_corrective(path: Path) -> dict[str, Any]:
    """Bind the accepted120 reconstruction before new detached diagnostics."""
    with path.open("rb") as handle:
        payload = handle.read(4 * 1024**2 + 1)
    if (
        len(payload) > 4 * 1024**2
        or hashlib.sha256(payload).hexdigest() != REUSED_CORRECTIVE_SHA256
    ):
        raise ValueError("accepted120 corrective binding differs")
    control = json.loads(payload)
    if (
        control["complete"] is not True
        or control["source_unchanged"] is not True
        or control["unreturned_stage_work_unknown"] is not False
        or control["correction_fit_protocol"]["rows"] != 148
    ):
        raise ValueError("accepted120 corrective endpoint is incomplete")
    _require_longer_continuation(control["correction_continuation"], correction_fit=True)
    return control


def read_reused_corrective_readouts(path: Path) -> dict[str, Any]:
    """Reuse accepted uniform fit diagnostics without rerunning its model."""
    with path.open("rb") as handle:
        payload = handle.read(4 * 1024**2 + 1)
    if (
        len(payload) > 4 * 1024**2
        or hashlib.sha256(payload).hexdigest() != REUSED_CORRECTIVE_READOUTS_SHA256
    ):
        raise ValueError("accepted uniform corrective readouts binding differs")
    control = json.loads(payload)
    if (
        control["complete"] is not True
        or control["source_unchanged"] is not True
        or control["unreturned_stage_work_unknown"] is not False
    ):
        raise ValueError("accepted uniform corrective readouts incomplete")
    _require_longer_continuation(control["correction_continuation"], correction_fit=True)
    _require_corrective_readout(control["corrective_readout112"], 112, 9832)
    _require_corrective_readout(control["corrective_readout120"], 120, 11016)
    return control


def read_reused_longer(path: Path) -> dict[str, Any]:
    """Bind the accepted fixed dose before evaluating its native endpoint."""
    with path.open("rb") as handle:
        payload = handle.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2 or hashlib.sha256(payload).hexdigest() != REUSED_LONGER_SHA256:
        raise ValueError("accepted longer TRAIN fit binding differs")
    control = json.loads(payload)
    if (
        control["complete"] is not True
        or control["source_unchanged"] is not True
        or control["unreturned_stage_work_unknown"] is not False
        or control["longer_readout"]["complete"] is not True
        or control["pick"] != []
    ):
        raise ValueError("accepted longer TRAIN fit is incomplete")
    _require_longer_continuation(control["longer_continuation"])
    return control


def read_reused_retention(path: Path) -> dict[str, Any]:
    """Reuse the accepted mixed48 physical controls without rerunning them."""
    with path.open("rb") as handle:
        payload = handle.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2 or hashlib.sha256(payload).hexdigest() != REUSED_RETENTION_SHA256:
        raise ValueError("accepted mixed retention binding differs")
    control = json.loads(payload)
    if (
        control["protocol"] != "push_cube_mixed_retention_train_v1"
        or control["complete"] is not True
        or control["source_unchanged"] is not True
        or control["unreturned_stage_work_unknown"] is not False
        or len(control["pick"]) != 3
        or tuple(raw["seed"] for raw in control["pick"]) != (17, 19, 20)
        or any(raw["complete"] is not True for raw in control["pick"])
        or control["after_push"]["complete"] is not True
    ):
        raise ValueError("accepted mixed retention is incomplete")
    return control


def _require_reused_longer(control: dict[str, Any], report: dict[str, Any]) -> None:
    """Require the entire accepted112 TRAIN endpoint, not just its counters."""
    keys = ("longer_continuation", "longer_readout")
    expected, actual = (deepcopy({key: value[key] for key in keys}) for value in (control, report))
    # Why: duration is descriptive; all weights' publications/data/work/readouts agree.
    for value in (expected, actual):
        del value["longer_continuation"]["elapsed_seconds"]
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise RuntimeError("reconstructed longer TRAIN endpoint differs from accepted fit")


def read_reused_control(path: Path) -> dict[str, Any]:
    """Reuse one accepted control; refuse replacement before admitting new work."""
    with path.open("rb") as handle:
        payload = handle.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2 or hashlib.sha256(payload).hexdigest() != REUSED_CONTROL_SHA256:
        raise ValueError("accepted Push source control binding differs")
    control = json.loads(payload)
    if (
        control["protocol"] != "push_cube_continuation_train_v1"
        or control["complete"] is not True
        or control["source_unchanged"] is not True
        or control["push_initial_exact"] is not True
        or control["before_push"]["complete"] is not True
    ):
        raise ValueError("accepted Push source control is incomplete")
    return control


def _require_reused_source(control: dict[str, Any], report: dict[str, Any]) -> None:
    """Match the actual producer, ignoring only its descriptive stream duration."""
    keys = (
        "source",
        "acquisition",
        "input_receipts",
        "training",
        "references",
        "b_original",
        "b_projected_actual",
        "reused_push_sha256",
    )
    expected = deepcopy({key: control[key] for key in keys})
    actual = deepcopy({key: report[key] for key in keys})
    # Why: wall-clock duration varies; all model/work/data evidence must agree.
    for value in (expected, actual):
        del value["source"]["stream"]["elapsed_seconds"]
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise RuntimeError("reconstructed source differs from accepted native control")


def _same_push_reset(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Compare every bound reset value across live tuples and saved JSON lists."""
    binding = ("task_id", "seed", "role", "context", "recipe", "controller", "horizon")

    def values(report):
        return dict(
            observation=report["transitions"][0]["observation"],
            metadata={key: report["metadata"][key] for key in binding},
        )

    # Why: serialized controls use lists while live asdict rows retain tuples.
    # JSON normalization changes representation only; no field/value is omitted.
    return json.dumps(values(before), sort_keys=True, allow_nan=False) == json.dumps(
        values(after), sort_keys=True, allow_nan=False
    )


def read_reused_mixed(path: Path) -> dict[str, Any]:
    """Admit the accepted partial negative without upgrading its failed study."""
    with path.open("rb") as handle:
        payload = handle.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2 or hashlib.sha256(payload).hexdigest() != REUSED_MIXED_SHA256:
        raise ValueError("accepted mixed Push endpoint binding differs")
    control = json.loads(payload)
    if (
        control["protocol"] != "push_cube_mixed_continuation_train_v1"
        or control["complete"] is not False
        or control["error"] != "RuntimeError: fixed Push before/after reset binding differs"
        or control["pick"] != []
        or control["unreturned_stage_work_unknown"] is not False
        or control["continuation"]["complete"] is not True
        or control["continuation"]["source_exact"] is not True
        or control["continuation"]["actor_exact"] is not True
        or control["continuation"]["unreturned_training_work_unknown"] is not False
        or control["after_push"]["complete"] is not True
        or control["after_push"]["source_unchanged"] is not True
        or control["after_push"]["unreturned_prediction_possible"] is not False
        or control["after_push"]["unreturned_native_work_unknown"] is not False
        or control["after_push"]["any_success"] is not False
        or control["after_push"]["final_success"] is not False
        or control["after_push"]["sparse_return"] != 0.0
        or not _same_push_reset(control["before_push"], control["after_push"])
    ):
        raise ValueError("accepted mixed Push endpoint is incomplete or differs")
    return control


def _require_reused_continuation(
    control: dict[str, Any], report: dict[str, Any], *, key: str = "continuation"
) -> None:
    """Keep every native/receipt/clock/data value; exempt only descriptive duration."""
    expected, actual = (deepcopy(value[key]) for value in (control, report))
    for value in (expected, actual):
        del value["elapsed_seconds"]
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise RuntimeError("reconstructed mixed continuation differs from accepted Push producer")


def read_push_training(path: Path) -> ContinuousEpisode:
    """Read the exact accepted release; never regenerate labels or native work."""
    with path.open("rb") as handle:
        payload = handle.read(1024**2 + 1)
    if len(payload) > 1024**2 or hashlib.sha256(payload).hexdigest() != PUSH_SHA256:
        raise ValueError("Push training receipt differs from the frozen accepted release")
    raw = json.loads(payload)
    if raw["complete"] is not True or raw["training_release"] is not True:
        raise ValueError("Push training release is incomplete")
    collection = raw["collection"]
    episode = _episode(collection["episode"])
    result = PushCubeTeacherResult(
        episode=episode,
        definition=collection["definition"],
        seed=collection["seed"],
        role=collection["role"],
        states=tuple(
            CubeTaskSpaceState(
                state["is_grasped"],
                tuple(state["tcp_position"]),
                tuple(state["object_position"]),
                tuple(state["goal_position"]),
            )
            for state in collection["states"]
        ),
        decisions=tuple(
            PushCubeDecision(value["stage"], tuple(value["action"]))
            for value in collection["decisions"]
        ),
        transitions=_episode(
            dict(collection["episode"], transitions=collection["transitions"])
        ).transitions,
        attempted_actions=tuple(tuple(row) for row in collection["attempted_actions"]),
        returned_actions=tuple(tuple(row) for row in collection["returned_actions"]),
        decision_attempts=collection["decision_attempts"],
        decision_returns=collection["decision_returns"],
        native_work=PushCubeWork(**collection["native_work"]),
        metadata=collection["metadata"],
        error=collection["error"],
        unreturned_work_possible=collection["unreturned_work_possible"],
    )
    training = result.training_episode()
    if (
        asdict(training) != asdict(_episode(raw["training_episode"]))
        or len(training.transitions) != 18
    ):
        raise ValueError("Push exported TRAIN rows changed")
    return training


def _pin(learner):
    return learner._state, learner._policy, _snapshot(learner._state, learner._policy)


def _require(learner, pin):
    native, policy, snapshot = pin
    if (
        learner._state is not native
        or learner._policy is not policy
        or not same_continuous_state(snapshot, _snapshot(learner._state, learner._policy))
    ):
        raise RuntimeError("frozen study learner changed")


def _pin_push_nearest(learner):
    if (
        type(learner) is not NumpyNearestContinuousBCLearner
        or type(learner._policy) is not FixedInputNearestContinuousBCPolicy
        or type(learner._policy.training_role) is not str
        or learner._policy.training_role != "train"
        or learner._policy.spec != cube_taskspace_spec()
    ):
        raise ValueError("Push nearest rollout requires exact fixed-input TRAIN cubeD12 memory")
    native, policy = learner._state, learner._policy
    snapshot = validate_push_nearest_snapshot(native, policy)
    if not 1 <= snapshot.training_examples <= 256:
        raise ValueError("Push nearest rollout requires1..256 released prototypes")
    return (
        native,
        policy,
        native.policy,
        native.prototype_features,
        native.prototype_actions,
        snapshot,
    )


def _require_push_nearest(learner, pin, *, observe=True):
    def native():
        if (
            type(learner) is not NumpyNearestContinuousBCLearner
            or learner._state is not pin[0]
            or learner._policy is not pin[1]
            or learner._state.policy is not pin[2]
            or learner._state.prototype_features is not pin[3]
            or learner._state.prototype_actions is not pin[4]
            or not same_nearest_state(
                pin[5], validate_push_nearest_snapshot(learner._state, learner._policy)
            )
        ):
            raise RuntimeError("frozen Push nearest owner or prototypes changed")

    native()
    if observe:
        public, policy = learner.snapshot_state(), learner.policy
        if not same_nearest_state(pin[5], public) or policy != pin[5].policy:
            raise RuntimeError("Push nearest public/native state differs")
    native()


def _push_prediction_readback(value):
    if type(value) is np.ndarray:
        return dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            values=[repr(item) for item in value.flat[:16]],
            values_omitted=max(0, value.size - 16),
        )
    return dict(type=type(value).__name__)


def _push_nearest_readback(value):
    def scalar(item, limit=256):
        try:
            text = repr(item)[:limit]
        except Exception as error:
            text = f"repr failed: {type(error).__name__}"
        return dict(type=type(item).__name__, repr=text)

    if type(value) is not NearestPrediction:
        return dict(type=type(value).__name__, returned=scalar(value))
    return dict(
        type="NearestPrediction",
        distance_candidates=scalar(value.distance_candidates),
        distance_feature_values=scalar(value.distance_feature_values),
        selected_indices=scalar(value.selected_indices, 1024),
        actions=_push_prediction_readback(value.actions),
    )


def _roll_push_nearest(learner, *, label: str, seed: int = 17) -> dict[str, Any]:
    """Freeze existing fixed-input retrieval; retain paid receipts before checks."""
    if type(seed) is not int or seed not in (17, 19, 20):
        raise ValueError("Push frozen rollout seed must be an exact TRAIN integer17,19,20")
    pin = _pin_push_nearest(learner)
    prototypes = pin[5].training_examples
    report: dict[str, Any] = dict(
        complete=False,
        error=None,
        seed=seed,
        role="train",
        task_id="PushCube-v1",
        episode=None,
        transitions=[],
        metadata=None,
        work=None,
        prediction_attempts=0,
        prediction_returns=0,
        prediction_outputs=[],
        control_attempts=0,
        control_returns=0,
        source_unchanged=False,
        unreturned_prediction_possible=False,
        unreturned_native_work_unknown=False,
        source_updates=pin[5].completed_updates,
        source_training_rows=prototypes,
        nearest_prediction_attempts=0,
        nearest_prediction_returns=0,
        nearest_distance_candidate_upper_bound=0,
        nearest_distance_feature_upper_bound=0,
        nearest_distance_candidates_completed=0,
        nearest_distance_feature_values_completed=0,
        nearest_selected_indices=[],
        nearest_prediction_reports=[],
        unvalidated_nearest_prediction_work_unknown=False,
    )
    records: list[Any] = []
    published = report["nearest_prediction_reports"]
    validated = 0
    port = None
    started = monotonic()

    def require():
        _require_push_nearest(learner, pin)
        if report["nearest_prediction_reports"] is not published or len(published) != len(records):
            raise RuntimeError("Push nearest paid receipt list changed")
        for index, (raw, action_owner, copied, seal) in enumerate(records):
            if (
                published[index] is not copied
                or (type(raw) is NearestPrediction and raw.actions is not action_owner)
                or json.dumps(_push_nearest_readback(raw), sort_keys=True, allow_nan=False) != seal
                or json.dumps(copied, sort_keys=True, allow_nan=False) != seal
            ):
                raise RuntimeError("preceding Push nearest prediction receipt changed")
        _require_push_nearest(learner, pin, observe=False)

    try:
        require()
        port = ManiSkillPushCubeEpisode(f"push-frozen-{label}", seed=seed, role="train")
        report["metadata"] = port.metadata
        require()
        for _ in range(50):
            require()
            observation = port.observation
            require()
            report["prediction_attempts"] += 1
            report["nearest_prediction_attempts"] += 1
            report["nearest_distance_candidate_upper_bound"] += prototypes
            report["nearest_distance_feature_upper_bound"] += 12 * prototypes
            returned = learner.predict_with_work(np.asarray([observation], dtype=np.float64))
            report["prediction_returns"] += 1
            report["nearest_prediction_returns"] += 1
            # Raw returned work/actions are paid even when malformed. Preserve
            # a bounded JSON readback before any source/work/control admission.
            copied = deepcopy(_push_nearest_readback(returned))
            published.append(copied)
            seal = json.dumps(copied, sort_keys=True, allow_nan=False)
            actions = returned.actions if type(returned) is NearestPrediction else None
            records.append((returned, actions, copied, seal))
            report["prediction_outputs"].append(_push_prediction_readback(actions))
            require()
            if (
                type(returned) is not NearestPrediction
                or type(returned.distance_candidates) is not int
                or returned.distance_candidates != prototypes
                or type(returned.distance_feature_values) is not int
                or returned.distance_feature_values != 12 * prototypes
                or type(returned.selected_indices) is not tuple
                or len(returned.selected_indices) != 1
                or type(returned.selected_indices[0]) is not int
                or not 0 <= returned.selected_indices[0] < prototypes
            ):
                raise ValueError("Push nearest prediction work or selected prototype differs")
            report["nearest_distance_candidates_completed"] += returned.distance_candidates
            report["nearest_distance_feature_values_completed"] += returned.distance_feature_values
            report["nearest_selected_indices"].extend(returned.selected_indices)
            validated += 1
            if (
                type(actions) is not np.ndarray
                or actions.dtype != np.float64
                or actions.shape != (1, 4)
                or not np.isfinite(actions).all()
                or np.any(actions < -1.0)
                or np.any(actions > 1.0)
            ):
                raise ValueError("Push nearest must return one finite normalized float64 action")
            require()
            report["control_attempts"] += 1
            row = port.step(tuple(float(value) for value in actions[0]))
            report["control_returns"] += 1
            report["transitions"].append(asdict(row))
            report["work"] = asdict(port.work)
            require()
            if row.terminated or row.truncated:
                break
        report["episode"] = asdict(port.episode())
        require()
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        work = getattr(error, "push_cube_work", None)
        if work is not None:
            report["work"] = asdict(work)
        for name in ("push_cube_attempted_actions", "push_cube_returned_actions"):
            if hasattr(error, name):
                report[name] = getattr(error, name)
        report["unreturned_native_work_unknown"] = port is None or work is None
    finally:
        if port is not None:
            try:
                port.close()
            except Exception as error:
                report["complete"] = False
                report["close_error"] = f"{type(error).__name__}: {error}"
            try:
                work = port.work
                report["work"] = asdict(work)
                if type(work) is PushCubeWork:
                    report["unreturned_native_work_unknown"] = (
                        work.construction_returns != 1
                        or work.reset_returns != 1
                        or work.step_attempts != work.completed_steps
                        or work.possible_failed_physics_steps != 0
                        or work.state_encoding_attempts != work.state_encoding_returns
                        or work.grasp_query_attempts != work.grasp_query_returns
                        or work.close_attempts != work.close_returns
                    )
            except Exception as error:
                report["complete"] = False
                report["work_error"] = f"{type(error).__name__}: {error}"
                report["unreturned_native_work_unknown"] = True
        report["elapsed_seconds"] = monotonic() - started
        try:
            require()
            report["source_unchanged"] = True
        except Exception as error:
            report["complete"] = False
            report["state_error"] = f"{type(error).__name__}: {error}"
        report["unreturned_prediction_possible"] = (
            report["prediction_attempts"] != report["prediction_returns"]
        )
        report["unvalidated_nearest_prediction_work_unknown"] = (
            report["nearest_prediction_returns"] != validated
        )
        rows = report["transitions"]
        report["any_success"] = any(row["reward"] == 1 for row in rows)
        report["final_success"] = bool(
            rows and (rows[-1]["terminated"] or rows[-1]["truncated"]) and rows[-1]["reward"] == 1
        )
        report["sparse_return"] = sum(row["reward"] for row in rows)
    return report


def roll_push(learner, *, label: str, seed: int = 17) -> dict[str, Any]:
    """One bounded frozen TRAIN Push return; complete negative scores are valid."""
    if type(seed) is not int or seed not in (17, 19, 20):
        raise ValueError("Push frozen rollout seed must be an exact TRAIN integer17,19,20")
    if type(learner) is NumpyNearestContinuousBCLearner:
        if seed == 17:
            return _roll_push_nearest(learner, label=label)
        return _roll_push_nearest(learner, label=label, seed=seed)
    if (
        type(learner) is not NumpyContinuousBCLearner
        or learner._policy.spec != cube_taskspace_spec()
    ):
        raise ValueError("Push rollout requires an exact same-spec native BP")
    pin = _pin(learner)
    report: dict[str, Any] = dict(
        complete=False,
        error=None,
        seed=seed,
        role="train",
        task_id="PushCube-v1",
        episode=None,
        transitions=[],
        metadata=None,
        work=None,
        prediction_attempts=0,
        prediction_returns=0,
        prediction_outputs=[],
        control_attempts=0,
        control_returns=0,
        source_unchanged=False,
        unreturned_prediction_possible=False,
        unreturned_native_work_unknown=False,
    )
    port = None
    started = monotonic()
    try:
        port = ManiSkillPushCubeEpisode(f"push-frozen-{label}", seed=seed, role="train")
        report["metadata"] = port.metadata
        for _ in range(50):
            _require(learner, pin)
            observation = port.observation
            _require(learner, pin)
            report["prediction_attempts"] += 1
            prediction = learner.predict(np.asarray([observation], dtype=np.float64))
            report["prediction_returns"] += 1
            # Retain the returned readout even when validation/control refuses it.
            # String values also preserve NaN/Inf without emitting invalid JSON.
            if type(prediction) is np.ndarray:
                report["prediction_outputs"].append(
                    dict(
                        shape=list(prediction.shape),
                        dtype=str(prediction.dtype),
                        values=[repr(value) for value in prediction.flat[:16]],
                        values_omitted=max(0, prediction.size - 16),
                    )
                )
            else:
                report["prediction_outputs"].append(dict(type=type(prediction).__name__))
            _require(learner, pin)
            if (
                type(prediction) is not np.ndarray
                or prediction.shape != (1, 4)
                or not np.isfinite(prediction).all()
            ):
                raise ValueError("Push policy must return one finite four-axis action")
            report["control_attempts"] += 1
            row = port.step(tuple(float(value) for value in prediction[0]))
            report["control_returns"] += 1
            report["transitions"].append(asdict(row))
            report["work"] = asdict(port.work)
            _require(learner, pin)
            if row.terminated or row.truncated:
                break
        report["episode"] = asdict(port.episode())
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        work = getattr(error, "push_cube_work", None)
        if work is not None:
            report["work"] = asdict(work)
        for name in ("push_cube_attempted_actions", "push_cube_returned_actions"):
            if hasattr(error, name):
                report[name] = getattr(error, name)
        report["unreturned_native_work_unknown"] = port is None or work is None
    finally:
        if port is not None:
            try:
                port.close()
            except Exception as error:
                report["complete"] = False
                report["close_error"] = f"{type(error).__name__}: {error}"
            try:
                work = port.work
                report["work"] = asdict(work)
                if type(work) is PushCubeWork:
                    report["unreturned_native_work_unknown"] = (
                        work.construction_returns != 1 or work.reset_returns != 1
                    )
            except Exception as error:
                report["complete"] = False
                report["work_error"] = f"{type(error).__name__}: {error}"
                report["unreturned_native_work_unknown"] = True
        report["elapsed_seconds"] = monotonic() - started
        try:
            _require(learner, pin)
            report["source_unchanged"] = True
        except Exception as error:
            report["complete"] = False
            report["state_error"] = f"{type(error).__name__}: {error}"
        report["unreturned_prediction_possible"] = (
            report["prediction_attempts"] != report["prediction_returns"]
        )
        rows = report["transitions"]
        report["any_success"] = any(row["reward"] == 1 for row in rows)
        report["final_success"] = bool(
            rows and (rows[-1]["terminated"] or rows[-1]["truncated"]) and rows[-1]["reward"] == 1
        )
        report["sparse_return"] = sum(row["reward"] for row in rows)
    return report


def _write(output, report, *, exclusive=False):
    text = json.dumps(report, sort_keys=True, allow_nan=False)
    maximum = 6 if report.get("protocol") == "push_cube_aggregated_correction_train_v1" else 4
    if len(text.encode("utf8")) > maximum * 1024**2:
        raise ValueError(f"study report exceeds{maximum}MiB")
    with output.open("x" if exclusive else "w", encoding="utf8") as handle:
        handle.write(text + "\n")


def _require_longer_continuation(
    report, *, correction_fit=False, row_weights=None, aggregated=False
):
    """Validate this fixed paid scope before its read-only endpoint work."""
    if type(correction_fit) is not bool or type(aggregated) is not bool:
        raise ValueError("continuation proof selection must be boolean")
    weighted = row_weights is not None
    if weighted and not correction_fit:
        raise ValueError("weighted continuation proof requires the corrective scope")
    if aggregated and (not correction_fit or not weighted):
        raise ValueError("aggregated continuation proof requires weighted corrective scope")
    rows, updates = (198, 8) if aggregated else (148, 8) if correction_fit else (98, 64)
    initial = (
        dict(updates=112, training_rows=9832)
        if correction_fit
        else dict(updates=48, training_rows=3560)
    )
    final = (
        dict(updates=120, training_rows=11016)
        if correction_fit
        else dict(updates=112, training_rows=9832)
    )
    if aggregated:
        initial = dict(updates=120, training_rows=11016)
        final = dict(updates=128, training_rows=12600)
        if report.get("row_scope") != dict(
            maximum_rows=198,
            default_maximum_rows=148,
            scope="explicit expanded released TRAIN inputs; no authority or update-budget renewal",
        ):
            raise RuntimeError("aggregated continuation row scope differs")
    counters = (
        report["rows"],
        report["budget"]["max_training_updates"],
        *report["initial_counts"].values(),
        *report["final_counts"].values(),
        *report["progress"].values(),
    )
    if (
        report["complete"] is not True
        or report.get("unreturned_training_work_unknown") is not False
        or any(type(value) is not int for value in counters)
        or report["source_exact"] is not True
        or report["actor_exact"] is not True
        or report["rows"] != rows
        or report["protocol"]
        != (
            "continuous_train_row_weighted_mixed_gripper_continuation_v1"
            if weighted
            else "continuous_train_mixed_gripper_continuation_v1"
        )
        or report["budget"]["max_training_updates"] != updates
        or report["initial_counts"] != initial
        or report["final_counts"] != final
        or report["progress"]
        != dict(
            updates_completed=updates, replay_examples_completed=0, logical_tick=999 + 2 * updates
        )
        or len(report["native_updates"]) != updates
        or len(report["applied_updates"]) != updates
        or len(report["events"]) != updates
        or report["native_updates"] != report["sealed_native_updates"]
        or report["native_updates"] != report["validated_native_updates"]
        or report["applied_updates"] != report["sealed_applied_updates"]
    ):
        raise RuntimeError("fixed longer TRAIN continuation incomplete or unvalidated")
    if weighted:
        masses = _row_masses(np.asarray(row_weights, dtype=np.float64))
        expected_masses = [1.0 / rows] * rows if masses is None else masses.tolist()
        if (
            len(row_weights) != rows
            or report["row_weighting"]["definition"] != WEIGHTED_MIXED_OBJECTIVE
            or report["row_weighting"]["values"] != list(row_weights)
            or report["row_weighting"]["normalized_masses"] != expected_masses
            or report["policy_transition"]["candidate_objective"] != WEIGHTED_MIXED_OBJECTIVE
        ):
            raise RuntimeError("weighted corrective continuation supervision differs")
    for index, (native, receipt, event) in enumerate(
        zip(report["native_updates"], report["applied_updates"], report["events"], strict=True)
    ):
        if (
            event["deferred_exact"] is not True
            or event["success_validated"] is not True
            or any(
                type(receipt[key]) is not int
                for key in ("update_number", "observed_at", "arrived_at", "applied_at")
            )
            or type(receipt["update_number"]) is not int
            or receipt["update_number"] != index + 1
            or (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"])
            != (1000 + 2 * index, 1001 + 2 * index, 1001 + 2 * index)
            or receipt["episode_id"] != "continuous-TRAIN-continuation"
            or receipt["sample_id"] != f"continuation-{index}"
            or receipt["event_id"] != f"continuation-label-{index}"
            or receipt["actor_version"] != "actor-0"
            or receipt["learner_version"] != "candidate-0"
            or native["training_rows"] != rows
            or type(native["training_rows"]) is not int
            or native["diagnostic"] != receipt["diagnostic"]
            or receipt["diagnostic"]["definition"]
            != (WEIGHTED_MIXED_OBJECTIVE if weighted else MIXED_OBJECTIVE)
            or (
                weighted
                and (
                    native["objective_row_weights"] != list(row_weights)
                    or native["work"]["objective_weight_values"] != rows
                    or native["work"]["objective_weight_numeric_bytes"] != 8 * rows
                    or native["work"]["native_row_weight_mass_admissions"] != 2
                )
            )
        ):
            raise RuntimeError("fixed longer TRAIN receipt or original clock differs")


def _require_corrective_readout(report, updates, training_rows, *, label=None, aggregated=False):
    if type(aggregated) is not bool:
        raise ValueError("readout proof selection must be boolean")
    names: tuple[str, ...] = ("corrections", "push", "a", "b", "train20")
    sizes: tuple[int, ...] = (50, 18, 15, 50, 15)
    if aggregated:
        names = ("current_corrections", *names)
        sizes = (50, *sizes)
    if (
        report["complete"] is not True
        or report["label"] != (f"corrective{updates}" if label is None else label)
        or type(report["source_updates"]) is not int
        or report["source_updates"] != updates
        or type(report["source_training_examples"]) is not int
        or report["source_training_examples"] != training_rows
        or report["group_order"] != list(names)
        # Canonical JSON sorts mapping keys; group_order carries semantic order.
        or set(report["groups"]) != set(names)
        or tuple(report["groups"][name]["rows"] for name in names) != sizes
        or report["native_objective"] != MIXED_OBJECTIVE
        or report["physical_objective"] != "normalized_active_axis_half_mse"
        or report["work"]["unknown_unreturned_work"] is not False
        or report["phase_readouts"]["forks"] != 1
        or report["phase_readouts"]["validated"] != 1
        or len(report["operations"]) != 2 * len(names)
    ):
        raise RuntimeError("corrective TRAIN readout incomplete or differs from declared endpoint")
    for entry, (name, kind, rows) in zip(
        report["operations"],
        (
            (name, kind, rows)
            for name, rows in zip(names, sizes, strict=True)
            for kind in ("conditioning", "residual")
        ),
        strict=True,
    ):
        if any(entry[key] is not True for key in ("attempted", "returned", "validated")) or (
            entry["group"],
            entry["kind"],
            entry["rows"],
        ) != (name, kind, rows):
            raise RuntimeError("corrective TRAIN diagnostic operation is not validated")


def _corrective_deltas(before, after):
    result = {}
    for name in before["group_order"]:
        old, new = (value["groups"][name] for value in (before, after))
        previous, current = (value["residual"]["residuals"]["overall"] for value in (old, new))
        axes = []
        for a, b in zip(previous["axes"], current["axes"], strict=True):
            if (a["index"], a["unit"]) != (b["index"], b["unit"]):
                raise RuntimeError("corrective physical residual axes differ")
            axes.append(
                dict(
                    index=b["index"],
                    unit=b["unit"],
                    **{
                        key: b[key] - a[key]
                        for key in ("normalized_bias", "normalized_mae", "normalized_half_mse")
                    },
                )
            )
        result[name] = dict(
            native_objective=new["conditioning"]["conditioning"]["objective_loss"]
            - old["conditioning"]["conditioning"]["objective_loss"],
            physical_half_mse=current["normalized_half_mse"] - previous["normalized_half_mse"],
            axes=axes,
        )
    return result


def _release_push_corrections(collected, old_push, *, checkpoint):
    """Release only complete observed labels on the exact accepted student path."""
    if (
        collected.report["complete"] is not True
        or collected.corrections is None
        or collected.report["source_unchanged"] is not True
        or collected.report["training_release"] is not True
        or any(
            collected.report[key] is not False
            for key in (
                "unreturned_query_possible",
                "unreturned_prediction_possible",
                "unreturned_native_work_unknown",
            )
        )
    ):
        raise RuntimeError("Push correction collection is incomplete")
    batch = collected.corrections
    if (
        type(batch) is not PushCubeCorrectionBatch
        or asdict(batch.episode) != collected.report["episode"]
        or [asdict(label) for label in batch.labels] != collected.report["labels"]
    ):
        raise RuntimeError("Push correction returned packet differs from paid labels")
    if not _same_push_reset(old_push, collected.report) or json.dumps(
        old_push["transitions"], sort_keys=True, allow_nan=False
    ) != json.dumps(collected.report["transitions"], sort_keys=True, allow_nan=False):
        raise RuntimeError(
            f"Push correction student trajectory differs from accepted{checkpoint} endpoint"
        )
    return batch.training_episode(now=1100)


def run_study(
    output: Path,
    *,
    profile: str = "head_gauss_newton",
    training_diagnostics: bool = False,
    diagnostics_only: bool = False,
    reserved_probes: bool = False,
    longer_fit: bool = False,
    native_endpoint: bool = False,
    push_corrections: bool = False,
    correction_fit: bool = False,
    correction_readouts: bool = False,
    balanced_correction_fit: bool = False,
    zero_velocity_fit: bool = False,
    balanced_push_corrections: bool = False,
    aggregated_correction_fit: bool = False,
) -> dict[str, Any]:
    """One fixed source→new-task continuation, using the existing private runtime."""
    if type(profile) is not str or profile not in (
        "head_gauss_newton",
        "bp_mixed",
        "bp_mixed_retention",
    ):
        raise ValueError("study profile must be head_gauss_newton, bp_mixed or bp_mixed_retention")
    if any(
        type(value) is not bool
        for value in (
            training_diagnostics,
            diagnostics_only,
            reserved_probes,
            longer_fit,
            native_endpoint,
            push_corrections,
            correction_fit,
            correction_readouts,
            balanced_correction_fit,
            zero_velocity_fit,
            balanced_push_corrections,
            aggregated_correction_fit,
        )
    ):
        raise ValueError("study diagnostic selections must be exact booleans")
    if aggregated_correction_fit and (
        profile != "bp_mixed_retention"
        or not longer_fit
        or not native_endpoint
        or not correction_fit
        or training_diagnostics
        or diagnostics_only
        or reserved_probes
        or push_corrections
        or correction_readouts
        or balanced_correction_fit
        or zero_velocity_fit
        or balanced_push_corrections
    ):
        raise ValueError(
            "aggregated_correction_fit requires only retention longer-fit corrective native endpoint"
        )
    if balanced_push_corrections and (
        profile != "bp_mixed_retention"
        or not longer_fit
        or not correction_fit
        or native_endpoint
        or diagnostics_only
        or push_corrections
        or correction_readouts
        or balanced_correction_fit
        or zero_velocity_fit
        or training_diagnostics
        or reserved_probes
    ):
        raise ValueError(
            "balanced_push_corrections requires only retention longer-fit correction-fit"
        )
    if zero_velocity_fit and (
        profile != "bp_mixed_retention"
        or not longer_fit
        or not diagnostics_only
        or not correction_fit
        or not correction_readouts
        or native_endpoint
        or balanced_correction_fit
        or push_corrections
        or training_diagnostics
        or reserved_probes
    ):
        raise ValueError(
            "zero_velocity_fit requires only retention longer-fit corrective readouts diagnostics-only"
        )
    if balanced_correction_fit and (
        profile != "bp_mixed_retention"
        or not longer_fit
        or not native_endpoint
        or not correction_fit
        or diagnostics_only
        or correction_readouts
        or push_corrections
        or training_diagnostics
        or reserved_probes
    ):
        raise ValueError(
            "balanced_correction_fit requires only retention longer-fit corrective native endpoint"
        )
    if diagnostics_only and (
        not (training_diagnostics or reserved_probes or longer_fit)
        or profile != "bp_mixed_retention"
    ):
        raise ValueError("diagnostics_only requires readouts or probes and bp_mixed_retention")
    if reserved_probes and (not diagnostics_only or profile != "bp_mixed_retention"):
        raise ValueError("reserved_probes requires diagnostics_only and bp_mixed_retention")
    if longer_fit and (
        profile != "bp_mixed_retention"
        or diagnostics_only == (native_endpoint or push_corrections or balanced_push_corrections)
        or training_diagnostics
        or reserved_probes
    ):
        raise ValueError(
            "longer_fit requires retention diagnostics_only or native_endpoint without other readouts/probes"
        )
    if native_endpoint and (not longer_fit or profile != "bp_mixed_retention" or diagnostics_only):
        raise ValueError(
            "native_endpoint requires longer_fit and retention without diagnostics_only"
        )
    if push_corrections and (
        not longer_fit or profile != "bp_mixed_retention" or diagnostics_only or native_endpoint
    ):
        raise ValueError("push_corrections requires longer-fit retention without other endpoints")
    if correction_fit and (
        not (native_endpoint or correction_readouts or balanced_push_corrections)
        or not longer_fit
        or profile != "bp_mixed_retention"
        or push_corrections
    ):
        raise ValueError(
            "correction_fit requires retention longer-fit endpoint or correction-readouts"
        )
    if correction_readouts and (
        not correction_fit
        or not diagnostics_only
        or native_endpoint
        or push_corrections
        or training_diagnostics
        or reserved_probes
    ):
        raise ValueError(
            "correction_readouts requires corrective diagnostics-only without other readouts"
        )
    paths = tuple(output.with_name(output.name + f".pick-{seed}.json") for seed in (17, 19, 20))
    if not output.parent.is_dir() or any(p.exists() or p.is_symlink() for p in (output, *paths)):
        raise ValueError("study requires four fresh output paths")
    control = read_reused_control(REUSED_CONTROL) if profile != "head_gauss_newton" else None
    mixed = read_reused_mixed(REUSED_MIXED) if profile == "bp_mixed_retention" else None
    longer = (
        read_reused_longer(REUSED_LONGER)
        if native_endpoint or push_corrections or correction_readouts or balanced_push_corrections
        else None
    )
    retention = (
        read_reused_retention(REUSED_RETENTION)
        if native_endpoint or push_corrections or correction_readouts or balanced_push_corrections
        else None
    )
    prior_native = read_reused_native(REUSED_NATIVE) if push_corrections or correction_fit else None
    corrective_control = (
        read_reused_corrective(REUSED_CORRECTIVE)
        if correction_readouts
        or balanced_correction_fit
        or balanced_push_corrections
        or aggregated_correction_fit
        else None
    )
    corrective_readouts = (
        read_reused_corrective_readouts(REUSED_CORRECTIVE_READOUTS)
        if balanced_correction_fit or balanced_push_corrections or aggregated_correction_fit
        else None
    )
    inherited_balanced = (
        _read_push_study(REUSED_BALANCED)
        if zero_velocity_fit or balanced_push_corrections or aggregated_correction_fit
        else None
    )
    correction_packet, correction_receipt = (
        load_push_correction_inputs() if correction_fit else (None, None)
    )
    current_packet, current_receipt = (
        load_balanced_push_correction_inputs() if aggregated_correction_fit else (None, None)
    )
    control_seal = json.dumps(control, sort_keys=True, allow_nan=False)
    mixed_seal = json.dumps(mixed, sort_keys=True, allow_nan=False)
    endpoint_seal = json.dumps(
        (
            longer,
            retention,
            prior_native,
            corrective_control,
            corrective_readouts,
            inherited_balanced,
        ),
        sort_keys=True,
        allow_nan=False,
    )
    training = read_push_training(PUSH)
    inputs, receipts = read_joint_acquisition_inputs(*INPUTS)
    warm, new, corrections, curated, retained = inputs
    recipe = PickCubeTaskSpaceHistoryRecipe(mode="executed")
    corrections.training_pairs(now=1100)  # Original release gate before projection.
    view = action_history_correction_view(corrections, recipe)
    _, targets = view.model.training_pairs(now=1100)
    # This is a target packet, not native execution: encoded history stays the
    # original student's history, while action supplies the released teacher label.
    b = replace(
        view.model.episode,
        episode_id="derived-B19-supervision",
        transitions=tuple(
            replace(row, action=target)
            for row, target in zip(view.model.episode.transitions, targets, strict=True)
        ),
    )
    references = (
        action_history_training_episode(warm.episode, recipe),
        b,
        action_history_training_episode(retained.episode, recipe),
    )
    if tuple(len(e.transitions) for e in references) != (15, 50, 15) or any(
        e.spec != training.spec or e.role != "train" for e in references
    ):
        raise ValueError("fixed TRAIN references differ")
    balanced_weights, balanced_recipe = None, None
    if (
        balanced_correction_fit
        or zero_velocity_fit
        or balanced_push_corrections
        or aggregated_correction_fit
    ):
        assert correction_packet is not None
        corrective_groups = tuple(
            zip(
                ("corrections", "push", "a", "b", "train20"),
                (correction_packet, training, *references),
                strict=True,
            )
        )
        balanced_weights, balanced_recipe = build_balanced_correction_weights(corrective_groups)
    aggregated_groups: tuple[tuple[str, ContinuousEpisode], ...] = ()
    aggregated_weights, aggregated_recipe = None, None
    if aggregated_correction_fit:
        assert current_packet is not None and correction_packet is not None
        aggregated_groups = tuple(
            zip(
                ("current_corrections", "corrections", "push", "a", "b", "train20"),
                (current_packet, correction_packet, training, *references),
                strict=True,
            )
        )
        aggregated_weights, aggregated_recipe = build_aggregated_correction_weights(
            aggregated_groups
        )
    probe_inputs: tuple[Any, ...] = ()
    probe_readouts: tuple[ContinuousEpisode, ...] = ()
    plan: Any = None
    probe_receipts = None
    if reserved_probes:
        plan, probe_receipts = load_reserved_probe_inputs()
        probe_readouts = tuple(
            action_history_training_episode(item.episode, recipe) for item in (warm, new, retained)
        )
        if tuple(len(item.transitions) for item in probe_readouts) != (15, 22, 15):
            raise ValueError("original probe TRAIN readouts differ")
        plan = copy_probe_plan(plan, (*probe_readouts, training))
        probe_inputs = (plan, *probe_readouts)
    report: dict[str, Any] = dict(
        protocol="push_cube_continuation_train_v1",
        complete=False,
        input_receipts=receipts,
        reused_push_receipt=str(PUSH),
        reused_push_sha256=PUSH_SHA256,
        source=None,
        acquisition=None,
        continuation=None,
        before_push=None,
        after_push=None,
        pick=[],
        attempted=[],
        references=[asdict(e) for e in references],
        b_original=asdict(corrections),
        b_projected_actual=asdict(view.model),
        reference_projection_work=dict(
            rows=80,
            source_views=160,
            source_values=6720,
            projected_values=1920,
            history_advances=80,
            b_label_observation_values=600,
            b_label_target_values=200,
        ),
        training=asdict(training),
        error=None,
    )
    if control is not None:
        report["protocol"] = "push_cube_mixed_continuation_train_v1"
        report["reused_control"] = dict(
            path=str(REUSED_CONTROL),
            sha256=REUSED_CONTROL_SHA256,
            scope="accepted nonconcurrent common-MSE source40/2776",
        )
    if mixed is not None:
        report["protocol"] = "push_cube_mixed_retention_train_v1"
        report["reused_after_push"] = dict(
            path=str(REUSED_MIXED),
            sha256=REUSED_MIXED_SHA256,
            scope="accepted complete negative endpoint from failed mixed study",
        )
    if training_diagnostics:
        report["training_diagnostics"] = dict(
            scope="released TRAIN fit/conditioning, no diagnostic updates or native score selection",
            diagnostics_only=diagnostics_only,
            groups=["push", "a", "b", "train20"],
        )
    if reserved_probes:
        report["reserved_probes"] = dict(
            plan=asdict(plan),
            receipts=probe_receipts,
            training_readouts=[asdict(item) for item in probe_readouts],
            scope="two independent disposable probes; source/actor unchanged; no native scenes",
        )
    if longer_fit:
        report["longer_fit_protocol"] = dict(
            additional_updates=64,
            expected_initial_counts=[48, 3560],
            expected_final_counts=[112, 9832],
            historical_after_push_checkpoint=48,
            scope="fixed longer TRAIN dose; no probe/final or native endpoint selection",
        )
    if native_endpoint:
        report["native_endpoint_protocol"] = dict(
            checkpoint=120 if correction_fit else 112,
            training_rows=11016 if correction_fit else 9832,
            baseline_checkpoint=112 if correction_fit else 48,
            reused_fit=dict(path=str(REUSED_LONGER), sha256=REUSED_LONGER_SHA256),
            reused_retention=dict(path=str(REUSED_RETENTION), sha256=REUSED_RETENTION_SHA256),
            scenes=["push-17", "pick-17", "pick-19", "pick-20"],
            scope="fixed TRAIN physical endpoint; no additional training or scene-dependent selection",
        )
    if push_corrections:
        report["push_correction_protocol"] = dict(
            checkpoint=112,
            source_training_rows=9832,
            seed=17,
            role="train",
            prior_native=dict(path=str(REUSED_NATIVE), sha256=REUSED_NATIVE_SHA256),
            scope="provisional labels on frozen student states; no correction training or Pick scenes",
        )
    if correction_fit:
        if native_endpoint:
            report["native_endpoint_protocol"]["reused_native"] = dict(
                path=str(REUSED_NATIVE), sha256=REUSED_NATIVE_SHA256
            )
        report["correction_fit_protocol"] = dict(
            updates=8,
            rows=148,
            initial_counts=[112, 9832],
            final_counts=[120, 11016],
            input_receipt=correction_receipt,
            scope="all50corrections+18expert+80oldTRAIN; no row selection",
        )
    if correction_readouts:
        report["corrective_readouts_protocol"] = dict(
            checkpoints=[112, 120],
            groups=["corrections", "push", "a", "b", "train20"],
            rows=[50, 18, 15, 50, 15],
            reused_corrective=dict(path=str(REUSED_CORRECTIVE), sha256=REUSED_CORRECTIVE_SHA256),
            scope="detached TRAIN fit only; reconstruction charged; no new dose or native scenes",
        )
    if balanced_correction_fit or balanced_push_corrections or aggregated_correction_fit:
        assert balanced_recipe is not None and corrective_readouts is not None
        if balanced_correction_fit:
            report["protocol"] = "push_cube_prefix_group_balanced_correction_train_v1"
            report["native_endpoint_protocol"].update(
                baseline_checkpoint=120,
                supplemental_baseline_checkpoint=112,
                reused_uniform120=dict(
                    path=str(REUSED_CORRECTIVE), sha256=REUSED_CORRECTIVE_SHA256
                ),
            )
        report["balanced_correction_protocol"] = dict(
            **balanced_recipe,
            reused_uniform120=dict(path=str(REUSED_CORRECTIVE), sha256=REUSED_CORRECTIVE_SHA256),
            reused_readouts=dict(
                path=str(REUSED_CORRECTIVE_READOUTS), sha256=REUSED_CORRECTIVE_READOUTS_SHA256
            ),
        )
        report["reused_uniform120_readout"] = deepcopy(corrective_readouts["corrective_readout120"])
    if zero_velocity_fit:
        assert balanced_recipe is not None and inherited_balanced is not None
        if (
            inherited_balanced["complete"] is not True
            or inherited_balanced["source_unchanged"] is not True
            or inherited_balanced["unreturned_stage_work_unknown"] is not False
            or inherited_balanced["protocol"]
            != "push_cube_prefix_group_balanced_correction_train_v1"
        ):
            raise ValueError("inherited velocity control is incomplete")
        original_recipe = inherited_balanced["balanced_correction_protocol"]
        if balanced_recipe != {name: original_recipe[name] for name in balanced_recipe}:
            raise ValueError("private velocity control fixed recipe differs")
        report["protocol"] = "push_cube_private_zero_velocity_fit_train_v1"
        report["zero_velocity_protocol"] = dict(
            **balanced_recipe,
            reused_inherited=dict(path=str(REUSED_BALANCED), sha256=PUSH_STUDY_SHA256),
            scope="private TRAIN velocity-only control; all148rows/eightweightedupdates; no native scenes",
        )
        report["reused_inherited_readout"] = deepcopy(inherited_balanced["balanced_readout120"])
        report["corrective_readouts_protocol"]["scope"] = (
            "private zero-velocity TRAIN fit; paid reconstruction and readouts; no physical success inference"
        )
    if balanced_push_corrections or aggregated_correction_fit:
        assert inherited_balanced is not None
        if (
            inherited_balanced["complete"] is not True
            or inherited_balanced["source_unchanged"] is not True
            or inherited_balanced["unreturned_stage_work_unknown"] is not False
            or inherited_balanced["protocol"]
            != "push_cube_prefix_group_balanced_correction_train_v1"
            or report["balanced_correction_protocol"]
            != inherited_balanced["balanced_correction_protocol"]
        ):
            raise ValueError("accepted balanced student or fixed recipe differs")
        report["protocol"] = "push_cube_balanced_student_corrections_train_v1"
        report["push_correction_protocol"] = dict(
            checkpoint=120,
            source_training_rows=11016,
            seed=17,
            role="train",
            prior_native=dict(path=str(REUSED_BALANCED), sha256=PUSH_STUDY_SHA256),
            scope="provisional labels on frozen balanced120 states; original fit reconstructed only; no new-label fit or Pick scenes",
        )
    if aggregated_correction_fit:
        assert aggregated_recipe is not None and current_packet is not None
        report["protocol"] = "push_cube_aggregated_correction_train_v1"
        report["aggregated_correction_protocol"] = dict(
            **aggregated_recipe,
            input_receipt=current_receipt,
            updates=8,
            initial_counts=[120, 11016],
            final_counts=[128, 12600],
            reused_balanced=dict(path=str(REUSED_BALANCED), sha256=PUSH_STUDY_SHA256),
        )
        report["current_correction_packet"] = asdict(current_packet)
        del report["push_correction_protocol"]
        report["native_endpoint_protocol"].update(
            checkpoint=128,
            training_rows=12600,
            baseline_checkpoint=120,
            reused_balanced=dict(path=str(REUSED_BALANCED), sha256=PUSH_STUDY_SHA256),
        )
    pins: list[Any] = []
    paid: list[Any] = []
    observed: list[str] = []
    seal = json.dumps(report, sort_keys=True, allow_nan=False)
    input_seal = json.dumps(
        [
            asdict(value)
            for value in (
                *inputs,
                training,
                *references,
                *probe_inputs,
                *((correction_packet,) if correction_fit else ()),
                *((current_packet,) if aggregated_correction_fit else ()),
            )
        ],
        sort_keys=True,
        allow_nan=False,
    )

    def guard():
        if json.dumps(control, sort_keys=True, allow_nan=False) != control_seal:
            raise RuntimeError("accepted reused control changed")
        if json.dumps(mixed, sort_keys=True, allow_nan=False) != mixed_seal:
            raise RuntimeError("accepted reused mixed endpoint changed")
        if (
            json.dumps(
                (
                    longer,
                    retention,
                    prior_native,
                    corrective_control,
                    corrective_readouts,
                    inherited_balanced,
                ),
                sort_keys=True,
                allow_nan=False,
            )
            != endpoint_seal
        ):
            raise RuntimeError("accepted reused longer/retention endpoint changed")
        if json.dumps(report, sort_keys=True, allow_nan=False) != seal:
            raise RuntimeError("study preceding saved evidence changed")
        if (
            json.dumps(
                [
                    asdict(value)
                    for value in (
                        *inputs,
                        training,
                        *references,
                        *probe_inputs,
                        *((correction_packet,) if correction_fit else ()),
                        *((current_packet,) if aggregated_correction_fit else ()),
                    )
                ],
                sort_keys=True,
                allow_nan=False,
            )
            != input_seal
        ):
            raise RuntimeError("study original TRAIN inputs changed")
        for owner, pin in pins:
            _require(owner, pin)
        for holder, raw, candidate, payload, saved, location in paid:
            current = report["pick"][location] if type(location) is int else report[location]
            if current is not saved:
                raise RuntimeError("study preceding saved report owner changed")
            if holder is not None and (
                holder.report is not raw or getattr(holder, "candidate", None) is not candidate
            ):
                raise RuntimeError("study preceding returned owner changed")
            if json.dumps(raw, sort_keys=True, allow_nan=False) != payload:
                raise RuntimeError("study preceding returned evidence changed")

    def save(*, exclusive=False):
        nonlocal seal
        seal = json.dumps(report, sort_keys=True, allow_nan=False)
        _write(output, report, exclusive=exclusive)
        guard()

    def call(key, function, *, pick=False):
        guard()
        report["attempted"].append(key)
        save()
        returned = function()
        observed.append(key)
        holder = None if type(returned) is dict else returned
        raw = returned if holder is None else holder.report
        if type(raw) is not dict:
            raise RuntimeError("study requires an original report dictionary")
        try:
            guard()  # Validate the preceding success before admitting this return.
        except Exception:
            # The return is observed and paid even when prior evidence was corrupted.
            report["refused_return"] = dict(stage=key, report=deepcopy(raw))
            raise
        saved = deepcopy(raw)
        location = len(report["pick"]) if pick else key
        paid.append(
            (
                holder,
                raw,
                getattr(holder, "candidate", None),
                json.dumps(raw, sort_keys=True, allow_nan=False),
                saved,
                location,
            )
        )
        if pick:
            report["pick"].append(saved)
        else:
            report[key] = saved
        save()
        return returned

    def probe(key, owner, pin):
        raw = call(
            key,
            lambda: run_continuous_probe(
                owner, plan, training_readouts=probe_readouts, check_boundary=guard
            ),
        )
        # Reuse the single-scope proof already required between paired probes:
        # local budgets never waive original receipts, clocks or global counts.
        require_completed_probe(raw, pin[2], plan)
        expected_plan = dict(
            updates=1,
            training_id=plan.training.episode_id,
            evaluation_id=plan.evaluation.episode_id,
            context=plan.training.context,
            diagnostics=True,
        )
        if json.dumps(raw["plan"], sort_keys=True, allow_nan=False) != json.dumps(
            expected_plan, sort_keys=True, allow_nan=False
        ):
            raise RuntimeError("reserved probe returned plan differs from the fixed split")
        objective = OBJECTIVE if owner.policy.binary_gripper_axis is None else MIXED_OBJECTIVE
        if raw["native_updates"][0]["diagnostic"]["definition"] != objective:
            raise RuntimeError("reserved probe objective differs from original checkpoint")
        guard()

    failed = False

    def fail_capture(error):
        nonlocal report, seal, failed
        failed = True
        rejected = []
        missing = object()

        def observation(value):
            try:
                return json.loads(json.dumps(value, allow_nan=False))
            except (TypeError, ValueError, RecursionError):
                return dict(type=type(value).__name__, representation=repr(value)[:4096])

        for holder, raw, candidate, payload, saved, location in paid:
            picks = report.get("pick")
            if type(location) is int:
                current = (
                    picks[location] if type(picks) is list and location < len(picks) else missing
                )
            else:
                current = report.get(location, missing)
            if current is missing:
                rejected.append(dict(location=location, kind="saved", missing=True))
            elif current is not saved or observation(current) != json.loads(payload):
                rejected.append(
                    dict(
                        location=location,
                        kind="saved",
                        replaced=current is not saved,
                        observed=observation(current),
                    )
                )
            if observation(raw) != json.loads(payload):
                rejected.append(dict(location=location, kind="returned", observed=observation(raw)))
            if holder is not None and holder.report is not raw:
                rejected.append(
                    dict(
                        location=location,
                        kind="returned_replacement",
                        observed=observation(holder.report),
                    )
                )
        refused = report.get("refused_return")
        # Restore the bounded last intended capture, preserving corrupt evidence
        # separately. A failed proof never overwrites the original paid payload.
        report = json.loads(seal)
        report["complete"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["evidence_error"] = report["error"]
        if rejected:
            report["rejected_evidence"] = rejected
        if refused is not None:
            report["refused_return"] = refused
        seal = json.dumps(report, sort_keys=True, allow_nan=False)

    save(exclusive=True)
    started = monotonic()
    try:
        guard()
        built = call(
            "source",
            lambda: run_joint_readiness_stream(
                warm, new, corrections, curated, corrections_now=99, curated_now=199
            ),
        )
        if (
            built.report["complete"] is not True
            or type(built.candidate) is not NumpyContinuousBCLearner
        ):
            raise RuntimeError("source construction incomplete")
        pins.append((built.candidate, _pin(built.candidate)))
        acquired = call(
            "acquisition",
            lambda: run_joint_readiness_corrections(
                built.candidate,
                warm,
                corrections,
                curated,
                now=1100,
                action_history_recipe=recipe,
                retained_reference=retained,
            ),
        )
        source = acquired.candidate
        if acquired.report["complete"] is not True or type(source) is not NumpyContinuousBCLearner:
            raise RuntimeError("joint acquisition incomplete")
        source_pin = _pin(source)
        pins.append((source, source_pin))
        if (source_pin[2].completed_updates, source_pin[2].training_examples) != (40, 2776):
            raise RuntimeError("fixed acquired source counters differ")
        if control is not None:
            _require_reused_source(control, report)
        if mixed is not None:
            _require_reused_source(mixed, report)
        if reserved_probes:
            probe("probe_before", source, source_pin)
        diagnostic_groups = tuple(
            zip(("push", "a", "b", "train20"), (training, *references), strict=True)
        )
        if training_diagnostics:
            diagnosed = call(
                "fit_before",
                lambda: run_training_readout(
                    source, diagnostic_groups, label="source40", guard=guard
                ),
            )
            if diagnosed["complete"] is not True:
                raise RuntimeError("source TRAIN fit diagnosis incomplete")
        before = call(
            "before_push",
            lambda: (
                deepcopy(control["before_push"])
                if control is not None
                else roll_push(source, label="before")
            ),
        )
        if before["complete"] is not True:
            raise RuntimeError("pre-update Push return incomplete")
        continued = call(
            "continuation",
            lambda: (
                run_continuous_train_continuation(
                    source,
                    training,
                    references=references,
                    updates=8,
                    update_rule="bp_mixed",
                )
                if control is not None
                else run_continuous_train_continuation(
                    source, training, references=references, updates=8
                )
            ),
        )
        candidate = continued.candidate
        if (
            continued.report["complete"] is not True
            or type(candidate) is not NumpyContinuousBCLearner
        ):
            raise RuntimeError("new-task continuation incomplete")
        candidate_pin = _pin(candidate)
        pins.append((candidate, candidate_pin))
        if (candidate_pin[2].completed_updates, candidate_pin[2].training_examples) != (48, 3560):
            raise RuntimeError("fixed continuation counters differ")
        if mixed is not None:
            _require_reused_continuation(mixed, report)
        if native_endpoint or push_corrections or correction_readouts or balanced_push_corrections:
            assert longer is not None and retention is not None
            for cached in (longer, retention):
                _require_reused_source(cached, report)
                _require_reused_continuation(cached, report)
        native_candidate = candidate
        if longer_fit:
            extended = call(
                "longer_continuation",
                lambda: run_continuous_train_continuation(
                    candidate, training, references=references, updates=64, update_rule="bp_mixed"
                ),
            )
            _require_longer_continuation(extended.report)
            if type(extended.candidate) is not NumpyContinuousBCLearner:
                raise RuntimeError("longer TRAIN continuation returned no native candidate")
            extended_pin = _pin(extended.candidate)
            pins.append((extended.candidate, extended_pin))
            if (extended_pin[2].completed_updates, extended_pin[2].training_examples) != (
                112,
                9832,
            ):
                raise RuntimeError("longer TRAIN native counters differ")
            if extended_pin[2].policy.binary_gripper_axis != 3:
                raise RuntimeError("longer TRAIN native objective differs")
            diagnosed = call(
                "longer_readout",
                lambda: run_training_readout(
                    extended.candidate, diagnostic_groups, label="mixed112", guard=guard
                ),
            )
            if diagnosed["complete"] is not True:
                raise RuntimeError("longer TRAIN endpoint diagnosis incomplete")
            if (
                native_endpoint
                or push_corrections
                or correction_readouts
                or balanced_push_corrections
            ):
                assert longer is not None
                _require_reused_longer(longer, report)
                native_candidate = extended.candidate
        if reserved_probes:
            probe("probe_after", candidate, candidate_pin)
        if training_diagnostics:
            diagnosed = call(
                "fit_after",
                lambda: run_training_readout(
                    candidate,
                    diagnostic_groups,
                    label="mixed48" if control is not None else "head48",
                    guard=guard,
                ),
            )
            if diagnosed["complete"] is not True:
                raise RuntimeError("candidate TRAIN fit diagnosis incomplete")
        after = call(
            "after_push",
            lambda: (
                deepcopy(mixed["after_push"])
                if mixed is not None
                else roll_push(candidate, label="after")
            ),
        )
        if after["complete"] is not True:
            raise RuntimeError("post-update Push return incomplete")
        report["push_initial_exact"] = _same_push_reset(before, after)
        save()
        if report["push_initial_exact"] is not True:
            raise RuntimeError("fixed Push before/after reset binding differs")
        if push_corrections:
            assert prior_native is not None
            _require_reused_source(prior_native, report)
            _require_reused_continuation(prior_native, report)
            _require_reused_longer(prior_native, report)
            collected = call(
                "push_corrections",
                lambda: collect_push_cube_corrections(
                    native_candidate, seed=17, role="train", check_boundary=guard
                ),
            )
            packet = _release_push_corrections(
                collected, prior_native["native_push"], checkpoint=112
            )
            guard()  # Release validation is a boundary too; do not publish a corrupted return.
            report["push_correction_target_packet"] = asdict(packet)
            report["push_correction_release_at"] = 1100
            save()
        endpoint_updates, endpoint_rows = (120, 11016) if correction_fit else (112, 9832)
        if correction_fit:
            assert prior_native is not None and correction_packet is not None
            _require_reused_source(prior_native, report)
            _require_reused_continuation(prior_native, report)
            _require_reused_longer(prior_native, report)
            if balanced_push_corrections or aggregated_correction_fit:
                assert inherited_balanced is not None
                _require_reused_source(inherited_balanced, report)
                _require_reused_continuation(inherited_balanced, report)
                _require_reused_longer(inherited_balanced, report)
                if (
                    report["correction_fit_protocol"]
                    != inherited_balanced["correction_fit_protocol"]
                ):
                    raise RuntimeError("balanced student original correction data/dose differs")
            if balanced_correction_fit or balanced_push_corrections or aggregated_correction_fit:
                assert corrective_control is not None and corrective_readouts is not None
                for cached in (corrective_control, corrective_readouts):
                    _require_reused_source(cached, report)
                    _require_reused_continuation(cached, report)
                    _require_reused_longer(cached, report)
                    if report["correction_fit_protocol"] != cached["correction_fit_protocol"]:
                        raise RuntimeError(
                            "balanced corrective original data/dose differs from uniform control"
                        )
            if correction_readouts:
                assert corrective_control is not None
                _require_reused_source(corrective_control, report)
                _require_reused_continuation(corrective_control, report)
                _require_reused_longer(corrective_control, report)
                corrective_groups = tuple(
                    zip(
                        ("corrections", "push", "a", "b", "train20"),
                        (correction_packet, training, *references),
                        strict=True,
                    )
                )
                diagnosed = call(
                    "corrective_readout112",
                    lambda: run_training_readout(
                        native_candidate,
                        corrective_groups,
                        label="corrective112",
                        guard=guard,
                        corrective=True,
                    ),
                )
                _require_corrective_readout(diagnosed, 112, 9832)
            if zero_velocity_fit:
                assert inherited_balanced is not None
                _require_reused_source(inherited_balanced, report)
                _require_reused_continuation(inherited_balanced, report)
                _require_reused_longer(inherited_balanced, report)
                if (
                    report["correction_fit_protocol"]
                    != inherited_balanced["correction_fit_protocol"]
                ):
                    raise RuntimeError("private velocity control original data/dose differs")
                _require_corrective_readout(
                    inherited_balanced["balanced_readout120"], 120, 11016, label="balanced120"
                )
                prepared = call(
                    "zero_velocity_preparation",
                    lambda: prepare_zero_velocity_bc(native_candidate, check_boundary=guard),
                )
                preparation = prepared.report
                expected_work = dict(
                    source_fork_attempts=1,
                    source_fork_returns=1,
                    additional_source_owners=1,
                    velocity_zero_array_attempts=4,
                    velocity_zero_array_returns=4,
                    velocity_values_zeroed=1092,
                    velocity_numeric_bytes_zeroed=8736,
                    restore_attempts=1,
                    restore_returns=1,
                    fits=0,
                    parameter_initializations=0,
                    seeded_normal_values=0,
                    parameter_momentum_numeric_bytes_per_source=17472,
                )
                if (
                    preparation["complete"] is not True
                    or preparation["source_unchanged"] is not True
                    or preparation["definition"] != "continuous_private_zero_velocity_bc_v1"
                    or preparation["initial_counts"] != dict(updates=112, training_rows=9832)
                    or any(
                        type(preparation["work"].get(key)) is not int
                        or preparation["work"][key] != value
                        for key, value in expected_work.items()
                    )
                    or type(prepared.candidate) is not NumpyContinuousBCLearner
                ):
                    raise RuntimeError("private zero-velocity preparation incomplete")
                prepared_pin = _pin(prepared.candidate)
                if (prepared_pin[2].completed_updates, prepared_pin[2].training_examples) != (
                    112,
                    9832,
                ):
                    raise RuntimeError("private zero-velocity source counters differ")
                pins.append((prepared.candidate, prepared_pin))
                native_candidate = prepared.candidate
            weight_options: dict[str, Any] = (
                {"row_weights": np.asarray(balanced_weights, dtype=np.float64)}
                if balanced_correction_fit
                or zero_velocity_fit
                or balanced_push_corrections
                or aggregated_correction_fit
                else {}
            )
            corrected = call(
                "correction_continuation",
                lambda: run_continuous_train_continuation(
                    native_candidate,
                    correction_packet,
                    references=(training, *references),
                    updates=8,
                    update_rule="bp_mixed",
                    **weight_options,
                ),
            )
            if (
                balanced_correction_fit
                or zero_velocity_fit
                or balanced_push_corrections
                or aggregated_correction_fit
            ):
                _require_longer_continuation(
                    corrected.report, correction_fit=True, row_weights=balanced_weights
                )
            else:
                _require_longer_continuation(corrected.report, correction_fit=True)
            if type(corrected.candidate) is not NumpyContinuousBCLearner:
                raise RuntimeError("corrective fit returned no native candidate")
            corrected_pin = _pin(corrected.candidate)
            if (corrected_pin[2].completed_updates, corrected_pin[2].training_examples) != (
                120,
                11016,
            ) or corrected_pin[2].policy.binary_gripper_axis != 3:
                raise RuntimeError("corrective fit native endpoint differs")
            pins.append((corrected.candidate, corrected_pin))
            native_candidate = corrected.candidate
            if correction_readouts:
                assert corrective_control is not None
                if not zero_velocity_fit:
                    _require_reused_continuation(
                        corrective_control, report, key="correction_continuation"
                    )
                if json.dumps(
                    report["correction_fit_protocol"], sort_keys=True, allow_nan=False
                ) != json.dumps(
                    corrective_control["correction_fit_protocol"], sort_keys=True, allow_nan=False
                ):
                    raise RuntimeError("reconstructed corrective data/protocol differs")
                diagnosed = call(
                    "corrective_readout120",
                    lambda: run_training_readout(
                        native_candidate,
                        corrective_groups,
                        label="corrective120",
                        guard=guard,
                        corrective=True,
                    ),
                )
                _require_corrective_readout(diagnosed, 120, 11016)
            if balanced_push_corrections or aggregated_correction_fit:
                assert inherited_balanced is not None
                _require_reused_continuation(
                    inherited_balanced, report, key="correction_continuation"
                )
            if balanced_correction_fit or balanced_push_corrections or aggregated_correction_fit:
                diagnosed = call(
                    "balanced_readout120",
                    lambda: run_training_readout(
                        native_candidate,
                        corrective_groups,
                        label="balanced120",
                        guard=guard,
                        corrective=True,
                    ),
                )
                _require_corrective_readout(diagnosed, 120, 11016, label="balanced120")
        if aggregated_correction_fit:
            assert (
                inherited_balanced is not None
                and current_packet is not None
                and correction_packet is not None
            )
            if json.dumps(
                report["balanced_readout120"], sort_keys=True, allow_nan=False
            ) != json.dumps(
                inherited_balanced["balanced_readout120"], sort_keys=True, allow_nan=False
            ):
                raise RuntimeError("reconstructed balanced student readout differs")
            diagnosed = call(
                "aggregated_readout120",
                lambda: run_training_readout(
                    native_candidate,
                    aggregated_groups,
                    label="aggregated120",
                    guard=guard,
                    corrective=True,
                    aggregated=True,
                ),
            )
            _require_corrective_readout(
                diagnosed, 120, 11016, label="aggregated120", aggregated=True
            )
            expanded = call(
                "aggregated_continuation",
                lambda: run_continuous_train_continuation(
                    native_candidate,
                    current_packet,
                    references=(correction_packet, training, *references),
                    updates=8,
                    update_rule="bp_mixed",
                    maximum_rows=198,
                    row_weights=np.asarray(aggregated_weights, dtype=np.float64),
                ),
            )
            _require_longer_continuation(
                expanded.report,
                correction_fit=True,
                row_weights=aggregated_weights,
                aggregated=True,
            )
            if type(expanded.candidate) is not NumpyContinuousBCLearner:
                raise RuntimeError("aggregated fit returned no native candidate")
            expanded_pin = _pin(expanded.candidate)
            if (expanded_pin[2].completed_updates, expanded_pin[2].training_examples) != (
                128,
                12600,
            ) or expanded_pin[2].policy.binary_gripper_axis != 3:
                raise RuntimeError("aggregated fit native endpoint differs")
            pins.append((expanded.candidate, expanded_pin))
            native_candidate = expanded.candidate
            diagnosed = call(
                "aggregated_readout128",
                lambda: run_training_readout(
                    native_candidate,
                    aggregated_groups,
                    label="aggregated128",
                    guard=guard,
                    corrective=True,
                    aggregated=True,
                ),
            )
            _require_corrective_readout(
                diagnosed, 128, 12600, label="aggregated128", aggregated=True
            )
            endpoint_updates, endpoint_rows = 128, 12600
        if balanced_push_corrections:
            assert inherited_balanced is not None
            # Why: queries may run only after exact observed reconstruction evidence;
            # matching counters alone cannot establish the current student binding.
            if json.dumps(
                report["balanced_readout120"], sort_keys=True, allow_nan=False
            ) != json.dumps(
                inherited_balanced["balanced_readout120"], sort_keys=True, allow_nan=False
            ):
                raise RuntimeError("reconstructed balanced student readout differs")
            collected = call(
                "push_corrections",
                lambda: collect_push_cube_corrections(
                    native_candidate, seed=17, role="train", check_boundary=guard
                ),
            )
            if (
                type(collected.report["source_updates"]) is not int
                or collected.report["source_updates"] != 120
                or type(collected.report["source_training_rows"]) is not int
                or collected.report["source_training_rows"] != 11016
            ):
                raise RuntimeError("balanced student collection source counters differ")
            packet = _release_push_corrections(
                collected, inherited_balanced["native_push"], checkpoint=120
            )
            guard()
            report["push_correction_target_packet"] = asdict(packet)
            report["push_correction_release_at"] = 1100
            save()
        if native_endpoint:
            native_push = call(
                "native_push", lambda: roll_push(native_candidate, label=f"mixed{endpoint_updates}")
            )
            if (
                native_push["complete"] is not True
                or native_push["source_unchanged"] is not True
                or native_push["unreturned_prediction_possible"] is not False
                or native_push["unreturned_native_work_unknown"] is not False
            ):
                raise RuntimeError(f"native{endpoint_updates} Push return incomplete")
            report["native_push_initial_exact"] = _same_push_reset(
                inherited_balanced["native_push"]
                if aggregated_correction_fit and inherited_balanced is not None
                else corrective_control["native_push"]
                if balanced_correction_fit and corrective_control is not None
                else prior_native["native_push"]
                if correction_fit and prior_native is not None
                else before,
                native_push,
            )
            save()
            if report["native_push_initial_exact"] is not True:
                raise RuntimeError(f"native{endpoint_updates} Push reset binding differs")
        native_seeds = (
            ()
            if diagnostics_only or push_corrections or balanced_push_corrections
            else (17, 19, 20)
        )
        native_paths = (
            () if diagnostics_only or push_corrections or balanced_push_corrections else paths
        )
        for seed, path in zip(native_seeds, native_paths, strict=True):
            rolled = call(
                f"pick-{seed}",
                lambda: run_frozen_rollout(
                    path, learner=native_candidate, train_seed=seed, action_history_recipe=recipe
                ),
                pick=True,
            )
            if rolled.report["complete"] is not True or rolled.corrections is not None:
                raise RuntimeError("post-update frozen Pick return incomplete")
            if native_endpoint:
                assert retention is not None
                raw = rolled.report
                if (
                    raw["source_updates"] != endpoint_updates
                    or raw["source_training_rows"] != endpoint_rows
                    or type(raw["source_updates"]) is not int
                    or type(raw["source_training_rows"]) is not int
                    or raw["frozen_state_exact"] is not True
                    or raw["original_owner_exact"] is not True
                    or raw["action_history_exact"] is not True
                    or raw["optimizer_updates"] != 0
                    or type(raw["prediction_attempts"]) is not int
                    or raw["prediction_attempts"] != raw["prediction_returns"]
                    or raw["prediction_returns"] != len(raw["episode"]["transitions"])
                ):
                    raise RuntimeError(
                        f"native{endpoint_updates} Pick endpoint or frozen prediction work differs"
                    )
                native_baseline = (
                    inherited_balanced
                    if aggregated_correction_fit
                    else corrective_control
                    if balanced_correction_fit
                    else prior_native
                    if correction_fit
                    else retention
                )
                assert native_baseline is not None
                old = native_baseline["pick"][len(report["pick"]) - 1]
                binding = ("seed", "role", "task", "context", "controller", "observation_recipe")
                expected, actual = (
                    dict(
                        observation=value["episode"]["transitions"][0]["observation"],
                        metadata={key: value[key] for key in binding},
                        history_recipe=value["action_history"]["recipe"],
                    )
                    for value in (old, raw)
                )
                if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
                    expected, sort_keys=True, allow_nan=False
                ):
                    raise RuntimeError(
                        f"native{endpoint_updates} Pick reset binding differs from accepted control"
                    )
        guard()
        keys = ("final_success", "any_success", "sparse_return")
        report["comparisons"] = dict(
            direction="after-minus-before; positive success/return is better; positive MSE is worse",
            push={key: float(after[key]) - float(before[key]) for key in keys},
            pick=[
                dict(
                    seed=seed,
                    **{key: float(raw[key]) - 1.0 for key in keys},
                    controls=len(raw["episode"]["transitions"]),
                    control_delta=len(raw["episode"]["transitions"]) - old_controls,
                )
                for seed, raw, old_controls in zip(
                    native_seeds,
                    report["pick"],
                    ()
                    if diagnostics_only or push_corrections or balanced_push_corrections
                    else (17, 20, 20),
                    strict=True,
                )
            ],
            reused_pick_source="artifacts/runs/joint-acquisition-native-cli-20261021/run.json",
            reused_pick_scope="known TRAIN source40/2776, nonconcurrent accepted controls",
            source_acquisition_fit_deltas={
                group: acquired.report["residuals"]["after"][group]["result"]["overall"][
                    "normalized_half_mse"
                ]
                - acquired.report["residuals"]["before"][group]["result"]["overall"][
                    "normalized_half_mse"
                ]
                for group in ("a", "b", "curated_a", "retained_train20")
            },
            continuation_group_fit=(
                "measured in explicit fit_before/fit_after TRAIN readouts"
                if training_diagnostics
                else "not measured; no unbudgeted extra predictions"
            ),
        )
        if diagnostics_only:
            report["comparisons"]["scope"] = (
                "isolated reserved-reset probes; Push scores historical; no new native results"
                if reserved_probes
                else "TRAIN readouts only; Push scores historical; no new native results"
            )
        if reserved_probes:
            report["comparisons"]["reserved_probe_metrics"] = {
                key: deepcopy(report[key]["metrics"]) for key in ("probe_before", "probe_after")
            }
            report["comparisons"]["reserved_probe_directions"] = (
                "fresh gains=before-minus-after, higher better; old retention loss=after-minus-before, "
                "lower better; different native objectives, no causal optimizer/family advantage"
            )
        if longer_fit:
            report["comparisons"]["scope"] = (
                "TRAIN fit at mixed112 only; historical Push scores belong to mixed48; "
                "no new native results for the longer candidate"
            )
            report["comparisons"]["continuation_group_fit"] = (
                "measured at mixed112 in longer_readout"
            )
        if native_endpoint:
            assert retention is not None
            report["comparisons"]["scope"] = (
                "native_push and pick are mixed112; after_push is historical mixed48; "
                "source40 comparisons and mixed48 comparisons are nonconcurrent descriptive controls"
            )
            old_push = retention["after_push"]
            report["comparisons"]["native_endpoint_vs_mixed48"] = dict(
                checkpoint=endpoint_updates,
                baseline_checkpoint=48,
                push=dict(
                    **{key: float(native_push[key]) - float(old_push[key]) for key in keys},
                    controls=len(native_push["transitions"]),
                    control_delta=len(native_push["transitions"]) - len(old_push["transitions"]),
                ),
                pick=[
                    dict(
                        seed=seed,
                        **{key: float(raw[key]) - float(old[key]) for key in keys},
                        controls=len(raw["episode"]["transitions"]),
                        control_delta=len(raw["episode"]["transitions"])
                        - len(old["episode"]["transitions"]),
                    )
                    for seed, raw, old in zip(
                        native_seeds, report["pick"], retention["pick"], strict=True
                    )
                ],
            )
        if (
            correction_fit
            and not correction_readouts
            and not balanced_push_corrections
            and not aggregated_correction_fit
        ):
            assert prior_native is not None
            report["comparisons"]["continuation_group_fit"] = (
                "original mixed112 measured in longer_readout; no post120 TRAIN readout"
            )
            report["comparisons"]["scope"] = (
                "native_push/pick are mixed120; original112 controls, historical48 traces remain distinct"
            )
            report["comparisons"]["native_endpoint_vs_mixed112"] = dict(
                checkpoint=120,
                baseline_checkpoint=112,
                push=dict(
                    **{
                        k: float(native_push[k]) - float(prior_native["native_push"][k])
                        for k in keys
                    },
                    controls=len(native_push["transitions"]),
                    control_delta=len(native_push["transitions"])
                    - len(prior_native["native_push"]["transitions"]),
                ),
                pick=[
                    dict(
                        seed=seed,
                        **{k: float(raw[k]) - float(old[k]) for k in keys},
                        controls=len(raw["episode"]["transitions"]),
                        control_delta=len(raw["episode"]["transitions"])
                        - len(old["episode"]["transitions"]),
                    )
                    for seed, raw, old in zip(
                        native_seeds, report["pick"], prior_native["pick"], strict=True
                    )
                ],
            )
            report["native_endpoint_protocol"].update(checkpoint=120, training_rows=11016)
        if correction_readouts:
            report["comparisons"]["scope"] = (
                "corrective_readout112/120 are detached TRAIN fit; all native scores historical"
            )
            report["comparisons"]["continuation_group_fit"] = (
                "five fixed TRAIN groups measured at112 and120; no physical acquisition inference"
            )
            report["comparisons"]["corrective_fit_deltas"] = _corrective_deltas(
                report["corrective_readout112"], report["corrective_readout120"]
            )
            report["comparisons"]["corrective_direction"] = (
                "120-minus112; lower physical half-MSE/native objective better; all axes and initial/remainder kept in raw readouts"
            )
        if balanced_correction_fit:
            assert corrective_control is not None and corrective_readouts is not None
            report["comparisons"].update(
                scope=(
                    "balanced120 physical TRAIN endpoints versus accepted uniform120; "
                    "equal new update/row dose, nonconcurrent and unequal total compute; "
                    "all groups/scenes retained, no generalization or CPC inference"
                ),
                continuation_group_fit="balanced120 detached five-group readout; uniform112/120 readouts reused",
                balanced_fit_vs_uniform120=_corrective_deltas(
                    corrective_readouts["corrective_readout120"], report["balanced_readout120"]
                ),
                balanced_fit_vs_112=_corrective_deltas(
                    corrective_readouts["corrective_readout112"], report["balanced_readout120"]
                ),
                balanced_fit_direction="balanced-minus-control; lower physical MSE/unweighted mixed diagnostic loss better",
                balanced_native_vs_uniform120=dict(
                    checkpoint=120,
                    baseline_checkpoint=120,
                    push=dict(
                        **{
                            key: float(native_push[key])
                            - float(corrective_control["native_push"][key])
                            for key in keys
                        },
                        controls=len(native_push["transitions"]),
                        control_delta=len(native_push["transitions"])
                        - len(corrective_control["native_push"]["transitions"]),
                    ),
                    pick=[
                        dict(
                            seed=seed,
                            **{key: float(raw[key]) - float(old[key]) for key in keys},
                            controls=len(raw["episode"]["transitions"]),
                            control_delta=len(raw["episode"]["transitions"])
                            - len(old["episode"]["transitions"]),
                        )
                        for seed, raw, old in zip(
                            native_seeds, report["pick"], corrective_control["pick"], strict=True
                        )
                    ],
                ),
            )
        if zero_velocity_fit:
            assert inherited_balanced is not None
            report["comparisons"].update(
                scope="private TRAIN velocity-only fit; same148rows/weights/eightupdates, inherited control reused nonconcurrently; no physical results",
                zero_velocity_fit_vs_inherited=_corrective_deltas(
                    inherited_balanced["balanced_readout120"], report["corrective_readout120"]
                ),
                zero_velocity_fit_vs_112=_corrective_deltas(
                    report["corrective_readout112"], report["corrective_readout120"]
                ),
                zero_velocity_direction="zero-minus-inherited; lower physical half-MSE/unweighted objective better; all axes/stages retained",
            )
        if push_corrections:
            report["comparisons"]["scope"] = (
                "push_corrections is frozen112 student supervision only; Push scores historical48; "
                "no correction fit or new acquisition claim"
            )
        if balanced_push_corrections:
            report["comparisons"]["scope"] = (
                "frozen balanced120 student supervision only; historical48 Push scores; "
                "same accepted fit reconstructed, no new-label fit or new acquisition claim"
            )
            report["comparisons"]["continuation_group_fit"] = (
                "balanced120 detached five-group readout exactly matches accepted control"
            )
        if aggregated_correction_fit:
            assert inherited_balanced is not None
            report["comparisons"].update(
                scope="aggregated128 native TRAIN endpoints versus accepted balanced120; unequal data/compute, nonconcurrent; historical48/source40 distinct; no generalization or CPC inference",
                continuation_group_fit="six fixed TRAIN groups measured at120 and128; all axes/initial/remainder retained",
                aggregated_fit_vs120=_corrective_deltas(
                    report["aggregated_readout120"], report["aggregated_readout128"]
                ),
                aggregated_fit_direction="128-minus120; lower physical half-MSE/unweighted native objective better",
                aggregated_native_vs_balanced120=dict(
                    checkpoint=128,
                    baseline_checkpoint=120,
                    push=dict(
                        **{
                            key: float(native_push[key])
                            - float(inherited_balanced["native_push"][key])
                            for key in keys
                        },
                        controls=len(native_push["transitions"]),
                        control_delta=len(native_push["transitions"])
                        - len(inherited_balanced["native_push"]["transitions"]),
                    ),
                    pick=[
                        dict(
                            seed=seed,
                            **{key: float(raw[key]) - float(old[key]) for key in keys},
                            controls=len(raw["episode"]["transitions"]),
                            control_delta=len(raw["episode"]["transitions"])
                            - len(old["episode"]["transitions"]),
                        )
                        for seed, raw, old in zip(
                            native_seeds, report["pick"], inherited_balanced["pick"], strict=True
                        )
                    ],
                ),
            )
        report["source_unchanged"] = True
        report["complete"] = True
        save()
    except Exception as error:
        fail_capture(error)
    finally:
        elapsed = monotonic() - started
        if not failed:
            try:
                # The final clock is opaque too; check before complete capture.
                guard()
            except Exception as error:
                fail_capture(error)
        report["elapsed_seconds"] = elapsed
        report["observed_stage_returns"] = list(observed)
        report["unreturned_stage_work_unknown"] = len(observed) != len(report["attempted"])
        if report["complete"]:
            try:
                save()
            except Exception as error:
                fail_capture(error)
                _write(output, report)
        else:
            _write(output, report)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--profile",
        choices=("head_gauss_newton", "bp_mixed", "bp_mixed_retention"),
        default="head_gauss_newton",
    )
    parser.add_argument("--training-diagnostics", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--reserved-probes", action="store_true")
    parser.add_argument("--longer-fit", action="store_true")
    parser.add_argument("--native-endpoint", action="store_true")
    parser.add_argument("--push-corrections", action="store_true")
    parser.add_argument("--correction-fit", action="store_true")
    parser.add_argument("--correction-readouts", action="store_true")
    parser.add_argument("--balanced-correction-fit", action="store_true")
    parser.add_argument("--zero-velocity-fit", action="store_true")
    parser.add_argument("--balanced-push-corrections", action="store_true")
    parser.add_argument("--aggregated-correction-fit", action="store_true")
    args = parser.parse_args()
    result = run_study(
        args.output,
        profile=args.profile,
        training_diagnostics=args.training_diagnostics,
        diagnostics_only=args.diagnostics_only,
        reserved_probes=args.reserved_probes,
        longer_fit=args.longer_fit,
        native_endpoint=args.native_endpoint,
        push_corrections=args.push_corrections,
        correction_fit=args.correction_fit,
        correction_readouts=args.correction_readouts,
        balanced_correction_fit=args.balanced_correction_fit,
        zero_velocity_fit=args.zero_velocity_fit,
        balanced_push_corrections=args.balanced_push_corrections,
        aggregated_correction_fit=args.aggregated_correction_fit,
    )
    print(
        json.dumps(
            dict(
                complete=result["complete"],
                error=result["error"],
                elapsed_seconds=result["elapsed_seconds"],
            )
        )
    )
    raise SystemExit(0 if result["complete"] else 1)
