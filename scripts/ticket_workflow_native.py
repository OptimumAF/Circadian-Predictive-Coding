"""Native details for the existing chronological ticket recipe, not a runner.

Creates counted forks and JSON-safe complete checkpoint observations. Schedules,
data admission, task effects and budgets stay in the existing recipe/stream.
"""

from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
import json
from typing import Any

from src.adapters.torch_decision import TorchDecisionPredictor, _same_values
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.shared.torch_runtime import require_torch

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def counted_learner(method, feature_dim, num_classes, before_update, after_update):
    """Preserve one physical-call counter across the stream's actual owned forks."""

    class CountedNativeLearner(TorchFeatureLearner):
        def fork(self):
            child = CountedNativeLearner(self.policy)
            child.restore_state(self.snapshot_state())
            return child

        def train_batch(self, features, targets):
            before_update(features)
            result = super().train_batch(features, targets)
            after_update()
            return result

        def train_batch_with_rate_multiplier(self, features, targets, multiplier):
            if multiplier == 1:
                return super().train_batch_with_rate_multiplier(features, targets, multiplier)
            before_update(features)
            result = super().train_batch_with_rate_multiplier(features, targets, multiplier)
            after_update()
            return result

        def train_batch_with_gate_setting(self, features, targets, setting):
            if setting == "unchanged":
                return super().train_batch_with_gate_setting(features, targets, setting)
            before_update(features)
            result = super().train_batch_with_gate_setting(features, targets, setting)
            after_update()
            return result

    return CountedNativeLearner(
        TorchFeaturePolicy(method, feature_dim, 3, 17, num_classes=num_classes)
    )


def snapshot_record(value: Any) -> Any:
    """Complete scalar observation, not a persistence/restore format or authority."""
    torch = require_torch()
    if type(value) is torch.Tensor:
        return dict(dtype=str(value.dtype), shape=list(value.shape), values=value.tolist())
    if is_dataclass(value):
        return {f.name: snapshot_record(getattr(value, f.name)) for f in fields(value)}
    if type(value) is dict:
        return {key: snapshot_record(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [snapshot_record(item) for item in value]
    return value


def tensor_bytes(value: Any) -> int:
    torch = require_torch()
    if type(value) is torch.Tensor:
        return int(value.numel() * value.element_size())
    if is_dataclass(value):
        return sum(tensor_bytes(getattr(value, f.name)) for f in fields(value))
    if type(value) is dict:
        return sum(tensor_bytes(item) for item in value.values())
    if type(value) in (list, tuple):
        return sum(tensor_bytes(item) for item in value)
    return 0


def checkpoint(point, state):
    return dict(
        source_updates=point,
        source_rows=state.training_examples,
        source_mistakes=None,  # Native CE/local energy is not a perceptron mistake count.
        state=snapshot_record(state),
        tensor_bytes=tensor_bytes(state),
    )


def readout(spec, state):
    return TorchDecisionPredictor(spec, state)


def same_state(left, right):
    return _same_values(left, right)


def _policy_phase_records(report):
    """Validate small clock/role metadata before any native state is decoded."""
    from scripts.ticket_controller_streams import POLICY_CHANGE_CHECKPOINTS

    if report.get("policy_change") is not True or report.get("policy_capture_complete") is not True:
        raise ValueError("policy development requires complete original phase captures")
    records = report.get("policy_checkpoints")
    if type(records) is not list or len(records) != 4:
        raise ValueError("policy development requires exactly four phase captures")
    for row, (tick, version) in zip(records, POLICY_CHANGE_CHECKPOINTS, strict=True):
        if type(row) is not dict or type(row.get("checkpoint")) is not dict:
            raise ValueError("policy phase requires complete scalar observation records")
        point = row.get("checkpoint", {})
        if type(point.get("state")) is not dict:
            raise ValueError("policy phase requires complete scalar native state")
        updates = point.get("source_updates")
        if (
            type(row.get("opportunity")) is not int
            or row["opportunity"] != tick
            or row.get("policy_version") != version
            or type(updates) is not int
            or not 0 <= updates <= tick
            or type(row.get("applied_receipts")) is not int
            or row["applied_receipts"] != updates
            or type(point.get("source_rows")) is not int
            or point.get("source_rows") != 576 + 6 * updates
            or type(point["state"].get("completed_updates")) is not int
            or point.get("state", {}).get("completed_updates") != 96 + updates
            or type(point["state"].get("training_examples")) is not int
            or point.get("state", {}).get("training_examples") != 576 + 6 * updates
        ):
            raise ValueError("policy phase clocks/version/receipt counts differ")
    if records[-1]["checkpoint"] != report.get("candidate_checkpoint"):
        raise ValueError("policy final phase differs from the original final candidate")
    return records


def policy_development_readout(spec, report, require_input):
    """Score four frozen phases after TRAIN; no update or final-source authority.

    Existing native decoder/readout and sandbox own model copies and task effects.
    A deferred caller controls its later development-evaluation chronology.
    """
    from scripts.recurring_ticket_workflow_example import episode_slots, summarize_policy_changes
    from src.adapters.torch_bp_record import bp_snapshot_from_record
    from src.app.ticket_workflow import run_workflow_episode
    from src.core.ticket_workflow import ACTION_IDS, WorkflowCapabilities, WorkflowState

    records = _policy_phase_records(report)
    pin = deepcopy(records)

    def require_original():
        require_input()
        if (
            report.get("policy_change") is not True
            or report.get("policy_capture_complete") is not True
            or report.get("policy_checkpoints") != pin
            or report.get("candidate_checkpoint") != pin[-1]["checkpoint"]
        ):
            raise ValueError("original policy phase evidence changed during readout")

    episodes = report["policy_development_episodes"] = []
    paid = report["policy_readout_work"] = []
    report["policy_readout_complete"] = False
    for row in records:
        require_original()
        state = bp_snapshot_from_record(
            json.loads(json.dumps(row["checkpoint"]["state"], allow_nan=False))
        )
        reader = readout(spec, state)
        work = dict(opportunity=row["opportunity"], work=asdict(reader.work))
        paid.append(work)

        def predict(request):
            require_original()
            try:
                return reader(request)
            finally:
                work["work"] = asdict(reader.work)

        slots = tuple(("ordinary", c) for c in ("invoice", "parcel", "password")) + tuple(
            (s, c) for s, c in episode_slots(row["opportunity"]) if s != "ordinary"
        )
        for scenario, concept in slots:
            require_original()
            episode = run_workflow_episode(
                WorkflowState(
                    f"policy-dev-{row['opportunity']}-{concept}",
                    concept,
                    policy_version=row["policy_version"],
                ),
                predict,
                capabilities=WorkflowCapabilities(ACTION_IDS),
                max_steps=6,
                scenario=scenario,
            )
            episodes.append(
                dict(
                    opportunity=row["opportunity"],
                    source_updates=row["checkpoint"]["source_updates"],
                    policy_version=row["policy_version"],
                    concept=concept,
                    role="outer_selection",
                    source_group="development-native-policy-workflow",
                    episode=episode,
                )
            )
            require_original()
            reader.assert_unchanged()
            if not episode["complete"]:
                raise ValueError("uncertain paid policy development episode")
    report["policy_summary"] = summarize_policy_changes(episodes, opportunity_axis=True)
    report["policy_readout_complete"] = True


def native_costs(report, attempted_rows, method):
    forwards = sum(row["work"]["native_forward_attempts"] for row in report["native_readout_work"])
    # Why this: these nominal attempted affine terms are not executed FLOPs on
    # failed calls and do not include backward/latent work, validation or copies.
    terms = 7 * 3 + 3 * 6
    return dict(
        model_random_draws=None,
        native_initialization="seed-local scalar slots and physical constructors externally metered",
        attempted_training_nominal_affine_terms=attempted_rows * terms,
        attempted_readout_nominal_affine_terms=forwards * terms,
        attempted_pc_latent_iterations=(
            0 if method == "bp" else 2 * report["training_work"]["attempted_updates"]
        ),
        returned_bp_optimizer_steps=(
            report["training_work"]["returned_updates"] if method == "bp" else 0
        ),
        retained_checkpoint_tensor_bytes=sum(row["tensor_bytes"] for row in report["checkpoints"]),
        qualification="nominal attempted full-class affine terms; BP backward/PC latent steps, tanh, initialization, RNG, copies, validation, decoding, Python and other live owners additional; failures may execute less or leave work unknown",
    )
