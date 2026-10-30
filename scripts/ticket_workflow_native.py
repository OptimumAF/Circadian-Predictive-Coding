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
from src.app.coupled_attribution import _same_value

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


class ClassBalancedUpdates:
    """At most twelve current-batch effects; the existing stream owns label release.

    Why: static TRAIN diagnostics include future contexts. Only targets delivered
    to the actual train call may set these weights. No teacher/readout is used.
    """

    def __init__(self, report):
        self.report = report
        self.rows = report["class_balanced_updates"] = []
        self.pins = []
        self.receipt_pins: tuple[tuple[Any, dict[str, Any]], ...] = ()
        self.pending = None
        self.attempts = self.returns = 0

    def require_unchanged(self):
        if (
            self.report.get("class_balanced_updates") is not self.rows
            or len(self.rows) != len(self.pins)
            or any(
                row is not original or not same_state(row, value)
                for row, (original, value) in zip(self.rows, self.pins, strict=True)
            )
        ):
            raise ValueError("class-balanced paid evidence changed")

    def require_receipts(self, receipts):
        self.require_unchanged()
        if (
            len(receipts) != len(self.pins)
            or len(self.receipt_pins) > len(receipts)
            or any(
                receipt is not owner or not same_state(asdict(receipt), value)
                for receipt, (owner, value) in zip(receipts, self.receipt_pins)
            )
            or any(
                not value["complete"]
                or not same_state(
                    asdict(receipt), dict(**value["source_receipt"], diagnostic=value["diagnostic"])
                )
                for receipt, (_, value) in zip(receipts, self.pins, strict=True)
            )
        ):
            raise ValueError("class-balanced causal receipt mismatch")
        self.receipt_pins = tuple((receipt, asdict(receipt)) for receipt in receipts)

    def admit(self, source, label, now, learner_version):
        """Witness the exact ready source/label before delegating native work."""
        from src.core.experience import Experience, LabelArrival

        self.require_unchanged()
        if (
            self.pending is not None
            or type(source) is not Experience
            or type(label) is not LabelArrival
            or source.role != "train"
            or label.role != "train"
            or source.permissions.training is not True
            or source.key != label.key
            or source.model_version != label.model_version
            or source.observed_at + 1 != label.arrived_at
            or label.arrived_at > now
        ):
            raise ValueError("class-balanced update requires one original arrived TRAIN label")
        witness = dict(
            sample_id=source.sample_id,
            episode_id=source.episode_id,
            event_id=label.event_id,
            actor_version=source.model_version,
            learner_version=learner_version,
            observed_at=source.observed_at,
            arrived_at=label.arrived_at,
            applied_at=now,
            update_number=len(self.pins) + 1,
        )
        self.pending = (source, label, deepcopy((source, label)), witness)

    def train(self, learner, features, targets):
        import numpy as np
        from src.adapters.torch_array_weighted import (
            TORCH_ARRAY_WEIGHTED_CE_ID,
            train_class_weighted_batch,
        )
        from src.core.demonstration_class_weights import validate_class_weights

        self.require_unchanged()
        if self.pending is None:
            raise ValueError("class-balanced update requires an admitted TRAIN label")
        source, label, source_pin, witness = self.pending
        if (
            not _same_value((source, label), source_pin)
            or not _same_value(features, source.features)
            or not _same_value(targets, label.targets)
        ):
            raise ValueError("class-balanced admitted TRAIN data changed")
        if len(self.pins) >= 12 or learner.completed_updates != len(self.pins):
            raise ValueError("class-balanced update prefix or twelve-update limit changed")
        if (
            type(targets) is not np.ndarray
            or targets.dtype != np.float64
            or targets.shape != (len(features), 1)
            or not 1 <= len(targets) <= 128
            or not np.isfinite(targets).all()
            or not ((targets >= 0) & (targets < learner.policy.num_classes)).all()
            or not (targets == np.floor(targets)).all()
        ):
            raise ValueError("class-balanced update requires current hard TRAIN targets")
        counts = tuple(
            int(x)
            for x in np.bincount(
                targets[:, 0].astype(np.int64), minlength=learner.policy.num_classes
            )
        )
        present = sum(count > 0 for count in counts)
        weights = validate_class_weights(
            tuple(float(len(targets) / (present * count)) if count else 1.0 for count in counts),
            learner.policy.num_classes,
        )
        row = dict(
            source_update=len(self.pins) + 1,
            role="train",
            rows=len(targets),
            class_counts=counts,
            class_weights=weights,
            objective=TORCH_ARRAY_WEIGHTED_CE_ID,
            source_receipt=deepcopy(witness),
            attempted_updates=1,
            returned_updates=0,
            complete=False,
            unreturned_work_unknown=True,
        )
        self.attempts += 1
        try:
            result = train_class_weighted_batch(learner, features, targets, weights)
            self.returns += 1
            row.update(returned_updates=1, unreturned_work_unknown=False, diagnostic=asdict(result))
            self.require_unchanged()
            if not _same_value((source, label), source_pin):
                raise ValueError("class-balanced admitted TRAIN data changed")
            if result.definition != TORCH_ARRAY_WEIGHTED_CE_ID:
                raise ValueError("class-balanced objective changed")
            row["complete"] = True
            return result
        finally:
            self.pending = None
            self.rows.append(row)
            self.pins.append((row, deepcopy(row)))

    def finish(self):
        """Keep physical return counts and the original paid prefix on failure."""
        self.report["class_balanced_work"] = dict(
            attempted_updates=self.attempts,
            returned_updates=self.returns,
            unreturned_work_unknown=self.attempts != self.returns,
        )
        try:
            self.require_unchanged()
        except ValueError:
            self.report["preserved_class_balanced_updates"] = [deepcopy(v) for _, v in self.pins]
            self.report.update(complete=False, error="class-balanced paid evidence changed")


def counted_learner(
    method, feature_dim, num_classes, before_update, after_update, *, balanced=None
):
    """Preserve one physical-call counter across the stream's actual owned forks."""
    if balanced is not None and (
        type(balanced) is not ClassBalancedUpdates
        or method != "bp"
        or feature_dim != 9
        or num_classes != 6
    ):
        raise ValueError("class-balanced factory requires the BP D9/K6 recipe")

    class CountedNativeLearner(TorchFeatureLearner):
        def fork(self):
            child = CountedNativeLearner(self.policy)
            child.restore_state(self.snapshot_state())
            return child

        def train_batch(self, features, targets):
            before_update(features)
            result = (
                super().train_batch(features, targets)
                if balanced is None
                else balanced.train(self, features, targets)
            )
            after_update()
            return result

        def train_batch_with_rate_multiplier(self, features, targets, multiplier):
            if balanced is not None and multiplier != 1:
                raise ValueError("class-balanced recipe has no rate controller")
            if multiplier == 1:
                return super().train_batch_with_rate_multiplier(features, targets, multiplier)
            before_update(features)
            result = super().train_batch_with_rate_multiplier(features, targets, multiplier)
            after_update()
            return result

        def train_batch_with_gate_setting(self, features, targets, setting):
            if balanced is not None and setting != "unchanged":
                raise ValueError("class-balanced recipe has no gate controller")
            if setting == "unchanged":
                return super().train_batch_with_gate_setting(features, targets, setting)
            before_update(features)
            result = super().train_batch_with_gate_setting(features, targets, setting)
            after_update()
            return result

        def train_batch_with_rate_and_gate(self, features, targets, multiplier, setting):
            if balanced is not None and (multiplier != 1 or setting != "unchanged"):
                raise ValueError("class-balanced recipe has no composite controller")
            if multiplier == 1 or setting == "unchanged":
                return super().train_batch_with_rate_and_gate(
                    features, targets, multiplier, setting
                )
            before_update(features)
            result = super().train_batch_with_rate_and_gate(features, targets, multiplier, setting)
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


def require_readout(predictor, spec, state):
    """Bind complete frozen state/spec while permitting paid inference counters."""
    if (
        type(predictor) is not TorchDecisionPredictor
        or predictor._spec != spec
        or predictor._initial_spec != spec
        or not same_state(predictor._initial, state)
    ):
        raise ValueError("context frozen native readout binding changed")
    predictor.assert_unchanged()


def same_state(left, right):
    return _same_values(left, right)


def prepare_train_fit(spec):
    """Original TRAIN truth only; never an online feedback or selection source."""
    from scripts.ticket_train_fit import context_teacher
    from scripts.run_ticket_workflow import training_cases

    return context_teacher(spec, training_cases)


def record_train_fit(predictor, source, row, require_source):
    """Reuse the frozen readout; retain attempted work even when no logits return."""
    import numpy as np
    from scripts.ticket_train_fit import teacher_fit
    from scripts.run_ticket_workflow import CONTEXT_TASKS

    row.update(complete=False, role="train")
    attempted = returned = returned_rows = summarized = 0

    def save_work():
        row["work"] = dict(
            native_forward_attempts=attempted,
            rows_attempted=12 * attempted,
            native_forward_returns=returned,
            rows_returned=returned_rows,
            summary_returns=summarized,
            unreturned_work_unknown=attempted != returned,
        )

    try:
        cases, batch = source
        require_source()
        if len(cases) != 12 or any(c.role != "train" for c in cases) or batch.role != "train":
            raise ValueError("fit diagnostics require original TRAIN roles")
        if batch.features.shape != (12, 9) or batch.targets.shape != (12, 1):
            raise ValueError("fit diagnostics require original D9 TRAIN geometry")
        attempted += 1
        save_work()
        work_owner = row["work"]
        work_pin = dict(work_owner)
        logits = predictor._learner.predict_logits(batch.features)
        returned += 1
        if type(logits) is np.ndarray and logits.ndim == 2:
            returned_rows = len(logits)
        if row.get("work") is not work_owner or not same_state(row["work"], work_pin):
            raise ValueError("pending TRAIN fit work changed")
        require_source()
        if (
            type(logits) is not np.ndarray
            or logits.shape != (12, 6)
            or not np.isfinite(logits).all()
        ):
            raise ValueError("fit diagnostics require finite complete native logits")
        row["fit"] = teacher_fit(logits, cases, batch, tasks=CONTEXT_TASKS)
        summarized += 1
        require_source()
        row["complete"] = True
    finally:
        save_work()


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


def native_costs(report, attempted_rows, method, *, feature_dim=7):
    forwards = sum(row["work"]["native_forward_attempts"] for row in report["native_readout_work"])
    # Why this: these nominal attempted affine terms are not executed FLOPs on
    # failed calls and do not include backward/latent work, validation or copies.
    # The existing factory fixes hidden width3/classes6; context return has D9.
    terms = feature_dim * 3 + 3 * 6
    costs = dict(
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
    if "train_fit_readouts" in report:
        work = [r.get("work") if type(r) is dict else None for r in report["train_fit_readouts"]]

        def known_total(field):
            total = 0
            for row in work:
                value = row.get(field) if type(row) is dict else None
                if type(value) is not int or value < 0:
                    return None
                total += value
            return total

        attempts = known_total("native_forward_attempts")
        returns = known_total("native_forward_returns")
        rows = known_total("rows_attempted")
        returned_rows = known_total("rows_returned")
        costs.update(
            train_fit_native_forward_attempts=attempts,
            train_fit_rows_attempted=rows,
            train_fit_rows_returned=returned_rows,
            train_fit_unreturned_work_unknown=(
                any(v is None for v in (attempts, returns, rows, returned_rows))
                or any(
                    type(w) is not dict
                    or w.get("unreturned_work_unknown", True)
                    or w.get("native_forward_attempts") != w.get("native_forward_returns")
                    for w in work
                )
            ),
            attempted_train_fit_nominal_affine_terms=None if rows is None else rows * terms,
        )
    return costs
