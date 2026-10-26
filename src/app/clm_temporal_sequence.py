"""Fixed twelve-update CLM temporal-rule use case with delayed TRAIN labels.

Consumes already-bound frozen features and an injected native learner. Retains
unscored readouts and actual inbox receipts, never scores, loads backends, writes
files, promotes a learner or renews spent work. Hooks are synchronous boundaries.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from typing import Any

from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_head import (
    CLMHeadPolicy,
    CLMTargets,
    FrozenChoiceBatch,
    FrozenEmbeddingBatch,
    FrozenEncoderIdentity,
    _exact,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.temporal_decision_sequence import CATALOG, CHECKPOINTS, PHASES
from src.core.temporal_catalog_support import catalog_training_examples
from src.core.clm_catalog import CATALOG_OBJECTIVE, CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_linear_catalog import (
    CONTROL_KIND,
    uniform_linear_controls,
    CLMLinearCatalogPolicy,
    OBJECTIVE as LINEAR_OBJECTIVE,
    linear_catalog_work,
)

_KEYS = (
    "raw_readouts",
    "frozen_A_probabilities",
    "rows",
    "stream",
    "learner_work",
    "sequence_complete",
)


def _tensor_traits(value, shape):
    # Backend-independent read-only tensor traits; no backend import or forward.
    if (
        str(getattr(value, "dtype", None)) != "torch.float32"
        or getattr(getattr(value, "device", None), "type", None) != "cpu"
        or str(getattr(value, "layout", None)) != "torch.strided"
        or tuple(getattr(value, "shape", ())) != shape
        or not value.isfinite().all().item()
    ):
        raise ValueError("temporal features require finite CPU float32 declared shapes")


def _feature_pin(batch):
    return tuple(
        (value, value._version, value.data_ptr()) for value in (batch.states, batch.actions)
    )


def _require_features(batch, pin):
    for value, original in zip((batch.states, batch.actions), pin, strict=True):
        if (
            value is not original[0]
            or value._version != original[1]
            or value.data_ptr() != original[2]
        ):
            raise ValueError("original frozen temporal feature owner or value changed")


def _admit(
    learner,
    train_batches,
    choice_batches,
    report,
    before_step,
    save,
    *,
    catalog_support="none",
    catalog_training=False,
    linear_catalog=False,
):
    if type(report) is not dict or any(key in report for key in _KEYS):
        raise ValueError("temporal stream requires a fresh report scope")
    if not callable(before_step) or not callable(save):
        raise ValueError("temporal boundaries must be callable")
    if any(
        not callable(getattr(learner, name, None)) for name in ("train_batch", "predict_choices")
    ):
        raise ValueError("temporal stream requires the native paired learner and choice port")
    identity, policy, work = learner.identity, learner.policy, learner.work
    _exact(identity, FrozenEncoderIdentity)
    _exact(policy, CLMLinearCatalogPolicy if linear_catalog else CLMHeadPolicy)
    if (
        policy.max_updates != 12
        or policy.max_rows < 4
        or learner.completed_updates != 0
        or learner.training_examples != 0
        or learner.generation != 0
        or type(work) is not dict
        or any(type(value) not in (int, bool) or value != 0 for value in work.values())
    ):
        raise ValueError("temporal stream requires a fresh twelve-update physical owner")
    required: tuple[str, ...] = (
        "train_attempts",
        "training_rows_attempted",
        "loss_forward_attempts",
        "loss_forward_returns",
        "prediction_attempts",
        "prediction_returns",
        "prediction_rows_attempted",
        "choice_prediction_attempts",
        "choice_prediction_returns",
        "choice_candidate_rows_attempted",
        "backward_attempts",
        "backward_returns",
        "optimizer_attempts",
        "optimizer_returns",
        "scheduler_attempts",
        "scheduler_returns",
        "commits",
        "snapshot_attempts",
        "snapshot_returns",
        "restore_attempts",
        "restore_returns",
        "unreturned_work_unknown",
    )
    if catalog_training:
        required += ("catalog_candidate_rows_attempted", "catalog_logit_values_attempted")
    if linear_catalog:
        required = tuple(linear_catalog_work())
    if set(work) != set(required):
        raise ValueError("native learner does not expose the exact physical work schema")
    if type(train_batches) is not tuple or len(train_batches) != 12:
        raise ValueError("expected twelve fixed paired TRAIN batches")
    phases = tuple(phase for _, phase in CHECKPOINTS)
    if type(choice_batches) is not dict or tuple(choice_batches) != phases:
        raise ValueError("choice batches must preserve the four checkpoint phases")
    catalog = {candidate.candidate_id: index for index, candidate in enumerate(CATALOG)}
    train_pins = []
    for index, pair in enumerate(train_batches):
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("each TRAIN entry must be an exact batch/target pair")
        batch, targets = pair
        expected = catalog_training_examples(
            PHASES[index // 4], catalog_support, exposure=index % 4
        )
        if catalog_training:
            codes = _admit_catalog_pair(
                batch, targets, expected, identity, choice_batches[PHASES[index // 4]]
            )
        else:
            _exact(batch, FrozenEmbeddingBatch)
            _exact(targets, CLMTargets)
            codes = tuple(catalog[example.target_id] for example in expected)
            if batch.identity != identity or targets.group_codes != codes:
                raise ValueError(
                    "TRAIN identity or catalog group codes differ from fixed phase/order"
                )
            _tensor_traits(batch.states, (len(expected), identity.embedding_dim))
            _tensor_traits(batch.actions, (len(expected), identity.embedding_dim))
        train_pins.append((batch, targets, deepcopy(codes), _feature_pin(batch)))
    choice_pins = []
    for phase, batch in choice_batches.items():
        _exact(batch, FrozenChoiceBatch)
        if batch.identity != identity or batch.role != "outer_selection":
            raise ValueError("temporal readouts require bound development choice features")
        _tensor_traits(batch.states, (4, identity.embedding_dim))
        _tensor_traits(batch.actions, (3, identity.embedding_dim))
        choice_pins.append((phase, batch, _feature_pin(batch)))
    return identity, policy, tuple(train_pins), tuple(choice_pins)


def _admit_catalog_pair(batch, targets, examples, identity, choice):
    _exact(batch, FrozenCatalogTrainingBatch)
    _exact(targets, CLMCatalogTargets)
    _exact(choice, FrozenChoiceBatch)
    ids = tuple(candidate.candidate_id for candidate in examples[0].request.allowed_candidates)
    indices = tuple(ids.index(example.target_id) for example in examples)
    if (
        batch.identity != identity
        or batch.candidate_ids != ids
        or targets.candidate_ids != ids
        or targets.candidate_indices != indices
    ):
        raise ValueError("catalog TRAIN identity, candidate order or observed targets differ")
    _tensor_traits(batch.states, (len(examples), identity.embedding_dim))
    _tensor_traits(batch.actions, (len(ids), identity.embedding_dim))
    if not batch.actions.equal(choice.actions):
        raise ValueError(
            "catalog TRAIN action features differ from the full active readout catalog"
        )
    return indices


def _admit_train_choices(train_choice_batches, train_batches, choice_batches, identity, report):
    if train_choice_batches is None:
        return ()
    if (
        type(train_choice_batches) is not dict
        or tuple(train_choice_batches) != PHASES
        or "train_raw_readouts" in report
    ):
        raise ValueError("TRAIN diagnostics require a fresh ordered PHASES choice dictionary")
    pins = []
    for index, (phase, batch) in enumerate(train_choice_batches.items()):
        _exact(batch, FrozenChoiceBatch)
        if batch.identity != identity or batch.role != "train":
            raise ValueError("TRAIN diagnostic choice identity or role differs")
        _tensor_traits(
            batch.states, (len(train_batches[index * 4][0].states), identity.embedding_dim)
        )
        _tensor_traits(batch.actions, (3, identity.embedding_dim))
        if not batch.states.equal(train_batches[index * 4][0].states) or not batch.actions.equal(
            choice_batches[phase].actions
        ):
            raise ValueError("TRAIN diagnostic features differ from phase TRAIN/full catalog")
        pins.append((phase, batch, _feature_pin(batch)))
    return tuple(pins)


def _probabilities(value, *, rows=4, candidates=3):
    if type(candidates) is not int or not 2 <= candidates <= 64:
        raise ValueError("probability candidate count must be an integer in2..64")
    _tensor_traits(value, (rows, candidates))
    rows = value.detach().tolist()
    if any(
        any(type(p) is not float or not 0 <= p <= 1 for p in row)
        or not math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        for row in rows
    ):
        raise ValueError("native choice rows must be complete probability distributions")
    return rows


def run_temporal_stream(
    learner,
    train_batches,
    choice_batches,
    report,
    *,
    before_step=lambda: None,
    save=lambda: None,
    train_choice_batches=None,
    catalog_support="none",
    on_checkpoint=None,
    catalog_training=False,
    allow_train_replay=False,
    native_observer=None,
    linear_catalog=False,
):
    """Mutate/return a bounded partial report; admission failures precede work."""
    if type(linear_catalog) is not bool or (
        linear_catalog
        and (
            catalog_training is not True
            or catalog_support != "none"
            or allow_train_replay
            or native_observer is not None
            or on_checkpoint is not None
        )
    ):
        raise ValueError(
            "linear catalog requires one original TRAIN stream without native extensions"
        )
    if type(allow_train_replay) is not bool or (
        allow_train_replay
        and (
            catalog_support != "none"
            or (catalog_training and on_checkpoint is None)
            or (not catalog_training and (native_observer is None or on_checkpoint is not None))
        )
    ):
        raise ValueError("TRAIN replay consent requires a catalog checkpoint or paired observer")
    if native_observer is not None and not callable(native_observer):
        raise ValueError("native_observer must be callable or None")
    if on_checkpoint is not None and not callable(on_checkpoint):
        raise ValueError("checkpoint operation must be callable or None")
    if type(catalog_training) is not bool:
        raise ValueError("catalog training selection must be an exact boolean")
    if type(catalog_support) is not str or catalog_support not in ("none", "repeat", "route"):
        raise ValueError("catalog_support must be exactly none, repeat or route")
    # Support changes packet supervision; reserved checkpoint probes keep the none recipe.
    if catalog_training and catalog_support != "none" and on_checkpoint is not None:
        raise ValueError("catalog support recipes cannot use checkpoint callbacks")
    if catalog_training:
        identity, policy, train_pins, choice_pins = _admit(
            learner,
            train_batches,
            choice_batches,
            report,
            before_step,
            save,
            catalog_support=catalog_support,
            catalog_training=True,
            linear_catalog=linear_catalog,
        )
    elif catalog_support == "none" and type(catalog_support) is str:
        identity, policy, train_pins, choice_pins = _admit(
            learner, train_batches, choice_batches, report, before_step, save
        )
    else:
        identity, policy, train_pins, choice_pins = _admit(
            learner,
            train_batches,
            choice_batches,
            report,
            before_step,
            save,
            catalog_support=catalog_support,
        )
    packet_rows = tuple(len(pin[2]) for pin in train_pins)
    if "catalog_support" in report and (
        type(report["catalog_support"]) is not str or report["catalog_support"] != catalog_support
    ):
        raise ValueError("report catalog support declaration differs")
    diagnostic_pins = _admit_train_choices(
        train_choice_batches, train_batches, choice_batches, identity, report
    )
    report_keys: tuple[str, ...] = _KEYS + ("train_raw_readouts",) if diagnostic_pins else _KEYS
    if diagnostic_pins:
        report["train_raw_readouts"] = []
    if linear_catalog:
        controls = uniform_linear_controls()
        if (
            report.get("control_kind", CONTROL_KIND) != CONTROL_KIND
            or report.get("control_probabilities", controls) != controls
        ):
            raise ValueError("linear stream requires its declared untrained uniform control")
        report["control_kind"], report["control_probabilities"] = CONTROL_KIND, controls
        report_keys += ("control_kind", "control_probabilities")
    if catalog_support != "none":
        report_keys += ("catalog_support",)
        report["catalog_support"] = catalog_support
    if catalog_training:
        objective = LINEAR_OBJECTIVE if linear_catalog else CATALOG_OBJECTIVE
        if "training_objective" in report and report["training_objective"] != objective:
            raise ValueError("catalog report objective declaration differs")
        report_keys += ("training_objective",)
        report["training_objective"] = objective
    if allow_train_replay:
        if "training_replay_consent" in report and report["training_replay_consent"] is not True:
            raise ValueError("original TRAIN replay consent changed")
        report_keys += ("training_replay_consent",)
        report["training_replay_consent"] = True
    elif "training_replay_consent" in report:
        raise ValueError("TRAIN replay consent was not selected at stream creation")
    permissions = ExperiencePermissions(training=True, replay=allow_train_replay)
    report.update(
        raw_readouts=[],
        frozen_A_probabilities=None,
        rows=[],
        stream=dict(completed_updates=0, training_pairs=0, receipts=[]),
        learner_work=learner.work,
        sequence_complete=False,
    )
    clock = LogicalClock()
    limits = ToyExecutionBudget(max_training_updates=12)
    budget = ToyBudgetSession(limits, lambda: 0.0)
    birth = (budget, limits, budget.clock, budget.started_at, clock)
    inbox = ExperienceInbox(
        learner, clock=clock, budget=budget, learner_version="clm-temporal", max_experiences=12
    )
    receipts: list[dict[str, Any]] = []
    probe_records = report.get("reserved_probes") if on_checkpoint is not None else None
    if probe_records is not None and (type(probe_records) is not list or probe_records):
        raise ValueError("checkpoint probes require a fresh bounded record list")
    probe_owners: tuple = ()
    probe_proof = "[]"
    tick = 0
    reads = 0
    train_reads = 0
    train_query_rows = 0
    proof = ""

    def require_probe_prefix(*, adding=False):
        if probe_records is None:
            return
        expected = len(probe_owners) + int(adding)
        if report.get("reserved_probes") is not probe_records or len(probe_records) != expected:
            raise ValueError("original completed checkpoint probe list changed")
        if any(probe_records[index] is not owner for index, owner in enumerate(probe_owners)):
            raise ValueError("original completed checkpoint probe was replaced")
        if (
            json.dumps(probe_records[: len(probe_owners)], sort_keys=True, allow_nan=False)
            != probe_proof
        ):
            raise ValueError("completed checkpoint probe evidence changed")

    def scalar_state():
        return json.dumps(
            {key: report[key] for key in report_keys if key != "learner_work"},
            sort_keys=True,
            allow_nan=False,
        )

    def refresh():
        nonlocal proof
        if proof and scalar_state() != proof:
            raise ValueError("recorded temporal prefix changed before native return capture")
        actual = [asdict(receipt) for receipt in inbox.applied_updates]
        if actual[: len(receipts)] != receipts or len(actual) > len(receipts) + 1:
            raise ValueError("actual temporal receipt prefix changed")
        for receipt in actual[len(receipts) :]:
            index = len(receipts)
            phase, exposure = PHASES[index // 4], index % 4
            sample_id = f"clm-{phase}-{exposure}"
            if (
                receipt["update_number"] != index + 1
                or receipt["observed_at"] != 2 * index
                or receipt["arrived_at"] != 2 * index + 1
                or receipt["applied_at"] != 2 * index + 1
                or receipt["sample_id"] != sample_id
                or receipt["episode_id"] != "clm-temporal"
                or receipt["event_id"] != "label-" + sample_id
                or receipt["actor_version"] != phase
                or receipt["learner_version"] != "clm-temporal"
                or receipt["diagnostic"]["definition"]
                != (
                    (LINEAR_OBJECTIVE if linear_catalog else CATALOG_OBJECTIVE)
                    if catalog_training
                    else "clm_paired_group_masked_symmetric_cross_entropy_v1"
                )
            ):
                raise ValueError("actual temporal receipt clocks or update differ")
            report["rows"].append(
                dict(
                    update=index + 1,
                    phase=PHASES[index // 4],
                    exposure=index % 4,
                    **(
                        {
                            "candidate_ids": list(train_pins[index][0].candidate_ids),
                            "candidate_indices": list(train_pins[index][2]),
                        }
                        if catalog_training
                        else {"group_codes": list(train_pins[index][2])}
                    ),
                    loss=receipt["diagnostic"]["value"],
                )
            )
            receipts.append(deepcopy(receipt))
        report["stream"] = dict(
            completed_updates=budget.updates_completed,
            training_pairs=sum(packet_rows[: len(receipts)]),
            receipts=deepcopy(actual),
        )
        report["learner_work"] = learner.work
        proof = scalar_state()

    def guard():
        require_probe_prefix()
        if allow_train_replay:
            _exact(permissions, ExperiencePermissions)
            if permissions != ExperiencePermissions(training=True, replay=True):
                raise ValueError("original arrived TRAIN replay consent changed")
            for item in inbox._experiences.values():
                _exact(item.permissions, ExperiencePermissions)
                if item.permissions != permissions:
                    raise ValueError("original arrived TRAIN replay consent changed")
        if (
            budget is not birth[0]
            or budget.budget is not birth[1]
            or budget.clock is not birth[2]
            or budget.started_at != birth[3]
            or clock is not birth[4]
            or clock.now() != tick
            or budget.budget != limits
            or budget.updates_completed != len(receipts)
        ):
            raise ValueError("temporal stream original clock or budget changed")
        if (
            learner.identity != identity
            or learner.policy != policy
            or learner.completed_updates != len(receipts)
            or learner.training_examples != sum(packet_rows[: len(receipts)])
            or learner.generation != len(receipts)
        ):
            raise ValueError("native temporal learner identity, allowance or counters changed")
        work = learner.work
        expected_work = dict.fromkeys(
            (
                "train_attempts",
                "commits",
                "loss_forward_attempts",
                "loss_forward_returns",
                "backward_attempts",
                "backward_returns",
                "optimizer_attempts",
                "optimizer_returns",
                "scheduler_attempts",
                "scheduler_returns",
            ),
            len(receipts),
        )
        expected_work.update(
            dict.fromkeys(
                (
                    "prediction_attempts",
                    "prediction_returns",
                    "choice_prediction_attempts",
                    "choice_prediction_returns",
                ),
                reads,
            )
        )
        expected_work.update(
            training_rows_attempted=sum(packet_rows[: len(receipts)]),
            choice_candidate_rows_attempted=3 * reads,
            prediction_rows_attempted=4 * (reads - train_reads) + train_query_rows,
            snapshot_attempts=0,
            snapshot_returns=0,
            restore_attempts=0,
            restore_returns=0,
            unreturned_work_unknown=False,
        )
        if catalog_training:
            expected_work.update(
                catalog_candidate_rows_attempted=3 * len(receipts),
                catalog_logit_values_attempted=3 * sum(packet_rows[: len(receipts)]),
            )
        if linear_catalog:
            expected_work = linear_catalog_work(
                len(receipts),
                sum(packet_rows[: len(receipts)]),
                reads,
                4 * (reads - train_reads) + train_query_rows,
            )
        if (
            type(work) is not dict
            or set(work) != set(expected_work)
            or any(
                type(work[key]) is not type(value) or work[key] != value
                for key, value in expected_work.items()
            )
        ):
            raise ValueError("native temporal physical work differs from paid stream prefix")
        for index, (pair, pin) in enumerate(zip(train_batches, train_pins, strict=True)):
            if pair[0] is not pin[0] or pair[1] is not pin[1]:
                raise ValueError("original TRAIN pair changed")
            if catalog_training:
                _exact(pair[0], FrozenCatalogTrainingBatch)
                _exact(pair[1], CLMCatalogTargets)
                examples = catalog_training_examples(
                    PHASES[index // 4], catalog_support, exposure=index % 4
                )
                ids = tuple(
                    candidate.candidate_id for candidate in examples[0].request.allowed_candidates
                )
                if (
                    pair[0].identity != identity
                    or pair[0].candidate_ids != ids
                    or pair[1].candidate_ids != ids
                    or pair[1].candidate_indices != pin[2]
                ):
                    raise ValueError("original catalog TRAIN metadata changed")
            else:
                _exact(pair[0], FrozenEmbeddingBatch)
                _exact(pair[1], CLMTargets)
                if pair[0].identity != identity or pair[1].group_codes != pin[2]:
                    raise ValueError("original TRAIN metadata changed")
            _require_features(pair[0], pin[3])
        for phase, batch, pin in choice_pins:
            if choice_batches.get(phase) is not batch:
                raise ValueError("original choice batch changed")
            _exact(batch, FrozenChoiceBatch)
            if batch.identity != identity or batch.role != "outer_selection":
                raise ValueError("original choice metadata changed")
            _require_features(batch, pin)
        if diagnostic_pins:
            if type(train_choice_batches) is not dict or tuple(train_choice_batches) != PHASES:
                raise ValueError("original TRAIN diagnostic phase dictionary changed")
            for phase, batch, pin in diagnostic_pins:
                if train_choice_batches[phase] is not batch:
                    raise ValueError("original TRAIN diagnostic batch changed")
                _exact(batch, FrozenChoiceBatch)
                if batch.identity != identity or batch.role != "train":
                    raise ValueError("original TRAIN diagnostic metadata changed")
                _require_features(batch, pin)
        if [asdict(receipt) for receipt in inbox.applied_updates] != receipts:
            raise ValueError("actual temporal receipts changed")
        if proof and scalar_state() != proof:
            raise ValueError("recorded temporal paid prefix changed")

    def boundary():
        guard()
        before_step()
        guard()

    def persist():
        nonlocal proof
        report["learner_work"] = learner.work
        proof = scalar_state()
        save()
        guard()

    def readout(phase, *, frozen=False, train=False):
        nonlocal reads, train_reads, train_query_rows, proof
        boundary()
        row = dict(phase=phase, completed_updates=len(receipts), probabilities=None)
        if train:
            report["train_raw_readouts"].append(row)
        elif not frozen:
            report["raw_readouts"].append(row)
        proof = scalar_state()
        selected = train_choice_batches[phase] if train else choice_batches[phase]
        value = learner.predict_choices(selected)
        reads += 1
        if train:
            train_reads += 1
            train_query_rows += packet_rows[PHASES.index(phase) * 4]
        probabilities = (
            _probabilities(value, rows=packet_rows[PHASES.index(phase) * 4])
            if train
            else _probabilities(value)
        )
        if scalar_state() != proof:
            raise ValueError("recorded temporal prefix changed during choice readout")
        if frozen:
            report["frozen_A_probabilities"] = probabilities
        else:
            row["probabilities"] = probabilities
        proof = scalar_state()
        guard()
        persist()

    def checkpoint(phase):
        nonlocal probe_owners, probe_proof
        if on_checkpoint is not None:
            boundary()
            on_checkpoint(phase, learner)
            if probe_records is not None:
                require_probe_prefix(adding=True)
                child = probe_records[-1]
                if (
                    type(child) is not dict
                    or child.get("complete") is not True
                    or child.get("phase") != phase
                    or child.get("source_update") != len(receipts)
                ):
                    raise ValueError(
                        "checkpoint probe did not return a complete phase-bound record"
                    )
                probe_owners = tuple(probe_records)
                probe_proof = json.dumps(probe_records, sort_keys=True, allow_nan=False)
            guard()
            persist()

    try:
        refresh()
        readout("baseline")
        controls = report.get("control_probabilities")
        if type(controls) is dict and "v1" in controls:
            baseline = report["raw_readouts"][0]["probabilities"]
            reference = controls["v1"]
            if (
                type(reference) is not list
                or len(reference) != 4
                or any(
                    type(row) is not list
                    or len(row) != 3
                    or any(
                        type(value) not in (int, float) or not math.isfinite(value) for value in row
                    )
                    for row in reference
                )
            ):
                raise ValueError("initial frozen control requires a finite4x3 matrix")
            error = max(
                abs(a - b)
                for row, expected in zip(baseline, reference, strict=True)
                for a, b in zip(row, expected, strict=True)
            )
            report["initial_control_max_abs_error"] = error
            if error > 2e-6:
                raise ValueError("initial learner differs from the unchanged released control")
            persist()
        checkpoint("baseline")
        for index, (batch, targets) in enumerate(train_batches):
            boundary()
            tick = 2 * index
            clock.advance_to(tick)
            phase, exposure = PHASES[index // 4], index % 4
            sample_id = f"clm-{phase}-{exposure}"
            source = Experience(
                sample_id,
                "clm-temporal",
                tick,
                phase,
                batch,
                "train",
                permissions,
            )
            inbox.record_experience(source)
            if inbox.drain(max_updates=1):
                raise ValueError("temporal TRAIN update occurred before label arrival")
            guard()
            tick += 1
            clock.advance_to(tick)
            inbox.record_label(
                LabelArrival(
                    "label-" + sample_id, sample_id, "clm-temporal", tick, phase, targets, "train"
                )
            )
            try:
                applied = (
                    inbox.drain(max_updates=1, native_observer=native_observer)
                    if native_observer is not None
                    else inbox.drain(max_updates=1)
                )
            finally:
                refresh()
            if len(applied) != 1:
                raise ValueError("expected one original temporal update receipt")
            guard()
            persist()
            if len(receipts) in (4, 8, 12):
                readout(phase)
                if len(receipts) == 4:
                    readout("after_B", frozen=True)
                if diagnostic_pins:
                    readout(phase, train=True)
                checkpoint(phase)
        guard()
        if (
            len(receipts) != 12
            or reads != (8 if diagnostic_pins else 5)
            or train_reads != (3 if diagnostic_pins else 0)
        ):
            raise ValueError("temporal sequence did not finish its fixed work")
        report["sequence_complete"] = True
        persist()
    except Exception as error:
        report["sequence_complete"] = False
        report["error"] = type(error).__name__ + ": " + str(error)[:512]
        raise
    finally:
        report["learner_work"] = learner.work
        report["stream"] = dict(
            completed_updates=budget.updates_completed,
            training_pairs=sum(packet_rows[: len(inbox.applied_updates)]),
            receipts=[asdict(receipt) for receipt in inbox.applied_updates],
        )
    return report
