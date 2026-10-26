"""Complete borrowed coupled graphs, original admission, and detached ownership.

Native objects are assembled from explicit records; no native constructor or
operational method is needed to test the capture representation boundary.
"""

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import src.adapters.coupled_composite_capture as module
from src.adapters.coupled_elapsed_retention import make_coupled_elapsed_lifecycle
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.coupled_managed_serving import _evaluator
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_experience import ManagedExperienceOwner
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import (
    PromotableActor,
    ServingPromotionController,
    _Bundle,
    _Pending,
    _Slot,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionProgress
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_learner_state import CoupledUpdatePolicy
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledWakeState,
)
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.managed_composite_state import CompositeCaptureLimits, SourceRecord
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.promotion_guard import (
    PromotionEvidence,
    PromotionGuardReport,
    PromotionPolicy,
    decide_promotion,
)
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import (
    CachedPrediction,
    PreparedPromotion,
    PromotionReceipt,
    ServingConfiguration,
)
from src.shared.process_memory import ProcessRssSampler


LIMITS = CompositeCaptureLimits(LifecycleCaptureLimits(64, 128), 8192, 64, 65536, 32)


def values(record):
    assert type(record) is SourceRecord
    return dict(record.fields)


def native_snapshot(learner):
    model = learner._model
    wake = None
    if type(model) is CoupledCircadianPredictiveCodingNetwork:
        c = model._config
        wake = CoupledWakeState(
            c.chemical_decay,
            c.chemical_buildup_rate,
            c.plasticity_sensitivity,
            c.min_plasticity,
            model._chemical_state,
            model._plasticity_state,
            model._wake_updates,
        )
    return deepcopy(CoupledSnapshot(1, model._parameters, model._precisions, wake))


def synthetic_learner(family, guarded):
    kind = (
        CoupledPredictiveCodingNetwork
        if family == "pc"
        else CoupledCircadianPredictiveCodingNetwork
    )
    model: Any = object.__new__(kind)
    model._input_dim, model._hidden_dims, model._precisions = 2, (3, 2), (0.7, 1.3)
    model._parameters = CoupledParameters(
        (
            np.array([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]]),
            np.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6]]),
        ),
        (np.array([[-0.0, 0.2, -0.1]]), np.array([[0.3, -0.0]])),
        np.array([[0.4], [-0.2]]),
        np.array([[0.5]]),
    )
    if family != "pc":
        model._config = CoupledCircadianConfig(min_plasticity=1.0 if family == "neutral" else 0.2)
        model._chemical_state = np.array([0.0, 0.0])
        model._plasticity_state = np.array([1.0, 1.0])
        model._wake_updates = 7
    learner = object.__new__(CoupledLearner)
    learner._model = model
    learner._learning_rate, learner._inference_steps, learner._inference_learning_rate = (
        0.025,
        3,
        0.07,
    )
    learner._inference_policy = CoupledInferencePolicy(1, 1e6, None) if guarded else None
    return learner


class Reading:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


@pytest.fixture
def graph_factory(monkeypatch):
    # Why: exact native types exercise the projection guard while fixture forks
    # duplicate only explicit Python records, never native restoration methods.
    monkeypatch.setattr(CoupledLearner, "fork", lambda self: deepcopy(self))
    monkeypatch.setattr(CoupledLearner, "snapshot_state", native_snapshot)
    graphs: list[dict[str, Any]] = []

    def build(family="pc", rich=False, guarded=False):
        learner = synthetic_learner(family, guarded)
        wall, rss = Reading(8.0), Reading(100)
        sampler = ProcessRssSampler(read_rss_bytes=rss)
        sampler.sample()
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0, max_wall_seconds=50, max_process_rss_bytes=1000
            ),
            clock=wall,
            progress=ToyExecutionProgress(),
        )
        budget.attach_memory(sampler)
        actor = (
            None
            if not rich
            else PromotableActor(
                learner,
                version="actor",
                configuration=ServingConfiguration(2, 2),
                feature_digest=lambda *a: "unused",
                metadata={},
            )
        )
        runtime = ActorShadowRuntime(
            learner,
            actor_version="actor",
            candidate_version="candidate",
            clock=LogicalClock(0),
            budget=budget,
            max_experiences=4,
            actor=actor,
        )
        shared = ResourceSharedRuntime(
            runtime, ServingPriorityGate(SharingLimits(2, 0, 1), resource_available=lambda: True)
        )
        owner = ManagedExperienceOwner(shared, limits=LifecycleLimits(8, 8))
        life = make_coupled_elapsed_lifecycle(
            owner,
            policy=DataRetentionPolicy(
                4096,
                20,
                PayloadOwnershipLimits(8, 12),
                owned_payload_copies=PayloadCopyLimits(65536),
                max_retention_seconds=1000.0,
            ),
        )
        owner.declare(
            LifecycleDeclaration(
                ("episode", "one"),
                DataProvenance("local", "person", True, False),
                DataConsent(True, True),
                "replay",
            )
        )
        owner.record_experience(
            Experience(
                "one",
                "episode",
                0,
                "actor",
                np.array([[0.2, -0.3]]),
                "train",
                ExperiencePermissions(True, True),
            )
        )
        owner.record_label(
            LabelArrival("label", "one", "episode", 0, "actor", np.array([[0.5]]), "train")
        )
        shared._sharing.pause()
        checkpoint = CandidateCheckpointController(
            shared,
            build_learner=lambda *a: pytest.fail("builder"),
            state_digest=lambda *a: "a" * 64,
            policy_digest=lambda *a: "b" * 64,
        )
        token = checkpoint.capture() if rich else None
        promotion = None
        if rich:
            assert actor is not None
            source = runtime._inbox._experiences[("episode", "one")].features
            target = runtime._inbox._labels[("episode", "one")].targets
            current = actor._slot.bundle
            current = _Bundle(
                current.learner,
                current.version,
                current.configuration,
                {"c" * 64: CachedPrediction(2, target)},
                {"shared": source, "equal": source.copy()},
                0,
            )
            actor._slot = _Slot(current, 3, current)
            builder = lambda *a: pytest.fail("promotion builder")
            evaluator: Any = PromotionGuardEvaluator(
                build_learner=builder,
                utility=lambda *a: 0.0,
                actions=lambda *a: (),
                state_valid=lambda *a: True,
                prediction_valid=lambda *a: True,
                resource_bytes=lambda *a: 0,
                state_digest=lambda *a: "a" * 64,
                clock=wall,
            )
            policy = PromotionPolicy("utility", "bytes", 0.0, 0.0, 0.0, 1.0, 4096, ("safe",))
            promotion = ServingPromotionController(
                actor,
                policy=policy,
                evaluator=evaluator,
                build_learner=builder,
                state_digest=lambda *a: "a" * 64,
            )
            evidence = PromotionEvidence(0.0, 0.0, 0.0, 0.0, 0.0, 0, True, True)
            report = PromotionGuardReport(
                policy,
                "actor",
                "candidate",
                (0, 0, 0),
                "a" * 64,
                "b" * 64,
                "new",
                "old",
                (("new", "one"),),
                (("old", "one"),),
                (0, 0),
                (),
                0,
                evidence,
                decide_promotion(policy, evidence),
            )
            ticket = PreparedPromotion(0, report)
            promotion._pending[ticket] = _Pending(runtime, report, current, 0, builder)
            receipt = PromotionReceipt(0, "previous", "actor")
            promotion._latest = (receipt, replace(receipt))
        result = dict(
            owner=owner,
            life=life,
            runtime=runtime,
            shared=shared,
            checkpoint=checkpoint,
            token=token,
            promotion=promotion,
            wall=wall,
            rss=rss,
            sampler=sampler,
        )
        graphs.append(result)
        return result

    yield build
    for graph in graphs:
        assert graph["runtime"]._budget.updates_completed == 0
        assert graph["sampler"]._thread is None


def forbid_operations(monkeypatch):
    for kind in (
        CoupledLearner,
        CoupledPredictiveCodingNetwork,
        CoupledCircadianPredictiveCodingNetwork,
    ):
        for name in (
            "snapshot_state",
            "restore_state",
            "predict",
            "predict_proba",
            "train_batch",
            "train_epoch",
            "fork",
            "checkpoint_policy",
        ):
            if hasattr(kind, name):
                monkeypatch.setattr(
                    kind, name, lambda *a, **k: pytest.fail("native method during capture")
                )


def capture(graph, limits=LIMITS):
    return module.capture_coupled_managed_composite(graph["owner"], limits=limits)


def candidate(result):
    return next(
        values(row)
        for _, row in result.state.holders
        if row.kind == "src.app.actor_shadow.ActorShadowRuntime"
    )


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
@pytest.mark.parametrize("rich", (False, True))
def test_should_detach_complete_family_graph_without_native_methods(
    graph_factory, monkeypatch, family, rich
):
    graph = graph_factory(family, rich, guarded=rich)
    forbid_operations(monkeypatch)
    before = graph["life"]._copy_budget._charged
    result = capture(graph)
    assert result.state.retained_payload_bytes > 0
    if not rich:
        # Native arrays belong to the independent graph bound; this ledger
        # admits only sixteen feature bytes and eight label bytes.
        assert result.state.retained_payload_bytes == 24
    assert graph["life"]._copy_budget._charged == before + result.state.retained_payload_bytes
    learner = values(candidate(result)["_candidate"])
    assert set(learner) == {
        "_model",
        "_learning_rate",
        "_inference_steps",
        "_inference_learning_rate",
        "_inference_policy",
    }
    assert learner["_learning_rate"] == 0.025 and learner["_inference_steps"] == 3
    assert learner["_inference_learning_rate"] == 0.07
    assert learner["_inference_policy"] == graph["runtime"]._candidate._inference_policy
    native = values(learner["_model"])
    original = graph["runtime"]._candidate._model
    assert set(native) == set(vars(original))
    assert native["_input_dim"] == 2 and native["_hidden_dims"] == (3, 2)
    assert native["_precisions"] == (0.7, 1.3)
    assert not np.shares_memory(native["_parameters"].output_bias, original._parameters.output_bias)
    np.testing.assert_array_equal(
        native["_parameters"].output_bias, original._parameters.output_bias
    )
    if family != "pc":
        assert native["_config"] == original._config and native["_wake_updates"] == 7
        assert not np.shares_memory(native["_chemical_state"], original._chemical_state)
    assert len({reference.path for reference in result.authority}) == len(result.authority)
    assert any(reference.value is original for reference in result.authority)
    assert values(result.state.sharing)["_paused"] is True
    assert values(result.state.budget)["started_at"] == 8.0
    if rich:
        actor = next(
            values(row)
            for _, row in result.state.holders
            if row.kind == "src.app.serving_promotion.PromotableActor"
        )
        slot = values(actor["_slot"])
        assert slot["bundle"] is slot["previous"]
        inbox = values(candidate(result)["_inbox"])
        source = inbox["_experiences"][("episode", "one")].features
        bundle = values(slot["bundle"])
        assert bundle["metadata"]["shared"] is source
        assert bundle["metadata"]["equal"] is not source
        checkpoint = next(
            values(row)
            for _, row in result.state.holders
            if row.kind.endswith(".CandidateCheckpointController")
        )
        view = values(checkpoint["_pending"][0][1])["view"]
        assert type(view.state) is CoupledSnapshot and view.state.parameters.hidden_weights[
            0
        ].shape == (2, 3)


@pytest.mark.parametrize(
    "fault",
    (
        "learner_missing",
        "learner_extra",
        "model_missing",
        "model_extra",
        "dimension",
        "width",
        "precision",
        "parameter_fields",
        "wake_count",
        "gates",
        "chemistry",
        "guard_cap",
    ),
)
def test_should_refuse_malformed_complete_native_before_copy_or_charge(
    graph_factory, monkeypatch, fault
):
    graph = graph_factory("active")
    learner = graph["runtime"]._candidate
    model = learner._model
    if fault == "learner_missing":
        monkeypatch.delattr(learner, "_learning_rate")
    elif fault == "learner_extra":
        monkeypatch.setattr(learner, "foreign", None, raising=False)
    elif fault == "model_missing":
        monkeypatch.delattr(model, "_precisions")
    elif fault == "model_extra":
        monkeypatch.setattr(model, "foreign", None, raising=False)
    elif fault == "dimension":
        monkeypatch.setattr(model, "_input_dim", 3)
    elif fault == "width":
        monkeypatch.setattr(model, "_hidden_dims", (3, 3))
    elif fault == "precision":
        monkeypatch.setattr(model, "_precisions", (True, 1.0))
    elif fault == "parameter_fields":
        monkeypatch.setattr(model, "_parameters", object())
    elif fault == "wake_count":
        monkeypatch.setattr(model, "_wake_updates", True)
    elif fault == "gates":
        monkeypatch.setattr(model, "_plasticity_state", np.array([0.5, 1.0]))
    elif fault == "chemistry":
        monkeypatch.setattr(model, "_chemical_state", np.array([-1.0, 0.0]))
    else:
        monkeypatch.setattr(learner, "_inference_policy", CoupledInferencePolicy(4, None, None))
    before = graph["life"]._copy_budget._charged
    forbid_operations(monkeypatch)
    with patch.object(
        module, "copy_coupled_composite", side_effect=AssertionError("premature copy")
    ):
        with pytest.raises(ValueError):
            capture(graph)
    assert graph["life"]._copy_budget._charged == before


@pytest.mark.parametrize(
    "fault",
    (
        "opaque",
        "cycle",
        "overlap",
        "object_array",
        "noncontiguous",
        "nonfinite",
        "byte_bound",
        "node_bound",
        "depth_bound",
        "dimension_bound",
    ),
)
def test_should_bound_full_graph_before_admission_or_opaque_hooks(
    graph_factory, monkeypatch, fault
):
    graph = graph_factory("pc", True)
    metadata = graph["runtime"].actor._slot.bundle.metadata
    limits = LIMITS
    if fault == "opaque":

        class Poison:
            def __deepcopy__(self, memo):
                pytest.fail("opaque copy hook")

        metadata["bad"] = Poison()
    elif fault == "cycle":
        metadata["bad"] = metadata
    elif fault == "overlap":
        metadata["bad"] = metadata["shared"].view()
    elif fault == "object_array":
        metadata["bad"] = np.empty((1,), dtype=object)
    elif fault == "noncontiguous":
        metadata["bad"] = np.ones((2, 6))[:, ::2]
    elif fault == "nonfinite":
        metadata["bad"] = np.array([[np.nan]])
    else:
        overrides: dict[str, Any] = {
            "byte_bound": {"max_array_bytes": 1},
            "node_bound": {"max_nodes": 1},
            "depth_bound": {"max_depth": 1},
            "dimension_bound": {"max_dimension": 1},
        }[fault]
        limits = replace(LIMITS, **overrides)
    before = graph["life"]._copy_budget._charged
    finite_guard = (
        patch.object(
            np, "isfinite", side_effect=AssertionError("scan before native byte admission")
        )
        if fault == "byte_bound"
        else nullcontext()
    )
    with (
        finite_guard,
        patch.object(
            module, "copy_coupled_composite", side_effect=AssertionError("premature copy")
        ),
    ):
        with pytest.raises(ValueError):
            capture(graph, limits)
    assert graph["life"]._copy_budget._charged == before


@pytest.mark.parametrize(
    "fault",
    (
        "not_paused",
        "serving",
        "training",
        "checkpointing",
        "expired",
        "wall",
        "rss",
        "opted_out",
        "revoked",
        "exhausted",
    ),
)
def test_should_refuse_original_admission_before_projection_or_copy(
    graph_factory, monkeypatch, fault
):
    graph = graph_factory()
    life, sharing = graph["life"], graph["shared"]._sharing
    if fault == "not_paused":
        sharing._paused = False
    elif fault in ("serving", "training", "checkpointing"):
        setattr(sharing, "_" + fault, 1 if fault == "serving" else True)
    elif fault == "expired":
        graph["wall"].value = 1008.0
    elif fault == "wall":
        graph["wall"].value = 58.0
    elif fault == "rss":
        graph["rss"].value = 1001
    elif fault == "opted_out":
        graph["owner"]._opted_out.add("person")
    elif fault == "revoked":
        graph["owner"]._revoked_keys.add(("episode", "one"))
    else:
        life._copy_budget.reserve(65536 - life._copy_budget._charged)
    before = life._copy_budget._charged
    with patch.object(
        module, "copy_coupled_composite", side_effect=AssertionError("premature copy")
    ):
        with pytest.raises((ValueError, RuntimeError)):
            capture(graph)
    assert life._copy_budget._charged == before


@pytest.mark.parametrize(
    "fault",
    (
        "pending_owner",
        "pending_budget",
        "pending_integrity",
        "pending_precision",
        "foreign_clock",
        "foreign_guard",
    ),
)
def test_should_validate_pending_fullstate_and_original_bindings(graph_factory, monkeypatch, fault):
    graph = graph_factory("active", True)
    pending = graph["checkpoint"]._pending[graph["token"]]
    if fault == "pending_owner":
        monkeypatch.setattr(pending, "owner", object())
    elif fault == "pending_budget":
        monkeypatch.setattr(pending, "budget", object())
    elif fault == "pending_integrity":
        monkeypatch.setattr(pending, "integrity", "0" * 64)
    elif fault == "pending_precision":
        monkeypatch.setattr(
            pending,
            "view",
            replace(pending.view, state=replace(pending.view.state, precisions=(True, 1.0))),
        )
    elif fault == "foreign_clock":
        monkeypatch.setattr(graph["runtime"]._inbox, "_clock", LogicalClock(0))
    else:
        monkeypatch.setattr(graph["runtime"]._inbox, "_training_guard", lambda *a: True)
    before = graph["life"]._copy_budget._charged
    with patch.object(
        module, "copy_coupled_composite", side_effect=AssertionError("premature copy")
    ):
        with pytest.raises(ValueError):
            capture(graph)
    assert graph["life"]._copy_budget._charged == before


def test_should_preserve_cross_native_raw_alias_and_fortran_readonly_storage(
    graph_factory, monkeypatch
):
    graph = graph_factory("pc")
    model = graph["runtime"]._candidate._model
    target = graph["runtime"]._inbox._labels[("episode", "one")].targets
    first = np.asfortranarray(model._parameters.hidden_weights[0])
    first.flags.writeable = False
    model._parameters = replace(
        model._parameters,
        hidden_weights=(first, model._parameters.hidden_weights[1]),
        output_bias=target,
    )
    forbid_operations(monkeypatch)
    result = capture(graph)
    data = candidate(result)
    parameters = values(values(data["_candidate"])["_model"])["_parameters"]
    inbox = values(data["_inbox"])
    assert parameters.output_bias is inbox["_labels"][("episode", "one")].targets
    copied = parameters.hidden_weights[0]
    assert (
        copied.flags.f_contiguous and not copied.flags.c_contiguous and not copied.flags.writeable
    )
    assert not np.shares_memory(copied, first)
    assert np.signbit(parameters.hidden_biases[0][0, 0])


@pytest.mark.parametrize(
    "fault", ("copy_error", "late_consent", "late_wall", "late_rss", "late_binding")
)
def test_should_keep_consumed_copy_charge_and_return_no_partial_capture(
    graph_factory, monkeypatch, fault
):
    graph = graph_factory("pc", True)
    before = graph["life"]._copy_budget._charged
    original = module.copy_coupled_composite
    calls = []

    def copying(state, limits, memo):
        calls.append(memo)
        if fault == "copy_error":
            raise RuntimeError("injected copy failure")
        detached = original(state, limits, memo)
        if fault == "late_consent":
            graph["owner"]._revoked_keys.add(("episode", "one"))
        elif fault == "late_wall":
            graph["wall"].value = 58.0
        elif fault == "late_rss":
            graph["rss"].value = 1001
        else:
            graph["checkpoint"]._pending[graph["token"]].build = object()
        return detached

    monkeypatch.setattr(module, "copy_coupled_composite", copying)
    with pytest.raises((ValueError, RuntimeError)):
        capture(graph)
    assert len(calls) == 1 and graph["life"]._copy_budget._charged > before
    assert graph["runtime"]._budget.updates_completed == 0


def test_should_refresh_only_original_observations_with_the_same_detached_arrays(
    graph_factory, monkeypatch
):
    graph = graph_factory()
    original = module.copy_coupled_composite
    copies = []

    def copying(state, limits, memo):
        result = original(state, limits, memo)
        array = values(candidate(SimpleNamespace(state=result))["_inbox"])["_labels"][
            ("episode", "one")
        ].targets
        copies.append((memo, array))
        graph["wall"].value, graph["rss"].value = 9.0, 200
        return result

    monkeypatch.setattr(module, "copy_coupled_composite", copying)
    result = capture(graph)
    assert len(copies) == 2 and copies[0][0] is copies[1][0] and copies[0][1] is copies[1][1]
    budget = values(result.state.budget)
    assert budget["started_at"] == 8.0 and budget["last_clock"] == 9.0
    assert budget["process_rss_segment"].peak_bytes == 200
    assert budget["process_rss_segment"] is values(budget["progress"])["process_rss_segment"]


@pytest.mark.parametrize("limits", (None, True, {}))
def test_should_refuse_non_native_limits_before_consulting_the_owner(limits):
    class PoisonOwner:
        def __getattribute__(self, name):
            pytest.fail("owner consulted before independent limits validation")

    with pytest.raises(ValueError):
        module.capture_coupled_managed_composite(PoisonOwner(), limits=limits)


def test_should_capture_actual_evaluator_helper_without_invoking_callbacks_or_builder(
    graph_factory, monkeypatch
):
    graph = graph_factory("active", True, guarded=True)
    promotion = graph["promotion"]
    callback_calls: list[object] = []

    def forbidden_callback(*args, **kwargs):
        callback_calls.append(args)
        pytest.fail("capture invoked evaluator callback")

    callbacks: Any = SimpleNamespace(state_digest=forbidden_callback)
    counts = dict(
        feature_digest=0, state_valid=0, prediction_valid=0, utility=0, actions=0, resource=0
    )
    evaluator = _evaluator(
        promotion._build,
        callbacks,
        CoupledUpdatePolicy(0.025, 3, 0.07, CoupledInferencePolicy(1, 1e6, None)),
        graph["wall"],
        counts,
    )
    monkeypatch.setattr(promotion, "_evaluator", evaluator)
    forbid_operations(monkeypatch)
    result = capture(graph)
    retained = next(
        values(record)
        for _, record in result.state.holders
        if record.kind == "src.app.serving_promotion.ServingPromotionController"
    )
    projected = values(retained["_evaluator"])
    references = {reference.path: reference.value for reference in result.authority}
    assert references[projected["_build"].path] is promotion._build
    assert references[projected["_digest"].path] is evaluator._digest
    assert not callback_calls and all(value == 0 for value in counts.values())
