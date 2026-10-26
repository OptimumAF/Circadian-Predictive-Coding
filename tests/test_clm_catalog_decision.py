"""Current trained catalog state binds once into the existing frozen typed predictor."""

from copy import copy, deepcopy
from dataclasses import replace

import pytest

pytest.importorskip("torch")

import src.adapters.clm_catalog_decision as binding_adapter
import src.adapters.clm_catalog_learner as catalog
import src.adapters.clm_head_learner as native
from src.adapters.clm_catalog_decision import CLMCatalogPredictorBinding
from src.adapters.clm_catalog_probe import CLMCatalogProbe
from src.adapters.clm_decision import CLMDecisionPredictor
from src.adapters.clm_frozen_encoder import FrozenCLMEncoder
from src.core.clm_catalog import CATALOG_OBJECTIVE, CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_head import CLMHeadPolicy, FrozenChoiceBatch
from src.core.typed_decision import DecisionCandidate, DecisionRequest
from test_clm_catalog_learning import parts
from test_clm_frozen_encoder import FakeTokenizer, backbone, encoder

IDS = ("alpha", "beta", "gamma")
MAPPING = {"query\n\nChoose.": [1], "first": [2], "second": [3], "third": [4]}


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def request(*, reverse=False):
    candidates = tuple(
        DecisionCandidate(name, text)
        for name, text in zip(IDS, ("first", "second", "third"), strict=True)
    )
    return DecisionRequest("learned-choice", "query", candidates[::-1] if reverse else candidates)


def trained_source(torch, *, forwards=4, retain_zero=False):
    frozen, model, tokenizer = encoder(torch, mapping=MAPPING, forwards=forwards)
    # These are actual outputs of the same literal fake encoder used by the predictor.
    values = frozen.embed_texts(tuple(MAPPING), ("tail", "head", "head", "head"))
    templates = parts(torch)
    source = catalog.CLMCatalogLearner(
        *templates,
        identity=frozen.identity,
        policy=CLMHeadPolicy(projection_dim=2, max_updates=3, max_rows=4),
    )
    training = FrozenCatalogTrainingBatch(
        frozen.identity, values[[0, 0]].clone(), values[1:].clone(), IDS
    )
    targets = CLMCatalogTargets(IDS, (0, 0))
    zero = catalog.CLMCatalogSnapshot(**vars(source._capture())) if retain_zero else None
    source.train_batch(training, targets)
    return dict(
        source=source,
        encoder=frozen,
        model=model,
        tokenizer=tokenizer,
        templates=templates,
        training=training,
        targets=targets,
        values=values,
        zero=zero,
    )


def binding(case):
    return CLMCatalogPredictorBinding(
        case["source"], case["encoder"], instructions="Choose.", head_revision="learned-fixture"
    )


def tensor_elements(torch, value):
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, dict):
        return sum(tensor_elements(torch, item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_elements(torch, item) for item in value)
    return 0


@pytest.fixture(scope="module")
def shared_source(torch):
    return trained_source(torch, retain_zero=True)


@pytest.fixture(scope="module")
def successful_binding(torch):
    case = trained_source(torch)
    source, frozen = case["source"], case["encoder"]
    source_state = source._capture()
    source_work, encoder_work = source.work, frozen.work
    owners = tuple(source._owners)
    caller_rng = torch.get_rng_state().clone()
    factory = binding(case)
    assert source.work == source_work and frozen.work == encoder_work
    assert all(value == 0 for value in factory.work.values()) and factory.receipt is None
    with pytest.MonkeyPatch.context() as patch:
        snapshot = source.snapshot_state
        returned = []

        def observed_snapshot():
            saved = snapshot()
            returned.append(saved)
            return saved

        patch.setattr(source, "snapshot_state", observed_snapshot)
        predictor = factory.bind()
    assert len(returned) == 1
    expected_work = dict(source_work)
    expected_work["snapshot_attempts"] += 1
    expected_work["snapshot_returns"] += 1
    assert source.work == expected_work and frozen.work == encoder_work
    assert native._equal(vars(source_state), vars(source._capture()))
    assert all(left is right for left, right in zip(owners, source._owners, strict=True))
    snapshot_elements = tensor_elements(torch, vars(returned[0]))
    expected = source.predict_choices(
        FrozenChoiceBatch(source.identity, case["values"][:1], case["values"][1:])
    )[0]
    first, reordered = predictor(request()), predictor(request(reverse=True))
    actor_weights = (
        deepcopy(predictor._state_head.state_dict()),
        deepcopy(predictor._action_head.state_dict()),
    )
    stale = binding(case)
    source.train_batch(case["training"], case["targets"])
    returned[0].state_head["weight"].zero_()
    returned[0].logit_scale.add_(2)
    after = predictor(request())
    return dict(
        case=case,
        factory=factory,
        predictor=predictor,
        expected=expected,
        first=first,
        reordered=reordered,
        after=after,
        actor_weights=actor_weights,
        source_state=source_state,
        source_work=source_work,
        encoder_work=encoder_work,
        source_owners=owners,
        rng=caller_rng,
        snapshot=returned[0],
        snapshot_elements=snapshot_elements,
        stale=stale,
    )


def test_should_return_the_existing_typed_predictor_with_trained_probabilities_and_candidate_order(
    successful_binding,
):
    case = successful_binding
    predictor = case["predictor"]
    assert type(predictor) is CLMDecisionPredictor
    assert [key for key, _ in case["first"].probabilities] == list(IDS)
    assert [key for key, _ in case["reordered"].probabilities] == list(reversed(IDS))
    assert dict(case["first"].probabilities) == pytest.approx(
        dict(zip(IDS, case["expected"].tolist(), strict=True)), abs=2e-7
    )
    assert dict(case["reordered"].probabilities) == pytest.approx(
        dict(case["first"].probabilities), abs=2e-7
    )
    assert case["first"].selected_id == case["reordered"].selected_id
    assert case["first"].reported_confidence == pytest.approx(
        case["reordered"].reported_confidence, abs=2e-7
    )


def test_should_freeze_independent_actor_heads_and_ignore_later_learning_or_snapshot_edits(
    torch, successful_binding
):
    saved = successful_binding
    source, actor = saved["case"]["source"], saved["predictor"]
    assert source.completed_updates == 2 and saved["after"] == saved["first"]
    for actual, expected in zip(
        (actor._state_head, actor._action_head), saved["actor_weights"], strict=True
    ):
        assert native._equal(actual.state_dict(), expected)
        assert not actual.training and all(
            not parameter.requires_grad for parameter in actual.parameters()
        )
    for head, original in zip(
        (actor._state_head, actor._action_head), source._bundle[:2], strict=True
    ):
        assert all(
            a is not b and a.data_ptr() != b.data_ptr()
            for a, b in zip(head.parameters(), original.parameters(), strict=True)
        )
    assert source._bundle[3].state and all(
        parameter.grad is not None for head in source._bundle[:2] for parameter in head.parameters()
    )
    assert torch.equal(saved["rng"], torch.get_rng_state())
    for name in ("train_batch", "snapshot_state", "restore_state", "fork", "export", "install"):
        assert not hasattr(actor, name)


def test_should_charge_one_snapshot_three_guards_and_two_head_copies_in_defensive_receipt(
    successful_binding,
):
    saved = successful_binding
    factory = saved["factory"]
    work = factory.work
    assert work["binding_attempts"] == work["binding_returns"] == 1
    assert work["snapshot_attempts"] == work["snapshot_returns"] == 1
    assert work["guard_capture_attempts"] == work["guard_capture_returns"] == 3
    assert work["head_copy_attempts"] == work["head_copy_returns"] == 2
    assert (
        work["head_parameter_elements_attempted"] == work["head_parameter_elements_returned"] == 8
    )
    assert work["scale_copy_elements_attempted"] == work["scale_copy_elements_returned"] == 2
    assert work["snapshot_tensor_elements_copied"] == 4 * saved["snapshot_elements"]
    assert work["unreturned_work_unknown"] is False
    receipt = factory.receipt
    assert receipt["objective"] == CATALOG_OBJECTIVE
    assert receipt["source_completed_updates"] == 1 and receipt["source_training_examples"] == 2
    assert receipt["source_generation"] == 1 and receipt["work"] == work
    assert receipt["identity"] == vars(saved["case"]["source"].identity)
    assert receipt["policy"] == vars(saved["case"]["source"].policy)
    assert receipt["instructions"] == "Choose." and receipt["head_revision"] == "learned-fixture"
    receipt["work"].clear()
    detached = factory.work
    detached.clear()
    assert factory.work == work and factory.receipt["work"] == work
    with pytest.raises(ValueError):
        factory.bind()
    assert factory.work == work


def test_should_refuse_factory_stale_after_real_source_update_before_paid_snapshot(
    successful_binding,
):
    saved = successful_binding
    factory, source = saved["stale"], saved["case"]["source"]
    before, encoded = source.work, saved["case"]["encoder"].work
    with pytest.raises(ValueError):
        factory.bind()
    work = factory.work
    assert work["binding_attempts"] == 1 and work["binding_returns"] == 0
    assert work["snapshot_attempts"] == work["head_copy_attempts"] == 0
    assert work["unreturned_work_unknown"] is False and factory.receipt is None
    assert source.work == before and saved["case"]["encoder"].work == encoded
    with pytest.raises(ValueError):
        factory.bind()
    assert factory.work == work


def test_should_preserve_existing_encoder_budget_and_failstop_actor_behavior(successful_binding):
    saved = successful_binding
    actor, frozen = saved["predictor"], saved["case"]["encoder"]
    before = frozen.work
    with pytest.raises(ValueError, match="allowance exhausted"):
        actor(request())
    paid = actor.work
    assert frozen.work == before and paid["request_attempts"] == 4 and paid["request_returns"] == 3
    assert paid["encoder_attempts"] == 4 and paid["encoder_returns"] == 3
    assert paid["unreturned_work_unknown"] is True
    with pytest.raises(ValueError):
        actor(request())
    assert actor.work == paid


@pytest.fixture(scope="module")
def other_sources(shared_source):
    source = shared_source["source"]
    kwargs = dict(identity=source.identity, policy=source.policy)
    paired = native.CLMHeadLearner(*shared_source["templates"], **kwargs)
    untrained = catalog.CLMCatalogLearner(*shared_source["templates"], **kwargs)
    return paired, untrained


@pytest.mark.parametrize(
    "fault",
    (
        "paired",
        "subclass",
        "probe",
        "untrained",
        "encoder_type",
        "instructions",
        "instructions_control",
        "revision",
        "revision_bound",
    ),
)
def test_should_refuse_unadmitted_sources_or_text_before_native_capture(
    shared_source, other_sources, fault
):
    case = shared_source
    source, frozen = case["source"], case["encoder"]
    instructions, revision = "Choose.", "fixture"
    if fault == "paired":
        source = other_sources[0]
    elif fault == "subclass":

        class Derived(catalog.CLMCatalogLearner):
            pass

        source = Derived.__new__(Derived)
    elif fault == "probe":
        source = CLMCatalogProbe.__new__(CLMCatalogProbe)
    elif fault == "untrained":
        source = other_sources[1]
    elif fault == "encoder_type":
        frozen = object()
    elif fault == "instructions":
        instructions = " Choose. "
    elif fault == "instructions_control":
        instructions = "Choose.\n"
    elif fault == "revision":
        revision = ""
    else:
        revision = "x" * 129
    work, encoder_work = case["source"].work, case["encoder"].work
    with pytest.raises(ValueError):
        CLMCatalogPredictorBinding(
            source, frozen, instructions=instructions, head_revision=revision
        )
    assert case["source"].work == work and case["encoder"].work == encoder_work


def test_should_refuse_healthy_foreign_encoder_identity_without_encoding(torch, shared_source):
    model, tokenizer = backbone(torch), FakeTokenizer(MAPPING)
    declared = replace(shared_source["encoder"].identity, revision="foreign")
    foreign = FrozenCLMEncoder(model, tokenizer, identity=declared, max_tokens=4)
    source_work = shared_source["source"].work
    with pytest.raises(ValueError):
        CLMCatalogPredictorBinding(
            shared_source["source"], foreign, instructions="Choose.", head_revision="fixture"
        )
    assert shared_source["source"].work == source_work and not model.calls and not tokenizer.calls


@pytest.mark.parametrize("fault", ("stale", "corrupt", "paired", "extra", "lost_return"))
def test_should_preserve_paid_snapshot_cost_and_refuse_stale_or_corrupt_native_returns(
    torch, shared_source, monkeypatch, fault
):
    case = shared_source
    source, frozen = case["source"], case["encoder"]
    original, source_work, encoded = source._capture(), source.work, frozen.work
    real_snapshot = source.snapshot_state

    def damaged_snapshot():
        saved = real_snapshot()
        if fault == "stale":
            return case["zero"]
        if fault == "corrupt":
            saved.state_head["weight"].add_(0.1)
        elif fault == "paired":
            return native.CLMHeadSnapshot(**vars(saved))
        elif fault == "extra":
            object.__setattr__(saved, "undeclared", True)
        else:
            raise RuntimeError("public catalog snapshot return lost")
        return saved

    factory = binding(case)
    monkeypatch.setattr(source, "snapshot_state", damaged_snapshot)
    with pytest.raises((ValueError, RuntimeError)):
        factory.bind()
    paid = factory.work
    assert paid["binding_attempts"] == 1 and paid["binding_returns"] == 0
    assert paid["snapshot_attempts"] == 1 and paid["head_copy_attempts"] == 0
    assert paid["snapshot_returns"] == (0 if fault == "lost_return" else 1)
    assert paid["unreturned_work_unknown"] is True
    assert factory.receipt is None
    expected = dict(source_work)
    expected["snapshot_attempts"] += 1
    expected["snapshot_returns"] += 1
    assert source.work == expected and frozen.work == encoded
    assert native._equal(vars(original), vars(source._capture()))
    with pytest.raises(ValueError):
        factory.bind()
    assert factory.work == paid and source.work == expected


@pytest.mark.parametrize("mutation", ("source_update", "encoder_work"))
def test_should_refuse_changes_during_predictor_copy_and_keep_observed_costs(
    torch, monkeypatch, mutation
):
    case = trained_source(torch)
    factory = binding(case)
    copy_predictor = binding_adapter._copy_predictor
    before = case["source"].work
    encoded = case["encoder"].work

    def changed(*args, **kwargs):
        predictor = copy_predictor(*args, **kwargs)
        if mutation == "source_update":
            case["source"].train_batch(case["training"], case["targets"])
        else:
            case["encoder"].embed_texts(("first",), ("head",))
        return predictor

    monkeypatch.setattr(binding_adapter, "_copy_predictor", changed)
    with pytest.raises(ValueError):
        factory.bind()
    paid = factory.work
    assert paid["binding_attempts"] == 1 and paid["binding_returns"] == 0
    assert paid["snapshot_attempts"] == paid["snapshot_returns"] == 1
    assert paid["head_copy_attempts"] == paid["head_copy_returns"] == 2
    assert (
        paid["head_parameter_elements_attempted"] == paid["head_parameter_elements_returned"] == 8
    )
    assert paid["scale_copy_elements_attempted"] == paid["scale_copy_elements_returned"] == 2
    assert paid["unreturned_work_unknown"] is True and factory.receipt is None
    assert case["source"].work["snapshot_attempts"] == before["snapshot_attempts"] + 1
    if mutation == "source_update":
        assert case["source"].completed_updates == 2
        assert case["source"].work["train_attempts"] == before["train_attempts"] + 1
        assert case["encoder"].work == encoded
    else:
        assert case["source"].completed_updates == 1
        assert case["encoder"].work["forward_returns"] == encoded["forward_returns"] + 1
    with pytest.raises(ValueError):
        factory.bind()
    assert factory.work == paid


def test_should_retain_existing_actor_partial_head_failure_charges_after_binding(torch):
    case = trained_source(torch)
    factory = binding(case)
    actor = factory.bind()
    original, work = case["source"]._capture(), case["source"].work

    def fail(module, args, output):
        raise RuntimeError("bound action head lost return")

    actor._action_head.register_forward_hook(fail)
    with pytest.raises(RuntimeError, match="lost return"):
        actor(request())
    paid = actor.work
    assert paid["encoder_attempts"] == paid["encoder_returns"] == 1
    assert paid["head_forward_attempts"] == 2 and paid["head_forward_returns"] == 1
    assert paid["request_attempts"] == 1 and paid["request_returns"] == 0
    assert paid["unreturned_work_unknown"] is True
    assert case["source"].work == work and native._equal(
        vars(original), vars(case["source"]._capture())
    )
    with pytest.raises(ValueError):
        actor(request())
    assert actor.work == paid


@pytest.mark.parametrize(
    "mutation",
    (
        "source_owner",
        "optimizer_groups",
        "source_identity",
        "source_rng",
        "source_mode",
        "source_gradient",
        "encoder_tokenizer",
        "encoder_identity",
        "encoder_limits",
        "encoder_work_owner",
        "encoder_mode",
    ),
)
def test_should_refuse_changed_birth_owners_and_declarations_before_any_snapshot(
    torch, shared_source, mutation
):
    case = shared_source
    source, frozen = case["source"], case["encoder"]
    factory = binding(case)
    before, encoded = source.work, frozen.work
    with pytest.MonkeyPatch.context() as patch:
        if mutation == "source_owner":
            patch.setattr(factory, "_source", copy(source))
        elif mutation == "optimizer_groups":
            patch.setattr(source._bundle[3], "param_groups", list(source._bundle[3].param_groups))
        elif mutation == "source_identity":
            patch.setattr(source, "_identity", replace(source._identity))
        elif mutation == "source_rng":
            patch.setattr(source, "_rng", source._rng.clone())
        elif mutation == "source_mode":
            patch.setattr(source._bundle[0], "training", not source._bundle[0].training)
        elif mutation == "source_gradient":
            parameter = next(source._bundle[0].parameters())
            patch.setattr(parameter, "grad", parameter.grad.clone())
        elif mutation == "encoder_tokenizer":
            patch.setattr(frozen, "_tokenizer", FakeTokenizer(MAPPING))
        elif mutation == "encoder_identity":
            patch.setattr(frozen, "_identity", replace(frozen._identity))
        elif mutation == "encoder_limits":
            patch.setattr(frozen, "_limits", (4, 16, 5))
        elif mutation == "encoder_work_owner":
            patch.setattr(frozen, "_work", dict(frozen._work))
        else:
            patch.setattr(case["model"], "training", True)
        with pytest.raises(ValueError):
            factory.bind()
    paid = factory.work
    assert paid["binding_attempts"] == 1 and paid["binding_returns"] == 0
    assert (
        paid["guard_capture_attempts"]
        == paid["snapshot_attempts"]
        == paid["head_copy_attempts"]
        == 0
    )
    assert paid["unreturned_work_unknown"] is False and factory.receipt is None
    assert source.work == before and frozen.work == encoded
    with pytest.raises(ValueError):
        factory.bind()
    assert factory.work == paid
