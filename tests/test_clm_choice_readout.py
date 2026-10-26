"""Rectangular CLM port checks on literal heads, with no encoder or random init."""

from copy import deepcopy
from dataclasses import replace
from math import exp, sqrt

import pytest

pytest.importorskip("torch")

from src.adapters.clm_head_learner import CLMHeadLearner, _equal
from src.core.clm_head import (
    CLMHeadPolicy,
    CLMTargets,
    FrozenChoiceBatch,
    FrozenEmbeddingBatch,
    FrozenEncoderIdentity,
)
from src.core.temporal_decision_sequence import temporal_examples


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def make_owner(torch):
    def linear(weights):
        head = torch.nn.Linear.__new__(torch.nn.Linear)
        torch.nn.Module.__init__(head)
        head.in_features = head.out_features = 2
        head.weight = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        head.register_parameter("bias", None)
        return head

    state, action = linear([[1.0, 0.2], [-0.1, 0.9]]), linear([[0.8, -0.2], [0.3, 1.1]])
    state.eval()
    declared = FrozenEncoderIdentity("fake", "literal", "literal", "last", "rectangular-fixture", 2)
    learner = CLMHeadLearner(
        state,
        action,
        torch.nn.Parameter(torch.tensor(0.2)),
        identity=declared,
        policy=CLMHeadPolicy(projection_dim=2, max_updates=4, max_rows=4, max_restores=1),
    )
    choice = FrozenChoiceBatch(
        declared,
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.2, -0.8]]),
        torch.tensor([[1.0, 0.1], [0.0, 1.0], [-0.5, 0.8]]),
    )
    paired = FrozenEmbeddingBatch(declared, choice.states[:2].clone(), choice.actions[:2].clone())
    return learner, choice, paired, state, action


def scalar_probabilities(snapshot, states, actions):
    """Independent scalar affine/cosine/softmax; creates no oracle head modules."""

    def projected(values, weights):
        result = [sum(x * w for x, w in zip(values, row)) for row in weights]
        norm = max(sqrt(sum(x * x for x in result)), 1e-12)
        return [x / norm for x in result]

    state_weights = snapshot.state_head["weight"].tolist()
    action_weights = snapshot.action_head["weight"].tolist()
    scale = min(exp(snapshot.logit_scale.item()), 100.0)
    candidates = [projected(row, action_weights) for row in actions]
    result = []
    for state in states:
        query = projected(state, state_weights)
        logits = [scale * sum(a * b for a, b in zip(query, candidate)) for candidate in candidates]
        masses = [exp(value - max(logits)) for value in logits]
        result.append([value / sum(masses) for value in masses])
    return result


@pytest.fixture(scope="module")
def readout_case(torch):
    learner, choice, paired, state, action = make_owner(torch)
    examples = temporal_examples("after_B", "outer_selection")
    query_features = dict(zip((example.request.query for example in examples), choice.states))
    candidate_ids = tuple(item.candidate_id for item in examples[0].request.allowed_candidates)
    candidate_features = dict(zip(candidate_ids, choice.actions))
    choice = FrozenChoiceBatch(
        choice.identity,
        torch.stack([query_features[example.request.query] for example in examples]),
        torch.stack(
            [
                candidate_features[candidate.candidate_id]
                for candidate in examples[0].request.allowed_candidates
            ]
        ),
    )
    original = deepcopy(state.state_dict()), deepcopy(action.state_dict())
    inputs = choice.states.clone(), choice.actions.clone()
    caller_rng = torch.get_rng_state().clone()
    learner.train_batch(paired, CLMTargets((0, 1)))
    before = learner.snapshot_state()
    rectangular = learner.predict_choices(choice)
    singleton = learner.predict_choices(
        FrozenChoiceBatch(choice.identity, choice.states[2:3], choice.actions)
    )
    permutation = [2, 0, 1]
    reordered = learner.predict_choices(
        FrozenChoiceBatch(choice.identity, choice.states, choice.actions[permutation])
    )
    square = FrozenEmbeddingBatch(choice.identity, choice.states[:3], choice.actions)
    paired_result = learner.predict(square)
    square_result = learner.predict_choices(
        FrozenChoiceBatch(choice.identity, square.states, square.actions, "train")
    )
    after = learner.snapshot_state()
    saved = rectangular.clone()
    rectangular.zero_()
    detached_preserved = _equal(vars(after), vars(learner._capture()))
    learner.train_batch(paired, CLMTargets((0, 1)))
    continued = learner.snapshot_state()
    paid_before_restore = learner.work
    learner.restore_state(before)
    restored = learner.snapshot_state()
    return dict(
        learner=learner,
        examples=examples,
        choice=choice,
        paired=paired,
        sources=(state, action),
        original=original,
        inputs=inputs,
        rng=caller_rng,
        before=before,
        after=after,
        rectangular=saved,
        singleton=singleton,
        reordered=reordered,
        square=paired_result,
        square_choice=square_result,
        detached=detached_preserved,
        continued=continued,
        paid=paid_before_restore,
        restored=restored,
    )


def test_should_match_independent_four_query_three_candidate_cosines(torch, readout_case):
    case = readout_case
    expected = scalar_probabilities(
        case["before"], case["choice"].states.tolist(), case["choice"].actions.tolist()
    )
    actual = case["rectangular"]
    assert actual.shape == (4, 3) and actual.dtype == torch.float32
    assert actual.device.type == "cpu" and not actual.requires_grad
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=2e-7)
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(4), rtol=0, atol=1e-7)


def test_should_preserve_candidate_permutation_and_single_query(torch, readout_case):
    case = readout_case
    torch.testing.assert_close(case["singleton"], case["rectangular"][2:3], rtol=0, atol=2e-7)
    torch.testing.assert_close(
        case["reordered"], case["rectangular"][:, [2, 0, 1]], rtol=0, atol=2e-7
    )


def test_should_keep_existing_square_prediction_numerically_equal(torch, readout_case):
    torch.testing.assert_close(
        readout_case["square_choice"], readout_case["square"], rtol=0, atol=0
    )


def test_should_preserve_complete_nonempty_native_state_sources_and_rng(torch, readout_case):
    case = readout_case
    assert _equal(vars(case["before"]), vars(case["after"])) and case["detached"]
    assert case["before"].optimizer_state["state"]
    assert all(value is not None for value in case["before"].gradients)
    assert torch.equal(case["choice"].states, case["inputs"][0])
    assert torch.equal(case["choice"].actions, case["inputs"][1])
    for head, saved in zip(case["sources"], case["original"]):
        for name, value in head.state_dict().items():
            assert torch.equal(value, saved[name])
    assert not case["sources"][0].training and case["sources"][1].training
    assert torch.equal(case["rng"], torch.get_rng_state())


def test_should_continue_paired_training_without_refunding_readouts_on_restore(readout_case):
    case, owner = readout_case, readout_case["learner"]
    assert case["continued"].completed_updates == 2 and case["continued"].training_examples == 4
    assert not _equal(case["before"].state_head, case["continued"].state_head)
    assert _equal(vars(case["before"]), vars(case["restored"]))
    assert owner.completed_updates == 1 and owner.work["train_attempts"] == 2
    for field, expected in (
        ("prediction_attempts", 5),
        ("prediction_returns", 5),
        ("prediction_rows_attempted", 15),
        ("choice_prediction_attempts", 4),
        ("choice_prediction_returns", 4),
        ("choice_candidate_rows_attempted", 12),
    ):
        assert owner.work[field] == case["paid"][field] == expected
    work = owner.work
    work.clear()
    assert owner.work["choice_prediction_attempts"] == 4


def test_should_admit_existing_four_case_temporal_development_without_training(readout_case):
    examples = readout_case["examples"]
    assert examples == temporal_examples("after_B", "outer_selection")
    assert len(examples) == readout_case["rectangular"].shape[0] == 4
    candidate_ids = tuple(item.candidate_id for item in examples[0].request.allowed_candidates)
    assert candidate_ids == ("defer", "inspect", "route")
    assert all(
        tuple(item.candidate_id for item in example.request.allowed_candidates) == candidate_ids
        for example in examples
    )
    observed = [dict(zip(candidate_ids, row)) for row in readout_case["rectangular"].tolist()]
    assert all(set(row) == set(candidate_ids) for row in observed)
    assert all(example.role == "outer_selection" for example in examples)
    assert readout_case["learner"].work["commits"] == 2


@pytest.fixture(scope="module")
def fresh_case(torch):
    return make_owner(torch)[:3]


@pytest.mark.parametrize(
    "fault",
    [
        "probe",
        "final",
        "identity",
        "dtype",
        "nonfinite",
        "empty_queries",
        "too_many_queries",
        "one_candidate",
        "too_many_candidates",
        "query_width",
        "candidate_width",
    ],
)
def test_should_refuse_all_malformed_choice_inputs_before_work(torch, fresh_case, fault):
    owner, original, _ = fresh_case
    choice = FrozenChoiceBatch(original.identity, original.states.clone(), original.actions.clone())
    if fault in ("probe", "final"):
        object.__setattr__(choice, "role", "probe_train" if fault == "probe" else "final_test")
    elif fault == "identity":
        object.__setattr__(choice, "identity", replace(choice.identity, revision="foreign"))
    elif fault == "dtype":
        object.__setattr__(choice, "states", choice.states.double())
    elif fault == "nonfinite":
        choice.actions[0, 0] = float("nan")
    elif fault == "empty_queries":
        object.__setattr__(choice, "states", choice.states[:0])
    elif fault == "too_many_queries":
        object.__setattr__(choice, "states", torch.cat((choice.states, choice.states[:1])))
    elif fault == "one_candidate":
        object.__setattr__(choice, "actions", choice.actions[:1])
    elif fault == "too_many_candidates":
        object.__setattr__(choice, "actions", torch.cat((choice.actions, choice.actions[:2])))
    elif fault == "query_width":
        object.__setattr__(choice, "states", choice.states[:, :1])
    else:
        object.__setattr__(choice, "actions", choice.actions[:, :1])
    before = owner.work
    with pytest.raises(ValueError):
        owner.predict_choices(choice)
    assert owner.work == before and owner.completed_updates == 0


@pytest.mark.parametrize("operation", ["train", "paired_predict"])
def test_should_never_treat_square_choice_records_as_paired_training(fresh_case, operation):
    owner, original, _ = fresh_case
    square = FrozenChoiceBatch(original.identity, original.states[:3], original.actions, "train")
    before = owner.work
    with pytest.raises(ValueError):
        if operation == "train":
            owner.train_batch(square, CLMTargets((0, 1, 2)))
        else:
            owner.predict(square)
    assert owner.work == before


@pytest.mark.parametrize("role", ["probe_train", "final_test", "inner_guard"])
def test_should_reject_unadmitted_choice_roles_at_record_birth(fresh_case, role):
    with pytest.raises(ValueError):
        FrozenChoiceBatch(fresh_case[1].identity, None, None, role)


@pytest.mark.parametrize("fault", ["unreturned", "source_changed"])
def test_should_preserve_paid_readout_failure_and_poison_without_retry(torch, monkeypatch, fault):
    owner, choice, *_ = make_owner(torch)
    before = owner._capture()
    native_project = owner._project

    def project(bundle, states, actions):
        if fault == "unreturned":
            raise RuntimeError("rectangular native return unavailable")
        result = native_project(bundle, states, actions)
        with torch.no_grad():
            owner._bundle[0].weight.add_(0.1)
        return result

    monkeypatch.setattr(owner, "_project", project)
    with pytest.raises((ValueError, RuntimeError)):
        owner.predict_choices(choice)
    paid = owner.work
    assert paid["prediction_attempts"] == paid["choice_prediction_attempts"] == 1
    assert paid["prediction_rows_attempted"] == 4 and paid["choice_candidate_rows_attempted"] == 3
    assert (
        paid["prediction_returns"]
        == paid["choice_prediction_returns"]
        == (0 if fault == "unreturned" else 1)
    )
    assert paid["commits"] == 0 and paid["unreturned_work_unknown"] is True
    if fault == "unreturned":
        assert _equal(vars(before), vars(owner._capture()))
    else:
        assert not _equal(before.state_head, owner._capture().state_head)
    with pytest.raises(ValueError):
        owner.predict_choices(choice)
    assert owner.work == paid
