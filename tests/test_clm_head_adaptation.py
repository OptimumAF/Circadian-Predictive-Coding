"""Tiny native-head stream checks; no encoder or released model is loaded."""

from copy import deepcopy
from dataclasses import replace
from math import log
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_head_adaptation as study
from src.adapters.clm_head_learner import CLMHeadLearner
from src.core.clm_head import CLMHeadPolicy, FrozenEmbeddingBatch, FrozenEncoderIdentity


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def make_owner(torch, *, maximum=12):
    def linear(values):
        # Literal parameters avoid calling any random Linear initializer.
        head = torch.nn.Linear.__new__(torch.nn.Linear)
        torch.nn.Module.__init__(head)
        head.in_features = head.out_features = 2
        head.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        head.register_parameter("bias", None)
        return head

    state = linear([[1.0, 0.1], [0.2, 1.0]])
    action = linear([[0.9, -0.1], [0.1, 1.1]])
    state.eval()
    declared = FrozenEncoderIdentity("fake", "fixture", "fixture", "last", "literal-three-pairs", 2)
    owner = CLMHeadLearner(
        state,
        action,
        torch.nn.Parameter(torch.tensor(0.0)),
        identity=declared,
        policy=CLMHeadPolicy(projection_dim=2, max_updates=maximum),
    )
    batch = FrozenEmbeddingBatch(
        declared,
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]),
        torch.tensor([[0.8, 0.2], [0.1, 0.9], [-0.7, 0.6]]),
    )
    return owner, batch, state, action


@pytest.fixture(scope="module")
def completed_stream(torch):
    owner, batch, state, action = make_owner(torch)
    source = deepcopy(state.state_dict()), deepcopy(action.state_dict())
    features = batch.states.clone(), batch.actions.clone()
    caller_rng = torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    saves, paid_batches, scored_at = [], [], []
    train = owner.train_batch

    def record_train(features, targets):
        paid_batches.append(
            (features.states.clone(), features.actions.clone(), targets.group_codes)
        )
        return train(features, targets)

    owner.train_batch = record_train

    def save():
        saves.append(
            {
                "updates": owner.completed_updates,
                "scored": "scores" in report,
                "readouts": len(report.get("raw_readouts", [])),
            }
        )

    score = study.score_readouts

    def checked_score(readouts):
        scored_at.append(owner.completed_updates)
        assert owner.completed_updates == 12
        return score(readouts)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "score_readouts", checked_score)
        study.run_stream(owner, batch, report, save=save)
    return (
        owner,
        batch,
        state,
        action,
        source,
        features,
        caller_rng,
        report,
        saves,
        paid_batches,
        scored_at,
    )


def test_should_apply_fixed_A_B_return_dose_and_only_score_after_all_updates(
    torch, completed_stream
):
    owner, batch, _, _, _, _, _, report, saves, paid_batches, scored_at = completed_stream
    assert report["stream_complete"] is True
    assert owner.completed_updates == 12 and owner.training_examples == 24
    assert report["stream"]["completed_updates"] == 12
    assert report["stream"]["training_pairs"] == 24
    assert [row["update"] for row in report["rows"]] == list(range(1, 13))
    assert [row["phase"] for row in report["rows"]] == ["A"] * 4 + ["B"] * 4 + ["return_A"] * 4
    assert scored_at == [12]
    assert all(not row["scored"] for row in saves if row["updates"] < 12)
    assert [(row["name"], row["completed_updates"]) for row in report["raw_readouts"]] == [
        ("baseline0", 0),
        ("after_A4", 4),
        ("after_B8", 8),
        ("after_return12", 12),
    ]
    assert len(paid_batches) == 12
    for index, (states, actions, codes) in enumerate(paid_batches):
        chosen = [0, 1] if index < 4 or index >= 8 else [2, 1]
        assert torch.equal(states, batch.states[chosen])
        assert torch.equal(actions, batch.actions[chosen])
        assert len(codes) == 2 and codes[0] != codes[1]
    assert paid_batches[0][2] == paid_batches[8][2]
    assert paid_batches[0][2][1] == paid_batches[4][2][1]


def test_should_preserve_frozen_features_control_heads_and_caller_rng(torch, completed_stream):
    owner, batch, state, action, source, features, rng, report, *_ = completed_stream
    for head, expected in zip((state, action), source):
        for key, value in head.state_dict().items():
            assert torch.equal(value, expected[key])
    assert torch.equal(batch.states, features[0]) and torch.equal(batch.actions, features[1])
    assert not state.training and action.training
    assert torch.equal(torch.get_rng_state(), rng)
    work = report["learner_work"]
    assert work == owner.work
    assert work["train_attempts"] == work["commits"] == 12
    assert work["training_rows_attempted"] == 24
    assert work["prediction_attempts"] == work["prediction_returns"] == 4
    for field in (
        "backward_attempts",
        "backward_returns",
        "optimizer_attempts",
        "optimizer_returns",
        "scheduler_attempts",
        "scheduler_returns",
    ):
        assert work[field] == 12
    assert work["unreturned_work_unknown"] is False


def test_should_keep_probability_readouts_as_raw_observations_and_score_each_task(completed_stream):
    report = completed_stream[7]
    for row in report["raw_readouts"]:
        matrix = row["probabilities"]
        metrics = report["scores"][row["name"]]
        assert len(matrix) == 3 and all(len(values) == 3 for values in matrix)
        expected_accuracy, expected_ce = [], []
        for index, task in enumerate(("password", "parcel", "refund")):
            assert sum(matrix[index]) == pytest.approx(1.0, abs=1e-6)
            winner = max(range(3), key=matrix[index].__getitem__)
            expected_accuracy.append(float(winner == index))
            expected_ce.append(-log(matrix[index][index]))
            assert metrics["tasks"][task]["accuracy"] == expected_accuracy[-1]
            assert metrics["tasks"][task]["ce"] == pytest.approx(expected_ce[-1])
        assert metrics["all"]["accuracy"] == pytest.approx(sum(expected_accuracy) / 3)
        assert metrics["all"]["ce"] == pytest.approx(sum(expected_ce) / 3)


def test_should_use_delayed_unique_train_receipts_and_no_replay(completed_stream):
    receipts = completed_stream[7]["stream"]["receipts"]
    assert len(receipts) == 12
    assert [row["update_number"] for row in receipts] == list(range(1, 13))
    assert [row["observed_at"] for row in receipts] == list(range(0, 24, 2))
    assert [row["arrived_at"] for row in receipts] == list(range(1, 24, 2))
    assert [row["applied_at"] for row in receipts] == list(range(1, 24, 2))
    assert len({row["sample_id"] for row in receipts}) == 12


def test_should_preserve_first_paid_receipt_when_save_fails_after_commit(torch):
    owner, batch, *_ = make_owner(torch)
    report: dict[str, Any] = {}
    saved = []

    def save():
        saved.append(deepcopy(report))
        if owner.completed_updates == 1:
            raise RuntimeError("paid diagnostic save failed")

    with pytest.raises(RuntimeError, match="paid diagnostic"):
        study.run_stream(owner, batch, report, save=save)
    assert report["stream_complete"] is False and "scores" not in report
    assert report["stream"]["completed_updates"] == owner.completed_updates == 1
    assert report["stream"]["training_pairs"] == 2
    assert len(report["stream"]["receipts"]) == len(report["rows"]) == 1
    assert len(report["raw_readouts"]) == 1
    assert report["learner_work"]["train_attempts"] == report["learner_work"]["commits"] == 1
    assert report["learner_work"]["prediction_attempts"] == 1
    assert saved[-1]["stream"]["receipts"] == report["stream"]["receipts"]
    paid = owner.work
    with pytest.raises(ValueError):
        study.run_stream(owner, batch, {})
    assert owner.work == paid


@pytest.fixture(scope="module")
def fresh_owner(torch):
    return make_owner(torch)[:2]


@pytest.mark.parametrize("malformation", ["role", "identity", "rows", "dtype", "nonfinite"])
def test_should_refuse_invalid_stream_inputs_before_paid_work(torch, fresh_owner, malformation):
    owner, original = fresh_owner
    batch = FrozenEmbeddingBatch(
        original.identity, original.states.clone(), original.actions.clone()
    )
    if malformation == "role":
        object.__setattr__(batch, "role", "probe_train")
    elif malformation == "identity":
        object.__setattr__(batch, "identity", replace(batch.identity, tokenizer_revision="foreign"))
    elif malformation == "rows":
        object.__setattr__(batch, "states", batch.states[:2])
        object.__setattr__(batch, "actions", batch.actions[:2])
    elif malformation == "dtype":
        object.__setattr__(batch, "states", batch.states.double())
    else:
        batch.states[0, 0] = float("nan")
    before = owner.work
    with pytest.raises(ValueError):
        study.run_stream(owner, batch, {})
    assert owner.work == before and owner.completed_updates == 0


def test_should_refuse_wrong_update_budget_before_baseline_readout(torch):
    owner, batch, *_ = make_owner(torch, maximum=8)
    before = owner.work
    with pytest.raises(ValueError):
        study.run_stream(owner, batch, {})
    assert owner.work == before and owner.completed_updates == 0


def test_should_preserve_fixed_native_strings_and_candidate_order():
    states, actions = study.fixed_texts()
    assert states == (
        "I forgot my password and cannot sign in.\n\n" + study.INSTRUCTIONS,
        "Where is my parcel and when will it arrive?\n\n" + study.INSTRUCTIONS,
        "I was charged twice and need an invoice refund.\n\n" + study.INSTRUCTIONS,
    )
    assert actions == (
        "Reset account password",
        "Check parcel tracking",
        "Request an invoice refund",
    )


def readouts(matrix=None):
    matrix = matrix or [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.25, 0.25, 0.5]]
    return [
        dict(name=name, completed_updates=4 * index, probabilities=deepcopy(matrix))
        for index, name in enumerate(("baseline0", "after_A4", "after_B8", "after_return12"))
    ]


def test_should_compute_independent_task_and_aggregate_metrics_without_mutating_inputs():
    values = readouts()
    original = deepcopy(values)
    result = study.score_readouts(values)
    expected = [-log(0.8), -log(0.7), -log(0.5)]
    for point in result.values():
        assert point["selected_ids"] == ["password", "parcel", "refund"]
        assert point["all"] == pytest.approx({"accuracy": 1.0, "ce": sum(expected) / 3})
        for task, loss in zip(("password", "parcel", "refund"), expected):
            assert point["tasks"][task] == pytest.approx({"accuracy": 1.0, "ce": loss})
    result["baseline0"]["selected_ids"].clear()
    assert values == original


def test_should_keep_wrong_tied_and_zero_probability_scores_descriptive():
    result = study.score_readouts(readouts([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]))
    for point in result.values():
        assert point["selected_ids"] == ["parcel", "password", "password"]
        assert point["all"]["accuracy"] == 0
        assert point["tasks"]["password"]["ce"] == pytest.approx(-log(1e-12))
        assert point["tasks"]["parcel"]["ce"] == pytest.approx(-log(0.5))
        assert point["tasks"]["refund"]["ce"] == pytest.approx(-log(1e-12))


@pytest.mark.parametrize(
    "fault", ["missing", "order", "counter", "shape", "mass", "negative", "nonfinite"]
)
def test_should_refuse_malformed_readouts_without_repair(fault):
    values = readouts()
    if fault == "missing":
        values.pop()
    elif fault == "order":
        values[0], values[1] = values[1], values[0]
    elif fault == "counter":
        values[1]["completed_updates"] = 5
    elif fault == "shape":
        values[1]["probabilities"][0].pop()
    elif fault == "mass":
        values[1]["probabilities"][0] = [0.8, 0.2, 0.2]
    elif fault == "negative":
        values[1]["probabilities"][0] = [-0.1, 0.5, 0.6]
    else:
        values[1]["probabilities"][0][0] = float("nan")
    with pytest.raises(ValueError):
        study.score_readouts(values)


def test_should_refuse_existing_cli_output_before_loading(monkeypatch, tmp_path):
    output = tmp_path / "saved.json"
    output.write_text("prior paid evidence", encoding="utf8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm-adaptation",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(output),
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("existing output reached native loader")

    monkeypatch.setattr(study, "run_case", forbidden)
    with pytest.raises(ValueError, match="refusing"):
        study.main()
    assert output.read_text(encoding="utf8") == "prior paid evidence"


def test_should_publish_cli_report_from_injected_case_without_native_work(monkeypatch, tmp_path):
    import json

    output = tmp_path / "fresh.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm-adaptation",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(output),
        ],
    )
    called = []

    def case(model_dir, head, report, *, save):
        called.append((model_dir, head))
        report.update(complete=True, fixture="metadata only")
        save()

    monkeypatch.setattr(study, "run_case", case)
    study.main()
    assert called == [(tmp_path, tmp_path / "head.pt")]
    assert json.loads(output.read_text(encoding="utf8")) == {
        "complete": True,
        "fixture": "metadata only",
    }


def test_should_refuse_spent_physical_allowance_before_baseline_readout(monkeypatch, fresh_owner):
    owner, batch = fresh_owner
    # Synthetic permanent-work history proves that logical zero alone cannot renew work.
    with monkeypatch.context() as patch:
        patch.setitem(owner._work, "train_attempts", 1)
        paid = owner.work
        with pytest.raises(ValueError):
            study.run_stream(owner, batch, {})
        assert owner.work == paid and owner.completed_updates == 0


def test_should_distinguish_endpoint_passes_from_acquisition_retention_and_recovery():
    def scores(rows):
        return {
            point: {"tasks": {task: {"accuracy": value} for task, value in zip(study.IDS, row)}}
            for point, row in zip(study.POINTS, rows)
        }

    unchanged = study._usefulness(scores([(1, 1, 1)] * 4))
    assert unchanged["A_endpoint_pass"] and unchanged["B_endpoint_pass"]
    assert unchanged["return_A_endpoint_pass"] and unchanged["retained_password_after_B"]
    assert not unchanged["newly_acquired_password"] and not unchanged["new_refund_acquisition"]
    assert not unchanged["recovered_password"]
    changed = study._usefulness(scores([(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]))
    assert changed["newly_acquired_password"] and changed["new_refund_acquisition"]
    assert changed["refund_corrected_during_B"] and changed["recovered_password"]
    assert not changed["retained_password_after_B"]
