"""One successful repeat TRAIN stream followed by four real typed requests.

The extra invoice row is repeated exposure, not another task or new evidence.
One module fixture owns all twelve updates; cache refusers never retrain it.
"""

from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_head_learner as native
from src.adapters.clm_catalog_learner import CLMCatalogLearner
from src.core.clm_catalog import CATALOG_OBJECTIVE
from src.core.temporal_decision_sequence import temporal_examples
from test_clm_typed_stream_readout import execute, json_record


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def repeated(torch):
    packets = []
    original_train = CLMCatalogLearner.train_batch

    def observed_train(self, batch, targets):
        # Scalar trace makes the actual native packet observable without copies
        # of native continuation state or another training/encoding operation.
        packets.append(
            dict(
                entered_at_update=self.completed_updates,
                role=batch.role,
                states=batch.states.tolist(),
                candidates=list(batch.candidate_ids),
                targets=list(targets.candidate_indices),
            )
        )
        return original_train(self, batch, targets)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CLMCatalogLearner, "train_batch", observed_train)
        result = execute(torch, mode="repeat")
    result["packets"] = packets
    return result


def test_should_finish_twelve_updates_and_keep_permanent_native_work_after_typed_reads(repeated):
    report = repeated["report"]
    learner = repeated["handles"]["learner"]
    assert report["complete"] is report["sequence_complete"] is True
    assert report["catalog_support"] == "repeat"
    assert report["training_objective"] == CATALOG_OBJECTIVE
    assert learner.completed_updates == learner.generation == 12
    assert learner.training_examples == report["stream"]["training_pairs"] == 44
    work = report["learner_work"]
    assert work == learner.work
    for key in (
        "train_attempts",
        "commits",
        "backward_attempts",
        "backward_returns",
        "optimizer_attempts",
        "optimizer_returns",
        "scheduler_attempts",
        "scheduler_returns",
    ):
        assert work[key] == 12
    assert work["training_rows_attempted"] == 44
    assert work["catalog_candidate_rows_attempted"] == 36
    assert work["catalog_logit_values_attempted"] == 132
    assert work["snapshot_attempts"] == work["snapshot_returns"] == 1
    assert work["restore_attempts"] == work["restore_returns"] == 0
    assert work["choice_prediction_attempts"] == work["choice_prediction_returns"] == 5
    assert work["prediction_rows_attempted"] == 20
    assert work["choice_candidate_rows_attempted"] == 15
    assert work["unreturned_work_unknown"] is False
    assert all(
        update == 12
        for update, predictions, scored in repeated["observations"]
        if predictions or scored
    )


def test_should_apply_literal_repeat_packets_with_original_observation_and_arrival_clocks(repeated):
    report = repeated["report"]
    packets = repeated["packets"]
    assert len(packets) == len(report["rows"]) == len(report["stream"]["receipts"]) == 12
    for index, (packet, row, receipt) in enumerate(
        zip(packets, report["rows"], report["stream"]["receipts"], strict=True)
    ):
        phase = ("after_A", "after_B", "after_return")[index // 4]
        exposure = index % 4
        sample = f"clm-{phase}-{exposure}"
        expected_ids = (
            ["inspect", "retire", "route"] if index < 4 else ["defer", "inspect", "route"]
        )
        expected_targets = [2, 1, 0] if index < 4 else [1, 0, 0, 1] if index < 8 else [1, 0, 1, 1]
        assert packet["entered_at_update"] == index and packet["role"] == "train"
        assert packet["candidates"] == row["candidate_ids"] == expected_ids
        assert packet["targets"] == row["candidate_indices"] == expected_targets
        assert len(packet["states"]) == (3 if index < 4 else 4)
        if index >= 4:
            assert packet["states"][3] == packet["states"][0]
            assert len({tuple(state) for state in packet["states"]}) == 3
        assert row["phase"] == phase and row["exposure"] == exposure
        assert receipt["update_number"] == row["update"] == index + 1
        assert receipt["sample_id"] == sample and receipt["event_id"] == "label-" + sample
        assert receipt["episode_id"] == receipt["learner_version"] == "clm-temporal"
        assert receipt["actor_version"] == phase
        assert receipt["observed_at"] == 2 * index
        assert receipt["arrived_at"] == receipt["applied_at"] == 2 * index + 1
        assert receipt["diagnostic"] == {"definition": CATALOG_OBJECTIVE, "value": row["loss"]}


def test_should_return_all_four_canonical_typed_requests_and_cached_probability_parity(repeated):
    report = repeated["report"]
    record = report["typed_development"]
    examples = temporal_examples("after_return", "outer_selection")
    assert record["complete"] is True
    assert record["requests"] == [json_record(example.request) for example in examples]
    assert len(record["predictions"]) == 4
    differences: list[float] = []
    for raw, example, cached in zip(
        record["predictions"], examples, report["raw_readouts"][-1]["probabilities"], strict=True
    ):
        assert raw["request_id"] == example.request.request_id
        assert [pair[0] for pair in raw["probabilities"]] == ["defer", "inspect", "route"]
        observed = [pair[1] for pair in raw["probabilities"]]
        assert observed == pytest.approx(cached, abs=2e-7)
        differences.extend(abs(a - b) for a, b in zip(observed, cached, strict=True))
    assert record["max_abs_cached_difference"] == max(differences)
    actor = record["predictor_work"]
    assert actor["request_attempts"] == actor["request_returns"] == 4
    assert actor["encoder_attempts"] == actor["encoder_returns"] == 4
    assert actor["head_forward_attempts"] == actor["head_forward_returns"] == 8
    assert actor["projected_rows"] == 16 and actor["unreturned_work_unknown"] is False


def test_should_bind_exact_repeat_source_and_charge_copies_and_source_guards(repeated):
    report = repeated["report"]
    record = report["typed_development"]
    learner = repeated["handles"]["learner"]
    work, receipt, guard = (
        record["binding_work"],
        record["binding_receipt"],
        record["source_guard_work"],
    )
    assert work["binding_attempts"] == work["binding_returns"] == 1
    assert work["snapshot_attempts"] == work["snapshot_returns"] == 1
    assert work["guard_capture_attempts"] == work["guard_capture_returns"] == 3
    assert work["head_copy_attempts"] == work["head_copy_returns"] == 2
    assert (
        work["head_parameter_elements_attempted"] == work["head_parameter_elements_returned"] == 8
    )
    assert work["scale_copy_elements_attempted"] == work["scale_copy_elements_returned"] == 2
    # Nine parameters + nine gradients + eighteen moments + three Adam steps;
    # RNG size is platform-native. Four binding snapshots and six source guards.
    capture_elements = 39 + learner._rng.numel()
    assert work["snapshot_tensor_elements_copied"] == 4 * capture_elements
    assert guard["capture_attempts"] == guard["capture_returns"] == 6
    assert guard["tensor_elements_copied"] == 6 * capture_elements
    assert receipt["work"] == work and receipt["objective"] == CATALOG_OBJECTIVE
    assert receipt["source_completed_updates"] == receipt["source_generation"] == 12
    assert receipt["source_training_examples"] == 44
    assert receipt["identity"] == report["identity"]
    assert receipt["policy"] == record["source_policy"] == vars(learner.policy)
    assert receipt["instructions"] == study.INSTRUCTIONS


def test_should_keep_one_nonrenewable_encoder_and_unchanged_loaded_controls(torch, repeated):
    report = repeated["report"]
    encoders = repeated["handles"]["encoders"]
    assert len(encoders) == 1 and encoders[0]._limits[2] == 5
    assert [len(call["ids"]) for call in repeated["model"].calls] == [13, 4, 4, 4, 4]
    before = report["typed_development"]["encoder_work_before"]
    after = report["typed_development"]["encoder_work_after"]
    assert before["forward_returns"] == 1 and after["forward_returns"] == 5
    assert before["sequences_returned"] == 13 and after["sequences_returned"] == 29
    assert after == report["encoder_work"]
    paid = encoders[0].work
    with pytest.raises(ValueError, match="allowance exhausted"):
        encoders[0].embed_texts(("invoice",), ("tail",))
    assert encoders[0].work == paid
    state, action, scale = repeated["original"]
    assert native._equal(repeated["loaded"].state_head.state_dict(), state)
    assert native._equal(repeated["loaded"].action_head.state_dict(), action)
    assert torch.equal(repeated["loaded"].logit_scale, scale)
    assert torch.equal(torch.get_rng_state(), repeated["rng"])
    assert report["control_unchanged"] is report["caller_rng_unchanged"] is True


def test_should_admit_complete_typed_cache_and_retain_every_original_metric(repeated):
    report = repeated["report"]
    source, frozen = repeated["handles"]["learner"], repeated["handles"]["encoders"][0]
    paid, encoded = source.work, frozen.work
    original = deepcopy(report)
    study._require_catalog_completed(report)
    rescored = study.score_sequence(report)
    assert rescored["checkpoints"] == report["checkpoints"]
    assert rescored["contrasts"] == report["contrasts"]
    assert rescored["cached_typed_decodes"] == report["cached_typed_decodes"] == 88
    assert [point["phase"] for point in rescored["checkpoints"]] == [
        "baseline",
        "after_A",
        "after_B",
        "after_return",
    ]
    assert report == original and source.work == paid and frozen.work == encoded


@pytest.mark.parametrize("fault", ("source_rows", "receipt_clock", "actor_schema"))
def test_should_refuse_corrupted_repeat_cache_before_scoring_without_more_native_work(
    repeated, monkeypatch, fault
):
    report = deepcopy(repeated["report"])
    if fault == "source_rows":
        report["typed_development"]["binding_receipt"]["source_training_examples"] = 36
    elif fault == "receipt_clock":
        report["stream"]["receipts"][-1]["observed_at"] += 1
    else:
        report["typed_development"]["predictor_work"]["unaccounted"] = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("corrupted repeat cache reached scoring")

    monkeypatch.setattr(study, "_score_cached", forbidden)
    source, frozen = repeated["handles"]["learner"], repeated["handles"]["encoders"][0]
    paid, encoded = source.work, frozen.work
    with pytest.raises(ValueError):
        study.score_sequence(report)
    assert source.work == paid and frozen.work == encoded


def test_should_route_existing_repeat_and_typed_cli_flags_without_another_numeric_case(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm-temporal",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(tmp_path / "new.json"),
            "--catalog-training",
            "--catalog-support",
            "repeat",
            "--typed-development-readout",
        ],
    )
    calls = []

    def case(model_dir: Path, head: Path, report, *, train_catalog_diagnostics, save, **kwargs):
        calls.append(kwargs)
        report.update(complete=True, fixture="CLI metadata only")
        save()

    monkeypatch.setattr(study, "run_case", case)
    study.main()
    assert calls == [
        {"catalog_training": True, "catalog_support": "repeat", "typed_development_readout": True}
    ]
