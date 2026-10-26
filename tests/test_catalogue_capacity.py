"""Finite catalogue capacity and new-C suffix behavior, not scientific outcomes."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_catalogue_capacity as study
from src.adapters import minigrid_context_linear_learner as port
from src.adapters import minigrid_train_catalogue as catalogue
from test_arrived_cue_catalogue import episode
from test_retained_context_linear_games import state


def sources():
    return tuple(
        episode(f"minigrid-source-{context}-{name}", cue, mark=mark, action=action, context=context)
        for context, mark, action in (("A", 1, 2), ("B", 3, 5), ("C_train", 5, 1))
        for name, cue in (("blue", 2), ("red", 0))
    )


@pytest.mark.parametrize("capacity", [True, False, 0, 3, 1.0, "2"])
def test_should_refuse_non_exact_capacity_before_source_admission(monkeypatch, capacity):
    monkeypatch.setattr(catalogue, "_validate_sources", lambda *a: pytest.fail("source reached"))
    with pytest.raises(ValueError, match="exact integer"):
        catalogue.compose_arrived_cue_catalogue((), retained_per_cue=capacity)


def test_should_compose_arrived_abc_with_complete_provenance_and_real_capacity_costs():
    packets = sources()
    before = pickle.dumps(packets), pickle.dumps(np.random.get_state())
    plan = catalogue.compose_arrived_cue_catalogue(packets, retained_per_cue=2)
    assert [len(b.features) for b in plan.training] == [9, 9, 18, 18, 27, 27]
    assert (plan.arrival_row_exposures, plan.replay_row_exposures, plan.fit_row_exposures) == (
        54,
        54,
        108,
    )
    assert plan.input_array_bytes == 108 * 297 * 8
    assert plan.peak_joint_template_array_bytes == 27 * 297 * 8
    assert plan.peak_bank_rows == 36
    assert plan.peak_bank_observation_target_float64_equivalent_bytes == 36 * 593 * 8
    for i, (batch, wake) in enumerate(zip(plan.training, plan.wakes, strict=True)):
        old_indices = tuple(range(i % 2, i, 2))[-2:]
        expected = tuple(packets[j].episode_id for j in old_indices)
        assert wake.retained_episode_ids == expected
        assert wake.retained_arrival_ticks == tuple(j + 1 for j in old_indices)
        assert all(t <= wake.observed_at for t in wake.retained_arrival_ticks)
        assert [s.episode_id for s in wake.source_spans] == [*expected, packets[i].episode_id]
        assert [(s.start, s.stop) for s in wake.source_spans] == [
            (j * 9, (j + 1) * 9) for j in range(len(expected) + 1)
        ]
        assert wake.replay_rows == 9 * len(expected)
        assert wake.retained_episode_id == (expected[-1] if expected else None)
        assert wake.retained_arrived_at == (old_indices[-1] + 1 if old_indices else None)
        assert (
            batch.episode is None
            and batch.role == "train"
            and len(set(batch.row_ids)) == len(batch.features)
        )
    assert plan.wakes[-1].bank_episode_ids_after == tuple(
        packets[j].episode_id for j in (2, 4, 3, 5)
    )
    assert pickle.dumps(packets) == before[0] and pickle.dumps(np.random.get_state()) == before[1]


def test_should_preserve_default_one_packet_rows_costs_and_scalar_metadata():
    default = catalogue.compose_arrived_cue_catalogue(sources())
    explicit = catalogue.compose_arrived_cue_catalogue(sources(), retained_per_cue=1)
    assert default.retained_per_cue == 1 and default.fit_row_exposures == 90
    assert default.replay_row_exposures == 36 and default.peak_bank_rows == 18
    assert [asdict(w) for w in default.wakes] == [asdict(w) for w in explicit.wakes]
    for a, b in zip(default.training, explicit.training, strict=True):
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.targets, b.targets)


def test_should_evict_oldest_same_public_cue_packet_and_keep_arrays_disjoint():
    packets = tuple(episode(f"packet{i}") for i in range(6))
    plan = catalogue.compose_arrived_cue_catalogue(packets, retained_per_cue=2)
    assert plan.fit_row_exposures == 135 and plan.peak_bank_rows == 18
    for i, wake in enumerate(plan.wakes):
        assert wake.retained_episode_ids == tuple(f"packet{j}" for j in range(max(0, i - 2), i))
        assert wake.bank_episode_ids_after == tuple(
            f"packet{j}" for j in range(max(0, i - 1), i + 1)
        )
    arrays = [a for b in plan.training for a in (b.features, b.targets)]
    assert not any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])
    last = plan.training[-1].features.copy()
    plan.training[0].features.fill(0)
    np.testing.assert_array_equal(plan.training[-1].features, last)


def test_should_route_only_by_public_cue_despite_misleading_context_and_names():
    packets = tuple(episode(f"red-name{i}", context="red", mark=i + 1) for i in range(3))
    plan = catalogue.compose_arrived_cue_catalogue(packets, retained_per_cue=2)
    assert [w.cue for w in plan.wakes] == ["blue"] * 3
    assert plan.wakes[-1].retained_episode_ids == ("red-name0", "red-name1")


@pytest.mark.parametrize("fault", ["early", "late", "rows", "head", "failed"])
def test_should_refuse_incompatible_saved_state_before_owner(monkeypatch, fault):
    saved = state()
    if fault == "early":
        saved = replace(saved, completed_updates=2)
    elif fault == "late":
        saved = replace(saved, completed_updates=6)
    elif fault == "rows":
        saved = replace(saved, training_examples=55)
    elif fault == "head":
        saved = replace(saved, heads=(None, saved.heads[1]))
    else:
        saved = replace(saved, failed_work=(1, 18, 0, 0, True))
    monkeypatch.setattr(study, "ContextLinearLearner", lambda: pytest.fail("owner reached"))
    with pytest.raises(ValueError):
        study.compare_new_c_capacity(sources(), saved)


@pytest.mark.parametrize("fault", ["role", "order", "cue"])
def test_should_refuse_invalid_new_task_source_before_owner(monkeypatch, fault):
    packets = sources()
    if fault == "role":
        packets = (*packets[:-1], replace(packets[-1], role="final_test"))
    elif fault == "order":
        packets = packets[2:] + packets[:2]
    else:
        packets = tuple(episode(e.episode_id, context=e.context) for e in packets)
    monkeypatch.setattr(study, "ContextLinearLearner", lambda: pytest.fail("owner reached"))
    with pytest.raises(ValueError):
        study.compare_new_c_capacity(packets, state())


def test_should_train_only_c_suffix_with_shared_budget_and_score_only_after_both_arms(monkeypatch):
    saved, packets = state(), sources()
    before = pickle.dumps(saved), pickle.dumps(packets), pickle.dumps(np.random.get_state())
    fit = port.fit_categorical_linear
    predict = port.ContextLinearLearner.predict
    fits, forwards = [], []

    def fitting(x, y, classes):
        fits.append(len(x))
        return fit(x, y, classes)

    def predicting(self, x):
        assert fits == [18, 18, 27, 27]
        forwards.append((self.completed_updates, len(x)))
        return predict(self, x)

    monkeypatch.setattr(port, "fit_categorical_linear", fitting)
    monkeypatch.setattr(port.ContextLinearLearner, "predict", predicting)
    result = study.compare_new_c_capacity(packets, saved)
    assert (
        result["status"] == "complete"
        and result["training_complete"]
        and result["source_unchanged"]
    )
    assert (
        result["budget"]["updates_completed"] == 4
        and result["budget"]["replay_examples_completed"] == 0
    )
    assert result["score_calls_completed"] == 14 and result["score_rows_completed"] == 126
    assert set(result["initial_new_task_scores"]) == {e.episode_id for e in packets[4:]}
    for i, (arm, rows) in enumerate(zip(result["arms"], (36, 54), strict=True)):
        assert arm["status"] == "complete" and arm["source_unchanged"] and arm["actor_unchanged"]
        assert (
            arm["stream"]["completed_updates"] == 2 and arm["stream"]["training_examples"] == rows
        )
        assert (
            arm["stream"]["score_forward_calls"] == 0
            and arm["stream"]["evaluation_scope"] == "training_only"
        )
        assert arm["last_checkpoint"]["suffix_updates"] == 2
        native = arm["last_checkpoint"]["state"]
        assert native["completed_updates"] == 6 and native["training_examples"] == 54 + rows
        assert [h["fit_rows"] for h in native["heads"]] == [rows // 2] * 2
        assert arm["catalogue"]["committed_rows"] == dict(
            arrivals=18, replay=rows - 18, native=rows
        )
        assert len(arm["scores"]) == 6
        receipts = arm["committed_receipts"]
        assert [r["update_number"] for r in receipts] == [i * 2 + 1, i * 2 + 2]
        assert [(r["observed_at"], r["arrived_at"], r["applied_at"]) for r in receipts] == [
            (0, 1, 1),
            (1, 2, 2),
        ]
    assert forwards == [(4, 9)] * 2 + [(6, 9)] * 12
    assert (
        pickle.dumps(saved),
        pickle.dumps(packets),
        pickle.dumps(np.random.get_state()),
    ) == before


def test_should_keep_paid_first_arm_and_failed_second_arm_prefix_without_scores(monkeypatch):
    fit = port.fit_categorical_linear
    calls = []

    def fitting(x, y, classes):
        calls.append(len(x))
        if len(calls) == 4:
            raise RuntimeError("deliberate last suffix fit failure")
        return fit(x, y, classes)

    monkeypatch.setattr(port, "fit_categorical_linear", fitting)
    result = study.compare_new_c_capacity(sources(), state())
    assert calls == [18, 18, 27, 27] and result["status"] == "failed"
    assert not result["training_complete"] and result["score_calls_attempted"] == 0
    assert result["budget"]["updates_completed"] == 3
    first, failed = result["arms"]
    assert first["status"] == "complete" and len(first["committed_receipts"]) == 2
    assert failed["status"] == "failed" and len(failed["committed_receipts"]) == 1
    assert failed["last_checkpoint"]["state"]["training_examples"] == 81
    assert failed["catalogue"]["committed_rows"] == dict(arrivals=9, replay=18, native=27)
    assert failed["catalogue"]["failed_attempt_replay_rows_upper_bound"] == 18
    assert failed["failure_work"]["work"]["fit_rows_attempted"] == 27
    assert failed["catalogue"]["failure_work_may_be_unobserved"] is True
    assert first["scores"] == failed["scores"] == {}


@pytest.mark.parametrize("fault", ["counter", "budget"])
def test_should_refuse_changed_native_offset_or_budget_before_accepting_checkpoint(
    monkeypatch, fault
):
    saved = state()

    class BrokenStream:
        def __init__(self, source, *a, **options):
            current = replace(
                saved, completed_updates=5, training_examples=72 if fault == "budget" else 73
            )
            if fault == "budget":
                options["budget"].clock = lambda: 0.0
            options["observe_checkpoint"](1, current, None)
            pytest.fail("incompatible checkpoint accepted")

    monkeypatch.setattr(study, "AttributionStream", BrokenStream)
    result = study.compare_new_c_capacity(sources(), saved)
    assert result["status"] == "failed" and result["score_calls_attempted"] == 0
    assert (
        "counters differ" in result["reason"]
        if fault == "counter"
        else "budget changed" in result["reason"]
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--context-linear-new-c", "--describe"],
        ["--context-linear-new-c", "--fit-updates", "1"],
        ["--context-linear-new-c", "--context-linear-catalogue"],
        ["--context-linear-new-c", "--source", "train"],
        ["--retained-source", "retained", "--describe"],
    ],
)
def test_should_describe_without_source_io_and_refuse_invalid_mode_options(
    monkeypatch, capsys, args
):
    monkeypatch.setattr("sys.argv", ["control", *args])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: pytest.fail("TRAIN opened"))
    monkeypatch.setattr(
        cli, "_qualified_context_catalogue_bytes", lambda *a: pytest.fail("fit opened")
    )
    if args == ["--context-linear-new-c", "--describe"]:
        cli.main()
        result = json.loads(capsys.readouterr().out)
        assert result["new_fit_calls"] == 4 and result["new_fit_rows"] == 90
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2


def test_should_save_failed_new_c_cli_output_exclusively_without_generic_source_read(
    monkeypatch, tmp_path
):
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "control",
            "--context-linear-new-c",
            "--source",
            "train",
            "--retained-source",
            "saved",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        cli, "read_training_sources", lambda *a: pytest.fail("generic decode reached")
    )

    def run(path, saved):
        assert str(path) == "train" and str(saved) == "saved"
        return dict(status="failed", arms=[dict(status="failed")])

    monkeypatch.setattr(cli, "run_saved_context_linear_new_c", run)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and json.loads(output.read_text(encoding="utf8"))["arms"] == [
        dict(status="failed")
    ]
    with pytest.raises(FileExistsError):
        cli.main()


def test_should_bind_real_saved_source_and_model_without_new_fit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    train = root / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    retained = root / "artifacts/runs/arrived-cue-catalogue-20261024/actual.json"
    if not train.is_file() or not retained.is_file():
        pytest.skip("accepted local source unavailable")

    def compare(packets, saved, *, check_boundary):
        assert len(packets) == 6 and saved.completed_updates == 4 and saved.training_examples == 54
        check_boundary()
        return dict(status="scripted_source_binding")

    monkeypatch.setattr(study, "compare_new_c_capacity", compare)
    result = cli.run_saved_context_linear_new_c(train, retained)
    assert result["source_bindings"]["training_sha256"] == cli.QUALIFIED_CAPACITY_TRAIN_SHA
    assert result["source_bindings"]["retained_bytes"] == 522500


def test_should_refuse_declared_source_hash_before_decoding_targets(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="declared saved-model"):
        cli.read_training_sources(path, expected_sha256=cli.QUALIFIED_CAPACITY_TRAIN_SHA)


def test_should_refuse_replaced_prior_evidence_at_suffix_boundary(monkeypatch):
    monkeypatch.setattr(cli, "_qualified_context_catalogue_bytes", lambda *a: b"{}")
    monkeypatch.setattr(cli, "_causal_affine_states", lambda *a, **k: (None, state(), None))
    monkeypatch.setattr(cli, "_qualified_capacity_train_bytes", lambda *a: b"source")
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: sources())

    def compare(*a, check_boundary):
        monkeypatch.setattr(cli, "_qualified_context_catalogue_bytes", lambda *a: b"replaced")
        check_boundary()
        pytest.fail("replacement accepted")

    monkeypatch.setattr(study, "compare_new_c_capacity", compare)
    with pytest.raises(ValueError, match="retained evidence changed"):
        cli.run_saved_context_linear_new_c("train", "retained")
