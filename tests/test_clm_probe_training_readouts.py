"""Observe arrived TRAIN on disposable probes without replaying or selecting it."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_checkpoint_probes as checkpoint
import src.adapters.clm_head_learner as native
import src.app.clm_reserved_probe as probe_app
from src.adapters.clm_catalog_probe import CLMCatalogProbe
from src.adapters.clm_frozen_encoder import text_recipe
from src.core.clm_catalog import FrozenCatalogTrainingBatch
from src.core.experience import ExperiencePermissions
from src.core.temporal_decision_sequence import CHECKPOINTS, temporal_examples
from src.infra.clm_local_loader import LoadedCLM
from test_clm_catalog_learning import identity, owner, parts
from test_clm_frozen_encoder import FakeTokenizer, backbone
from test_clm_probe_rehearsal import budget, data
from test_clm_typed_stream_readout import LiteralSampler


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def factory(source):
    return CLMCatalogProbe(
        source, policy=replace(source.policy, max_updates=6, max_restores=0), max_readouts=8
    )


@pytest.fixture(scope="module")
def full_case(torch):
    state_texts, action_texts = study.feature_texts(include_reserved_probe=True)
    state, action, scale = parts(torch)
    loaded = LoadedCLM(
        backbone(torch),
        FakeTokenizer({text: [index + 1] for index, text in enumerate(state_texts + action_texts)}),
        state,
        action,
        scale,
        replace(identity(), recipe=text_recipe(2048)),
        {},
    )
    original = (
        deepcopy(loaded.state_head.state_dict()),
        deepcopy(loaded.action_head.state_dict()),
        loaded.logit_scale.detach().clone(),
    )
    rng = torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    observed = []
    training_reads = []
    real_probe = probe_app.run_reserved_probe
    real_predict = CLMCatalogProbe.predict_choices

    def checked(source, *args, **kwargs):
        before, work = source._capture(), source.work
        owners = tuple(source._owners)
        assert report["sequence_complete"] is False
        assert "checkpoints" not in report and "reserved_probe_scores" not in report
        pair = kwargs.get("training_readout")

        def predict(probe, choice):
            if choice.role == "train":
                assert pair is not None
                assert choice.states is pair[0].states and choice.actions is pair[0].actions
                assert choice.identity is pair[0].identity
                training_reads.append((kwargs["phase"], probe.completed_updates))
            return real_predict(probe, choice)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(CLMCatalogProbe, "predict_choices", predict)
            result = real_probe(source, *args, **kwargs)
        assert native._equal(vars(before), vars(source._capture()))
        assert source.work == work
        assert all(left is right for left, right in zip(owners, source._owners, strict=True))
        observed.append((kwargs["phase"], source.completed_updates, kwargs.get("training_readout")))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "load_local_clm", lambda *args, **kwargs: loaded)
        patch.setattr(study, "ProcessRssSampler", LiteralSampler)
        patch.setattr(probe_app, "run_reserved_probe", checked)
        study.run_case(
            Path("literal-model"),
            Path("literal-head"),
            report,
            catalog_training=True,
            reserved_task_probes=True,
            probe_rehearsal=True,
            probe_train_readouts=True,
        )
    return dict(
        report=report,
        observed=observed,
        training_reads=training_reads,
        loaded=loaded,
        original=original,
        rng=rng,
    )


def test_should_observe_original_arrived_train_without_new_updates_or_main_changes(
    torch, full_case
):
    report = full_case["report"]
    assert report["complete"] is True and report["sequence_complete"] is True
    assert report["probe_train_readouts"] is True and report["probe_rehearsal"] is True
    assert report["stream"]["completed_updates"] == 12
    assert report["stream"]["training_pairs"] == 36
    assert report["learner_work"]["train_attempts"] == 12
    assert report["learner_work"]["catalog_logit_values_attempted"] == 108
    assert [(phase, update) for phase, update, _ in full_case["observed"]] == [
        (phase, update) for update, phase in CHECKPOINTS
    ]
    assert full_case["observed"][0][2] is None
    assert full_case["training_reads"] == [
        (phase, update) for _, phase in CHECKPOINTS[1:] for update in (0, 6)
    ]
    baseline, *children = report["reserved_probes"]
    assert "arrived_train_readouts" not in baseline
    assert baseline["probe_work"] == probe_app._expected_work(6, 6, 16, catalog_training=True)
    for child, (_, _, pair) in zip(children, full_case["observed"][1:], strict=True):
        batch, targets = pair
        assert type(batch) is FrozenCatalogTrainingBatch and batch.role == "train"
        assert child["complete"] is True and child["stream"]["completed_updates"] == 6
        assert child["stream"]["training_pairs"] == 24
        assert child["probe_work"] == probe_app._expected_work(
            6, 8, 22, catalog_training=True, rehearsal_rows=2
        )
        assert child["rehearsal"]["replay_rows_attempted"] == 12
        record = child["arrived_train_readouts"]
        assert set(record) == {
            "source_phase",
            "source_role",
            "source_row_indices",
            "candidate_ids",
            "candidate_indices",
            "retained_tensor_storage_bytes",
            "readouts",
        }
        assert record["source_phase"] == child["phase"]
        assert record["source_role"] == "train" and record["source_row_indices"] == [0, 1, 2]
        assert record["candidate_ids"] == list(batch.candidate_ids)
        assert record["candidate_indices"] == list(targets.candidate_indices)
        assert type(record["retained_tensor_storage_bytes"]) is int
        assert 1 <= record["retained_tensor_storage_bytes"] <= 1024
        assert [row["point"] for row in record["readouts"]] == ["before", "after"]
        for row in record["readouts"]:
            assert set(row) == {"point", "probabilities"}
            assert len(row["probabilities"]) == 3
            assert all(
                len(values) == 3 and abs(sum(values) - 1) <= 1e-6 for values in row["probabilities"]
            )
        assert [(row["point"], row["group"]) for row in child["readouts"]] == [
            (point, group)
            for point in ("before", "after")
            for group in ("probe_train", "probe_test", "old_task")
        ]
    assert study._control_matches(torch, full_case["loaded"], full_case["original"])
    assert torch.equal(torch.get_rng_state(), full_case["rng"])


def test_should_score_train_separately_from_suffix_heldout_only_after_all_work(full_case):
    report = full_case["report"]
    saved = deepcopy(report)
    scores = study.score_reserved_probes(report)
    assert report == saved and scores == report["reserved_probe_scores"]
    assert scores["cached_probability_rows"] == 82
    assert len(scores["checkpoints"]) == 4
    for point in scores["checkpoints"]:
        groups = point["groups"]
        expected = ("probe_train", "probe_test", "old_task")
        assert tuple(groups) == expected + (
            () if point["phase"] == "baseline" else ("arrived_train",)
        )
        if point["phase"] == "baseline":
            continue
        group = groups["arrived_train"]
        examples = temporal_examples(point["phase"], "train")
        for endpoint in ("before", "after"):
            cases = group[endpoint]["cases"]
            assert len(cases) == 3
            assert [case["role"] for case in cases] == ["train"] * 3
            assert [case["target_id"] for case in cases] == [item.target_id for item in examples]
            assert [case["request"] for case in cases] != [
                case["request"] for case in groups["old_task"][endpoint]["cases"]
            ]
        assert set(group["after_minus_before"]) == {
            "accuracy",
            "nll_mean",
            "nll_status",
            "brier_mean",
            "ece_5bin",
            "obsolete_rule_error_count",
        }


@pytest.mark.parametrize(
    "fault",
    (
        "missing_record",
        "phase",
        "role",
        "row_order",
        "boolean_index",
        "ids",
        "labels",
        "storage",
        "negative_storage",
        "extra",
        "point",
        "rows",
        "nan",
        "probability",
        "missing_endpoint",
        "extra_endpoint",
        "extra_read",
        "work_queries",
        "work_candidates",
        "partial",
        "mode",
        "missing_mode",
        "baseline_record",
        "main_partial",
    ),
)
def test_should_refuse_corrupt_training_readouts_before_scoring(full_case, monkeypatch, fault):
    report = deepcopy(full_case["report"])
    child = report["reserved_probes"][-1]
    record = child["arrived_train_readouts"]
    if fault == "missing_record":
        child.pop("arrived_train_readouts")
    elif fault == "phase":
        record["source_phase"] = "after_B"
    elif fault == "role":
        record["source_role"] = "probe_test"
    elif fault == "row_order":
        record["source_row_indices"].reverse()
    elif fault == "boolean_index":
        record["source_row_indices"][0] = False
    elif fault == "ids":
        record["candidate_ids"].reverse()
    elif fault == "labels":
        record["candidate_indices"][0] = (record["candidate_indices"][0] + 1) % 3
    elif fault == "storage":
        record["retained_tensor_storage_bytes"] = True
    elif fault == "negative_storage":
        record["retained_tensor_storage_bytes"] = -1
    elif fault == "extra":
        record["replay_rows_attempted"] = 0
    elif fault == "point":
        record["readouts"][0]["point"] = "after"
    elif fault == "rows":
        record["readouts"][0]["probabilities"].pop()
    elif fault == "nan":
        record["readouts"][0]["probabilities"][0][0] = float("nan")
    elif fault == "probability":
        record["readouts"][0]["probabilities"][0] = [1.0, 1.0, 1.0]
    elif fault == "missing_endpoint":
        record["readouts"].pop()
    elif fault == "extra_endpoint":
        record["readouts"].append(deepcopy(record["readouts"][-1]))
    elif fault == "extra_read":
        child["readouts"].append(deepcopy(child["readouts"][-1]))
    elif fault == "work_queries":
        child["probe_work"]["prediction_rows_attempted"] = 16
    elif fault == "work_candidates":
        child["probe_work"]["choice_candidate_rows_attempted"] = 18
    elif fault == "partial":
        child["complete"] = False
    elif fault == "mode":
        report["probe_train_readouts"] = False
    elif fault == "missing_mode":
        report.pop("probe_train_readouts")
    elif fault == "baseline_record":
        report["reserved_probes"][0]["arrived_train_readouts"] = deepcopy(record)
    else:
        report["sequence_complete"] = False

    def forbidden(*args, **kwargs):
        raise AssertionError("corrupt training readout exposed quality")

    monkeypatch.setattr(study, "_reserved_probe_metrics", forbidden)
    with pytest.raises(ValueError):
        study.score_reserved_probes(report)


@pytest.fixture(scope="module")
def source_after_a(torch):
    _, _, benchmark, _, _ = data(torch)
    source, _ = owner(torch, maximum=12)
    for batch, targets in benchmark[:4]:
        source.train_batch(batch, targets)
    return source


@pytest.mark.parametrize("mode", ("ordinary", "marker_control"))
def test_should_allow_train_observation_without_rehearsal_or_replay_consent(
    torch, source_after_a, mode
):
    training, readouts, _, _, arrived = data(torch)
    extra = {}
    if mode == "marker_control":
        extra["marker_control"] = checkpoint.build_probe_marker_control(
            training["after_A"], phase="after_A"
        )
    report: dict[str, Any] = {}
    before, work = source_after_a._capture(), source_after_a.work
    probe_app.run_reserved_probe(
        source_after_a,
        training["after_A"],
        readouts["after_A"],
        report,
        phase="after_A",
        probe_factory=factory,
        permissions=ExperiencePermissions(training=True),
        budget=budget(),
        catalog_training=True,
        training_readout=arrived["after_A"],
        **extra,
    )
    assert report["complete"] is True and "rehearsal" not in report
    assert report["probe_work"] == probe_app._expected_work(
        6, 8, 22, catalog_training=True, extra_marker_rows=2 if extra else 0
    )
    assert report["arrived_train_readouts"]["source_role"] == "train"
    assert source_after_a.work == work
    assert native._equal(vars(before), vars(source_after_a._capture()))


@pytest.mark.parametrize(
    "fault",
    ("baseline", "paired", "list", "role", "labels", "identity", "actions", "shape"),
)
def test_should_refuse_unbound_training_readout_before_probe_creation(torch, source_after_a, fault):
    training, readouts, _, _, arrived = data(torch)
    pair, phase, catalog_training = arrived["after_A"], "after_A", True
    if fault == "baseline":
        phase = "baseline"
    elif fault == "paired":
        catalog_training = False
    elif fault == "list":
        pair = list(pair)
    elif fault == "role":
        object.__setattr__(pair[0], "role", "probe_test")
    elif fault == "labels":
        object.__setattr__(pair[1], "candidate_indices", (0, 0, 0))
    elif fault == "identity":
        object.__setattr__(pair[0], "identity", replace(identity(), revision="foreign"))
    elif fault == "actions":
        pair[0].actions.add_(0.125)
    else:
        object.__setattr__(pair[0], "states", pair[0].states[:2])

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted training readout constructed probe")

    with pytest.raises(ValueError):
        probe_app.run_reserved_probe(
            source_after_a,
            training[phase],
            readouts[phase],
            {},
            phase=phase,
            probe_factory=forbidden,
            permissions=ExperiencePermissions(training=True),
            budget=budget(),
            catalog_training=catalog_training,
            training_readout=pair,
        )


@pytest.mark.parametrize("fault", ("states", "indices_owner", "identity_owner"))
def test_should_refuse_changed_training_evidence_after_one_paid_update(
    torch, source_after_a, fault
):
    training, readouts, _, _, arrived = data(torch)
    pair = arrived["after_A"]
    report: dict[str, Any] = {}
    session = budget()
    work = source_after_a.work

    def save():
        if report.get("stream", {}).get("completed_updates") == 1:
            if fault == "states":
                pair[0].states.add_(0.25)
            elif fault == "indices_owner":
                object.__setattr__(
                    pair[1], "candidate_indices", tuple(list(pair[1].candidate_indices))
                )
            else:
                object.__setattr__(pair[0], "identity", replace(pair[0].identity))

    with pytest.raises(ValueError):
        probe_app.run_reserved_probe(
            source_after_a,
            training["after_A"],
            readouts["after_A"],
            report,
            phase="after_A",
            probe_factory=factory,
            permissions=ExperiencePermissions(training=True),
            budget=session,
            catalog_training=True,
            training_readout=pair,
            save=save,
        )
    assert report["complete"] is False
    assert report["stream"]["completed_updates"] == session.updates_completed == 1
    assert report["probe_work"]["commits"] == 1
    assert len(report["stream"]["receipts"]) == 1
    assert report["stream"]["training_pairs"] == 2
    assert source_after_a.work == work


@pytest.mark.parametrize(
    "flags",
    (
        [],
        ["--catalog-training"],
        ["--reserved-task-probes"],
        ["--catalog-training", "--reserved-task-probes", "--catalog-support", "repeat"],
        ["--catalog-training", "--reserved-task-probes", "--compare-catalog-support"],
        ["--catalog-training", "--reserved-task-probes", "--typed-development-readout"],
    ),
)
def test_should_refuse_invalid_train_readout_cli_before_loading(monkeypatch, tmp_path, flags):
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(tmp_path / "new.json"),
            "--probe-train-readouts",
            *flags,
        ],
    )
    monkeypatch.setattr(
        study, "load_local_clm", lambda *args, **kwargs: pytest.fail("invalid CLI loaded")
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and not (tmp_path / "new.json").exists()


@pytest.mark.parametrize("mode", (None, "probe-rehearsal", "probe-marker-control"))
def test_should_route_train_readout_cli_with_existing_modes(monkeypatch, tmp_path, mode):
    calls = []
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(tmp_path / "new.json"),
            "--catalog-training",
            "--reserved-task-probes",
            "--probe-train-readouts",
            *([] if mode is None else ["--" + mode]),
        ],
    )

    def execute(model_dir, head, report, *, train_catalog_diagnostics, save, **kwargs):
        calls.append(kwargs)
        report.update(complete=True)
        save()

    monkeypatch.setattr(study, "run_case", execute)
    study.main()
    expected = dict(catalog_training=True, reserved_task_probes=True, probe_train_readouts=True)
    if mode:
        expected[mode.replace("-", "_")] = True
    assert calls == [expected]


@pytest.mark.parametrize(
    "options",
    (
        {},
        {"catalog_training": True},
        {"reserved_task_probes": True},
        {"catalog_training": True, "reserved_task_probes": True, "catalog_support": "route"},
        {"catalog_training": True, "reserved_task_probes": True, "compare_catalog_support": True},
        {"catalog_training": True, "reserved_task_probes": True, "typed_development_readout": True},
        {"catalog_training": True, "reserved_task_probes": True, "probe_train_readouts": 1},
    ),
)
def test_should_refuse_invalid_train_readout_api_before_loading(monkeypatch, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid TRAIN readout API loaded model")

    monkeypatch.setattr(study, "load_local_clm", forbidden)
    with pytest.raises(ValueError):
        study.run_case(
            Path("unused"), Path("unused"), {}, **{"probe_train_readouts": True, **options}
        )
