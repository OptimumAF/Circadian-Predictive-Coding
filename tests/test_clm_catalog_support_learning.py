"""Equal-row full-catalog support recipes through the existing stream and scorer."""

from copy import deepcopy
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_head_learner as native
from src.adapters.clm_catalog_learner import CLMCatalogLearner
from src.app.clm_temporal_sequence import run_temporal_stream
from src.core.clm_catalog import CATALOG_OBJECTIVE, CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_head import CLMHeadPolicy
from src.core.temporal_catalog_support import SUPPORT_QUERY, catalog_training_examples
from src.core.temporal_decision_sequence import CATALOG, PHASES, VOCABULARY
from test_clm_catalog_learning import parts
from test_clm_catalog_support import features


def make_owner(torch, declared):
    source = parts(torch)
    learner = CLMCatalogLearner(
        *source,
        identity=declared,
        policy=CLMHeadPolicy(projection_dim=2, max_updates=12, max_rows=4),
    )
    return learner, source


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("mode", ("repeat", "route"))
def test_should_bind_all_recipe_queries_to_full_active_catalog_and_observed_targets(torch, mode):
    declared, raw = features(torch)
    training, choices = study.build_catalog_training_batches(declared, raw, catalog_support=mode)
    all_ids = tuple(candidate.candidate_id for candidate in CATALOG)
    assert [len(batch.states) for batch, _ in training] == [3] * 4 + [4] * 8
    for index, (batch, targets) in enumerate(training):
        phase = PHASES[index // 4]
        examples = catalog_training_examples(phase, mode, exposure=index % 4)
        ids = tuple(candidate.candidate_id for candidate in examples[0].request.allowed_candidates)
        query_indices = [
            8 if example.request.query == SUPPORT_QUERY else VOCABULARY.index(example.request.query)
            for example in examples
        ]
        assert type(batch) is FrozenCatalogTrainingBatch and type(targets) is CLMCatalogTargets
        assert batch.identity == declared and batch.role == "train"
        assert batch.candidate_ids == targets.candidate_ids == ids
        assert targets.candidate_indices == tuple(
            ids.index(example.target_id) for example in examples
        )
        assert torch.equal(batch.states, raw[query_indices])
        assert torch.equal(batch.actions, raw[[9 + all_ids.index(candidate) for candidate in ids]])
        assert torch.equal(batch.actions, choices[phase].actions)
    diagnostics = study.build_train_choices(training, choices)
    assert [len(diagnostics[phase].states) for phase in PHASES] == [3, 4, 4]
    for phase in PHASES:
        assert diagnostics[phase].role == "train"
        assert torch.equal(diagnostics[phase].actions, choices[phase].actions)
    if mode == "repeat":
        assert torch.equal(training[4][0].states[0], training[4][0].states[-1])
        assert training[4][1].candidate_indices == (1, 0, 0, 1)
    else:
        assert torch.equal(training[4][0].states[-1], raw[8])
        assert training[4][1].candidate_indices == (1, 0, 0, 2)
    original = training[4][0].states.clone()
    raw.zero_()
    assert torch.equal(training[4][0].states, original)


@pytest.mark.parametrize("mode", (None, True, "unknown"))
def test_should_refuse_undeclared_catalog_support_modes_before_building_features(torch, mode):
    declared, raw = features(torch)
    with pytest.raises(ValueError):
        study.build_catalog_training_batches(declared, raw, catalog_support=mode)


@pytest.fixture(scope="module")
def streams(torch):
    declared, raw = features(torch)
    original, rng = raw.clone(), torch.get_rng_state().clone()
    reports, owners, sources, observations = {}, {}, {}, []
    for mode in ("repeat", "route"):
        training, choices = study.build_catalog_training_batches(
            declared, raw, catalog_support=mode
        )
        learner, source = make_owner(torch, declared)
        source_weights = [deepcopy(head.state_dict()) for head in source[:2]]
        sources[mode] = source, source_weights
        report: dict[str, Any] = {}

        def save():
            observations.append(
                (
                    mode,
                    learner.completed_updates,
                    "checkpoints" in report,
                    "train_catalog_diagnostics" in report,
                )
            )

        run_temporal_stream(
            learner,
            training,
            choices,
            report,
            catalog_training=True,
            catalog_support=mode,
            train_choice_batches=study.build_train_choices(training, choices),
            save=save,
        )
        reports[mode], owners[mode] = report, learner
    return dict(
        reports=reports,
        owners=owners,
        sources=sources,
        observations=observations,
        raw=raw,
        original=original,
        rng=rng,
    )


def test_should_finish_both_equal_row_catalog_arms_without_exposing_scores_between_training(
    streams,
):
    assert tuple(streams["reports"]) == ("repeat", "route")
    assert all(
        not scored and not train_scored for _, _, scored, train_scored in streams["observations"]
    )
    for mode, report in streams["reports"].items():
        assert report["sequence_complete"] is True and report["catalog_support"] == mode
        assert report["training_objective"] == CATALOG_OBJECTIVE
        assert (
            report["stream"]["completed_updates"] == 12 and report["stream"]["training_pairs"] == 44
        )
        work = report["learner_work"]
        assert work == streams["owners"][mode].work
        assert work["train_attempts"] == work["commits"] == work["backward_returns"] == 12
        assert work["training_rows_attempted"] == 44
        assert (
            work["catalog_candidate_rows_attempted"] == 36
            and work["catalog_logit_values_attempted"] == 132
        )
        assert work["choice_prediction_attempts"] == work["choice_prediction_returns"] == 8
        assert (
            work["prediction_rows_attempted"] == 31
            and work["choice_candidate_rows_attempted"] == 24
        )
        assert work["snapshot_attempts"] == work["restore_attempts"] == 0
        assert work["unreturned_work_unknown"] is False
        assert [len(point["probabilities"]) for point in report["train_raw_readouts"]] == [3, 4, 4]


def test_should_keep_causal_receipts_indexed_labels_and_the_entire_shared_A_prefix(streams):
    repeat, route = streams["reports"].values()
    for mode, report in streams["reports"].items():
        receipts = report["stream"]["receipts"]
        assert len(receipts) == len(report["rows"]) == 12
        assert [row["observed_at"] for row in receipts] == list(range(0, 24, 2))
        assert (
            [row["arrived_at"] for row in receipts]
            == [row["applied_at"] for row in receipts]
            == list(range(1, 24, 2))
        )
        for index, (receipt, row) in enumerate(zip(receipts, report["rows"], strict=True)):
            phase = PHASES[index // 4]
            examples = catalog_training_examples(phase, mode, exposure=index % 4)
            ids = tuple(
                candidate.candidate_id for candidate in examples[0].request.allowed_candidates
            )
            assert receipt["diagnostic"]["definition"] == CATALOG_OBJECTIVE
            assert row["candidate_ids"] == list(ids)
            assert row["candidate_indices"] == [
                ids.index(example.target_id) for example in examples
            ]
            assert "group_codes" not in row
    assert repeat["rows"][:4] == route["rows"][:4]
    assert repeat["stream"]["receipts"][:4] == route["stream"]["receipts"][:4]
    assert repeat["raw_readouts"][:2] == route["raw_readouts"][:2]
    assert repeat["train_raw_readouts"][0] == route["train_raw_readouts"][0]
    assert repeat["frozen_A_probabilities"] == route["frozen_A_probabilities"]
    left, right = streams["owners"].values()
    assert left is not right and left._bundle[3] is not right._bundle[3]


def test_should_leave_encoder_features_source_head_values_and_caller_rng_unchanged(torch, streams):
    assert torch.equal(streams["raw"], streams["original"])
    assert torch.equal(torch.get_rng_state(), streams["rng"])
    for source, weights in streams["sources"].values():
        for head, expected in zip(source[:2], weights, strict=True):
            assert native._equal(head.state_dict(), expected)


@pytest.fixture(scope="module")
def comparison(streams):
    report = {"arms": deepcopy(streams["reports"])}
    for arm in report["arms"].values():
        arm["control_probabilities"] = {
            version: [[1 / 3] * 3 for _ in range(4)] for version in ("v1", "v2")
        }
    original = deepcopy(report)
    result = study.score_catalog_support_comparison(report)
    assert report == original
    return report, result


def test_should_report_full_train_catalog_margins_and_keep_auxiliary_out_of_development(comparison):
    _, result = comparison
    assert all(result["shared_A_equal"].values())
    for mode, score in result["arm_scores"].items():
        assert score["cached_typed_decodes"] == 88
        diagnostics = score["train_catalog_diagnostics"]
        assert [len(point["rows"]) for point in diagnostics] == [3, 4, 4]
        support = diagnostics[1]["rows"][-1]
        assert support["query"] == ("invoice" if mode == "repeat" else SUPPORT_QUERY)
        assert support["target_id"] == ("inspect" if mode == "repeat" else "route")
        assert support["target_minus_best_alternative"] == pytest.approx(
            support["target_probability"] - support["best_alternative_probability"]
        )
        for point in score["checkpoints"]:
            assert len(point["learned"]["cases"]) == 4
            assert all(row["concept"] != "route_support" for row in point["learned"]["cases"])


def test_should_keep_all_signed_route_minus_repeat_metrics_and_concept_outcomes(comparison):
    _, result = comparison
    assert [point["phase"] for point in result["recipe_contrasts"]] == ["baseline", *PHASES]
    arms = result["arm_scores"]
    for index, point in enumerate(result["recipe_contrasts"]):
        route = arms["route"]["checkpoints"][index]["learned"]
        repeat = arms["repeat"]["checkpoints"][index]["learned"]
        difference = point["route_minus_repeat"]
        for key in ("accuracy", "brier_mean", "ece_5bin"):
            assert difference[key] == pytest.approx(route["metrics"][key] - repeat["metrics"][key])
        if route["metrics"]["nll_mean"] is None or repeat["metrics"]["nll_mean"] is None:
            assert difference["nll_mean"] is None
        else:
            assert difference["nll_mean"] == pytest.approx(
                route["metrics"]["nll_mean"] - repeat["metrics"]["nll_mean"]
            )
        for concepts, key in (
            (("invoice", "legacy", "password"), "current_old_domain_accuracy"),
            (("parcel",), "new_domain_accuracy"),
        ):
            route_accuracy = sum(
                row["correct"] for row in route["cases"] if row["concept"] in concepts
            ) / len(concepts)
            repeat_accuracy = sum(
                row["correct"] for row in repeat["cases"] if row["concept"] in concepts
            ) / len(concepts)
            assert difference[key] == pytest.approx(route_accuracy - repeat_accuracy)
        assert (
            difference["obsolete_rule_error_count"]
            == route["temporal"]["obsolete_rule_errors"]
            - repeat["temporal"]["obsolete_rule_errors"]
        )
        for a, b in zip(route["cases"], repeat["cases"], strict=True):
            assert difference["by_concept"][a["concept"]]["correct"] == int(a["correct"]) - int(
                b["correct"]
            )
            assert difference["by_concept"][a["concept"]]["target_probability"] == pytest.approx(
                a["target_probability"] - b["target_probability"]
            )


@pytest.mark.parametrize(
    "fault",
    (
        "objective",
        "missing_objective",
        "mixed_objectives",
        "partial",
        "query_work",
        "candidate_work",
        "logit_work",
        "receipt",
        "clock",
        "stream_update_type",
        "stream_rows_type",
        "ids",
        "indices",
        "read_work",
        "diagnostic",
        "A_loss",
        "A_receipt",
        "frozen_A",
        "control",
        "development_cache",
    ),
)
def test_should_admit_both_objectives_and_all_second_arm_evidence_before_scoring(
    comparison, monkeypatch, fault
):
    # Copy only the route subtrees this fault mutates; keep accepted cached evidence immutable.
    report = {"arms": dict(comparison[0]["arms"])}
    arm = report["arms"]["route"] = dict(report["arms"]["route"])
    mutated = {
        "mixed_objectives": ("learner_work", "rows", "stream"),
        "query_work": ("learner_work",),
        "candidate_work": ("learner_work",),
        "logit_work": ("learner_work",),
        "receipt": ("stream",),
        "clock": ("stream",),
        "stream_update_type": ("stream",),
        "stream_rows_type": ("stream",),
        "ids": ("rows",),
        "indices": ("rows",),
        "read_work": ("learner_work",),
        "diagnostic": ("train_raw_readouts",),
        "A_loss": ("rows",),
        "A_receipt": ("stream",),
        "frozen_A": ("frozen_A_probabilities",),
        "control": ("control_probabilities",),
        "development_cache": ("raw_readouts",),
    }
    for key in mutated.get(fault, ()):
        arm[key] = deepcopy(arm[key])
    if fault == "objective":
        arm["training_objective"] = native.OBJECTIVE
    elif fault == "missing_objective":
        arm.pop("training_objective")
    elif fault == "mixed_objectives":
        arm.pop("training_objective")
        arm["learner_work"].pop("catalog_candidate_rows_attempted")
        arm["learner_work"].pop("catalog_logit_values_attempted")
        global_ids = tuple(candidate.candidate_id for candidate in CATALOG)
        for index, row in enumerate(arm["rows"]):
            examples = catalog_training_examples(PHASES[index // 4], "route", exposure=index % 4)
            row.pop("candidate_ids")
            row.pop("candidate_indices")
            row["group_codes"] = [global_ids.index(example.target_id) for example in examples]
        for receipt in arm["stream"]["receipts"]:
            receipt["diagnostic"]["definition"] = native.OBJECTIVE
    elif fault == "partial":
        arm["sequence_complete"] = False
    elif fault == "query_work":
        arm["learner_work"]["training_rows_attempted"] -= 1
    elif fault == "candidate_work":
        arm["learner_work"]["catalog_candidate_rows_attempted"] -= 1
    elif fault == "logit_work":
        arm["learner_work"]["catalog_logit_values_attempted"] -= 1
    elif fault == "receipt":
        arm["stream"]["receipts"][-1]["diagnostic"]["definition"] = native.OBJECTIVE
    elif fault == "clock":
        arm["stream"]["receipts"][-1]["applied_at"] += 1
    elif fault == "stream_update_type":
        arm["stream"]["completed_updates"] = 12.0
    elif fault == "stream_rows_type":
        arm["stream"]["training_pairs"] = 44.0
    elif fault == "ids":
        arm["rows"][-1]["candidate_ids"].reverse()
    elif fault == "indices":
        arm["rows"][-1]["candidate_indices"][-1] = 1
    elif fault == "read_work":
        arm["learner_work"]["choice_prediction_returns"] = 7
    elif fault == "diagnostic":
        arm["train_raw_readouts"][-1]["probabilities"].pop()
    elif fault == "A_loss":
        arm["rows"][0]["loss"] += 0.1
    elif fault == "A_receipt":
        arm["stream"]["receipts"][0]["applied_at"] += 1
    elif fault == "frozen_A":
        arm["frozen_A_probabilities"][0][0] += 0.01
        arm["frozen_A_probabilities"][0][1] -= 0.01
    elif fault == "control":
        arm["control_probabilities"]["v1"][0][0] += 0.01
        arm["control_probabilities"]["v1"][0][1] -= 0.01
    else:
        arm["raw_readouts"][-1]["probabilities"][0][0] += 0.1

    def forbidden(*args, **kwargs):
        raise AssertionError("unbound second catalog arm exposed earlier scores")

    monkeypatch.setattr(study, "score_sequence", forbidden)
    monkeypatch.setattr(study, "score_train_choices", forbidden)
    with pytest.raises(ValueError, match="objective" if fault == "mixed_objectives" else None):
        study.score_catalog_support_comparison(report)


@pytest.fixture(scope="module")
def fresh_source(torch):
    declared, _ = features(torch)
    learner, _ = make_owner(torch, declared)
    return learner, declared


@pytest.mark.parametrize(
    "fault",
    (
        "recipe",
        "ids",
        "indices",
        "role",
        "rows",
        "diagnostic",
        "diagnostic_role",
        "report_mode",
        "report_objective",
        "checkpoint",
    ),
)
def test_should_refuse_misbound_recipe_before_any_catalog_prediction(torch, fresh_source, fault):
    learner, declared = fresh_source
    _, raw = features(torch)
    mode = "repeat" if fault == "recipe" else "route"
    training, choices = study.build_catalog_training_batches(declared, raw, catalog_support=mode)
    diagnostics = study.build_train_choices(training, choices)
    report = {}
    if fault == "ids":
        object.__setattr__(
            training[4][0], "candidate_ids", tuple(reversed(training[4][0].candidate_ids))
        )
    elif fault == "indices":
        object.__setattr__(training[4][1], "candidate_indices", (1, 0, 0, 1))
    elif fault == "role":
        object.__setattr__(training[4][0], "role", "probe_train")
    elif fault == "rows":
        object.__setattr__(training[4][0], "states", training[4][0].states[:3])
    elif fault == "diagnostic":
        object.__setattr__(diagnostics["after_B"], "states", diagnostics["after_B"].states[:3])
    elif fault == "diagnostic_role":
        object.__setattr__(diagnostics["after_B"], "role", "outer_selection")
    elif fault == "report_mode":
        report["catalog_support"] = "repeat"
    elif fault == "report_objective":
        report["training_objective"] = native.OBJECTIVE
    before = learner.work
    extra = {"on_checkpoint": lambda *args: None} if fault == "checkpoint" else {}
    with pytest.raises(ValueError):
        run_temporal_stream(
            learner,
            training,
            choices,
            report,
            catalog_training=True,
            catalog_support="route",
            train_choice_batches=diagnostics,
            **extra,
        )
    assert learner.work == before and learner.completed_updates == 0


def test_should_capture_first_paid_catalog_receipt_and_refuse_changed_prefix_before_more_work(
    torch,
):
    declared, raw = features(torch)
    training, choices = study.build_catalog_training_batches(declared, raw, catalog_support="route")
    learner, _ = make_owner(torch, declared)
    report: dict[str, Any] = {}

    def save():
        if learner.completed_updates == 1:
            report["rows"][0]["candidate_indices"][0] = 0

    with pytest.raises(ValueError, match="prefix"):
        run_temporal_stream(
            learner,
            training,
            choices,
            report,
            catalog_training=True,
            catalog_support="route",
            save=save,
        )
    assert report["sequence_complete"] is False and learner.completed_updates == 1
    assert report["stream"]["completed_updates"] == 1 and report["stream"]["training_pairs"] == 3
    assert len(report["stream"]["receipts"]) == 1
    assert report["stream"]["receipts"][0]["diagnostic"]["definition"] == CATALOG_OBJECTIVE
    assert report["learner_work"]["train_attempts"] == report["learner_work"]["commits"] == 1
    assert report["learner_work"]["catalog_candidate_rows_attempted"] == 3
    assert report["learner_work"]["catalog_logit_values_attempted"] == 9
    assert report["learner_work"]["choice_prediction_returns"] == 1
    paid = deepcopy(report["learner_work"])
    with pytest.raises(ValueError):
        run_temporal_stream(
            learner, training, choices, report, catalog_training=True, catalog_support="route"
        )
    assert report["learner_work"] == paid


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (
            ["--catalog-training", "--catalog-support", "repeat"],
            {"catalog_training": True, "catalog_support": "repeat"},
        ),
        (
            ["--catalog-training", "--catalog-support", "route"],
            {"catalog_training": True, "catalog_support": "route"},
        ),
        (
            ["--catalog-training", "--compare-catalog-support"],
            {"catalog_training": True, "compare_catalog_support": True},
        ),
    ],
)
def test_should_route_explicit_catalog_recipe_cli_options(
    monkeypatch, tmp_path, arguments, expected
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
            *arguments,
        ],
    )
    calls = []

    def case(model_dir, head, report, *, train_catalog_diagnostics, save, **kwargs):
        calls.append(kwargs)
        report.update(complete=True, fixture="metadata only")
        save()

    monkeypatch.setattr(study, "run_case", case)
    study.main()
    assert calls == [expected]


@pytest.mark.parametrize(
    "other",
    (
        ["--catalog-support", "repeat", "--reserved-task-probes"],
        ["--catalog-support", "route", "--reserved-task-probes"],
        ["--compare-catalog-support", "--reserved-task-probes"],
        ["--catalog-support", "route", "--compare-catalog-support"],
    ),
)
def test_should_refuse_incompatible_catalog_recipe_cli_before_any_loader(
    monkeypatch, tmp_path, other
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
            *other,
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible recipe CLI reached native loader")

    monkeypatch.setattr(study, "run_case", forbidden)
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2
