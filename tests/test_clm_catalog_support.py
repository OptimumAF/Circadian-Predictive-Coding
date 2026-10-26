"""Declared catalog-support recipes and equal-dose tiny native stream checks."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
from src.app.clm_temporal_sequence import run_temporal_stream
from src.core.clm_head import FrozenEncoderIdentity
from src.core.temporal_catalog_support import (
    SUPPORT_QUERY,
    CatalogSupportExample,
    catalog_training_examples,
)
from src.core.temporal_decision_sequence import CATALOG, PHASES, temporal_examples
from test_clm_temporal_sequence import owner


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def features(torch):
    declared = FrozenEncoderIdentity(
        "fake", "literal", "literal", "last", "catalog-support-literal", 2
    )
    values = torch.tensor(
        [[(index + 1) / 14, (-1) ** index * (index + 2) / 15] for index in range(13)],
        dtype=torch.float32,
    )
    return declared, values


@pytest.mark.parametrize("mode", ["none", "repeat", "route"])
def test_should_preserve_original_three_examples_and_explicitly_declare_the_fourth(mode):
    for phase in PHASES:
        original = temporal_examples(phase, "train", exposure=2)
        selected = catalog_training_examples(phase, mode, exposure=2)
        assert selected[:3] == original
        if phase == "after_A" or mode == "none":
            assert selected == original
        elif mode == "repeat":
            assert len(selected) == 4 and selected[3] is selected[0]
            assert selected[3].target_id == "inspect" and selected[3].request.query == "invoice"
        else:
            support = selected[3]
            assert type(support) is CatalogSupportExample
            assert support.request.query == SUPPORT_QUERY == "original workflow"
            assert support.request.request_id == f"train-{phase}-2-route-support"
            assert support.target_id == "route" and support.obsolete_target_id is None
            assert support.domain == "auxiliary" and support.concept == "route_support"
            assert support.role == "train" and support.source_group == "train-catalog-route"
            assert support.policy_version == "v2"
            assert support.request.allowed_candidates == original[0].request.allowed_candidates
            assert support.request.query not in {example.request.query for example in original}


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "after_A"),
        ("exposure", True),
        ("role", "probe_train"),
        ("target_id", "inspect"),
        ("policy_version", "v1"),
        ("source_group", "train-token"),
        ("obsolete_target_id", "route"),
    ],
)
def test_should_refuse_relabeling_or_heldout_auxiliary_records(field, value):
    support = catalog_training_examples("after_B", "route")[3]
    with pytest.raises(ValueError):
        replace(support, **{field: value})


@pytest.mark.parametrize("mode", [None, True, "unknown"])
def test_should_refuse_unknown_or_coerced_support_modes(mode):
    with pytest.raises(ValueError):
        catalog_training_examples("after_B", mode)


def test_should_keep_default_texts_and_add_one_explicit_shared_auxiliary_template():
    original, actions = study.feature_texts()
    selected, selected_actions = study.feature_texts(include_catalog_support=True)
    assert len(original) == 8 and len(selected) == 9
    assert selected[:8] == original and selected_actions == actions
    assert selected[8] == SUPPORT_QUERY + "\n\nChoose the ticket workflow tool."
    assert actions == tuple(item.description for item in CATALOG)


@pytest.mark.parametrize(
    "mode,last_codes",
    [("repeat", ((1, 0, 0, 1), (1, 0, 1, 1))), ("route", ((1, 0, 0, 3), (1, 0, 1, 3)))],
)
def test_should_bind_selected_packet_shapes_labels_and_auxiliary_feature_index(
    torch, mode, last_codes
):
    declared, raw = features(torch)
    training, choices = study.build_feature_batches(declared, raw, catalog_support=mode)
    assert [len(pair[0].states) for pair in training] == [3] * 4 + [4] * 8
    assert (
        training[4][1].group_codes == last_codes[0] and training[8][1].group_codes == last_codes[1]
    )
    ids = [item.candidate_id for item in CATALOG]
    for index, (batch, targets) in enumerate(training):
        phase = PHASES[index // 4]
        examples = catalog_training_examples(phase, mode, exposure=index % 4)
        expected_indices = [
            8
            if example.request.query == SUPPORT_QUERY
            else ("invoice", "legacy", "parcel", "password").index(example.request.query)
            for example in examples
        ]
        assert torch.equal(batch.states, raw[expected_indices])
        assert torch.equal(
            batch.actions, raw[[9 + ids.index(example.target_id) for example in examples]]
        )
        assert targets.group_codes == tuple(ids.index(example.target_id) for example in examples)
    diagnostics = study.build_train_choices(training, choices)
    assert [len(diagnostics[phase].states) for phase in PHASES] == [3, 4, 4]
    for phase in PHASES:
        assert torch.equal(diagnostics[phase].actions, choices[phase].actions)
    before = training[4][0].states.clone()
    raw.zero_()
    assert torch.equal(training[4][0].states, before)


@pytest.fixture(scope="module")
def support_streams(torch):
    declared, raw = features(torch)
    original = raw.clone()
    rng = torch.get_rng_state().clone()
    reports, owners, sources, observations = {}, {}, {}, []
    for mode in ("repeat", "route"):
        training, choices = study.build_feature_batches(declared, raw, catalog_support=mode)
        learner, state, action = owner(torch, declared)
        sources[mode] = (state, action, deepcopy(state.state_dict()), deepcopy(action.state_dict()))
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
            train_choice_batches=study.build_train_choices(training, choices),
            catalog_support=mode,
            save=save,
        )
        reports[mode], owners[mode] = report, learner
    return dict(
        reports=reports,
        owners=owners,
        sources=sources,
        raw=raw,
        original=original,
        rng=rng,
        observations=observations,
    )


def test_should_run_two_independent_equal_dose_streams_without_scoring_between_arms(
    torch, support_streams
):
    case = support_streams
    assert tuple(case["reports"]) == ("repeat", "route")
    assert all(
        not scored and not train_scored for _, _, scored, train_scored in case["observations"]
    )
    for mode, report in case["reports"].items():
        assert report["catalog_support"] == mode and report["sequence_complete"] is True
        assert (
            report["stream"]["completed_updates"] == 12 and report["stream"]["training_pairs"] == 44
        )
        assert len(report["rows"]) == len(report["stream"]["receipts"]) == 12
        assert [len(row["probabilities"]) for row in report["train_raw_readouts"]] == [3, 4, 4]
        assert report["learner_work"]["train_attempts"] == report["learner_work"]["commits"] == 12
        assert report["learner_work"]["training_rows_attempted"] == 44
        assert (
            report["learner_work"]["choice_prediction_attempts"]
            == report["learner_work"]["choice_prediction_returns"]
            == 8
        )
        assert report["learner_work"]["prediction_rows_attempted"] == 31
        assert report["learner_work"]["choice_candidate_rows_attempted"] == 24
    assert torch.equal(case["raw"], case["original"]) and torch.equal(
        torch.get_rng_state(), case["rng"]
    )
    for state, action, initial_state, initial_action in case["sources"].values():
        assert all(
            torch.equal(value, initial_state[name]) for name, value in state.state_dict().items()
        )
        assert all(
            torch.equal(value, initial_action[name]) for name, value in action.state_dict().items()
        )


def test_should_match_the_entire_common_A_prefix_before_interpreting_v2_changes(support_streams):
    repeat, route = support_streams["reports"].values()
    assert repeat["rows"][:4] == route["rows"][:4]
    assert repeat["stream"]["receipts"][:4] == route["stream"]["receipts"][:4]
    assert repeat["raw_readouts"][:2] == route["raw_readouts"][:2]
    assert repeat["frozen_A_probabilities"] == route["frozen_A_probabilities"]
    assert repeat["train_raw_readouts"][0] == route["train_raw_readouts"][0]
    for point in repeat["train_raw_readouts"][1:]:
        assert point["probabilities"][0] == point["probabilities"][-1]
    first = support_streams["owners"]["repeat"]
    second = support_streams["owners"]["route"]
    assert first is not second and first._bundle[3] is not second._bundle[3]
    for left, right in zip(first._bundle[:2], second._bundle[:2]):
        for a, b in zip(left.parameters(), right.parameters()):
            assert a is not b and a.data_ptr() != b.data_ptr()


def test_should_keep_none_recipe_shape_and_refuse_wrong_feature_union(torch):
    declared, raw = features(torch)
    original = torch.cat((raw[:8], raw[9:]))
    training, _ = study.build_feature_batches(declared, original)
    assert all(len(batch.states) == 3 for batch, _ in training)
    assert training[4][1].group_codes == (1, 0, 0)
    with pytest.raises(ValueError):
        study.build_feature_batches(declared, raw)
    for mode in ("repeat", "route"):
        with pytest.raises(ValueError):
            study.build_feature_batches(declared, original, catalog_support=mode)


@pytest.fixture(scope="module")
def fresh_source(torch):
    declared, raw = features(torch)
    learner, *_ = owner(torch, declared)
    return learner, declared, raw


@pytest.mark.parametrize(
    "fault",
    ["unknown_mode", "wrong_recipe", "missing_support", "relabel", "diagnostic_shape", "role"],
)
def test_should_refuse_misbound_support_packets_before_any_native_work(torch, fresh_source, fault):
    learner, declared, raw = fresh_source
    mode = "repeat" if fault == "wrong_recipe" else "route"
    training, choices = study.build_feature_batches(declared, raw, catalog_support=mode)
    diagnostics = study.build_train_choices(training, choices)
    if fault == "missing_support":
        object.__setattr__(training[4][0], "states", training[4][0].states[:3])
        object.__setattr__(training[4][0], "actions", training[4][0].actions[:3])
        object.__setattr__(training[4][1], "group_codes", training[4][1].group_codes[:3])
    elif fault == "relabel":
        object.__setattr__(training[4][1], "group_codes", (1, 0, 0, 1))
    elif fault == "diagnostic_shape":
        object.__setattr__(diagnostics["after_B"], "states", diagnostics["after_B"].states[:3])
    elif fault == "role":
        object.__setattr__(training[4][0], "role", "probe_train")
    before = learner.work
    with pytest.raises(ValueError):
        run_temporal_stream(
            learner,
            training,
            choices,
            {},
            catalog_support="unknown" if fault == "unknown_mode" else "route",
            train_choice_batches=diagnostics,
        )
    assert learner.work == before and learner.completed_updates == 0


@pytest.fixture(scope="module")
def comparison(support_streams):
    arms = deepcopy(support_streams["reports"])
    for report in arms.values():
        report["control_probabilities"] = {
            version: [[1 / 3] * 3 for _ in range(4)] for version in ("v1", "v2")
        }
    report = {"arms": arms}
    original = deepcopy(report)
    score = study.score_sequence
    calls = []

    def checked_score(arm):
        assert all(
            learner.completed_updates == 12 for learner in support_streams["owners"].values()
        )
        calls.append(arm["catalog_support"])
        return score(arm)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "score_sequence", checked_score)
        scores = study.score_catalog_support_comparison(report)
    assert report == original and calls == ["repeat", "route"]
    saved = deepcopy(scores)
    scores["arm_scores"]["route"]["train_catalog_diagnostics"][1]["rows"][-1][
        "probabilities"
    ].clear()
    assert report == original
    return report, saved


def test_should_score_both_completed_arms_and_keep_auxiliary_out_of_development(comparison):
    _, scores = comparison
    assert all(scores["shared_A_equal"].values())
    assert tuple(scores["arm_scores"]) == ("repeat", "route")
    for mode, arm in scores["arm_scores"].items():
        assert arm["cached_typed_decodes"] == 88
        assert [len(point["rows"]) for point in arm["train_catalog_diagnostics"]] == [3, 4, 4]
        for point in arm["checkpoints"]:
            assert len(point["learned"]["cases"]) == 4
            assert all(row["concept"] != "route_support" for row in point["learned"]["cases"])
        support = arm["train_catalog_diagnostics"][1]["rows"][-1]
        if mode == "repeat":
            assert support["query"] == "invoice" and support["target_id"] == "inspect"
        else:
            assert support["query"] == SUPPORT_QUERY and support["target_id"] == "route"
            assert support["obsolete_target_id"] is None


def test_should_report_signed_route_minus_repeat_differences_without_endpoint_selection(comparison):
    _, scores = comparison
    arms = scores["arm_scores"]
    assert [point["phase"] for point in scores["recipe_contrasts"]] == ["baseline", *PHASES]
    for index, point in enumerate(scores["recipe_contrasts"]):
        route = arms["route"]["checkpoints"][index]["learned"]
        repeat = arms["repeat"]["checkpoints"][index]["learned"]
        observed = point["route_minus_repeat"]
        for key in ("accuracy", "brier_mean", "ece_5bin"):
            assert observed[key] == pytest.approx(route["metrics"][key] - repeat["metrics"][key])
        if route["metrics"]["nll_mean"] is None or repeat["metrics"]["nll_mean"] is None:
            assert observed["nll_mean"] is None
        else:
            assert observed["nll_mean"] == pytest.approx(
                route["metrics"]["nll_mean"] - repeat["metrics"]["nll_mean"]
            )
        for a, b in zip(route["cases"], repeat["cases"]):
            assert observed["by_concept"][a["concept"]]["correct"] == int(a["correct"]) - int(
                b["correct"]
            )
            assert observed["by_concept"][a["concept"]]["target_probability"] == pytest.approx(
                a["target_probability"] - b["target_probability"]
            )


@pytest.mark.parametrize(
    "fault",
    [
        "order",
        "partial",
        "work",
        "missing_diag",
        "A_loss",
        "A_receipt",
        "A_readout",
        "frozen_A",
        "control",
    ],
)
def test_should_refuse_unbound_comparison_before_scoring_either_arm(comparison, monkeypatch, fault):
    report = deepcopy(comparison[0])
    route = report["arms"]["route"]
    if fault == "order":
        report["arms"] = dict(reversed(tuple(report["arms"].items())))
    elif fault == "partial":
        route["sequence_complete"] = False
    elif fault == "work":
        route["learner_work"]["choice_prediction_returns"] = 7
    elif fault == "missing_diag":
        route["train_raw_readouts"].pop()
    elif fault == "A_loss":
        route["rows"][0]["loss"] += 0.1
    elif fault == "A_receipt":
        route["stream"]["receipts"][0]["applied_at"] += 1
    elif fault == "A_readout":
        route["raw_readouts"][1]["probabilities"][0][0] += 0.001
        route["raw_readouts"][1]["probabilities"][0][1] -= 0.001
    elif fault == "frozen_A":
        route["frozen_A_probabilities"][0][0] += 0.001
        route["frozen_A_probabilities"][0][1] -= 0.001
    else:
        route["control_probabilities"]["v1"][0][0] += 0.01
        route["control_probabilities"]["v1"][0][1] -= 0.01

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid second arm exposed first-arm scores")

    monkeypatch.setattr(study, "score_sequence", forbidden)
    monkeypatch.setattr(study, "score_train_choices", forbidden)
    with pytest.raises(ValueError):
        study.score_catalog_support_comparison(report)


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], {}),
        (["--catalog-support", "repeat"], {"catalog_support": "repeat"}),
        (["--compare-catalog-support"], {"compare_catalog_support": True}),
    ],
)
def test_should_route_only_explicit_new_cli_options(monkeypatch, tmp_path, arguments, expected):
    output = tmp_path / "new.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm-temporal",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(output),
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


def test_should_refuse_mixed_single_and_comparison_cli_modes_before_native_loading(
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
            "--catalog-support",
            "route",
            "--compare-catalog-support",
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid CLI modes reached native loading")

    monkeypatch.setattr(study, "run_case", forbidden)
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2
