"""Pure BP observer admission, protocol declaration and honest observed work."""

from copy import deepcopy

import pytest

from src.adapters import minigrid_bp_update_route as route


FLAGS = dict(
    public_history=True,
    invariant_history=True,
    joint_catalogue=True,
    distinct_probe=True,
    old_task_readouts=True,
    return_to_a=True,
    qualification_method="bp",
)


def test_should_allow_original_bp_route_and_pure_bounds():
    route.validate_option(True, **FLAGS, train_fit_bounds=True)


def test_should_preserve_other_recipes_when_disabled():
    route.validate_option(False, qualification_method="pc", cue_relative_history=True)


@pytest.mark.parametrize("value", [None, 0, 1, "yes", [], {}])
def test_should_refuse_nonboolean_option(value):
    with pytest.raises(ValueError, match="exact bool"):
        route.validate_option(value, **FLAGS)


@pytest.mark.parametrize("name", route.REQUIRED)
@pytest.mark.parametrize("value", [False, 1, None])
def test_should_require_exact_true_prerequisites(name, value):
    with pytest.raises(ValueError, match="requires"):
        route.validate_option(True, **{**FLAGS, name: value})


@pytest.mark.parametrize("name", [*route.FORBIDDEN, "probe_inference"])
def test_should_refuse_other_recipes_and_diagnostics(name):
    with pytest.raises(ValueError, match="requires"):
        route.validate_option(True, **FLAGS, **{name: True})


@pytest.mark.parametrize("method", [None, 1, "pc", "neutral_cpc", "gated_cpc", "pc_relax64"])
def test_should_require_explicit_bp(method):
    with pytest.raises(ValueError, match="requires"):
        route.validate_option(True, **{**FLAGS, "qualification_method": method})


def protocol():
    return dict(
        protocol_id="original_joint_catalogue_v1",
        methods=["public_invariant_bp"],
        probe=dict(source_updates=[0, 128], updates=16),
        return_phase=dict(updates=16),
        joint_catalogue=dict(updates=128),
        seed=17,
        limitations="negative results preserved",
        primary_comparisons=[],
        baseline_controls=dict(original="frozen"),
        planned_work=dict(
            native_train_calls=192,
            training_rows=7488,
            replay_rows=0,
            score_forwards=94,
            score_rows=846,
        ),
    )


def test_should_describe_disposable_extra_costs_without_rewriting_score_recipe():
    actual = protocol()
    before = deepcopy(actual)
    route.describe(actual)
    assert actual["protocol_id"] == "original_joint_catalogue_bp_update_diagnostics_v1"
    assert actual["planned_work"]["native_train_calls"] == 195
    assert actual["planned_work"]["training_rows"] == 7515
    assert actual["planned_work"]["replay_rows"] == 18
    assert actual["planned_work"]["score_forwards"] == 94
    assert actual["planned_work"]["score_rows"] == 846
    extra = actual["bp_update_diagnostics"]
    assert extra["source_update"] == 128
    assert extra["forward_calls"] == 15 and extra["forward_rows"] == 135
    assert extra["backward_calls"] == 9 and extra["backward_rows"] == 81
    assert extra["admission_projection_rows"] == 54
    assert extra["admission_projection_array_equivalent_bytes"] == 127872
    assert "no final or reserved F" in extra["isolation"]
    for key in (
        "methods",
        "seed",
        "probe",
        "return_phase",
        "primary_comparisons",
        "baseline_controls",
    ):
        assert actual[key] == before[key]


@pytest.mark.parametrize(
    "replacement",
    [
        dict(methods=["public_invariant_pc"]),
        dict(methods=["public_cue_relative_bp"]),
        dict(probe=dict(source_updates=[0, 64], updates=16)),
        dict(probe=dict(source_updates=[0, 128], updates=8)),
        dict(return_phase=dict(updates=8)),
    ],
)
def test_should_refuse_other_protocol_cost_declarations(replacement):
    actual = {**protocol(), **replacement}
    before = deepcopy(actual)
    with pytest.raises(ValueError, match="requires"):
        route.describe(actual)
    assert actual == before


def record():
    work = {name: 0 for name in route.COST_FIELDS.values()}
    work.update(
        native_train_calls_attempted=3,
        native_train_calls_returned=3,
        native_training_rows_attempted=27,
        native_training_rows_returned=27,
        charged_updates=3,
        charged_replay_rows=18,
        forward_calls_attempted=15,
        forward_calls_completed=15,
        forward_rows_attempted=135,
        forward_rows_completed=135,
        backward_calls_attempted=9,
        backward_calls_completed=9,
        backward_rows_attempted=81,
        backward_rows_completed=81,
        admission_projection_rows=54,
        admission_projection_array_equivalent_bytes=127872,
        budget_binding_preserved=True,
        failure_work_may_be_unobserved=False,
    )
    return dict(status="complete", work=work)


def test_should_report_observed_costs_without_mutating_original_evidence():
    original = record()
    before = deepcopy(original)
    actual = route.aggregate(original)
    assert actual["native_train_calls"] == actual["returned_native_train_calls"] == 3
    assert actual["training_rows"] == actual["attempted_training_rows"] == 27
    assert actual["replay_rows"] == 18
    assert actual["forward_rows_completed"] == 135
    assert actual["backward_rows_completed"] == 81
    assert actual["budget_binding_preserved"] is True
    assert actual["partial_work_unknown"] is False
    assert original == before


def test_should_distinguish_paid_attempts_from_returned_rows():
    partial = record()
    partial["status"] = "incomplete"
    partial["work"].update(
        native_train_calls_attempted=2,
        native_train_calls_returned=1,
        native_training_rows_attempted=18,
        native_training_rows_returned=9,
        charged_updates=2,
        charged_replay_rows=9,
        failure_work_may_be_unobserved=True,
    )
    actual = route.aggregate(partial)
    assert actual["native_train_calls"] == 2
    assert actual["returned_native_train_calls"] == 1
    assert actual["training_rows"] == 9 and actual["attempted_training_rows"] == 18
    assert actual["replay_rows"] == 9
    assert actual["failure_work_may_be_unobserved"] is True


def test_should_treat_no_admission_as_known_zero_work():
    actual = route.aggregate(None)
    assert all(actual[name] == 0 for name in route.COST_FIELDS)
    assert actual["admitted"] is False
    assert actual["partial_work_unknown"] is False


@pytest.mark.parametrize(
    "partial", [dict(status="incomplete"), dict(status="incomplete", work=None)]
)
def test_should_never_impute_free_work_when_admitted_report_is_missing(partial):
    actual = route.aggregate(partial)
    assert all(actual[name] is None for name in route.COST_FIELDS)
    assert actual["budget_binding_preserved"] is False
    assert actual["failure_work_may_be_unobserved"] is True


def test_should_keep_individual_missing_cost_unknown_and_other_receipts_observed():
    partial = record()
    partial["work"].pop("native_training_rows_returned")
    actual = route.aggregate(partial)
    assert actual["training_rows"] is None
    assert actual["native_train_calls"] == 3
    assert actual["replay_rows"] == 18
    assert actual["partial_work_unknown"] is True


@pytest.mark.parametrize("value", [True, -1, 3.0, "3"])
def test_should_refuse_invalid_observed_counter_types(value):
    partial = record()
    partial["work"]["native_train_calls_attempted"] = value
    with pytest.raises(ValueError, match="nonnegative exact int"):
        route.aggregate(partial)


def test_should_add_native_costs_without_changing_benchmark_or_score_counts():
    result = dict(
        native_train_calls=192,
        training_rows=7488,
        replay_rows=0,
        benchmark_native_train_calls=128,
        score_forwards=94,
        failure_work_may_be_unobserved=False,
    )
    route.add_work(result, record())
    assert result["native_train_calls"] == 195 and result["training_rows"] == 7515
    assert result["replay_rows"] == 18
    assert result["benchmark_native_train_calls"] == 128 and result["score_forwards"] == 94
    assert result["observed_work_is_lower_bound"] is False


def test_should_propagate_unknown_and_known_partial_costs_into_existing_totals():
    partial = record()
    partial["status"] = "incomplete"
    partial["work"].pop("native_training_rows_returned")
    result = dict(native_train_calls=192, training_rows=7488, replay_rows=0)
    route.add_work(result, partial)
    assert result["native_train_calls"] == 195
    assert result["training_rows"] is None and result["replay_rows"] == 18
    assert result["failure_work_may_be_unobserved"] is True
    assert result["observed_work_is_lower_bound"] is True
