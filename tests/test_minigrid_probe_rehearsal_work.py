"""Pure public replay admission, honest partial-cost reduction and protocol costs."""

from copy import deepcopy

import pytest

from src.adapters import minigrid_probe_rehearsal as recipe


FLAGS = dict(
    public_history=True,
    invariant_history=True,
    joint_catalogue=True,
    distinct_probe=True,
    old_task_readouts=True,
    return_to_a=True,
    qualification_method="bp",
)


@pytest.mark.parametrize("method", recipe.METHODS)
def test_should_allow_original_native_recipe_and_pure_bounds(method):
    recipe.validate_option(True, **{**FLAGS, "qualification_method": method}, train_fit_bounds=True)


@pytest.mark.parametrize("value", [None, 0, 1, "yes", [], {}])
def test_should_refuse_nonboolean_option(value):
    with pytest.raises(ValueError, match="exact bool"):
        recipe.validate_option(value, **FLAGS)


def test_should_leave_other_legacy_recipes_unrestricted_when_disabled():
    recipe.validate_option(False, cue_relative_history=True, qualification_method="anything")


@pytest.mark.parametrize("name", [name for name in FLAGS if name != "qualification_method"])
@pytest.mark.parametrize("value", [False, 1, None])
def test_should_require_exact_true_route_prerequisites(name, value):
    with pytest.raises(ValueError, match="requires"):
        recipe.validate_option(True, **{**FLAGS, name: value})


@pytest.mark.parametrize("method", [None, 1, "pc_relax64", "public_invariant_bp", "recurrent_bc"])
def test_should_refuse_unmatched_native_method(method):
    with pytest.raises(ValueError, match="requires"):
        recipe.validate_option(True, **{**FLAGS, "qualification_method": method})


@pytest.mark.parametrize(
    "name",
    [
        "cue_relative_history",
        "pc_qualification",
        "acquisition_only",
        "memory_control",
        "weighted_bp",
        "weighted_pc",
        "weighted_catalogue",
        "activation_health",
        "gradient_geometry",
        "prior_gradient_geometry",
        "train_a_rollout",
        "probe_rollouts",
        "deferred_probe_rollouts",
        "train_action_margins",
        "pc_update_diagnostics",
        "probe_inference",
    ],
)
def test_should_refuse_other_objectives_encodings_and_extra_diagnostics(name):
    with pytest.raises(ValueError, match="requires"):
        recipe.validate_option(True, **FLAGS, **{name: True})


def protocol():
    return dict(
        protocol_id="original_joint_catalogue_v1",
        methods=["public_invariant_bp"],
        probe=dict(source_updates=[0, 128], updates=16),
        return_phase=dict(updates=16),
        old_task_readouts=dict(updates="readouts only", role="train"),
        joint_catalogue=dict(updates=128),
        planned_work=dict(native_train_calls=192, training_rows=7488, replay_rows=0),
        primary_comparisons=[],
        baseline_controls={"original": "frozen"},
        seed=17,
        limitations="negative results preserved",
    )


def test_should_describe_extra_exposure_without_rewriting_baseline_controls():
    actual = protocol()
    before = deepcopy(actual)
    recipe.describe(actual)
    assert actual["protocol_id"] == "original_joint_catalogue_old_a_rehearsal_v1"
    assert actual["planned_work"]["native_train_calls"] == 256
    assert actual["planned_work"]["training_rows"] == 8064
    assert actual["planned_work"]["replay_rows"] == 576
    extra = actual["planned_work"]["old_a_rehearsal"]
    assert extra["native_train_calls"] == 64
    assert extra["training_rows"] == extra["replay_rows"] == 576
    assert extra["bank_transitions_per_probe"] == 18
    assert extra["bank_encoded_bytes_per_probe"] == 87280
    assert extra["logical_bank_encoded_bytes"] == 174560
    assert extra["validation_projection_rows"] == 72
    assert extra["validation_projected_array_equivalent_bytes"] == 170496
    assert "remain readonly" in actual["old_task_readouts"]["updates"]
    assert "no further replay" in actual["old_a_rehearsal"]["return_phase"]
    assert "not matched" in actual["old_a_rehearsal"]["comparison"]
    for key in (
        "baseline_controls",
        "primary_comparisons",
        "seed",
        "methods",
        "probe",
        "return_phase",
    ):
        assert actual[key] == before[key]


@pytest.mark.parametrize(
    "replace",
    [
        {"methods": ["public_cue_relative_bp"]},
        {"methods": ["public_invariant_bp", "public_invariant_pc"]},
        {"probe": dict(source_updates=[0, 64], updates=16)},
        {"probe": dict(source_updates=[0, 128], updates=8)},
        {"return_phase": dict(updates=8)},
    ],
)
def test_should_refuse_protocol_costs_for_another_schedule(replace):
    actual = {**protocol(), **replace}
    before = deepcopy(actual)
    with pytest.raises(ValueError, match="requires"):
        recipe.describe(actual)
    assert actual == before


def work():
    return dict(
        charged_replay_examples=288,
        attempted_updates=32,
        returned_updates=32,
        committed_updates=32,
        returned_training_rows=288,
        bank_encoded_bytes=87280,
        validation_projection_rows=36,
        validation_projected_array_equivalent_bytes=85248,
        budget_binding_preserved=True,
        work_unknown=False,
    )


def test_should_sum_actual_two_probe_costs_and_logical_storage_without_mutation():
    rows = [dict(probes=[dict(rehearsal_work=work()), dict(rehearsal_work=work())])]
    before = deepcopy(rows)
    actual = recipe.aggregate(rows)
    assert actual["native_train_calls"] == actual["returned_native_train_calls"] == 64
    assert actual["committed_native_train_calls"] == 64
    assert actual["training_rows"] == actual["replay_rows"] == 576
    assert actual["logical_bank_encoded_bytes"] == 174560
    assert actual["validation_projection_rows"] == 72
    assert actual["validation_projected_array_equivalent_bytes"] == 170496
    assert actual["observed_slot_count"] == 2
    assert actual["unknown_slot_count"] == 0
    assert actual["budget_binding_preserved"] is True
    assert actual["partial_work_unknown"] is False
    assert rows == before


def test_should_distinguish_paid_attempt_from_returned_and_committed_work():
    partial = work()
    partial.update(
        charged_replay_examples=18,
        attempted_updates=2,
        returned_updates=1,
        committed_updates=0,
        returned_training_rows=9,
        work_unknown=True,
    )
    actual = recipe.aggregate([dict(probes=[dict(rehearsal_work=partial)])])
    assert actual["native_train_calls"] == 2
    assert actual["returned_native_train_calls"] == 1
    assert actual["committed_native_train_calls"] == 0
    assert actual["training_rows"] == 9
    assert actual["replay_rows"] == 18
    assert actual["partial_work_unknown"] is True


def test_should_preserve_unknown_evidence_without_imputing_zero():
    actual = recipe.aggregate([dict(probes=[dict(rehearsal_work=None)])])
    assert actual["native_train_calls"] is None
    assert actual["training_rows"] is None
    assert actual["replay_rows"] is None
    assert actual["logical_bank_encoded_bytes"] is None
    assert actual["unknown_slot_count"] == 1
    assert actual["failure_work_may_be_unobserved"] is True
    assert actual["budget_binding_preserved"] is False


def test_should_report_known_sums_as_partial_lower_bounds_when_other_slot_unknown():
    actual = recipe.aggregate(
        [dict(probes=[dict(rehearsal_work=work()), dict(rehearsal_work=None)])]
    )
    assert actual["native_train_calls"] == 32
    assert actual["replay_rows"] == 288
    assert actual["partial_work_unknown"] is True
    assert actual["unknown_slot_count"] == 1


def test_should_ignore_unreached_slots_without_factory_admission():
    actual = recipe.aggregate([dict(probes=[dict(report=None)])])
    assert actual["native_train_calls"] == 0
    assert actual["observed_slot_count"] == 0
    assert actual["partial_work_unknown"] is False


@pytest.mark.parametrize("value", [True, -1, 1.0, "32"])
def test_should_refuse_malformed_observed_costs_instead_of_silent_coercion(value):
    malformed = work()
    malformed["attempted_updates"] = value
    with pytest.raises(ValueError, match="nonnegative exact int"):
        recipe.aggregate([dict(probes=[dict(rehearsal_work=malformed)])])


def test_should_keep_individual_missing_cost_unknown_but_preserve_other_receipts():
    partial = work()
    partial.pop("returned_training_rows")
    partial["budget_binding_preserved"] = False
    actual = recipe.aggregate([dict(probes=[dict(rehearsal_work=partial)])])
    assert actual["native_train_calls"] == 32
    assert actual["training_rows"] is None
    assert actual["replay_rows"] == 288
    assert actual["partial_work_unknown"] is True
    assert actual["budget_binding_preserved"] is False
