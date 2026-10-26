"""Current-student labels require the exact accepted balanced reconstruction.

Only literal model owners and opaque stage doubles are used. The new30 cases
forecast17 source/15 acquisition/45 continuation/26 readout/8 collection calls;
all initialization, learning, inference, teacher and SDK primitives are forbidden.
"""

from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as bp
from src.core import continuous_bc as core
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.push_cube_teacher import PushCubeTeacher
from src.infra import maniskill_push_cube as native
from test_push_cube_continuation_study import run_routing_case, study


def forbidden(*args, **options):
    pytest.fail("balanced collection fixture admitted actual model, teacher or SDK work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts: dict[str, int] = defaultdict(int)
    with pytest.MonkeyPatch.context() as patch:
        for name in ("__init__", "fork", "train_batch", "train_head_gauss_newton_batch", "predict"):
            patch.setattr(bp.NumpyContinuousBCLearner, name, forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        patch.setattr(core, "continuous_bc_loss_and_gradients", forbidden)
        patch.setattr(core, "_forward", forbidden)
        patch.setattr(PushCubeTeacher, "__init__", forbidden)
        patch.setattr(PushCubeTeacher, "decide", forbidden)
        patch.setattr(native, "_make_environment", forbidden)
        yield counts
    if os.environ.get("BPC_FIXTURE_LEDGER"):
        with Path(os.environ["BPC_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["routing_source_calls"] == 17
    assert counts["routing_acquisition_calls"] == 15
    assert counts["routing_continuation_calls"] == 45
    assert counts["routing_fit_calls"] == 26
    assert counts["routing_collection_calls"] == 8
    assert counts["routing_push_calls"] == counts["routing_pick_calls"] == 0


@pytest.fixture
def source():
    # Why: exercise native identity/state guards without inventing training or
    # consuming an initializer. These counters are controlled routing tags.
    owner = object.__new__(bp.NumpyContinuousBCLearner)
    owner._policy = bp.ContinuousBCPolicy(cube_taskspace_spec(), hidden_dim=2, seed=17)
    parameters = (
        np.zeros((12, 2), np.float64),
        np.zeros(2, np.float64),
        np.zeros((2, 4), np.float64),
        np.zeros(4, np.float64),
    )
    owner._state = bp.ContinuousBCSnapshot(
        owner._policy,
        parameters,
        (parameters[0].copy(), parameters[1].copy(), parameters[2].copy(), parameters[3].copy()),
        40,
        2776,
    )
    return owner


@pytest.mark.parametrize(
    "options",
    [
        dict(balanced_push_corrections=1),
        dict(balanced_push_corrections=True),
        dict(
            balanced_push_corrections=True, profile="bp_mixed", longer_fit=True, correction_fit=True
        ),
        dict(balanced_push_corrections=True, profile="bp_mixed_retention", correction_fit=True),
        dict(balanced_push_corrections=True, profile="bp_mixed_retention", longer_fit=True),
        dict(
            balanced_push_corrections=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            correction_fit=True,
            native_endpoint=True,
        ),
        dict(
            balanced_push_corrections=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            correction_fit=True,
            diagnostics_only=True,
        ),
        dict(
            balanced_push_corrections=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            correction_fit=True,
            balanced_correction_fit=True,
        ),
        dict(
            balanced_push_corrections=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            correction_fit=True,
            training_diagnostics=True,
        ),
        dict(
            balanced_push_corrections=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            correction_fit=True,
            reserved_probes=True,
        ),
        dict(
            balanced_push_corrections=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            correction_fit=True,
            push_corrections=True,
        ),
        dict(
            balanced_push_corrections=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            correction_fit=True,
            correction_readouts=True,
        ),
        dict(
            balanced_push_corrections=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            correction_fit=True,
            zero_velocity_fit=True,
        ),
    ],
)
def test_should_refuse_invalid_collection_selection_before_any_saved_input_or_source(
    tmp_path,
    monkeypatch,
    options,
):
    for name in (
        "read_reused_control",
        "read_reused_mixed",
        "read_reused_longer",
        "read_reused_retention",
        "read_reused_native",
        "read_reused_corrective",
        "read_reused_corrective_readouts",
        "_read_push_study",
        "load_push_correction_inputs",
        "read_push_training",
        "read_joint_acquisition_inputs",
        "run_joint_readiness_stream",
    ):
        monkeypatch.setattr(study, name, forbidden)
    output = tmp_path / "invalid-balanced-collection.json"
    with pytest.raises(ValueError):
        study.run_study(output, **options)
    assert not output.exists()


@pytest.mark.parametrize(
    "collection_fault",
    [
        None,
        "cache",
        "recipe",
        "data",
        "reconstruction",
        "weights",
        "readout_incomplete",
        "readout_mismatch",
        "paid_readout",
        "counter",
        "incomplete",
        "unknown",
        "prior",
        "labels",
        "trajectory",
        "release",
        "collection_counter",
    ],
)
def test_should_collect_only_the_reproduced_balanced_student_and_refuse_paid_or_prior_corruption(
    tmp_path,
    monkeypatch,
    source,
    ledger,
    collection_fault,
):
    run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        longer_fit=True,
        correction_fit=True,
        balanced_push_corrections=True,
        collection_fault=collection_fault,
    )
