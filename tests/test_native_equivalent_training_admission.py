"""New literal-only materiality/provenance boundary; no learner or data imports."""

from pathlib import Path
import runpy

import pytest

SOURCE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/full-network-native-control-20261101/materiality-study.py"
    ),
    run_name="literal_materiality_only",
)


def test_raw_discrepancy_is_retained_and_native_equivalent_training_is_admitted():
    evidence = dict(SOURCE["CONSISTENCY_EVIDENCE"])
    SOURCE["require_native_consistency"](evidence)
    assert evidence["raw_conflicts"] == 1 and evidence["native_conflicts"] == 0
    assert evidence["source_complete"] is False  # Never upgrade the old case.


@pytest.mark.parametrize(
    "fault", ["native_conflict", "boolean_counter", "coverage", "provenance", "raw_erasure"]
)
def test_changed_or_ambiguous_prior_evidence_is_refused(fault):
    evidence = dict(SOURCE["CONSISTENCY_EVIDENCE"])
    if fault == "native_conflict":
        evidence["native_conflicts"] = 1
    elif fault == "boolean_counter":
        evidence["native_conflicts"] = False
    elif fault == "coverage":
        evidence["selected_pairs"] = 4586
    elif fault == "provenance":
        evidence["source_sha256"] = "0" * 64
    else:
        evidence["raw_conflicts"] = 0
    with pytest.raises(ValueError):
        SOURCE["require_native_consistency"](evidence)
