"""Saved BP reuse and fixed missing-arm admission, with no learner execution.

Parse the two fixed original evidence logs once, then reread bytes once to check
preservation. Every live arm, final-state
observer and neutral comparator is a nonnumeric stub: zero model initializers,
fits, updates, predictions, gradients, random draws or native calls. Only saved
report scalar arithmetic and finite JSON comparisons execute in the driver.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import continue_continuous_probe_control as study


FAMILIES = ("pc", "neutral-cpc", "active-cpc")


@pytest.fixture(scope="module")
def originals():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
    reference_path = root / "continuous-probe-direction-20261020" / "example-run.log"
    failed_path = root / "continuous-probe-control-20261020" / "example-run.log"
    reference_bytes, failed_bytes = reference_path.read_bytes(), failed_path.read_bytes()
    reference = json.loads(reference_bytes.decode("utf8"))
    failed, end = json.JSONDecoder().raw_decode(failed_bytes.decode("utf8"))
    assert failed["complete"] is False and "Traceback" in failed_bytes.decode("utf8")[end:]
    assert set(failed["reports"]) == {"bp"} and set(reference["reports"]) == {"bp", *FAMILIES}
    return reference, failed, reference_bytes, failed_bytes


def synthetic_result(reference, failed, family):
    report = deepcopy(reference["reports"][family])
    report["settings"] = deepcopy(failed["reports"]["bp"]["settings"])
    template = failed["reports"]["bp"]["probes"][0]["report"]
    for point in report["probes"]:
        inherited = point["report"]
        pair = deepcopy(template)
        pair.update(
            source_updates=inherited["source_updates"],
            source_training_examples=inherited["source_training_examples"],
            inherited=deepcopy(inherited),
            zero_velocity=deepcopy(inherited),
        )
        pair["contrast"] = dict(
            direction=template["contrast"]["direction"],
            fresh_acquisition_gain=0.0,
            fresh_training_gain=0.0,
            old_retention_loss={name: 0.0 for name in inherited["metrics"]["old_retention_loss"]},
        )
        point["report"] = pair
    candidate = SimpleNamespace(final=deepcopy(reference["final_states"][family]), family=family)
    return SimpleNamespace(report=report, candidate=candidate)


@pytest.fixture
def harness(originals, monkeypatch):
    reference, failed = deepcopy(originals[0]), deepcopy(originals[1])
    results = {family: synthetic_result(reference, failed, family) for family in FAMILIES}
    calls, comparisons = [], []

    def run_arm(family):
        calls.append(family)
        return results[family]

    def neutral(first, second):
        comparisons.append((first.candidate.family, second.candidate.family))
        return True

    monkeypatch.setattr(study, "_neutral_exact", neutral)
    monkeypatch.setattr(study, "_final_state", lambda candidate: deepcopy(candidate.final))
    monkeypatch.setattr(
        study, "_capture_owner", lambda candidate, _report: deepcopy(candidate.final), raising=False
    )

    def check_owner(candidate, expected):
        if candidate.final != expected:
            raise RuntimeError("stubbed preceding native owner changed")

    monkeypatch.setattr(study, "_check_owner", check_owner, raising=False)
    return reference, failed, results, calls, comparisons, run_arm


def test_should_reuse_failed_original_bp_and_run_only_missing_families_in_fixed_order(harness):
    reference, failed, results, calls, comparisons, run_arm = harness
    before = deepcopy(reference), deepcopy(failed)
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is True and payload["reused_families"] == ["bp"]
    assert (
        calls
        == payload["attempted_new_families"]
        == payload["returned_new_families"]
        == list(FAMILIES)
    )
    assert comparisons == [("pc", "neutral-cpc")] and payload["neutral_common_exact"] is True
    assert payload["reports"]["bp"] == failed["reports"]["bp"]
    assert payload["final_states"] == reference["final_states"]
    assert payload["saved_original_reports_exact"] == dict.fromkeys(("bp", *FAMILIES), True)
    assert (reference, failed) == before
    for family in FAMILIES:
        assert payload["reports"][family] == results[family].report


@pytest.mark.parametrize(
    "damage",
    [
        "original_complete",
        "bp_incomplete",
        "pair",
        "initial_scores",
        "inherited_scalar",
        "final_state",
        "bp_missing",
    ],
)
def test_should_validate_reused_bp_before_admitting_any_live_arm(harness, damage):
    reference, failed, _results, calls, comparisons, run_arm = harness
    bp = failed["reports"]["bp"]
    if damage == "original_complete":
        failed["complete"] = True
    elif damage == "bp_incomplete":
        bp["complete"] = False
    elif damage == "pair":
        bp["probes"][0]["report"]["zero_velocity"]["complete"] = False
    elif damage == "initial_scores":
        bp["probes"][0]["report"]["initial_scores_exact"] = False
    elif damage == "inherited_scalar":
        bp["probes"][1]["report"]["inherited"]["metrics"]["fresh_training_gain"] += 0.01
    elif damage == "final_state":
        failed["final_states"]["bp"]["training_examples"] += 1
    else:
        del failed["reports"]["bp"]
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is False and calls == comparisons == []
    assert payload["attempted_new_families"] == payload["returned_new_families"] == []
    assert "error" in payload


@pytest.mark.parametrize("family", FAMILIES)
def test_should_retain_returned_paid_report_before_refusing_incomplete_arm(harness, family):
    reference, failed, results, calls, _comparisons, run_arm = harness
    results[family].report["complete"] = False
    results[family].report["error"] = "paid native result failed later gate"
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    expected = list(FAMILIES[: FAMILIES.index(family) + 1])
    assert payload["complete"] is False
    assert (
        calls == payload["attempted_new_families"] == payload["returned_new_families"] == expected
    )
    assert payload["reports"][family] == results[family].report
    assert payload["reports"][family]["progress"]["updates_completed"] == 3
    assert len(payload["reports"][family]["applied_updates"]) == 3


@pytest.mark.parametrize("family", FAMILIES)
def test_should_distinguish_attempted_and_returned_families_when_callback_raises(harness, family):
    reference, failed, _results, calls, _comparisons, run_arm = harness

    def fail(selected):
        if selected == family:
            calls.append(selected)
            raise RuntimeError("unreturned arm attempt")
        return run_arm(selected)

    payload = study.continue_study(reference, failed, run_arm=fail)
    index = FAMILIES.index(family)
    assert payload["complete"] is False and "unreturned arm attempt" in payload["error"]
    assert calls == payload["attempted_new_families"] == list(FAMILIES[: index + 1])
    assert payload["returned_new_families"] == list(FAMILIES[:index])
    assert family not in payload["reports"]


def test_should_require_actual_neutral_gate_before_admitting_active_family(harness, monkeypatch):
    reference, failed, _results, calls, comparisons, run_arm = harness

    def refuse(first, second):
        comparisons.append((first.candidate.family, second.candidate.family))
        return False

    monkeypatch.setattr(study, "_neutral_exact", refuse)
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is False and payload["neutral_common_exact"] is False
    assert calls == ["pc", "neutral-cpc"] and comparisons == [("pc", "neutral-cpc")]
    assert payload["returned_new_families"] == calls
    assert "active-cpc" not in payload["reports"]


@pytest.mark.parametrize(
    "damage",
    [
        "seed",
        "metrics",
        "work",
        "checkpoint",
        "receipt",
        "control_metric",
        "contrast",
        "source_type",
        "pairs",
        "final",
        "candidate",
    ],
)
def test_should_refuse_changed_scalar_or_incomplete_state_after_retaining_paid_arm(harness, damage):
    reference, failed, results, calls, comparisons, run_arm = harness
    result = results["pc"]
    report, pair = result.report, result.report["probes"][1]["report"]
    if damage == "seed":
        report["policy"]["base"]["seed"] = 18
    elif damage == "metrics":
        key = next(iter(report["metrics"]))
        report["metrics"][key] += 0.01
    elif damage == "work":
        report["work"]["training_rows"] += 1
    elif damage == "checkpoint":
        report["checkpoints"][0]["state_exact"] = False
    elif damage == "receipt":
        pair["zero_velocity"]["applied_updates"][0]["event_id"] = "replaced-label"
    elif damage == "control_metric":
        pair["zero_velocity"]["metrics"]["fresh_training_gain"] += 0.01
    elif damage == "contrast":
        pair["contrast"]["fresh_acquisition_gain"] += 0.01
    elif damage == "source_type":
        pair["zero_velocity"]["source_updates"] = 1.0
    elif damage == "pairs":
        report["probes"].pop()
    elif damage == "final":
        result.candidate.final["training_examples"] += 1
    else:
        result.candidate = None
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is False and calls == ["pc"] and comparisons == []
    assert payload["reports"]["pc"] == report
    assert payload["returned_new_families"] == ["pc"]
    assert "error" in payload


def test_should_keep_returned_report_when_final_observer_throws(harness, monkeypatch):
    reference, failed, results, calls, _comparisons, run_arm = harness

    def fail(_candidate):
        raise RuntimeError("paid arm final observer failed")

    monkeypatch.setattr(study, "_final_state", fail)
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is False and calls == ["pc"]
    assert payload["reports"]["pc"] == results["pc"].report
    assert payload["returned_new_families"] == ["pc"]
    assert "final observer failed" in payload["error"]


def test_should_refuse_late_report_replacement_and_retain_original_owned_paid_copy(
    harness, monkeypatch
):
    reference, failed, results, calls, _comparisons, run_arm = harness
    expected = deepcopy(results["pc"].report)

    def final(candidate):
        results["pc"].report["work"]["training_rows"] += 1
        return deepcopy(candidate.final)

    monkeypatch.setattr(study, "_final_state", final)
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is False and calls == ["pc"]
    assert payload["reports"]["pc"] == expected
    assert "preceding" in payload["error"]


def test_should_revalidate_previous_paid_report_after_another_arm_callback(harness):
    reference, failed, results, calls, _comparisons, run_arm = harness
    expected = deepcopy(results["pc"].report)

    def run(selected):
        returned = run_arm(selected)
        if selected == "neutral-cpc":
            results["pc"].report["work"]["training_rows"] += 1
        return returned

    payload = study.continue_study(reference, failed, run_arm=run)
    assert payload["complete"] is False and calls == ["pc", "neutral-cpc"]
    assert payload["reports"]["pc"] == expected
    assert payload["reports"]["neutral-cpc"] == results["neutral-cpc"].report
    assert payload["returned_new_families"] == calls
    assert "preceding" in payload["error"]


def test_should_recheck_retained_evidence_after_neutral_observer_before_active(
    harness, monkeypatch
):
    reference, failed, results, calls, _comparisons, run_arm = harness
    expected = deepcopy(results["pc"].report)

    def neutral(_first, _second):
        results["pc"].report["metrics"][next(iter(results["pc"].report["metrics"]))] += 0.01
        return True

    monkeypatch.setattr(study, "_neutral_exact", neutral)
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is False and calls == ["pc", "neutral-cpc"]
    assert payload["reports"]["pc"] == expected and "active-cpc" not in payload["reports"]
    assert "preceding" in payload["error"]


def test_should_recheck_preceding_owner_after_later_callback_without_executing_model(harness):
    reference, failed, results, calls, _comparisons, run_arm = harness

    def run(selected):
        returned = run_arm(selected)
        if selected == "neutral-cpc":
            results["pc"].candidate.final["training_examples"] += 1
        return returned

    payload = study.continue_study(reference, failed, run_arm=run)
    assert payload["complete"] is False and calls == ["pc", "neutral-cpc"]
    assert payload["reports"]["neutral-cpc"] == results["neutral-cpc"].report
    assert "owner changed" in payload["error"]


def test_should_keep_original_failure_provenance_and_observed_work_distinct_from_planned_caps(
    harness,
):
    reference, failed, _results, _calls, _comparisons, run_arm = harness
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is True
    assert payload["original_invocation"]["complete"] is False
    assert payload["original_invocation"]["disposition"] == "FAILED/SPENT"
    assert payload["original_invocation"]["error"] == failed["error"]
    assert payload["planned_new_work"]["parameter_updates"] == 33
    assert payload["planned_new_work"]["training_rows"] == 69
    assert payload["observed_new_work"]["returned_reports"] == 3
    assert payload["observed_new_work"]["retained_applied_receipts"] == 33
    assert payload["observed_new_work"]["unobserved_attempts"] == []
    assert payload["simulator_calls"] == payload["final_rows"] == 0


def test_should_accept_negative_or_zero_gains_without_selecting_or_retrying(harness):
    reference, failed, _results, calls, _comparisons, run_arm = harness
    # The saved original contains negative inherited gains. Identical synthetic
    # control outcomes give zero contrasts and still pass all structural gates.
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is True and calls == list(FAMILIES)
    gains = [
        point["report"]["inherited"]["metrics"]["fresh_training_gain"]
        for report in payload["reports"].values()
        for point in report["probes"]
    ]
    assert any(value < 0 for value in gains)
    assert all(
        point["report"]["contrast"]["fresh_acquisition_gain"] == 0.0
        for family in FAMILIES
        for point in payload["reports"][family]["probes"]
    )


def test_should_keep_malformed_paid_observation_unknown_without_masking_original_failure(harness):
    reference, failed, results, calls, _comparisons, run_arm = harness
    results["pc"].report.update(
        complete=False, applied_updates=None, error="native report incomplete after paid work"
    )
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is False and calls == ["pc"]
    assert payload["reports"]["pc"] == results["pc"].report
    assert payload["observed_new_work"]["returned_reports"] == 1
    assert payload["observed_new_work"]["retained_applied_receipts"] is None
    assert payload["observed_new_work"]["retained_applied_receipts_by_family"] == {"pc": None}
    assert "incomplete" in payload["error"]


def test_should_return_owned_bp_metadata_without_modifying_original_failed_bytes(
    harness, originals
):
    reference, failed, _results, _calls, _comparisons, run_arm = harness
    expected = deepcopy(failed)
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    payload["reports"]["bp"]["work"]["training_rows"] += 1
    payload["final_states"]["bp"]["training_examples"] += 1
    assert failed == expected
    root = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
    assert (
        root / "continuous-probe-direction-20261020" / "example-run.log"
    ).read_bytes() == originals[2]
    assert (
        root / "continuous-probe-control-20261020" / "example-run.log"
    ).read_bytes() == originals[3]


def test_should_refuse_final_control_score_role_replacement_before_next_arm(harness):
    reference, failed, results, calls, _comparisons, run_arm = harness
    results["pc"].report["probes"][0]["report"]["zero_velocity"]["checkpoints"][1]["scores"][
        "fresh"
    ]["role"] = "train"
    payload = study.continue_study(reference, failed, run_arm=run_arm)
    assert payload["complete"] is False and calls == ["pc"]
    assert "identity/role" in payload["error"]


def test_should_finish_native_checks_after_all_public_result_getters(harness, monkeypatch):
    reference, failed, results, calls, _comparisons, run_arm = harness
    pc, neutral = results["pc"], results["neutral-cpc"]
    armed = [False]

    class LateObserver:
        @property
        def report(self):
            if armed[0]:
                pc.candidate.final["training_examples"] = 7
            return neutral.report

        @property
        def candidate(self):
            if armed[0]:
                pc.candidate.final["training_examples"] = 8
            return neutral.candidate

    def compare(_first, _second):
        armed[0] = True
        return True

    def run(family):
        returned = run_arm(family)
        return LateObserver() if family == "neutral-cpc" else returned

    monkeypatch.setattr(study, "_neutral_exact", compare)
    payload = study.continue_study(reference, failed, run_arm=run)
    assert payload["complete"] is False and calls == ["pc", "neutral-cpc"]
    assert "owner changed" in payload["error"]
    assert payload["reports"]["neutral-cpc"] == neutral.report


def test_should_refuse_non_pc_owner_before_native_snapshot_or_next_model():
    with pytest.raises(ValueError, match="exact continuous owner"):
        study._capture_owner(object(), {})
