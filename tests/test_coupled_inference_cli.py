"""The fixed CLI performs five training calls and four nested inference calls."""

import json
import subprocess
import sys


def test_should_expose_fixed_early_stop_and_fallback_with_neutral_parity():
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_inference"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["scope"] == "engineering_control_no_heldout_data"
    assert len(report["cases"]) == 5 and report["seed"] == 23
    fixed = report["cases"][0]
    assert fixed["scenario"] == "fixed" and fixed["inference"] is None
    assert fixed["fixed_inference_energy_evaluations"] == 7
    for scenario, reason, accepted in (
        ("early_stop", "gradient_tolerance", 2),
        ("fallback", "energy_increase", 0),
    ):
        pc, neutral = [c for c in report["cases"] if c["scenario"] == scenario]
        assert pc["inference"] == neutral["inference"]
        assert pc["updated_parameters"] == neutral["updated_parameters"]
        assert pc["energy"] == neutral["energy"]
        diagnostic = pc["inference"]
        assert diagnostic["reason"] == reason and diagnostic["accepted_steps"] == accepted
        assert diagnostic["gradient_norm_evaluations"] == 1 + accepted
        assert diagnostic["energy_comparisons"] == diagnostic["attempted_steps"]
        assert diagnostic["energy_evaluations"] == 1 + diagnostic["attempted_steps"]
        assert diagnostic["used_fallback"] == (scenario == "fallback")
        assert neutral["wake_state"]["wake_updates"] == 1
        assert neutral["wake_state"]["plasticity_state"] == [1.0, 1.0]
        assert any(value > 0 for value in neutral["wake_state"]["chemical_state"])
