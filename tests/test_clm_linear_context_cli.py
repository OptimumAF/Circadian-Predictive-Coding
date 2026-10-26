"""Invalid context options refuse before loading or creating an output."""

import pytest
import scripts.run_clm_ticket_workflow as study


@pytest.mark.parametrize(
    "other",
    [None, "linear_fit_control", "reuse_action_features", "record_train_fit", "learned_only"],
)
def test_should_refuse_incompatible_context_options_before_io(other, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "load_local_clm", lambda *a, **k: pytest.fail("loaded refused mode"))
    options = dict(linear_context_updates=True, public_observation_train=other is not None)
    if other:
        options[other] = True
    with pytest.raises(ValueError, match="requires"):
        study.run_case(None, None, {}, **options)
    output = tmp_path / "refused.json"
    args = [
        "workflow",
        "--model-dir",
        "unused",
        "--head",
        "unused",
        "--output",
        str(output),
        "--linear-context-updates",
    ]
    if other:
        args += ["--public-observation-train", "--" + other.replace("_", "-")]
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("entry", [study.run_case, study.run_loaded])
def test_should_refuse_non_boolean_context_mode(entry):
    with pytest.raises(ValueError, match="exact bool"):
        entry(None, None, {}, linear_context_updates=1) if entry is study.run_case else entry(
            None, {}, linear_context_updates=1
        )
