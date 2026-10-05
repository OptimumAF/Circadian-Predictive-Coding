"""Fixed CLI composition without a training/score/source/budget override."""

import sys

import pytest

from scripts import run_p612_complete_findings as module


def test_should_compose_two_unchanged_complete_bundle_readers_and_development(monkeypatch):
    calls = []

    def outcome(*args):
        calls.append(("outcome", args))
        return {}, {}, {}

    def matrix(*args):
        calls.append(("matrix", args))
        return {}, {}, {}

    monkeypatch.setattr(module, "read_completed_outcome_costs", outcome)
    monkeypatch.setattr(module, "read_completed_confirmation_matrix", matrix)

    def development():
        calls.append(("development", ()))
        return {}

    monkeypatch.setattr(module.development, "inspect_development_findings", development)
    readers = module._readers()
    readers.outcome_costs()
    readers.matrix()
    readers.development()
    assert [label for label, _ in calls] == ["outcome", "matrix", "development"]
    assert calls[0][1][-1] is module.outcomes._report
    assert calls[1][1][-1] is module.matrix._report


@pytest.mark.parametrize("argument", ["--seed", "--metric", "--baseline", "--budget", "--epochs"])
def test_should_reject_scientific_or_budget_override_before_publication(argument, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_p612_complete_findings", "--publish", argument, "1"])
    monkeypatch.setattr(
        module, "publish_complete_findings", lambda *_: pytest.fail("must reject override")
    )
    with pytest.raises(SystemExit):
        module.main()


@pytest.mark.parametrize("mode", ["--publish", "--read-only"])
def test_should_dispatch_selected_mode_and_report_its_whole_audit(
    mode, monkeypatch, capsys, tmp_path
):
    calls = []
    audit = {"status": "completed", "coverage": {"all": True}, "files": {"whole": True}}
    monkeypatch.setattr(module, "_readers", lambda: "complete ports")

    def publish(*args):
        calls.append(args)
        return audit

    def read(*args):
        calls.append(args)
        return {}, {}, audit

    monkeypatch.setattr(module, "publish_complete_findings", publish)
    monkeypatch.setattr(module, "read_completed_findings", read)
    monkeypatch.setattr(
        sys, "argv", ["run_p612_complete_findings", mode, "--output-dir", str(tmp_path / "new")]
    )
    module.main()
    assert len(calls) == 1 and calls[0][-1] == "complete ports"
    assert '"status": "completed"' in capsys.readouterr().out
