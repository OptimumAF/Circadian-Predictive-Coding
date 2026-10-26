"""Route the one fixed long-stream mode without modifying original CLI paths."""

import sys

import pytest

import scripts.run_stream_rehearsal_pilot as cli


def forbidden(*args, **kwargs):
    pytest.fail("unselected CLI callback reached")


@pytest.mark.parametrize("action", ["describe", "run"])
def test_should_route_only_the_fixed_long_stream_without_kwargs(action, monkeypatch, capsys):
    calls = []

    def selected():
        calls.append(action)
        return {"status": "complete"}

    for name in (
        "describe_protocol",
        "run_pilot",
        "describe_paired_protocol",
        "run_paired_pilot",
        "describe_tracked_protocol",
        "run_tracked_pilot",
        "describe_long_protocol",
        "run_long_pilot",
    ):
        expected = "describe_long_protocol" if action == "describe" else "run_long_pilot"
        monkeypatch.setattr(cli, name, selected if name == expected else forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", f"--{action}", "--scenario", "long-stream"])
    assert cli.main() == 0
    assert capsys.readouterr().out and calls == [action]


@pytest.mark.parametrize("action", ["describe", "run"])
@pytest.mark.parametrize("flag", ["--recent-only", "--diagnose-learnability"])
def test_should_refuse_unfrozen_single_flags_before_long_callback(action, flag, monkeypatch):
    monkeypatch.setattr(cli, "describe_long_protocol", forbidden)
    monkeypatch.setattr(cli, "run_long_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", f"--{action}", "--scenario", "long-stream", flag])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_exit_incomplete_long_run_without_retry(monkeypatch, capsys):
    calls = []

    def incomplete():
        calls.append(1)
        return {"status": "incomplete", "reason": "preserved failure"}

    monkeypatch.setattr(cli, "run_long_pilot", incomplete)
    monkeypatch.setattr(sys, "argv", ["pilot", "--run", "--scenario", "long-stream"])
    assert cli.main() == 2 and calls == [1]
    assert "preserved failure" in capsys.readouterr().out
