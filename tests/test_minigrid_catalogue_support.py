"""Explicit six-source support is the same set as128 repeated joint wakes."""

from dataclasses import replace

import pytest

from src.adapters import minigrid_probe_support as support
from src.adapters import minigrid_learnability as pilot
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode


def originals():
    return tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )


def test_should_match_repeated_training_support_and_charge_only_complete_sources_once():
    sources, data = originals(), raw_probe_data()
    repeated = pilot.shared._packets(sources)
    old = support.summarize_probe_support(repeated, data)
    new = support.summarize_probe_support(sources, data, catalogue_sources=True)
    assert old["training"]["rows"] == 1152 and old["work"]["history_rows"] == 1314
    assert new["training"]["packets"] == 6 and new["training"]["rows"] == 54
    assert new["work"]["history_rows"] == new["work"]["canonical_projection_rows"] == 216
    assert new["work"]["canonical_projected_array_equivalent_bytes"] == 216 * 2368
    assert new["training"]["raw"] == old["training"]["raw"]
    assert new["training"]["canonical"] == old["training"]["canonical"]
    assert new["probe"] == old["probe"]
    assert new["all_canonical_inputs_unseen"] is old["all_canonical_inputs_unseen"] is True
    assert new["work"]["native_updates"] == new["work"]["native_score_forwards"] == 0


@pytest.mark.parametrize("flag", [None, 0, 1, "true"])
def test_should_refuse_coerced_source_mode_before_rows(monkeypatch, flag):
    monkeypatch.setattr(support, "_rows", lambda *args: pytest.fail("unexpected support work"))
    with pytest.raises(ValueError):
        support.summarize_probe_support(originals(), raw_probe_data(), catalogue_sources=flag)


@pytest.mark.parametrize("fault", ["missing", "reordered", "renamed", "context", "role", "legacy"])
def test_should_keep_fixed_source_mode_and_default_legacy_admission_separate(fault):
    sources = originals()
    if fault == "missing":
        sources = sources[:-1]
    elif fault == "reordered":
        sources = sources[::-1]
    elif fault == "renamed":
        victim = sources[-1]
        assert victim.episode is not None
        sources = (
            *sources[:-1],
            replace(victim, episode=replace(victim.episode, episode_id="foreign-red")),
        )
    elif fault == "context":
        sources = (*sources[:-1], replace(sources[-1], context="A"))
    elif fault == "role":
        sources = (*sources[:-1], replace(sources[-1], role="probe_train"))
    with pytest.raises(ValueError):
        support.summarize_probe_support(
            sources, raw_probe_data(), catalogue_sources=fault != "legacy"
        )
