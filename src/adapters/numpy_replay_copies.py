"""Exact original NumPy builder identity port; no copying or consent decisions."""

from src.adapters.numpy_learners import ManagedNumpyBuilder, CircadianLearner


def replay_builder_source(builder: object) -> object:
    if type(builder) is not ManagedNumpyBuilder or type(builder._source) is not CircadianLearner:
        raise ValueError("replay copy requires exact original CPC builder")
    if getattr(builder.__call__, "__func__", None) is not ManagedNumpyBuilder.__call__:
        raise ValueError("original native builder operation changed")
    return builder._source
