"""Bounded original NumPy replay inventory, without projection or raw copies.

Empty Backprop state needs no replay origin. Nonempty copied CPC snapshots are
reported as separate original holders; current candidate metadata cannot bless them.
"""

from collections import deque
from typing import Any
from src.adapters.numpy_learners import BackpropLearner, BackpropSnapshot, CircadianLearner
from src.core.backprop_mlp import BackpropMLP
from src.app.candidate_checkpoint import CandidateCheckpointView
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    CircadianNetworkSnapshot,
    ReplaySnapshot,
)


def numpy_replay_inventory(groups, limits):
    if type(groups) is not tuple or len(groups) > limits.records.max_records:
        raise ValueError("replay holder inventory exceeds original capture bound")
    entries = {}
    visited = 0
    rows: Any
    for group in groups:
        for snapshot in group.references.models + group.references.snapshots:
            visited += 1
            if visited > limits.max_nodes:
                raise ValueError("replay native inventory exceeds original capture bound")
            if type(snapshot) is CandidateCheckpointView:
                snapshot = snapshot.state
            if type(snapshot) in (BackpropLearner, CircadianLearner):
                model = snapshot._model
                if type(snapshot) is BackpropLearner and type(model) is BackpropMLP:
                    continue
                if (
                    type(snapshot) is not CircadianLearner
                    or type(model) is not CircadianPredictiveCodingNetwork
                ):
                    raise ValueError("replay inventory requires supported original learner/model")
                rows = model._replay_memory
            elif type(snapshot) is BackpropSnapshot:
                continue
            elif type(snapshot) is CircadianNetworkSnapshot and type(snapshot.state) is dict:
                model, rows = snapshot, snapshot.state.get("_replay_memory")
            else:
                raise ValueError("replay inventory requires supported original native holder")
            if type(rows) not in (list, deque) or len(rows) > limits.max_nodes:
                raise ValueError("replay buffer exceeds original capture bounds")
            if any(type(row) is not ReplaySnapshot for row in rows):
                raise ValueError("replay buffer contains unsupported native row")
            entries[id(model)] = (model, tuple(rows))
    return tuple(entries.values())
