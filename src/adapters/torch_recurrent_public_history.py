"""Cold paired recurrent control for public first-frame history.

Input: a complete, untrained148-input paired snapshot. Output: an owned296-input
paired snapshot for existing encoded episodes/array ports. The additional input
weights start at zero so the initial law agrees with the supplied raw learner
within float32 roundoff. This module owns no data collection, training, stream,
evaluation, role conversion, or scientific comparison.
"""

from dataclasses import replace

from src.adapters import torch_recurrent_episode_batch as paired
from src.shared.torch_runtime import require_torch


def initialize_public_history_control(
    snapshot: paired.TorchRecurrentEpisodeBatchSnapshot,
) -> paired.TorchRecurrentEpisodeBatchSnapshot:
    """Extend a cold raw snapshot without resetting any paid training history.

    Use `encode_public_history_episode` for inputs ordered [first, current].
    Restore the returned snapshot into a296-input paired learner or existing
    TRAIN array adapter. A trained state requires a separately specified warm
    conversion and is deliberately refused here.
    """
    owned = paired._snapshot(snapshot)
    if owned.policy.feature_dim != 148:
        raise ValueError("public history initialization requires raw148-input paired state")
    if (owned.completed_updates, owned.training_episodes, owned.training_examples) != (0, 0, 0):
        raise ValueError("public history initialization requires a cold paired state")
    torch = require_torch()
    weights = owned.parameters["weight_input"].detach()
    # Why: changing seed tensor geometry also changes later random draws. Copy
    # the actual raw law instead of trusting same-seed wider initialization.
    parameters = dict(owned.parameters)
    parameters["weight_input"] = torch.cat((torch.zeros_like(weights), weights), dim=0)
    return paired._snapshot(
        replace(owned, policy=replace(owned.policy, feature_dim=296), parameters=parameters)
    )
