"""Prepare contextual or explicit shared-head packets for existing consolidation.

Retained encoded episodes must stay TRAIN. Current probe inputs are transient
and never enter the contextual bank; the shared path admits ordinary TRAIN only.
No runtime, fit, budget authority or data-role rewrite lives here.
"""

from src.core.context_joint_replay import packet_context, compose_joint_packet
from src.core.demonstration_episode import decode_episode, episode_arrays
from src.core.shared_replay_schedule import SharedReplaySelection
from src.core.coupled_energy import Array
from src.core.shared_linear_replay import SharedJointReplayPolicy, compose_shared_packet
from src.core.categorical_linear_fit import _features
from src.core.categorical_probability_scores import validate_class_targets


def prepare_joint_rehearsal(policy, schema, selection, current, maximum_rows, wake_role):
    current_index = packet_context(policy, *current)
    chosen: list[str] = []
    arrays: list[tuple[Array, Array]] = []
    seen = set()
    # Selection is newest first; choose by declared public context and content,
    # never prediction, loss, gradient or evaluation outcomes.
    wires = dict(zip(selection.sample_ids, selection.training_batches(), strict=True))
    for identifier in reversed(selection.sample_ids):
        pair = wires[identifier]
        episode = decode_episode(pair[0], pair[1], schema)
        if episode.role != "train":
            raise ValueError("joint rehearsal retains original TRAIN episodes only")
        x, y = episode_arrays(episode)
        context = packet_context(policy, x, y)
        key = (x.tobytes(), y.tobytes())
        # Why: an ordinary wake may already contain several complete episodes.
        # Do not spend scarce retention slots replaying an episode wholly present
        # in that paid packet; compare ordered row/target pairs, not outcomes.
        covered = any(
            (current[0][start : start + len(x)] == x).all()
            and (current[1][start : start + len(y)] == y).all()
            for start in range(len(current[0]) - len(x) + 1)
        )
        if context != current_index or covered or key in seen:
            continue
        seen.add(key)
        if len(current[0]) + sum(len(a) for a, _ in arrays) + len(x) > maximum_rows:
            continue
        chosen.append(identifier)
        arrays.append((x, y))
    selected = SharedReplaySelection(tuple(chosen), tuple(wires[key] for key in chosen))
    if not arrays:
        return selected, None, None
    features, targets = compose_joint_packet(policy, current, tuple(arrays))
    detail = dict(
        control=policy.control,
        current_role=wake_role,
        retained_role="train",
        public_context=current_index,
        current_packet_rows=len(current[0]),
        retained_rows_available=sum(len(x) for x, _ in arrays),
        retained_train_rows_used=sum(len(x) for x, _ in arrays) if policy.control == "joint" else 0,
        current_rows_used=len(current[0]) if policy.control == "joint" else len(features),
        total_fit_rows=len(features),
        selected_train_ids=chosen,
        qualification="same physical rows, not matched SVD/total compute; current packet already paid in wake",
    )
    return selected, (features, targets), detail


def prepare_shared_rehearsal(policy, schema, selection, current, maximum_rows, wake_role):
    """Reuse certified coverage selection, never infer physical TRAIN authority.

    The stream certifies arrival/consent. This boundary validates all selected
    TRAIN packets before dropping exact duplicates already present in the wake.
    """
    if type(policy) is not SharedJointReplayPolicy or wake_role != "train":
        raise ValueError("shared joint replay requires explicit ordinary TRAIN policy")
    policy.__post_init__()
    _features(current[0], policy.feature_dim)
    validate_class_targets(current[1], len(current[0]), policy.num_classes)
    if (schema.observation_dim, schema.num_classes) != (policy.feature_dim, policy.num_classes):
        raise ValueError("shared joint replay schema differs")
    decoded = []
    wires = dict(zip(selection.sample_ids, selection.training_batches(), strict=True))
    for identifier in reversed(selection.sample_ids):
        pair = wires[identifier]
        episode = decode_episode(pair[0], pair[1], schema)
        if episode.role != "train":
            raise ValueError("shared joint replay retains only original TRAIN")
        x, y = episode_arrays(episode)
        _features(x, policy.feature_dim)
        validate_class_targets(y, len(x), policy.num_classes)
        decoded.append((identifier, x, y))
    arrays: list[tuple[Array, Array]] = []
    chosen = []
    seen = set()
    for identifier, x, y in decoded:
        key = (x.tobytes(), y.tobytes())
        covered = any(
            (current[0][i : i + len(x)] == x).all() and (current[1][i : i + len(y)] == y).all()
            for i in range(len(current[0]) - len(x) + 1)
        )
        if (
            covered
            or key in seen
            or len(current[0]) + sum(len(a) for a, _ in arrays) + len(x) > maximum_rows
        ):
            continue
        seen.add(key)
        chosen.append(identifier)
        arrays.append((x, y))
    selected = SharedReplaySelection(tuple(chosen), tuple(wires[k] for k in chosen))
    if not arrays:
        return selected, None, None
    features, targets = compose_shared_packet(
        current, tuple(arrays), policy.num_classes, control=policy.control
    )
    detail = dict(
        control=policy.control,
        current_role="train",
        retained_role="train",
        current_packet_rows=len(current[0]),
        retained_rows_available=sum(len(x) for x, _ in arrays),
        retained_train_rows_used=sum(len(x) for x, _ in arrays) if policy.control == "joint" else 0,
        current_rows_used=len(current[0]) if policy.control == "joint" else len(features),
        total_fit_rows=len(features),
        selected_train_ids=chosen,
        packet_array_bytes=features.nbytes + targets.nbytes,
        qualification="one additional shared-head fit; charge all matrix rows, including already-paid current rows; not matched compute",
    )
    return selected, (features, targets), detail
