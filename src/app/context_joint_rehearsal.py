"""Prepare one same-context packet for the existing replay consolidation.

Retained encoded episodes must stay TRAIN. Current probe inputs are transient
and never enter the bank. No runtime, fit, budget authority or data-role rewrite.
"""

from src.core.context_joint_replay import packet_context, compose_joint_packet
from src.core.demonstration_episode import decode_episode, episode_arrays
from src.core.shared_replay_schedule import SharedReplaySelection
from src.core.coupled_energy import Array


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
