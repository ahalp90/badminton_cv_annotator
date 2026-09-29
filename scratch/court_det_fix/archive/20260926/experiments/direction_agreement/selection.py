"""Coverage allocation with bucket records, and precision representatives (E2)."""

from __future__ import annotations

import numpy as np
from vp_pruning import Settings

FOOT_ANCHOR = 'original_foot'
MIDPOINT_ANCHOR = 'projected_fragment_midpoint'
LEADER_RULE = 'original_greedy_leader'
PRECISION_RULE = 'lowest_mean_clipped_squared_angle_over_leader_support'
ARM_RULES = {
    'B': (FOOT_ANCHOR, LEADER_RULE),
    'M': (MIDPOINT_ANCHOR, LEADER_RULE),
    'R': (FOOT_ANCHOR, PRECISION_RULE),
    'MR': (MIDPOINT_ANCHOR, PRECISION_RULE),
}
MINIMUM_DIRECTIONS = 2


def support_arrays(residuals: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Membership masks, support counts and soft support as vp_pruning.estimate derives them.

    :param residuals: (candidates, lines) angles in degrees.
    :return: boolean masks, integer counts and ``sum(max(0, 1 - angle / angle_deg))``.
    """
    masks = residuals <= angle_deg
    counts = masks.sum(axis=1).astype(int)
    supports = np.maximum(0, 1 - residuals / angle_deg).sum(axis=1)
    return masks, counts, supports


def retain_pencils_with_buckets(
    masks: np.ndarray, counts: np.ndarray, supports: np.ndarray, candidate_ids: np.ndarray, settings: Settings,
) -> tuple[list[int], np.ndarray, list[dict]]:
    """Private copy of vp_pruning.retain_pencils that records each leader's suppression bucket.

    A bucket is the leader plus every currently eligible candidate its IoU suppression
    removes; the leader's own IoU is 1, so it is always a member. Buckets are disjoint
    because suppressed candidates leave the eligible set. Candidates still capped when
    the loop ends belong to no bucket.
    """
    if settings.pencil_selection not in ('ranked', 'coverage'):
        raise ValueError(f'Unknown pencil selection: {settings.pencil_selection}')
    eligible = counts >= 2
    status = np.where(eligible, 'capped', 'below_two_lines').astype('U16')
    covered = np.zeros(masks.shape[1], dtype=bool)
    retained: list[int] = []
    buckets: list[dict] = []
    for _ in range(settings.pencils):
        indices = np.flatnonzero(eligible)
        if not len(indices):
            break
        novelty = (
            (masks[indices] & ~covered).sum(axis=1) if settings.pencil_selection == 'coverage' else counts[indices]
        )
        order = np.lexsort((candidate_ids[indices], -supports[indices], -counts[indices], -novelty))
        index = int(indices[order[0]])
        retained.append(index)
        covered |= masks[index]
        union = np.count_nonzero(masks[indices] | masks[index], axis=1)
        intersection = np.count_nonzero(masks[indices] & masks[index], axis=1)
        redundant = indices[intersection / union > settings.overlap]
        eligible[redundant] = False
        status[redundant] = 'redundant'
        status[index] = 'retained'
        buckets.append({'leader_row': index, 'member_rows': redundant.tolist()})
    return retained, status, buckets


def precision_scores(residuals: np.ndarray, leader_mask: np.ndarray, rows: np.ndarray, angle_deg: float) -> np.ndarray:
    """Declared hypothesis score: ``mean(min(angle, angle_deg) ** 2)`` over the leader's support lines."""
    clipped = np.minimum(residuals[np.ix_(rows, np.flatnonzero(leader_mask))], angle_deg)
    return np.mean(np.square(clipped), axis=1)


def choose_precise(
    residuals: np.ndarray, leader_mask: np.ndarray, rows: np.ndarray, candidate_ids: np.ndarray, angle_deg: float,
) -> tuple[int, np.ndarray]:
    """Lowest precision score in the bucket; equal scores break by original candidate ID."""
    scores = precision_scores(residuals, leader_mask, rows, angle_deg)
    order = np.lexsort((candidate_ids[rows], scores))
    return int(rows[order[0]]), scores


def arm_record(
    arm: str, buckets: list[dict], residuals: np.ndarray, masks: np.ndarray, counts: np.ndarray,
    status: np.ndarray, candidate_ids: np.ndarray, candidates: np.ndarray, generators: list[list[int]],
    transform: np.ndarray, angle_deg: float,
) -> dict:
    """Assemble one arm from an allocation and its residual matrix.

    Leaders alone came from the coverage allocation. The representative is the leader for
    the leader rule, or the precision choice within the leader's bucket. Every bucket
    member's score is kept for all arms so the alternatives stay in evidence.
    """
    anchor_rule, representative_rule = ARM_RULES[arm]
    groups = []
    chosen_rows = []
    for group_index, bucket in enumerate(buckets):
        leader_row = bucket['leader_row']
        rows = np.asarray(bucket['member_rows'], dtype=int)
        leader_mask = masks[leader_row]
        precise_row, scores = choose_precise(residuals, leader_mask, rows, candidate_ids, angle_deg)
        chosen_row = precise_row if representative_rule == PRECISION_RULE else leader_row
        chosen_rows.append(chosen_row)
        groups.append({
            'group_index': group_index,
            'leader_candidate_id': int(candidate_ids[leader_row]),
            'leader_row': int(leader_row),
            'leader_generators': generators[leader_row],
            'leader_support_mask': leader_mask.tolist(),
            'leader_support_count': int(counts[leader_row]),
            'bucket_candidate_ids': candidate_ids[rows].tolist(),
            'bucket_scores': scores.tolist(),
            'bucket_size': len(rows),
            'chosen_candidate_id': int(candidate_ids[chosen_row]),
            'chosen_row': int(chosen_row),
            'chosen_generators': generators[chosen_row],
            'chosen_score': float(scores[np.flatnonzero(rows == chosen_row)[0]]),
            'chosen_support_mask': masks[chosen_row].tolist(),
            'chosen_support_count': int(counts[chosen_row]),
            'chosen_is_leader': bool(chosen_row == leader_row),
        })
    chosen = np.asarray(chosen_rows, dtype=int)
    points_normalised = candidates[chosen]
    points_working = points_normalised @ transform.T
    status_values, status_counts = np.unique(status, return_counts=True)
    direction_count = len(chosen)
    return {
        'arm': arm,
        'anchor_rule': anchor_rule,
        'representative_rule': representative_rule,
        'direction_count': direction_count,
        'matcher_eligible': direction_count >= MINIMUM_DIRECTIONS,
        'court_result': None if direction_count >= MINIMUM_DIRECTIONS else {
            'status': 'empty', 'reason': f'fewer than {MINIMUM_DIRECTIONS} directions survive allocation'},
        'leader_candidate_ids': [group['leader_candidate_id'] for group in groups],
        'representative_candidate_ids': candidate_ids[chosen].tolist(),
        'representatives_differ_from_leaders': int(sum(not group['chosen_is_leader'] for group in groups)),
        'points_normalised': points_normalised.tolist(),
        'points_working': points_working.tolist(),
        'support_masks': masks[chosen].tolist(),
        'support_counts': counts[chosen].tolist(),
        'allocation_status_counts': dict(zip(status_values.tolist(), status_counts.tolist(), strict=True)),
        'allocation_candidate_status': status.tolist(),
        'groups': groups,
    }
