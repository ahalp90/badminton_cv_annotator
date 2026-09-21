"""Executed synthetic checks for the read-only court-geometry critique.

Requires NumPy. No repository access, image search, matcher run, or real-frame
fit. The angle/bank/greedy rules follow evaluation/method_excerpts.md at 7299ff3.
Synthetic court corners use float64 metre coordinates, not stored float32.
Run: python synthetic_geometry_checks.py
"""
from itertools import combinations
import json
import numpy as np


def angles(lines: np.ndarray, points: np.ndarray,
           anchors: np.ndarray | None = None) -> np.ndarray:
    normals = lines[:, :2]
    if anchors is None:
        anchors = -lines[:, 2, None] * normals / np.sum(normals**2, axis=1)[:, None]
    tangents = np.stack((lines[:, 1], -lines[:, 0]), axis=1)
    rays = points[:, None, :2] - points[:, None, 2:] * anchors[None]
    dot = np.abs(np.einsum('pli,li->pl', rays, tangents))
    cross = np.abs(rays[..., 0] * tangents[:, 1] - rays[..., 1] * tangents[:, 0])
    result = np.degrees(np.arctan2(cross, dot))
    return np.where(np.linalg.norm(rays, axis=2) > 1e-12, result, 90.0)


def bank(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.array(list(combinations(range(len(lines)), 2)))
    intersections = np.cross(lines[pairs[:, 0]], lines[pairs[:, 1]])
    infinity = np.column_stack((lines[:, 1], -lines[:, 0], np.zeros(len(lines))))
    points = np.concatenate((intersections, infinity))
    norms = np.linalg.norm(points, axis=1)
    valid = norms > 1e-12
    return np.flatnonzero(valid), points[valid] / norms[valid, None]


def allocate(residuals: np.ndarray, ids: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    masks = residuals <= 1.5
    counts = masks.sum(axis=1)
    supports = np.maximum(0, 1 - residuals / 1.5).sum(axis=1)
    eligible = counts >= 2
    covered = np.zeros(masks.shape[1], dtype=bool)
    leaders, buckets = [], []
    for _ in range(16):
        indices = np.flatnonzero(eligible)
        if not len(indices):
            break
        novelty = (masks[indices] & ~covered).sum(axis=1)
        order = np.lexsort((ids[indices], -supports[indices], -counts[indices], -novelty))
        leader = int(indices[order[0]])
        leaders.append(leader)
        covered |= masks[leader]
        union = np.count_nonzero(masks[indices] | masks[leader], axis=1)
        intersection = np.count_nonzero(masks[indices] & masks[leader], axis=1)
        bucket = indices[intersection / union > .8]
        buckets.append(bucket)
        eligible[bucket] = False
    return leaders, buckets


def project(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.column_stack((xy, np.ones(len(xy)))) @ h.T
    if np.any(np.abs(p[:, 2]) < 1e-12):
        raise ValueError('Synthetic point projects to infinity')
    return p[:, :2] / p[:, 2, None]


def main() -> None:
    diagonal = float(np.hypot(960, 540))
    T = np.array([[diagonal, 0, 480], [0, diagonal, 270], [0, 0, 1.]])
    court = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]])

    # One physical line, two candidate VPs, and two different physical anchors.
    horizontal = np.array([[0., 1., 0.]])
    finite = np.array([[100., 1., 1.], [1., .1, 1.]])
    anchor_test = {
        'anchor_0_angles_deg': angles(horizontal, finite).ravel().tolist(),
        'anchor_99_angles_deg': angles(horizontal, finite, np.array([[99., 0.]])).ravel().tolist(),
    }

    exact_pencil = np.array([[1., 0., 0.], [0., 1., 0.]])
    exact_vp = np.array([[0., 0., 1.]])
    anchor_test['exact_origin_vp_foot_angles'] = angles(exact_pencil, exact_vp).ravel().tolist()
    anchor_test['exact_origin_vp_midpoint_angles'] = angles(
        exact_pencil, exact_vp, np.array([[0., 1.], [1., 0.]])).ravel().tolist()

    # Parallel short fragments, all tilted one degree by a coherent small error.
    epsilon = np.tan(np.deg2rad(1.))
    y = np.array([50., 191.6, 452.])
    pixel_lines = np.column_stack((np.full(3, -epsilon), np.ones(3), epsilon * 413.5 - y))
    normalised_lines = pixel_lines @ T
    normalised_lines /= np.linalg.norm(normalised_lines[:, :2], axis=1)[:, None]
    truth = np.array([1., 0., 0.])
    observed = np.array([1., epsilon, 0.])
    observed /= np.linalg.norm(observed)
    _, singular_values, vh = np.linalg.svd(normalised_lines)
    fitted = vh[-1]
    if fitted @ observed < 0:
        fitted = -fitted
    true_h = np.array([[70., 0., 200.], [0., 30., 50.], [0., 0., 1.]])
    # Exact coordinate-LS optimum with x VP=(1,epsilon,0), y VP=(0,1,0).
    alpha = 70 / (1 + epsilon**2)
    bad_h = np.array([[alpha, 0., 200 + (70 - alpha) * 3.05],
                      [alpha * epsilon, 30., 50 - alpha * epsilon * 3.05],
                      [0., 0., 1.]])
    affine_test = {
        'angles_before_after_deg': angles(normalised_lines, np.stack((truth, observed))).tolist(),
        'algebraic_rms_before_after': [float(np.sqrt(np.mean((normalised_lines @ v)**2)))
                                       for v in (truth, observed)],
        'svd_direction': fitted.tolist(),
        'singular_values': singular_values.tolist(),
        'endpoint_deviation_for_20px_fragment': float(10 * epsilon),
        'true_h_pixel': true_h.tolist(),
        'biased_direction_best_coordinate_ls_h_pixel': bad_h.tolist(),
        'best_ls_max_corner_error_px': float(np.linalg.norm(project(bad_h, court) - project(true_h, court), axis=1).max()),
    }

    # A second noise construction stays inside the similarity-camera subclass:
    # both true directions and both biased directions are mutually orthogonal.
    theta = np.deg2rad(1.)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    a_rotation = 30 * np.cos(theta) * rotation
    centre_court = court.mean(axis=0)
    t_rotation = np.array([300., 50.]) + (30 * np.eye(2) - a_rotation) @ centre_court
    h_rotation_true = np.array([[30., 0., 300.], [0., 30., 50.], [0., 0., 1.]])
    h_rotation_fit = np.column_stack((a_rotation, t_rotation))
    h_rotation_fit = np.vstack((h_rotation_fit, [0., 0., 1.]))
    horizontal_lines = np.column_stack((np.full(3, -epsilon), np.ones(3),
                                        epsilon * 391.5 - np.array([50., 191.6, 452.])))
    vertical_lines = np.column_stack((np.ones(3), np.full(3, epsilon),
                                      -np.array([300., 391.5, 483.]) - epsilon * 253.))
    rotation_groups = []
    for rows, true_point, biased_point in (
        (horizontal_lines, [1., 0., 0.], [1., epsilon, 0.]),
        (vertical_lines, [0., 1., 0.], [-epsilon, 1., 0.]),
    ):
        rows = rows @ T
        rows /= np.linalg.norm(rows[:, :2], axis=1)[:, None]
        vp = np.array([true_point, biased_point])
        vp /= np.linalg.norm(vp, axis=1)[:, None]
        _, sv, right = np.linalg.svd(rows)
        rotation_groups.append({
            'angles_before_after_deg': angles(rows, vp).tolist(),
            'algebraic_rms_before_after': np.sqrt(np.mean((rows @ vp.T)**2, axis=0)).tolist(),
            'svd_algebraic_rms': float(np.sqrt(np.mean((rows @ right[-1])**2))),
            'singular_values': sv.tolist(),
        })
    rotation_test = {
        'groups': rotation_groups,
        'true_h_pixel': h_rotation_true.tolist(),
        'best_coordinate_ls_h_pixel': h_rotation_fit.tolist(),
        'all_corner_errors_px': np.linalg.norm(project(h_rotation_fit, court)
                                                - project(h_rotation_true, court), axis=1).tolist(),
    }

    # A complete 3-line bank. Its first suppression bucket contains two very
    # different finite VPs and infinity; its minimum-residual representative
    # is not its greedy leader. This is NOT a complete real-frame court search.
    lines = np.array([[0., 1., 0.], [-.005, 1., .005], [.005, 1., .005]])
    ids, points = bank(lines)
    residuals = angles(lines, points)
    leaders, buckets = allocate(residuals, ids)
    leader = leaders[0]
    leader_mask = residuals[leader] <= 1.5
    scores = np.mean(np.minimum(residuals[:, leader_mask], 1.5)**2, axis=1)
    representative = min(buckets[0], key=lambda i: (scores[i], ids[i]))
    assert ids[leader] == 0 and ids[representative] == 3
    assert set(ids[buckets[0]]) == {0, 1, 3, 4, 5}

    # Hold the OTHER direction (0,1,0) available and fixed only for this
    # synthetic downstream comparison. The leader fits H_true exactly.
    # The representative forces an axis-aligned affine court. Its globally
    # optimal coordinate-LS fit is a pair of independent linear regressions.
    h_true = np.array([[.05, 0., 0.], [0., .015, -.005], [.05, 0., 1.]])
    reference = project(h_true, court)
    ax, tx = np.linalg.lstsq(np.column_stack((court[:, 0], np.ones(4))), reference[:, 0], rcond=None)[0]
    by, ty = np.linalg.lstsq(np.column_stack((court[:, 1], np.ones(4))), reference[:, 1], rcond=None)[0]
    h_rep = np.array([[ax, 0., tx], [0., by, ty], [0., 0., 1.]])
    max_error = float(np.linalg.norm(project(h_rep, court) - reference, axis=1).max() * diagonal)
    assert abs(max_error - 25.22792923898848) < 1e-8
    bucket_test = {
        'candidate_ids': ids.tolist(),
        'points_normalised': points.tolist(),
        'angles_deg': residuals.tolist(),
        'leader_ids': ids[leaders].tolist(),
        'first_bucket_ids': ids[buckets[0]].tolist(),
        'first_representative_id': int(ids[representative]),
        'capped_squared_score_leader_rep': [float(scores[leader]), float(scores[representative])],
        'true_h_normalised': h_true.tolist(),
        'representative_best_coordinate_ls_h_normalised': h_rep.tolist(),
        'representative_max_corner_error_working_px': max_error,
    }

    # Pair conditioning: a 0.1px normal-offset error on one intersecting line.
    phi = np.deg2rad(.5)
    N = np.array([[0., 1.], [-np.sin(phi), np.cos(phi)]])
    pair_test = {'line_angle_deg': .5, 'normal_offset_px': .1,
                 'intersection_shift_px': np.linalg.solve(N, [0., .1]).tolist()}
    print(json.dumps({'anchor': anchor_test, 'affine_noise': affine_test,
                      'similarity_rotation': rotation_test, 'bucket': bucket_test, 'pair_conditioning': pair_test}, indent=2))


if __name__ == '__main__':
    main()
