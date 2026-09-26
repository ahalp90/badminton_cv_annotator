#!/usr/bin/env python3
"""Show the centre-assignment capture band in the pinned court refit.

This is a minimal specialization of:

* stripe_observations.resolve_fragments: maximum reverse support over
  [centre, negative edge, positive edge], in that order;
* run_probe.relabel: polarity swaps only negative/positive edges and leaves
  centre assignments unchanged;
* fixed_stripe_refit.prepare: selected points must lie within 5 working pixels.

Coordinate conventions
----------------------
Court x increases left -> right and court y increases far -> near. Image x
increases right and image y increases down. The saved SS03-34 corners are from
commit 13f02fdbf8954ead15dbc7241a696a314473f9c5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
STRIPE_WIDTH_M = 0.04
HALF_STRIPE_M = STRIPE_WIDTH_M / 2
DISTANCE_SIGMA_PX = 2.0
PRESENT_SUPPORT = 0.55
SUPPORT_DISTANCE_PX = 5.0
MATCH_ANGLE_DEG = 5.0

LEFT_SINGLES_CENTRE_M = 0.48
NEAR_SHORT_CENTRE_M = 8.70

# Saved edge-polarity probe result, fits.polarity.corners_px.
SCENE34_POLARITY_CORNERS_PX = np.array(
    [
        [317.6658884157470, 215.4113812134875],
        [642.4361527459146, 214.9527820925585],
        [754.2593087957458, 492.1533939673320],
        [202.9437733454317, 492.3931133737459],
    ],
    dtype=float,
)

# Saved median signed brightness differences at +/-1 working pixel.
FRAGMENT_111_CONTRAST = -69.99860382080078
FRAGMENT_179_CONTRAST = +80.87715148925781

POSITION_NAMES = ("centre", "negative_edge", "positive_edge")
POSITION_OFFSETS_M = np.array([0.0, -HALF_STRIPE_M, +HALF_STRIPE_M])


@dataclass(frozen=True)
class OrientationResult:
    image_sign: int
    minimum_margin_after_match_tolerance_deg: float
    polarity_position: int


def homography_from_four(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Solve a 3x3 homography with h[2,2] fixed to one."""
    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    parameters = np.linalg.solve(np.asarray(rows), np.asarray(values))
    return np.append(parameters, 1.0).reshape(3, 3)


def project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = homogeneous @ homography.T
    return mapped[:, :2] / mapped[:, 2, None]


def canonical_direction(vector: np.ndarray) -> np.ndarray:
    """Match assignment.prepare_observations' lexicographic endpoint order."""
    vector = np.asarray(vector, dtype=float)
    if vector[0] < 0 or (vector[0] == 0 and vector[1] < 0):
        vector = -vector
    return vector / np.linalg.norm(vector)


def orientation_over_marking(
    homography: np.ndarray,
    points_m: np.ndarray,
    tangent_step_m: np.ndarray,
    positive_axis_step_m: np.ndarray,
    contrast: float,
) -> OrientationResult:
    """Infer edge position from polarity and verify the sign is angle-robust."""
    cosines = []
    signs = []
    for point_m in points_m:
        origin = project(homography, point_m[None])[0]
        tangent_end = project(homography, (point_m + tangent_step_m)[None])[0]
        direction = canonical_direction(tangent_end - origin)
        normal = np.array([-direction[1], direction[0]])
        positive_axis = project(homography, (point_m + positive_axis_step_m)[None])[0] - origin
        cosine = float(np.dot(positive_axis, normal) / np.linalg.norm(positive_axis))
        cosines.append(cosine)
        signs.append(int(np.sign(cosine)))

    assert len(set(signs)) == 1 and signs[0] != 0
    angles_to_normal = np.degrees(np.arccos(np.clip(np.asarray(cosines), -1.0, 1.0)))
    margin = float(np.min(90.0 - angles_to_normal - MATCH_ANGLE_DEG))
    assert margin > 0.0

    image_sign = signs[0]
    contrast_sign = int(np.sign(contrast))
    # run_probe.expected_bright_side returns image_sign for position 1 and
    # -image_sign for position 2.
    matches = [
        position
        for position, expected in ((1, image_sign), (2, -image_sign))
        if expected == contrast_sign
    ]
    assert len(matches) == 1
    return OrientationResult(image_sign, margin, matches[0])


def projected_half_width(
    homography: np.ndarray,
    centres_m: np.ndarray,
    edge_offset_m: np.ndarray,
) -> np.ndarray:
    edge = centres_m + edge_offset_m
    return np.linalg.norm(project(homography, edge) - project(homography, centres_m), axis=1)


def reverse_response(distance_px: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-0.5 * np.square(np.asarray(distance_px) / DISTANCE_SIGMA_PX))


def geometric_position(parent_shift_px: float, model_half_width_px: float) -> int:
    """Choose [centre, negative edge, positive edge] for a true positive edge."""
    observed = model_half_width_px
    predicted = parent_shift_px + np.array([0.0, -model_half_width_px, +model_half_width_px])
    distances = np.abs(observed - predicted)
    responses = reverse_response(distances)
    return int(np.argmax(responses))  # Exact ties choose centre because it is first.


def polarity_relabel(position: int, true_edge_position: int = 2) -> int:
    """Specialize run_probe.relabel: centre is unresolved; wrong edge is swapped."""
    if position == 0:
        return 0
    return true_edge_position


def fit_affine(model_x: np.ndarray, observed_u: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    design = np.column_stack((model_x, np.ones_like(model_x)))
    root = np.sqrt(weights)
    slope, intercept = np.linalg.lstsq(design * root[:, None], observed_u * root, rcond=None)[0]
    residual = observed_u - (slope * model_x + intercept)
    return float(slope), float(intercept), float(weights @ np.square(residual))


def anchored_coordinate_map(model: float, physical: float, opposite_boundary: float) -> tuple[float, float]:
    scale = (opposite_boundary - physical) / (opposite_boundary - model)
    offset = physical - scale * model
    return scale, offset


def main() -> None:
    court_corners = np.array(
        [[0.0, 0.0], [COURT_WIDTH_M, 0.0],
         [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
        dtype=float,
    )
    homography = homography_from_four(court_corners, SCENE34_POLARITY_CORNERS_PX)

    y_grid = np.linspace(0.0, COURT_LENGTH_M, 201)
    left_centres = np.column_stack((np.full_like(y_grid, LEFT_SINGLES_CENTRE_M), y_grid))
    left_orientation = orientation_over_marking(
        homography,
        left_centres,
        tangent_step_m=np.array([0.0, 0.01]),
        positive_axis_step_m=np.array([STRIPE_WIDTH_M, 0.0]),
        contrast=FRAGMENT_111_CONTRAST,
    )
    left_q = projected_half_width(homography, left_centres, np.array([HALF_STRIPE_M, 0.0]))

    x_grid = np.linspace(0.0, COURT_WIDTH_M, 201)
    near_centres = np.column_stack((x_grid, np.full_like(x_grid, NEAR_SHORT_CENTRE_M)))
    near_orientation = orientation_over_marking(
        homography,
        near_centres,
        tangent_step_m=np.array([0.01, 0.0]),
        positive_axis_step_m=np.array([0.0, STRIPE_WIDTH_M]),
        contrast=FRAGMENT_179_CONTRAST,
    )
    near_q = projected_half_width(homography, near_centres, np.array([0.0, -HALF_STRIPE_M]))

    assert left_orientation.polarity_position == 2  # inner/positive edge
    assert near_orientation.polarity_position == 1  # far/negative edge

    def print_capture(name: str, q: np.ndarray) -> None:
        maximum_centre_residual = float(q.max() / 2.0)
        minimum_reverse_support = float(reverse_response(maximum_centre_residual))
        print(f"{name} projected centre-to-edge q: {q.min():.6f} .. {q.max():.6f} px")
        print(f"  centre-capture parent-shift lower endpoint q/2: {q.min()/2:.6f} .. {q.max()/2:.6f} px")
        print(f"  centre-capture parent-shift upper endpoint 3q/2: {1.5*q.min():.6f} .. {1.5*q.max():.6f} px")
        print(f"  captured point-to-centre residual <= {maximum_centre_residual:.6f} px")
        print(f"  corresponding reverse support >= {minimum_reverse_support:.6f}")
        assert maximum_centre_residual < SUPPORT_DISTANCE_PX
        assert minimum_reverse_support > PRESENT_SUPPORT

    print("SS03-34 polarity identity from saved contrast and code orientation")
    print(
        "  fragment 111: image_sign=+1, contrast<0 -> positive/inner edge "
        f"(position {left_orientation.polarity_position}); orientation margin after 5 deg tolerance "
        f"{left_orientation.minimum_margin_after_match_tolerance_deg:.3f} deg"
    )
    print(
        "  fragment 179: image_sign=+1, contrast>0 -> negative/far edge "
        f"(position {near_orientation.polarity_position}); orientation margin after 5 deg tolerance "
        f"{near_orientation.minimum_margin_after_match_tolerance_deg:.3f} deg"
    )
    print()
    print_capture("left-singles", left_q)
    print_capture("near-short", near_q)

    # Exact two-constraint affine specialization. A true positive/inner edge is
    # geometrically assigned under three parent shifts. The middle shift lies in
    # the centre-capture band, so polarity cannot fix it and the zero-loss fit is inset.
    true_scale = 60.0
    true_left = 300.0
    stripe_centre_m = LEFT_SINGLES_CENTRE_M
    true_edge_m = stripe_centre_m + HALF_STRIPE_M
    model_half_width_px = true_scale * HALF_STRIPE_M
    observed = true_scale * np.array([true_edge_m, COURT_WIDTH_M]) + true_left
    weights = np.array([0.73, 0.27])

    print("\nMinimal assignment -> polarity -> frozen-refit calculation")
    for parent_shift in (0.5, 1.0, 2.0):
        geometric = geometric_position(parent_shift, model_half_width_px)
        final_position = polarity_relabel(geometric)
        model_edge_m = stripe_centre_m + POSITION_OFFSETS_M[final_position]
        _, boundary, loss = fit_affine(np.array([model_edge_m, COURT_WIDTH_M]), observed, weights.copy())
        print(
            f"  parent inward shift {parent_shift:.1f} px: geometry={POSITION_NAMES[geometric]}, "
            f"after polarity={POSITION_NAMES[final_position]}, left boundary={boundary:.6f} px, "
            f"inset={boundary-true_left:+.6f} px, loss={loss:.3g}"
        )

    # Projective proxy retained from the first report: correct only fragment 111's
    # model coordinate while holding the opposite outside boundary fixed.
    x_scale, x_offset = anchored_coordinate_map(LEFT_SINGLES_CENTRE_M, true_edge_m, COURT_WIDTH_M)
    wrong_world = court_corners.copy()
    wrong_world[:, 0] = x_scale * wrong_world[:, 0] + x_offset
    correction = project(homography, court_corners) - project(homography, wrong_world)
    print(
        "\nSS03-34 fragment-111 two-line projective proxy, upper-left correction: "
        f"({correction[0,0]:+.6f}, {correction[0,1]:+.6f}) working px"
    )
    assert correction[0, 0] < -1.0

    # Width/localisation generalization: with model half-width q_m and an observed
    # effective edge offset q_a, centre wins whenever |q_a-delta| <= q_m/2.
    print("\nGeneralized capture condition")
    print("  centre wins when |effective_edge_offset - parent_shift| <= model_half_width / 2")
    print("  therefore a width/localisation mismatch can select centre even at zero parent shift")


if __name__ == "__main__":
    main()
