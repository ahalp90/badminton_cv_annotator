#!/usr/bin/env python3
"""Minimal specialization of the fixed-stripe court objective.

Coordinate system
-----------------
Court x is in metres, increasing left -> right.  The physical outside doubles
boundaries are x=0 and x=6.10.  Image u is in working pixels, increasing right.
For vertical, interior samples the repository's point-to-finite-segment loss
reduces exactly to weighted squared residuals of an affine map u = a*x + b.

The first calculation proves that a physical paint edge frozen as the stripe
centre can give zero fitting loss while the inferred outside boundary is inset.
The second calculation uses the saved SS03-34 polarity-fit corners from commit
13f02fdbf8954ead15dbc7241a696a314473f9c5 to give a projective, two-line proxy
for the two retained position-0 fragments discussed in the accompanying report.
It is not a substitute for the real weighted refit.
"""

from __future__ import annotations

import numpy as np

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
STRIPE_WIDTH_M = 0.04

# Saved scratch/court_det_fix/edge_polarity/scene34_smoke.json.gz,
# fits.polarity.corners_px, pinned to the commit above.
SCENE34_POLARITY_CORNERS_PX = np.array(
    [
        [317.6658884157470, 215.4113812134875],
        [642.4361527459146, 214.9527820925585],
        [754.2593087957458, 492.1533939673320],
        [202.9437733454317, 492.3931133737459],
    ],
    dtype=float,
)


def fit_affine(model_x: np.ndarray, observed_u: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """Return slope, intercept and normalized weighted squared pixel loss."""
    model_x = np.asarray(model_x, dtype=float)
    observed_u = np.asarray(observed_u, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    design = np.column_stack((model_x, np.ones_like(model_x)))
    root = np.sqrt(weights)
    slope, intercept = np.linalg.lstsq(design * root[:, None], observed_u * root, rcond=None)[0]
    residual = observed_u - (slope * model_x + intercept)
    return float(slope), float(intercept), float(weights @ np.square(residual))


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


def anchored_coordinate_map(model: float, physical: float, opposite_boundary: float) -> tuple[float, float]:
    """Map model coordinate to physical coordinate while fixing the opposite boundary."""
    scale = (opposite_boundary - physical) / (opposite_boundary - model)
    offset = physical - scale * model
    return scale, offset


def main() -> None:
    # Exact 1-D counterexample: left-singles inner edge is x=0.50 m, but the
    # frozen interpretation says stripe centre x=0.48 m.  The right outside
    # boundary is correctly assigned.  Any positive weights give an exact fit.
    true_scale_px_per_m = 60.0
    true_left_boundary_px = 300.0
    physical_inner_edge_m = 0.50
    frozen_centre_m = 0.48
    observed = true_scale_px_per_m * np.array([physical_inner_edge_m, COURT_WIDTH_M]) + true_left_boundary_px
    weights = np.array([0.73, 0.27])

    _, correct_boundary, correct_loss = fit_affine(
        np.array([physical_inner_edge_m, COURT_WIDTH_M]), observed, weights
    )
    _, inset_boundary, inset_loss = fit_affine(
        np.array([frozen_centre_m, COURT_WIDTH_M]), observed, weights
    )
    expected_inset = (
        true_scale_px_per_m
        * (physical_inner_edge_m - frozen_centre_m)
        * COURT_WIDTH_M
        / (COURT_WIDTH_M - frozen_centre_m)
    )

    np.testing.assert_allclose(correct_boundary, true_left_boundary_px, atol=1e-10, rtol=0)
    np.testing.assert_allclose(correct_loss, 0.0, atol=1e-20, rtol=0)
    np.testing.assert_allclose(inset_loss, 0.0, atol=1e-20, rtol=0)
    np.testing.assert_allclose(inset_boundary - true_left_boundary_px, expected_inset, atol=1e-10, rtol=0)

    print("1-D exact specialization")
    print(f"  correct edge: boundary={correct_boundary:.6f} px, loss={correct_loss:.3g} px^2")
    print(f"  edge frozen as centre: boundary={inset_boundary:.6f} px, loss={inset_loss:.3g} px^2")
    print(f"  zero-loss inset={inset_boundary - true_left_boundary_px:.6f} px")

    court_corners = np.array(
        [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]]
    )
    homography = homography_from_four(court_corners, SCENE34_POLARITY_CORNERS_PX)
    projected_corners = project(homography, court_corners)

    # Fragment 111 hypothesis: left-singles inner edge x=0.50 was frozen as
    # centre x=0.48, while the right outside boundary remains fixed.
    x_scale, x_offset = anchored_coordinate_map(0.48, 0.50, COURT_WIDTH_M)
    x_wrong_world = court_corners.copy()
    x_wrong_world[:, 0] = x_scale * x_wrong_world[:, 0] + x_offset
    x_correction = projected_corners - project(homography, x_wrong_world)

    # Fragment 179 hypothesis: near-short negative edge y=8.68 was frozen as
    # centre y=8.70, while the near outside boundary remains fixed.
    y_scale, y_offset = anchored_coordinate_map(8.70, 8.68, COURT_LENGTH_M)
    y_wrong_world = court_corners.copy()
    y_wrong_world[:, 1] = y_scale * y_wrong_world[:, 1] + y_offset
    y_correction = projected_corners - project(homography, y_wrong_world)

    both_wrong_world = court_corners.copy()
    both_wrong_world[:, 0] = x_scale * both_wrong_world[:, 0] + x_offset
    both_wrong_world[:, 1] = y_scale * both_wrong_world[:, 1] + y_offset
    both_correction = projected_corners - project(homography, both_wrong_world)

    # Full projected stripe widths under the saved fit.  Both are below the
    # repository's 4 px RESOLVABLE_WIDTH_PX threshold throughout the line.
    y_grid = np.linspace(0.0, COURT_LENGTH_M, 101)
    left_outer = project(homography, np.column_stack((np.full_like(y_grid, 0.46), y_grid)))
    left_inner = project(homography, np.column_stack((np.full_like(y_grid, 0.50), y_grid)))
    left_width = np.linalg.norm(left_inner - left_outer, axis=1)
    x_grid = np.linspace(0.0, COURT_WIDTH_M, 101)
    near_far_edge = project(homography, np.column_stack((x_grid, np.full_like(x_grid, 8.68))))
    near_near_edge = project(homography, np.column_stack((x_grid, np.full_like(x_grid, 8.72))))
    near_width = np.linalg.norm(near_near_edge - near_far_edge, axis=1)

    print("\nSS03-34 projective two-line proxy")
    print(f"  left-singles projected 40 mm width: {left_width.min():.3f}..{left_width.max():.3f} px")
    print(f"  near-short projected 40 mm width: {near_width.min():.3f}..{near_width.max():.3f} px")
    print(f"  fragment 111-like correction, upper-left: ({x_correction[0,0]:+.3f}, {x_correction[0,1]:+.3f}) px")
    print(f"  fragment 179-like correction, upper-left: ({y_correction[0,0]:+.3f}, {y_correction[0,1]:+.3f}) px")
    print(f"  combined proxy correction, upper-left: ({both_correction[0,0]:+.3f}, {both_correction[0,1]:+.3f}) px")

    assert left_width.max() < 4.0
    assert near_width.max() < 4.0
    assert x_correction[0, 0] < -1.0 and abs(x_correction[0, 1]) < 0.05
    assert y_correction[0, 1] > 0.5  # Down, contrary to the requested extra upward shift.


if __name__ == "__main__":
    main()
