"""Small baseline-versus-patch probes; run with the project's cicd Python."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scratch/court_det_fix/frozen_helpers_20260914/legacy"))

from experiments.annotator.independent_court import assignment, detector, paint_geometry
from experiments.annotator.independent_court import stripe_observations as stripes
import zone_net


def old_interval(homography, inverse, interval, centre_samples, observations, size, centres, tolerance):
    normal_m = np.array([1.0, 0.0]) if interval < 6 else np.array([0.0, 1.0])
    offsets = paint_geometry.POSITION_OFFSETS_M[:, None] * normal_m
    shifted_segments = paint_geometry.positioned_segments(centres, np.full(3, interval), np.arange(3))
    projected, _ = detector.project(homography[None], shifted_segments)
    projected = projected.reshape(3, 2, 2)
    vectors = projected[:, 1] - projected[:, 0]
    directions = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    compatible = np.abs(directions @ observations.directions.T) >= np.cos(np.deg2rad(assignment.MATCH_ANGLE_DEG))
    court_samples, _ = detector.project(inverse[None], centre_samples)
    shifted_samples, _ = detector.project(homography[None], court_samples[0][None] + offsets[:, None])
    shifted_samples = shifted_samples.reshape(3, len(centre_samples), 2)
    inside = ((shifted_samples >= -tolerance)
              & (shifted_samples <= np.asarray(size) - 1 + tolerance)).all(axis=2)
    forward = []
    for position in range(len(paint_geometry.POSITION_OFFSETS_M)):
        distances = assignment.distances_to_segments(shifted_samples[position], observations.segments)
        response = np.exp(-0.5 * np.square(distances / assignment.DISTANCE_SIGMA_PX))
        forward.append(np.where(inside[position, :, None] & compatible[position, None], response, 0.0))
    distances = assignment.distances_to_segments(observations.samples.reshape(-1, 2), projected)
    distances = distances.reshape(len(observations.segments), observations.samples.shape[1], 3)
    distances = np.where(compatible.T[:, None], distances, np.inf)
    width = np.linalg.norm(shifted_samples[1] - shifted_samples[2], axis=1)
    resolvable = inside[1] & inside[2] & (width >= stripes.RESOLVABLE_WIDTH_PX)
    return np.asarray(forward), distances, resolvable


def exact_arrays(left, right):
    return all(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
               for a, b in zip(left, right, strict=True))


def main():
    size = (800, 450)
    h = np.array([[50., 0., 50.], [0., 25., 50.], [0., 0., 1.]])
    invalid = h.copy()
    invalid[0, 2] = 2000.
    transforms = np.stack([h, invalid])
    feet = np.array([[[100., 100.], [200., 250.]]])
    from importlib import import_module
    sys.path.insert(0, str(ROOT / "scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis"))
    valid, _ = import_module("scan_population").geometry(transforms, size)
    old_one, old_two = zone_net.player_fractions(transforms, feet)
    new_one = np.full(2, np.nan)
    new_two = np.full(2, np.nan)
    new_one[valid], new_two[valid] = zone_net.player_fractions(transforms[valid], feet)
    print("C1 valid", valid.tolist(), "old", old_one.tolist(), old_two.tolist(),
          "new", new_one.tolist(), new_two.tolist())
    print("C1 valid-row bits", exact_arrays((old_one[valid], old_two[valid]),
                                           (new_one[valid], new_two[valid])))
    print("C1 empty shapes", [a.shape for a in (np.array([]), np.full(0, np.nan))])

    rng = np.random.default_rng(20260924)
    batch = np.concatenate((np.repeat(h[None], 16, axis=0), invalid[None]))
    batch[:16, :2, 2] += rng.uniform(-10, 10, (16, 2))
    full = zone_net.player_fractions(batch, np.repeat(feet, 5, axis=0))
    part = zone_net.player_fractions(batch[:16], np.repeat(feet, 5, axis=0))
    print("C1 16-row subset bits", exact_arrays((full[0][:16], full[1][:16]), part))

    segment_sets = {
        "empty": np.empty((0, 2, 2)),
        "no_compatible": np.array([[[100., 100.], [200., 200.]]]),
        "mixed": np.array([[[100., 100.], [200., 200.]], [[100., 40.], [100., 350.]],
                           [[50., 100.], [600., 100.]]]),
    }
    for name, segments in segment_sets.items():
        observations = assignment.prepare_observations(segments, size)
        for centres_name, centres in (("default", detector.SEGMENTS_M),
                                       ("paint", paint_geometry.CENTRE_SEGMENTS_M)):
            for interval in (0, 7):
                if name == "no_compatible":
                    projected, _ = detector.project(h[None], paint_geometry.positioned_segments(
                        centres, np.full(3, interval), np.arange(3)))
                    projected = projected.reshape(3, 2, 2)
                    vectors = projected[:, 1] - projected[:, 0]
                    directions = vectors / np.linalg.norm(vectors, axis=1)[:, None]
                    assert not (np.abs(directions @ observations.directions.T)
                                >= np.cos(np.deg2rad(assignment.MATCH_ANGLE_DEG))).any()
                for tolerance in (0., 3.5):
                    endpoint = centres[interval]
                    samples, _ = detector.project(h[None],
                                                  endpoint[0] + np.linspace(0, 1, 7)[:, None]
                                                  * (endpoint[1] - endpoint[0]))
                    args = (h, np.linalg.inv(h), interval, samples[0], observations, size, centres, tolerance)
                    with np.errstate(all="ignore"):
                        old = old_interval(*args)
                        new = stripes.interval_evidence(*args)
                    if not exact_arrays(old, new):
                        print("C2 MISMATCH", name, centres_name, interval, tolerance,
                              [(a.shape, a.dtype, b.shape, b.dtype) for a, b in zip(old, new, strict=True)])
                        return
    print("C2 24 interval comparisons: shapes, dtype and bytes identical")

    observations = assignment.prepare_observations(segment_sets["mixed"], size)
    unusual_samples = observations.samples.copy()
    unusual_samples[0, 0] = [np.nan, np.inf]
    unusual_samples[1, 1, 0] = -0.0
    unusual_directions = observations.directions.copy()
    unusual_directions[0] = np.nan
    unusual = replace(observations, samples=unusual_samples, directions=unusual_directions)
    centres = paint_geometry.CENTRE_SEGMENTS_M
    endpoint = centres[0]
    samples, _ = detector.project(h[None], endpoint[0] + np.linspace(0, 1, 7)[:, None]
                                  * (endpoint[1] - endpoint[0]))
    args = (h, np.linalg.inv(h), 0, samples[0], unusual, size, centres, 3.5)
    with np.errstate(all="ignore"):
        old = old_interval(*args)
        new = stripes.interval_evidence(*args)
    print("C2 NaN/inf and negative-zero input bits", exact_arrays(old, new))

    original = stripes.interval_evidence
    try:
        stripes.interval_evidence = old_interval
        baseline = stripes.measure(h, observations, size, centres=centres, boundary_tolerance_px=3.5)
    finally:
        stripes.interval_evidence = original
    patched = stripes.measure(h, observations, size, centres=centres, boundary_tolerance_px=3.5)
    print("C2 measure bytes", exact_arrays(baseline.forward, patched.forward)
          and exact_arrays((baseline.reverse,), (patched.reverse,))
          and exact_arrays(baseline.resolvable, patched.resolvable)
          and exact_arrays((baseline.visible,), (patched.visible,)))

    map_segments = np.array([[1., 1., 29., 2.], [5., 1., 5., 20.]])
    maps_a = detector._distance_maps(detector._wide_line_families(map_segments), (32, 24))
    maps_b = detector._distance_maps(detector._wide_line_families(map_segments), (32, 24))
    print("C4 repeated map bytes", exact_arrays((maps_a,), (maps_b,)))


if __name__ == "__main__":
    main()
