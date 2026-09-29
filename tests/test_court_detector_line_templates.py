"""Line-template scoring: one NumPy/CuPy array definition, CPU by default, CUDA only on request."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from court_detector import (
    detect,
    directions,
    geometry,
    line_templates,
    run_video,
    template_arrays,
)
from court_detector.inputs import (
    ViewInputs,
    same_frame_provenance,
)
from scratch.court_det_fix.court_detector import run_views

SIZE = (320, 240)
NATIVE_SIZE = (640, 480)
# Singles sidelines crossed with the doubles long service lines.
INNER_LINES_M = np.array([
    [geometry.X_COORDS[1], geometry.Y_COORDS[1]],
    [geometry.X_COORDS[-2], geometry.Y_COORDS[1]],
    [geometry.X_COORDS[-2], geometry.Y_COORDS[-2]],
    [geometry.X_COORDS[1], geometry.Y_COORDS[-2]],
])


def court_homography() -> np.ndarray:
    """Court metres to working pixels for a pinhole camera 6 m behind and above a baseline.

    Its focal length is one of the camera check's trial values, so the camera error is zero.
    """
    width, height = SIZE
    focal = template_arrays.camera_focals(width)[60]
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    rotation, _ = cv2.Rodrigues(np.array([np.deg2rad(-65.0), 0.0, 0.0]))
    centre = np.array([3.05, -6.0, 6.0])
    # The sign puts the court in front of the camera; the pixels are the same either way.
    return -intrinsics @ np.column_stack([rotation[:, 0], rotation[:, 1], -rotation @ centre])


def court_segments(homography: np.ndarray) -> np.ndarray:
    """The 12 painted pieces as (x1, y1, x2, y2) working-pixel fragments."""
    return geometry.project(homography[None], geometry.SEGMENTS_M)[0].reshape(12, 4)


def cuda_available() -> bool:
    try:
        template_arrays.array_module("cuda")
    except Exception:  # noqa: BLE001 - any CuPy or driver failure means there is no GPU to test
        return False
    return True


def test_shared_projection_and_clipping_match_geometry_bit_for_bit() -> None:
    rng = np.random.default_rng(20260928)
    homographies = (court_homography() * (1 + rng.normal(scale=0.2, size=(64, 3, 3)))).astype(np.float32)
    view = template_arrays.place_view(np, np.zeros(SIZE[::-1], dtype=np.float32), SIZE, geometry)
    for points_m, homogeneous in ((geometry.CORNER_COURT_M, view.corner_points),
                                  (geometry.SEGMENTS_M, view.segment_points)):
        for actual, expected in zip(template_arrays.project(np, homographies, homogeneous),
                                    geometry.project(homographies, points_m), strict=True):
            assert actual.dtype == np.float32
            np.testing.assert_array_equal(actual, expected)

    endpoints = geometry.project(homographies, geometry.SEGMENTS_M)[0].reshape(-1, 12, 2, 2)
    # Vertical, horizontal, zero-length, off-image, too-short and part-visible lines reach every clipping
    # branch. One line moves 1e-9 px across, below the stationary threshold.
    edge_cases = np.array([
        [[0, 10], [0, 200]], [[10, 239], [300, 239]], [[50, 50], [50, 50]], [[-40, -5], [-10, -30]],
        [[100, 100], [105, 104]], [[-50, 120], [400, 60]], [[330, 10], [330, 200]], [[160, -20], [160, 260]],
        [[319, 0], [0, 239]], [[0, 20], [1e-9, 200]], [[5, 5], [6, 250]], [[-1, -1], [321, 241]],
    ], dtype=np.float32)
    endpoints = np.concatenate((endpoints, edge_cases[None]))
    shared = template_arrays.visible_samples(np, endpoints, view)
    original = geometry._visible_samples(endpoints, SIZE, template_arrays.SAMPLES_PER_LINE)
    assert shared[0].dtype == np.float32
    for actual, expected in zip(shared, original, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_real_camera_view_scores_full_support_by_direction() -> None:
    homography = court_homography()
    scored = homography[None].astype(np.float32)
    union_map = geometry.distance_map(court_segments(homography), SIZE)
    corners, means, valid, visibility = line_templates.geometry_and_support(scored, union_map, SIZE, geometry)
    assert valid.tolist() == [True]
    # float32 keeps the corners within a thousandth of a pixel.
    np.testing.assert_allclose(corners[0], geometry.project(homography[None], geometry.CORNER_COURT_M)[0][0],
                               atol=1e-3)
    np.testing.assert_array_equal(means, np.array([[1.0, 1.0]], dtype=np.float32))
    np.testing.assert_array_equal(visibility, np.array([[6, 6]], dtype=np.int16))
    # Zero up to float32 rounding, far inside the frontier recheck margin.
    assert line_templates.vector_camera_errors(scored, SIZE)[0] < 1e-5
    with pytest.raises(TypeError, match="float32"):
        line_templates.vector_camera_errors(homography[None], SIZE)

    # The first column holds the lengthwise pieces, so drawing only cross-court lines empties it.
    cross_court_map = geometry.distance_map(court_segments(homography)[6:], SIZE)
    _, means, _, _ = line_templates.geometry_and_support(scored, cross_court_map, SIZE, geometry)
    assert means[0, 0] < 0.5
    assert means[0, 1] == 1.0

    # A mirror image reverses the corner order, so no hypothesis survives; results stay typed.
    mirror = np.array([[-1.0, 0.0, SIZE[0] - 1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    corners, means, valid, visibility = line_templates.geometry_and_support(
        (mirror @ homography)[None].astype(np.float32), union_map, SIZE, geometry,
    )
    assert valid.tolist() == [False]
    assert (corners.shape, corners.dtype) == ((0, 4, 2), np.float32)
    assert (means.shape, means.dtype) == ((0, 2), np.float32)
    assert (visibility.shape, visibility.dtype) == ((0, 2), np.int16)


def test_frontier_recheck_replaces_only_errors_near_the_camera_limit() -> None:
    calls = []

    def net_segments(native_corners: np.ndarray, native_size: tuple[int, int]) -> tuple[None, float]:
        calls.append((native_corners, native_size))
        return None, 0.101

    vector_errors = np.array([0.05, 0.0995, 0.1008, 0.2], dtype=np.float32)
    corners = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    errors, count, largest_change = line_templates.recheck_camera_frontier(
        vector_errors, corners, np.array([2.0, 2.0]), NATIVE_SIZE, SimpleNamespace(net_segments=net_segments),
    )
    # float64, so the rechecked scalar values meet the camera limit unrounded.
    assert errors.dtype == np.float64
    np.testing.assert_array_equal(errors, [np.float32(0.05), 0.101, 0.101, np.float32(0.2)])
    assert count == 2
    assert largest_change == pytest.approx(0.0015)
    np.testing.assert_array_equal(vector_errors, np.array([0.05, 0.0995, 0.1008, 0.2], dtype=np.float32))
    np.testing.assert_array_equal(calls[0][0], (corners[1] * 2).astype(np.float32))


def rectangle_scene(monkeypatch: pytest.MonkeyPatch, positions: dict[int, np.ndarray]):
    """A view of the synthetic court whose coverage order lists the given rectangle quads.

    :param positions: rectangle ID to its (4, 2) working-pixel quad, in coverage order.
    """
    segments = court_segments(court_homography())
    ids = list(positions)
    monkeypatch.setattr(directions, "estimate", lambda *args: (np.empty((0, 3)), {}))
    monkeypatch.setattr(directions, "select", lambda *args: (None, {"selected_pair_product_ids": ids,
                                                                    "union_pair_product_ids": ids}))
    # The population holds one rectangle outside the coverage order.
    population_ids = [*ids, 999]
    quads = np.asarray([*positions.values(), [[0, 0], [50, 0], [50, 50], [0, 50]]], dtype=float)
    monkeypatch.setattr(directions, "rectangle_population", lambda *args: (quads, None, population_ids))
    context = SimpleNamespace(case_id="synthetic", segments=segments, size=SIZE, native_size=NATIVE_SIZE,
                              families=(segments[:6], segments[6:]), source={})
    runtime = {"zone": SimpleNamespace(net_segments=lambda *args: (None, 0.5)),
               "gate_evidence": lambda *args: {"checked": True}}
    return context, runtime


def court_rectangles() -> dict[int, np.ndarray]:
    """The true court's inner rectangle eighth in coverage order, behind a tiny one and six others.

    The CPU scores six rectangles per batch, so the true one lands in the second batch.
    """
    inner_quad = geometry.project(court_homography()[None], INNER_LINES_M)[0][0]
    others = {90 + position: np.array([[left, top], [left + 60, top + 5], [left + 55, top + 70], [left - 5, top + 60]],
                                      dtype=float)
              for position, (left, top) in enumerate([(20, 20), (200, 30), (40, 150), (230, 160), (120, 60), (90, 120)])}
    return {80: np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=float), **others, 5: inner_quad}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not cuda_available(), reason="needs CuPy and a CUDA GPU"))])
def test_generation_finds_the_court_across_batches(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    if device == "cpu":
        # CPU generation must not import CuPy.
        monkeypatch.setitem(sys.modules, "cupy", None)
    positions = court_rectangles()
    context, runtime = rectangle_scene(monkeypatch, positions)
    generated = line_templates.generate(context, runtime, geometry, device=device)

    # The template whose unit square spans the inner lines rebuilds the whole court from that rectangle.
    unit_corners = geometry.TEMPLATE_TRANSFORMS @ template_arrays.homogeneous(INNER_LINES_M).T
    [expected_template] = np.flatnonzero(np.all(np.isclose(unit_corners[:, :2], geometry.UNIT_CORNERS.T), axis=(1, 2)))
    best = generated.entries[0]
    assert best["proposal_id"] == f"rectangle_5:template_{expected_template}"
    assert best["rectangle_order"] == 7
    assert best["line_template"]["direction_means"] == [1.0, 1.0]
    assert (best["line_template"]["visible_lengthwise_pieces"], best["line_template"]["visible_cross_court_pieces"]) \
        == (6, 6)
    assert best["gates"] == {"checked": True}
    true_corners = geometry.project(court_homography()[None], geometry.CORNER_COURT_M)[0][0] * 2
    np.testing.assert_allclose(best["corners_px"], true_corners, atol=1e-3)
    order_by_id = {rectangle_id: order for order, rectangle_id in enumerate(positions)}
    assert all(entry["rectangle_order"] == order_by_id[entry["rectangle_id"]] for entry in generated.entries)
    assert generated.metadata["ordering"]["area_retained_rectangle_count"] == 7
    assert generated.metadata["generation"]["rectangle_template_hypotheses"] == 7 * line_templates.TEMPLATE_COUNT


def test_batch_size_leaves_generation_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU and GPU runs use different batch sizes, so comparing them relies on this."""
    context, runtime = rectangle_scene(monkeypatch, court_rectangles())
    outputs = []
    for batch_rectangles in (line_templates.BATCH_RECTANGLES["cpu"], 1, line_templates.BATCH_RECTANGLES["cuda"]):
        monkeypatch.setitem(line_templates.BATCH_RECTANGLES, "cpu", batch_rectangles)
        generated = line_templates.generate(context, runtime, geometry)
        generated.metadata["generation"].pop("elapsed_seconds")
        outputs.append(generated)
    assert all(generated == outputs[0] for generated in outputs[1:])


@pytest.mark.skipif(not cuda_available(), reason="needs CuPy and a CUDA GPU")
def test_cuda_generation_keeps_the_cpu_court(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU rounding can move a rare line sample, so later proposals may reorder; the court and geometry may not."""
    context, runtime = rectangle_scene(monkeypatch, court_rectangles())
    cpu = line_templates.generate(context, runtime, geometry, device="cpu")
    cuda = line_templates.generate(context, runtime, geometry, device="cuda")
    assert cuda.entries[0]["proposal_id"] == cpu.entries[0]["proposal_id"]
    assert cuda.entries[0]["line_template"]["direction_means"] == [1.0, 1.0]
    for count in ("valid_geometry_hypotheses", "camera_eligible_hypotheses"):
        assert cuda.metadata["generation"][count] == cpu.metadata["generation"][count]
    cuda_corners = {entry["proposal_id"]: entry["corners_px"] for entry in cuda.entries}
    # float32 rounding alone moves these corners by up to about 1e-3 native px on the CPU.
    for entry in cpu.entries:
        if entry["proposal_id"] in cuda_corners:
            np.testing.assert_allclose(cuda_corners[entry["proposal_id"]], entry["corners_px"], atol=1e-2)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not cuda_available(), reason="needs CuPy and a CUDA GPU",
))])
@pytest.mark.parametrize(("positions", "reason"), [
    ({}, "no_coverage_rectangles"),
    ({80: np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=float)}, "no_area_valid_rectangles"),
    # Listing the corners anticlockwise mirrors every template's court.
    ({7: np.array([[40, 40], [40, 200], [280, 200], [280, 40]], dtype=float)}, "no_geometry_valid_hypotheses"),
])
def test_generation_without_candidates_returns_an_empty_source(monkeypatch: pytest.MonkeyPatch,
                                                                positions: dict, reason: str, device: str) -> None:
    context, runtime = rectangle_scene(monkeypatch, positions)
    generated = line_templates.generate(context, runtime, geometry, device=device)
    assert generated.entries == ()
    assert (generated.metadata["status"], generated.metadata["reason"]) == ("empty", reason)
    assert generated.metadata["proposal_count"] == 0


def test_template_device_is_explicit_and_cuda_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    assert template_arrays.array_module("cpu") is np
    with pytest.raises(ValueError, match="template device"):
        template_arrays.array_module("gpu")
    with pytest.raises(ValueError, match="template_device"):
        detect.Switches(template_device="gpu")
    assert detect.Switches().template_device == "cpu"

    no_devices = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0)))
    monkeypatch.setitem(sys.modules, "cupy", no_devices)
    with pytest.raises(RuntimeError, match="found none"):
        template_arrays.array_module("cuda")
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(RuntimeError, match="needs CuPy"):
        template_arrays.array_module("cuda")
    # The detector checks at construction, before any view reaches the line templates.
    with pytest.raises(RuntimeError, match="needs CuPy"):
        detect.CourtDetector(detect.Switches(template_device="cuda"))


def test_detector_passes_its_template_device_to_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    def generate(*args, **kwargs):
        requests.append(kwargs)
        return SimpleNamespace(entries=[], metadata={})

    monkeypatch.setattr(detect.feet, "window_feet", lambda *args: SimpleNamespace(all_feet_px=[], _asdict=dict))
    monkeypatch.setattr(detect.search, "seed_points", lambda family: None)
    detector = object.__new__(detect.CourtDetector)
    detector.switches = detect.Switches(require_people=False, template_device="cuda")
    detector.live = SimpleNamespace(
        verifier=SimpleNamespace(view_context=lambda *args: SimpleNamespace(families=[None])), runtime={},
        court_model=None, line_template_source=SimpleNamespace(generate=generate),
        prepared_measurements=lambda verifier: nullcontext(),
    )
    detector.search = lambda *args, **kwargs: {}
    detector.score_and_choose = lambda view, *args: detect.CourtResult(view.view_id, None, "no_gated_court", None,
                                                                         None)
    view = ViewInputs("view", np.zeros((10, 20, 3), dtype=np.uint8), 0, (0, 10), np.empty((0, 4)),
                      np.empty((0, 4)), same_frame_provenance("view", 0))
    detector.detect(view, None, None)
    assert requests[0]["device"] == "cuda"


class SwitchesSeen(Exception):
    """Stops a runner once it has built the detector's switches."""


def stop_at_detector(switches: detect.Switches) -> None:
    raise SwitchesSeen(switches)


@pytest.mark.parametrize(("flag", "device"), [([], "cpu"), (["--template-device", "cuda"], "cuda")])
def test_runners_pass_the_template_device(monkeypatch: pytest.MonkeyPatch, tmp_path, flag: list[str],
                                          device: str) -> None:
    monkeypatch.setattr(run_views, "CourtDetector", stop_at_detector)
    monkeypatch.setattr("sys.argv", ["run_views", "--people", str(tmp_path), "--output", str(tmp_path), "view",
                                     *flag])
    with pytest.raises(SwitchesSeen) as views_stop:
        run_views.main()
    assert views_stop.value.args[0].template_device == device

    monkeypatch.setattr(run_video, "CourtDetector", stop_at_detector)
    monkeypatch.setattr(run_video, "VideoFrames", lambda path: nullcontext(SimpleNamespace(frame_count=1)))
    monkeypatch.setattr(run_video, "read_json", lambda path: {})
    monkeypatch.setattr(run_video.os, "sched_setaffinity", lambda *_: None)
    monkeypatch.setattr("sys.argv", ["run_video", "--video", "input.mp4", "--output", str(tmp_path / "out.json.gz"),
                                     "--saved-lines", "lines.json.gz", "--no-require-people", *flag])
    with pytest.raises(SwitchesSeen) as video_stop:
        run_video.main()
    assert video_stop.value.args[0].template_device == device


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not cuda_available(), reason="needs CuPy and a CUDA GPU"))])
def test_equal_sample_counts_keep_exact_template_ties(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    array_module = template_arrays.array_module(device)
    counts = np.asarray([[16, 18, 18, 20, 9, 20], [18, 20, 20, 18, 16, 9]])
    counts = np.tile(counts, (1, 2))
    distance_map = np.full(SIZE[::-1], 10, dtype=np.float32)
    samples = np.zeros((2, 12, template_arrays.SAMPLES_PER_LINE, 2), dtype=np.float32)
    for court in range(2):
        for interval in range(12):
            row = court * 12 + interval
            distance_map[row, :counts[court, interval]] = 0
            samples[court, interval, :, 0] = np.arange(template_arrays.SAMPLES_PER_LINE)
            samples[court, interval, :, 1] = row
    monkeypatch.setattr(template_arrays, "visible_samples", lambda *args: (
        array_module.asarray(samples), array_module.ones((2, 12), dtype=bool)))
    view = template_arrays.place_view(array_module, distance_map, SIZE, geometry)
    homographies = array_module.asarray(np.repeat(court_homography()[None], 2, axis=0), dtype=array_module.float32)

    _, _, _, means, _ = template_arrays.geometry_and_support(array_module, homographies, view)

    if device == "cuda":
        means = array_module.asnumpy(means)
    expected = np.full((2, 2), np.float32(101 / 144), dtype=np.float32)
    np.testing.assert_array_equal(means, expected)
