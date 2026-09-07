"""Candidate fitting must retain a usable court until final scene acceptance."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from annotator import court_evidence as evidence
from annotator.court_views import describe_court_view, matching_view_groups
from courtkeynet import court_corners as fallback
from courtkeynet.wrapper import CornerDetection

GOOD_CORNERS = np.array([[300, 150], [980, 150], [1050, 650], [230, 650]], dtype=np.float32)
DISTORTED_CORNERS = np.array(
    [[156.09236, 110.281006], [785.24194, 387.77365], [1050, 650], [230, 650]], dtype=np.float32,
)


def _candidate_scene(monkeypatch: pytest.MonkeyPatch) -> fallback.CourtQuad:
    good_homography = cv2.getPerspectiveTransform(fallback.CORNER_COURT_M, GOOD_CORNERS)
    distorted_homography = cv2.getPerspectiveTransform(fallback.CORNER_COURT_M, DISTORTED_CORNERS)
    segments = np.array([
        fallback._project(good_homography, np.stack(pair)).reshape(4)
        for pair in fallback.PAINTED_SEGMENTS_M
    ]) + np.array([3, 2, 3, 2])
    lines = [fallback._Line(*fallback._tls_line(segment.reshape(2, 2)),
                           float(np.linalg.norm(segment[2:] - segment[:2]))) for segment in segments]
    anchors = {fallback.BR: GOOD_CORNERS[fallback.BR], fallback.BL: GOOD_CORNERS[fallback.BL]}
    good = fallback._court_fit(good_homography, lines, anchors, (1280, 720))
    distorted = fallback._court_fit(distorted_homography, lines, anchors, (1280, 720))
    assert good is not None and distorted is not None
    assert distorted.line_error < good.line_error
    assert (distorted.n_lines, good.n_lines) == (2, 12)
    assert min(fallback.painted_line_support(good.corners, (segments,), (1280, 720))) > 0.98
    assert min(fallback.painted_line_support(distorted.corners, (segments,), (1280, 720))) == 0.0

    # Supply candidate hypotheses; exercise the real fit ranking and adapter gates.
    monkeypatch.setattr(fallback, '_frame_segments', lambda *_args: segments)
    monkeypatch.setattr(fallback, '_cluster_segments', lambda *_args: (lines, 0.0))
    monkeypatch.setattr(fallback, '_outer_lines', lambda *_args: {})
    monkeypatch.setattr(fallback, '_bootstrap_corners', lambda *_args: GOOD_CORNERS)
    monkeypatch.setattr(fallback, '_ransac_homography', lambda *_args: distorted_homography)
    peak = np.array([0.005, 0.005, 0.5, 0.5], dtype=np.float32)
    detection = CornerDetection(GOOD_CORNERS, peak, np.full(4, 0.3, dtype=np.float32), ())
    result = fallback.pick_scene_corners([np.zeros((720, 1280, 3), dtype=np.uint8)], [detection])
    assert result is not None
    return result


def _accept(quad: fallback.CourtQuad, player_count: int = 2) -> evidence.CourtEvidenceResult:
    bboxes = np.tile(np.array([[[600, 220, 650, 300], [600, 500, 650, 580]]], dtype=float), (10, 1, 1))
    scores = np.full((10, 2), 0.9)
    return evidence.build_detected_court_evidence(
        'candidate-test', 'detected', '7', (1280.0, 720.0), [(0, 10)],
        [evidence.SceneEvidence(0, 10, (0,), quad)], bboxes, scores, np.full(10, player_count),
        detector_resolution=(1280.0, 720.0),
    )


def test_lower_residual_candidate_retains_supported_alternative(monkeypatch: pytest.MonkeyPatch) -> None:
    quad = _candidate_scene(monkeypatch)
    np.testing.assert_allclose(quad.corners_px, DISTORTED_CORNERS)
    np.testing.assert_allclose(quad.alternative_corners_px, [GOOD_CORNERS])
    with pytest.raises(evidence.CourtConsensusError):
        _accept(replace(quad, alternative_corners_px=()))
    result = _accept(quad)
    record = result.scene_records[0]
    assert record.scene_valid
    assert result.keep_vote.all() and result.court_present.all()
    np.testing.assert_allclose(record.raw_corners_px, DISTORTED_CORNERS)
    np.testing.assert_allclose(record.active_corners_native_px, GOOD_CORNERS)
    assert record.fallback_diagnostics == quad.diagnostics
    assert min(record.painted_line_support) > 0.98
    assert result.inputs.homography_rows['upleft_x'].tolist() == [300.0]


def test_passing_preferred_court_keeps_its_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    quad = _candidate_scene(monkeypatch)
    preferred = replace(quad, corners_px=GOOD_CORNERS, alternative_corners_px=(DISTORTED_CORNERS,))
    result = _accept(preferred)
    np.testing.assert_allclose(result.scene_records[0].active_corners_native_px, GOOD_CORNERS)


def test_supported_alternative_still_requires_person_votes(monkeypatch: pytest.MonkeyPatch) -> None:
    quad = _candidate_scene(monkeypatch)
    with pytest.raises(evidence.CourtConsensusError):
        _accept(quad, player_count=1)


@pytest.mark.parametrize(('moved_corner', 'shift', 'recover_alternative'), [(2, 75, True), (0, 50, False)])
@pytest.mark.parametrize('native_scale', [0.4, 1.0, 1.5])
def test_sharing_preserves_target_anchors_and_stronger_line_support(
    monkeypatch: pytest.MonkeyPatch, moved_corner: int, shift: float, recover_alternative: bool, native_scale: float,
) -> None:
    target = _candidate_scene(monkeypatch)
    if not recover_alternative:
        target = replace(target, corners_px=GOOD_CORNERS, alternative_corners_px=())
    shared_proposal = GOOD_CORNERS.copy()
    shared_proposal[moved_corner, 0] -= shift
    local_support = fallback.painted_line_support(GOOD_CORNERS, target.line_segments_px, (1280, 720))
    shared_support = fallback.painted_line_support(shared_proposal, target.line_segments_px, (1280, 720))
    assert min(shared_support) >= evidence.MIN_PAINTED_LINE_SUPPORT
    assert sum(shared_support) < sum(local_support)

    frame = np.full((720, 1280, 3), 80, dtype=np.uint8)
    for first, last in target.line_segments_px[0].reshape(-1, 2, 2):
        cv2.line(frame, tuple(np.rint(first).astype(int)), tuple(np.rint(last).astype(int)), (240, 240, 240), 2)
    view = describe_court_view([frame] * 3)
    assert matching_view_groups([view] * 3, [GOOD_CORNERS, shared_proposal, shared_proposal], [True] * 3) == [[0, 1, 2]]

    target = replace(
        target, corners_px=target.corners_px * native_scale,
        line_segments_px=tuple(segments * native_scale for segments in target.line_segments_px),
        alternative_corners_px=tuple(corners * native_scale for corners in target.alternative_corners_px),
    )
    model = fallback.CourtQuad(
        shared_proposal * native_scale, np.full(4, 0.5, dtype=np.float32), 'model', ('model',) * 4, None,
    )
    intervals = [(0, 10), (10, 20), (20, 30)]
    scenes = [evidence.SceneEvidence(start, end, (start + 1, start + 5, start + 8), quad, view)
              for (start, end), quad in zip(intervals, [target, model, model])]
    bboxes = np.tile(np.array([[[600, 220, 650, 300], [600, 500, 650, 580]]], dtype=float), (30, 1, 1))
    result = evidence.build_detected_court_evidence(
        'sharing-test', 'detected', '7', (1280., 720.), intervals, scenes,
        bboxes, np.full((30, 2), 0.9), np.full(30, 2),
        detector_resolution=(1280 * native_scale, 720 * native_scale),
    )
    assert result.keep_vote.all() and result.court_present.all()
    assert all(record.scene_valid for record in result.scene_records)
    assert all(record.view_group_index is None for record in result.scene_records)
    np.testing.assert_allclose(result.scene_records[0].active_corners_native_px, GOOD_CORNERS * native_scale)
    np.testing.assert_allclose(result.scene_records[0].raw_corners_px, target.corners_px)
    assert min(result.scene_records[0].painted_line_support) > 0.98
