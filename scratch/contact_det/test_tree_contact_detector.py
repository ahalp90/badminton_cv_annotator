"""Focused tests for the isolated tree contact detector trial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import freeze_tree_contact_features as freezer
import numpy as np
import pytest
import score_tree_contact_detector as scorer


def test_impulses_keep_the_production_junction_alignment() -> None:
    from annotator.config import RallySegmentationThresholds
    from annotator.fps_constants import scale_for_fps
    from annotator.rally.contacts import span_impulses

    track = np.zeros((18, 3), dtype=np.float64)
    track[:, 2] = 1
    track[:, 0] = np.concatenate([np.linspace(0.1, 0.8, 9), np.linspace(0.8, 0.2, 9)])
    spans = ((2, 16),)
    signals = freezer._shuttle_signals(track, spans, 30.0)
    values = scale_for_fps(30.0)
    thresholds = RallySegmentationThresholds(
        values.rest_speed,
        values.rest_window,
        values.end_rest_frames,
        values.start_speed,
        values.start_min_frames,
        values.smooth_window,
        values.impulse_floor_half_window_frames,
        values.contact_dedup_radius_frames,
        values.contact_suppression_radius_frames,
        freezer.RELAXED_IMPULSE_MULTIPLE,
    )
    expected = span_impulses(track, 2, 16, thresholds)
    assert expected is not None
    np.testing.assert_allclose(signals["shuttle_impulse"][3:15], expected)
    assert np.isnan(signals["shuttle_impulse"][[2, 15]]).all()


def test_broad_regions_stay_inside_each_span_even_when_the_kernel_is_longer() -> None:
    n_frames = 30
    signals = {
        "shuttle_visible": np.ones(n_frames, dtype=np.float32),
        "shuttle_impulse_ratio": np.full(n_frames, np.nan, dtype=np.float32),
        "wrist_gap_min": np.full(n_frames, np.nan, dtype=np.float32),
    }
    signals["shuttle_impulse_ratio"][12] = 2.0
    signals["wrist_gap_min"][18] = 1.0
    regions = freezer.build_region_masks(
        signals,
        spans=((10, 14), (17, 21)),
        raw_contacts=({"contact_frame": 11},),
        scene_spans=((0, 17), (17, 30)),
        fps=30.0,
    )
    assert all(region.dtype == bool and region.shape == (n_frames,) for region in regions.values())
    assert regions["region_current_raw"][10:14].all()
    assert not regions["region_current_raw"][:10].any()
    assert not regions["region_current_raw"][14:].any()
    assert regions["region_wrist"][17:21].all()
    assert not regions["region_wrist"][:17].any()


def test_player_geometry_follows_current_frame_sticky_picks() -> None:
    track = np.asarray([[0.50, 0.50, 1], [0.55, 0.50, 1]], dtype=np.float64)
    pose_kps = np.zeros((2, 2, 17, 2), dtype=np.float64)
    pose_kps[0, 0, (9, 10), :] = (50.0, 50.0)
    pose_kps[0, 1, (9, 10), :] = (10.0, 10.0)
    pose_kps[1, 0, (9, 10), :] = (5.0, 5.0)
    pose_kps[1, 1, (9, 10), :] = (55.0, 50.0)
    sticky = SimpleNamespace(
        picks=np.asarray([[0, -1], [1, -1]]),
        distances_per_slot=np.asarray([[0.2, np.nan], [0.2, np.nan]]),
        ankle_pos=np.asarray([[[0.4, 0.6], [np.nan, np.nan]], [[0.5, 0.6], [np.nan, np.nan]]]),
        bbox_height=np.asarray([[50.0, np.nan], [50.0, np.nan]]),
    )
    signals = freezer._player_signals(track, pose_kps, sticky, (100.0, 100.0))
    np.testing.assert_allclose(signals["nearest_wrist_dx"], [0.0, 0.0])
    np.testing.assert_allclose(signals["nearest_wrist_dy"], [0.0, 0.0])


def _write_verified_fixture(tmp_path: Path) -> Path:
    families = freezer._feature_family_names()
    dtype = freezer._record_dtype(families)
    rows = np.zeros(3, dtype=dtype)
    rows["fixture"] = [b"sset_01", b"sset_15", b"sset_21"]
    rows["span_id"] = 0
    rows["frame"] = [10, 20, 30]
    rows["fps"] = [25.0, 25.0, 30.0]
    feature_path = tmp_path / freezer.FEATURE_FILENAME
    freezer._write_npy_xz(feature_path, rows)
    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    manifest = {
        "schema": freezer.MANIFEST_SCHEMA,
        "feature_schema": freezer.FEATURE_SCHEMA,
        "labels_read": False,
        "source_commit": "ad8da4f",
        "fixture_set": list(freezer.FIXTURE_SPECS),
        "feature_file": feature_path.name,
        "feature_sha256": digest,
        "row_count": len(rows),
        "feature_families": families,
        "identity_fields": list(freezer.IDENTITY_FIELDS),
        "region_fields": list(freezer.REGION_FIELDS),
    }
    manifest_path = tmp_path / freezer.MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_freeze_verification_rejects_a_labelled_or_changed_table(tmp_path: Path) -> None:
    manifest_path = _write_verified_fixture(tmp_path)
    verified = scorer.verify_freeze(manifest_path)
    assert len(verified.rows) == 3

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels_read"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="label-blind"):
        scorer.verify_freeze(manifest_path)

    manifest["labels_read"] = False
    manifest["feature_families"]["physics"].append("frame")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="producer contract"):
        scorer.verify_freeze(manifest_path)


def test_training_selection_keeps_positives_ignores_adjacent_frames() -> None:
    dtype = np.dtype([("fixture", "S7"), ("span_id", "i2"), ("frame", "i4"), ("fps", "f4")])
    rows = np.zeros(31, dtype=dtype)
    rows["fixture"] = b"sset_21"
    rows["frame"] = np.arange(85, 116)
    rows["fps"] = 30.0
    ground_truth = scorer.GroundTruth(
        frames={"sset_21": np.asarray([100], dtype=np.int32)},
        serves={"sset_21": {100}},
        rally_count=1,
    )
    selected, labels = scorer.build_training_mask(rows, ["sset_21"], ground_truth, seed=1)
    assert labels[rows["frame"] == 100].item() == 1
    assert labels[rows["frame"] == 99].item() == 1
    assert labels[rows["frame"] == 101].item() == 1
    assert not selected[(rows["frame"] == 102) | (rows["frame"] == 104)].any()
    assert selected[rows["frame"] == 105].item()


def test_temporal_nms_is_per_span_and_keeps_the_strongest_frame() -> None:
    frames = np.asarray([10, 12, 14, 12], dtype=np.int32)
    spans = np.asarray([0, 0, 0, 1], dtype=np.int16)
    probabilities = np.asarray([0.7, 0.9, 0.8, 0.6])
    kept = scorer.temporal_nms(frames, spans, probabilities, threshold=0.5, radius=3)
    assert set(kept.tolist()) == {1, 3}


def test_event_matching_is_one_to_one_and_reports_serve_split() -> None:
    ground_truth = scorer.GroundTruth(
        frames={"sset_21": np.asarray([100, 108], dtype=np.int32)},
        serves={"sset_21": {100}},
        rally_count=1,
    )
    metrics = scorer._event_counts(
        ground_truth,
        {"sset_21": np.asarray([104], dtype=np.int32)},
        tolerance_base30=5,
        fixtures=["sset_21"],
    )
    assert metrics["matched"] == 1
    assert metrics["predictions"] == 1
    assert metrics["serve_matched"] == 1
    assert metrics["nonserve_matched"] == 0


def test_region_coverage_distinguishes_strict_and_operational_centres() -> None:
    ground_truth = np.asarray([100, 120], dtype=np.int32)
    region_frames = np.asarray([100, 115], dtype=np.int32)
    strict = scorer._region_coverage(ground_truth, {100}, region_frames, tolerance=0)
    operational = scorer._region_coverage(ground_truth, {100}, region_frames, tolerance=5)
    assert strict["covered"] == 1
    assert operational["covered"] == 2
    assert strict["serve_covered"] == operational["serve_covered"] == 1
