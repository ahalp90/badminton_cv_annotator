"""Focused tests for the isolated tree contact detector trial."""

from __future__ import annotations

import hashlib
import json
import lzma
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


def test_broad_regions_stay_inside_each_eligible_interval() -> None:
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
        eligible_intervals=((10, 14), (17, 21)),
        rally_spans=((10, 14),),
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
    assert regions["region_serve_lookback"][:10].all()
    assert regions["region_serve_lookback"][14:17].all()
    assert not regions["region_serve_lookback"][17:21].any()


def test_eligible_intervals_split_around_excluded_frames() -> None:
    exclusion = np.asarray([True, False, False, True, False, False, False, True])
    intervals = freezer.build_eligible_intervals(((0, 4), (4, 8)), exclusion)
    assert intervals == [(1, 3), (4, 7)]


def test_serve_lookback_is_backward_only_and_clipped() -> None:
    n_frames = 120
    signals = {
        "shuttle_visible": np.ones(n_frames, dtype=np.float32),
        "shuttle_impulse_ratio": np.full(n_frames, np.nan, dtype=np.float32),
        "wrist_gap_min": np.full(n_frames, np.nan, dtype=np.float32),
    }
    regions = freezer.build_region_masks(
        signals,
        eligible_intervals=((10, 80), (90, 110)),
        rally_spans=(),
        raw_contacts=(),
        scene_spans=((0, 90), (90, 120)),
        fps=30.0,
    )
    lookback = regions["region_serve_lookback"]
    assert lookback[:10].all()
    assert not lookback[10:45].any()
    assert lookback[45:90].all()
    assert not lookback[90:].any()


def test_search_intervals_merge_overlapping_serve_lookbacks() -> None:
    intervals = freezer.extend_intervals_with_lookback(((50, 60), (80, 90)), 100, 30.0)
    assert intervals == [(5, 90)]


def test_feature_offsets_do_not_cross_an_eligible_interval_boundary() -> None:
    values = np.arange(12, dtype=np.float32)
    frames = np.asarray([4, 5, 6], dtype=np.int32)
    shifted = freezer._shift_inside_interval(values, frames, offset=-1, start=5, end=8)
    assert np.isnan(shifted[0])
    assert np.isnan(shifted[1])
    assert shifted[2] == 5


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


def _write_verified_fixture(tmp_path: Path, motion_mode: str | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    families = freezer._feature_family_names()
    dtype = freezer._record_dtype(families)
    rows = np.zeros(3, dtype=dtype)
    rows["fixture"] = [b"sset_01", b"sset_15", b"sset_21"]
    rows["interval_id"] = 0
    rows["frame"] = [10, 20, 30]
    rows["fps"] = [25.0, 25.0, 30.0]
    feature_path = tmp_path / freezer.FEATURE_FILENAME
    freezer._write_npy_xz(feature_path, rows)
    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    manifest = {
        "schema": freezer.MANIFEST_SCHEMA,
        "feature_schema": freezer.FEATURE_SCHEMA,
        "labels_read": False,
        "row_domain": "eligible tracker intervals plus 45-base-30 serve pre-roll",
        "model_search_surface": "seeded region union",
        "source_commit": "ad8da4f",
        "fixture_set": list(freezer.FIXTURE_SPECS),
        "feature_file": feature_path.name,
        "feature_sha256": digest,
        "row_count": len(rows),
        "feature_families": families,
        "identity_fields": list(freezer.IDENTITY_FIELDS),
        "region_fields": list(freezer.REGION_FIELDS),
        "fixtures": [
            {
                "fixture": fixture,
                "frame_count": 100,
                "tracker_intervals": [[0, 100]],
                "eligible_intervals": [[0, 100]],
                "search_intervals": [[frame, frame + 1]],
            }
            for fixture, frame in zip(freezer.FIXTURE_SPECS, (10, 20, 30), strict=True)
        ],
    }
    manifest_path = tmp_path / freezer.MANIFEST_FILENAME
    if motion_mode is not None:
        manifest["motion_mode"] = motion_mode
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


def test_freeze_verification_rejects_rows_outside_manifest_intervals(tmp_path: Path) -> None:
    manifest_path = _write_verified_fixture(tmp_path)
    feature_path = tmp_path / freezer.FEATURE_FILENAME
    with lzma.open(feature_path, "rb") as source:
        rows = np.load(source, allow_pickle=False)
    rows["frame"][0] = 101
    with lzma.open(feature_path, "wb") as destination:
        np.save(destination, rows, allow_pickle=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_sha256"] = freezer._sha256(feature_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="feature rows differ from search interval"):
        scorer.verify_freeze(manifest_path)


def test_freeze_verification_rejects_wrong_fixture_fps(tmp_path: Path) -> None:
    manifest_path = _write_verified_fixture(tmp_path)
    feature_path = tmp_path / freezer.FEATURE_FILENAME
    with lzma.open(feature_path, "rb") as source:
        rows = np.load(source, allow_pickle=False)
    rows["fps"][0] = 30.0
    with lzma.open(feature_path, "wb") as destination:
        np.save(destination, rows, allow_pickle=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_sha256"] = freezer._sha256(feature_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="feature row fps differs"):
        scorer.verify_freeze(manifest_path)


def test_freeze_motion_mode_defaults_to_legacy_raw_and_rejects_unknown_values(tmp_path: Path) -> None:
    manifest_path = _write_verified_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert scorer._manifest_motion_mode(manifest) == scorer.RAW_MOTION_MODE

    manifest["motion_mode"] = "raw_per_frame"
    assert scorer._manifest_motion_mode(manifest) == "raw_per_frame"
    manifest["motion_mode"] = "base30_per_frame"
    assert scorer._manifest_motion_mode(manifest) == "base30_per_frame"
    manifest["motion_mode"] = "raw"
    with pytest.raises(ValueError, match="motion_mode"):
        scorer._manifest_motion_mode(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="motion_mode"):
        scorer.verify_freeze(manifest_path)


def test_variant_feature_selection_and_motion_mode_contract(tmp_path: Path) -> None:
    raw_manifest = scorer.verify_freeze(_write_verified_fixture(tmp_path))
    assert len(scorer._feature_names(raw_manifest.manifest, "physics")) == 85
    assert (
        len(
            scorer._feature_names(
                raw_manifest.manifest,
                "physics",
                variant=scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT,
            )
        )
        == 55
    )
    with pytest.raises(ValueError, match="requires base30"):
        scorer._feature_names(
            raw_manifest.manifest,
            "physics",
            variant=scorer.PHYSICS_BASE30_MOTION_VARIANT,
        )

    base30_manifest = scorer.verify_freeze(_write_verified_fixture(tmp_path / "base30", "base30_per_frame"))
    assert (
        len(
            scorer._feature_names(
                base30_manifest.manifest,
                "physics",
                variant=scorer.PHYSICS_BASE30_MOTION_VARIANT,
            )
        )
        == 85
    )
    with pytest.raises(ValueError, match="requires raw"):
        scorer._validate_variant_for_manifest(base30_manifest.manifest, scorer.CANDIDATE_VARIANT)
    with pytest.raises(ValueError, match="requires raw"):
        scorer._validate_variant_for_manifest(
            base30_manifest.manifest,
            scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT,
        )


def test_trial_feature_selection_leaves_control_column_sets_unchanged(tmp_path: Path) -> None:
    manifest = scorer.verify_freeze(_write_verified_fixture(tmp_path)).manifest

    for feature_set in scorer.FEATURE_SETS:
        baseline = scorer._base_feature_names(manifest, feature_set)
        if feature_set == "physics":
            trial = scorer._feature_names(
                manifest,
                feature_set,
                variant=scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT,
            )
            assert len(baseline) - len(trial) == 30
        else:
            assert scorer._base_feature_names(manifest, feature_set) == baseline


def test_manifest_intervals_reject_overlaps() -> None:
    manifest = {
        "fixtures": [
            {
                "fixture": fixture,
                "frame_count": 20,
                "tracker_intervals": [[0, 10], [9, 20]] if fixture == "sset_21" else [[0, 20]],
            }
            for fixture in freezer.FIXTURE_SPECS
        ]
    }
    with pytest.raises(ValueError, match="overlap"):
        scorer._manifest_intervals(manifest, "tracker_intervals")


def test_training_selection_keeps_positives_ignores_adjacent_frames() -> None:
    dtype = np.dtype([("fixture", "S7"), ("interval_id", "i2"), ("frame", "i4"), ("fps", "f4")])
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


def test_candidate_scores_preserve_identity_and_current_decision() -> None:
    rows = np.zeros(
        6,
        dtype=[("fixture", "S7"), ("interval_id", "<i2"), ("frame", "<i4")],
    )
    rows["fixture"] = b"sset_01"
    rows["interval_id"] = [0, 0, 0, 0, 1, 1]
    rows["frame"] = [10, 12, 14, 20, 10, 12]
    probabilities = np.asarray([0.5, 0.9, 0.8, 0.4, 0.9, 0.9])
    kept = scorer.temporal_nms(rows["frame"], rows["interval_id"], probabilities, threshold=0.5, radius=3)

    candidates = scorer._candidate_score_rows(rows, probabilities, threshold=0.5, kept=kept)

    assert candidates[["fixture", "interval_id", "frame"]].tolist() == rows.tolist()
    assert candidates["decision"].tolist() == [
        scorer.CANDIDATE_NEARBY_DUPLICATE,
        scorer.CANDIDATE_RETAINED,
        scorer.CANDIDATE_NEARBY_DUPLICATE,
        scorer.CANDIDATE_BELOW_THRESHOLD,
        scorer.CANDIDATE_RETAINED,
        scorer.CANDIDATE_NEARBY_DUPLICATE,
    ]
    assert candidates["timing_score"].tolist() == probabilities.tolist()
    assert candidates["threshold"].tolist() == [0.5] * len(rows)


def test_outer_fold_preserves_the_existing_two_value_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    predictions = np.asarray([10, 20], dtype=np.int32)

    def fake_fold(*_args: object, **_kwargs: object) -> tuple[dict[str, str], np.ndarray, None]:
        return {"test_fixture": "sset_01"}, predictions, None

    monkeypatch.setattr(scorer, "_outer_fold_with_candidate_scores", fake_fold)

    fold, actual_predictions = scorer._outer_fold(
        np.empty(0),
        "sset_01",
        scorer.GroundTruth({}, {}, 0),
        (),
        "histogram_boosting",
    )

    assert fold == {"test_fixture": "sset_01"}
    assert actual_predictions is predictions


def test_candidate_score_manifest_binds_the_freeze_and_tree_result(tmp_path: Path) -> None:
    verified = scorer.verify_freeze(_write_verified_fixture(tmp_path))
    model_rows = verified.rows.copy()
    model_rows["region_wrist"] = 1
    verified = scorer.VerifiedFeatures(verified.manifest_path, verified.manifest, model_rows)

    chunks = []
    folds = []
    for fixture, row in zip(freezer.FIXTURE_SPECS, model_rows, strict=True):
        fixture_rows = np.asarray([row], dtype=model_rows.dtype)
        chunks.append(
            scorer._candidate_score_rows(
                fixture_rows,
                np.asarray([0.8]),
                threshold=0.5,
                kept=np.asarray([0], dtype=np.int32),
            )
        )
        folds.append(
            {
                "test_fixture": fixture,
                "train_fixtures": [name for name in freezer.FIXTURE_SPECS if name != fixture],
                "threshold": 0.5,
                "prediction_count": 1,
                "prediction_frames": [int(row["frame"])],
            }
        )
    candidates = np.concatenate(chunks)
    results = {
        "schema": scorer.RESULTS_SCHEMA,
        "source_commit": verified.manifest["source_commit"],
        "feature_sha256": verified.manifest["feature_sha256"],
        "row_count": len(model_rows),
        "model_row_count": len(model_rows),
        "model_search_surface": "seeded_union",
        "models": {
            "histogram_boosting": {
                "physics": {
                    "feature_count": 85,
                    "folds": folds,
                }
            }
        },
    }
    tree_result_path = tmp_path / "tree_results.json.gz"
    scorer.write_results(tree_result_path, results)
    candidate_path = tmp_path / "candidate_scores.npy.xz"
    candidate_manifest_path = tmp_path / "candidate_scores_manifest.json"
    scorer.write_candidate_scores(
        candidate_path,
        candidate_manifest_path,
        candidates,
        verified,
        tree_result_path,
        results,
    )

    retained = scorer.verify_candidate_scores(candidate_manifest_path, verified, tree_result_path)

    assert np.array_equal(retained.rows, candidates)
    changed_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    changed_manifest["folds"][0]["threshold"] = 0.6
    candidate_manifest_path.write_text(
        json.dumps(changed_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="candidate fold summary"):
        scorer.verify_candidate_scores(candidate_manifest_path, verified, tree_result_path)

    scorer.write_candidate_scores(
        candidate_path,
        candidate_manifest_path,
        candidates,
        verified,
        tree_result_path,
        results,
    )
    scorer.write_results(tree_result_path, {**results, "source_commit": "changed"})
    with pytest.raises(ValueError, match="tree result source_commit"):
        scorer.verify_candidate_scores(candidate_manifest_path, verified, tree_result_path)


def test_nonbaseline_result_records_and_verifies_its_variant(tmp_path: Path) -> None:
    verified = scorer.verify_freeze(_write_verified_fixture(tmp_path))
    result = {
        "schema": scorer.RESULTS_SCHEMA,
        "source_commit": verified.manifest["source_commit"],
        "feature_sha256": verified.manifest["feature_sha256"],
        "row_count": len(verified.rows),
        "model_row_count": 0,
        "model_search_surface": "seeded_union",
        "selected_variant": scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT,
        "models": {
            "histogram_boosting": {
                "physics": {
                    "feature_count": 55,
                    "folds": [
                        {
                            "test_fixture": fixture,
                            "train_fixtures": [name for name in freezer.FIXTURE_SPECS if name != fixture],
                            "threshold": 0.5,
                            "prediction_count": 0,
                            "prediction_frames": [],
                        }
                        for fixture in freezer.FIXTURE_SPECS
                    ],
                }
            }
        },
    }
    result_path = tmp_path / "variant_results.json.gz"
    scorer.write_results(result_path, result)
    verified_result = scorer.verify_tree_result(result_path, verified)
    assert verified_result["selected_variant"] == scorer.PHYSICS_WITHOUT_RAW_MOTION_VARIANT

    changed = {**result, "selected_variant": scorer.CANDIDATE_VARIANT}
    scorer.write_results(result_path, changed)
    with pytest.raises(ValueError, match="must omit"):
        scorer.verify_tree_result(result_path, verified)


def test_seeded_region_mask_selects_each_frame_once() -> None:
    families = freezer._feature_family_names()
    rows = np.zeros(4, dtype=freezer._record_dtype(families))
    rows["region_wrist"][[0, 2]] = 1
    rows["region_serve_lookback"][[1, 2]] = 1
    assert scorer.seeded_region_mask(rows).tolist() == [True, True, True, False]


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
