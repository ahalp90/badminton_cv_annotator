from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from experiments.build_multiscale_trials import (
    _candidate_records,
    _select_all_eligible,
    _select_pilot,
    build_parser,
    clip_segments,
    load_raw_cut_segments,
    scene_stratum,
    storyboard_source_frames,
)
from experiments.multiscale_schema import Segment

from experiments import build_multiscale_trials as builder


def test_load_and_clip_segments_keep_source_global_ids(tmp_path: Path) -> None:
    path = tmp_path / "court_evidence.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"raw_cuts": [[0, 100], [100, 180], [180, 300]]}, stream)

    segments = load_raw_cut_segments(path)
    clipped = clip_segments(segments, 50, 250)

    assert clipped == (
        Segment("S00000", 50, 100),
        Segment("S00001", 100, 180),
        Segment("S00002", 180, 250),
    )


def test_storyboard_keeps_both_sides_of_cuts_and_target_edges() -> None:
    segments = (
        Segment("S00010", 10_000, 10_100),
        Segment("S00011", 10_100, 10_180),
        Segment("S00012", 10_180, 10_300),
    )

    frames = storyboard_source_frames(
        segments,
        10_000,
        10_300,
        10_120,
        10_160,
        20,
    )

    assert frames is not None
    assert len(frames) == 20
    assert frames == tuple(sorted(set(frames)))
    assert {
        10_000,
        10_099,
        10_100,
        10_119,
        10_120,
        10_159,
        10_160,
        10_179,
        10_180,
        10_299,
    } <= set(frames)


def test_storyboard_rejects_a_case_whose_required_frames_do_not_fit() -> None:
    segments = tuple(
        Segment(f"S{index:05d}", index * 2, index * 2 + 2) for index in range(10)
    )

    frames = storyboard_source_frames(segments, 0, 20, 4, 8, 12)

    assert frames is None


def test_scene_strata_are_fixed_before_sampling() -> None:
    assert scene_stratum({"live": 0.92, "cutaway": 0.08}) == "clear_live"
    assert scene_stratum({"replay": 0.80, "live": 0.20}) == "replay_or_cutaway"
    assert scene_stratum({"live": 0.60, "cutaway": 0.40}) == "mixed"
    assert scene_stratum({"other": 1.0}) is None


def test_default_pilot_selection_remains_balanced() -> None:
    records = [
        {
            "video_id": video_id,
            "sort_frame": offset + frame,
            "span_id": frame,
            "stratum": stratum,
        }
        for video_id in ("sset_15", "sset_01")
        for stratum, offset in (
            ("clear_live", 0),
            ("replay_or_cutaway", 100),
            ("mixed", 200),
        )
        for frame in (10, 20, 30, 40)
    ]

    selected = _select_pilot(records, 12)

    assert [
        (record["video_id"], record["stratum"], record["sort_frame"])
        for record in selected
    ] == [
        ("sset_01", "clear_live", 10),
        ("sset_01", "clear_live", 40),
        ("sset_01", "replay_or_cutaway", 110),
        ("sset_01", "replay_or_cutaway", 140),
        ("sset_01", "mixed", 210),
        ("sset_01", "mixed", 240),
        ("sset_15", "clear_live", 10),
        ("sset_15", "clear_live", 40),
        ("sset_15", "replay_or_cutaway", 110),
        ("sset_15", "replay_or_cutaway", 140),
        ("sset_15", "mixed", 210),
        ("sset_15", "mixed", 240),
    ]


def test_parser_separates_pilot_and_all_eligible_modes() -> None:
    parser = build_parser()
    common = [
        "--artifacts-root",
        "artifacts",
        "--repo-root",
        "repo",
        "--scene-labels-dir",
        "labels",
        "--video",
        "sset_01",
        "--context-seconds",
        "90",
        "--context-seconds",
        "120",
        "--out",
        "out",
    ]

    default_args = parser.parse_args(common)
    assert default_args.pilot_cases is None
    assert not default_args.all_eligible

    all_args = parser.parse_args([*common, "--all-eligible"])
    assert all_args.all_eligible

    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--pilot-cases", "12", "--all-eligible"])


def test_all_eligible_selection_is_stable_with_uneven_strata() -> None:
    records = [
        {"video_id": "sset_15", "sort_frame": 20, "span_id": 3, "stratum": "mixed"},
        {
            "video_id": "sset_01",
            "sort_frame": 50,
            "span_id": 7,
            "stratum": "clear_live",
        },
        {
            "video_id": "sset_01",
            "sort_frame": 10,
            "span_id": 4,
            "stratum": "clear_live",
        },
        {"video_id": "sset_01", "sort_frame": 10, "span_id": 2, "stratum": "mixed"},
    ]

    selected = _select_all_eligible(records)

    assert [
        (record["video_id"], record["sort_frame"], record["span_id"])
        for record in selected
    ] == [
        ("sset_01", 10, 2),
        ("sset_01", 10, 4),
        ("sset_01", 50, 7),
        ("sset_15", 20, 3),
    ]
    assert {record["stratum"] for record in selected} == {"clear_live", "mixed"}


def test_wide_mode_includes_labelled_spans_outside_pilot_strata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = SimpleNamespace(
        name="sset_01",
        track=[None] * 4_000,
        result={"spans": [[1_000, 1_200]]},
        scene_labels=object(),
        fps=25.0,
    )
    segments = (Segment("S00000", 0, 4_000),)
    monkeypatch.setattr(builder, "_scene_fractions", lambda *_args: {"other": 1.0})
    monkeypatch.setattr(
        builder,
        "_truth_intervals",
        lambda *_args: [{"truth": "other"}],
    )

    pilot_eligible, pilot_excluded = _candidate_records(
        [video],
        {"sset_01": segments},
        (90, 120),
        96,
    )
    wide_eligible, wide_excluded = _candidate_records(
        [video],
        {"sset_01": segments},
        (90, 120),
        96,
        include_unstratified=True,
    )

    assert pilot_eligible == []
    assert len(pilot_excluded) == 1
    assert len(wide_eligible) == 1
    assert wide_eligible[0]["stratum"] == "other_or_unstratified"
    assert wide_excluded == []


def test_all_eligible_selection_is_recorded_in_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    shots_master = repo_root / "training/data/shuttleset/annotations/shots_master.csv"
    shots_master.parent.mkdir(parents=True)
    shots_master.write_text("vid\n1\n", encoding="utf-8")

    artifacts_root = tmp_path / "artifacts"
    source_paths = {
        name: artifacts_root / f"{name}-{suffix}"
        for name, suffix in (
            ("source", "video.avi"),
            ("result", "result.json.gz"),
            ("track", "track.npy.xz"),
            ("bboxes", "bboxes.npy.xz"),
            ("kps", "kps.npy.xz"),
            ("court_present", "court.npy.xz"),
            ("raw_mask", "raw.npy.xz"),
            ("definitive_mask", "definitive.npy.xz"),
            ("scene_labels", "scene.csv.gz"),
            ("cuts", "court_evidence.json.gz"),
        )
    }
    for path in source_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input")
    court_path = artifacts_root / "stages/court/sset_01/court_evidence.json.gz"
    court_path.parent.mkdir(parents=True, exist_ok=True)
    court_path.write_bytes(b"input")

    class FakeVideo:
        name = "sset_01"
        fps = 25.0
        track = (0,)
        result: ClassVar[dict[str, list[list[int]]]] = {"spans": [[0, 1]]}

        source_path = source_paths["source"]
        result_path = source_paths["result"]
        track_path = source_paths["track"]
        bboxes_path = source_paths["bboxes"]
        kps_path = source_paths["kps"]
        court_present_path = source_paths["court_present"]
        raw_mask_path = source_paths["raw_mask"]
        definitive_mask_path = source_paths["definitive_mask"]
        scene_labels_path = source_paths["scene_labels"]

    segment = Segment("S00000", 0, 1)

    def record(pair_id: str, sort_frame: int, span_id: int) -> dict:
        context = {
            "source_start_frame": 0,
            "source_end_frame": 1,
            "source_frames": (0,),
            "segments": (segment,),
        }
        return {
            "pair_id": pair_id,
            "video_id": "sset_01",
            "sort_frame": sort_frame,
            "span_id": span_id,
            "target_start_frame": 0,
            "target_end_frame": 1,
            "stratum": "mixed",
            "scene_fractions": {"live": 0.5, "cutaway": 0.5},
            "truth_intervals": [],
            "contexts": {90: context, 120: context},
        }

    eligible = [
        record("context-sset_01-r001", 20, 1),
        record("context-sset_01-r000", 10, 0),
    ]
    excluded = [
        {"pair_id": "context-sset_01-r002", "reason": "scene stratum is undefined"}
    ]
    monkeypatch.setattr(builder, "_load_video", lambda *_args: FakeVideo())
    monkeypatch.setattr(builder, "load_raw_cut_segments", lambda _path: (segment,))
    monkeypatch.setattr(
        builder, "_alignment_report", lambda *_args: {"video_id": "sset_01"}
    )
    monkeypatch.setattr(
        builder,
        "_candidate_records",
        lambda *_args, **_kwargs: (eligible, excluded),
    )
    monkeypatch.setattr(builder, "_automatic_priors", lambda *_args: {"span_id": 0})
    monkeypatch.setattr(
        builder,
        "_write_storyboard",
        lambda _video, case: case.clip_path.write_bytes(b"clip"),
    )
    monkeypatch.setattr(builder, "load_manifest", lambda _path: ())
    monkeypatch.setattr(builder, "validate_context_pairs", lambda _cases: None)

    output_dir = tmp_path / "output"
    builder.build_multiscale_trials(
        artifacts_root,
        repo_root,
        tmp_path / "labels",
        output_dir,
        video_names=("sset_01",),
        pilot_cases=12,
        context_seconds=(90, 120),
        max_frames=1,
        all_eligible=True,
    )

    provenance = json.loads(
        (output_dir / "scoring/provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["settings"]["selection_mode"] == "all_eligible"
    assert provenance["settings"]["selected_count"] == 2
    assert provenance["settings"]["eligible_count"] == 2
    assert provenance["settings"]["excluded_count"] == 1
