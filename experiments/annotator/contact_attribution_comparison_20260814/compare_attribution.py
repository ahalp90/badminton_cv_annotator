"""Compare contact-player distance normalisations on pinned ShuttleSet evidence."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from annotator import point_winner, rally_segmentation
from annotator.calibration.fixtures import FIXTURES, REPO_ROOT, SHARED_FILES, Fixture, verify_file
from annotator.calibration.scoring import RallyBoundary, classify_all, load_gt_rallies
from annotator.config import BaseAnnotatorConfig
from annotator.court_evidence import DETECTOR_RESOLUTION, build_net_band, detected_court_info
from annotator.resolve import resolve
from annotator.types import WRIST_L, WRIST_R, StickyResult
from shared.court import HOMOGRAPHY_RESOLUTION


REFERENCE_TAG = "shuttleset-annotator-heuristic-reference-v1"
REFERENCE_RUN = "measurement/current_annotator_8config_288p"
REFERENCE_PARENT = "detected_ckn_opencv_consensus"
METHODS = ("body_height", "image_pixels", "court_projection")
POSE_ROLES = ("bboxes", "scores", "kps", "ndet")
ACTIVE_CORNER_COLUMNS = (
    ("active_tl_x", "active_tl_y"),
    ("active_tr_x", "active_tr_y"),
    ("active_br_x", "active_br_y"),
    ("active_bl_x", "active_bl_y"),
)


@dataclass(frozen=True)
class VideoPaths:
    """Inputs for one fixed video from the public release and pinned pose root."""

    track: Path
    bboxes: Path
    scores: Path
    kps: Path
    ndet: Path
    court_present: Path
    scene_rows: Path
    court_scenes: Path
    annotations: Path


def parse_args() -> argparse.Namespace:
    """Parse paths for the extracted reference release, pose evidence, and output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_digest(path: Path, algorithm: str = "md5") -> str:
    """Hash one file in bounded chunks.

    :param path: File to read.
    :param algorithm: Hashlib algorithm name.
    :return: Lowercase hexadecimal digest.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_release_manifest(reference_root: Path) -> dict[str, dict[str, Any]]:
    """Load the release manifest keyed by release-relative path."""
    path = reference_root / "RELEASE_MANIFEST.json"
    payload = json.loads(path.read_text())
    records = {str(record["release_path"]): record for record in payload["files"]}
    if len(records) != int(payload["file_count"]):
        raise ValueError("release manifest has duplicate or missing file records")
    return records


def verify_released_file(
    reference_root: Path,
    relative_path: str,
    records: dict[str, dict[str, Any]],
) -> Path:
    """Verify one extracted release file against its published MD5."""
    if relative_path not in records:
        raise ValueError(f"release manifest does not contain {relative_path}")
    path = reference_root / relative_path
    expected = str(records[relative_path]["md5"])
    actual = file_digest(path)
    if actual != expected:
        raise ValueError(f"release MD5 mismatch for {relative_path}: {actual} != {expected}")
    return path


def video_paths(
    fixture: Fixture,
    reference_root: Path,
    pose_root: Path,
    release_records: dict[str, dict[str, Any]],
) -> VideoPaths:
    """Resolve and validate every input used for one video."""
    track_relative = f"inputs/{fixture.name}_track_npy/{fixture.name}_track.npy"
    result_relative = f"{REFERENCE_RUN}/{REFERENCE_PARENT}/{fixture.name}/tracknet-stride-8"
    released = {
        role: verify_released_file(reference_root, f"{result_relative}/{filename}", release_records)
        for role, filename in (
            ("court_present", "court_present.npy"),
            ("scene_rows", "scene_rows.csv"),
            ("court_scenes", "court_scenes.csv"),
            ("annotations", "annotations.json"),
        )
    }
    track = verify_released_file(reference_root, track_relative, release_records)
    expected_pose = {
        "bboxes": fixture.digests.bboxes,
        "scores": fixture.digests.scores,
        "kps": fixture.digests.kps,
        "ndet": fixture.digests.ndet,
    }
    pose_paths: dict[str, Path] = {}
    for role in POSE_ROLES:
        path = pose_root / fixture.pose_path(role)
        actual = file_digest(path)
        expected = expected_pose[role]
        if actual != expected:
            raise ValueError(f"pose MD5 mismatch for {fixture.name}/{role}: {actual} != {expected}")
        pose_paths[role] = path
    return VideoPaths(
        track=track,
        bboxes=pose_paths["bboxes"],
        scores=pose_paths["scores"],
        kps=pose_paths["kps"],
        ndet=pose_paths["ndet"],
        court_present=released["court_present"],
        scene_rows=released["scene_rows"],
        court_scenes=released["court_scenes"],
        annotations=released["annotations"],
    )


def load_active_court_info(court_scenes_path: Path) -> dict[str, object]:
    """Reconstruct the detected-video consensus homography from retained active quads."""
    frame = pd.read_csv(court_scenes_path)
    required = {column for pair in ACTIVE_CORNER_COLUMNS for column in pair}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"court scenes are missing active corner columns: {sorted(missing)}")
    valid = frame.dropna(subset=sorted(required))
    if valid.empty:
        raise ValueError("court scenes contain no active quads")
    quads_native = np.stack(
        [valid[[x_column, y_column]].to_numpy(dtype=float) for x_column, y_column in ACTIVE_CORNER_COLUMNS],
        axis=1,
    )  # (scene, corner, xy)
    consensus_native = np.median(quads_native, axis=0)  # (corner, xy)
    scale = np.asarray(HOMOGRAPHY_RESOLUTION, dtype=float) / np.asarray(DETECTOR_RESOLUTION, dtype=float)
    return detected_court_info(consensus_native * scale)


def load_sticky(
    fixture: Fixture,
    paths: VideoPaths,
    resolution_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StickyResult, dict[str, object], tuple[float, float]]:
    """Build current sticky-player evidence from the pinned arrays."""
    track = np.load(paths.track, mmap_mode="r")
    bboxes = np.load(paths.bboxes, mmap_mode="r")
    scores = np.load(paths.scores, mmap_mode="r")
    kps = np.load(paths.kps, mmap_mode="r")
    ndet = np.load(paths.ndet, mmap_mode="r")
    court_present = np.load(paths.court_present)
    scene_rows = pd.read_csv(paths.scene_rows)
    expected_frames = len(track)
    for name, array in (
        ("bboxes", bboxes),
        ("scores", scores),
        ("kps", kps),
        ("ndet", ndet),
        ("court_present", court_present),
    ):
        if len(array) != expected_frames:
            raise ValueError(f"{fixture.name} {name} has {len(array)} rows; expected {expected_frames}")
    if court_present.dtype != np.bool_:
        raise ValueError(f"{fixture.name} court_present must be boolean")

    court_info = load_active_court_info(paths.court_scenes)
    net_band = build_net_band(court_info, fixture.resolution)
    segments = rally_segmentation.tracker_segments(
        scene_rows.to_dict("records"), court_present, expected_frames
    )
    constants = resolve(BaseAnnotatorConfig(), fixture.fps).constants
    sticky = rally_segmentation.build_sticky_result(
        track,
        segments,
        bboxes,
        scores,
        kps,
        ndet,
        str(fixture.video_id),
        {str(fixture.video_id): court_info},
        resolution_table,
        fixture.resolution,
        constants.body_unit_half_window,
    )
    return track, bboxes, kps, sticky, court_info, net_band


def half_from_distances(
    frame: int,
    distances: np.ndarray,
    track: np.ndarray,
    sticky: StickyResult,
    bboxes: np.ndarray,
    net_band: tuple[float, float],
) -> point_winner.Half | None:
    """Select the nearest slot and retain production's foot-versus-net half check."""
    if track[frame, 2] != 1:
        return None
    if not np.isfinite(distances).any():
        return None
    slot_index = int(np.nanargmin(distances))
    pose_index = int(sticky.picks[frame, slot_index])
    if pose_index < 0:
        return None
    foot_y = float(bboxes[frame, pose_index, 3])
    if not np.isfinite(foot_y):
        return None
    if foot_y < net_band[0]:
        return point_winner.Half.TOP
    if foot_y > net_band[1]:
        return point_winner.Half.BOT
    return None


def court_distances(
    frame: int,
    track: np.ndarray,
    sticky: StickyResult,
    kps: np.ndarray,
    resolution: tuple[float, float],
    court_info: dict[str, object],
) -> np.ndarray:
    """Measure wrist-to-shuttle distance after a floor-plane homography projection."""
    distances = np.full(2, np.nan, dtype=float)
    shuttle_px = np.asarray(track[frame, :2], dtype=float) * np.asarray(resolution, dtype=float)
    shuttle_court = point_winner.project_pixels_to_court(
        shuttle_px[:, None], resolution, court_info
    )[:, 0]
    if not np.isfinite(shuttle_court).all():
        return distances
    for slot_index in range(2):
        pose_index = int(sticky.picks[frame, slot_index])
        if pose_index < 0:
            continue
        wrists = np.asarray(kps[frame, pose_index, (WRIST_L, WRIST_R), :], dtype=float)
        if not np.isfinite(wrists).all():
            continue
        projected = point_winner.project_pixels_to_court(wrists.T, resolution, court_info).T
        if np.isfinite(projected).all():
            distances[slot_index] = float(np.linalg.norm(projected - shuttle_court, axis=1).min())
    return distances


def image_distances(
    frame: int,
    track: np.ndarray,
    sticky: StickyResult,
    kps: np.ndarray,
    resolution: tuple[float, float],
) -> np.ndarray:
    """Recompute production's wrist numerator without the body-height divisor."""
    distances = np.full(2, np.nan, dtype=float)
    shuttle_px = np.asarray(track[frame, :2], dtype=float) * np.asarray(resolution, dtype=float)
    for slot_index in range(2):
        pose_index = int(sticky.picks[frame, slot_index])
        if pose_index < 0:
            continue
        wrists = np.asarray(kps[frame, pose_index, (WRIST_L, WRIST_R), :], dtype=float)
        distances[slot_index] = float(np.linalg.norm(wrists - shuttle_px, axis=1).min())
    return distances


def predictions_for_frame(
    frame: int,
    track: np.ndarray,
    bboxes: np.ndarray,
    kps: np.ndarray,
    sticky: StickyResult,
    resolution: tuple[float, float],
    court_info: dict[str, object],
    net_band: tuple[float, float],
) -> tuple[dict[str, point_winner.Half | None], dict[str, np.ndarray]]:
    """Return all three predictions and their two-slot distance vectors."""
    body = np.asarray(sticky.distances_per_slot[frame], dtype=float)
    image = image_distances(frame, track, sticky, kps, resolution)
    projected = court_distances(frame, track, sticky, kps, resolution, court_info)
    distances = {
        "body_height": body,
        "image_pixels": image,
        "court_projection": projected,
    }
    predictions = {
        method: half_from_distances(frame, values, track, sticky, bboxes, net_band)
        for method, values in distances.items()
    }
    return predictions, distances


def fitted_first(final: point_winner.Half | None, n_strokes: int) -> point_winner.Half | None:
    """Return the first alternating half implied by one final half and count."""
    if final is None or n_strokes == 0:
        return None
    return final if n_strokes % 2 == 1 else point_winner.OTHER_HALF[final]


def per_span_predictions(
    method: str,
    annotations: dict[str, Any],
    prediction_cache: dict[int, dict[str, point_winner.Half | None]],
) -> tuple[list[point_winner.Half | None], list[point_winner.Half | None]]:
    """Fit alternating player phases for the published detected contacts."""
    n_spans = len(annotations["spans"])
    contacts = {
        int(rally_id): [int(frame) for frame in frames]
        for rally_id, frames in annotations["filtered_by_rally"].items()
    }
    final_halves: list[point_winner.Half | None] = []
    first_halves: list[point_winner.Half | None] = []
    for rally_id in range(n_spans):
        frames = contacts.get(rally_id, [])
        guesses = [prediction_cache[frame][method] for frame in frames]
        final = point_winner.fit_alternation(guesses)
        final_halves.append(final)
        first_halves.append(fitted_first(final, len(frames)))
    return final_halves, first_halves


def end_to_end_counts(
    fixture: Fixture,
    master: pd.DataFrame,
    spans: list[tuple[int, int]],
    final_halves: list[point_winner.Half | None],
    first_halves: list[point_winner.Half | None],
) -> dict[str, dict[str, int]]:
    """Score rally-final player and server predictions under fixed boundaries and contacts."""
    rallies = load_gt_rallies(master, fixture.video_id)
    classifications = classify_all(spans, rallies)
    video_gt = master.loc[master["vid"] == fixture.video_id, ["frame_num", "player_side"]]
    frame_side = {
        int(frame): point_winner.Half.TOP if str(side) == "Top" else point_winner.Half.BOT
        for frame, side in video_gt.itertuples(index=False, name=None)
    }
    counts = {
        "player": {"primary_correct": 0, "primary_total": len(rallies), "covered_correct": 0, "covered_total": 0},
        "server": {"primary_correct": 0, "primary_total": len(rallies), "covered_correct": 0, "covered_total": 0},
    }
    for rally, (category, span_index) in zip(rallies, classifications):
        covered = category is RallyBoundary.COVERED
        player_gt = frame_side[rally.stroke_frames[-1]]
        server_gt = frame_side[rally.stroke_frames[0]]
        player_pred = final_halves[span_index] if covered and span_index is not None else None
        server_pred = first_halves[span_index] if covered and span_index is not None else None
        player_ok = player_pred == player_gt
        server_ok = server_pred == server_gt
        counts["player"]["primary_correct"] += int(covered and player_ok)
        counts["server"]["primary_correct"] += int(covered and server_ok)
        if covered:
            counts["player"]["covered_total"] += 1
            counts["server"]["covered_total"] += 1
            counts["player"]["covered_correct"] += int(player_ok)
            counts["server"]["covered_correct"] += int(server_ok)
    return counts


def summarise_contact_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact-contact coverage, accuracy, side splits, and disagreements."""
    summary: dict[str, Any] = {"methods": {}, "versus_body_height": {}}
    scopes: dict[str, list[dict[str, Any]]] = {"all": rows}
    for video_name in sorted({str(row["video"]) for row in rows}):
        scopes[video_name] = [row for row in rows if row["video"] == video_name]
    for gt_side in ("Top", "Bot"):
        scopes[f"gt_{gt_side.lower()}"] = [row for row in rows if row["gt_side"] == gt_side]

    for method in METHODS:
        summary["methods"][method] = {}
        for scope, scoped_rows in scopes.items():
            eligible = [row for row in scoped_rows if row[f"{method}_pred"] is not None]
            correct = sum(row[f"{method}_pred"] == row["gt_side"] for row in eligible)
            predicted_bot = sum(row[f"{method}_pred"] == "Bot" for row in eligible)
            summary["methods"][method][scope] = {
                "total": len(scoped_rows),
                "eligible": len(eligible),
                "correct": correct,
                "accuracy": correct / len(eligible) if eligible else None,
                "predicted_bot": predicted_bot,
            }

    for candidate in METHODS[1:]:
        summary["versus_body_height"][candidate] = {}
        for scope, scoped_rows in scopes.items():
            comparable = [
                row
                for row in scoped_rows
                if row["body_height_pred"] is not None and row[f"{candidate}_pred"] is not None
            ]
            fixes = sum(
                row["body_height_pred"] != row["gt_side"] and row[f"{candidate}_pred"] == row["gt_side"]
                for row in comparable
            )
            damages = sum(
                row["body_height_pred"] == row["gt_side"] and row[f"{candidate}_pred"] != row["gt_side"]
                for row in comparable
            )
            summary["versus_body_height"][candidate][scope] = {
                "comparable": len(comparable),
                "disagreements": fixes + damages,
                "fixes": fixes,
                "damages": damages,
                "net_fixes": fixes - damages,
                "mcnemar_exact_p": exact_two_sided_binomial(fixes, damages),
            }
    return summary


def exact_two_sided_binomial(fixes: int, damages: int) -> float | None:
    """Return the exact two-sided sign-test p-value for discordant pairs."""
    discordant = fixes + damages
    if discordant == 0:
        return None
    smaller = min(fixes, damages)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def nullable_float(value: float) -> float | None:
    """Convert non-finite diagnostics to JSON/CSV-safe nulls."""
    return float(value) if np.isfinite(value) else None


def write_gzip_text(path: Path, content: str) -> None:
    """Write deterministic UTF-8 gzip content."""
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(content.encode("utf-8"))


def write_contact_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the exact-contact comparison as deterministic compressed CSV."""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_gzip_text(path, output.getvalue())


def current_commit() -> str:
    """Return the source commit used for this comparison."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    """Run the three-method comparison and persist its evidence."""
    args = parse_args()
    reference_root = args.reference_root.resolve()
    pose_root = args.pose_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    release_records = load_release_manifest(reference_root)
    for pin in (SHARED_FILES[0], SHARED_FILES[2]):
        verify_file(pin)
    master = pd.read_csv(REPO_ROOT / SHARED_FILES[0].path)
    resolution_table = pd.read_csv(REPO_ROOT / SHARED_FILES[2].path).set_index("id")
    resolution_table.index = resolution_table.index.astype(str)

    contact_rows: list[dict[str, Any]] = []
    end_to_end: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
    provenance_files: dict[str, dict[str, str]] = {}
    for fixture in FIXTURES:
        paths = video_paths(fixture, reference_root, pose_root, release_records)
        track, bboxes, kps, sticky, court_info, net_band = load_sticky(
            fixture, paths, resolution_table
        )
        annotations = json.loads(paths.annotations.read_text())
        filtered_frames = {
            int(frame)
            for frames in annotations["filtered_by_rally"].values()
            for frame in frames
        }
        gt = master.loc[master["vid"] == fixture.video_id, ["frame_num", "player_side"]]
        gt_frames = {int(frame) for frame in gt["frame_num"]}
        needed_frames = sorted(gt_frames | filtered_frames)
        prediction_cache: dict[int, dict[str, point_winner.Half | None]] = {}
        distance_cache: dict[int, dict[str, np.ndarray]] = {}
        for frame in needed_frames:
            predictions, distances = predictions_for_frame(
                frame,
                track,
                bboxes,
                kps,
                sticky,
                fixture.resolution,
                court_info,
                net_band,
            )
            prediction_cache[frame] = predictions
            distance_cache[frame] = distances

        for frame, side in gt.itertuples(index=False, name=None):
            frame = int(frame)
            gt_side = "Top" if str(side) == "Top" else "Bot"
            predictions = prediction_cache[frame]
            distances = distance_cache[frame]
            row: dict[str, Any] = {
                "video": fixture.name,
                "frame": frame,
                "gt_side": gt_side,
            }
            for method in METHODS:
                prediction = predictions[method]
                row[f"{method}_pred"] = prediction.value if prediction is not None else None
                row[f"{method}_top_distance"] = nullable_float(float(distances[method][0]))
                row[f"{method}_bot_distance"] = nullable_float(float(distances[method][1]))
            row["top_bbox_height"] = nullable_float(float(sticky.bbox_height[frame, 0]))
            row["bot_bbox_height"] = nullable_float(float(sticky.bbox_height[frame, 1]))
            contact_rows.append(row)

        body_final, body_first = per_span_predictions(
            "body_height", annotations, prediction_cache
        )
        stored_final = [
            point_winner.Half(value) if value is not None else None
            for value in annotations["striker_halves"]
        ]
        stored_first = [
            point_winner.Half(value) if value is not None else None
            for value in annotations["fitted_first_all"]
        ]
        if body_final != stored_final or body_first != stored_first:
            raise ValueError(
                f"{fixture.name}: reconstructed body-height attribution does not reproduce "
                "the published baseline"
            )

        spans = [tuple(map(int, span)) for span in annotations["spans"]]
        for method in METHODS:
            final_halves, first_halves = (
                (body_final, body_first)
                if method == "body_height"
                else per_span_predictions(method, annotations, prediction_cache)
            )
            end_to_end[method][fixture.name] = end_to_end_counts(
                fixture, master, spans, final_halves, first_halves
            )

        provenance_files[fixture.name] = {
            "track_md5": file_digest(paths.track),
            "bboxes_md5": file_digest(paths.bboxes),
            "scores_md5": file_digest(paths.scores),
            "kps_md5": file_digest(paths.kps),
            "ndet_md5": file_digest(paths.ndet),
            "court_present_md5": file_digest(paths.court_present),
            "scene_rows_md5": file_digest(paths.scene_rows),
            "court_scenes_md5": file_digest(paths.court_scenes),
            "annotations_md5": file_digest(paths.annotations),
        }

    summary = {
        "schema_version": 1,
        "question": "Does another distance normalisation improve contact-player attribution over per-player bbox height?",
        "methods": {
            "body_height": "nearest wrist image distance divided by that player's mean bbox height",
            "image_pixels": "nearest wrist image distance with no per-player scale",
            "court_projection": "nearest wrist-to-shuttle distance after floor-plane homography projection",
        },
        "population": {
            "videos": [fixture.name for fixture in FIXTURES],
            "ground_truth_contacts": len(contact_rows),
            "end_to_end_boundaries_and_contacts": "published detected_ckn_opencv_consensus stride-8 annotations",
        },
        "provenance": {
            "comparison_commit": current_commit(),
            "reference_tag": REFERENCE_TAG,
            "reference_measurement_source_commit": "189c5af58e45d23ae827dde516924194eb238e18",
            "pose_source": str(pose_root),
            "reference_source": str(reference_root),
            "files": provenance_files,
        },
        "baseline_reproduced": True,
        "exact_contacts": summarise_contact_rows(contact_rows),
        "end_to_end": end_to_end,
        "limitations": [
            "The three videos are the existing calibration fixtures, not a held-out production test.",
            "Homography projection treats airborne wrists and shuttle positions as if they lie on the court plane.",
            "End-to-end comparisons hold the published rally spans and detected contact frames fixed.",
        ],
    }
    write_contact_rows(output_dir / "contact_rows.csv.gz", contact_rows)
    encoded_summary = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_gzip_text(output_dir / "summary.json.gz", encoded_summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
