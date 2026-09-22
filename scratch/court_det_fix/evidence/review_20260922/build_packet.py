"""Build the bounded, readable court-detector review packet."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

CASE_FRAME_PATHS = {
    "am1_window_00_frame_54": "scratch/court_det_fix/frozen_views/frames/amateur/am1/frame_00000054.png",
    "am2_window_00_frame_150": "scratch/court_det_fix/frozen_views/frames/amateur/am2/frame_00000150.png",
    "am2_window_01_frame_28019": "scratch/court_det_fix/frozen_views/frames/amateur/am2/frame_00028019.png",
    "am3_window_00_frame_0": "scratch/court_det_fix/frozen_views/frames/amateur/am3/frame_00000000.png",
    "am3_window_01_frame_10514": "scratch/court_det_fix/frozen_views/frames/amateur/am3/frame_00010514.png",
    "am4_window_00_frame_0": "scratch/court_det_fix/frozen_views/frames/amateur/am4/frame_00000000.png",
    "am4_window_01_frame_13782": "scratch/court_det_fix/frozen_views/frames/amateur/am4/frame_00013782.png",
    "centre_short_frame_36": "scratch/court_det_fix/frozen_views/frames/amateur/centre/frame_00000036.png",
    "letterboxed_short_frame_45": (
        "scratch/court_det_fix/frozen_views/frames/amateur/letterboxed/frame_00000045.png"
    ),
    "yellow_short_frame_14": "scratch/court_det_fix/frozen_views/frames/amateur/yellow/frame_00000014.png",
    "gxBQ_window_00_frame_0": "scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_00_frame_00000000.png",
    "gxBQ_window_00_frame_5": "scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_00_frame_00000005.png",
    "gxBQ_window_00_frame_689": "scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_00_frame_00000689.png",
    "gxBQ_window_01_frame_5111": "scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_01_frame_00005111.png",
    "gxBQ_window_02_frame_5766": "scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_02_frame_00005766.png",
    "gxBQ_window_03_frame_77876": "scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_03_frame_00077876.png",
    "gxBQ_window_04_frame_86088": "scratch/court_det_fix/frozen_views/frames/gx/images/gxBQ_window_04_frame_00086088.png",
    "shuttleset_03_scene_0016": "scratch/court_det_fix/frozen_views/frames/original/video_03/scene_0016_frame_00036665_cached_view.png",
    "shuttleset_03_scene_0017": "scratch/court_det_fix/frozen_views/frames/original/video_03/scene_0017_frame_00038343_cached_view.png",
    "shuttleset_03_scene_0019": "scratch/court_det_fix/frozen_views/frames/original/video_03/scene_0019_frame_00040691_cached_view.png",
    "shuttleset_03_scene_0029": "scratch/court_det_fix/frozen_views/frames/original/video_03/scene_0029_frame_00062591_cached_view.png",
    "shuttleset_03_scene_0034": "scratch/court_det_fix/frozen_views/frames/original/video_03/scene_0034_frame_00067324_cached_view.png",
    "shuttleset_03_scene_0038": "scratch/court_det_fix/frozen_views/frames/original/video_03/scene_0038_frame_00078284_cached_view.png",
    "shuttleset_21_scene_0000": "scratch/court_det_fix/frozen_views/frames/original/video_21/scene_0000_frame_00013558_cached_view.png",
    "shuttleset_21_scene_0010": "scratch/court_det_fix/frozen_views/frames/original/video_21/scene_0010_frame_00023159_cached_view.png",
    "shuttleset_21_scene_0020": "scratch/court_det_fix/frozen_views/frames/original/video_21/scene_0020_frame_00046403_cached_view.png",
    "shuttleset_21_scene_0039": "scratch/court_det_fix/frozen_views/frames/original/video_21/scene_0039_frame_00085565_cached_view.png",
}

W5_ARMS = {
    "3_3": "w5_directional_20260921_r5_33",
    "4_3": "w5_directional_20260921_r5_43",
    "5_3": "w5_directional_20260921_r5_53",
}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_gz(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(value, handle, separators=(",", ":"), sort_keys=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def repo_link(repo_root: Path, packet_root: Path, relative_source: str) -> str:
    source = repo_root / relative_source
    return os.path.relpath(source, packet_root).replace(os.sep, "/")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def compact_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    gates = candidate.get("gates", {})
    return {
        "origin_key": candidate.get("origin_key"),
        "candidate_id": candidate.get("candidate_id"),
        "source": candidate.get("source"),
        "source_memberships": candidate.get("source_memberships"),
        "kind": candidate.get("kind"),
        "parent_origin_key": candidate.get("parent_origin_key"),
        "corners_px": candidate.get("corners_px"),
        "homography_working": candidate.get("homography_working"),
        "camera_eligible": candidate.get("camera_eligible"),
        "gates": gates,
        "scores": {
            key: candidate.get(key)
            for key in (
                "q_geom",
                "q_paint10",
                "q_geom_span_weighted",
                "q_paint10_span_weighted",
                "exclusive_score",
                "directional",
            )
            if key in candidate
        },
    }


def candidate_image_name(case_id: str, origin_key: str, suffix: str = "full") -> str:
    return f"{case_id}__{safe_name(origin_key)}__{suffix}.png"


def convert_preview(source: Path, destination: Path, quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(destination, format="JPEG", quality=quality, optimize=True)


def convert_source_preview(
    repo_root: Path,
    packet_root: Path,
    source_relative: str,
    destination_relative: str,
    quality: int,
) -> str:
    source = repo_root / source_relative
    destination = packet_root / destination_relative
    if not source.is_file():
        raise FileNotFoundError(source)
    convert_preview(source, destination, quality)
    return destination_relative


def source_candidate_path(arm_directory: Path, case_id: str, origin_key: str) -> Path:
    return arm_directory / "gallery" / candidate_image_name(case_id, origin_key)


def frame_link(repo_root: Path, packet_root: Path, case_id: str) -> str:
    return repo_link(repo_root, packet_root, CASE_FRAME_PATHS[case_id])


def build_w5(repo_root: Path, packet_root: Path) -> None:
    base = repo_root / "scratch/court_det_fix/evidence/holistic_admission/directional_20260921_r5"
    comparison = read_json(base / "w5_directional_20260921_r5_comparison/comparison.json")
    review_by_arm: dict[str, dict[str, Any]] = {}
    for arm_id, run_name in W5_ARMS.items():
        review_by_arm[arm_id] = read_json(base / run_name / "review_candidates.json")

    selected_rows: list[dict[str, Any]] = []
    all_arm_rows: list[dict[str, Any]] = []
    for case in comparison["cases"]:
        case_id = case["case_id"]
        for arm_id, run_name in W5_ARMS.items():
            review = review_by_arm[arm_id][case_id]
            selected = review["selected"]
            candidate_map = review["candidates"]
            winner = case["arms"][arm_id]["c_winner"]
            if winner != selected["C"]:
                raise ValueError(f"W5 comparison/review mismatch for {case_id} {arm_id}")
            candidate_keys = {
                "A_line": review["A"].get("line"),
                "A_paint": review["A"].get("paint"),
                "B": selected.get("B"),
                "C": selected.get("C"),
            }
            row = {
                "case_id": case_id,
                "arm": arm_id,
                "run": run_name,
                "floor": list(map(int, arm_id.split("_"))),
                "winner": winner,
                "winner_source": case["arms"][arm_id].get("winner_source"),
                "status": case["arms"][arm_id].get("status"),
                "pool_counts": {
                    key: case["arms"][arm_id].get(key)
                    for key in (
                        "hypotheses_before",
                        "hypotheses_after",
                        "hypotheses_rejected",
                        "combined_admission_hypotheses",
                        "floor_zero_scanned_for_proposal_cap",
                        "floor_zero_selected_count",
                    )
                },
                "candidates": {
                    label: compact_candidate(candidate_map.get(origin_key))
                    for label, origin_key in candidate_keys.items()
                },
                "candidate_keys": candidate_keys,
                "reference_near": review.get("reference_near"),
                "source_review_record": repo_link(
                    repo_root,
                    packet_root,
                    f"scratch/court_det_fix/evidence/holistic_admission/directional_20260921_r5/{run_name}/review_candidates.json",
                ),
            }
            all_arm_rows.append(row)

            arm_directory = base / run_name
            image_source = source_candidate_path(arm_directory, case_id, winner)
            if not image_source.is_file():
                raise FileNotFoundError(image_source)
            selected_rows.append(
                {
                    "case_id": case_id,
                    "arm": arm_id,
                    "winner": winner,
                    "source_relative": str(image_source.relative_to(repo_root)).replace(os.sep, "/"),
                    "source_png": repo_link(repo_root, packet_root, str(image_source.relative_to(repo_root))),
                    "preview": None,
                    "candidate": compact_candidate(candidate_map.get(winner)),
                }
            )

    distinct_rows: list[dict[str, Any]] = []
    seen_selection: set[tuple[str, str]] = set()
    for row in selected_rows:
        key = (row["case_id"], row["winner"])
        if key in seen_selection:
            continue
        seen_selection.add(key)
        source = repo_root / row["source_relative"]
        destination = packet_root / "w5/overlays" / f"selected__{safe_name(row['case_id'])}__{safe_name(row['arm'])}.jpg"
        convert_preview(source, destination, quality=95)
        row["preview"] = os.path.relpath(destination, packet_root / "w5").replace(os.sep, "/")
        distinct_rows.append(row)

    if len(distinct_rows) != 30:
        raise ValueError(f"Expected 30 distinct W5 selections, found {len(distinct_rows)}")

    evaluation_gallery = base / "evaluation/gallery"
    ablation_images: dict[str, str] = {}
    for source in sorted(evaluation_gallery.glob("*.png")):
        destination_name = f"ablation__{source.stem}.jpg"
        destination = packet_root / "w5/overlays" / destination_name
        convert_preview(source, destination, quality=95)
        ablation_images[str(source.resolve())] = f"overlays/{destination_name}"
    landmark_gallery = base / "evaluation/visible_landmark_gallery"
    landmark_images: dict[str, str] = {}
    for source in sorted(landmark_gallery.glob("*.png")):
        destination_name = f"landmark__{source.stem}.jpg"
        destination = packet_root / "w5/overlays" / destination_name
        convert_preview(source, destination, quality=95)
        landmark_images[source.name] = f"overlays/{destination_name}"

    ablation = read_json_gz(base / "evaluation/ablation_winner_details.json.gz")
    for row in ablation["rows"]:
        source_path = row.pop("image_path", None)
        source = Path(source_path or "")
        if not source.is_absolute():
            source = repo_root / source
        source_key = str(source.resolve()) if source_path else ""
        if source_key and source.is_file() and source_key not in ablation_images:
            destination_name = f"ablation__{safe_name(source.parent.name)}__{source.stem}.jpg"
            destination = packet_root / "w5/overlays" / destination_name
            convert_preview(source, destination, quality=95)
            ablation_images[source_key] = f"overlays/{destination_name}"
        row["preview"] = ablation_images.get(source_key)
        row["source_path"] = (
            str(source.relative_to(repo_root)).replace(os.sep, "/") if source_path else None
        )
    write_json_gz(
        packet_root / "w5/results.json.gz",
        {
            "schema": "court-detector-review-w5/1",
            "distinct_final_selections": distinct_rows,
            "all_arm_selections": all_arm_rows,
            "admission_ablation": ablation["rows"],
            "visible_landmark_previews": landmark_images,
            "source_comparison": repo_link(
                repo_root,
                packet_root,
                "scratch/court_det_fix/evidence/holistic_admission/directional_20260921_r5/w5_directional_20260921_r5_comparison/comparison.json",
            ),
        },
    )

    per_view = {}
    with (base / W5_ARMS["4_3"] / "per_view.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            per_view[row["case_id"]] = row

    visual_rows: list[list[str]] = []
    visual_review = base / "visual_review.md"
    in_table = False
    for line in visual_review.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Case |"):
            in_table = True
            continue
        if in_table and line.startswith("| ---"):
            continue
        if in_table and line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == 4:
                visual_rows.append(cells)
        elif in_table and line.startswith("## "):
            break
    outcome_lines = [
        "# W5 final selections and review previews",
        "",
        "The 30 JPEGs are native-resolution quality-95 previews converted from the saved arm PNGs. They are lossy review copies; exact corners, working homographies, gates and scores are in [w5/results.json.gz](w5/results.json.gz).",
        "",
        "| Case | Arm | Final source-qualified winner | Preview | Raw frame | Camera error | Player fractions |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in distinct_rows:
        candidate = row["candidate"] or {}
        gates = candidate.get("gates", {})
        preview = f"w5/{row['preview']}"
        raw_frame = frame_link(repo_root, packet_root, row["case_id"])
        outcome_lines.append(
            f"| {row['case_id']} | {row['arm']} | `{row['winner']}` | [{preview}]({preview}) | [{row['case_id']}]({raw_frame}) | {gates.get('camera_error', '')} | `{gates.get('player_fractions', '')}` |"
        )
    outcome_lines.extend(
        [
            "",
            "The three arm-3,3-only selections are retained because they differ from the tied 4,3/5,3 result: `am2_window_01_frame_28019`, `yellow_short_frame_14`, and `am4_window_01_frame_13782`.",
            "",
            "## Gate and source ablations",
            "",
            "Rows below reproduce the saved admission-ablation winner IDs and scores. A blank preview means the saved source gallery did not contain that row's image.",
            "",
            "| Case | Filter | Winner | Rank | Eligible | Camera error | Player fractions | Preview |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in ablation["rows"]:
        preview = row.get("preview") or ""
        preview = f"w5/{preview}" if preview else ""
        preview_cell = f"[{preview}]({preview})" if preview else ""
        outcome_lines.append(
            f"| {row.get('case_id', '')} | {row.get('filter', '')} | `{row.get('winner', '')}` | {row.get('original_rank', '')} | {row.get('eligible_count', '')} | {row.get('camera_error', '')} | `{row.get('player_fractions', '')}` | {preview_cell} |"
        )
    outcome_lines.extend(
        [
            "",
            "## Full-frame rulings",
            "",
            "| Case | Floors | Ruling | Visible reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for case_id, floors, ruling, reason in visual_rows:
        outcome_lines.append(f"| {case_id} | {floors} | {ruling} | {reason} |")
    outcome_lines.extend(
        [
            "",
            "## Exact selected geometry",
            "",
            "Corners and working homographies below are copied from the saved candidate records. The compact JSON record retains the same values plus all gate and score fields.",
            "",
            "| Case | Arm | Origin | Corners px | Working homography |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in distinct_rows:
        candidate = row.get("candidate") or {}
        outcome_lines.append(
            f"| {row['case_id']} | {row['arm']} | `{row['winner']}` | `{json.dumps(candidate.get('corners_px'), separators=(',', ':'))}` | `{json.dumps(candidate.get('homography_working'), separators=(',', ':'))}` |"
        )
    outcome_lines.extend(
        [
            "",
            "### Failure-case alternatives",
            "",
            "The saved alternatives below make the three post-gate failures inspectable without opening the full W5 records.",
            "",
            "| Case | Candidate role | Origin | Corners px | Working homography |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in all_arm_rows:
        if row["arm"] != "4_3" or row["case_id"] not in {"yellow_short_frame_14", "am1_window_00_frame_54", "shuttleset_21_scene_0010"}:
            continue
        for role in ("A_line", "A_paint", "B", "C"):
            candidate = (row["candidates"].get(role) or {})
            if not candidate:
                continue
            outcome_lines.append(
                f"| {row['case_id']} | {role} | `{candidate.get('origin_key')}` | `{json.dumps(candidate.get('corners_px'), separators=(',', ':'))}` | `{json.dumps(candidate.get('homography_working'), separators=(',', ':'))}` |"
            )
    outcome_lines.extend(
        [
            "",
            "The ruling is qualitative development evidence, not a held-out accuracy estimate. The source review note is [visual_review.md](../holistic_admission/directional_20260921_r5/visual_review.md).",
        ]
    )
    write_text(packet_root / "w5.md", "\n".join(outcome_lines))
    geometry_lines = [
        "# W5 final-winner geometry",
        "",
        (
            "These are the 30 distinct arm winners, copied without rounding from the saved review records. "
            "Corners are native-image pixels in saved court order. Homographies map court metres to the "
            "960×540 working image. This table does not contain every post-hoc gate/source-ablation winner."
        ),
        "",
    ]
    for row in distinct_rows:
        candidate = row["candidate"]
        geometry_lines.extend([
            f"## {row['case_id']} — {row['winner']}",
            "",
            "```json",
            json.dumps({key: candidate[key] for key in ("corners_px", "homography_working")}, indent=2),
            "```",
            "",
        ])
    write_text(packet_root / "w5/geometry.md", "\n".join(geometry_lines))


def selected_population_candidate(
    populations: dict[tuple[str, str, str], dict[str, Any]],
    origin_key: str | None,
    case_id: str,
) -> dict[str, Any] | None:
    if not origin_key:
        return None
    source, _, candidate_id = origin_key.partition(":")
    candidate = populations.get((case_id, source, candidate_id))
    if candidate is None:
        return None
    return {
        "origin_key": origin_key,
        "candidate_id": candidate.get("candidate_id"),
        "source": source,
        "corners_px": candidate.get("corners_px"),
        "homography_working": candidate.get("homography_working"),
        "gates": candidate.get("gates"),
        "axis_score": candidate.get("axis_score"),
        "shortlist_score": candidate.get("shortlist_score"),
        "profile_score": candidate.get("profile", {}).get("score"),
        "stripe_s0": candidate.get("stripe_s0"),
        "stripe_s1": candidate.get("stripe_s1"),
    }


def build_g0_g1(repo_root: Path, packet_root: Path) -> None:
    base = repo_root / "scratch/court_det_fix/evidence/g0_g1/evaluation_20260922"
    summary = read_json_gz(base / "comparison_summary.json.gz")
    populations: dict[tuple[str, str, str], dict[str, Any]] = {}
    population_sources: dict[str, str] = {}
    for case_path in sorted((base / "populations").glob("*.json.gz")):
        payload = read_json_gz(case_path)
        case_id = payload["case_id"]
        population_sources[case_id] = str(case_path.relative_to(repo_root)).replace(os.sep, "/")
        for source in ("G0", "G1"):
            for candidate in payload[source]:
                populations[(case_id, source, candidate["candidate_id"])] = candidate

    output_cases: list[dict[str, Any]] = []
    table_lines = [
        "# G0/G1 crossed evaluation",
        "",
        "The table preserves all 27 cases and all six crossed cells. `nearest_camera_px` is the closest camera-eligible proposal; line and paint distances are the selected winners. Distances are working-resolution diagnostic values, not visual acceptance labels.",
        "",
        "| Case | Pool | Scorer | Candidates | Camera-eligible | Nearest camera px | Line winner (distance; score) | Paint winner (distance; score) |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for case in summary["cases"]:
        case_id = case["case_id"]
        case_output = {
            "case_id": case_id,
            "label": next(row["label"] for row in summary["case_table"] if row["case_id"] == case_id),
            "source_frame": CASE_FRAME_PATHS[case_id],
            "population_source": population_sources.get(case_id),
            "cells": {},
        }
        for cell_name, cell in case["cells"].items():
            line = cell.get("line") or {}
            paint = cell.get("paint") or {}
            pool = cell_name.split(",", 1)[0]
            line_candidate = selected_population_candidate(populations, line.get("origin_key"), case_id)
            paint_candidate = selected_population_candidate(populations, paint.get("origin_key"), case_id)
            output_cell = {
                "candidate_count": cell.get("candidate_count"),
                "eligible_count": cell.get("eligible_count"),
                "camera_eligible_count": cell.get("camera_eligible_count"),
                "nearest_all_distance_working_px": cell.get("nearest_all_distance_working_px"),
                "nearest_camera_distance_working_px": cell.get("nearest_camera_distance_working_px"),
                "line": {**line, "geometry": line_candidate},
                "paint": {**paint, "geometry": paint_candidate},
            }
            case_output["cells"][cell_name] = output_cell
            line_summary = f"`{line.get('origin_key', '')}` ({line.get('distance_working_px', '')}; {line.get('stripe_exclusive_score', '')})"
            paint_summary = f"`{paint.get('origin_key', '')}` ({paint.get('distance_working_px', '')}; {paint.get('profile_score', '')})"
            table_lines.append(
                f"| {case_id} | {pool} | {cell_name.split(',', 1)[1]} | {cell.get('candidate_count', '')} | {cell.get('camera_eligible_count', '')} | {cell.get('nearest_camera_distance_working_px', '')} | {line_summary} | {paint_summary} |"
            )
        output_cases.append(case_output)

    table_lines.extend(
        [
            "",
            "S1 is absent for `shuttleset_03_scene_0029`; its missing cell is preserved as an unavailable stage in the source summary rather than imputed.",
            "The saved crossed summary records nearest-proposal distances but does not retain a nearest-origin ID for every cell. The published geometry is therefore complete for every selected line/paint winner, while nearest-only identity is selectively unavailable.",
            "",
            "Source frame links are canonical frozen inputs:",
            "",
        ]
    )
    for case_id in [case["case_id"] for case in summary["cases"]]:
        table_lines.append(f"- [{case_id}]({frame_link(repo_root, packet_root, case_id)})")
    write_text(packet_root / "g0_g1.md", "\n".join(table_lines))
    write_json_gz(
        packet_root / "g0_g1/results.json.gz",
        {
            "schema": "court-detector-review-g0-g1/1",
            "scope": summary["scope"],
            "cases": output_cases,
            "source_summary": repo_link(repo_root, packet_root, str((base / "comparison_summary.json.gz").relative_to(repo_root))),
            "source_ledger": repo_link(repo_root, packet_root, "scratch/court_det_fix/evidence/g0_g1/cases.csv.gz"),
        },
    )

    gallery_lines = [
        "# G0/G1 selected-origin previews",
        "",
        "These 122 native-resolution JPEGs are quality-85 lossy previews of the saved full overlays. The exact source-qualified origin and roles are retained below; raw candidate geometry is in [results.json.gz](results.json.gz).",
        "",
        "| Case | Origin | Roles | Preview | Source PNG |",
        "| --- | --- | --- | --- | --- |",
    ]
    gallery_index = base / "gallery/index.md"
    for line in gallery_index.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith(("| case ", "| ---")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        case_id, origin, roles, _, full_link, _ = cells
        source_name_match = re.search(r"\(([^)]+__full\.png)\)", full_link)
        if source_name_match is None:
            raise ValueError(f"No full overlay in gallery row: {line}")
        source_name = source_name_match.group(1)
        source = base / "gallery" / source_name
        destination_name = f"{source.stem}.jpg"
        destination = packet_root / "g0_g1/overlays" / destination_name
        convert_preview(source, destination, quality=85)
        preview = f"overlays/{destination_name}"
        source_relative = str(source.relative_to(repo_root)).replace(os.sep, "/")
        gallery_lines.append(f"| {case_id} | `{origin}` | {roles} | [{preview}]({preview}) | `{source_relative}` |")
    write_text(packet_root / "g0_g1/gallery.md", "\n".join(gallery_lines))


def parse_mask_table(source: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    stage = ""
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Stage "):
            stage = line.removeprefix("## Stage ")
        if not line.startswith("| ") or line.startswith(("| View ", "| ---")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 4:
            rows.append([stage, *cells])
    return rows


def build_person_masks(repo_root: Path, packet_root: Path) -> None:
    base = repo_root / "scratch/court_det_fix/evidence/holistic_admission/box_repair/evaluation_20260922"
    run_dir = base / "runs/person_observations_repair_20260922/matcher"
    source_rows = parse_mask_table(run_dir / "comparison.md")
    image_links: dict[str, str] = {}
    image_sources: dict[str, Path] = {}
    for source in sorted((base / "gallery").rglob("*.png")):
        destination_name = f"{source.stem}.jpg"
        destination = packet_root / "person_masks/overlays" / destination_name
        convert_preview(source, destination, quality=90)
        image_links[source.name] = f"overlays/{destination_name}"
        image_sources[source.name] = source
    lines = [
        "# Corrected person-mask comparison",
        "",
        (
            "All five cases completed. Values are maximum corner distances to the frozen control in working pixels. "
            "Nearest means the pooled proposal before the global cap, not necessarily a retained or camera-eligible court. "
            "The `results` stage uses ordinary retention; `all_camera` is the separate uncapped-camera diagnostic. "
            "The JPEGs are quality-90 lossy previews of the saved line, paint and nearest diagnostics."
        ),
        "",
        "| Stage | View | Control | Baseline nearest / line / paint | Corrected-mask nearest / line / paint | Raw frame |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows: list[dict[str, Any]] = []
    for stage, view, control, baseline, repaired in source_rows:
        case_id_map = {
            "GX5": "gxBQ_window_00_frame_5",
            "SS03-16": "shuttleset_03_scene_0016",
            "SS03-17": "shuttleset_03_scene_0017",
            "SS03-19": "shuttleset_03_scene_0019",
            "SS21-20": "shuttleset_21_scene_0020",
        }
        case_id = case_id_map[view]
        lines.append(f"| {stage} | {view} | {control} | {baseline} | {repaired} | [{case_id}]({frame_link(repo_root, packet_root, case_id)}) |")
        rows.append({"stage": stage, "view": view, "control": control, "baseline": baseline, "corrected": repaired, "case_id": case_id})
    lines.extend(
        [
            "",
            "## Saved diagnostic previews",
            "",
            "| Preview | Source PNG |",
            "| --- | --- |",
        ]
    )
    for source_name, preview in sorted(image_links.items()):
        preview_link = f"person_masks/{preview}"
        source_relative = str(image_sources[source_name].relative_to(repo_root)).replace(os.sep, "/")
        lines.append(f"| [{preview_link}]({preview_link}) | `{source_relative}` |")
    lines.extend(
        [
            "",
            "The full saved stage records and the original comparison remain at [comparison.md](../holistic_admission/box_repair/evaluation_20260922/runs/person_observations_repair_20260922/matcher/comparison.md) and [diagnosis.json.gz](../holistic_admission/box_repair/evaluation_20260922/runs/person_observations_repair_20260922/matcher/diagnosis.json.gz).",
        ]
    )
    write_text(packet_root / "person_masks.md", "\n".join(lines))
    write_json_gz(
        packet_root / "person_masks/results.json.gz",
        {
            "schema": "court-detector-review-person-masks/1",
            "rows": rows,
            "previews": image_links,
            "source_comparison": repo_link(repo_root, packet_root, str((run_dir / "comparison.md").relative_to(repo_root))),
            "source_diagnosis": repo_link(repo_root, packet_root, str((run_dir / "diagnosis.json.gz").relative_to(repo_root))),
        },
    )


def temporal_candidate_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["court_id"]: candidate for candidate in payload["candidates"]}


def temporal_selection_rows(selection: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for access, cases in (
        ("native_per_frame", selection["native_per_frame"]),
        ("common_union_per_frame", selection["common_union_per_frame"]),
        ("target_eligible_per_frame", selection["target_eligible_per_frame"]),
    ):
        for case_id, result in cases.items():
            for criterion in ("line", "paint"):
                choice = result.get(criterion)
                if choice is None:
                    continue
                candidate = candidates.get(choice["court_id"])
                rows.append(
                    {
                        "access": access,
                        "case_id": case_id,
                        "criterion": criterion,
                        "choice": choice,
                        "homography_anchor_working": candidate.get("homography_anchor_working") if candidate else None,
                    }
                )
    shared = selection["common_union_temporal"]
    for criterion, key in (("line", "shared_line"), ("paint", "shared_paint")):
        choice = shared[key]
        candidate = candidates.get(choice["court_id"])
        rows.append(
            {
                "access": "common_union_temporal",
                "case_id": "shared",
                "criterion": criterion,
                "choice": choice,
                "homography_anchor_working": candidate.get("homography_anchor_working") if candidate else None,
            }
        )
    return rows


def build_temporal(repo_root: Path, packet_root: Path) -> None:
    base = repo_root / "scratch/court_det_fix/evidence/pixel_temporal/evaluation_20260922/results"
    rows: list[dict[str, Any]] = []
    source_map: dict[str, str] = {}
    for cohort in ("gx", "am3"):
        selection_path = base / cohort / "selection.json"
        candidate_path = base / cohort / "candidates.json.gz"
        selection = read_json(selection_path)
        candidates = temporal_candidate_map(read_json_gz(candidate_path))
        cohort_rows = temporal_selection_rows(selection, candidates)
        for row in cohort_rows:
            row["cohort"] = cohort
            rows.append(row)
        source_map[f"{cohort}_selection"] = repo_link(repo_root, packet_root, str(selection_path.relative_to(repo_root)))
        source_map[f"{cohort}_candidates"] = repo_link(repo_root, packet_root, str(candidate_path.relative_to(repo_root)))
        source_map[f"{cohort}_manifest"] = repo_link(repo_root, packet_root, str((base / cohort / "manifest.json").relative_to(repo_root)))

    for source in sorted((base / "am3/overlays").glob("*__full.png")):
        destination_name = f"{source.stem}.jpg"
        destination = packet_root / "temporal/am3_overlays" / destination_name
        convert_preview(source, destination, quality=95)
        source_map[f"am3_overlay:{source.name}"] = {
            "source": repo_link(repo_root, packet_root, str(source.relative_to(repo_root))),
            "preview": f"am3_overlays/{destination_name}",
        }

    lines = [
        "# Temporal union evaluation",
        "",
        "This table preserves native per-frame, common-union per-frame and shared-median choices for GX and Am3. Exact candidate IDs, scores and anchor homographies are in [temporal/results.json.gz](temporal/results.json.gz). Am3 full-frame previews are quality-95 lossy JPEGs; canonical GX review sheets and far-end crops remain linked raw evidence.",
        "",
        "| Cohort | Access | Target | Criterion | Origin case | Arm | Candidate | Line score | Paint score | Preview |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        choice = row["choice"]
        court_id = choice["court_id"]
        source_case, _, source_arm_candidate = court_id.partition("::")
        origin_arm, _, origin_candidate = source_arm_candidate.partition("::")
        preview = ""
        if row["cohort"] == "am3" and row["case_id"] != "shared":
            target = row["case_id"]
            prefix = f"{target}__{source_case}__{origin_arm}__{origin_candidate.replace(':', '_')}__full.jpg"
            if (packet_root / "temporal/am3_overlays" / prefix).is_file():
                preview = f"[temporal/am3_overlays/{prefix}](temporal/am3_overlays/{prefix})"
        lines.append(
            f"| {row['cohort']} | {row['access']} | {row['case_id']} | {row['criterion']} | {source_case} | {origin_arm} | `{origin_candidate}` | {choice.get('line_score', choice.get('median_line_score', ''))} | {choice.get('paint_profile_score', choice.get('median_paint_profile_score', ''))} | {preview} |"
        )
    lines.extend(
        [
            "",
            (
                "References were not used before selection. The complete saved score matrices and candidate populations "
                "are published, not just the winners. See the raw source links below."
            ),
        ]
    )
    lines.extend(["", "## GX full views and far-end detail", "",
                  "| Target | Six selected overlays | Raw and paint-overlay far-end crops |",
                  "| --- | --- | --- |"])
    for case_id in CASE_FRAME_PATHS:
        if not case_id.startswith("gxBQ_"):
            continue
        image_base = f"../pixel_temporal/evaluation_20260922/results/gx/review_sheets/{case_id}"
        lines.append(f"| {case_id} | [Full views]({image_base}.jpg) | [Far end]({image_base}__far_end.png) |")
    lines.extend(["", "## Complete numerical records", ""])
    for cohort in ("gx", "am3"):
        source_base = f"../pixel_temporal/evaluation_20260922/results/{cohort}"
        lines.append(
            f"- {cohort}: [selections]({source_base}/selection.json), "
            f"[all candidate geometry]({source_base}/candidates.json.gz), "
            f"[all per-frame scores]({source_base}/scores/), "
            f"[registration and run settings]({source_base}/manifest.json)"
        )
    write_text(packet_root / "temporal.md", "\n".join(lines))
    write_json_gz(
        packet_root / "temporal/results.json.gz",
        {
            "schema": "court-detector-review-temporal/1",
            "rows": rows,
            "sources": source_map,
        },
    )


def build_readme(repo_root: Path, packet_root: Path) -> None:
    text = """# Court-detector evaluation review packet (22 September 2026)

This is a bounded GitHub-readable evidence packet for the completed development evaluation. It keeps readable case tables, exact selected geometry in small `.json.gz` records, canonical raw-frame links, and lossy visual previews. The JPEGs are review previews, not replacement source data.

## Read in this order

- [W5 whole-court selections](w5.md): all 30 distinct final arm winners, saved gate/source ablations, failure alternatives and full-frame rulings.
- [G0/G1 crossed comparison](g0_g1.md): all 27 cases, six crossed pool/scorer cells, nearest-versus-selected distances and all 122 source-qualified selected-origin previews.
- [Corrected person masks](person_masks.md): all five stage comparisons and the saved diagnostic gallery.
- [Temporal union evaluation](temporal.md): native/common/shared choices for GX and Am3, exact scores and anchor homographies.

## Raw inputs and source records

The 27 unannotated detector input images are linked in [g0_g1.md](g0_g1.md) and live under `scratch/court_det_fix/frozen_views/frames/`: 10 broadcast scene medians and 17 single-frame views. The packet does not duplicate them.

For fitting accuracy, use the committed [original amateur annotations](../../../../data/amateur_court_corners/README.md), [September additions](../../../../data/amateur_court_corners/2026-09-08/README.md), and [GX clicked landmarks](../../../../data/amateur_court_corners/2026-09-09/hand_corners_landmarks.csv). Visible clicks and extrapolated corners have different meanings. Expanded frozen references are descriptive aids, not independently approved truth.

[W5 winner coordinates and homographies](w5/geometry.md) are readable directly on GitHub. Compressed records retain the remaining numerical detail. Post-hoc gate/source-ablation rows retain IDs, scores and overlays; 40 rows lack geometry in the compact source review, so their exact corners are not exported here. The full local case records retain them.

The small packet records link to the saved G0/G1, W5, person-mask and temporal source outputs. Those links are repository-relative and resolve in the published checkout. The multi-gigabyte W5 arrays, case records, rankings and caches, plus the large G0/G1 populations, remain local-only because the tables and selected geometry already expose the outcomes needed for independent review.

The panel is development data, and W5 usability is qualitative. Nearest-only G0/G1 distances do not retain a nearest-origin ID in the saved summary. This packet supports independent review of outcomes; it is not a self-contained rerun of the full detector search.

To rebuild the export, run `build_packet.py` with the complete local source packets present. It copies results and converts existing overlays; it does not run the detector or change the source evidence. JPEG quality is 95 for W5/Am3, 85 for G0/G1 and 90 for person-mask previews. The seven GX far-end sheets remain PNGs.
"""
    write_text(packet_root / "README.md", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    packet_root = Path(__file__).resolve().parent
    repo_root = (args.repo_root or packet_root.parents[3]).resolve()
    builders = (build_w5, build_g0_g1, build_person_masks, build_temporal)
    with ThreadPoolExecutor(max_workers=4) as executor:
        exports = [executor.submit(builder, repo_root, packet_root) for builder in builders]
        for export in exports:
            export.result()
    build_readme(repo_root, packet_root)
    print(f"built {packet_root}")


if __name__ == "__main__":
    main()
