"""Inspect cached DeepLSD segment support at saved net-post bases."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DEFAULT_OUTPUT = REPO / "local_scratch/net_recovery/20260923/base_probe"
sys.path[:0] = [str(ROOT / "colour_consistency"), str(ROOT / "wider_evaluation")]

import am1_net_selection_trial as net  # pyrefly: ignore[missing-import]
from run_cases import load_runtime  # pyrefly: ignore[missing-import]

SAVED_SCAN = ROOT / "net_recovery/saved_net_scan.json.gz"
AM1_SCAN = ROOT / "colour_consistency/am1_net_selection_trial.json.gz"
CONTROL_PACK = ROOT / "wider_evaluation/runs/20260922/control_inputs.json.gz"
INPUT_MANIFEST = ROOT / "wider_evaluation/runs/20260922/manifest.json.gz"
GX_CONTROLS = {"gxBQ_window_00_frame_0", "gxBQ_window_00_frame_5"}
INSPECTION_ANGLE_DEG = 16.0


def source_path(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def piece_samples(piece: np.ndarray, segments: np.ndarray, size: tuple[int, int]) -> dict:
    """Recreate the old 24-sample mask, retaining samples outside the image."""
    start, end = piece
    direction = (end - start) / np.linalg.norm(end - start)
    samples = start + np.linspace(0, 1, net.SAMPLES_PER_PIECE)[:, None] * (end - start)
    sample_x, sample_y = samples.T
    width, height = size
    in_frame = (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
    segment_start, segment_end = segments[:, :2], segments[:, 2:]
    segment_vector = segment_end - segment_start
    segment_length = np.linalg.norm(segment_vector, axis=1)
    segment_direction = segment_vector / segment_length[:, None]
    aligned = np.abs(segment_direction @ direction) >= np.cos(np.radians(net.DIRECTION_TOLERANCE_DEG))
    covered = np.zeros(net.SAMPLES_PER_PIECE, dtype=bool)
    for index, sample in enumerate(samples):
        offset = sample - segment_start
        along = (offset * segment_direction).sum(axis=1)
        within = (along >= -net.EXTENT_MARGIN_WORKING_PX) & (along <= segment_length + net.EXTENT_MARGIN_WORKING_PX)
        perpendicular = np.abs(offset[:, 0] * segment_direction[:, 1] - offset[:, 1] * segment_direction[:, 0])
        covered[index] = bool((aligned & within & (perpendicular <= net.PERPENDICULAR_TOLERANCE_WORKING_PX)).any())
    visible = int(in_frame.sum())
    effective = covered & in_frame
    result = {
        "coordinates_working_px": samples.tolist(),
        "in_frame": in_frame.tolist(),
        "covered_old": covered.tolist(),
        "covered_in_frame": effective.tolist(),
        "visible_count": visible,
        "covered_count": int(effective.sum()),
        "coverage": float(effective.sum() / visible) if visible else None,
    }
    assert (result["coverage"], visible) == net.piece_coverage(piece, segments, size)
    return result


def post_fragments(piece: np.ndarray, segments: np.ndarray, samples: dict) -> list[dict]:
    """Describe every fragment within 16 degrees of the post's base-to-top axis."""
    base, top = piece
    post_vector = top - base
    post_length = np.linalg.norm(post_vector)
    post_direction = post_vector / post_length
    rows = []
    for fragment_id, segment in enumerate(segments):
        first, last = segment[:2], segment[2:]
        vector = last - first
        length = np.linalg.norm(vector)
        fragment_direction = vector / length
        angle = float(np.degrees(np.arccos(np.clip(abs(fragment_direction @ post_direction), 0, 1))))
        if angle > INSPECTION_ANGLE_DEG:
            continue
        endpoints = np.stack((first, last))
        along = ((endpoints - base) @ post_direction) / post_length
        endpoint_distances = np.linalg.norm(endpoints - base, axis=1)
        nearest_index = int(endpoint_distances.argmin())
        base_offset = base - first
        perpendicular = float(abs(base_offset[0] * fragment_direction[1] - base_offset[1] * fragment_direction[0]))
        old_eligible = bool(abs(fragment_direction @ post_direction) >= np.cos(np.radians(net.DIRECTION_TOLERANCE_DEG)))
        covered_indices = []
        if old_eligible:
            for index, (sample, in_frame) in enumerate(zip(samples["coordinates_working_px"], samples["in_frame"])):
                if not in_frame:
                    continue
                offset = np.asarray(sample) - first
                extent = float(offset @ fragment_direction)
                distance = float(abs(offset[0] * fragment_direction[1] - offset[1] * fragment_direction[0]))
                if (-net.EXTENT_MARGIN_WORKING_PX <= extent <= length + net.EXTENT_MARGIN_WORKING_PX
                        and distance <= net.PERPENDICULAR_TOLERANCE_WORKING_PX):
                    covered_indices.append(index)
        rows.append({
            "fragment_id": fragment_id,
            "angle_mismatch_deg": angle,
            "old_8deg_eligible": old_eligible,
            "along_post_interval_base_to_top_lengths": [float(along.min()), float(along.max())],
            "base_to_infinite_line_working_px": perpendicular,
            "nearest_endpoint_distance_working_px": float(endpoint_distances[nearest_index]),
            "nearest_endpoint_working_px": endpoints[nearest_index].tolist(),
            "nearest_endpoint_index": nearest_index,
            "covered_sample_indices_old": covered_indices,
        })
    return rows


def inspect_choice(choice: dict, segments: np.ndarray, size: tuple[int, int]) -> dict:
    net_row = choice["net"]
    if net_row["state"] != "measured":
        raise ValueError(f"{choice['origin_key']}: selected net projection failed")
    pieces = {}
    for name, coordinates in zip(net.PIECE_NAMES, net_row["pieces_working_px"]):
        piece = np.asarray(coordinates, dtype=float)
        samples = piece_samples(piece, segments, size)
        assert samples["coverage"] == net_row["coverage"][name], (choice["origin_key"], name, "coverage")
        assert samples["visible_count"] == net_row["visible_samples"][name], (choice["origin_key"], name, "visible")
        piece_row = {"samples": samples}
        if name.startswith("post_"):
            fragments = post_fragments(piece, segments, samples)
            covered_by_fragments = set()
            for fragment in fragments:
                covered_by_fragments.update(fragment["covered_sample_indices_old"])
            assert covered_by_fragments == {
                index for index, covered in enumerate(samples["covered_in_frame"]) if covered
            }, (choice["origin_key"], name, "fragment union")
            lower_visible = sum(samples["in_frame"][:6])
            lower_covered = sum(samples["covered_in_frame"][:6])
            eligible_near_base = [fragment for fragment in fragments if (
                fragment["old_8deg_eligible"]
                and fragment["base_to_infinite_line_working_px"] <= net.PERPENDICULAR_TOLERANCE_WORKING_PX
                and fragment["along_post_interval_base_to_top_lengths"][1] > 0
            )]
            nearest = min(eligible_near_base, key=lambda fragment: fragment["nearest_endpoint_distance_working_px"],
                          default=None)
            piece_row.update({
                "fragments_within_16deg": fragments,
                "base_quarter_first_6": {
                    "visible_count": lower_visible,
                    "covered_count": lower_covered,
                    "coverage": lower_covered / lower_visible if lower_visible else None,
                },
                "base_point_first_sample": {
                    "in_frame": samples["in_frame"][0],
                    "covered": samples["covered_in_frame"][0],
                },
                "nearest_old_aligned_endpoint_within_4px_base_line_and_upward_extent": None if nearest is None else {
                    "fragment_id": nearest["fragment_id"],
                    "distance_working_px": nearest["nearest_endpoint_distance_working_px"],
                },
            })
        pieces[name] = piece_row
    return {
        "origin_key": choice["origin_key"],
        "candidate_id": choice["candidate_id"],
        "source": choice["source"],
        "native_size_wh": choice["native_size_wh"],
        "working_size_wh": choice["working_size_wh"],
        "corners_native_px": choice["corners_native_px"],
        "corners_working_px": choice["corners_working_px"],
        "homography_working": choice["homography_working"],
        "paint_score": choice.get("paint_score"),
        "old_coverage": net_row["coverage"],
        "old_visible_samples": net_row["visible_samples"],
        "pieces_native_px": net_row["pieces_native_px"],
        "pieces_working_px": net_row["pieces_working_px"],
        "pieces": pieces,
    }


def markdown_table(cases: list[dict]) -> str:
    lines = [
        "# Predicted net-post base fragment support",
        "",
        "L/R: full-post / first 6/24 / first sample / nearest endpoint (working px).",
        "Qualifying endpoints belong to old 8° aligned fragments within 4 px of the base's infinite line",
        "that extend upward along the post. A dash means no in-frame samples or qualifying fragment.",
        "",
        "| Case | Role | Left: full / lower / base / endpoint px | Right: full / lower / base / endpoint px |",
        "|---|---|---|---|",
    ]
    for case in cases:
        for role in ("baseline", "trial"):
            choice = case["choices"][case["roles"][role]]
            cells = []
            for name in ("post_left", "post_right"):
                piece = choice["pieces"][name]
                full = piece["samples"]["coverage"]
                lower = piece["base_quarter_first_6"]["coverage"]
                base = piece["base_point_first_sample"]
                endpoint = piece["nearest_old_aligned_endpoint_within_4px_base_line_and_upward_extent"]
                values = ["–" if value is None else f"{value:.2f}" for value in (full, lower)]
                values.append("–" if not base["in_frame"] else ("Y" if base["covered"] else "N"))
                values.append("–" if endpoint is None else f"{endpoint['distance_working_px']:.1f}")
                cells.append(" / ".join(values))
            lines.append(f"| {case['case_id']} | {role} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


def main(output_dir: Path) -> None:
    cv2.setNumThreads(1)
    saved = read_json_gz(SAVED_SCAN)
    am1 = read_json_gz(AM1_SCAN)
    if not saved["complete"] or saved["changed_selection_count"] != 13:
        raise ValueError("Saved scan is incomplete or does not contain the expected 13 changes")
    selected = [case for case in saved["cases"] if case["changed"] or case["case_id"] in GX_CONTROLS]
    if len(selected) != 15 or not GX_CONTROLS.issubset({case["case_id"] for case in selected}):
        raise ValueError("Expected 13 changed cases plus GX0 and GX5")
    seeded = [case for case in am1["cases"] if case["label"] == net.POOL_LABEL]
    if len(seeded) != 1:
        raise ValueError("Expected one accepted Am1 seeded case")
    selected.extend(seeded)
    manifest = {row["case_id"]: row for row in read_json_gz(INPUT_MANIFEST)["cases"]}
    _, verifier, _ = load_runtime(ROOT, CONTROL_PACK)
    results = []
    for case in selected:
        case_id = case["case_id"]
        context = verifier.prepare_view(ROOT, case_id)
        if case["frame_path"] != context.frame_relative_path:
            raise ValueError(f"{case_id}: prepared frame differs from scan")
        if case["native_size_wh"] != list(context.native_size) or case["working_size_wh"] != list(context.size):
            raise ValueError(f"{case_id}: prepared dimensions differ from scan")
        record_path = REPO / case["source_record"]
        record = read_json_gz(record_path)
        if record["case_id"] != case_id or record["provenance"]["frame_path"] != context.frame_relative_path:
            raise ValueError(f"{case_id}: source record provenance differs")
        if record["provenance"]["native_dimensions"] != list(context.native_size):
            raise ValueError(f"{case_id}: source record native dimensions differ")
        if record["provenance"]["working_dimensions"] != list(context.size):
            raise ValueError(f"{case_id}: source record working dimensions differ")
        if case.get("provenance"):
            for field, value in case["provenance"].items():
                if record["provenance"][field] != value:
                    raise ValueError(f"{case_id}: source record {field} differs from scan")
        frame_md5 = hashlib.md5((ROOT / context.frame_relative_path).read_bytes()).hexdigest()
        manifest_row = manifest[case_id]
        if manifest_row["image"] != context.frame_relative_path:
            raise ValueError(f"{case_id}: manifest frame path differs from prepared view")
        manifest_md5 = manifest_row["image_md5"]
        if frame_md5 != manifest_md5:
            raise ValueError(f"{case_id}: local frame MD5 differs from saved manifest")
        candidates = {candidate["origin_key"]: candidate for candidate in record["parents"] + record["valid_children"]}
        paint_criterion = record["rankings"]["C"]["r2_criterion"]
        choices = {}
        roles = {}
        for role in ("baseline", "trial"):
            choice = case[role]
            if choice is None:
                raise ValueError(f"{case_id}: missing {role} selection")
            key = choice["origin_key"]
            roles[role] = key
            if key not in choices:
                old_paint_score = candidates[key]["evidence"][paint_criterion]
                if choice.get("paint_score") is not None and choice["paint_score"] != old_paint_score:
                    raise ValueError(f"{case_id}: {key} saved paint score differs from source record")
                measured_choice = {**choice, "paint_score": old_paint_score}
                choices[key] = inspect_choice(measured_choice, context.segments, context.size)
        results.append({
            "case_id": case_id,
            "scan_label": case["label"],
            "source_scan": source_path(AM1_SCAN if case is seeded[0] else SAVED_SCAN),
            "source_record": source_path(record_path),
            "frame_path": context.frame_relative_path,
            "frame_md5": frame_md5,
            "manifest_frame_md5": manifest_md5,
            "native_size_wh": list(context.native_size),
            "working_size_wh": list(context.size),
            "segments_working_px": context.segments.tolist(),
            "roles": roles,
            "choices": choices,
        })
        print(f"{case_id}: {roles['baseline']} -> {roles['trial']}", flush=True)
    output = {
        "schema": "net-base-fragment-probe/1",
        "sources": {"saved_scan": source_path(SAVED_SCAN), "am1_scan": source_path(AM1_SCAN),
                    "control_pack": source_path(CONTROL_PACK), "input_manifest": source_path(INPUT_MANIFEST)},
        "measurement": {"samples_per_piece": net.SAMPLES_PER_PIECE,
                        "old_perpendicular_px": net.PERPENDICULAR_TOLERANCE_WORKING_PX,
                        "old_angle_deg": net.DIRECTION_TOLERANCE_DEG,
                        "old_extent_margin_px": net.EXTENT_MARGIN_WORKING_PX,
                        "inspection_angle_deg": INSPECTION_ANGLE_DEG},
        "cases": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_dir / "base_probe.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(output, stream, allow_nan=False)
    (output_dir / "summary.md").write_text(markdown_table(results), encoding="utf-8")
    print(f"wrote {len(results)} cases to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output_dir)
