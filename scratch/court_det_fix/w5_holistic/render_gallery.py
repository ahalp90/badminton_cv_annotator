"""Render compact prediction-only W5 overlays and a Markdown gallery index."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def import_verifier(root: Path):
    import sys

    sys.path.insert(0, str(root / "w5_holistic"))
    from verifier import (
        CASE_IDS,
        CASE_LABELS,
        CASE_PACKS,
        PACK_OF,
        frame_path,
        load_source,
        read_json_gz,
    )

    return {
        "CASE_IDS": CASE_IDS,
        "CASE_LABELS": CASE_LABELS,
        "CASE_PACKS": CASE_PACKS,
        "PACK_OF": PACK_OF,
        "frame_path": frame_path,
        "load_source": load_source,
        "read_json_gz": read_json_gz,
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_native_frame(root: Path, source: dict, verifier: dict) -> np.ndarray:
    path = verifier["frame_path"](root, source)
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(path)
    expected = (source["dimensions"]["height"], source["dimensions"]["width"])
    if frame.shape[:2] != expected:
        raise ValueError(f"{source['id']}: {frame.shape[:2]} != {expected}")
    return frame


def draw_reference(canvas: np.ndarray, reference: dict | None) -> np.ndarray:
    if not reference or not reference.get("corners_px"):
        return canvas
    corners = np.rint(np.asarray(reference["corners_px"], dtype=float)).astype(int)
    cv2.polylines(canvas, [corners], True, (255, 0, 255), 2, cv2.LINE_AA)
    for point in corners:
        cv2.circle(canvas, tuple(point), 5, (255, 0, 255), -1, cv2.LINE_AA)
    return canvas


def render_prediction(
    frame: np.ndarray, candidate: dict, context: dict, roles: list[str], output_stem: Path,
    reference: dict | None,
) -> list[str]:
    from experiments.annotator.independent_court import detector, paint_geometry

    canvas = frame.copy()
    native_size = np.asarray([context["dimensions"]["width"], context["dimensions"]["height"]], dtype=float)
    working_size = np.asarray(context["working_dimensions"], dtype=float)
    scale = native_size / working_size
    homography = np.asarray(candidate["homography_working"], dtype=float)
    paint_working, _ = detector.project(homography[None], paint_geometry.CENTRE_SEGMENTS_M)
    paint_native = paint_working[0].reshape(-1, 2, 2) * scale
    corners = np.asarray(candidate["corners_px"], dtype=float)
    cv2.polylines(canvas, [np.rint(corners).astype(int)], True, (255, 255, 255), 3, cv2.LINE_AA)
    for interval_index, segment in enumerate(paint_native):
        colour = (255, 190, 0) if interval_index != 2 and interval_index != 3 else (255, 120, 0)
        cv2.line(canvas, tuple(np.rint(segment[0]).astype(int)), tuple(np.rint(segment[1]).astype(int)), colour, 2,
                 cv2.LINE_AA)
    label = ", ".join(roles)
    cv2.putText(canvas, f"{context['id']} {candidate['origin_key']} [{label}]", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    full_path = output_stem.with_name(output_stem.name + "__full.png")
    if not cv2.imwrite(str(full_path), canvas):
        raise OSError(full_path)
    visible = paint_native.reshape(-1, 2)
    visible = visible[np.isfinite(visible).all(axis=1)]
    if len(visible):
        lower = np.maximum(np.floor(visible.min(axis=0) - 70), 0).astype(int)
        upper = np.minimum(np.ceil(visible.max(axis=0) + 70), native_size - 1).astype(int)
        if upper[0] - lower[0] > 1200:
            midpoint = int((upper[0] + lower[0]) / 2)
            lower[0], upper[0] = max(0, midpoint - 600), min(int(native_size[0] - 1), midpoint + 600)
        if upper[1] - lower[1] > 800:
            midpoint = int((upper[1] + lower[1]) / 2)
            lower[1], upper[1] = max(0, midpoint - 400), min(int(native_size[1] - 1), midpoint + 400)
        crop = canvas[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
    else:
        crop = canvas
    crop_path = output_stem.with_name(output_stem.name + "__crop.png")
    if not cv2.imwrite(str(crop_path), crop):
        raise OSError(crop_path)
    links = [full_path.name, crop_path.name]
    if reference and reference.get("corners_px"):
        reference_canvas = draw_reference(canvas.copy(), reference)
        reference_path = output_stem.with_name(output_stem.name + "__reference.png")
        if not cv2.imwrite(str(reference_path), reference_canvas):
            raise OSError(reference_path)
        links.append(reference_path.name)
    return links


def render_case(root: Path, run_dir: Path, case_id: str, packet: dict, verifier: dict) -> dict:
    source = verifier["load_source"](root, case_id)
    frame = load_native_frame(root, source, verifier)
    context = {
        "id": case_id,
        "dimensions": source["dimensions"],
        "working_dimensions": packet["provenance"]["working_dimensions"],
    }
    reference_pack = verifier["read_json_gz"](root / verifier["CASE_PACKS"][verifier["PACK_OF"][case_id]])
    reference = reference_pack.get("references", {}).get(case_id)
    candidates = packet["review_candidates"]
    selections = {
        "A_paint": candidates.get(packet["A"].get("paint")),
        "B": candidates.get(packet["B"].get("selected_origin_key")),
        "C": candidates.get(packet["C"].get("selected_origin_key")),
    }
    rendered: dict[str, dict] = {}
    by_origin: dict[str, list[str]] = {}
    for role, candidate in selections.items():
        if candidate is None:
            continue
        by_origin.setdefault(candidate["origin_key"], []).append(role)
    for rank_name in ("B", "C"):
        ranking = packet[rank_name]
        rank_keys = ranking["provisional_rank"] or ranking.get("ungated_provisional_rank", [])
        ungated = not ranking["provisional_rank"] and bool(rank_keys)
        for rank_index, origin_key in enumerate(rank_keys[:3]):
            if origin_key in candidates:
                if rank_index == 0 and ungated:
                    role = f"{rank_name}-ungated"
                else:
                    role = rank_name if rank_index == 0 else f"{rank_name}-alternative-{rank_index + 1}"
                by_origin.setdefault(origin_key, []).append(role)
    reference_near = packet.get("reference_near")
    if reference_near and reference_near["origin_key"] in candidates:
        by_origin.setdefault(reference_near["origin_key"], []).append("reference-near")
    for candidate in packet["diagnostic_controls"]:
        by_origin.setdefault(candidate["origin_key"], []).append("control")
    for origin_key, roles in by_origin.items():
        candidate = candidates.get(origin_key)
        if candidate is None:
            candidate = next((item for item in packet["diagnostic_controls"] if item["origin_key"] == origin_key), None)
        if candidate is None:
            continue
        stem = run_dir / "gallery" / safe_name(f"{case_id}__{origin_key}")
        stem.parent.mkdir(parents=True, exist_ok=True)
        links = render_prediction(frame, candidate, context, roles, stem, reference if "control" not in roles else None)
        rendered[origin_key] = {"links": links, "roles": roles}
    return {"case_id": case_id, "label": packet["label"], "rendered": rendered, "selections": selections}


def ranking_position(roles: list[str], prefix: str) -> int:
    positions = []
    for role in roles:
        if role in (prefix, f"{prefix}-ungated"):
            positions.append(0)
        elif role.startswith(f"{prefix}-alternative-"):
            positions.append(int(role.rsplit("-", 1)[-1]))
    return min(positions, default=10_000)


def format_role_links(role_links: dict[str, list[tuple[int, str]]], role: str) -> str:
    return "/".join(link for _, link in sorted(role_links[role])) or "—"


def write_index(run_dir: Path, rendered_cases: list[dict]) -> None:
    lines = [
        "# W5 pilot gallery",
        "",
        "Prediction overlays show the automatic outside boundary and physical finite paint template. Reference overlays are separate files where frozen references exist.",
        "",
        (
            "| view | legacy paint-first | original candidates | original + adjusted candidates | "
            "diagnostic controls | reference-near |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in rendered_cases:
        if case.get("stopped_reason"):
            lines.append(f"| {case['label']} | stopped: {case['stopped_reason']} | — | — | — | — |")
            continue
        links_by_origin = case["rendered"]
        role_links = {role: [] for role in ("A_paint", "B", "C", "control", "reference-near")}
        for origin, rendered in links_by_origin.items():
            links = rendered["links"]
            label = safe_name(origin)
            crop = next((link for link in links if link.endswith("__crop.png")), links[0])
            link = f"[{label}]({crop})"
            roles = rendered["roles"]
            if any(role == "A_paint" for role in roles):
                role_links["A_paint"].append((0, link))
            if any(role == "B" or role.startswith("B-") for role in roles):
                role_links["B"].append((ranking_position(roles, "B"), link))
            if any(role == "C" or role.startswith("C-") for role in roles):
                role_links["C"].append((ranking_position(roles, "C"), link))
            if "control" in roles:
                role_links["control"].append((0, link))
            if "reference-near" in roles:
                role_links["reference-near"].append((0, link))

        lines.append(
            f"| {case['label']} | {format_role_links(role_links, 'A_paint')} | "
            f"{format_role_links(role_links, 'B')} | {format_role_links(role_links, 'C')} | "
            f"{format_role_links(role_links, 'control')} | {format_role_links(role_links, 'reference-near')} |"
        )
    index = run_dir / "gallery/index.md"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run", required=True)
    parser.add_argument("--cases", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    run_dir = root / "w5_holistic/runs" / args.run
    verifier = import_verifier(root)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    packet = {}
    for case_id in args.cases or manifest["cases"]:
        packet_path = run_dir / "case_records" / f"{case_id}.json.gz"
        case_record = verifier["read_json_gz"](packet_path)
        packet[case_id] = case_record
    reviews = json.loads((run_dir / "review_candidates.json").read_text())
    rendered_cases = []
    for case_id, case_packet in packet.items():
        case_packet["review_candidates"] = reviews[case_id]["candidates"]
        case_packet["A"] = reviews[case_id]["A"]
        case_packet["B"] = case_packet["rankings"]["B"]
        case_packet["C"] = case_packet["rankings"]["C"]
        case_packet["label"] = verifier["CASE_LABELS"][case_id]
        case_packet["diagnostic_controls"] = reviews[case_id]["diagnostic_controls"]
        case_packet["reference_near"] = reviews[case_id].get("reference_near")
        rendered_cases.append(render_case(root, run_dir, case_id, case_packet, verifier))
    for stopped in manifest.get("stopped_views", []):
        rendered_cases.append({
            "case_id": stopped["case_id"],
            "label": verifier["CASE_LABELS"].get(stopped["case_id"], stopped["case_id"]),
            "stopped_reason": stopped["reason"],
            "rendered": {},
        })
    write_index(run_dir, rendered_cases)


if __name__ == "__main__":
    main()
