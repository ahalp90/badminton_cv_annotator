"""Render paired legacy and physical paint-refit boundary diagnostics."""

from __future__ import annotations

import argparse
import gzip
import html
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .render_stripe_overlays import draw_lines

SIZE = (1280, 720)
HEADER_HEIGHT = 110
FIT_COLOUR = "#cc00cc"
REFERENCE_COLOUR = "#0072b2"
FONT = ImageFont.truetype("DejaVuSans.ttf", 22)
SMALL_FONT = ImageFont.truetype("DejaVuSans.ttf", 16)
MODELS = (("legacy", "Legacy paint / stripe ranking"), ("physical", "Physical paint / stripe ranking"))


def read_gzip_json(path: Path) -> dict:
    """Read one gzip-compressed JSON document."""
    return json.loads(gzip.decompress(path.read_bytes()))


def read_references(path: Path) -> dict:
    """Read manual references from the frozen marking replay archive."""
    with ZipFile(path) as archive:
        return json.loads(gzip.decompress(archive.read("marking_inputs.json.gz")))["references"]


def outside_polygon(corners_px: list[list[float]], dimensions: dict[str, int]) -> np.ndarray:
    """Scale native four-corner coordinates into the 1280x720 display."""
    corners = np.asarray(corners_px, dtype=float)
    scale = np.asarray(SIZE, dtype=float) / np.array([dimensions["width"], dimensions["height"]])
    corners = corners * scale
    return np.stack((corners, np.roll(corners, -1, axis=0)), axis=1)


def metric_text(metrics: dict, name: str) -> str:
    """Format one saved boundary metric, preserving missing values as n/a."""
    value = metrics.get(name)
    return "n/a" if value is None else f"{value:.2f}px"


def pick_parent_id(entry: dict) -> str:
    """Return the saved parent ID, deriving it from compact pick rows when needed."""
    if "parent_id" in entry:
        return entry["parent_id"]
    return entry["id"].rsplit("/", 2)[0]


def draw_panel(
    raw: Image.Image,
    case_id: str,
    model_title: str,
    entry: dict | None,
    reference_lines: np.ndarray,
    dimensions: dict[str, int],
) -> Image.Image:
    """Draw one labelled panel over an existing 1280x720 frame."""
    panel = Image.new("RGB", (SIZE[0], SIZE[1] + HEADER_HEIGHT), "white")
    panel.paste(raw.copy(), (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(panel)
    draw.text((16, 9), f"{case_id} · {model_title}", fill="black", font=FONT)
    if entry is None:
        draw.text((16, 43), "No eligible court", fill="black", font=SMALL_FONT)
        draw.text((16, 69), "Boundary landmark RMS: n/a · Clicked corner RMS: n/a", fill="black", font=SMALL_FONT)
    else:
        draw.text((16, 41), f"Parent: {pick_parent_id(entry)} · Pick: {entry['id']}", fill="black", font=SMALL_FONT)
        metrics = entry.get("boundary_metrics", {})
        boundary = metric_text(metrics, "boundary_landmark_rms_px")
        clicked = metric_text(metrics, "clicked_corner_rms_px")
        draw.text((16, 67), f"Boundary landmark RMS: {boundary} · Clicked corner RMS: {clicked}",
                  fill="black", font=SMALL_FONT)

    clip_rect = (0, 0, SIZE[0], SIZE[1])
    visible_frame = panel.crop((0, HEADER_HEIGHT, SIZE[0], HEADER_HEIGHT + SIZE[1]))
    draw_lines(visible_frame, reference_lines, REFERENCE_COLOUR, dashed=True, clip_rect=clip_rect)
    if entry is not None:
        predicted_lines = outside_polygon(entry["corners_px"], dimensions)
        draw_lines(visible_frame, predicted_lines, FIT_COLOUR, dashed=False, clip_rect=clip_rect)
    panel.paste(visible_frame, (0, HEADER_HEIGHT))
    return panel


def render_case(
    record: dict, pick_record: dict, reference: dict, frames: Path, output: Path,
) -> None:
    """Render one case's legacy and physical panels."""
    case_id = record["id"]
    dimensions = record["dimensions"]
    frame_path = frames / f"{case_id}__frame.jpg"
    with Image.open(frame_path) as image:
        raw = image.convert("RGB")
    if raw.size != SIZE:
        raise ValueError(f"Expected 1280x720 frame for {case_id}, got {raw.size}")
    reference_lines = outside_polygon(reference["corners_px"], dimensions)
    panels = [
        draw_panel(raw, case_id, title, pick_record[model], reference_lines, dimensions)
        for model, title in MODELS
    ]
    comparison = Image.new("RGB", (SIZE[0] * len(panels), SIZE[1] + HEADER_HEIGHT), "white")
    for index, panel in enumerate(panels):
        comparison.paste(panel, (index * SIZE[0], 0))
    comparison.save(output / f"{case_id}.jpg", quality=94)


def page_intro() -> str:
    """Return the visible-boundary diagnostic caption."""
    return (
        "Visible boundary diagnostic: magenta is the saved stripe-ranking fit's outside-corner polygon; "
        "blue dashed lines are the manual reference corners. Reference corner clicks use different "
        "conventions across the annotation sets, and extrapolated corners are historical guides rather "
        "than exact outside labels. Both panels use stripe-only ranking."
    )


def render_index(case_ids: list[str], output: Path) -> None:
    """Write a standalone HTML index with relative image links."""
    intro = html.escape(page_intro())
    sections = []
    for case_id in case_ids:
        escaped_id = html.escape(case_id)
        sections.append(
            f'<section id="{escaped_id}"><h2>{escaped_id}</h2>'
            f'<img src="{escaped_id}.jpg" alt="Paired legacy and physical boundary panels for {escaped_id}">'
            f'<p><a href="../stripe_overlays/{escaped_id}__comparison.jpg">Historical stripe gallery '
            "(different ranking)</a></p></section>"
        )
    document = (
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Paint-refit boundary diagnostics</title>"
        "<style>body{font:16px system-ui;margin:24px;background:#fafafa;color:#222}"
        "header{max-width:1280px}section{margin:36px 0}img{display:block;max-width:100%;height:auto}"
        "</style><header><h1>Paint-refit boundary diagnostics</h1>"
        f"<p>{intro}</p></header>{''.join(sections)}</html>"
    )
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="gzip JSON from run_paint_refit")
    parser.add_argument("--frames", type=Path, required=True, help="directory containing {id}__frame.jpg files")
    parser.add_argument("--references", type=Path, required=True, help="marking_refit_replay.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = read_gzip_json(args.results)
    references = read_references(args.references)
    records = {record["id"]: record for record in results["records"]}
    picks = {pick["id"]: pick for pick in results["picks"]}
    if len(picks) != 20 or set(picks) != set(records):
        raise ValueError("Expected exactly twenty matching paint-refit case records")
    args.output.mkdir(parents=True, exist_ok=True)
    case_ids = list(picks)
    for case_id in case_ids:
        render_case(records[case_id], picks[case_id]["picks"], references[case_id], args.frames, args.output)
    render_index(case_ids, args.output)


if __name__ == "__main__":
    main()
