#!/usr/bin/env python3
"""Draw CourtKeyNet's court quad + gate signals onto chosen video frames.

For a video and a set of frame indices, runs the CourtKeyNet detector on each
frame and saves one annotated PNG per frame so gate passes and failures can be
checked by eye. Each PNG shows:

- the detected TL->TR->BR->BL quad as a closed orange polygon (thickness scaled
  to the frame resolution) with a filled dot on each corner
- a top-left panel: one line per corner (peak + entropy) and a verdict line,
  ``gate PASS`` or ``gate FAIL: low_peak|bad_area``, white on a dark translucent
  backing so it stays legible over bright court frames

Frame selection is either an explicit comma list (``--frames``) or a column of a
CSV (``--frames-csv`` + ``--frame-col``), so a filtered eval CSV can drive it.
Outputs are named ``<video_stem>_f<frame:07d>_<resize_mode>.png``. If pngquant is
on PATH each PNG is shrunk in place at quality 40-60; without it the plain PNG is
kept (larger).

Usage::

    python src/courtkeynet/validation_scripts/render_court_overlay.py \\
        --video local_scratch/.../pilot_288p.mp4 \\
        --frames "55898,58444" \\
        --out-dir /tmp/court_overlays \\
        --device cpu --resize-mode pad
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# This file lives at src/courtkeynet/validation_scripts/<this>, so the repo root
# (which must be importable for the `src.` package path below) is three up.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.courtkeynet.wrapper import CornerDetection, CourtKeyNetDetector  # noqa: E402

# Orange from the Wong colourblind-safe palette (#D55E00); the quad reads clearly
# for protan/deutan vision against green court and white lines. Panel text white.
QUAD_COLOUR = (213, 94, 0, 255)
TEXT_COLOUR = (255, 255, 255, 255)
PANEL_FILL = (0, 0, 0, 190)

CORNER_LABELS = ("TL", "TR", "BR", "BL")

# DejaVu ships with the base Linux font packages; fall back to Pillow's scalable
# default if it is absent so the >=16 px panel floor still holds.
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """:return: a TrueType font at ``size`` px, DejaVu if present else Pillow's default.

    :param size: pixel height for the font
    """
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default(size=size)


def parse_frame_indices(args: argparse.Namespace) -> list[int]:
    """Resolve the requested frame indices from either CLI mode.

    ``--frames`` keeps the caller's order (dupes dropped); ``--frames-csv`` reads
    the named column as sorted unique ints so a filtered eval CSV drives it.

    :param args: parsed CLI namespace carrying ``frames`` or ``frames_csv``
    :return: frame indices to render
    """
    if args.frames is not None:
        # dict.fromkeys drops duplicates while keeping first-seen order.
        return list(dict.fromkeys(int(tok) for tok in args.frames.split(",") if tok.strip()))
    df = pd.read_csv(args.frames_csv)
    if args.frame_col not in df.columns:
        sys.exit(f"ERROR: column {args.frame_col!r} not in {args.frames_csv} (has {list(df.columns)})")
    return sorted({int(v) for v in df[args.frame_col].dropna()})


def read_frame(cap: cv2.VideoCapture, frame_idx: int, video: Path) -> np.ndarray:
    """Seek to and decode one frame; fail loud on an index past the video end.

    :param cap: an open VideoCapture for ``video``
    :param frame_idx: zero-based frame index to fetch
    :param video: source path, for the error message only
    :return: (H, W, 3) uint8 BGR frame
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        sys.exit(f"ERROR: could not read frame {frame_idx} of {video} (past the end, or unreadable)")
    return frame


def verdict_line(detection: CornerDetection) -> str:
    """:return: 'gate PASS' or 'gate FAIL: <flag>|<flag>' for the panel/progress.

    :param detection: one frame's gated corner detection
    """
    if detection.passed:
        return "gate PASS"
    return f"gate FAIL: {'|'.join(detection.flags)}"


def draw_overlay(frame_bgr: np.ndarray, detection: CornerDetection) -> Image.Image:
    """Render the quad + corner dots + info panel onto a copy of the frame.

    :param frame_bgr: (H, W, 3) uint8 BGR frame the detection came from
    :param detection: its gated corner detection (corners in this frame's pixels)
    :return: an RGB PIL image ready to save
    """
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    height, width = frame_bgr.shape[:2]

    # Scale line/dot sizes to the smaller frame extent so the quad stays visible
    # from 288p up to broadcast HD.
    thickness = max(2, round(min(height, width) / 180))
    dot_radius = max(3, thickness * 2)

    # corners_px is TL, TR, BR, BL; close the loop back to TL for a full outline.
    corners = [(float(x), float(y)) for x, y in detection.corners_px]
    draw.line([*corners, corners[0]], fill=QUAD_COLOUR, width=thickness)
    for cx, cy in corners:
        draw.ellipse([cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius], fill=QUAD_COLOUR)

    lines = [
        f"{label} peak {float(detection.peak[i]):.3f} ent {float(detection.entropy[i]):.2f}"
        for i, label in enumerate(CORNER_LABELS)
    ]
    lines.append(verdict_line(detection))
    draw_panel(draw, lines, height)
    return img


def draw_panel(draw: ImageDraw.ImageDraw, lines: list[str], frame_h: int) -> None:
    """Draw the top-left text panel on its dark translucent backing rectangle.

    Font size floors at 16 px so the panel stays legible at the saved resolution,
    and grows with frame height for larger inputs. The backing is sized to the
    measured text so it never crops a long ``gate FAIL: ...`` verdict.

    :param draw: an RGBA-mode ImageDraw bound to the target image
    :param lines: panel text lines, top to bottom (corners then verdict)
    :param frame_h: frame height in pixels, drives font size and margin
    """
    font_size = max(16, round(frame_h / 20))
    font = load_font(font_size)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    inner_pad = round(font_size * 0.4)
    margin = max(6, round(frame_h / 48))

    text_w = max(draw.textlength(line, font=font) for line in lines)
    panel_w = text_w + inner_pad * 2
    panel_h = line_h * len(lines) + inner_pad * 2

    draw.rectangle([margin, margin, margin + panel_w, margin + panel_h], fill=PANEL_FILL)
    for i, line in enumerate(lines):
        draw.text((margin + inner_pad, margin + inner_pad + i * line_h), line, fill=TEXT_COLOUR, font=font)


def maybe_pngquant(path: Path, pngquant_bin: str | None) -> None:
    """Shrink ``path`` in place with pngquant when the binary is available.

    :param path: the PNG just written
    :param pngquant_bin: pngquant's resolved path, or None to leave the PNG as-is
    """
    if pngquant_bin is None:
        return
    result = subprocess.run(
        [pngquant_bin, "--quality", "40-60", "--force", "--output", str(path), str(path)],
        check=False,
    )
    # Exit 99 = pngquant could not reach the minimum quality and wrote nothing; the
    # plain PNG stays, which is the documented no-pngquant behaviour. Anything else
    # non-zero is a real failure.
    if result.returncode not in (0, 99):
        raise subprocess.CalledProcessError(result.returncode, result.args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)

    frame_mode = parser.add_mutually_exclusive_group(required=True)
    frame_mode.add_argument("--frames", type=str, help='Comma list of frame indices, e.g. "12,99,1042".')
    frame_mode.add_argument("--frames-csv", type=Path, help="CSV whose --frame-col column lists frame indices.")
    parser.add_argument("--frame-col", type=str, default="frame", help="Column read from --frames-csv (default 'frame').")

    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default cpu).")
    parser.add_argument("--resize-mode", choices=("pad", "squash"), default="pad", help="Detector resize (default pad).")
    parser.add_argument("--batch", type=int, default=10, help="Frames per detector forward pass (default 10).")
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"ERROR: --video does not exist: {args.video}")

    frame_indices = parse_frame_indices(args)
    if not frame_indices:
        sys.exit("ERROR: no frame indices resolved from the given selection.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pngquant_bin = shutil.which("pngquant")
    if pngquant_bin is None:
        print("pngquant not on PATH; keeping plain (larger) PNGs.")

    detector = CourtKeyNetDetector(device=args.device, resize_mode=args.resize_mode)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video {args.video}")

    video_stem = args.video.stem
    print(f"Rendering {len(frame_indices)} frames of {args.video} ({args.resize_mode}) -> {args.out_dir}")

    try:
        for start in range(0, len(frame_indices), args.batch):
            chunk = frame_indices[start : start + args.batch]
            frames = [read_frame(cap, idx, args.video) for idx in chunk]
            detections = detector.detect_batch(frames)
            for frame_idx, frame_bgr, detection in zip(chunk, frames, detections):
                img = draw_overlay(frame_bgr, detection)
                out_path = args.out_dir / f"{video_stem}_f{frame_idx:07d}_{args.resize_mode}.png"
                img.save(out_path)
                maybe_pngquant(out_path, pngquant_bin)
                verdict = "PASS" if detection.passed else f"FAIL {'|'.join(detection.flags)}"
                print(f"  f{frame_idx:07d}  {verdict}  peak_min={float(detection.peak.min()):.3f}  -> {out_path.name}")
    finally:
        cap.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
