#!/usr/bin/env python3
"""Render the court landmark naming diagram to a PNG.

Every landmark in ``court_landmarks.LANDMARKS`` is named ``row_x_column``; the
diagram draws the court grid with labelled rows and columns so an annotator can
decode any name at a glance. The same drawing also lives inside the annotation
tool's window (top right, during a capture); this file is for a printable or
shareable copy.

Usage::

    python src/courtkeynet/validation_scripts/make_landmark_key.py --out landmark_key.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import court_landmarks as court  # noqa: E402
from annotate_court_corners_offframe import draw_court_key  # noqa: E402

SCALE = 60  # px per court metre
LEFT, TOP, RIGHT, BOTTOM = 250, 130, 70, 60
WHITE, GREY = (255, 255, 255), (170, 170, 170)


def render_key() -> np.ndarray:
    """:return: the BGR key image."""
    width = LEFT + int(court.COURT_WIDTH_M * SCALE) + RIGHT
    height = TOP + int(court.COURT_LENGTH_M * SCALE) + BOTTOM
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    draw_court_key(canvas, (LEFT, TOP), SCALE, highlight=None, placed=(), corner_slot=None)

    def text(value: str, x: float, y: float, scale: float = 0.45, colour: tuple = WHITE) -> None:
        cv2.putText(canvas, value, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)

    text("Landmark key: every name is  row_x_column", 12, 28, 0.55)
    text("far/near and left/right are as seen on screen (far = top of the picture)", 12, 52, 0.42, GREY)

    # Column labels, staggered on two rows with ticks down to the court top.
    for index, (name, x_m) in enumerate(court.X_LINES.items()):
        column_x = LEFT + x_m * SCALE
        label_y = 86 if index % 2 == 0 else 108
        text(name, max(6, column_x - len(name) * 7.5 / 2), label_y, 0.4)
        cv2.line(canvas, (int(column_x), label_y + 4), (int(column_x), TOP - 14), (110, 110, 110), 1, cv2.LINE_AA)

    # Row labels on the left, ticks across to the line.
    for name, y_m in court.Y_LINES.items():
        row_y = TOP + y_m * SCALE
        text(name, 12, row_y + 4, 0.4)
        cv2.line(canvas, (LEFT - 110, int(row_y)), (LEFT - 16, int(row_y)), (110, 110, 110), 1, cv2.LINE_AA)
    text("net (not painted, no landmarks)", 12, TOP + court.NET_Y_M * SCALE + 4, 0.38, GREY)

    text("example: far_short_service_x_left_singles", 12, height - 24, 0.42, GREY)
    example_x, example_y = court.LANDMARKS["far_short_service_x_left_singles"]
    cv2.circle(canvas, (int(LEFT + example_x * SCALE), int(TOP + example_y * SCALE)), 9, (0, 94, 213), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("landmark_key.png"))
    args = parser.parse_args()
    if not cv2.imwrite(str(args.out), render_key()):
        sys.exit(f"ERROR: could not write {args.out}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
