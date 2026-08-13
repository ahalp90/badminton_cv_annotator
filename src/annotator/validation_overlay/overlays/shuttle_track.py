"""Render TrackNetV3 shuttle tracks over selected source spans.

The documented entry point is
``PYTHONPATH=src python -m annotator.validation_overlay.overlays.shuttle_track``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from annotator.shuttle_track import validate_shuttle_track
from annotator.validation_overlay.core.cli import (
    DrawFn,
    build_shared_parser,
    make_render_plan,
    render,
)
from annotator.validation_overlay.core.hud import HudStyle, draw_mark_label
from annotator.validation_overlay.core.timeline import read_segments
from annotator.video_metadata import probe_video_metadata


BOX_COLOUR = (240, 16, 255)


def build_parser() -> argparse.ArgumentParser:
    """Build the TrackNetV3 overlay parser."""
    parser = argparse.ArgumentParser(
        description="Render TrackNetV3 shuttle marks over selected video spans",
        parents=[build_shared_parser()],
    )
    parser.add_argument("--track", type=Path, required=True, help="(n_frames, 3) TrackNetV3 .npy array")
    return parser


def load_track(track_path: Path, nb_frames: int) -> np.ndarray:
    """Load and validate the per-source-frame TrackNetV3 array."""
    track_path = Path(track_path)
    if not track_path.is_file():
        raise FileNotFoundError(f"track is not a regular file: {track_path}")
    track = np.load(track_path, allow_pickle=False)
    validate_shuttle_track(track, nb_frames)
    return track


def _clip_endpoint(value: int, upper_bound: int) -> int:
    return min(max(value, 0), upper_bound - 1)


def make_draw(track: np.ndarray, mark_label: str, style: HudStyle) -> DrawFn:
    """Bind validated track data to the core draw contract."""
    half_edge = max(1, int(round(25.0 * style.scale)))
    line_width = max(1, int(round(7.0 * style.scale)))

    def draw(image: np.ndarray, source_idx: int, in_target_span: bool) -> list[str]:
        del in_target_span
        x, y, visibility = track[source_idx]
        if visibility == 1.0:
            centre_x = int(round(float(x) * style.output_width))
            centre_y = int(round(float(y) * style.output_height))
            top_left = (
                _clip_endpoint(centre_x - half_edge, style.output_width),
                _clip_endpoint(centre_y - half_edge, style.output_height),
            )
            bottom_right = (
                _clip_endpoint(centre_x + half_edge, style.output_width),
                _clip_endpoint(centre_y + half_edge, style.output_height),
            )
            cv2.rectangle(image, top_left, bottom_right, BOX_COLOUR, line_width)
            if mark_label:
                draw_mark_label(
                    image,
                    mark_label,
                    centre_x + half_edge + style.padding,
                    centre_y - half_edge,
                    style,
                )
        return [f"x={x:.3f} y={y:.3f} vis={int(visibility)}"]

    return draw


def main(argv: Sequence[str] | None = None) -> int:
    """Validate inputs, render the selected spans and return a process status."""
    args = build_parser().parse_args(argv)
    try:
        info = probe_video_metadata(args.video)
        segments = read_segments(
            args.segments,
            info.frame_count,
            start_col=args.start_col,
            end_col=args.end_col,
            label_col=args.label_col,
        )
        track = load_track(args.track, info.frame_count)
        plan = make_render_plan(
            info,
            segments,
            args.out,
            render_width=args.render_width,
            hud_height=args.hud_height,
            lead_in=args.lead_in,
            lead_out=args.lead_out,
            spacer=args.spacer,
            verify=args.verify,
        )
        result = render(plan, make_draw(track, args.label, plan.hud_style))
    except Exception as exc:  # CLI boundary: turn hard validation/render errors into status 1.
        print(f"validation overlay failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {result.output} ({result.output_frames} frames)")
    if args.verify:
        print(f"verified {result.verified_distinct_indices} distinct source indices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
