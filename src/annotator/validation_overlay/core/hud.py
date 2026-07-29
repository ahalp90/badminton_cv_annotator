"""Scaled OpenCV HUD and mark-label drawing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from annotator.validation_overlay.core.timeline import SpanState


FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _scaled(value: float, scale: float) -> int:
    return max(1, int(round(value * scale)))


def _font_scale_for_cap_height(cap_height: int, thickness: int) -> float:
    font_scale = max(0.01, cap_height / cv2.getTextSize("H", FONT, 1.0, thickness)[0][1])
    for _ in range(8):
        measured = cv2.getTextSize("H", FONT, font_scale, thickness)[0][1]
        if measured == cap_height:
            return font_scale
        font_scale *= cap_height / max(1, measured)
    return font_scale


@dataclass(frozen=True)
class HudStyle:
    """All HUD geometry derived from the actual output width."""

    output_width: int
    output_height: int
    scale: float
    inset: int
    padding: int
    line_spacing: int
    text_height: int
    text_thickness: int
    font_scale: float


def make_hud_style(output_width: int, output_height: int, hud_height: int = 14) -> HudStyle:
    """Build a fixed-spacing HUD style for an output resolution."""
    if output_width <= 0 or output_height <= 0:
        raise ValueError(f"output dimensions must be positive, got {output_width}x{output_height}")
    if hud_height <= 0:
        raise ValueError(f"hud_height must be positive, got {hud_height}")
    scale = output_width / 1920.0
    text_height = _scaled(hud_height, scale)
    text_thickness = _scaled(1.0, scale)
    return HudStyle(
        output_width=output_width,
        output_height=output_height,
        scale=scale,
        inset=_scaled(12.0, scale),
        padding=_scaled(8.0, scale),
        line_spacing=_scaled(24.0, scale),
        text_height=text_height,
        text_thickness=text_thickness,
        font_scale=_font_scale_for_cap_height(text_height, text_thickness),
    )


def draw_mark_label(
    image: np.ndarray,
    label: str,
    x: int,
    y: int,
    style: HudStyle,
) -> None:
    """Draw a mark label at a position supplied by an overlay."""
    cv2.putText(
        image,
        label,
        (x, y),
        FONT,
        style.font_scale,
        WHITE,
        style.text_thickness,
        cv2.LINE_AA,
    )


def draw_hud(
    image: np.ndarray,
    source_idx: int,
    state: SpanState,
    segment_label: str | None,
    show_segment_label: bool,
    extra_lines: list[str] | None,
    style: HudStyle,
) -> None:
    """Draw the core HUD after an overlay has drawn its marks."""
    lines = [f"f{source_idx}  {state.value}"]
    if show_segment_label and segment_label is not None:
        lines.append(segment_label)
    if extra_lines:
        lines.extend(extra_lines)

    text_sizes = [cv2.getTextSize(line, FONT, style.font_scale, style.text_thickness) for line in lines]
    max_width = max(size[0][0] for size in text_sizes)
    first_baseline = style.inset + style.padding + text_sizes[0][0][1]
    last_baseline = first_baseline + style.line_spacing * (len(lines) - 1)
    _, last_baseline_offset = text_sizes[-1]
    rectangle_right = style.inset + style.padding * 2 + max_width
    rectangle_bottom = last_baseline + last_baseline_offset + style.padding
    cv2.rectangle(
        image,
        (style.inset, style.inset),
        (rectangle_right, rectangle_bottom),
        BLACK,
        thickness=-1,
    )
    text_x = style.inset + style.padding
    for line_index, line in enumerate(lines):
        baseline = first_baseline + style.line_spacing * line_index
        cv2.putText(
            image,
            line,
            (text_x, baseline),
            FONT,
            style.font_scale,
            WHITE,
            style.text_thickness,
            cv2.LINE_AA,
        )
