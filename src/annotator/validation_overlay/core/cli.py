"""Shared CLI flags, frame composition, identity verification and rendering."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from annotator.validation_overlay.core.decode import iter_span_frames
from annotator.video_metadata import VideoMetadata
from annotator.validation_overlay.core.encode import encode_frames
from annotator.validation_overlay.core.hud import HudStyle, draw_hud, make_hud_style
from annotator.validation_overlay.core.timeline import (
    Segment,
    SegmentPlan,
    SpanState,
    SpacerPlan,
    TimelinePlan,
    build_timeline,
)


DrawFn = Callable[[np.ndarray, int, bool], list[str] | None]

# x264's own ceiling. A corrupt or absurd sample aspect ratio can derive a
# gigantic canvas, and this catches it before anything allocates a frame.
_MAX_DIMENSION = 16384


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _nonnegative_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected non-negative seconds, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(f"expected non-negative seconds, got {value!r}")
    return parsed


def build_shared_parser() -> argparse.ArgumentParser:
    """Return the argparse parent shared by independent overlay CLIs."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", type=Path, required=True, help="source video")
    parser.add_argument("--segments", type=Path, required=True, help="inclusive segment CSV")
    parser.add_argument("--out", type=Path, required=True, help="output video path")
    parser.add_argument("--start-col", default="start_frame", help="CSV start-frame column")
    parser.add_argument("--end-col", default="end_frame", help="CSV end-frame column")
    parser.add_argument("--label-col", default="label", help="optional CSV label column")
    parser.add_argument(
        "--render-width",
        type=_positive_int,
        default=1920,
        help="minimum output width in pixels (default: 1920)",
    )
    parser.add_argument(
        "--hud-height",
        type=_positive_int,
        default=14,
        help="HUD and mark text height at the 1920 reference width (default: 14)",
    )
    parser.add_argument(
        "--lead-in",
        type=_nonnegative_seconds,
        default=2.5,
        help="lead-in seconds, 62 frames at 25 fps (default: 2.5)",
    )
    parser.add_argument(
        "--lead-out",
        type=_nonnegative_seconds,
        default=2.5,
        help="lead-out seconds, 62 frames at 25 fps (default: 2.5)",
    )
    parser.add_argument(
        "--spacer",
        type=_nonnegative_seconds,
        default=1.0,
        help="black spacer seconds, 25 frames at 25 fps (default: 1.0)",
    )
    parser.add_argument("--label", default="shuttle", help="mark label (default: shuttle)")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run the source-resolution SHA-256 identity gate before encoding",
    )
    return parser


@dataclass(frozen=True)
class RenderPlan:
    """Operational render plan assembled after all static validation."""

    timeline: TimelinePlan
    video: Path
    source_width: int
    source_height: int
    output: Path
    output_width: int
    output_height: int
    hud_style: HudStyle
    verify: bool

    @property
    def fps(self) -> Fraction:
        return self.timeline.fps

@dataclass(frozen=True)
class RenderResult:
    """Counts reported by a successful render."""

    output: Path
    output_frames: int
    verified_distinct_indices: int


def _current_umask() -> int:
    """Read the process umask. Reading it requires setting it, so put it back."""
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _round_fraction(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    if remainder * 2 >= value.denominator:
        quotient += 1
    return quotient


def make_render_plan(
    info: VideoMetadata,
    segments: Sequence[Segment],
    output: Path,
    *,
    render_width: int = 1920,
    hud_height: int = 14,
    lead_in: float | Fraction = 2.5,
    lead_out: float | Fraction = 2.5,
    spacer: float | Fraction = 1.0,
    verify: bool = False,
) -> RenderPlan:
    """Build the operational plan after metadata and input validation."""
    if render_width <= 0:
        raise ValueError(f"render_width must be positive, got {render_width}")
    if hud_height <= 0:
        raise ValueError(f"hud_height must be positive, got {hud_height}")
    for name, duration in (("lead_in", lead_in), ("lead_out", lead_out), ("spacer", spacer)):
        if duration < 0 or not math.isfinite(float(duration)):
            raise ValueError(f"{name} must be finite and non-negative, got {duration}")
    output = Path(output)
    if output.exists() and output.is_dir():
        raise ValueError(f"output path is a directory: {output}")
    if output.parent.exists() and not output.parent.is_dir():
        raise ValueError(f"output parent is not a directory: {output.parent}")

    # Anamorphic sources are written at their true display shape rather than
    # refused. Widening to at least coded_width * SAR first matters: sizing off
    # the coded width alone would satisfy the ratio by SHRINKING the other axis,
    # throwing away real detail on a wide-pixel source.
    aspect = info.sample_aspect_ratio
    square_width = -((-info.width * aspect.numerator) // aspect.denominator)  # ceil
    output_width = max(info.width, render_width, int(square_width))
    if output_width % 2:
        output_width += 1
    output_height = _round_fraction(Fraction(info.height * output_width, info.width) / aspect)
    if output_height % 2:
        output_height += 1
    output_height = max(2, output_height)
    if output_width > _MAX_DIMENSION or output_height > _MAX_DIMENSION:
        raise ValueError(
            f"output {output_width}x{output_height} exceeds the {_MAX_DIMENSION} px limit; "
            f"check the source's sample aspect ratio ({aspect}) and --render-width"
        )
    timeline = build_timeline(segments, info.frame_count, info.fps, lead_in, lead_out, spacer)
    hud_style = make_hud_style(output_width, output_height, hud_height)
    return RenderPlan(
        timeline=timeline,
        video=info.source_path,
        source_width=info.width,
        source_height=info.height,
        output=output,
        output_width=output_width,
        output_height=output_height,
        hud_style=hud_style,
        verify=verify,
    )


class _IdentityGate:
    def __init__(self, reference_hashes: dict[int, str]) -> None:
        self.reference_hashes = reference_hashes
        self.seen_indices: set[int] = set()
        self.compared_frames = 0

    def check(self, source_idx: int, frame: np.ndarray) -> None:
        reference = self.reference_hashes.get(source_idx)
        if reference is None:
            raise RuntimeError(f"identity gate saw unplanned source frame {source_idx}")
        actual = hashlib.sha256(frame.tobytes(order="C")).hexdigest()
        if actual != reference:
            raise RuntimeError(
                f"identity gate mismatch at source frame {source_idx}, "
                f"output frame {self.compared_frames}"
            )
        self.seen_indices.add(source_idx)
        self.compared_frames += 1

    def finish(self) -> int:
        expected = set(self.reference_hashes)
        if self.seen_indices != expected:
            missing = sorted(expected - self.seen_indices)
            raise RuntimeError(f"identity gate did not compare planned source indices: {missing[:5]}")
        return len(expected)


def _make_identity_gate(plan: RenderPlan) -> _IdentityGate:
    distinct_indices = plan.timeline.distinct_source_indices
    if not distinct_indices:
        raise ValueError("identity verification requires at least one source frame")
    largest_index = max(distinct_indices)
    reference_hashes: dict[int, str] = {}
    for source_idx, frame in enumerate(
        iter_span_frames(
            plan.video,
            0,
            largest_index,
            plan.fps,
            plan.source_width,
            plan.source_height,
        )
    ):
        if source_idx in distinct_indices:
            reference_hashes[source_idx] = hashlib.sha256(frame.tobytes(order="C")).hexdigest()
    if set(reference_hashes) != distinct_indices:
        raise RuntimeError(
            f"identity reference covered {len(reference_hashes)} indices, "
            f"expected {len(distinct_indices)}"
        )
    return _IdentityGate(reference_hashes)


def compose_frames(plan: RenderPlan, draw_fn: DrawFn, identity_gate: _IdentityGate | None = None) -> Iterator[np.ndarray]:
    """Yield the exact frames that ``render`` sends to the encoder."""
    for part in plan.timeline.parts:
        if isinstance(part, SpacerPlan):
            for _ in range(part.count):
                yield np.zeros((plan.output_height, plan.output_width, 3), dtype=np.uint8)
            continue

        if not isinstance(part, SegmentPlan):
            raise TypeError(f"unknown timeline part: {type(part).__name__}")
        decoded_count = 0
        decoded_frames = iter_span_frames(
            plan.video,
            part.effective_first,
            part.effective_last,
            plan.fps,
            plan.source_width,
            plan.source_height,
        )
        for decoded_count, source_frame in enumerate(decoded_frames, start=1):
            frame_spec = part.frames[decoded_count - 1]
            if identity_gate is not None:
                identity_gate.check(frame_spec.source_idx, source_frame)
            image = cv2.resize(
                source_frame,
                (plan.output_width, plan.output_height),
                interpolation=cv2.INTER_NEAREST,
            )
            extra_lines = draw_fn(
                image,
                frame_spec.source_idx,
                frame_spec.state is SpanState.TARGET,
            )
            if extra_lines is not None and not isinstance(extra_lines, list):
                raise TypeError("draw_fn must return list[str] or None")
            draw_hud(
                image,
                frame_spec.source_idx,
                frame_spec.state,
                frame_spec.segment_label,
                frame_spec.show_segment_label,
                extra_lines,
                plan.hud_style,
            )
            yield image
        if decoded_count != len(part.frames):
            raise RuntimeError(
                f"decoded {decoded_count} frames for planned span [{part.effective_first}, "
                f"{part.effective_last}], expected {len(part.frames)}"
            )


def render(plan: RenderPlan, draw_fn: DrawFn) -> RenderResult:
    """Compose and atomically encode one output video from a plan and draw function."""
    plan.output.parent.mkdir(parents=True, exist_ok=True)

    # Claim the temporary output before verifying. The identity gate costs a full
    # decode, and an undiscovered unwritable destination would waste all of it.
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{plan.output.name}.",
            suffix=plan.output.suffix or ".mp4",
            dir=plan.output.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

        identity_gate = _make_identity_gate(plan) if plan.verify else None
        verified_count = len(identity_gate.reference_hashes) if identity_gate else 0

        output_frames = encode_frames(
            compose_frames(plan, draw_fn, identity_gate),
            temporary_path,
            plan.output_width,
            plan.output_height,
            plan.fps,
        )
        if output_frames != plan.timeline.output_frame_count:
            raise RuntimeError(
                f"encoder accepted {output_frames} frames, expected {plan.timeline.output_frame_count}"
            )
        if identity_gate is not None:
            verified_count = identity_gate.finish()
        # NamedTemporaryFile creates 0600. These renders get opened in a player
        # and passed around, so give them the same mode a normal file would get.
        temporary_path.chmod(0o666 & ~_current_umask())
        os.replace(temporary_path, plan.output)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return RenderResult(plan.output, output_frames, verified_count)
