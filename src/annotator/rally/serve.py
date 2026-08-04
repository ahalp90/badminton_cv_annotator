"""Serve-start options and sticky serve-setup policy."""

from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import NamedTuple
import warnings

import numpy as np

from ..types import Slot, StickyResult


class ServeSetupInputs(NamedTuple):
    """Per-frame evidence for the body-height-unit serve-setup gate.

    Ankles are image-fraction centroids and heights are image-HEIGHT fractions.
    ``wrist_dist`` is the raw wrist-to-shuttle distance as an image-height
    fraction, with ``+inf`` outside analysed coverage and NaN when a frame or
    slot has no measured distance. Dividing the raw pixel distance by the same
    image height used for ``top_height`` and ``bot_height`` makes the gate's
    ratio cancel exactly to body heights, with no hidden normalisation step.
    """

    count: np.ndarray
    wrist_dist: np.ndarray
    analysed: np.ndarray
    top_ankles: np.ndarray
    bot_ankles: np.ndarray
    top_height: np.ndarray
    bot_height: np.ndarray

    def validate(self) -> None:
        """Validate shapes, dtypes, and the count contract."""
        arrays = {name: np.asarray(value) for name, value in self._asdict().items()}
        trailing_shapes = {
            'count': (), 'wrist_dist': (2,), 'analysed': (),
            'top_ankles': (2,), 'bot_ankles': (2,),
            'top_height': (), 'bot_height': (),
        }
        for name, trailing in trailing_shapes.items():
            value = arrays[name]
            if value.ndim != 1 + len(trailing) or value.shape[1:] != trailing:
                raise ValueError(f'{name} has wrong shape/rank')
        if len({value.shape[0] for value in arrays.values()}) != 1:
            raise ValueError('ServeSetupInputs fields must have equal first-axis length')

        count = arrays['count']
        if (np.issubdtype(count.dtype, np.bool_) or
                not np.issubdtype(count.dtype, np.number) or
                np.issubdtype(count.dtype, np.complexfloating)):
            raise ValueError('count must be numeric real values')
        if not np.all(np.isfinite(count)) or np.any(count < 0) or np.any(count != np.floor(count)):
            raise ValueError('count must be finite, nonnegative, integer-valued reals')
        if not np.issubdtype(arrays['analysed'].dtype, np.bool_):
            raise ValueError('analysed must have boolean dtype')
        for name in ('wrist_dist', 'top_ankles', 'bot_ankles', 'top_height', 'bot_height'):
            if not np.issubdtype(arrays[name].dtype, np.floating):
                raise ValueError(f'{name} must have floating-point dtype')


def series_drift(points: np.ndarray) -> tuple[float, int]:
    """Return median-half drift for a sentinel-coded point series.

    ``points`` may contain NaN rows and the paired-zero ``(0, 0)`` sentinel;
    it is not suitable for arbitrary geometry where the origin is meaningful.
    Both coordinates must be finite and the pair must not be zero, while
    ``(0, y)`` and ``(x, 0)`` remain detected.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError('points must have shape (n, 2)')
    if (np.issubdtype(points.dtype, np.bool_) or
            not np.issubdtype(points.dtype, np.number) or
            np.issubdtype(points.dtype, np.complexfloating)):
        raise ValueError('points must have real numeric dtype')
    detected = np.all(np.isfinite(points), axis=1) & np.any(points != 0, axis=1)
    points = points[detected]
    detected_count = len(points)
    if detected_count < 2:
        return float('nan'), detected_count
    split = (detected_count + 1) // 2
    first = np.median(points[:split], axis=0)
    second = np.median(points[split:], axis=0)
    return float(np.linalg.norm(second - first)), detected_count


# Presence floor per required player in the serve-setup gate.
PLAYER_PRESENT_MIN_FRAC = 0.5


def serve_setup_still(
    inputs: ServeSetupInputs,
    claimed_serve_frame: int,
    window_frames: int,
    threshold_bh: float,
    slots: tuple[Slot, ...],
) -> bool:
    """Return whether every requested player is still through the serve frame."""
    inputs.validate()
    if isinstance(window_frames, bool) or not isinstance(window_frames, (int, np.integer)) or window_frames <= 0:
        raise ValueError('window_frames must be a positive integer')
    t = len(inputs.count)
    if isinstance(claimed_serve_frame, bool) or not isinstance(claimed_serve_frame, (int, np.integer)):
        raise ValueError('claimed_serve_frame must be an integer in range [0, t)')
    if not 0 <= claimed_serve_frame < t:
        raise ValueError('claimed_serve_frame must be in range [0, t)')
    if not np.isfinite(threshold_bh) or threshold_bh < 0:
        raise ValueError('threshold_bh must be finite and nonnegative')
    # Element check before set(): an unhashable member must ValueError, not TypeError.
    if (not isinstance(slots, tuple) or not slots or
            any(not isinstance(slot, Slot) for slot in slots) or
            len(set(slots)) != len(slots)):
        raise ValueError('slots must be a nonempty duplicate-free tuple of Slot values')

    end = int(claimed_serve_frame) + 1
    window = slice(max(0, end - int(window_frames)), end)
    for slot in slots:
        ankles = inputs.top_ankles[window] if slot is Slot.TOP else inputs.bot_ankles[window]
        heights = inputs.top_height[window] if slot is Slot.TOP else inputs.bot_height[window]
        drift, _ = series_drift(ankles)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            body_unit = float(np.nanmean(heights))
        if not np.isfinite(body_unit) or body_unit <= 0:
            return False
        ratio = drift / body_unit
        if not np.isfinite(ratio) or ratio > threshold_bh:
            return False
    return True


def build_serve_setup_inputs(
    sticky: 'StickyResult', resolution: tuple[float, float],
) -> ServeSetupInputs:
    """Build validated sticky-sourced evidence for the serve-setup gate."""
    if (not isinstance(resolution, tuple) or len(resolution) != 2 or
            any(isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0 for value in resolution)):
        raise ValueError('resolution must be two finite positive real components')

    height = float(resolution[1])
    count = np.asarray(sticky.standing_count).copy()
    wrist_dist = np.asarray(sticky.wrist_dist_px, dtype=np.float64).copy() / height
    analysed = np.asarray(sticky.analysed, dtype=bool).copy()
    top_ankles = np.full_like(sticky.ankle_pos[:, Slot.TOP], np.nan, dtype=float)
    bot_ankles = np.full_like(sticky.ankle_pos[:, Slot.BOTTOM], np.nan, dtype=float)
    top_height = np.full(len(count), np.nan, dtype=float)
    bot_height = np.full(len(count), np.nan, dtype=float)

    for slot, ankles_out, heights_out in (
        (Slot.TOP, top_ankles, top_height), (Slot.BOTTOM, bot_ankles, bot_height),
    ):
        ankles = sticky.ankle_pos[:, slot]
        box_height = sticky.bbox_height[:, slot]
        ankle_valid = np.all(np.isfinite(ankles), axis=1) & np.any(ankles != 0, axis=1)
        height_valid = np.isfinite(box_height) & (box_height > 0)
        ankles_out[ankle_valid] = ankles[ankle_valid]
        heights_out[height_valid] = box_height[height_valid] / height

    inputs = ServeSetupInputs(
        count=count, wrist_dist=wrist_dist, analysed=analysed,
        top_ankles=top_ankles, bot_ankles=bot_ankles,
        top_height=top_height, bot_height=bot_height,
    )
    inputs.validate()
    return inputs


class ServeStartMode(StrEnum):
    """What a region whose bursts are none of them serve-setup-preceded does.

    TRIM keeps the span at the region's first burst (the stock pick), so coverage is only
    ever traded by a later start on the QUALIFYING regions. REJECT drops the region outright,
    the stronger anti-weld / anti-spurious lever, at the risk of dropping a GT rally that
    happens to sit in a no-qualify region.
    """

    TRIM = 'trim'
    REJECT = 'reject'


class ServeStartClose(StrEnum):
    """Where a split span closes; the serve-start split axis (None = single span, off).

    In split mode every serve-setup-qualifying burst opens a span. BURST closes the previous
    span exactly at the next qualifying burst, so the split spans union back to the single
    span and coverage is unchanged. LAST_REST closes it at the start of the last rest run
    before that burst, leaving the between-rally dead tail (where the junk contacts live)
    outside the span, at the risk of uncovering a rally whose own serve failed the gate.
    """

    BURST = 'burst'
    LAST_REST = 'last_rest'


class ServeStartOptions(NamedTuple):
    """Serve-start gating for segment_video(serve_start=...).

    The sticky setup evidence is built by the caller from the unmasked track. The committed
    measurement convention builds serve-start evidence before any replay mask is applied.

    :param dist: retired raw-distance carrier. It must remain None.
    :param threshold: serve-setup gate distance as a multiple of body height.
    :param mode: fallback for a region with no qualifying burst (TRIM / REJECT).
    :param close: optional split placement; None opens one span per region (the default).
    :param diagnostics: optional caller-supplied dict; when given, the span rule fills it in
        place with the per-call region counts / spacings (single writer, valid to read straight
        after the call IN THE SAME PROCESS: the in-place fill does not cross a multiprocessing
        worker boundary, so the pooled sweep runner leaves it None). None (the default) collects nothing.
    :param setup: sticky-sourced evidence (build_serve_setup_inputs).
    :param stillness_threshold_bh: optional stillness bound in body heights for the sticky
        gate; None (the default) leaves the stillness check off. Sticky path only.
    :param lookback_frames: resolved setup-window length in frames (the fps table's
        serve_start_lookback_frames row); required with setup. Sticky path only.
    :param stillness_window_frames: resolved stillness-window length in frames (the
        serve_stillness_window_frames row); required once stillness_threshold_bh is set.
        Sticky path only.
    """

    dist: np.ndarray | None
    threshold: float
    mode: ServeStartMode
    close: ServeStartClose | None = None
    diagnostics: dict | None = None
    setup: ServeSetupInputs | None = None
    stillness_threshold_bh: float | None = None
    lookback_frames: int | None = None
    stillness_window_frames: int | None = None


def _serve_distance_ratio_passes(
    window_dist: np.ndarray, window_height: np.ndarray, threshold_bh: float,
) -> bool:
    """Return whether paired distance evidence passes the body-height threshold.

    The finite-distance mask selects the matching heights. Sticky setup construction writes both
    values for a picked slot on the same frame.
    """
    mask = np.isfinite(window_dist)
    if not mask.any():
        return False
    ratio = np.median(window_dist[mask]) / np.mean(window_height[mask])
    return bool(ratio <= threshold_bh)


def _sticky_serve_setup_before(
    setup: ServeSetupInputs, burst: int, threshold: float, lookback_frames: int,
    stillness_threshold_bh: float | None, stillness_window_frames: int | None,
) -> bool:
    """Apply the sticky coverage gate and its three serve-setup lanes."""
    if not 0 <= burst < len(setup.count):
        raise ValueError('claimed_serve_frame must be in range [0, t)')
    setup_window = slice(max(0, burst - lookback_frames), burst)
    if setup_window.start == setup_window.stop:
        return False
    if stillness_threshold_bh is None:
        stillness_window = setup_window
    else:
        assert stillness_window_frames is not None
        stillness_window = slice(max(0, burst + 1 - stillness_window_frames), burst + 1)
    coverage_start = min(setup_window.start, stillness_window.start)
    coverage_stop = max(setup_window.stop, stillness_window.stop)
    if not np.all(setup.analysed[coverage_start:coverage_stop]):
        return False

    count = setup.count[setup_window]
    median_count = float(np.median(count))
    distances = setup.wrist_dist[setup_window]
    heights = (setup.top_height[setup_window], setup.bot_height[setup_window])
    valid = np.empty(distances.shape, dtype=bool)
    for slot in Slot:
        valid[:, slot] = np.isfinite(distances[:, slot]) & np.isfinite(heights[slot])

    if median_count >= 2:
        # Presence floor AND the primitive's minimum detections: below two valid
        # rows a drift cannot be split into halves, so the window fails closed
        # even with the stillness gate off.
        if any(np.mean(valid[:, slot]) < PLAYER_PRESENT_MIN_FRAC or
               np.count_nonzero(valid[:, slot]) < 2 for slot in Slot):
            return False
        if not any(
            _serve_distance_ratio_passes(distances[:, slot], heights[slot], threshold)
            for slot in Slot
        ):
            return False
        return stillness_threshold_bh is None or serve_setup_still(
            setup, burst, stillness_window_frames, stillness_threshold_bh, (Slot.TOP, Slot.BOTTOM),
        )

    if median_count >= 1:
        for slot in Slot:
            slot_valid = valid[:, slot]
            if np.mean(slot_valid) < PLAYER_PRESENT_MIN_FRAC or np.count_nonzero(slot_valid) < 2:
                continue
            if not _serve_distance_ratio_passes(distances[:, slot], heights[slot], threshold):
                continue
            if stillness_threshold_bh is None:
                return True
            masked = setup._replace(
                top_ankles=setup.top_ankles.copy(), bot_ankles=setup.bot_ankles.copy(),
                top_height=setup.top_height.copy(), bot_height=setup.bot_height.copy(),
            )
            ankles_out = masked.top_ankles if slot is Slot.TOP else masked.bot_ankles
            heights_out = masked.top_height if slot is Slot.TOP else masked.bot_height
            stillness_distances = setup.wrist_dist[stillness_window, slot]
            stillness_ankles = ankles_out[stillness_window]
            stillness_heights = heights_out[stillness_window]
            stillness_valid = (
                np.isfinite(stillness_distances) & np.all(np.isfinite(stillness_ankles), axis=1) &
                np.isfinite(stillness_heights)
            )
            stillness_indexes = np.arange(stillness_window.start, stillness_window.stop)
            ankles_out[stillness_indexes[~stillness_valid]] = np.nan
            heights_out[stillness_indexes[~stillness_valid]] = np.nan
            if serve_setup_still(masked, burst, stillness_window_frames, stillness_threshold_bh, (slot,)):
                return True
        return False

    # TODO: position-based count means a tight close-up CAN read 1 and route partial; the old
    # "close-ups read zero" claim was size-banded and does not describe the permanent
    # court-scale box and slot paths.
    return False



@dataclass(frozen=True)
class _ServeGate:
    """Validated serve-start evidence and thresholds."""

    setup: ServeSetupInputs
    threshold: float
    lookback_frames: int
    stillness_threshold_bh: float | None
    stillness_window_frames: int | None

    def qualifies(self, burst: int) -> bool:
        """Return whether one burst has qualifying setup evidence."""
        return _sticky_serve_setup_before(
            self.setup,
            burst,
            self.threshold,
            self.lookback_frames,
            self.stillness_threshold_bh,
            self.stillness_window_frames,
        )


def _valid_serve_window(value: object, name: str) -> int:
    """Return one positive integer window length."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _valid_serve_threshold(value: object, name: str) -> float:
    """Return one finite nonnegative serve threshold."""
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def _resolve_serve_gate(options: ServeStartOptions) -> _ServeGate:
    """Validate serve-start options and return the executable gate."""
    if options.dist is not None:
        raise ValueError('legacy serve-start dist is no longer supported; supply setup')
    if options.setup is None:
        raise ValueError('serve-start setup must be supplied')
    threshold = _valid_serve_threshold(options.threshold, 'threshold')
    lookback_frames = _valid_serve_window(options.lookback_frames, 'lookback_frames')
    if options.stillness_threshold_bh is not None:
        stillness_threshold_bh = _valid_serve_threshold(
            options.stillness_threshold_bh, 'stillness_threshold_bh',
        )
        stillness_window_frames = _valid_serve_window(
            options.stillness_window_frames, 'stillness_window_frames',
        )
    else:
        stillness_threshold_bh = None
        # Still validated when supplied: a bad window would silently wrap the
        # coverage-gate slice even with the stillness gate off.
        stillness_window_frames = (
            None if options.stillness_window_frames is None
            else _valid_serve_window(options.stillness_window_frames, 'stillness_window_frames')
        )
    options.setup.validate()
    return _ServeGate(
        options.setup,
        threshold,
        lookback_frames,
        stillness_threshold_bh,
        stillness_window_frames,
    )
