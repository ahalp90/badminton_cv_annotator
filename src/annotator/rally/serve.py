"""Recognise serve preparation from player positions and wrist-to-shuttle distance."""

import warnings
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import NamedTuple

import numpy as np

from ..types import Slot, StickyResult


class ServeSetupInputs(NamedTuple):
    """Player measurements used to recognise serve preparation at each frame.

    Ankle coordinates are fractions of image width and height. Player heights
    and wrist-to-shuttle distances are fractions of image height, so their ratio
    expresses distance in player body heights. Wrist distance is ``+inf`` outside
    analysed frames and NaN where a player has no measured distance.
    """

    count: np.ndarray
    wrist_dist: np.ndarray
    analysed: np.ndarray
    top_ankles: np.ndarray
    bot_ankles: np.ndarray
    top_height: np.ndarray
    bot_height: np.ndarray

    def validate(self) -> None:
        """Check array shapes, dtypes and nonnegative whole-number player counts."""
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
    """Measure movement between the first and second halves of a position series.

    Ignore non-finite positions and ``(0, 0)``, which denotes a missing detection.
    Split the remaining positions in time order, keeping any extra point in the
    first half. Return the distance between the two median positions and the
    number of detections. Fewer than two detections give a NaN distance.
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
        raise TypeError('claimed_serve_frame must be an integer in range [0, t)')
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
    """Prepare player measurements from sticky tracking for serve detection."""
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
    """What to do when no fast burst in a region follows detected serve preparation.

    TRIM starts the rally at the first fast burst. REJECT discards the region.
    """

    TRIM = 'trim'
    REJECT = 'reject'


class ServeStartClose(StrEnum):
    """Where to end a rally when another burst follows detected serve preparation.

    BURST ends it at that next burst, so the split rallies cover the same frames
    as one rally. LAST_REST ends it at the start of the last rest period between
    the bursts, or at the next burst if there is no such rest. Excluding the rest
    can remove contacts between rallies.
    Setting ``ServeStartOptions.close`` to None keeps one rally per region.
    """

    BURST = 'burst'
    LAST_REST = 'last_rest'


class ServeStartOptions(NamedTuple):
    """Settings and sticky player evidence for serve-start gating.

    Build setup evidence from the original track before applying replay masks.

    :param threshold: Serve-setup distance as a multiple of body height.
    :param mode: Fallback for a region with no qualifying burst (TRIM / REJECT).
    :param close: Optional split placement; None opens one span per region.
    :param diagnostics: Optional dict filled with region counts and spacings in
        the calling process. Worker processes do not share this mapping.
    :param setup: Evidence from build_serve_setup_inputs.
    :param stillness_threshold_bh: Optional stillness bound in body heights;
        None disables the stillness check.
    :param lookback_frames: Required resolved setup-window length in frames.
    :param stillness_window_frames: Resolved stillness-window length, required
        when stillness_threshold_bh is set.
    """

    threshold: float
    mode: ServeStartMode
    close: ServeStartClose | None = None
    diagnostics: dict | None = None
    setup: ServeSetupInputs | None = None
    stillness_threshold_bh: float | None = None
    lookback_frames: int | None = None
    stillness_window_frames: int | None = None


def serve_distance_ratio_passes(
    window_dist: np.ndarray, window_height: np.ndarray, threshold_bh: float,
) -> bool:
    """Check median wrist-to-shuttle distance relative to mean player height.

    Use only frames with a finite distance. Sticky tracking supplies the matching
    player height on those frames.
    """
    mask = np.isfinite(window_dist)
    if not mask.any():
        return False
    ratio = np.median(window_dist[mask]) / np.mean(window_height[mask])
    return bool(ratio <= threshold_bh)


def sticky_serve_setup_before(
    setup: ServeSetupInputs, burst: int, threshold: float, lookback_frames: int,
    stillness_threshold_bh: float | None, stillness_window_frames: int | None,
) -> bool:
    """Check for serve preparation in the frames before a fast burst.

    All required frames must have been analysed. When the window's median standing
    count is at least two, require measurements for both players and a nearby
    shuttle for at least one. Both must pass stillness checks when enabled. For
    a median count of at least one but below two, require one sufficiently observed player
    with a nearby shuttle and, when enabled, stillness. Reject lower median counts.
    """
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
        # Each player needs measurements in at least half the window. The two-frame
        # minimum comes from the drift check and applies even with stillness off.
        if any(np.mean(valid[:, slot]) < PLAYER_PRESENT_MIN_FRAC or
               np.count_nonzero(valid[:, slot]) < 2 for slot in Slot):
            return False
        if not any(
            serve_distance_ratio_passes(distances[:, slot], heights[slot], threshold)
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
            if not serve_distance_ratio_passes(distances[:, slot], heights[slot], threshold):
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

    return False



@dataclass(frozen=True)
class ServeGate:
    """Validated serve-start evidence and thresholds."""

    setup: ServeSetupInputs
    threshold: float
    lookback_frames: int
    stillness_threshold_bh: float | None
    stillness_window_frames: int | None

    def qualifies(self, burst: int) -> bool:
        """Return whether one burst has qualifying setup evidence."""
        return sticky_serve_setup_before(
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


def resolve_serve_gate(options: ServeStartOptions) -> ServeGate:
    """Validate the measurements and thresholds used to recognise serve preparation."""
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
        # Validate an explicitly supplied window even when stillness is disabled.
        stillness_window_frames = (
            None if options.stillness_window_frames is None
            else _valid_serve_window(options.stillness_window_frames, 'stillness_window_frames')
        )
    options.setup.validate()
    return ServeGate(
        options.setup,
        threshold,
        lookback_frames,
        stillness_threshold_bh,
        stillness_window_frames,
    )
