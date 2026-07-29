"""Detect invented track content by improbable exact recurrence.

The idea. InpaintNet's fill for an evidence-free window is a fixed function of
the weights, so it emits the SAME position sequence every time it fires. Real
footage cannot reproduce W consecutive exact pixel positions at many unrelated
moments in the same video. So the fabrication is detectable as a pattern whose
recurrence is impossible for footage, WITHOUT naming any coordinate.

Why this beats matching known values:
  - no coordinate is hardcoded; attractors are discovered per video
  - it survives a retrained or swapped checkpoint, since the fabricated values
    change but their exact recurrence does not
  - one rule covers both shapes. The nonoverlap loop is a 16-position cycle;
    the weight-mode constant is the degenerate pattern of W identical positions
  - it would catch a fabrication shape nobody has seen, provided the shape
    repeats

Protecting real detections:
  - a genuinely resting shuttle makes ONE episode at whatever pixel it landed
    on, not hundreds at the same pixel, so it never reaches the threshold
  - a shuttle passing through an attractor pixel occupies it for a frame or
    two, never the full W, so it forms no pattern occurrence
  - short informed fills bridged from real detections differ every time, so
    they are never attractors
  - output is graded, so blend-contaminated frames are marked separately from
    evidence-free ones
"""
from collections import defaultdict

import numpy as np

BASE = ('/home/ariel/Documents/COSC594/badminton_stroke_classification'
        '/local_scratch/autograder_architecture')

CLEAN, INVENTED, NO_DETECTION, DEGRADED = 0, 1, 2, 3


def pattern_episodes(track, window):
    """Group identical W-position sequences and count unrelated occurrences.

    Occurrences closer together than 2*window are one event: that folds a long
    constant run (contiguous starts) and one loop zone (starts spaced by the
    cycle length) each into a single episode.
    """
    n_frames = len(track)
    blank = (track[:, 0] == 0) & (track[:, 1] == 0)
    blank_cumsum = np.concatenate([[0], np.cumsum(blank)])
    coords = np.ascontiguousarray(track[:, :2])

    starts_by_pattern = defaultdict(list)
    for start in range(n_frames - window + 1):
        # a window containing a no-detection frame is not a fill pattern
        if blank_cumsum[start + window] - blank_cumsum[start] > 0:
            continue
        starts_by_pattern[coords[start:start + window].tobytes()].append(start)

    merge_gap = 2 * window
    episodes = {}
    for key, starts in starts_by_pattern.items():
        count, previous = 1, starts[0]
        for start in starts[1:]:
            if start - previous > merge_gap:
                count += 1
            previous = start
        episodes[key] = count
    return starts_by_pattern, episodes


def build_mask(track, window=16, min_episodes=8):
    starts_by_pattern, episodes = pattern_episodes(track, window)

    attractors = {k for k, n in episodes.items() if n >= min_episodes}
    codes = np.full(len(track), CLEAN, np.uint8)
    blank = (track[:, 0] == 0) & (track[:, 1] == 0)

    invented = np.zeros(len(track), bool)
    for key in attractors:
        for start in starts_by_pattern[key]:
            invented[start:start + window] = True
    codes[invented] = INVENTED

    # blend contamination reaches window-1 frames either side of an evidence-free
    # run, because that is how far the overlapping fill windows blend
    halo = np.zeros(len(track), bool)
    padded = np.concatenate([[False], invented, [False]])
    edges = np.diff(padded.astype(np.int8))
    for start in np.where(edges == 1)[0]:
        halo[max(0, start - (window - 1)):start] = True
    for stop in np.where(edges == -1)[0]:
        halo[stop:min(len(track), stop + window - 1)] = True
    codes[halo & ~invented] = DEGRADED
    codes[blank] = NO_DETECTION

    return codes, episodes, attractors


def report(track, label, window=16, min_episodes=8):
    codes, episodes, attractors = build_mask(track, window, min_episodes)
    counts = np.bincount(codes, minlength=4)
    print(f'\n=== {label} (W={window}, K={min_episodes}) ===')
    for name, code in (('clean', CLEAN), ('invented', INVENTED),
                       ('no_detection', NO_DETECTION), ('degraded', DEGRADED)):
        print(f'  {name:13s} {counts[code]:7d}  {counts[code] / len(track):.4f}')

    ranked = sorted(episodes.values(), reverse=True)
    non_attr = sorted((n for n in episodes.values() if n < min_episodes),
                      reverse=True)
    print(f'  distinct patterns: {len(episodes)}   attractors: {len(attractors)}')
    print(f'  episode counts, top 8: {ranked[:8]}')
    print(f'  highest NON-attractor episode count: '
          f'{non_attr[0] if non_attr else 0}')
    attr_min = min((episodes[k] for k in attractors), default=0)
    print(f'  lowest attractor episode count      : {attr_min}')
    if non_attr and attr_min:
        print(f'  SAFETY MARGIN: {attr_min / max(non_attr[0], 1):.1f}x between '
              f'the quietest fabrication and the loudest real pattern')
    return codes, episodes


if __name__ == '__main__':
    old = np.load(f'{BASE}/pilot_track_npy/1.npy')
    new = np.load(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_weight.npy')

    old_codes, old_eps = report(old, 'OLD track, nonoverlap (loop shape)')
    new_codes, new_eps = report(new, 'NEW track, weight (constant shape)')

    # --- how does the threshold behave? the gap should be huge, not delicate
    print('\n=== threshold sensitivity: invented share vs K ===')
    for min_ep in (2, 4, 8, 16, 32, 64):
        oc, _, _ = build_mask(old, 16, min_ep)
        nc, _, _ = build_mask(new, 16, min_ep)
        print(f'  K={min_ep:3d}: old {(oc == INVENTED).mean():.4f}   '
              f'new {(nc == INVENTED).mean():.4f}')

    print('\n=== window-length sensitivity (K=8) ===')
    for win in (8, 12, 16, 24, 32):
        oc, _, _ = build_mask(old, win, 8)
        nc, _, _ = build_mask(new, win, 8)
        print(f'  W={win:3d}: old {(oc == INVENTED).mean():.4f}   '
              f'new {(nc == INVENTED).mean():.4f}')

    # --- agreement with the coordinate-matching sidecar it is meant to replace
    print('\n=== agreement with the hardcoded-coordinate sidecar ===')
    for stem, codes in (('pilot_nonoverlap', old_codes),
                        ('pilot_weight', new_codes)):
        old_side = np.load(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/{stem}_fillmask.npy')
        mine = codes == INVENTED
        theirs = old_side == 1
        both = (mine & theirs).sum()
        print(f'  {stem}: coordinate rule {theirs.sum()}, recurrence rule '
              f'{mine.sum()}, agree {both}')
        print(f'    recurrence-only {(mine & ~theirs).sum()}, '
              f'coordinate-only {(~mine & theirs).sum()}')

    np.save(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_weight_recurrence_mask.npy',
            new_codes)
    np.save(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_nonoverlap_recurrence_mask.npy',
            old_codes)
    print('\nwrote recurrence masks')
