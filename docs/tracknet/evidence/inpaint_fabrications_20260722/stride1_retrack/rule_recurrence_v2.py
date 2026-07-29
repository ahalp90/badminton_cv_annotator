"""Recurrence rule, second pass. Fixes two faults found in the first.

Fault 1: the episode threshold K was a guess. On the nonoverlap track any K
between 8 and 64 gives the same answer, but on the weight track K=8 sat inside a
continuum (attractors at 8-14 episodes against real patterns at 7), so the
safety margin collapsed to 1.1x. The counts are actually bimodal: 418 for the
fabrication, then 14 and below. So derive K from the largest ratio gap in the
sorted episode counts instead of naming it.

Fault 2: a fabricated run shorter than W never forms a full pattern, so v1
missed 1,107 constant frames the crude coordinate rule caught. Recover them
WITHOUT hardcoding: take the positions of the DISCOVERED attractors and mark
frames sitting on them, but at reduced confidence, since a real shuttle may
legitimately pass through such a pixel for a frame or two.
"""
import numpy as np

from rule_recurrence import (BASE, CLEAN, DEGRADED, INVENTED, NO_DETECTION,
                             pattern_episodes)


def adaptive_threshold(episodes, floor=2):
    """K from the largest ratio gap between consecutive episode counts.

    A fabrication attractor recurs orders of magnitude more often than any real
    pattern, so the counts are bimodal and the gap locates itself. `floor`
    ignores the noise of patterns seen once or twice.
    """
    counts = sorted({n for n in episodes.values() if n >= floor}, reverse=True)
    if len(counts) < 2:
        return (counts[0] if counts else 1), 1.0
    ratios = [(counts[i] / counts[i + 1], counts[i]) for i in range(len(counts) - 1)]
    best_ratio, threshold = max(ratios)
    return threshold, best_ratio


def build_mask(track, window=16):
    starts_by_pattern, episodes = pattern_episodes(track, window)
    threshold, margin = adaptive_threshold(episodes)
    attractors = {k for k, n in episodes.items() if n >= threshold}

    codes = np.full(len(track), CLEAN, np.uint8)
    invented = np.zeros(len(track), bool)
    for key in attractors:
        for start in starts_by_pattern[key]:
            invented[start:start + window] = True
    codes[invented] = INVENTED

    # blend contamination either side of an evidence-free run
    halo = np.zeros(len(track), bool)
    edges = np.diff(np.concatenate([[False], invented, [False]]).astype(np.int8))
    for start in np.where(edges == 1)[0]:
        halo[max(0, start - (window - 1)):start] = True
    for stop in np.where(edges == -1)[0]:
        halo[stop:min(len(track), stop + window - 1)] = True

    # positions the DISCOVERED attractors occupy; sitting on one without the
    # full pattern is suspicious but not proof, so it grades down not out
    attractor_positions = set()
    for key in attractors:
        for point in np.frombuffer(key).reshape(window, 2):
            attractor_positions.add((point[0], point[1]))
    on_attractor = np.zeros(len(track), bool)
    if attractor_positions:
        for position in attractor_positions:
            on_attractor |= ((track[:, 0] == position[0])
                             & (track[:, 1] == position[1]))

    codes[(halo | on_attractor) & ~invented] = DEGRADED
    codes[(track[:, 0] == 0) & (track[:, 1] == 0)] = NO_DETECTION
    return codes, threshold, margin, len(attractors)


if __name__ == '__main__':
    old = np.load(f'{BASE}/pilot_track_npy/1.npy')
    new = np.load(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_weight.npy')

    for track, label, stem in ((old, 'OLD nonoverlap (loop)', 'pilot_nonoverlap'),
                               (new, 'NEW weight (constant)', 'pilot_weight')):
        codes, threshold, margin, n_attr = build_mask(track)
        counts = np.bincount(codes, minlength=4)
        print(f'\n=== {label} ===')
        print(f'  derived K = {threshold} episodes, chosen at a {margin:.1f}x '
              f'gap   attractors: {n_attr}')
        for name, code in (('clean', CLEAN), ('invented', INVENTED),
                           ('no_detection', NO_DETECTION),
                           ('degraded', DEGRADED)):
            print(f'  {name:13s} {counts[code]:7d}  '
                  f'{counts[code] / len(track):.4f}')

        crude = np.load(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/{stem}_fillmask.npy')
        mine_hard = codes == INVENTED
        mine_any = (codes == INVENTED) | (codes == DEGRADED)
        theirs = crude == 1
        print(f'  vs coordinate rule: it flags {theirs.sum()}, '
              f'I flag {mine_hard.sum()} hard')
        print(f'    missed by me at hard confidence: '
              f'{(~mine_hard & theirs).sum()}')
        print(f'    of those, recovered as degraded: '
              f'{(~mine_hard & theirs & mine_any).sum()}')
        print(f'    still missed entirely          : '
              f'{(~mine_any & theirs).sum()}')
        np.save(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/{stem}_recurrence_v2.npy', codes)

    print('\nwrote v2 masks')
