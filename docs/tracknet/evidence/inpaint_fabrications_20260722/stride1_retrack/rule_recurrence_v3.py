"""Recurrence rule v3: my discovery mechanism, Sol's epistemic split.

Sol's decisive point, which v2 got wrong. v2 called both fabrication shapes
"invented" because both are improbable-recurrence attractors. They are not
equally provable:

  - a VARYING 16-position motif recurring hundreds of times is proof. Footage
    cannot reproduce 16 exact moving positions on many unrelated occasions
  - a FLAT constant recurring hundreds of times is strong suspicion, not proof.
    A stationary object CAN hold one pixel, and if TrackNet locks onto a static
    image feature the frame is a genuine detection that InpaintNet passes
    through untouched (predict.py:301). No function of (x, y, visibility) can
    separate that from a fill, because provenance is discarded

So the flat attractor keeps a separate code and stays usable by default.
Destroying it would breach the constraint that a resting shuttle must survive.

Kept from v2: attractors discovered per track rather than hardcoded, the
threshold derived from the largest ratio gap rather than hand-set, and scanning
every offset rather than only the 16-frame lattice, which is what let v2 find
1,256 frames the lattice-aligned rules miss.

Added from Sol: split-half validation as a standing check.

Codes renumbered 2026-07-22: no-detection folds into 0; fabricated/suspect/
degraded are 1/2/3 (was 0-4 with no_detection=2).
"""
import numpy as np

from rule_recurrence import BASE, pattern_episodes
from rule_recurrence_v2 import adaptive_threshold

NO_FLAG, FABRICATED, SUSPECT_FLAT, DEGRADED = 0, 1, 2, 3
NAMES = {NO_FLAG: 'no flag', FABRICATED: 'fabricated (proof)',
         SUSPECT_FLAT: 'suspect_flat (keep)', DEGRADED: 'degraded'}


def discover(track, window=16):
    """Return attractor patterns split by whether they vary or sit still."""
    starts_by_pattern, episodes = pattern_episodes(track, window)
    threshold, margin = adaptive_threshold(episodes)
    varying, flat = {}, {}
    for key, count in episodes.items():
        if count < threshold:
            continue
        points = np.frombuffer(key).reshape(window, 2)
        moves = np.ptp(points[:, 0]) > 0 or np.ptp(points[:, 1]) > 0
        (varying if moves else flat)[key] = starts_by_pattern[key]
    return varying, flat, threshold, margin


def build_mask(track, window=16):
    varying, flat, threshold, margin = discover(track, window)
    codes = np.full(len(track), NO_FLAG, np.uint8)

    def cover(groups):
        hit = np.zeros(len(track), bool)
        for starts in groups.values():
            for start in starts:
                hit[start:start + window] = True
        return hit

    proven = cover(varying)
    suspect = cover(flat)

    # blend contamination reaches window-1 frames either side of any attractor
    core = proven | suspect
    halo = np.zeros(len(track), bool)
    edges = np.diff(np.concatenate([[False], core, [False]]).astype(np.int8))
    for start in np.where(edges == 1)[0]:
        halo[max(0, start - (window - 1)):start] = True
    for stop in np.where(edges == -1)[0]:
        halo[stop:min(len(track), stop + window - 1)] = True

    # sitting on a discovered attractor coordinate without forming the full
    # pattern: suspicious, never proof, so it grades down rather than out
    positions = set()
    for key in list(varying) + list(flat):
        for point in np.frombuffer(key).reshape(window, 2):
            positions.add((point[0], point[1]))
    on_attractor = np.zeros(len(track), bool)
    for pos_x, pos_y in positions:
        on_attractor |= (track[:, 0] == pos_x) & (track[:, 1] == pos_y)

    codes[(halo | on_attractor) & ~core] = DEGRADED
    codes[suspect] = SUSPECT_FLAT
    codes[proven] = FABRICATED
    # blank frames stay unflagged even under a halo: the guard has nothing to
    # say there, and the track's own visibility column already records them
    codes[(track[:, 0] == 0) & (track[:, 1] == 0)] = NO_FLAG
    return codes, dict(threshold=threshold, margin=margin,
                       n_varying=len(varying), n_flat=len(flat))


def split_half_check(track, window=16):
    """Sol's validation: attractors derived on one half must hold on the other."""
    midpoint = len(track) // 2
    halves = [track[:midpoint], track[midpoint:]]
    keysets = []
    for half in halves:
        varying, flat, _, _ = discover(half, window)
        keysets.append((set(varying), set(flat)))
    (v_a, f_a), (v_b, f_b) = keysets
    print(f'  split-half: varying {len(v_a)} vs {len(v_b)}, '
          f'shared {len(v_a & v_b)}')
    print(f'              flat    {len(f_a)} vs {len(f_b)}, '
          f'shared {len(f_a & f_b)}')
    agree = (not (v_a ^ v_b)) and (not (f_a ^ f_b))
    print(f'              halves agree exactly: {agree}')


if __name__ == '__main__':
    old = np.load(f'{BASE}/pilot_track_npy/1.npy')
    new = np.load(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/pilot_weight.npy')

    for track, label, stem in ((old, 'OLD nonoverlap (loop)', 'pilot_nonoverlap'),
                               (new, 'NEW weight (constant)', 'pilot_weight')):
        codes, info = build_mask(track)
        counts = np.bincount(codes, minlength=4)
        print(f'\n=== {label} ===')
        print(f'  derived K={info["threshold"]} at a {info["margin"]:.1f}x gap; '
              f'{info["n_varying"]} varying, {info["n_flat"]} flat attractors')
        for code in (NO_FLAG, FABRICATED, SUSPECT_FLAT, DEGRADED):
            print(f'  {NAMES[code]:22s} {counts[code]:7d}  '
                  f'{counts[code] / len(track):.4f}')
        split_half_check(track)
        np.save(f'{BASE}/inpaint_fabrications_investigation/stride1_retrack/{stem}_recurrence_v3.npy', codes)

    print('\nwrote v3 masks')
