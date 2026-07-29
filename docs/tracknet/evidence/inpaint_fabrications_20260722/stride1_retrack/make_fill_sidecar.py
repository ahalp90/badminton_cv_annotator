"""Build a per-frame fill sidecar for a TrackNetV3 track.

InpaintNet's evidence-free output takes one of two shapes, and both are
algorithmic, so both can be flagged without the discarded fill mask:

  shape A, nonoverlap mode: fill windows tile from frame 0, so a fully empty
    window emits the fixed 16-value cycle verbatim. Flag a window when 14 or
    more of its 16 y rows match the cycle exactly. A real shuttle cannot hit 14
    exact integer rows in the exact order on the exact lattice
  shape B, weight mode: windows overlap by one frame, so a deep-gap frame gets
    the tent-weighted blend of that same cycle: one constant position. Flag
    exact equality with it

Codes written per frame:
  0 clean        a position that is neither invented nor absent
  1 invented     shape A window, or shape B constant
  2 no_detection stored as (0,0); the tracker found nothing and nothing was
                 invented over it
  3 blended      within 15 frames of an invented run in weight mode. The tent
                 spans 16 frames, so these values are part real evidence and
                 part fixed cycle. Informed, but contaminated

Codes 1 and 3 are the "flag, not delete" split: 1 carries no evidence, 3 carries
some. Consumers decide per rule.
"""
import json
from collections import Counter

import numpy as np

BASE = ('/home/ariel/Documents/COSC594/badminton_stroke_classification'
        '/local_scratch/autograder_architecture')
OUT = f'{BASE}/inpaint_fabrications_investigation/stride1_retrack'

CLEAN, INVENTED, NO_DETECTION, BLENDED = 0, 1, 2, 3
BLEND_SPAN = 15   # tent covers 16 frames, so contamination reaches 15 out


def recover_cycle(track):
    """The 16 fixed cycle values, indexed by frame % 16.

    Only valid on a nonoverlap track, where fill windows tile from frame 0 so
    within-window index equals frame % 16.
    """
    y = track[:, 1]
    eq = y[:len(y) - 16] == y[16:]
    spans, start = [], None
    for i, matched in enumerate(eq):
        if matched and start is None:
            start = i
        elif not matched and start is not None:
            if i - start >= 32:
                spans.append((start, i))
            start = None
    looping = [(a, b) for a, b in spans if np.ptp(y[a:a + 16]) > 0]
    if not looping:
        raise ValueError('no repeating cycle found; is this a nonoverlap track?')
    frames = np.unique(np.concatenate(
        [np.arange(a, min(b + 16, len(track))) for a, b in looping]))
    cycle = np.zeros((16, 2))
    for phase in range(16):
        at_phase = frames[frames % 16 == phase]
        cycle[phase, 0] = Counter(track[at_phase, 0]).most_common(1)[0][0]
        cycle[phase, 1] = Counter(track[at_phase, 1]).most_common(1)[0][0]
    return cycle


def tent_weights(seq_len=16):
    """Mirrors get_ensemble_weight(seq_len, 'weight') in inference_utils.py."""
    weights = np.ones(seq_len)
    for i in range((seq_len + 1) // 2):
        weights[i] = i + 1
        weights[seq_len - i - 1] = i + 1
    return weights / weights.sum()


def predicted_constant(cycle):
    """Quantisation INTERVAL for the blend, not a single pixel.

    The stored cycle is already truncated to whole pixels, so each true value
    sits up to one pixel higher, and the pipeline truncates only once at the
    end. That widens the prediction to two candidates per axis.
    """
    weights = tent_weights()
    mean_x = (weights * cycle[:, 0]).sum() * 512
    mean_y = (weights * cycle[:, 1]).sum() * 288
    return ({int(mean_x), int(mean_x) + 1}, {int(mean_y), int(mean_y) + 1},
            mean_x, mean_y)


def runs_of(mask):
    spans, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def build_sidecar(track, cycle, mode):
    codes = np.full(len(track), CLEAN, np.uint8)
    blank = (track[:, 0] == 0) & (track[:, 1] == 0)
    codes[blank] = NO_DETECTION
    detail = {}

    if mode == 'nonoverlap':
        # whole-window rule: 14+ exact y matches against the cycle
        invented = np.zeros(len(track), bool)
        for start in range(0, len(track) - 15, 16):
            window = track[start:start + 16, 1]
            if (window == cycle[:, 1]).sum() >= 14:
                invented[start:start + 16] = True
        codes[invented & ~blank] = INVENTED
        detail['rule'] = '>=14 of 16 y rows match the cycle, per tiled window'

    elif mode == 'weight':
        x_set, y_set, mean_x, mean_y = predicted_constant(cycle)
        non_blank = Counter(map(tuple, track[~blank][:, :2]))
        (obs_x, obs_y), held = non_blank.most_common(1)[0]
        px, py = round(obs_x * 512), round(obs_y * 288)
        # fail loud: the busiest position must be the one the weights predict
        if px not in x_set or py not in y_set:
            raise ValueError(
                f'busiest position ({px},{py}) is outside the predicted '
                f'interval x in {sorted(x_set)}, y in {sorted(y_set)}. '
                f'Different checkpoint or mode?')
        invented = (track[:, 0] == obs_x) & (track[:, 1] == obs_y)
        codes[invented] = INVENTED
        for start, stop in runs_of(invented):
            lo = max(0, start - BLEND_SPAN)
            hi = min(len(track), stop + BLEND_SPAN)
            neighbourhood = np.zeros(len(track), bool)
            neighbourhood[lo:start] = True
            neighbourhood[stop:hi] = True
            codes[neighbourhood & (codes == CLEAN)] = BLENDED
        detail.update({
            'constant_px': [int(px), int(py)],
            'predicted_mean_px': [round(mean_x, 4), round(mean_y, 4)],
            'predicted_interval_px': {'x': sorted(x_set), 'y': sorted(y_set)},
            'held_frames': int(held),
            'rule': 'exact equality with the tent-blended cycle constant',
        })
    else:
        raise ValueError(f'unknown mode {mode}')

    return codes, detail


PROVENANCE = {
    'source_video': {
        'shuttleset_vid': 1,
        'match': 'Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals',
        'note': 'ShuttleSet video 1. No YouTube id recorded in local records; '
                'videos_full/pull_ledger.csv covers sset_21 only',
        'tracked_file': 'pilot_288p.mp4 (512x288, 25 fps, 154393 frames)',
        'tracked_file_md5': '20b7b2cf29ccfd853e542575d8031af3',
        'source_1080p_on_node':
            '/scratch/comp320a/ShuttleSet/raw_video/1 Kento_...Finals.mp4',
    },
    'checkpoints': {
        'tracknet_md5': '6540c256b1237cacdea3d05c16de8353',
        'inpaintnet_md5': '25aecc665050480a9bfb2fe2df275d14',
        'note': 'a different checkpoint changes the cycle; re-derive before use',
    },
    'codes': {
        '0': 'clean', '1': 'invented', '2': 'no_detection', '3': 'blended',
    },
}

if __name__ == '__main__':
    old = np.load(f'{BASE}/pilot_track_npy/1.npy')
    new = np.load(f'{OUT}/pilot_weight.npy')
    cycle = recover_cycle(old)

    manifest = dict(PROVENANCE)
    manifest['tracks'] = {}
    for track, mode, stem in ((old, 'nonoverlap', 'pilot_nonoverlap'),
                              (new, 'weight', 'pilot_weight')):
        codes, detail = build_sidecar(track, cycle, mode)
        np.save(f'{OUT}/{stem}_fillmask.npy', codes)
        counts = {name: int((codes == code).sum()) for name, code in
                  (('clean', CLEAN), ('invented', INVENTED),
                   ('no_detection', NO_DETECTION), ('blended', BLENDED))}
        shares = {k: round(v / len(codes), 4) for k, v in counts.items()}
        manifest['tracks'][stem] = {
            'eval_mode': mode, 'frames': int(len(codes)),
            'sidecar': f'{stem}_fillmask.npy',
            'counts': counts, 'shares': shares, **detail}
        print(f'--- {stem} ({mode}) ---')
        for name in ('clean', 'invented', 'no_detection', 'blended'):
            print(f'  {name:13s} {counts[name]:7d}  {shares[name]:.4f}')

    manifest['cycle_px'] = [[int(round(x * 512)), int(round(y * 288))]
                            for x, y in cycle]
    with open(f'{OUT}/fill_sidecar_manifest.json', 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f'\nwrote sidecars + fill_sidecar_manifest.json to {OUT}')
