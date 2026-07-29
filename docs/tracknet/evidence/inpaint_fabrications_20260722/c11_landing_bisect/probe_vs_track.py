"""Diff the checkpoint probe cycle against the saved fixture tracks.

The probe values below are the recorded output of
probe_inpaint_cycle.py (bourbaki, 2026-07-22): the InpaintNet
checkpoint's response to one fully-missing 16-frame window, after
predict.py:60's int conversion. If the mechanism in
inpaint_source_findings.md is right, every fully-missing window in
a saved track holds probe[f mod 16] at frame f, because non-overlap
windows tile from frame zero.

Recorded results (laptop, 2026-07-22), loop windows = windows
matching the probe y cycle at 14+ of 16 positions:

    pilot   3,289 of 9,649 windows (34.1%)
    vid15   3,419 of 9,342 windows (36.6%)
    sset21  2,080 of 6,271 windows (33.2%)

In every video the track equals the probe at all 16 phases in y and
15 of 16 in x, unanimously. The sole deviation is the same in all
three: at phase 5 the probe gives x=245 while every saved window
holds 244. The production GPU passes and this CPU probe land on
opposite sides of the 245.0 float boundary before int() truncates.
"""
import numpy as np

probe_x = np.array([240, 244, 244, 245, 242, 245, 244, 244,
                    245, 246, 242, 240, 241, 241, 241, 247])
probe_y = np.array([81, 73, 70, 69, 67, 68, 69, 70,
                    68, 68, 71, 73, 77, 80, 83, 84])

BASE = ('/home/ariel/Documents/COSC594/badminton_stroke_classification/'
        'local_scratch/autograder_architecture/')
TRACKS = [
    ('pilot', 'pilot_track_npy/1.npy'),
    ('vid15', 'vid15_track_npy/15.npy'),
    ('sset21', 'sset21_track_npy/21.npy'),
]

for name, rel_path in TRACKS:
    arr = np.load(BASE + rel_path)
    # Saved values are int pixels divided by 512/288; rint recovers the ints.
    x_px = np.rint(arr[:, 0] * 512).astype(int)
    y_row = np.rint(arr[:, 1] * 288).astype(int)

    phase = np.arange(len(arr)) % 16
    match_y = y_row == probe_y[phase]
    match_xy = match_y & (x_px == probe_x[phase])

    # Window-level view; the trailing partial window falls outside the reshape.
    n_win = len(arr) // 16
    xw = x_px[:n_win * 16].reshape(n_win, 16)
    yw = y_row[:n_win * 16].reshape(n_win, 16)
    loop_wins = (yw == probe_y).sum(axis=1) >= 14

    print(f'{name}: {len(arr)} frames, phase-locked y match {100 * match_y.mean():.1f}%, '
          f'xy {100 * match_xy.mean():.1f}%')
    print(f'  windows: {n_win}, loop windows (>=14/16 y match): {loop_wins.sum()} '
          f'({100 * loop_wins.sum() / n_win:.1f}%)')
    for p in range(16):
        ys, yc = np.unique(yw[loop_wins, p], return_counts=True)
        xs, xc = np.unique(xw[loop_wins, p], return_counts=True)
        y_mode, y_n = max(zip(ys, yc), key=lambda t: t[1])
        x_mode, x_n = max(zip(xs, xc), key=lambda t: t[1])
        y_unanimous_match = y_mode == probe_y[p] and y_n == loop_wins.sum()
        x_unanimous_match = x_mode == probe_x[p] and x_n == loop_wins.sum()
        if not (y_unanimous_match and x_unanimous_match):
            print(f'  p{p:2d} deviates: probe y={probe_y[p]} x={probe_x[p]}, '
                  f'track y {y_mode}x{y_n}, x {x_mode}x{x_n}')
    print('  (all phases not listed match the probe unanimously)')
