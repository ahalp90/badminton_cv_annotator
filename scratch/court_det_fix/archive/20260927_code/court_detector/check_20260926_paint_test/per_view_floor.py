"""Does bare-floor contrast differ enough between views for a per-view pass bar to matter?

For each hand-marked frame: the 90th percentile of bare-floor contrast, and pass rates with the
global bar of 9 and with that per-view bar. Usage, from the repository root: python per_view_floor.py
"""
import cv2
import numpy as np

from experiments.annotator.independent_court.paint_geometry import CENTRE_SEGMENTS_M
from scratch.court_det_fix.court_detector.check_20260926_paint_test import pass_bar
from scratch.court_det_fix.court_detector.run_views import pack_sources
from scratch.court_det_fix.w5_holistic import verifier

statistics = pass_bar.statistics
manifest = statistics.read(statistics.MANIFEST)
references = statistics.load_references(manifest)
sources, provenances, _ = pack_sources(verifier.CASE_PACKS)
GLOBAL_BAR = 9.0
print('view\tfloor p90\tfloor pass @9\tlines pass @9\tfar pass @9\t| per-view bar: lines pass\tfar pass\tfloor pass')
totals = {'global': [], 'view': []}
for row in manifest['cases']:
    view = row['case_id']
    landmarks = references.get(view, {}).get('landmarks')
    if not landmarks:
        continue
    homography, _ = cv2.findHomography(np.array([m['court_m'] for m in landmarks], float),
                                       np.array([m['image_px'] for m in landmarks], float))
    frame = cv2.imread(str(pass_bar.COURT_ROOT / row['image']))
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    npw = max(1.0, max(frame.shape[:2]) / 960)
    boxes = np.asarray(sources[view]['bbox_px'], float).reshape(-1, 4) if provenances[view].has_same_image_boxes else np.empty((0, 4))
    lines = [pass_bar.contrasts(grey, homography, line_m, verifier.parallel_neighbours_m(line_m), boxes, npw)
             for line_m in CENTRE_SEGMENTS_M]
    far = np.concatenate([lines[index] for index in pass_bar.FAR_LINES])
    lines = np.concatenate(lines)
    floor = np.concatenate([pass_bar.contrasts(grey, homography, line_m,
                                               verifier.LENGTHWISE_CENTRES_M if line_m[0, 0] == line_m[1, 0]
                                               else verifier.TRANSVERSE_CENTRES_M, boxes, npw)
                            for line_m in pass_bar.midway_lines_m()])
    view_bar = float(np.quantile(floor, 0.9))
    for name, bar in (('global', GLOBAL_BAR), ('view', view_bar)):
        totals[name].append(((lines >= bar).mean(), (floor >= bar).mean()))
    rate = lambda values, bar: f'{(values >= bar).mean():.2f}' if len(values) else '-'
    print(f'{view}\t{view_bar:.1f}\t{rate(floor, 9)}\t{rate(lines, 9)}\t{rate(far, 9)}\t| {rate(lines, view_bar)}\t{rate(far, view_bar)}\t{rate(floor, view_bar)}')
for name, pairs in totals.items():
    line_rates, floor_rates = np.array(pairs).T
    print(f'{name} bar, mean over views: lines {line_rates.mean():.3f}, floor {floor_rates.mean():.3f}, difference {(line_rates - floor_rates).mean():.3f}')
