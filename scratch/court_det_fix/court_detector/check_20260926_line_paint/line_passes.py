"""Per-line contrast, strength and fragment support for three courts on each hand-marked view.

A line's strength is its contrast as a share of the view's strongest line, as line_paint.py (version 4)
scores it; versions 1-3 passed a line at a fixed bar instead, and their tables show * for a pass.

The courts are the blend arm's pick, the line-paint arm's pick and the eligible court closest to the hand
marks before refit. Errors are the largest hand-mark error in floor metres, before refit.

Usage, from the repository root: line_passes.py REPLAY_OUT UPRIGHT_RUN VIEW [VIEW ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from scratch.court_det_fix.court_detector import line_paint
from scratch.court_det_fix.court_detector.check_20260926_line_paint.compare_line_paint import (
    COURT_ROOT,
    largest_floor_error,
    read_gz,
    statistics,
)
from scratch.court_det_fix.court_detector.run_views import pack_sources
from scratch.court_det_fix.w5_holistic import verifier

MARKING_NAMES = ("left doubles", "left singles", "centre", "right singles", "right doubles", "baseline y0",
                 "long service y0", "short service y0", "short service y13", "long service y13", "baseline y13")


def main() -> None:
    replay_out, upright_run, views = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    manifest = statistics.read(statistics.MANIFEST)
    references = statistics.load_references(manifest)
    images = {row["case_id"]: row["image"] for row in manifest["cases"]}
    sources, provenances, _ = pack_sources(verifier.CASE_PACKS)
    for view in views:
        landmarks = references[view]["landmarks"]
        frame = cv2.imread(str(COURT_ROOT / images[view]))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # The working image is the frame shrunk to at most 960 px on its longer side.
        native_per_working = np.full(2, max(1.0, max(frame.shape[:2]) / 960))
        boxes = np.asarray(sources[view]["bbox_px"], dtype=float).reshape(-1, 4)
        if not provenances[view].has_same_image_boxes:
            boxes = np.empty((0, 4))
        record = read_gz(upright_run / "artefacts" / f"{view}.json.gz")["w5"]["record"]
        candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
        rows = read_gz(replay_out / "blend" / "artefacts" / f"{view}.json.gz")["net_choice"]["rows"]
        errors = {row["origin_key"]: largest_floor_error(candidates[row["origin_key"]]["homography_working"],
                                                         landmarks, native_per_working[0]) for row in rows}
        courts = {"blend pick": read_gz(replay_out / "blend" / "artefacts" / f"{view}.json.gz")["net_choice"]["chosen"],
                  "line-paint pick": read_gz(replay_out / "line_paint" / "artefacts" / f"{view}.json.gz")[
                      "net_choice"]["chosen"],
                  "closest court": min(errors, key=lambda key: errors[key])}
        contrasts = {row["origin_key"]: line_paint.court_contrasts(
            np.asarray(candidates[row["origin_key"]]["homography_working"]), grey, native_per_working, boxes)
            for row in rows}
        reference = line_paint.view_reference(list(contrasts.values()))
        print(f"\n{view}: per line, contrast in grey levels / strength / fragment support; the view's strongest "
              f"line reads {reference:.1f}")
        print("line\t" + "\t".join(f"{name} ({errors[key]:.2f} m)" for name, key in courts.items()))
        for marking in range(len(line_paint.MARKING_INTERVALS)):
            cells = []
            for key in courts.values():
                support = candidates[key]["evidence"]["markings"][marking]["q_geom"]
                contrast = contrasts[key][marking]
                if support is None:
                    cells.append("unseen")
                elif contrast is None:
                    cells.append(f"unmeasured / {support:.2f}")
                else:
                    cells.append(f"{contrast:6.1f} / {max(0.0, contrast) / reference:.2f} / {support:.2f}")
            print(f"{MARKING_NAMES[marking]}\t" + "\t".join(cells))
        scores = []
        for name, key in courts.items():
            candidate = candidates[key]
            paint = line_paint.court_paint(candidate["evidence"], contrasts[key], reference)
            scores.append(f"{name}: W5 paint {candidate['evidence']['q_paint10_span_weighted']:.3f}, "
                          f"line paint {paint:.3f}, geometry {candidate['evidence']['q_geom_span_weighted']:.3f}")
        print("\n".join(scores))


if __name__ == "__main__":
    main()
