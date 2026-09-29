# Court-detector evaluation review packet (22 September 2026)

This is a bounded GitHub-readable evidence packet for the completed development evaluation. It keeps readable case tables, exact selected geometry in small `.json.gz` records, canonical raw-frame links, and lossy visual previews. The JPEGs are review previews, not replacement source data.

## Read in this order

- [W5 whole-court selections](w5.md): all 30 distinct final arm winners, saved gate/source ablations, failure alternatives and full-frame rulings.
- [G0/G1 crossed comparison](g0_g1.md): all 27 cases, six crossed pool/scorer cells, nearest-versus-selected distances and all 122 source-qualified selected-origin previews.
- [Corrected person masks](person_masks.md): all five stage comparisons and the saved diagnostic gallery.
- [Temporal union evaluation](temporal.md): native/common/shared choices for GX and Am3, exact scores and anchor homographies.

## Raw inputs and source records

The 27 unannotated detector input images are linked in [g0_g1.md](g0_g1.md) and live under `scratch/court_det_fix/frozen_views/frames/`: 10 broadcast scene medians and 17 single-frame views. The packet does not duplicate them.

For fitting accuracy, use the committed [original amateur annotations](../../../../data/amateur_court_corners/README.md), [September additions](../../../../data/amateur_court_corners/2026-09-08/README.md), and [GX clicked landmarks](../../../../data/amateur_court_corners/2026-09-09/hand_corners_landmarks.csv). Visible clicks and extrapolated corners have different meanings. Expanded frozen references are descriptive aids, not independently approved truth.

[W5 winner coordinates and homographies](w5/geometry.md) are readable directly on GitHub. Compressed records retain the remaining numerical detail. Post-hoc gate/source-ablation rows retain IDs, scores and overlays; 40 rows lack geometry in the compact source review, so their exact corners are not exported here. The full local case records retain them.

The small packet records link to the saved G0/G1, W5, person-mask and temporal source outputs. Those links are repository-relative and resolve in the published checkout. The multi-gigabyte W5 arrays, case records, rankings and caches, plus the large G0/G1 populations, remain local-only because the tables and selected geometry already expose the outcomes needed for independent review.

The panel is development data, and W5 usability is qualitative. Nearest-only G0/G1 distances do not retain a nearest-origin ID in the saved summary. This packet supports independent review of outcomes; it is not a self-contained rerun of the full detector search.

To rebuild the export, run `build_packet.py` with the complete local source packets present. It copies results and converts existing overlays; it does not run the detector or change the source evidence. JPEG quality is 95 for W5/Am3, 85 for G0/G1 and 90 for person-mask previews. The seven GX far-end sheets remain PNGs.
