# L2 scoring handoff

This is the expanded 17 September 2026 handoff for the realised L2 scope. The
initial handoff contained only the compact result and four image witnesses.
This folder now also carries the small comparison table, selected-candidate
witness records and the replay script for code review. Raw populations and
large matcher inputs remain outside the uploadable handoff.

## Scope and evidence

The fixed cases are GX0, Am2-150, Am2-28019 and Am3-0. G0 is the saved
baseline generation population. G1 is the saved `paint_observations`
generation population. S0 and S1 are the original and paint-filtered fragment
observations. The union is ordered G0 followed by G1.

The handoff contains 24 comparison cells. All eight saved diagonal winner
checks passed. The four diagnostic traces reproduce both the saved aggregate
profiles and the existing profile implementation. The diagnostic candidates
are evidence witnesses only. They are not added to the automatic union.

## Files

- `result.md`: compact interpretation and limitations.
- `comparison.csv`: winner, eligibility and nearest-court results for every cell.
- `witnesses.json`: selected-candidate records and complete diagnostic traces.
- `run_l2_scoring.py`: review copy of the bounded replay script.
- `witness_*.png`: native-pixel diagnostic overlays.

The canonical runnable script is `../../L2_scoring/run_l2_scoring.py` from this
folder's repository location. It expects the repository's saved inputs and
helpers, so this handoff is evidence-complete for review but not a standalone
rerun bundle.
