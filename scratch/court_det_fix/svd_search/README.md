# SVD12 search depth

`run.py` measures three independent automatic G0 search depths on six frozen
views. It uses the live SVD12 generator, W5 parent measurement, refitter and
ranker. It does not add frames or change scoring. `RUN_READY.md` gives the
Carmack launch and input list.

Results are one compressed generation record and one compact candidate/result
record per case and arm. `build_gallery.py` turns a complete run into an offline
four-column comparison. The detector's selection and reference-agreement
oracles are labelled separately. Each oracle considers only generated or refitted
candidates in the ranker's selectable `provisional_rank`. The reference annotation
is used after ranking and does not affect generation or selection. All saved
candidate geometries remain available for diagnostics. The earlier W5 selection
comes from the wider W5 records listed in `saved_selected.json.gz`; its selected
key identifies whether it came from G0, G1 or the line-template population.

Total wall and CPU time starts before view preparation and fresh direction
estimation, and ends immediately after generation, measurement, refit and ranking.
Runtime/module setup, reference evaluation, reporting and serialisation are
excluded. This measures the search on prepared views, not end-to-end scene runtime.

The full run uses no smoke flag. A one-pair smoke is available through
`--max-matched-pairs 1` with a fresh output directory.
