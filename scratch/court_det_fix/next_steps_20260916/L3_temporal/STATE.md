# L3 temporal opportunity — resume state

**Next action:** Hand the completed L3 packet to W3. Do not rerun the scorer.

**Inputs / outputs:** Task prompt at
`scratch/court_det_fix/worklog/webui_further_followups_16092026/prompts/L3_cached_temporal_opportunity.md`;
source evidence under `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/`,
`experiments/annotator/independent_court/`, and `scratch/court_det_fix/next_steps_20260916/`;
all L3 outputs belong under `scratch/court_det_fix/next_steps_20260916/L3_temporal/`.

**Completed:** Recorded basis HEAD `820597ffcc02dd143d16a60448ad4ce53391338b` and used
one process. The frozen/local pack MD5 values match. The runner produced 30 courts × 7
frames, all 210 scores `ok`, exact origin diagonals, six successful registrations and
six inspected overlays. Scoped Python compile, Ruff and Serena/Pyrefly diagnostics pass.

**Current result:** Sampled alignment supports a common pixel homography. Line rankings
change across separated frames, but the shared median winner is the wrong 106:93818
alternative (post-lock reference median 799.36 working px). Classify as discriminating
cross-frame evidence seen, with wrong alternatives still dominating this matrix.

**Unresolved question:** Whether a fixed, reference-blind combination of the existing line
and paint cues can promote the low-error 22:* geometry without per-frame hand selection.

**Scope update:** The user authorised expanding the scoring sample for representation.
Use all seven cached GX frame indices (0, 5, 689, 5111, 5766, 77876, 86088), while
retaining one video, two origin frames and at most 32 automatic courts.

**Outputs:** `result.md`, `W3_PACKET.md`, `worklog.md`, `source_manifest.json`,
`score_matrix.{json,csv}`, `view_alignment.json`, and six `alignment_0_to_*.png` overlays.
