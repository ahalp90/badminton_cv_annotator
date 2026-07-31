# Annotator measurement history

Chronological record of the contact-detection and rally-segmentation
measurement campaigns behind today's annotator. Every headline number below
was measured against the pre-W2.9 sticky build; treat them as historical
context, not current chain behaviour. The current live reference is
`tests/data/annotator_calibration/reference/`, generated after W2.9 by
`annotator.calibration.gt_scoring --capture`. The W2.9 behavioural delta
is preserved in
`docs/architecture_notes/completed_general_refactors/annotator_cleanup/w2_9_delta.diff`.

Historical fixture aliases (`pilot`, `vid15`, `sset21`) survive in quoted
tables and CSVs; canonical stems `sset_01` / `sset_15` / `sset_21` are used
in surrounding prose. Fixture identity is derived from `Fixture.name` in
`src/annotator/calibration/fixtures.py`.

## Ground-truth substrate

Contact ground truth comes from ShuttleSet per-stroke rows: sset_01 carries
1,641 annotated strokes over 113 rallies (`pilot`), sset_15 carries 824
strokes over 104 rallies (`vid15`), and sset_21 carries 663 strokes over 75
rallies (`sset21`). Pooled measurements combine sset_01 and sset_15;
sset_21 was added later and appears alongside the pair rather than inside
the pool.

Scoring uses the fps-scaled canonical tolerance
(`annotator.calibration.gt_scoring.canonical_tolerance`, base-30 "5" band,
so 4 frames at 25 fps and 5 frames at 30 fps). The `+/-10` window in the
older campaign notes is the ruled usability yardstick for pooled recall /
precision; wider tolerances flatter, because the median gap between GT
strokes is 21–24 frames.

## S26–S27 (mid-2026): impulse finder plus body-unit gate

S26 ran a 33-arm end-to-end matrix on the two labelled videos. The
composed chain of an impulse finder (dimensionless velocity change divided
by a local median floor, window 12, multiple 4), a body-unit wrist gate at
1.4 bbox heights, and gate-first suppression dominated everything else:
recall lifted 60.9 → 82.4%, precision 54.0 → 70.1%, exact-stroke-count
rallies roughly doubled (7.8 → 16.1%). S27 promoted the arm from harness to
pipeline with an MD5 acceptance pin proving harness-pipeline reproduction.

Source: the former `s33_drafts/campaign_results.md`, preserved through
`local_scratch/autograder_architecture/NOW_TRACKED_MAP.md`.

## S28 (mid-July 2026): sticky-anchor swap and cell census

S28 swapped the candidate source to the sticky-anchor tracker and measured
four (impulse-multiple, suppression-radius) cells. The chain that shipped
into the pre-W2.9 sticky build was m4 / r9. Pooled recall / precision / F1
at +/-10:

| cell | recall | precision | F1 |
|---|---:|---:|---:|
| m4 / r9 (shipped in pre-W2.9 build) | 0.8296 | 0.7120 | 0.7664 |
| m4 / r7 | 0.8385 | 0.6983 | 0.7620 |
| m2 / r9 | 0.9428 | 0.5941 | 0.7289 |
| m2 / r7 | 0.9574 | 0.5560 | 0.7034 |

The r7-versus-r9 default was settled by a pre-agreed rule: whichever wins
pooled F1, with near-ties (within 0.2 points) staying r9. r9 won by 0.44
points.

End-to-end at S28 (pooled sset_01 + sset_15, 30-frame tolerance from the
S28 console): rally coverage 0.899 (195/217), hit timing 0.850, exact
stroke counts on 16.6% of covered rallies, striker 0.513, server 0.518,
rally winner 0.526 on covered. sset_01 covered 0.982 with 4 spurious
spans; sset_15 covered 0.808 with 54. sset_15 was consistently harder.

A census of the 435 remaining misses found 69% were "no candidate at all":
the shuttle track offered nothing near the stroke. The finder, not the
filters, is the recall wall. Two precision signals surfaced from that
census: `burst_ratio` (shuttle speed out over speed in) and
`post-flag visible run` (how long the shuttle stays visible after the
flag).

Source: the former `s33_drafts/campaign_results.md`; end-to-end console record
(`h_end_to_end_console_s28.txt`, gitignored).

## S29 (2026-07-16): the 95/95 sweep

S29 measured whether `burst_ratio` and `visible-run` post-filters, applied
to the m2 recall flood, could reach the standing 95% recall / 95% precision
target. **The target was not reached; the gap is structural.**

At the only cell that holds 95% pooled recall (m2 / r7), the best
precision any shape buys at that recall is **0.5684, against a no-filter
baseline of 0.5560** (five shapes land within 0.3 points of each other, so
shape choice is immaterial at this operating point). Burning recall down
to 0.80 lifts precision to 0.6219.

Two structural findings from that sweep:

- **Thresholds transfer; videos do not.** LOVO recall gaps were 0.01–0.06,
  but precision moved 0.19–0.24 points with the video (pilot 0.64,
  vid15 0.45 at the headline point). No global threshold pair makes the
  two videos equivalent; precision follows each video's junk load.
- **The two signals are near-independent but jointly too weak here.** At
  the 95-pooled point the kill sets barely intersect (22 fail burst only,
  62 fail run only, 0 both). Junk-side Spearman between the signals is
  ~0.02, so stacking them is right in principle; the residual junk that
  survives the sticky gate mostly looks like a true hit on both.

Wilson 95% intervals at the headline: pooled precision [0.550, 0.584],
pooled recall [0.941, 0.958] — the recall CI spans the 0.95 line, so treat
the operating point as a grid pick, not a population guarantee.

**S29 did not reach 95/95.** **S34 was killed 2026-07-28 before build**:
74 of 2,045 true contacts for 171 junk was ruled no real value; the ruling
predates any post-W2.9 remeasurement.

Source: the former `s29_sweep_readout.md`, preserved through
`local_scratch/autograder_architecture/NOW_TRACKED_MAP.md`;
kill record `records/s34_r7_session_end_20260728.md` (gitignored).

## B5 (mid-July 2026): amateur-footage scoping

Three instructional YouTube videos (25 / 50 / 60 fps) went through the
full extraction chain overnight as a proof of concept. Two components were
broadcast-only:

- The court detector's confidence gate passed 3.0% / 0.0% / 0.3% of sampled
  frames on the three videos; peaks sit at 0.002–0.005, so a scrape preset
  cannot be a looser broadcast preset.
- Rally segmentation did not meaningfully segment casual play. On one
  amateur video it produced 2 spans of 237 s each covering 97.7% of the
  video with over a thousand raw contacts per span. The 90-frame end-rest
  constant assumes broadcast-style stillness between rallies (and at 50
  fps means 1.8 s, not its designed 3.6 s), and continuous casual play
  never offers it.

Cost profile (proof-of-concept only, not a production benchmark): pose is
67% of extraction wall time, shuttle tracking 26%; whole chain 0.38×
realtime on an A100. The commentary lane worked cleanly: 117 kept chunks,
131 of 234 (chunk, model) cells scored with zero failures.

Source: the former `s33_drafts/campaign_results.md` § B5, preserved through
`local_scratch/autograder_architecture/NOW_TRACKED_MAP.md`.

## Post-W2.9 boundary (2026-07-28 onward)

W2.9 shipped `SmoothingMode.IGNORE_INVISIBLE`, base-30 demotion bound
`75`, `ReentryGuardVariant.TWO_SIDED`, and re-entry buffer `0.05` together
in `BaseAnnotatorConfig` (`eee3e29`). Contact F1 improved on all three
fixtures; landing and getpoint moved mixed. The full pre/post surface is
`docs/architecture_notes/completed_general_refactors/annotator_cleanup/w2_9_delta.diff`.

`tests/data/annotator_calibration/reference/` is the live capture after
that flip. Commit `85b8751` re-pinned its predecessor, and W3.1 refreshed
the fixture names. The tracked reference replaces every current-behaviour
claim above. The four
frozen S28 CSV pins under `scripts/archive/autoseg_trials/` (r7/pilot,
r7/vid15, r9/pilot, r9/vid15) are the authoritative artefacts for the
pre-W2.9 sticky-build measurements they represent; do not attempt to
re-derive their numbers from the current tip.

## Related historical measurements

- **Serve-miss scope, 2026-07-23.** 136 GT serves missed by the pre-W2.9
  chain across the three fixtures. Historical CSVs and canonical mapping
  live under
  `docs/scraper_pipeline/evidence/serve_prepend/historical_20260723/`; the
  archived handover context is `docs/archive/serve_prepend_lookback.md`.
  Superseded by
  the future rerun spec in `local_scratch/autograder_architecture/TODO.md`.
- **Inpaint fabrications, 2026-07-22.** Roughly a third of every reference
  shuttle track is invented fill. Full evidence pack at
  `docs/tracknet/evidence/inpaint_fabrications_20260722/`; consumer state
  and open work at `docs/tracknet/inpaint_sidecar_consumption.md`;
  producer contract at `docs/tracknet/inpaint_sidecar.md`.

## Retirement note

The S28 sticky-anchor pin harness does not import at the current tip: W2.1
removed the scorer surfaces it uses, W2.2 archived its yardstick
dependencies, and W2.7 replaced the `CourtBox` shape it constructs. Do not
treat it as a runnable current gate. The clean GT-injected regression
harness is a separate TODO, designed in
`docs/scraper_pipeline/annotator_regression_harness.md`.
