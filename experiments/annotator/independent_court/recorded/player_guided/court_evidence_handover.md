# Court evidence investigation: session handover

The fixed-budget seed comparison is complete. Keep the original sampler: the
new rule improves generated coverage on five of seven GX frames, accepts none,
and regresses on both previously accepted controls. Read
[seed selection](seed_selection.md) for results and replay instructions.

The earlier raster-preserving experiment passes the usable GX frame-5 court
and rejects both known wrong courts. It also rejects one established good
Amateur-3 fit. That result remains diagnostic evidence, not a replacement scorer.

The committed and pushed experiment checkpoint is
`58b2f47a30c5459bd33fbf80b9228832eb0c3da9` on `fix/court-det`. Production and the original detector remain unchanged.

## Start here

Update, 13 September: the user inspected the four disputed Amateur-3 fragments.
Raw 0/62/84 are definitely not court lines, probably sunlight through windows.
Raw 108 follows the outer right court line with a slight inset. None supports
the far short-service marking. The old scorer credited incidental crossings;
these matches do not justify widening the direction tolerance.

Read the mandatory repository instructions and
[raster court matching](raster_court_matching.md). That report contains the one
result table, fixed comparison, decisive matches and limitations. Its
[replay archive](raster_court_matching.zip) supplies all portable inputs/results,
producer scripts, saved checks and replay instructions. No broad history reread
is needed to recover this checkpoint.

The experiment holds 63 courts across 22 frames fixed. The first four arms
separate original-family/projected-direction fragment access from
original-family/all-fragment merged pools. The fifth removes fragment filtering
at the same all-fragment pool. All five keep original raster distances, corners,
thresholds, residual matching and the 32-line merge cap. No merged-line angle
filter is added. The all-fragment pool can lose lines through merging and its cap.

## Verified findings and retained judgements

- Combined access passes GX references at frames 0, 5 and 5111, plus the saved
  frame-5 complete-search court. Neither access change alone passes those courts.
  The saved frame-5 court needs both restored fragment support and the broader
  merged pool to reach three distinct lengthwise lines.
- The user considers that specific frame-5 geometry usable. Left and bottom
  follow the outside of the paint; the mild right-edge inset is acceptable.
  Full right-side paint width is a preference, not a usability requirement.
  See the right-hand court in the existing
  [full-court comparison](gx_trace_overlays/gxBQ_window_00_frame_5__cap.jpg).
- Amateur-3 frame 0, stripe_legacy_0000_original, passes the original raster rule
  but fails directional fragment access under either merged pool. Far short-service
  support falls from 17/24 to 11/24, crosscourt distinct lines from three to two,
  and crosscourt mean from 0.6875 to 0.5486. Fragments 0, 62, 84 and 108 supply the
  six lost samples; their angles differ from that marking by about 9–13 degrees.
- The wrong broadcast junction court in scene 0017 fails directional access with
  lengthwise mean 0.5486, just below 0.55. Unrestricted access passes it. The wrong
  GX court fails either access change. Its Lines rejection loses two matched
  groups at the shared 32-line cap, not through better geometric discrimination.
- Seven older reference courts retain their passes. Five of six saved good stripe
  fits pass combined access. All 24 placements on eight non-court frames reject
  in every arm. These placements do not measure a search false-acceptance rate.
- All 63 baseline controls match the original scorer and previous saved raster
  support/count/score fields exactly. Corners and settings stay fixed. Six
  decisive intervals have exact raw-fragment/union-raster coverage agreement.
- Frame-0 legacy refit remains a useful but visibly imperfect start: the user
  reported top-left undershoot and bottom-right overshoot. The physical refit
  was marginally worse. Neither geometry received the frame-5 usability judgement.

## State after the 13 September comparisons

The disputed-fragment question is resolved. The archived inspect_matches.py
and decisive_matches.json.gz retain the exact coordinates for case
am3_window_00_frame_0, court stripe_legacy_0000_original, interval
far_short_service (index 8).

The subsequent attribution also explains the reference's 14/24 directional
short-service samples versus the stripe fit's 11/24. Four samples move outside
the four-pixel radius of raw 186; one sample moves inside the radius of raw 97.
See the updated [result report](raster_court_matching.md) for the distances.
This explains the numerical difference through sample placement. It does not
establish every retained match's identity or missing visibility.

The seed comparison kept the original scorer and gates in both arms. The pilot
on GX frames 0 and 5 was extended unchanged to the other five GX frames and two
accepted controls. The new sample loses the Amateur-2 output and worsens the
broadcast selected maximum corner error from 10.31 to 54.02 px at 1280×720.
Do not adopt this rule or combine it with a scorer change on these results.

The user judged the new closest frame-5 proposal a significant regression from
the earlier approved complete-search court. No particular edge was identified.
The visual comparison is complete; do not treat the numerical coverage gain as
recovery of the approved geometry. Missing
far-centre support still does not establish occlusion. No new experiment is queued.

The optional refit from the approved complete-search frame-5 court is also
complete. Legacy/physical boundary RMS moves from 1.47 to 1.01/1.35 px, while
maximum corner error rises from 3.37 to 3.97/5.09 px. Both still fail original
floor support. See the [close-start control](gx_proposal_trace.md#optional-close-start-control--13-september).
The new refits have no visual judgement yet. More informed rectangle selection
is the next triggered follow-up; no follow-up implementation has started.

The 9 September session stopped after the single experiment and handover. Seed selection,
refitting, paint-width corrections, threshold tuning, graph search and production
integration were excluded. No automatic continuation into those tasks is queued.

## Validation and on-demand history

The seed comparison's scoped lint, synthetic selection checks and completed
runs passed, exit 0. Original GX counts and final outputs match the seven saved
baseline traces. Opus reviewed the selector and two-frame pilot; the extension
and controls were outside its scope. The report records floating-point tie and
partial-round ordering limitations. Production and original detector code remain
unchanged. The checks below belong to the earlier raster experiment.

38 relevant tests and scoped lint/types passed, exit 0. Input-byte equality,
source provenance, saved arithmetic and archive checks passed. The focused
Fable 5.1 review reproduced two courts and checked all 63 recorded results.
No implementation defect was established. Its shared-cap finding was verified
by inspecting pre-cap groups; no additional score arm was run.
The report passed factual-fidelity review and scored 96.9/100 overall: 89.6/100
mechanically and 4/4 on all four cold-reader dimensions. No material reading
problem remains. No experiment or review process remains active.

Use the [previous finite-distance experiment](fixed_court_matching.md) only for
the earlier distance-method regression. The [GX proposal trace](gx_proposal_trace.md)
records search coverage and previous refits. Its complete-search examples are
label-guided coverage diagnostics, not automatic selection results. The frozen
inputs and original annotations remain unchanged throughout these comparisons.
