# Net-evidence pre-evaluation — 23 September 2026

## Resume

**The frontier packet does not justify another net experiment before
integration.** Its most practical idea is to stop net tape being counted as
floor paint. The completed automatic colour trials did not produce a useful
rejection rule; this leaves the Am1 failure unresolved.

This is a pre-evaluation, not a new detector result. No net detector, ranking
change or gallery was implemented. The negative colour result alone does not
validate an alternative net cue. Follow [pickup](pickup.md) for the next stage.

## Evidence and judgement

The [frontier packet](webui_net_evidence_frontier_handover/README.md) accurately
distinguishes projected geometry from an independent observation of a net.
Current W5 retains the camera-error diagnostic; it discards the projected net
segments before ranking. The older scorer samples generic line support along
the projection. A wall line or floor stripe can therefore reward it.

The historical replay still establishes a real Yellow improvement: its selected
worst-corner error changes from 221.19 to 26.01 pixels at 1280 × 720. The
Letterboxed change from 10.06 to 17.38 comes from breaking an exact floor-score
tie. That explains the mechanism but does not erase the regression. Fine
alignment remains part of the user's standard. These values were checked in
`replay.zip:joint_short/results.json.gz` under
`experiments/annotator/independent_court/recorded/player_guided/`.

The old formula also lowers a candidate's score when no net component is
visible: zero net support turns the blend into 0.75 times the floor score.
That behaviour conflicts with the intended treatment of missing evidence.
Restoring this scorer would require a new evaluation, not just wiring it in.

The modern W5 coverage diagnostics report no close Am1-54 or Yellow14 proposal:
the best maximum visible-landmark errors are 97.7 and 304.4 working pixels,
respectively. Those statements concern the inspected W5 pools. A new rejection
cue could still usefully cause abstention, but ranking cannot recover geometry
absent from those pools. See the
[coverage assessment](archive/20260922/evaluation_results_20260922.md#what-the-remaining-failures-need).

## What to retain from the independent review

Opus 5-5 high completed one read-only review without time limits. Its useful
findings were the missing-support penalty, the Letterboxed tie, and the need to
separate candidate coverage from ranking. Source inspection and saved replay
records support those findings.

Several recommendations need qualification:

- Do not run its optional absence-based rejection test. Visible projected
  segments with weak line support can reflect occlusion or missed extraction.
  That is not a positive observation contradicting the court.
- An oracle test does not require annotating all 71 views. A few directly
  marked nets could test whether accurate net observations distinguish named
  alternatives. It would establish information value, not a deployable net
  detector. Projecting those annotations from reference courts would make
  the test circular.
- The reported net projections from reference courts are model calculations.
  They do not verify the actual posts or tape. No agent inspected images;
  visual judgement remains with the user.
- Keep any older full-search coverage lead separate from this net decision.
  It does not justify restarting the historical experiment catalogue.

The raw review and launch record are local-only under
`local_scratch/external_delegate/20260923-net-frontier-audit-unlimited/`.
The result reports only `claude-opus-5-5` in model usage. The earlier timed
launch was stopped after the user requested no time limits; its replacement
completed with exit 0.

## Colour trial preparation

The saved SVD runs retain capped parent pools and successful refit children for
Am2, GX0, GX5 and Am1. They preserve native-pixel geometry and selected keys,
but omit the full final ranking and ranker scores. Rejection of a saved choice
can therefore be tested directly. Reselection would first require recovering
the original ordering; a reference-best candidate must never be inserted as a
replacement.

The automatic floor trial is now closed without a useful rejection. Existing person boxes
are not automatically reliable court-floor labels. GX5 also has frame-6 boxes
for a frame-5 image. The current colour diagnostic uses those raw boxes, while
`verifier.mask_boxes_working()` correctly checks the frame relationship. This
limits the existing GX5 occlusion masking and must be addressed before reusing
it in a decision trial. No effect size from that mismatch has been measured.

Sol owns gallery building. Its brief must require literal reuse of
`svd_search/gallery_template.html`, clearly labelled before/after decisions,
explicit abstention, unchanged good controls, identical candidate access in
each comparison, and visible locations of the evidence used. Preserve the
working zoom and overlay controls. The user performs visual inspection.

## Checks and scope

The reviewer checked all 15 packet manifest entries and replayed the supplied
net/mesh diagnostic with matching stdout (exit 0). The main session checked
the material score calculations, current W5 call path and GX5 box provenance
with read-only commands (exit 0). The independent pool inventory also made no
edits. No production code changed and no runtime test suite was required.
Local Markdown link targets and `git diff --check` passed with exit 0.

The review briefly wrote replay output to `/dev/shm` despite its scoped write
boundary, then removed it. No review target was changed.
