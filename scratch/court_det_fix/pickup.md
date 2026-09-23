# Court-detector pickup

Resume here. This is the only current handover; the linked records supply detail
when a specific question needs it. On 23 September 2026, the WebUI objective
return was replayed on the exact local evidence. The user prefers centre-to-edge
across the four-case gallery, confirming it as the winner after an initial
"marginally better" ruling. Retain it as a candidate
partial improvement; production fitting is unchanged. The fitting investigation
is complete: read [the assessment](edge_polarity/local_audit/ASSESSMENT.md).
The saved automatic SVD check also passes its coverage condition: all eight
historically approved automatic witnesses survive the 12-family screen.
**Integrated: the 12-family SVD screen in fresh experimental W5 generation.**
The default is 12; the explicit 16-family comparator remains. Code review and
a real matcher/scoring smoke pass. Matching depth remains a separate adjustment.
**Running now: the final case of the nine-case matcher benchmark.** The
Carmack job runs at revision `7935ce1`; use
[its worklog](svd_runtime/WORKLOG.md) for the PID, log and completion receipt.
Do not relaunch without checking that job. Opus also completed a requested
[compute-efficiency audit](svd_runtime/COMPUTE_AUDIT.md); fixes remain separate
from this fixed-matcher timing.
Read [the retention findings](evidence/webui_followup3_20260922/review_20260923/automatic_retention/README.md)
for the supporting coverage evidence. Three existing score-winner roles are lost, so the screen
does change selection opportunities. Keep the full-source comparator and G0.
The requested [eight-source amateur gallery](edge_polarity/local_audit/gallery/amateur/index.html)
has been reviewed. The user reports one insignificant regression (case unnamed)
and judges Am1 frame 54 very bad, regressing against an also-bad saved court.
The sample uses the earliest saved frame from each amateur source, including
GX. The user identifies Am1's false far baseline as the net's bottom white band.
The [WebUI net-versus-court audit brief](edge_polarity/local_audit/webui_net_audit/PROMPT.md)
and case packet are committed and pushed as `94821ee`; its source and images
are already on GitHub. This separate reasoning task needs no upload.
Do not restart the wider evaluation.

## Where we are

The target is a CourtKeyNet-free, scene-level detector. It should sample a few
frames, share useful proposals within a verified stable camera view, and reuse
the court with cheap change checks. The experimental parts exist; that complete
runtime and its acceptance/fallback behaviour are not integrated yet.

- The wider comparison is complete: **47 frozen cases plus 24 controls**.
  Image review stopped at 59 cases at the user's request. The other 12 have
  numeric analysis only. Controls comprise eight labelled non-court cases and
  16 unlabelled cases. Both source arms accept one of the eight negatives.
  Existing full-source W5 `(4,3)` is the comparator; rejection remains unresolved
- G1 paint-filtered proposals plus line templates retain 24/27 selections
  previously judged usable for development. These are not 24 clean fits.
  Removing G0 loses a tolerable Am4-319 result; keep G0 until that is resolved
- Geometry-only assignment can call an inner paint edge the outer edge.
  Correcting strong paint-side contradictions partly reduces the left inset:
  the six video-03 scenes' median static-grid disagreement falls **3.745 →
  1.799 working pixels**. Scores are mixed; the approved GX fit moves <0.05 px
- SS03-34 moves **1.078 px left and 0.529 px up**, short of the user's preferred
  diagnostic shift of 3 px left and 2 px up. Its paint score falls slightly.
  Polarity is useful evidence, not a finished correction or acceptance rule
- Eight earlier approved automatic gallery fits remain in current G0. The
  approved GX5 fit reproduces with today's fitter. Core fitting code is
  unchanged from 14 September; broadcast inward bias already existed then
- WebUI follow-ups 1 and 2 reproduce locally. Task 1 supplies fixed-candidate
  ranking counterexamples: lower overall landmark error can hide worse far-end
  clipping. Task 2 shows a wall selection surviving three samples. Neither
  return has produced an integrated detector change. Both SVD packages now have
  a [critical assessment and local replay](evidence/webui_followup3_20260922/ASSESSMENT.md).
  Their 12-family screen preserves all nine best cached reference-fit pairs;
  the subsequent automatic check retains all eight historical approved fits
  and all nine best reference-agreement candidates. Actual speed is untested

## Fitting investigation: original protocol

The fixed-data runs below have now been performed. Results and the user's
visual ruling are in [the current worklog](edge_polarity/local_audit/WORKLOG.md).
Fragment 111 alone adds (−0.497, −0.070) working pixels of upper-left movement
on SS03-34. Resolving both strong centre fragments adds (−0.506, −0.061).
The user prefers centre-to-edge across this small sample despite mixed paint
scores. On SS03-19, the user reports that polarity-only marginally undershoots
the back line, while centre-to-edge hugs the inner edge of the white line.
The default overlay marks the outside boundary; this is a relative preference,
not a ruling that every corner or boundary is exact.

The question is: **which fixed constraints still favour the inset after their
paint-side labels are corrected, and why?** Use `shuttleset_03_scene_0034`.
Keep SS03-29 as an improvement case, SS03-19 as a contrary case, and
`gxBQ_window_00_frame_5` as the previously approved control.

1. Verify the branch and current files. Use the saved
   [probe and results](edge_polarity/README.md#reproduction-and-checks).
   All seven baseline refits already reproduce within 1.1e-9 native pixels;
   repeat only what new instrumentation can affect
2. Decompose the corrected SS03-34 residuals by marking and fragment. Compare
   the corrected fit with the user-preferred diagnostic corner position.
   Identify the constraints that oppose that movement, including unchanged
   centre labels and fragments without strong polarity
3. For those named fragments, measure signed cross-line brightness profiles
   and the projected 40 mm stripe width. Separate wrong edge identity,
   observed edge displacement from blur, and inconsistent geometric width.
   Fragment 236 is the verified inner/outer mistake; it is not the whole cause
4. State a predicted numerical effect before changing the fit. Test one
   mechanism with the same parents, points and weights and the four controls.
   Stop that hypothesis if the predicted effect is absent. Preserve the
   contrary result instead of tuning until its score improves

Finish with an explained residual and one supported candidate change, or a
clear reason to leave fitting unchanged and proceed to the SVD candidate check.
No fixed corner offsets. No new full search or large gallery is needed here.

The [WebUI objective-audit prompt](edge_polarity/WEBUI_OBJECTIVE_AUDIT.md) is
ready to paste into a fresh WebUI session. It asks for independent reasoning,
a derivation or small counterexample, exact code anchors and a predicted local
test. The return is in [webui_return](edge_polarity/webui_return/); its analytic
examples and exact fragment-111 counterfactual reproduce locally.

## Immediately after fitting: SVD and search coverage

Update, 23 September: the saved automatic retention check is complete.
It retains 8/8 approved witnesses and all 34 candidates within one working
pixel of each full pool's best reference agreement. This numerical neighbourhood
is not a usability judgement. Candidate count falls from 19,286 to 8,657.
Both GX5 score winners and Am2-28019's paint winner are dropped. Proceed to
actual matcher timing; do not enable production pruning from cached counts.

The two SVD returns have been critically read and their cached calculations
reproduced. Read the assessment before tuning selection further. Cheaper
search could support a broader meaningful candidate pool at the same cost and
shorten later evaluations. The commissioned screen ranks line families by SVD
residual and prunes some; this is narrower than that broader opportunity.
Rejecting that screen would not rule out SVD-based search reduction generally.

Check retained useful candidates and actual matcher work/runtime first. Then
test whether reinvesting any savings in broader search improves coverage.
Keep those comparisons separate and preserve the full-source baseline.
More candidates alone do not establish better selection. Greyscale reuse and
removal of unused diagnostics can join this step with output-equality checks.

## Leads worth keeping, in order

| Lead | Evidence and bounded question |
| --- | --- |
| Remaining width/edge bias | [Polarity test](edge_polarity/README.md): partial correction only. Which residuals resist the preferred geometry? |
| SVD and search coverage | [Critical assessment](evidence/webui_followup3_20260922/ASSESSMENT.md): the cached screen passes at 12 families. Next check saved automatic-candidate retention, then actual work saved and broader search at equal cost |
| Ranking versus missing proposals | [Historical audit](evidence/independent_proposals/history_audit_20260922.md): approved geometries survive but rank lower. Compare fixed candidates before changing generation; keep Am4-319's G0 fallback |
| Source caps | Read-only worker lead, not yet parent-verified: W5 rebuilds capped G0 pools and loads capped G1 results rather than complete historical `all_camera` populations. Inspect `w5_holistic/run_w5.py:load_g0/load_g1` only if a named useful candidate is missing. The eight retained approved witnesses do not establish equality of whole populations |
| Far-end selection | [WebUI 1/2 return and local replay](evidence/webui_followups_20260922/README.md): GX86088 has less-clipped alternatives, still imperfect. Compare their existing evidence; no blanket visual-success claim |
| Sparse scene decisions | The same return uses archived registrations and candidate pools. A wall selection survives three samples. Adaptive two-to-three sampling, actual scene sampling and camera-change handling remain untested |
| Measurement cost | Greyscale reuse already works in the wider/polarity wrappers; the general runtime still needs it and removal of unused junction diagnostics |
| Difficult-camera coverage | Yellow14 and Am1 remain proposal problems. Earlier pooled-fragment/net-image evidence is a conditional lead in [decisions](DETECTOR_DECISIONS.md#older-ideas-worth-bringing-back), not another automatic sweep |

## A six-session route to branch completion

This is a planning target, not a promise of adequate quality. Combine stages
when the evidence permits. A failed finishing condition needs an explicit
scope or fallback decision, rather than another open-ended sweep.

| Session | Work | Finish when |
| --- | --- | --- |
| 1 | Explain the remaining fitting bias with the fixed cases above | One mechanism is supported or rejected; any candidate has a predicted, checked effect |
| 2 | Test the reviewed SVD screen on automatic candidate retention and actual search cost; remove measured waste | Useful-candidate retention and actual work saved are checked. Any broader search is assessed separately at equal cost; an unsuccessful screen is explicitly rejected |
| 3 | Settle selection and fallback on named failures and retained good fits | The chosen source/score rule preserves necessary G0 coverage; far-end failures have an explicit decision |
| 4 | Test sparse scene agreement and camera-change behaviour | Sampling, proposal reuse, resampling and abstention have concrete rules tested on short source clips |
| 5 | Integrate that bounded design and remove CourtKeyNet dependencies | The annotator uses the agreed scene interface; relevant pipeline gates pass |
| 6 | Check unseen cameras, selected failure cases and practical latency; close docs and branch | Acceptance/fallback behaviour and remaining limits are explicit; broader gates pass and the user can judge a small decisive visual set |

Use saved populations for numerical comparisons. Review images only when a
specific unresolved visual judgement can change a decision. Reserve any new
visual batch for that question; no routine review of another 50-plus images.

## Evidence and working agreements

- **Quality:** imperceptible misalignment is the visual ideal. Hugging the
  outside of the paint is preferred, not a minimum deployment threshold.
  Numerical fit supports visual judgement; there is no agreed pixel cutoff
- **References:** ShuttleSet has one static homography per video. It applies
  only to standard-view play, including neither side-on shots nor transitions.
  SS21-10's edit transition does not require reliable detection; confidence
  rejection there would be useful but is optional
- **Resources:** six workers where suitable, with one native numeric thread
  each for independent cases. Reuse greyscale; do not repeatedly decode or
  convert the same frame. Read `~/.codex/remote_hpc.md` before any HPC work
- **Delegation:** use bounded Luna Max tasks for mechanics. The coordinator
  checks their leads and makes the ruling. Claude Code Opus audit is authorised
  when useful, without a time limit; project sharing with Anthropic and HPC
  document uploads are authorised. All project sharing with delegated Codex
  agents is explicitly authorised. Use GPT-6 Sol medium, non-priority for
  bounded experiment runners and galleries. Do not cancel an existing external run
- **Git:** commits and pushes on `fix/court-det` are authorised. Use plain,
  succinct messages at meaningful checkpoints. Never commit to `main`
- **Checks:** choose validation for the change. The last code checkpoint passed
  ten synthetic tests, scoped Ruff and whole-project Pyrefly, all exit 0.
  Close-out changes are documentation/filing only; check paths and preservation

The code/results baseline is `4666526` (paint-side probe), preceded by
`76e390b` (historical fits) and `a042361` (WebUI local replay). All were pushed.
Use `git log -4` for the subsequent documentation checkpoint and actual tip.
No experiment or remote job was started by this close-out.

Twenty raw frozen-frame PNGs are intentionally local and untracked. Do not
stage or delete them during routine checkpointing. Large G0/G1 populations
remain under local `worklog/remote_records_20260921/`. Recovery archives stay
sealed. G1 scene 0029 lacks the original local input/estimator pair; seek it
only if exact generation replay requires it. The W2 checksum caveat is also
already recorded; neither issue needs a fresh search at pickup.

Use [INDEX](INDEX.md) for the experiment history, [FP_INDEX](FP_INDEX.md) for
files by idea, and [decisions](DETECTOR_DECISIONS.md) for settled findings.
The [archive map](archive/README.md) explains moved names and preserves prior
worklogs. Historical resume sections are records, not today's instructions.


## Appended close-out: SVD read, 23 September

Historical record below: the proposed automatic retention check has since
completed. Use the current next step at the top of this file.

The first package's arithmetic and code support a narrow positive result:
12 of 16 families retain the exact best cached reference-fit pair on 9/9 views,
with 132 rather than 240 possible ordered pairs. This is not a measured speedup
or proof that automatically generated useful courts survive. The 8- and
10-family cuts fail on most views. Preserve the frozen normalisation.

The supplement reproduces alternative-ranking failures and old SVD-refit
potentials. Its proposed per-assignment fit before duplicate selection/capping
is plausible but unimplemented and untested. Keep that separate from the
12-family screen; do not replace the current plan with this redesign.

Next SVD action, after the fitting decision: check the 12-family mask against
saved automatic candidate records with matching directions and IDs. Proceed
to a 12-versus-16 matcher/work comparison only if useful coverage survives.
No fresh detector run is needed merely to understand the returned findings.
The [assessment](evidence/webui_followup3_20260922/ASSESSMENT.md) records the
scientific limits, exact input checks, small protocol omission and rerun paths.
All three returned numerical scripts ran locally with exit 0; no image review,
new fits or matcher run occurred. Twenty local frozen PNGs remain untracked.
