# What was tried and decided

This file records decisions and links to their evidence. Each entry describes
what was known at the time. Later entries can replace an earlier choice;
[pickup.md](pickup.md) owns the current state and next work.

The detector searches using **all detected line fragments (G0)** and using
**only fragments that look like court paint (G1)**. It also builds courts from
crossing-line templates. The **scoring stage (W5)** measures these possible
courts and their refits against the image. Code labels appear below only when
needed to identify a saved experiment or court.

The test sets were used to develop the method. Their counts do not estimate
accuracy on unseen footage. Finding the played court, fitting its markings
accurately and rejecting a frame without a court are separate outcomes.

## Earlier search and fitting work

<a id="d01"></a>
### D01: Replace the CourtKeyNet repair chain

The old chain passed 0/11 labelled amateur frames. Keep its useful safeguards
for borrowing a court from another scene, but exclude CourtKeyNet from the
replacement. Removing old model-loading dependencies remains part of wiring
the replacement into the annotator. [Evidence](evidence/retirement/README.md).

<a id="d02"></a>
### D02: Keep the earlier courts and inputs; retire the failed rules

Better lines and more possible courts did not reliably identify the court
being played. Earlier acceptance rules are closed as defaults. Their inputs,
possible courts and failures remain useful evidence.
[Earlier trials](evidence/independent_proposals/README.md).

<a id="d03"></a>
### D03: Distinguish the different uses of SVD

SVD is a matrix method used in several unrelated trials here. Changing line
midpoints or replacing whole direction groups caused regressions. Fitting
existing groups with SVD differs from dropping four direction groups before
searching, the shortcut tested in D12. Refitting each line assignment before
applying its search limit remains an untested idea.
[Direction trials](evidence/direction_search/README.md);
[returned SVD study](evidence/webui_followup3_20260922/ASSESSMENT.md).

<a id="d04"></a>
### D04: Reject the tested rule for keeping more varied line matches

Sorting by score and keeping 512 line assignments can discard useful matches.
The tested alternative helped one case but badly worsened another. D13 tests
a larger limit instead. Neither limit is the separate limit of 256 crossing-line
templates. [Traces](evidence/direction_search/README.md#cap-and-duplicate-corrections).

<a id="d05"></a>
### D05: Search both all lines and paint-like lines

Searching paint-like fragments finds useful courts. Searching all fragments
finds others that would be lost. This does not show that scoring only against
paint-like fragments always works better. Adding courts can also make the
final choice worse. [Crossed comparison](evidence/g0_g1/README.md).

<a id="d06"></a>
### D06: Keep the 24/27 result in its original scope

The `(4,3)` and `(5,3)` minimum-line settings chose identical courts. Of 27
development cases, 21 choices were usable; requiring players to fit the court
raised that to 24. Searching paint-like lines plus crossing-line templates
kept that count. The wider test then needed the all-line search for Am4-319.
No tested floor score was sufficient to accept a court on its own.
[Original results](archive/20260922/evaluation_results_20260922.md).

<a id="d07"></a>
### D07: Hiding people helps some searches

All five corrected comparisons are complete. Useful broadcast courts survive,
including SS03-17 despite its poor paint-score winner. Hiding people alone
does not fix GX5. Older trials also changed line directions, so their results
do not isolate the effect of hiding people.
[Checks and image origins](evidence/holistic_admission/box_provenance.md).

<a id="d08"></a>
### D08: Sharing possible courts helps identify the played court

Using the same set of possible courts across aligned GX frames let paint scores
identify the played court on 7/7 frames, against 5/7 when each frame used only
its own courts. The far-end fits still had errors. Shared line scores picked a
wall on all seven. This supports trying reuse across a stable camera view;
repeated agreement alone does not certify the court.
[Temporal trials](evidence/pixel_temporal/README.md).

<a id="d09"></a>
### D09: The wider test needs all-line search and exposes narrow fits

The test covered 47 frozen cases and 24 controls. Removing the all-line search
lost the tolerable Am4-319 fit. Both tested versions accepted one of eight
labelled non-court frames. Fits tended to pull the sidelines inward; SS03-34
mistook an inner paint edge for an outer one.
[Wider comparison](archive/20260922/wider_evaluation_20260922.md#completed-numeric-comparison).

<a id="d10"></a>
### D10: Checking stripe edges partly fixes the left-side inset

On six video-03 scenes, the fixed-point stripe check reduced median left-edge
error against the shared static grid from 3.745 to 1.799 working pixels. The
approved GX control moved under 0.05 pixels. Scores were mixed, and this trial
did not change production defaults.
[Seven-case report](archive/20260926/experiments/edge_polarity/README.md).

<a id="d11"></a>
### D11: Relabelling stripe centres as edges helps, but does not fix Am1

The project owner preferred the changed labels in four reviewed cases. The
eight-source amateur extension did not fix Am1's confusion between net tape
and a court line. Neither a fixed corner offset nor a changed physical stripe
width was justified.
[Assessment](archive/20260926/experiments/edge_polarity/local_audit/ASSESSMENT.md).

<a id="d12"></a>
### D12: The 12-direction shortcut was accepted, then rejected

On 23 September, keeping 12 of 16 direction groups retained all eight approved
courts in nine development cases. Three score winners changed, with mixed
substitutes. The project owner accepted the tradeoff. The benchmark saved
52.9% of matcher time, not whole-detector time.
[Benchmark](archive/20260926/experiments/svd_runtime/RESULTS.md).

The later full-detector check changed five of 20 court views and clearly
worsened `letterboxed_short_frame_78` and `shuttleset_21_scene_0044`.
The joined detector therefore searches all directions. The earlier acceptance
still belongs in the history; it is no longer the default to carry forward.
[Later check](archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#svd12-screen-against-full-search).

<a id="d13"></a>
### D13: Searching more line matches recovered one useful court

In six cases using all fragments, 640 rather than 512 axis hypotheses cost
26.6% more total trial time and produced a visually suitable Am2 court where
the baseline failed. Keeping a wider set of courts cost 10.0% more without a
comparable win. The deeper Am2 and both deeper SS03 choices reproduced their
saved geometry. No global search-depth default changed.
[Run history and rulings](archive/20260926/experiments/svd_search/WORKLOG.md).

## Colour, net posts and stripe edges

<a id="d14"></a>
### D14: Close the tested colour-rejection rules

The floor trial rejected 0/9 saved choices. The final hue-only test rejected
0/12; one comparison was usable and eleven were inconclusive. Grouping raw
colours by hue rejected Am1 but failed a pale same-hue control. Similar paint
cannot establish correct geometry. These were tests on fixed fits, not a
successful colour-driven detector change.
[Complete colour record](archive/20260926/experiments/colour_consistency/PLAN.md).

<a id="d15"></a>
### D15: Recover Am1 without letting net evidence dominate

Three automatic starting points from line groups exposed a useful Am1-54
court. Net support selected it and the project owner accepted the result.
Unrestricted net preference then changed 13 choices in a 71-case scan and
lost good courts on Letterboxed78 and frame 52563. Reject that unrestricted
rule. Missing net support does not prove a court wrong; D17 limits its reward.
[Am1 record](archive/20260926/experiments/colour_consistency/AM1_RECOVERY.md);
[net worklog](archive/20260926/experiments/net_recovery/WORKLOG.md).

<a id="d16"></a>
### D16: Adjust stripe labels automatically when the evidence is clear

In the six-case gallery, the automatic rule looked practically identical to
the bright-paint rule, and both looked better than the original. It also
handled synthetic dark stripes at modest extra cost. Real dark courts remain
unproven. Keep original labels when the evidence cannot distinguish a stripe's
centre from its edges. This adjustment alone did not fix Am1.
[Edge policy](archive/20260926/experiments/edge_polarity/local_audit/ASSESSMENT.md).
D19 records its later inclusion in the joined detector.

<a id="d17"></a>
### D17: Give supported net posts a small, bounded reward

The 23 September rule uses net weight 0.04 and allows a matched fragment to
extend at most four working pixels below the predicted post base. Zero, one
or two supported posts add 0, 0.02 or 0.04 to a court's score. Missing evidence
is neutral. Only 4/26 supported posts on changed choices touched the base
sample, so this is evidence near lower posts, not verified post detection.
[Exact rule and limits](archive/20260926/experiments/net_recovery/ASSESSMENT.md).

The trial held 72 court pools from 71 frames, including old and seeded Am1-54
pools. Weight 0.04 changed 15/72 choices. The main comparison used the seeded
Am1 pool. On 45 frames with references, median per-frame errors at weights
0, 0.02, 0.04 and 0.08 were 3.40, 2.75, 2.75 and 2.76 working pixels.
The data did not establish a statistically reliable best weight. Am1-5352
and Yellow14 still had large errors; every setting accepted one of eight
non-court controls. [Statistics](net_recovery/statistics/paired_reference_report.md).

The project owner called the 20-case gallery very usable despite tiny amateur
imperfections. It chose a court with the bounded net rule, then adjusted the
stripe fit. This was one overall gallery judgement, not 20 separately labelled
outcomes. The saved-pool trial did not test the later fresh detector.
[Trial history](archive/20260926/experiments/net_recovery/WORKLOG.md).

## Speed and the joined detector

<a id="d18"></a>
### D18: Meet a whole-video time budget

The project owner's target is about **30 seconds for a five-minute video**,
with **90 seconds as the upper end**. Report start-up and warmup separately.
The earlier 5%-of-duration/15-second target was rejected. For longer videos,
expensive searches should mainly follow distinct compatible camera views.
[Recorded requirement](archive/20260925_optimisation_handover/handover_20260923/00_SHARED_CONTRACT.md).

The earlier six-case trials took 4.3–33.6 minutes per frame. They used the
12-direction shortcut and all-line search only, with cached line inputs.
Building courts took 91.6% of total trial time; refitting and scoring alone
took 59–107 seconds per frame. These are historical measurements, not timings
for the joined detector. [Run history](archive/20260926/experiments/svd_search/WORKLOG.md).

<a id="d19"></a>
### D19: The research steps now run together in memory

The 25 September joined detector passed the recorded 13 checks on all 28 views.
Chosen courts matched the research chain bit for bit. The comparison checked
crossing-line templates by count and metadata, and excluded research-only
scoring fields. Reproducing this historical comparison now requires turning
off the newer camera filter and geometry blend.
[Check and its limits](court_detector/check_20260925/README.md).

The joined path keeps all-line search, paint-line search, seeded crossing-line
templates, the bounded net reward and the automatic stripe refit. It searches
all directions. The earlier exact speed-ups are recorded by commit in the
[archived speed-up account](archive/20260926/originals/court_detector_optimisation_handover/README.md#built-in).

The old fresh-versus-saved score drift was traced to OpenCV's IPP distance
transform rounding differently with memory alignment. Use matching environments
and controls for numerical comparisons.
[Investigation](archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#score-drift-and-its-cause).

<a id="d20"></a>
### D20: Skip courts that need a sideways or upside-down camera

The 26 September filter reduced summed stage time over 28 views from 6,434
to 3,703 seconds, about 42%. Jobs ran eight views at a time on Carmack, one
thread each. Start-up and video decoding were excluded. This is not whole-video
latency. Three of 20 court views changed: one improved and two slipped at the
far end. Two of eight non-court controls received courts.
[Filter check](court_detector/check_20260926_upright/README.md).

<a id="d21"></a>
### D21: Keep a 10% share of line support in the final score

The 26 September choice uses 90% paint score and 10% geometry score, plus the
net-post reward. Here geometry score means support from detected line
fragments without requiring bright paint. It fixed
`gxBQ_window_00_frame_689`, reducing largest floor-coordinate error after refit
from 0.94 to 0.32 m. `gxBQ_window_03_frame_77876` still slips, at 1.06 m.
[Choice check](court_detector/check_20260926_court_choice/README.md#decision).

The Carmack run finished all 28 views. It matched the laptop replay on all 27
views that could be replayed; the other control kept its previous court.
Shared-server noise prevented a useful estimate of the blend's timing cost.
[Completed run](court_detector/check_20260926_court_choice/README.md#carmack-run).

<a id="d22"></a>
### D22: Keep both search shortlists at 256

Keeping 128 courts per search preserved the chosen courts in a replay but
removed close alternatives on six hard views. The check after the camera
filter reached the same conclusion. Keep 256 per pair and 256 per search.
[Replay and reasoning](archive/20260926/webui_final_opt_handover/README.md#shortlist-caps-of-128).

<a id="d23"></a>
### D23: The other 26 September scoring and filtering trials failed

| Trial | Why it was removed or set aside | Record |
| --- | --- | --- |
| Stricter paint test, bounded by the next painted line | Fixed neither far-end GX slip, worsened another view and took 12% longer | [Paint check](court_detector/check_20260926_paint_test/README.md) |
| Refit the top 15 courts before choosing | Fixed one view but worsened two others by about 0.7 m | [Choice check](court_detector/check_20260926_court_choice/README.md#decision) |
| Average paint contrast over a whole line | Four versions failed; distance, light and stripe width prevent simple cross-line comparison | [Line-paint check](court_detector/check_20260926_line_paint/README.md) |
| Reject courts that imply absurd player widths | Let a slipped court win on `am2_window_01_frame_28019`; no clear speed gain | [Player-size check](court_detector/check_20260926_player_size/README.md) |
| Add the search stage's score to the final choice | Did not fix the slips | [Saved-score test](court_detector/check_20260926_player_size/README.md#the-search-stages-score-does-not-fix-it) |
| Combine lengthwise and crosswise scores differently | The version that fixed Am2 slipped a whole end on two GX views | [Saved-score test](court_detector/check_20260926_player_size/README.md#combining-the-two-directions-differently-does-not-fix-it-either) |
| Use the average line-match score instead of a cheap court score | Ranked important courts about as poorly as build order | [Rebuilt courts](archive/20260926/webui_final_opt_handover/README.md#ruled-out-choosing-the-k-courts-by-their-line-guess-average) |
| Reject cameras looking nearly straight down | Would not help GX; other cases depend on assuming the lens | [Horizon check](archive/20260926/webui_final_opt_handover/README.md#skipping-courts-that-imply-a-camera-looking-straight-down) |

These tests used the same ten hand-marked views where applicable. Repeating
them on that set would not add independent evidence.

<a id="d24"></a>
### D24: Try a cheap score before the full score

On 26 September the project owner agreed to try 16 samples per marking, then
fully score the best 2,048 courts per pair. Keep a switch to score every court.
The saved replay covered 23 views that reached a search. The deepest cheap
ranks were 189 for a chosen court, 367 for a top-five court and 1,211 for a
court kept by a complete search. At 2,048, 217 pair-level courts were lost but
all were below their search's overall cut.
[Replay](archive/20260926/webui_final_opt_handover/README.md#the-cascade-item-16-now-to-be-tried).

This does not guarantee the same choices on new footage. The replay and old
speed estimates predate the camera filter. The
[design](court_detector/PERFORMANCE.md#score-cheaply-before-scoring-in-full)
owns the implementation and checks; the live queue owns whether it has run.

<a id="d25"></a>
### D25: Reuse a court only after checking it on the new scene

On 26 September the project owner agreed to move a court with the fitted
camera warp, check it against the new scene, and fall back to a full search.
Flag a camera group when the fallback disagrees. The proposed default is one
full search before reuse, with three available to combine separately searched
courts. The [design](court_detector/PERFORMANCE.md#reuse-a-court-when-the-camera-returns)
owns the details and unanswered tolerance question.

## Read accuracy claims carefully

The 24/27 result means visually usable courts in a development review. The
7/7 GX result means finding the played court, not fitting every marking well.
Use **clean fit**, **tolerable fallback** and **unacceptable fit** separately.
A fit can look right at the near end while losing a whole strip at the far end.
Report both the severity and frequency of such mistakes.

The project owner preferred outer edges of white markings on 22 September,
without setting a pixel cutoff. Later gallery feedback accepted tiny amateur
imperfections. Existing amateur references mix stripe centres and edges, so
small pixel differences can be ambiguous. Keep full images beside far-end
crops. [Original fit rulings](archive/20260926/originals/DETECTOR_DECISIONS.md#how-to-judge-fit-quality).

The earlier colour gallery displayed preserved fits, not fits changed by
colour. Its repeated columns were not evidence of a colour improvement.
[Complete account](archive/20260926/originals/DETECTOR_DECISIONS.md#interpret-the-earlier-colour-diagnostic-gallery-correctly).

## Conditional ideas to recover when needed

The earlier shared-camera replay, safeguards for borrowed courts, refitting
failures and search-limit traces remain in the
[archived decision account](archive/20260926/originals/DETECTOR_DECISIONS.md#older-ideas-worth-bringing-back).
It preserves their numbers, contrary cases and source links. These are leads
for a specific new problem, not another work queue.
