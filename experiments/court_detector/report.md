# Court detection: purpose, development and remaining problems

The court detector estimates the four outer corners of a badminton court from video.
Those corners let the annotation pipeline translate positions in an image into positions
on the court floor. The current approach combines detected lines, visible paint and
player positions with the known court layout. The central challenge is identifying which
visible markings belong to the court in use. Line detection, candidate selection and
combining observations each required separate experiments.

The project needs this geometry to build structured evidence about player performance,
including footage from unfamiliar professional and amateur venues. Supplied benchmark
mappings describe one camera per match. They help evaluate that camera view, but do not
provide geometry for arbitrary footage or another angle.

An earlier learned detector, CourtKeyNet with subsequent repairs, produced no courts
eligible for comparison with the reference corners on 11 labelled amateur frames.
Relaxing its checks admitted bad geometry. That result motivated the replacement. The
replacement uses pretrained models to find lines and people, then fits badminton
geometry without training a separate court model.

## What the detector returns

For each scene—a continuous shot interval—the detector returns either four court
corners in the original image’s pixels or a no-court result.
The corners define a **homography**, a perspective mapping between the image and a flat
court floor. This supports floor positions and distances. It does not recover the height
of an airborne shuttle, correct lens distortion, or follow camera movement when the
corners remain fixed.

DeepLSD supplies straight line fragments. Person and pose models locate people. The
detector searches for arrangements that could match the known badminton markings, using
both all detected lines and a separate selection of paint-like lines. The two searches
recover different useful candidates. Candidate courts must also imply an upright camera.
Refitting then adjusts the geometry to the centres or edges of painted stripes.

Visible paint provides most of the court score. Smaller geometry terms and a capped
net-post reward provide additional evidence. Player-position checks help distinguish the
court in use from other plausible arrangements. The score ranks candidate courts; it is
not a probability that a court is correct. A convincing fit to the wrong lines can still
score well.

## What the early experiments taught

The first line-based attempts showed that plausible geometry was easy to find and hard
to identify correctly. An OpenCV prototype accepted three of the 11 labelled amateur
frames, and all three courts were wrong. Frozen neural line extractors also failed to
select accurate amateur courts. Improving line extraction alone did not resolve line
identity or candidate ranking.

Paint evidence helped distinguish court markings from other edges. Filtering for
bright painted stripes improved accepted courts on the development broadcast examples
and rejected the non-court controls. It still failed on the labelled amateur frames.
That made paint a useful source of evidence within the search, while amateur court
selection remained unresolved. The [prototype record](../annotator/independent_court/README.md)
retains the numerical comparisons.

People supplied another useful constraint. Applying a player check retrospectively to
saved candidate rankings increased visually usable choices from 21/27 to 24/27
development cases. “Usable” was a visual judgement of the whole frame. These same
cases guided development of the checks, so the result describes improvement during
development.

Refitting exposed a different problem. A correction improved some already selected
courts without worsening the others. Applying the same rule throughout a larger
candidate search produced worse selected fits. Refitting changes which candidates win,
as well as their geometry. A comparison limited to the original winners had missed that
effect. The [refit comparisons](comparisons/earlier_approaches.md#better-local-fits-can-produce-worse-final-courts)
retain the counts and other attempted changes to candidate selection.

These experiments shaped the present combination of paint support, geometry checks and
player evidence. Detailed rejected approaches are in [earlier
approaches](comparisons/earlier_approaches.md); development measurements
and comparisons are in [evaluation](comparisons/development_evaluation.md).

## Using more than one frame

A single frame may hide a useful stripe behind a player or capture a transition.
Three-frame scene composition searches the first, middle and last sampled frames
independently, then combines their clearest markings into a fitted court. The composite
and accepted individual courts are compared on the same middle frame, giving the
alternatives a common basis for selection.

In a reviewed scene from video 005, the composite followed the court well while
all three single-frame fits had serious faults. Its paint score was lower than the
single-frame comparison, despite the visual improvement. This example helped motivate
composition and exposed a weakness in score-based selection. The review preceded the
current rule for choosing between composite and individual courts.

Evidence can also travel between scenes when the same camera view returns. Chronological
reuse aligns, refits and checks an earlier court to save another full search. A reused
result cannot itself become a donor, which prevents successive transfers from
accumulating drift.

Offline video pooling groups matching views and combines independent observations. It
can use later evidence to help an earlier scene. Pooling has its own failure mode: two
scenes in one cached experiment assigned different marking names to the same physical
stripe. Combining them unconditionally placed the court on blank floor. The
implementation therefore compares pooled fits with eligible complete courts. A receiving
scene may keep its own court if the shared choice fails its checks; prepared matching
scenes without a court can also receive the chosen court.

The modes use different combinations of these operations. `fast-robust` uses
middle-frame search and cross-scene pooling. `video-robust` also uses three-frame
composition. In the cached 11-view comparison, `fast-robust` scored higher. The inspected
outline placed the far boundary a few pixels above the visible paint. The [sampling and reuse
guide](../../docs/court_detector/sampling_and_reuse.md) describes each mode’s inputs and
outputs.

## What the released dataset establishes

The released snapshot covers 86 ShuttleSet and ShuttleSet22 videos and 44,810 scenes. A
sharing repair preserved donor eligibility when a receiving scene rejected a court, and
then allowed courtless matching scenes to receive one. Reapplying sharing over unchanged individual detections measured the effect of
that repair.

Against the supplied main-camera references at 1280 × 720, the number of videos whose
representative main-camera court had a mean four-corner error within 10 pixels rose from
74/86 to 85/86. A representative court is an actual detection that most closely agrees
with the others in the view covering most labelled rally time. This measures one
selected court per video; other scenes can still have wrong or missing courts. Other
camera angles may disagree with the main-camera reference even when their courts are
correct.

A trial on eight deliberately selected videos compared the original player checks with
score-first selection and a search without player rejection. At 1280 × 720, all three
methods placed every corner within 10 pixels of the reference in all 726 compared
main-camera scenes after sharing. Sharing concealed some wrong individual fits, and
image review found false courts among the changed detections. The trial preceded the
change allowing courtless scenes to receive a shared court. The evidence favours
retaining the existing player-required search.

The [release evaluation](comparisons/release.md) and [search-policy
trial](comparisons/search.md) retain the detailed counts, examples and
qualifications. Later detector changes, including scene composition and image-only
`--fast`/`--full` options, were not rerun across all 86 videos.

## What remains to improve

Wrong interior markings can be selected as outer boundaries. Boards and net tape can
resemble court evidence. Sampling can capture a court sliver or transition even when a
clearer court appears later. Player requirements can reject a visible court, while
camera movement and false alignments can undermine sharing.

Processing cost also remains substantial. Persistent workers, CPU parallelism, compiled
numerical work and optional GPU scoring helped. In controlled single runs on the same
five-minute clip, before composition, an NVIDIA L40 GPU and up to eight CPU cores were
used. With that setup, GPU template scoring reduced total time from about 275 to 227
seconds with identical outcomes for all 33 scenes. The target of 30 seconds, with 90
seconds at the upper end, remained unmet. These timings do not describe the current full
mode.

The next useful work is to compare sampled frames with clearer moments later in a scene
and improve transition and partial-court handling. A held-out test labelled across new
venues, including views without a court, is needed before making broader accuracy
claims. The existing checks have not demonstrated improved downstream contact or rally
recovery.

The [usage guide](../../docs/court_detector/usage.md) covers setup and operation. The
[design notes](../../docs/court_detector/design.md) explain maintained decisions;
[reproduction commands](tools/README.md) and [saved fixtures](saved_views/README.md)
support checking the recorded results.
