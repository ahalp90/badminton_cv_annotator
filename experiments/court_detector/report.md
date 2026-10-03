# Court detector results

**Fixing court sharing made the biggest difference.** Across 86 videos, the
number whose main camera view matched the official court corners rose from
74 to 85. Changing how the detector searched for individual courts gave little
further benefit in the eight-video trial and introduced some bad detections.

The [released dataset](../../data/court_detections/sset_and_sset22/extractions_20261003/README.md)
contains the repaired results for 40 ShuttleSet and 46 ShuttleSet22 videos:
44,810 scenes in total. These courts are used to turn player positions in the
image into positions and distances on the court.

## What the sharing fix changed

When several scenes show the same camera angle, a good court found in one scene
can help the others. The old sharing code could discard that good court because
it failed a check in one receiving scene. The repair keeps that failure local
to the receiving scene. A later change also lets a scene with no court of its
own receive one from a matching view.

This comparison reran sharing over the saved individual detections from all
86 videos. The individual detections stayed the same. Sharing changed 4,237
final courts and supplied courts to 108 scenes that previously had none.

The official labels describe the main camera view of each match. The table
below compares predicted corners with those labels at 1280 × 720 resolution.
A 10-pixel difference is the reporting threshold used here.

| Court comparison | Before the fix | After the fix |
| --- | ---: | ---: |
| One typical court per video's main camera, average corner difference within 10 px | 74 / 86 videos | 85 / 86 videos |
| Main-camera scenes, every corner within 10 px | 5,400 / 6,748 scenes | 6,522 / 6,748 scenes |
| One court per rally, average corner difference within 10 px | 5,228 / 6,833 rallies | 6,171 / 6,833 rallies |

A video can pass the first row while still containing wrong courts in other
scenes.

Most of the improvement came from repairing large errors. The average corner
difference for the video-level courts fell from 45.3 to 2.9 pixels, while the
median barely changed: 2.73 to 2.58 pixels. The remaining video above the
10-pixel threshold is ShuttleSet 32, at 10.8 pixels.

The fix did not improve every court. The video-level error increased in 24
videos, by at most 2.2 pixels. But no video or rally moved from within the
10-pixel threshold to outside it.

There is one important limit to these numbers: a court from another camera
angle will disagree with the main-camera labels even if it is correct. The
[gallery](released_dataset_evaluation/gallery.md) provides images alongside the
measurements; reference disagreement alone cannot judge those other views.

## Did changing the search help?

A separate trial ran fresh detections on eight selected videos: ShuttleSet
11, 21, 30 and 36, and ShuttleSet22 27, 43, 44 and 51. These included known
failures and otherwise good videos with difficult scenes.

The trial compared three ways of finding and choosing a court:

| Method | Change from the original search |
| --- | --- |
| Original player checks | Candidates must satisfy the existing checks on where people stand. |
| Score-first selection | Choose the highest court score; player support breaks exact ties. Earlier search filters stay in place. |
| Broader search | Also remove player-based rejection during the search. Geometry and camera checks remain. |

All three then shared courts between matching views. After sharing, all three
had every corner within 10 pixels of the reference in all 726 main-camera
scenes being compared. The search changes did not improve that final count.

Before sharing, the alternatives did repair some individual fits. With every
corner required to be within 20 pixels, score-first repaired 16 and broke one; broader search
repaired 18 and broke the same one. That failure was ShuttleSet 11 scene 154:
its worst corner went from 9.4 pixels away to 1,418.6 pixels away. Sharing later
replaced it with a court within 2.3 pixels. The good final result therefore
concealed a bad individual detection.

Broader search supplied courts to 13 rallies that previously had none and
removed one elsewhere. Only two of those additions matched the main-camera
reference within 10 pixels. Across all 668 trial rallies, the agreeing count
rose from 625 to 627; score-first reached 626.

The separate score-first review checked eight detections it added and eight
it removed.
None of the 16 review images--which were supposed to have courts--actually had a court.

A separate review of the 13 added-rally images found three clear courts, two
fragments where leaving the court missing was preferable, and eight images
with no court. All 13 rally images show a different frame from the one the
detector analysed;
the [detection-frame images](search_policy_trial/rally_review/easy_court_misses/)
show what it actually saw in the three clear-court cases.

The trial ran before the later change that lets courtless scenes receive a
shared court. It also used eight chosen videos, rather than a held-out set of
new venues. Its results favour keeping the existing player-required search;
they do not establish a general accuracy rate for other footage.

## Where courts are still wrong or missing

In [ShuttleSet22 video 43, scenes 89 and 176](search_policy_trial/rally_review/shared_court_check),
the released court has roughly the right orientation but sits one line inward
in both directions. It affects 171 labelled rally frames. Mistaking seated
judges for players is a possible explanation, but has not been verified.

Two scenes show why the sampled moment matters. ShuttleSet22 video
27 scene 27 is sampled when only a sliver of court is visible. Video 44 scene
349 is sampled during a transition overlay. A clear court appears later in
both scenes. The released result for video 27 fits the sliver frame and differs
from the official corners by 134 pixels on average. Video 44 still has no court.

ShuttleSet 36 scene 383 was different: the court was clearly visible, but the
player check passed in only 15 of 31 samples. The later sharing repair recovered
that court, with a 3.4-pixel average corner difference and a 4.5-pixel worst
corner difference. It also recovered two other courtless scenes in that video.

These examples point to better handling of transitions and partial courts as
more useful follow-up work than widening the search by default. The
[review images](search_policy_trial/rally_review/README.md) show the cases.

## Processing time

Reapplying sharing to the 86 saved detections took 24.8 hours of processing
summed across videos, with a median of 15.8 minutes per video. Parallel jobs
completed 85 videos in 70 minutes 33 seconds, using up to 27 workers. ShuttleSet
36 had already completed separately in about 15.5 minutes. These times cover
sharing; they exclude the original court searches.

The search trial ran two methods together in each job so they could reuse
search work. The original/score-first pair took 16.95 summed hours. The
broader-search pair included an extra version that kept the final player check. Together they took 25.17 hours, 48.5% more. The logs do not separate the
individual methods' times, so that is a job-cost comparison rather than an
exact slowdown for broader search alone.

## Measurement details and files

For each video, detections are grouped by camera view. The view covering the
most labelled rally time is the main view. Its typical court is the detection
that most closely agrees with the other detections from that view. For a rally,
the same choice is made within the view covering most of that rally. Missing courts count as misses. The rally
comparison excludes 377 rallies whose contact-frame lists did not run forward
in time.

The scene comparison uses the same 6,748 original main-camera scenes before
and after repair. The eight-video trial has its own fixed set of 726 scenes.
That keeps a method from improving its result simply by changing which scenes
are counted.

The [full-corpus tables](released_dataset_evaluation/README.md) and
[search-trial tables](search_policy_trial/selection_results/README.md) contain
the detailed counts and per-video results. [Reproduction commands](reproduce.md)
rebuild the tables from saved predictions.

Later changes to the detector code, including scene composition and the
`--fast`/`--full` image-only options, were not rerun across all 86 videos.
The released dataset contains the sharing repair described here.
