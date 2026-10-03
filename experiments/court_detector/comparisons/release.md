# Evaluation of the released court detections

Repairing court sharing raised the number of videos whose representative main-camera
court agreed with the supplied reference from 74 to 85 out of 86. Agreement here means
an average corner difference of at most 10 pixels at 1280 × 720. This comparison
measures the sharing repair over saved detections; it does not evaluate every later
detector change.

The [released
predictions](../../../data/court_detections/sset_and_sset22/extractions_20261003/README.md)
cover 40 ShuttleSet and 46 ShuttleSet22 videos, with 44,810 scenes in total. This page
explains that extraction's results and remaining errors. For the reasons the detector
was built and the experiments behind its design, [the development story](../report.md)
gives the wider account.

## What the sharing fix changed

When several scenes show the same camera angle, a good court found in one scene can help
the others. The old sharing code could discard that good court because it failed a check
in one receiving scene. The repair keeps that failure local to the receiving scene. A
later change also lets a scene with no court of its own receive one from a matching
view.

This comparison reran sharing over the saved individual detections from all 86 videos.
The individual detections stayed the same. Sharing changed 4,237 final courts and
supplied courts to 108 scenes that previously had none.

The official labels describe the main camera view of each match. The table below
compares predicted corners with those labels at 1280 × 720 resolution. A 10-pixel
difference is the reporting threshold used here.

| Court comparison | Before the fix | After the fix |
| --- | ---: | ---: |
| One typical court per video's main camera, average corner difference within 10 px | 74 / 86 videos | 85 / 86 videos |
| Main-camera scenes, every corner within 10 px | 5,400 / 6,748 scenes | 6,522 / 6,748 scenes |
| One court per rally, average corner difference within 10 px | 5,228 / 6,833 rallies | 6,171 / 6,833 rallies |

A video can pass the first row while still containing wrong courts in other scenes.

Most of the improvement came from repairing large errors. The average corner difference
for the video-level courts fell from 45.3 to 2.9 pixels, while the median barely
changed: 2.73 to 2.58 pixels. The remaining video above the 10-pixel threshold is
ShuttleSet 32, at 10.8 pixels.

The video-level error increased in 24 videos, by at most 2.2 pixels. Every video and
rally previously within the 10-pixel threshold remained within it.

The labels apply to the main camera. Courts from other camera angles need visual review
against the lines in those images. The [gallery](gallery.md) provides images alongside
the measurements; reference disagreement alone cannot judge those other views.

## Where courts are still wrong or missing

In [ShuttleSet22 video 43, scenes 89 and
176](../evidence/search/rally_review/shared_court_check), the released court has
roughly the right orientation but sits one line inward in both directions. It affects
171 labelled rally frames. Mistaking seated judges for players is a possible
explanation, but has not been verified.

Two scenes show why the sampled moment matters. ShuttleSet22 video 27 scene 27 is
sampled when only a sliver of court is visible. Video 44 scene 349 is sampled during a
transition overlay. A clear court appears later in both scenes. The released result for
video 27 fits the sliver frame and differs from the official corners by 134 pixels on
average. Video 44 still has no court.

ShuttleSet 36 scene 383 was different: the court was clearly visible, but the player
check passed in only 15 of 31 samples. The later sharing repair recovered that court,
with a 3.4-pixel average corner difference and a 4.5-pixel worst corner difference. It
also recovered two other courtless scenes in that video.

These examples point to better handling of transitions and partial courts as more useful
follow-up work than widening the search by default. The [review
images](search.md#images-and-clips-from-the-search-trial) show the cases.

## Processing time

Reapplying sharing to the 86 saved detections took 24.8 hours of processing summed
across videos, with a median of 15.8 minutes per video. Parallel jobs completed 85
videos in 70 minutes 33 seconds, using up to 27 workers. ShuttleSet 36 had already
completed separately in about 15.5 minutes. These times cover sharing; they exclude the
original court searches.

## How the comparison was measured

For each video, detections are grouped by camera view. The view covering the most
labelled rally time is the main view. Its typical court is the detection that most
closely agrees with the other detections from that view. For a rally, the same choice is
made within the view covering most of that rally. Missing courts count as misses. The
rally comparison excludes 377 rallies whose contact-frame lists did not run forward in
time.

The scene comparison uses the same 6,748 original main-camera scenes before and after
repair. Both versions are therefore judged on the same scenes.

The tables below retain the detailed counts and per-video results. [Reproduction
commands](../tools/README.md) rebuild them from saved predictions.

Later changes to the detector code, including scene composition and the
`--fast`/`--full` image-only options, were not rerun across all 86 videos. The released
dataset contains the sharing repair described here.

## Results files

These files are in [evidence/release/](../evidence/release/).

| File or directory | Use |
| --- | --- |
| `per_video.csv.gz`, `summary.json.gz` | Per-video measurements and full-corpus summary |
| `per_scene.csv.gz` | Final corners and reference errors for every scene |
| `per_rally.csv.gz` | Each labelled rally scored using its longest-overlapping scene |
| `rally_views/` | Each rally scored using the camera group occupying most of its duration; this supplies the report's rally-view results |
| `comparison_*.csv.gz` | Before/after measurements on the same videos and scenes |
| `pooling_diagnostic.csv.gz` | Cases where shared courts need closer investigation |
| `courts/`, `hard_samples/`, `rally_samples/` | One typical court per video, eight large reference disagreements, and stills from three rallies |
| `render_requests.json.gz`, `rally_samples.csv.gz` | Frame choices and captions for reproducing the images |

Errors use the references' 1280 × 720 resolution. Native corner fields retain the source
video resolution. Default-camera errors do not establish accuracy for alternate views.
