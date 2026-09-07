# Shared courts for repeated camera views

Repeated scenes can now share one court calibration when their sampled images
support the same camera projection. The development comparison on ShuttleSet22
videos 17 and 53 removes variation between repeated-view court estimates. Both
videos retain exactly the same final contact and rally records with frozen models.

This follows the [scene geometry repair](scene_geometry_repair.md). That repair
preserves different camera views. Grouping stabilises repeated instances of a view.

## Matching a view

The court detector already decodes scene samples. Grouping reuses the first,
middle and last samples to compute
[PySceneDetect perceptual hashes](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.hash_detector.HashDetector)
and a median greyscale image at 960×540. It adds no neural inference pass.

Hash distance measures the fraction of differing bits. For a scene pair, the
comparison takes the median distance across its sampled-frame pairs. Every member
must have distance at most 0.30 from one fixed representative. That representative
has the smallest total hash distance to the remaining scenes.

A perceptual hash can tolerate a small zoom or crop. An
[OpenCV image-alignment check](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)
therefore checks each candidate against the same representative image. Matching
a neighbour is insufficient: every member must align directly. Alignment uses image
intensity within an expanded court mask. It must achieve correlation of at least
0.8 and move the court corners by at most one pixel at the 1280×720 reference
resolution. That corresponds to 1.5 pixels in a 1920×1080 source.

After accepting a group, the remaining scenes can form their own groups around
other representatives. A failed representative remains eligible to join a group
using another reference image. Failed or ambiguous matches retain their existing courts.
The thresholds were established on the two development examples below. The
representative-based grouping revision was tested on original ShuttleSet.

## Choosing the shared court

Groups need at least three accepted scenes. The shared court uses the median x
and y coordinates of each corner across the initial group. This limits the effect
of isolated corner errors without favouring long scenes or a single donor.
Videos with only one or two scenes still use their individual detected courts;
the minimum applies only to sharing a calibration.

The shared quadrilateral must pass the existing geometry checks. Each target
scene must pass a fresh person vote. The proposal must stay within the existing
anchor tolerance of that scene's original confident model-derived corners.
Fallback scenes also need the existing painted-line floor in both court directions,
and total line support must not decrease from their current accepted court.
A failing scene retains its prior court; at least three scenes must pass before
any member changes. The median is computed once from the initial image group;
survivors validate that same proposal rather than recomputing it.

Raw detector corners and sources remain available. `view_group_index` identifies
the first scene sharing a final calibration. Members save identical active
corners. Scene rows carry that geometry into tracking, side assignment, landing
calculations and player-feature projection. Older saved records remain readable;
scene evidence without image summaries retains its previous geometry.

## Development results

These results describe the initial grouping revision `4aa75fb`. The later
representative-based revision is assessed separately below.

The comparison uses the same cached detections, pose and shuttle inputs as the
previous repair. It regenerates annotations and contact features before applying
the same four tree models and fixed selection policy. No model was retrained,
refitted or tuned.

| Measurement | Video 17 | Video 53 |
| --- | ---: | ---: |
| Accepted scenes | 28 | 97 |
| Scenes sharing one calibration | 27 | 97 |
| Maximum corner variation before sharing, source pixels | 3.135 | 24.799 |
| Maximum image movement measured within that group, source pixels | 0.762 | 0.259 |
| Maximum coordinate disagreement before sharing, metres | 0.077 | 0.661 |
| Correct rallies at ±10 frames, before → after | 17 → 17 | 35 → 35 |
| Correct rallies at ±5 frames, before → after | 14 → 14 | 31 → 31 |

Video 17's opening view stays separate. The coordinate disagreement calculation
uses 105 grid locations over a 6.1×13.4 m court. Each group median projects the grid
into the image; the original scene calibrations map those identical image points
back into court coordinates. The table reports the largest pairwise difference.
It measures inconsistency, not error against ground truth. Shared calibrations
remove that variation by construction.

Both final contact/rally streams are identical to the repaired baseline. Grouping
improves coordinate consistency on these examples; it does not improve their
frozen-model contact or rally scores.

The production implementation reproduces the experiment's active corners,
homography rows, person votes and court-presence flags for both videos. The four
tree model files and fixed policy also match the prior experiment exactly.

Existing player-feature inputs were regenerated for both complete videos:

| Player-position change, before → shared calibration | Video 17 | Video 53 |
| --- | ---: | ---: |
| Player-frames valid in both runs | 179,091 | 63,845 |
| Median position change, metres | 0.000 | 0.016 |
| 95th percentile position change, metres | 0.006 | 0.133 |
| Maximum position change, metres | 0.044 | 0.391 |
| Newly available player-frames | 0 | 1 |

One player-frame is one player slot in one video frame. A valid position has
finite court coordinates. Position change is the straight-line distance between
the before and after court-space estimates, converted to metres.

These are changes in estimated position, not measured accuracy gains. Whole-video
statistics include the unchanged opening view in video 17. No player-frames lost
validity. Posture values at matching valid frames and interpolation flags remain
identical. Movement and recovery summaries were also regenerated using matching
saved annotation spans and contacts. Their numeric values change with the court
projection; they are not identical between runs. For example, mean movement
inefficiency in video 53 changes from 0.100973 to 0.100930 for the top player and
from 0.075668 to 0.075607 for the bottom player, in normalised court coordinates.
Recovery excludes one span in video 17 and eight in video 53 because the saved
prediction does not identify which court side supplies the initial striker.

## Matching revision on original ShuttleSet

The first implementation required every pair of scenes in a preliminary group to
pass the hash bound. In the original video-21 control, ten scenes were left in
groups too small for alignment. Seven matched 33 or 34 of the main group's 35
members, but a mismatch with another member prevented admission.

The revised implementation requires both hash compatibility and geometric alignment
to one fixed representative. It removes the preliminary all-pair hash partition;
it retains the 0.30 hash, 0.8 correlation, one-reference-pixel movement and three-
member requirements. Chained matches and changed zooms still fail the tests.

Using the same cached geometry and sampled images, original video 3 retains its
39 matched scenes. Video 21 increases from 35 to 42, with no previously grouped
member lost. The seven additions were visually checked against the same wide view.
Shared-median reference-corner error stays 4.631 px in video 3 and changes from
4.636 to 4.632 px in video 21. This improves coverage of repeated views; it does
not establish a meaningful calibration-accuracy gain.

These are image-grouping controls using locally valid courts. They exclude the
production person gate, so 42 matched scenes is not a full-pipeline acceptance
count. The sharing adapter applies geometry, person and target-evidence checks; the
subsequent safeguard below reduces how many matches actually share a court. Matching requires images and court outlines; it does not
use neural confidence or depend on CourtKeyNet's candidate-recovery rules.

## Sharing must preserve the target's evidence

A later synthetic review reproduced a recovery being undone by sharing. Two
model-only scenes agreed on a misplaced corner. Their median passed the fallback
target's absolute line floor, but moved its confident anchor by 75 reference
pixels and weakened the accepted local fit. Identical images establish the same
camera view; they do not establish that its majority court estimate is correct.

Sharing now applies the existing donor anchor bound to every target's original
confident corners. A fallback target also retains its accepted calibration when
the proposed median reduces total painted-line support. This protects an accepted
alternative through the later sharing step. Regressions cover both checks, native
resolution scaling, fresh person votes and sharing among three remaining survivors.

The stricter check trades shared calibration for stronger local evidence:

| Original control | Image-matched scenes | Scenes sharing after safeguards | Mean corner error, previous sharing → guarded |
| --- | ---: | ---: | ---: |
| 3 | 39 | 26 | 4.631 → 4.492 px |
| 21 | 42 | 22 | 4.632 → 4.684 px |

The comparison reuses the same cached original frames and locally valid courts.
Person votes are forced true to isolate the geometry guards; these are not final
production acceptance counts. Errors cover the same 39/42 matching-view populations,
including the scenes that now retain local courts. No court is discarded. The
accuracy effects are small and mixed; line support is not ground-truth accuracy.
Earlier complete downstream results remain measurements of their stated revision.
The safeguards have not been rescored end to end on the development videos.

## Feature coverage and deferred work

The contact comparison regenerates the existing image-space shuttle and player
features, missingness, and candidate-region context. Court geometry can affect
these through player selection and scene acceptance. The closing selectors' frozen
physical-feature join uses 60 physics and 25 missingness values for each candidate
and fixed contact, alongside ten action features. It does not introduce court-space
positions into those tree inputs.

Player-performance code already projects player positions through each scene's
court. Posture uses selected pose coordinates. Sharing changes the calibration
supplied to that existing path; it adds no new feature definitions.

Remaining work includes:

- Shared calibration accuracy: the tested pooled-line fit worsened corner error
  on two original-data controls, so the median stays. A common bias can remain.
- Broader view-specific ground truth. The completed frozen fresh-broadcast audit
  and its limits are recorded in the [follow-up worklog](scene_geometry_repair_worklog.md).
- Moving cameras within a scene, general partial-court recovery and rally splitting.
- Any new court-space tree features or changes to the fitted tree models.

Image-space angles alone cannot select a valid court under general perspective.
Mapping the same four estimated corners back to a rectangle also supplies no
independent check. Additional visible court lines and intersections offer stronger
evidence for a future shared fit. The median remains the measured baseline.
