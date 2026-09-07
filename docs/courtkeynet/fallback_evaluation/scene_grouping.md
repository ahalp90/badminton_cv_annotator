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
comparison takes the median distance across its sampled-frame pairs. Every pair
in a candidate group must have distance at most 0.30. This prevents a chain of
approximately matching scenes from joining distinct views.

A perceptual hash can tolerate a small zoom or crop. An
[OpenCV image-alignment check](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)
therefore checks each candidate against a representative image. The representative
has the smallest total hash distance to the other members. Alignment uses image
intensity within an expanded court mask. It must achieve correlation of at least
0.8 and move the court corners by at most one pixel at the 1280×720 reference
resolution. That corresponds to 1.5 pixels in a 1920×1080 source.

A different recurring projection within a broad hash group can form its own
aligned subgroup. Failed or ambiguous matches retain their existing courts.
These are development settings, established on the two examples below.

## Choosing the shared court

Groups need at least three accepted scenes. The shared court uses the median x
and y coordinates of each corner across the initial group. This limits the effect
of isolated corner errors without favouring long scenes or a single donor.
Videos with only one or two scenes still use their individual detected courts;
the minimum applies only to sharing a calibration.

The shared quadrilateral must pass the existing geometry checks. Each target
scene must pass a fresh person vote. Fallback scenes also need the existing
painted-line support in both court directions. A failing scene retains its prior
court; at least three scenes must pass before any member changes.

Raw detector corners and sources remain available. `view_group_index` identifies
the first scene sharing a final calibration. Members save identical active
corners. Scene rows carry that geometry into tracking, side assignment, landing
calculations and player-feature projection. Older saved records remain readable;
scene evidence without image summaries retains its previous geometry.

## Development results

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

- Comparing a shared fit against pooled painted-line evidence across the group.
  A median can preserve a bias shared by most original detections.
- Evaluation on additional broadcasts and appropriate view-specific ground truth.
- Moving cameras within a scene, general partial-court recovery and rally splitting.
- Any new court-space tree features or changes to the fitted tree models.

Image-space angles alone cannot select a valid court under general perspective.
Mapping the same four estimated corners back to a rectangle also supplies no
independent check. Additional visible court lines and intersections offer stronger
evidence for a future shared fit. The median remains the measured baseline.
