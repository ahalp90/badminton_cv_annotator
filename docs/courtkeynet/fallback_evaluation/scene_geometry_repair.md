# Court geometry repair — issue #148

This work fixes two cases where court correction made detections worse. The repair
recovers valid courts while preserving distinct camera views. On the two reported
ShuttleSet22 videos, it preserves every previously correct rally section.
Video 17 keeps 17 fully correct sections and matches 29 more contacts, with a small
precision loss. Video 53 gains 28 fully correct sections.
Reliable court geometry supports the project's automatic player, contact and rally
annotations for its badminton dataset.

The [companion worklog](scene_geometry_repair_worklog.md) records failed
approaches, the decisions they informed and the outstanding geometry work.
The [evidence bundle](../../../experiments/annotator/court_geometry_repair/README.md)
contains saved results, scoring checks and reproduction instructions.

## What was tested

The comparison uses the saved production court evidence as its baseline
(`a111181`). Both runs use the same pose and shuttle arrays, contact models,
rally-selection models and settings. Court-dependent annotations and features
were regenerated. No models were retrained, and the tree models were not tuned.
The reconstructed baseline matches the saved reference contact and final-section
records on both videos.

These are development checks on known failures, not an untouched test-set claim.
The two videos have 976 and 937 labelled contacts, across 73 and 76 labelled
rallies respectively. Contact matching uses a tolerance of ±10 frames at 30 fps;
±5 frames supplies a stricter check. F1 balances contact precision and recall.
A fully correct section matches a complete labelled rally's contact sequence
and player sides. Sections are paired by labelled rally identity across runs.

Separate geometry controls use the original ShuttleSet videos 3 and 21. Old and
new fallback code receive identical sampled frames and neural predictions.
The 1080p sources were resized to 1280×720; these are fresh controls, not exact
reproductions of the archived video proxies. Official static homographies supply
corner ground truth for matching wide-court views. Camera changes and close-ups
were assessed visually because a static homography does not describe them.

## What changed

Three faults interacted:

1. **Wrong boundary selection.** The fallback could select a short diagonal as
   an outer court line. It now checks edge direction and all available corner
   anchors. Both the fitted court and the final mixture of fitted corners and
   retained anchors must form a valid quadrilateral.
2. **Whole-video replacement.** Consensus could overwrite a correct model court
   from a different camera view. Model courts now keep their own geometry.
   Fallback repair uses a majority of donor scenes agreeing with the target's
   confident anchors. The target's painted lines must corroborate the repair.
3. **Camera changes treated as replay.** Once different court views were retained,
   the replay mask discarded video 17's entire later camera view. Camera difference
   no longer supplies an automatic replay veto or biases the speed reference.
   Court absence and the existing slow-motion check still supply exclusions.

Finite painted-line coverage closes another acceptance gap: a plausible polygon
containing two people could pass while fitting shirt, net or advertising edges.
The check measures actual fragment extent in sampled frames with valid corner shapes.
It takes each painted line's median coverage across frames, then averages within
each court direction.
For fallback and borrowed courts, both directions require at least 50% coverage,
within 10 reference pixels and 10 degrees. A borrowed outline must improve
coverage and pass a fresh person check. Coverage fractions are saved with each
scene's evidence.

Tracking, contact-side assignment, landing projection, hit-height estimation and
feature export now use the accepted court for the current scene. Landing searches
stop at its boundary. Calibration scoring also uses the relevant scene geometry.

## Results

| ShuttleSet22 result, ±10 frames | Video 17: before → after | Video 53: before → after |
| --- | ---: | ---: |
| Matched / labelled contacts | 842/976 → 871/976 | 195/937 → 889/937 |
| Predicted contacts | 945 → 1,002 | 202 → 914 |
| Precision | 89.1% → 86.9% | 96.5% → 97.3% |
| Recall | 86.3% → 89.2% | 20.8% → 94.9% |
| Contact F1 | 0.877 → 0.881 | 0.342 → 0.961 |
| Fully correct sections / labelled rallies | 17/73 → 17/73 | 7/76 → 35/76 |
| Previously correct sections lost | 0 | 0 |

At ±5 frames, F1 changes from 0.868 to 0.872 on video 17 and from 0.332 to
0.944 on video 53. Fully correct sections remain 14 on video 17 and rise from
6 to 31 on video 53, again with no previously correct sections lost.

Video 17 retains its correct opening court and selects both players at the two
reported frames, 46045 and 47276. Video 53 recovers 74 previously rejected
wide-court scenes, including the reported scene 334. Two old close-up acceptances
are removed. The final court evidence contains 28 accepted scenes on video 17
and 97 on video 53.

| Original ShuttleSet geometry control | Video 3 | Video 21 |
| --- | ---: | ---: |
| Sampled scenes | 47 | 45 |
| Accepted final courts | 46 | 45 |
| Advertising-board errors repaired | 7 | 0 |
| Mean corner error after boundary fix, before donor repair | 21.34 px | 4.74 px |
| Mean corner error after validated donor repair | 5.17 px | 4.74 px |
| Maximum final corner error | 19.05 px | 7.16 px |

Errors use 1280×720 reference pixels. Each mean averages all four corners across
the same 46 accepted video-3 scenes or 45 accepted video-21 scenes before and after
donor repair. The unaccepted video-3 scene is excluded from these error measures.
Every previously recovered control court is retained, and the boundary fix
recovers one additional video-3 scene.

## Limits and next work

The controls support this repair on the examined broadcast views. They do not
establish general performance on amateur footage. The fallback still requires
at least two confident neural corners and assumes a static camera within a scene.
The coverage rule was checked against known good courts, false close-ups and
advertising-board failures; it is not a separately trained scene classifier.

Video 17's long opening scene still spans multiple rallies. Its improved player
evidence does not by itself solve those rally boundaries. The fixed contact
models also produce more false positives there.

The [scene-grouping follow-up](scene_grouping.md) gives repeated camera views a
shared calibration to prevent coordinate drift between separate court estimates.
Partial-court usefulness still needs evaluation; the
[companion worklog](scene_geometry_repair_worklog.md) describes that question.
Further work includes evaluating the remaining broadcasts and splitting rallies
within long scenes. Regenerate court-dependent features before model retraining.
Replay detection still needs evidence beyond camera novelty; this change does
not claim to identify every replay.

Regression tests cover boundary selection, final court shape, finite line extent,
transient noise, image scaling, donor ambiguity, scene-specific consumers and
the replay-mask dependency. Persisted evidence records remain readable when
they predate the new coverage fields.
