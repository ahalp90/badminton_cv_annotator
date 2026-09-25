# Court detector

This folder finds a badminton court in one image from a video, usually a single
frame. It runs the accepted research method as one piece of code, and keeps
every step's results in memory. On the 28 test views it gives the same results
as the research scripts. It is not ready for new videos yet: a view takes about
4 minutes, its inputs come from other tools, and it still loads code from
research folders.

**Names from the research.** The accepted method is called the D17 chain, and
`../d17_timing/run_d17.py` is the research script that runs it. The code labels
its steps too: G0 and G1 are the two court searches, and W5 is the scoring
stage. This page uses plain names and gives each label once, in brackets, so
you can find it in the code.

## What goes in and what comes out

`CourtDetector.detect(view, people, frames)` in `detect.py` takes three
inputs. Their types are in `inputs.py`.

- `view` (`ViewInputs`): the image, which is a video frame or a composite of
  several. Also its line fragments, which are short line segments found by the
  DeepLSD line detector. Also the person boxes, the video frame the image
  stands for, the first and last frame of its scene, and a note of which frames
  the image and boxes came from (`provenance`)
- `people` (`PeopleSource`): the people in the video frames around that one,
  each with a box and 17 body-joint positions (the COCO-17 pose layout)
- `frames` (`FrameReader`): those video frames themselves

It returns a `CourtResult`:

- `corners_native_px`: the court's four corners in the original image's
  pixels. The detector projects the whole court, so a corner can lie outside
  the image. `None` means no court
- `no_court_reason`: why there is no court. `no_gated_court` means no
  candidate passed the full-court checks in step 5. `rank_deficient` and the
  other refit reasons mean the final fit in step 7 failed. For example,
  `rank_deficient` means it had too little independent information to fix all
  four corners
- `chosen_key`: which candidate won
- `stage_seconds`: seconds per step, when timing is on

## How it finds the court

Most steps work on a working image, which is the image shrunk until its
longest side is at most 960 pixels. A 16:9 frame becomes 960×540.

1. **Feet** (`feet.py`). Looks at 31 video frames around the image, 10 a
   second, which covers 3 s. Near the edge of a scene, the window shifts to
   stay inside it; a scene shorter than the window raises `ValueError`. Keeps
   the unbroken run of frames around the image's frame that stay in its shot.
   A frame leaves the shot when a small grey thumbnail of it differs from the
   image's by more than 8 grey levels on average. Takes the bottom centre of
   each person's box as their feet, and drops people whose pose shows them
   seated.
2. **Set-up**. Shrinks the image and the line fragments to working size, and
   sorts the fragments by direction. The person boxes hide people from the
   paint measurements, but only when the boxes come from the image itself.
3. **Court search** (G0 and G1). Finds up to 16 main line directions in the
   view. For every pair of them, tried both ways round, it matches lines to the
   court's markings and builds candidate courts. It rejects courts the players'
   feet do not fit. It keeps up to 256 of the best-scoring, distinct courts per
   pair, and 256 overall. It runs twice: once on every line fragment (G0), and
   once on only the fragments that look like white paint (G1). A paint fragment
   is a pale line, clearly brighter than its surroundings and not strongly
   coloured.
4. **Line templates**. Builds more candidates from rectangles of crossing
   lines. As well as its own starting points, it tries the point where each
   pair of the three longest lengthwise lines meets. A template must show at
   least 4 lengthwise and 3 cross-court lines, and a camera with a plausible
   lens must be able to see it that way.
5. **Scoring** (W5). Merges duplicate candidates and sets aside invalid
   shapes. For each remaining candidate, it measures how well its markings sit
   on painted stripes, and tries refitting it to those stripes. Then it ranks
   the candidates and their successful refits together. Only candidates that
   pass the full-court checks (gated) can win. A gated court has a valid
   shape, and a real camera could see it that way. The players' feet also land
   on it or just outside it, within 15% of its size. At least one player must
   do so in every sampled frame, and one in each half in at least half the
   frames.
6. **Net choice** (`net_choice.py`). Each gated candidate implies where the
   two net posts stand. Its score is its scoring-stage evidence plus a small
   bonus when line fragments support those posts: 0.02 for one post, 0.04 for
   both. The highest score wins. With no gated candidate, there is no court.
7. **Stripe refit** (`stripe_refit.py`). A line fragment can sit on a
   painted stripe's centre or on either edge. For the winner, this step checks
   the brightness and colour on either side of each fragment it fits. It moves
   a fragment to a different centre or edge position only when that evidence
   is strong enough. Then it refits the corners. If that fit fails, there is no
   court.

Steps 3 to 5 take almost all the time.

## Current state

- **It matches the research scripts.** On all 28 test views, the chosen
  court, the refit and the other results the check compares are identical,
  bit for bit. `check_20260925/README.md` has the evidence and says exactly
  what was compared.
- **Results on the 28 views.** 20 are court views and 8 are control views
  with no court (`sset_21_…`, from one ShuttleSet video). All 20 court views
  get a court. Of the 8 control views, 6 correctly get no gated court, and one
  gets none because its final fit fails (`rank_deficient`). One control view,
  `sset_21_gloiZ_gTJaE_frame_00100347`, wrongly gets a court.
- **Accuracy** is the research method's. 18 court views have hand-marked
  court landmarks. On those, the median error per view is 1.2 to 3.6 working
  pixels, and 2.1 for the middle view
  (`../court_detector_optimisation_handover/claude_evidence/d17/d17_results.txt`).
- **Speed.** About 230 s a view on one core, from 20 s for a view with no
  court to 546 s. That comes from 6,434 s over the 28 views, with self-checks
  off and 8 views at a time on Carmack. It leaves out start-up and video
  decoding. The search takes about two thirds of the time and scoring most of
  the rest. It is 8-12% faster than the research script, which is about the
  size of run-to-run noise.
- **Tests.** In the repository's `tests/` folder, `test_court_detector_feet.py`
  checks the feet step against the scripts that made the test set's feet, and
  against BST-X's seated-person rule. `test_court_detector_modules.py` checks
  that the detector loads the same research modules as the research script.
  It also checks that research scripts can import the detector's own modules
  back safely.

## Running it

Run from the repository root, with the repository and its `src/` folder on
Python's path (`PYTHONPATH=.:src`). Before numpy loads, set six thread
variables to 1, and give OpenCV's video reader one decoding thread.
`run_views.py` shows both. Then:

```python
from scratch.court_det_fix.court_detector.detect import CourtDetector, Switches

detector = CourtDetector(Switches())  # loads the research code once
result = detector.detect(view, people, frames)  # one call per view
```

On the test views, run the test harness. Each `VIEW` is a view ID from
`../court_detector_optimisation_handover/claude_evidence/fresh_feet/views.json`,
and its image, line fragments and person boxes come from the saved research
inputs:

```
python -m scratch.court_det_fix.court_detector.run_views --people PEOPLE_DIR --output OUT VIEW [VIEW ...]
```

`PEOPLE_DIR` holds one record per view of the people and poses around its
image, and each record names its video. The 28-view check used records and
videos on Carmack; they are not in git. Add
`--baseline ARM_DIR --feet FEET_FILE --artefacts` to compare each view with
the saved research run. The `run_views.py` docstring gives the details.

| Switch | Default | What it does |
| --- | --- | --- |
| `self_checks` | On | Checks that each step's output is consistent, for example by replaying the scoring stage's fit before the refit. Costs little. It does not compare with the research run; `run_views.py --baseline` does that |
| `enforce_scene_consistency` | On | Uses only feet from the image's own shot. Planned to default to off once a scene cutter supplies real scene ranges |
| `timing` | Off | Reports seconds per step |
| `artefacts_dir` | None | Writes each view's intermediate results to this folder |

The accepted improvements have no on/off switches yet: the seeded templates,
the paint-only search, the net choice and the stripe refit. `STRIPPED.md` says
how to add each one.

## What it leaves out

It skips research-only work that never changes the chosen court: saved files,
extra rankings and diagnostics, and extra evidence the old search kept for its
reports. `STRIPPED.md` lists each piece and where it plugs back in.

## Before it can run on new videos

- **Speed.** About 4 minutes a view.
  `../court_detector_optimisation_handover/CLAUDE_FOLLOWUPS.md` lists what is
  left to try, such as searching direction pairs in parallel and reusing a
  court across the scenes of one camera.
- **Inputs.** Nothing builds the inputs from a new video yet. That needs
  DeepLSD for the line fragments, and a person and pose detector such as
  RTMLib for the people. It also needs a scene cutter for the scene range, and
  PySceneDetect is the candidate. `inputs.same_frame_provenance` makes the
  provenance for a frame taken straight from the video.
- **Research code.** The detector still imports 15 files from the research
  folders. Some of those folders hold files with the same module name. At
  start-up, `detect.py` checks which copy Python loaded and stops if it is the
  wrong one. Where the detector should finally live is still open.
- **Coverage.** Its recorded evaluation covers these 28 views.

## Files

| File | What it holds |
| --- | --- |
| `detect.py` | `CourtDetector`, `Switches` and `CourtResult`; runs the steps in order |
| `inputs.py` | The input types |
| `feet.py` | Step 1 |
| `search.py` | Step 3's settings and paint filter, and step 4's extra starting points |
| `net_choice.py` | Step 6 |
| `stripe_refit.py` | Step 7 |
| `run_views.py` | The test harness for the 28 views |
| `STRIPPED.md` | What the detector leaves out |
| `check_20260925/` | The 28-view comparison with the research scripts |

`../d17_timing/WIRING.md` explains how the research stages join up, and why
each piece is kept or dropped.
