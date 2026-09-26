# Run the court detector

`CourtDetector` finds a badminton court in a prepared image. It runs the
search and scoring steps in memory and returns four court corners, or a reason that no
court was found. This page owns its inputs, behaviour, settings and commands.

Read [pickup.md](../pickup.md) for current work,
[the decisions](../DETECTOR_DECISIONS.md#d19) for measured results, and
[PERFORMANCE.md](PERFORMANCE.md) for the speed-up design.

## Inputs and result

`CourtDetector.detect(view, people, frames)` takes three inputs from `inputs.py`:

- `view` (`ViewInputs`): an image, its detected line fragments, person boxes,
  represented video frame, scene boundaries, and the source frames for the
  image and boxes. The detector uses that source information to avoid hiding
  people in the wrong image
- `people` (`PeopleSource`): boxes and 17 COCO-layout body joints for people
  in nearby video frames
- `frames` (`FrameReader`): access to those nearby video frames

A caller must supply these inputs. The detector does not yet build them from
a new video.

The returned `CourtResult` contains:

| Field | Meaning |
| --- | --- |
| `corners_native_px` | Four corners in the original image's pixels, or `None`. Corners can lie outside the image |
| `no_court_reason` | Why no court was returned. `no_gated_court` means none passed the court checks; `rank_deficient` means the final fit lacked enough independent information |
| `chosen_key` | The saved identifier of the chosen court |
| `stage_seconds` | Time per step when timing is enabled |

## How it finds the court

Most steps shrink the image to a longest side of at most 960 pixels.
A 16:9 image becomes 960×540.

1. **Find players' feet.** Sample 31 frames at 10 per second. Shift the window
   to stay within the scene; a scene shorter than the window raises
   `ValueError`. Keep the continuous run of frames from the same shot around
   the target. Use the bottom centre of each box, excluding people whose
   poses indicate they are seated
2. **Prepare the lines and image.** Shrink line fragments and group them by
   direction. Hide person boxes from paint measurements only when the boxes
   belong to this image
3. **Search for courts.** Match pairs of line directions to court markings.
   Run once with all fragments (G0), and once with only paint-like fragments
   (G1). The second search uses pale lines that are brighter than their
   surroundings and not strongly coloured. Players' feet must fit the court.
   Reject sideways and upside-down camera geometry. Keep up to 256 distinct
   courts per pair and 256 per complete search
4. **Build courts from crossing lines.** Use rectangles and extra starting
   points from the three longest lengthwise lines. Require at least four
   lengthwise and three cross-court lines, and a plausible camera
5. **Score and refit the possible courts.** Merge duplicates, check shapes,
   measure paint support and try refitting each court to its stripes. Rank
   the courts and successful refits together. A court can win only if its
   geometry and camera are valid and the players fit. At least one player
   must fit in every sampled frame, and one in each half in at least half
   the frames. The code calls this stage W5
6. **Choose a court.** Combine 90% paint score and 10% line-support score.
   Add 0.02 for each supported net post, up to 0.04. Missing net support is
   neutral. If no court passed the checks, return no court
7. **Adjust the stripe fit.** Check whether each fitted line lies on the
   centre or an edge of its painted stripe. Change its label only when the
   image evidence is clear, then refit the corners. A failed fit returns no
   court

The court checks allow feet up to 15% beyond the court. The shot check keeps
frames whose small grey thumbnail differs from the target by no more than
eight grey levels on average. These are existing rules, not extra input
preparation supplied by the caller.

## Run it

Run from the repository root with `PYTHONPATH=.:src`. Follow `run_views.py` for
single-threaded numerical libraries and OpenCV decoding. It sets these before
loading NumPy.

```python
from scratch.court_det_fix.court_detector.detect import CourtDetector, Switches

detector = CourtDetector(Switches())
result = detector.detect(view, people, frames)
```

The saved-view runner takes IDs from the
[28-view list](../court_detector_optimisation_handover/claude_evidence/fresh_feet/views.json).
Its image, line and box inputs come from saved research records. `PEOPLE_DIR`
holds one record per view with people, poses and the source video's path.

```bash
PYTHONPATH=.:src python -m scratch.court_det_fix.court_detector.run_views \
  --people PEOPLE_DIR \
  --output OUT \
  VIEW [VIEW ...]
```

Add `--baseline ARM_DIR --feet FEET_FILE --artefacts` to compare against a saved
research run. To reproduce the original 25 September comparison, also use
`--any-camera-roll --geometry-weight 0`. The newer defaults intentionally
change results. The [original check](check_20260925/README.md) states exactly
what was compared; the runner docstring supplies the full command details.

The people records and videos used on Carmack are not in git. See
[the data map](../FP_INDEX.md#data-that-is-not-in-git) before a remote rerun or
new checkout.

Use `--workers 8` to search independent direction pairs in eight processes.
Results are collected in their original order before choosing courts. Keep the
total CPU allocation at eight cores on Carmack: run one eight-worker view at a
time, or several serial views whose combined allocation stays within that limit.
The runner reports peak memory for the parent and largest worker separately;
those figures do not measure the combined peak of every process.

## Settings

| `Switches` field | Default | Behaviour |
| --- | --- | --- |
| `self_checks` | On | Check intermediate results, including replaying the scoring-stage fit; this is separate from comparison with a saved run |
| `enforce_scene_consistency` | On | Restrict the foot samples to the target's shot |
| `upright_camera` | On | Reject courts requiring a sideways or upside-down camera; the runner's `--any-camera-roll` disables it |
| `geometry_weight` | 0.1 | Share of line support in the final score; the rest is paint support. The runner accepts `--geometry-weight` |
| `workers` | 1 | Processes for independent direction pairs; the runner accepts `--workers` |
| `timing` | Off | Report seconds per step |
| `artefacts_dir` | None | Optionally write intermediate results |

The camera filter skips a direction pair when its horizon tilts more than
45 degrees. It also skips individual courts that imply an upside-down camera.
A horizon more than ten image diagonals away passes both tests.

Seeded templates, the paint-line search, the bounded net reward and stripe
refitting have no individual switches. [STRIPPED.md](STRIPPED.md) records the
research features left out and where they could be restored.

## Code map

| File | Responsibility |
| --- | --- |
| [detect.py](detect.py) | `CourtDetector`, `Switches`, `CourtResult`, and the order of the steps |
| [inputs.py](inputs.py) | Input types and image/box source information |
| [feet.py](feet.py) | Player feet and shot checks |
| [search.py](search.py) | Search settings, paint-like fragments and extra starting points |
| [net_choice.py](net_choice.py) | Final choice and net-post reward |
| [stripe_refit.py](stripe_refit.py) | Adjust stripe labels and refit |
| [run_views.py](run_views.py) | Run the prepared views and compare with saved results |

The search, scoring and geometry code now lives in this package. Imports use
normal package paths and leave `sys.path` unchanged. Saved-view inputs remain
in the older data folders listed in [FP_INDEX.md](../FP_INDEX.md#code-and-input-paths-to-keep-stable).

| Files | Responsibility |
| --- | --- |
| [generation.py](generation.py), [candidate_pool.py](candidate_pool.py), [proposals.py](proposals.py) | Search direction pairs and keep candidate courts |
| [directions.py](directions.py), [line_matching.py](line_matching.py), [prepare_lines.py](prepare_lines.py) | Estimate directions, match markings and prepare fragments |
| [line_templates.py](line_templates.py) | Build candidate courts from crossing lines |
| [scoring.py](scoring.py), [measurements.py](measurements.py), [sampling.py](sampling.py) | Measure and rank courts, refit candidates and share image samples |
| [geometry.py](geometry.py), [candidate_geometry.py](candidate_geometry.py), [camera.py](camera.py) | Court coordinates, transforms and camera checks |
| [court_checks.py](court_checks.py), [players.py](players.py) | Check court shape and whether players fit |
| [line_observations.py](line_observations.py), [stripe_measurements.py](stripe_measurements.py), [stripe_fitting.py](stripe_fitting.py) | Group fragments, measure stripes and fit corners |
| [paint_geometry.py](paint_geometry.py), [net_geometry.py](net_geometry.py), [junctions.py](junctions.py) | Painted markings, projected net and line crossings |
| [image_sources.py](image_sources.py), [search_records.py](search_records.py) | Image/box source types, frozen inputs and search-record checks |
| [paint_profiles.py](paint_profiles.py) | Optional paint diagnostics retained for research callers |

The [27 September archive map](../archive/20260927_code/README.md) records the
former code locations. Historical scripts there keep their original paths;
they need a matching checkout or path repair before a rerun.
