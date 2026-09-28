# Run the court detector

`CourtDetector` finds a badminton court in a prepared image. It runs the
search and scoring steps in memory and returns four court corners, or a reason that no
court was found. This page owns its inputs, behaviour, settings and commands.

Read [pickup.md](../pickup.md) for current work,
[the decisions](../DETECTOR_DECISIONS.md#d19) for measured results, and
[PERFORMANCE.md](PERFORMANCE.md) for the speed-up design.

## Inputs and result

`CourtDetector.detect(view, people, frames)` takes three inputs from `inputs.py`:

- `view` (`ViewInputs`): an image, its detected line fragments as float32
  native pixels, person boxes, represented video frame, scene boundaries, and
  the source frames for the image and boxes. The detector uses that source
  information to avoid hiding people in the wrong image
- `people` (`PeopleSource`): boxes and 17 COCO-layout body joints for people
  in nearby video frames
- `frames` (`FrameReader`): access to those nearby video frames

A caller supplies these inputs directly, or uses the video runner below to
build them from video frames, line detections and people detections.

The returned `CourtResult` contains:

| Field | Meaning |
| --- | --- |
| `corners_native_px` | Four float64 corners in the original image's pixels, or `None`. Corners can lie outside the image |
| `no_court_reason` | Why no court was returned. `no_gated_court` means none passed the court checks; `rank_deficient` means the final fit lacked enough independent information |
| `chosen_key` | The saved identifier of the chosen court |
| `stage_seconds` | Time per step when timing is enabled |
| `paint_score` | Paint support after the final stripe fit, for checking later reuse |
| `reused_from` | Earlier view used for checked reuse, or `None` for a full search |

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
   Run once with all fragments (`all_lines`), and once with only paint-like
   fragments (`painted_lines`). Older saved records and notes call these G0 and
   G1. The second search uses pale lines that are brighter than their
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
   the frames. The code calls this stage scoring; older records and notes
   call it W5
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

People are required by default. For a plausible fallback without boxes or
keypoints, use `Switches(require_people=False)` and pass `people=None` with an
empty `(0, 4)` box array in the view. The detector then uses line-template
proposals, court geometry, camera checks and the usual stripe fit. Player
measurements remain unavailable rather than being treated as passing checks.
This fallback has not been tuned for the same precision as detection with people.

Required mode skips the candidate search when the sampled people cannot meet
the existing occupancy checks. Runs with full artefacts still build candidate
records so saved-run comparisons can inspect them.

When people are supplied in optional mode, their boxes still mask occlusions
and support the existing proposal search. Missing or off-court people do not
veto the final choice. Short scenes use their anchor's people when available.

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
research run. The baseline arm uses the older names: G0 and G1 for the two
searches and W5 in some field names. The runner reads them in the current names
and leaves the saved files unchanged. To reproduce the original 25 September
comparison, also use `--any-camera-roll --geometry-weight 0`. The newer defaults
intentionally change results. The [original check](check_20260925/README.md)
states exactly what was compared; the runner docstring supplies the full command
details.

The people records and videos used on Carmack are not in git. See
[the data map](../FP_INDEX.md#data-that-is-not-in-git) before a remote rerun or
new checkout.

Use `--workers 8` to search direction pairs and score candidates in eight processes.
These stages run in sequence. Results are collected in their original order before choosing courts. Keep the
total CPU allocation at eight cores on Carmack: run one eight-worker view at a
time, or several serial views whose combined allocation stays within that limit.
The runner reports peak memory for the parent and largest worker separately;
those figures do not measure the combined peak of every process.

Both runners keep one set of worker processes for the whole run, so each worker
starts and imports the detector once. The workers close when the run ends,
including when it fails. A script that calls `CourtDetector.detect` itself
shares workers the same way inside `with CourtDetector(switches) as detector:`.
Without the `with` block, each search and scoring step starts its own workers.
A worker holds the last view it scored until it scores another view or closes.
Parallel candidate scoring writes one temporary input pickle per view. Each
worker loads it once into memory; results return through the process pool.
Workers never write to the file, which is removed after the tasks finish.
The saved-view batch runner replaces a crashed pool before trying the next view.

Direction-pair support scoring uses a serial Numba kernel. Numba is a project
runtime dependency. The first call compiles and caches the kernel; later
processes load that cache. It adds no threads beyond the selected worker count.
Bulk scoring arrays and responses use float32. Fitting, sensitive geometry and
the compiled scorer's small projection/clipping calculations use float64.

Both runners accept `--template-device cuda` as an explicit option. It scores the
line-template hypotheses on a GPU with CuPy. Rectangle setup, admission and the
court checks stay on the CPU, and other stages keep their own settings.
CuPy is not a project dependency, so install it where a GPU is available. The
detector stops at setup when CuPy or a CUDA GPU is missing; it never falls back
to the CPU. GPU rounding can change the last bits of results, such as each line
template's `camera_error_before_candidate_gates`. It can also occasionally move a
line sample to a neighbouring pixel. Saved-run comparisons report any such
differences.

## Run from video

The video caller can run DeepLSD for lines and read frame-aligned RTMLib pose
extracts. With the default `--require-people`, omit `--people` to run the project's
RTMLib extractor on the requested frames. Both live models load once.
Use `--no-require-people` without `--people` to skip pose inference and use the
line-only fallback. This mode also analyses scenes shorter than the usual
three-second people window. Supplying `--people` still uses those saved inputs,
but leaves the player checks optional. The saved-view runner supports the same
gate toggle while retaining its required fixture directory.

Live pose extraction temporarily reuses `bst_x.preparing_data.rtmlib_pose`,
which also imports BST-X pipeline configuration. This is a practical compromise.
Move the reusable extractor and its pose-format constants into shared code so
both callers can use it without that dependency. The dependency is loaded only
when the live-pose adapter is constructed.

DeepLSD uses gradient validation by default. This matches the saved amateur
example's line extract; the less selective hard variant remains available through
`DeepLSDLines(..., grad_nfa=False)`.

```bash
PYTHONPATH=.:src python -m scratch.court_det_fix.court_detector.run_video \
  --video VIDEO.mp4 \
  --deeplsd-source DEEPLSD_CHECKOUT \
  --deeplsd-weights DEEPLSD_WEIGHTS.tar \
  --people POSE_DIR \
  --pyscenedetect --workers 8 \
  --output courts.json.gz
```

`POSE_DIR` contains `pose_bboxes.npy.xz`, `pose_kps.npy.xz` and
`pose_ndet.npy.xz`. They must use source-video frame indices and native image
pixels. The reader drops padded detection slots using `ndet`.

For saved-line trials, replace the DeepLSD arguments with `--saved-lines FILE`.
That file is a gzipped JSON object mapping source frame numbers to `(N, 4)`
native-pixel line arrays, read as float32. Missing frame entries raise an error.

PySceneDetect is optional. Its adapter uses the existing ContentDetector cut
settings and can supply a normalised luminance histogram for each scene.
Histograms can prioritise camera comparisons; they do not establish that courts
match. The `SceneSource` and `LineSource` protocols allow other input providers.
The prepared-image detector needs neither PySceneDetect nor DeepLSD installed.

Use `--scenes FILE` for a gzipped list of inclusive `[first_frame, last_frame]`
ranges instead of detecting cuts. With neither scene option, the caller analyses
one middle frame from the whole video. It analyses one middle frame per supplied
scene and keeps foot samples inside that scene. Short scenes that cannot hold
the existing foot window receive `scene_too_short_for_feet`; they remain
unanalysed. Source codecs need a seek-versus-sequential frame check before
benchmarking a new dataset.

The output separates model/input setup, scene detection and per-scene work.
It records whether people and lines were supplied from saved extracts, so those
timings are not mistaken for complete live inference.

`--reuse-courts` enables an optional trial for returning camera views. The caller
keeps up to eight recent courts found by full searches and tries up to three
for each later scene. Histograms order those attempts when available; otherwise
the most recent court comes first. A reused court never becomes a new template.

Video reuse aligns three-frame median images to reduce moving-player interference.
The samples are the first, middle and last frames of the existing feet window.
Optional-people scenes shorter than that window use their scene endpoints and
middle frame. One image is shared by the alignment attempts and any new stored
reference. Court search and stripe refitting still use the actual middle frame.

Each attempt aligns the images, refits the court to the current stripes, and
checks the current geometry, camera, players' feet and paint support. It also
bounds the refit's movement in court metres, including the far baseline. Failed
attempts fall through to the full search using the same prepared inputs. The
paint-ratio and movement limits are provisional until checked on representative
video pairs. Reuse stays off by default during that evaluation.

Prepared-image callers can pass `known_courts=` to `detect()`. Build each entry
with `reuse.make_known_court()` from a fully searched result and its input frame.
These callers use the raw image unless they supply `ViewInputs.alignment_image`
and the matching `alignment_image=` when building a reference. Prepared alignment
images are greyscale uint8 at `court_views.VIEW_RESOLUTION`.

## Settings

| `Switches` field | Default | Behaviour |
| --- | --- | --- |
| `self_checks` | On | Check intermediate results, including replaying the scoring-stage fit; this is separate from comparison with a saved run |
| `enforce_scene_consistency` | On | Restrict the foot samples to the target's shot |
| `upright_camera` | On | Reject courts requiring a sideways or upside-down camera; the runner's `--any-camera-roll` disables it |
| `geometry_weight` | 0.1 | Share of line support in the final score; the rest is paint support. The runner accepts `--geometry-weight` |
| `workers` | 1 | Processes for direction pairs and candidate scoring; the runner accepts `--workers` |
| `full_score_limit` | `None` | Optional trial: score each pair's courts with 16 samples, then fully score only the best given number with 64 samples. Both runners accept `--full-score-limit`; omitting it keeps exhaustive scoring |
| `template_device` | `"cpu"` | Line-template scoring device; `"cuda"` uses CuPy on a GPU and fails at setup without one. Both runners accept `--template-device` |
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
| [line_templates.py](line_templates.py), [template_arrays.py](template_arrays.py) | Build candidate courts from crossing lines; score them with NumPy or CuPy |
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
