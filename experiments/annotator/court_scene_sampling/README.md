# Three-frame court sampling comparison

This experiment compares three ways to fit a scene's court from the first,
middle and last frames of the midpoint's 3 s player window. A production
baseline runs alongside them. Production sampling is unchanged; nothing here
feeds the annotation pipeline.

The selection rule under test picks the accepted court with the highest final
paint score. **Paint score is not an accuracy measure.** Detector scores,
agreement between frames and agreement with the baseline are diagnostics, not
ground truth.

## Run

From the repository root, in the court detector's GPU environment:

```bash
PYTHONPATH=.:src python -m experiments.annotator.court_scene_sampling.run \
  --manifest manifest.json.gz --output-dir results/ \
  --deeplsd-source DEEPLSD_CHECKOUT --deeplsd-weights DEEPLSD_WEIGHTS.tar
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--device` | `cuda` | DeepLSD and RTMLib device |
| `--template-device` | `cuda` | Line-template scoring with CuPy |
| `--workers` | 8 | Search and scoring workers; the process keeps at most 8 CPU cores |
| `--scene-limit N` | all | Run only each full video's first N scenes, for a smoke check |

The output directory must not exist. `results.json.gz` is rewritten after every
scene, so a stopped run keeps its finished scenes.

Draw courts afterwards with plain 1 px red dashed outlines on the frames they fit:

```bash
PYTHONPATH=.:src python -m experiments.annotator.court_scene_sampling.render \
  --results results/results.json.gz --output-dir overlays/ --scene SCENE_ID
```

Repeat `--scene` for each scene; omit it to draw every court. File names read
`scene__arm__route_role_frame`, with `__chosen` on each method's selected court.
Courts with equal corners on one frame share an image; `index.tsv` lists every
label.

## Manifest

A `.json` or `.json.gz` object. Frame ranges are half-open: `end_frame` is the
first frame after the scene. IDs become file names, so they must be plain names.

```json
{
  "schema": "court-scene-sampling-manifest/1",
  "videos": [
    {
      "id": "video_003",
      "video": "/path/am1.mp4",
      "people": "/path/poses/video_003",
      "scenes": [
        {"id": "003_3076", "start_frame": 1986, "end_frame": 4168, "midpoint_frame": 3076},
        {"id": "003_16208", "start_frame": 4275, "end_frame": 28142, "midpoint_frame": 16208}
      ],
      "later_scenes": [
        {"id": "003_16208_return", "start_frame": 4275, "end_frame": 28142}
      ]
    },
    {
      "id": "video_040_full",
      "video": "/path/match4.mp4",
      "full_video_scenes": "/path/full_video_scenes.json"
    }
  ]
}
```

- `people` is optional: a directory of `pose_{bboxes,kps,ndet}.npy.xz` in native
  pixels. Without it, RTMLib runs live on the frames each scene needs.
- `midpoint_frame` is optional. It replaces the scheduled midpoint,
  `(start_frame + end_frame) // 2`, to reproduce a reviewed frame. Results
  record both frames.
- A video gives either `scenes`, with optional `later_scenes`, or
  `full_video_scenes` alone.
- `full_video_scenes` is the saved cut pass: an object whose `scenes` list holds
  `{"start_frame", "end_frame", "histogram"}` in order. The scenes must cover the
  whole video without gaps. Its optional `scene_seconds` is recorded as a
  one-off cost shared by every arm. Scenes are named `<id>_scene_0000` onwards.

Scenes run in manifest order, then later scenes. Put them in chronological order
when later scenes should test reuse of earlier ones.

## Shared inputs

Each scene's inputs are prepared once, timed, and shared by every arm:

- the midpoint's 31-sample window at 10 fps (`feet.window_frames`), decoded once;
- the midpoint's standing feet from that window, after the grey same-shot check;
- the window's **first, middle and last scheduled frames**, each with its own
  DeepLSD lines and person boxes. The endpoints stay the scheduled window ends
  even when the grey check drops them from the feet;
- run_video's reuse alignment image: the median of the three frames' grey images.

Every frame uses the **midpoint's feet**. An endpoint's window is never
recentred on itself. A scene too short for the window is recorded as
`scene_too_short_for_feet` for every arm, as run_video does.

## Arms

Each arm first tries the courts **it** established in earlier scenes, on the
middle frame, as run_video's optional `--reuse-courts` path does (production
defaults leave it off). A passing court ends the scene for that
arm: no endpoint work runs. Only a scene that reuses none of them reaches the
arm's own work below. Reviewed `scenes` skip this step, so every arm evaluates
them in full.

| Arm | Work after history reuse fails |
| --- | --- |
| `baseline` | Production `CourtDetector.detect()` on the middle frame, with its own feet step |
| `full_three` | Full middle-frame search. After a court, full independent searches of both endpoints |
| `cheap_first` | Search all three frames, finish the frame with the best cheap score, then refit its court on the other two |
| `seed_refit` | Full middle-frame search. After a court, refit it on each endpoint; an endpoint whose refit fails gets a full search |

**Stop rule.** If a method's first evaluation returns no court, it stops for
that scene. That evaluation is the middle frame for `full_three` and
`seed_refit`, and the leading frame for `cheap_first`. Too few standing players
in the shared feet stop every arm before any search, as `detect()` does.

**Selection.** Among a method's accepted courts, the highest final paint score
wins; exact ties go middle, then first, then last. Every frame's result stays in
the results.

**Refits** use the detector's own reuse check (`reuse.try_reuse`): image
alignment, stripe refit, validity, camera, players, upright camera, floor shift
and paint ratio. Within a scene the refitted court's reference image is its own
frame, and each target frame aligns its own image. Earlier scenes' courts align
to the new scene's median image.

### What cheap_first makes cheap

The early phase still runs each frame's context, both line searches and line
templates, with each direction pair fully scoring only its
best 2048 courts by 16-sample line support (`full_score_limit=2048`). The saved
search stays on the frame.

The frame ranking is cheap. It takes the best 16-sample line support among the
frame's search entries and line templates that pass the detector's hard-validity
rule, measured against the frame's wide-family line maps. That costs little next
to the search.

The deferred and costly part is candidate scoring, the net choice and the
stripe refit. `cheap_first` runs it on the leading frame first. It runs it on
another frame only when the lead's court fails that frame's refit check, and
then from the saved search: no frame is searched twice. `cheap_first` differs
from the exhaustive arms through its score limit, so its middle court can
differ from the baseline's.

## Histories

Each arm keeps its own history, as run_video keeps one per video:

- newest first, at most 8 courts;
- at most 3 tried per scene, ordered by nearest scene histogram when the scene
  has one, else newest first;
- only a court found by a search, with positive paint, joins. **Reused courts
  never join**, whether reused from history or refitted within the scene;
- a method's scene adds its best searched court. This is the chosen court unless
  a within-scene refit scored higher. `established.is_chosen` records which;
- a middle-frame court keeps the median image as its reference, as run_video
  does. An endpoint court keeps its own frame's image, which its corners fit.

A fit failure fails that arm's scene alone and adds nothing to its history. Any
other error stops the run and records `stopped_by`.

## Frame coordinates

Every result's corners are in the native pixels of **the frame they fit**. To
compare an endpoint court with the middle frame, `in_middle_frame` carries it
across with the ECC image alignment behind the reuse check. It records the
correlation and the largest corner movement; no motion limit applies.
`comparable` is false when ECC fails or the correlation is below the reuse
check's 0.8. `vs_baseline` compares only comparable courts, as the largest corner
distance in pixels and the largest floor movement in metres. It measures
disagreement with the baseline, not error.

## Timing

Models load once (`cold_setup_seconds`). Measure timings on one machine with the
same GPU and at most 8 cores.

- `detector_walltime_measured`: the arm's own measured wall time for its
  detector work on the scene. It excludes the shared inputs.
- `shared_inputs_charged`: the shared measurements this arm needed: the window
  decode, feet and median image, plus lines and boxes for its `frames_used` only.
- `assembled_total_not_walltime`: their sum. **It adds separate measurements and
  is not an end-to-end wall time.**
- The baseline's `detect()` repeats the feet step on already-read frames and
  people. That repeat is subtracted from its detector time; the raw call is
  `detect_walltime_measured_with_repeat_feet`.
- `diagnostics_not_charged`: evidence and alignment for the results, not
  charged to any arm.
- A full video's `cut_pass_seconds_shared` is the saved cut pass's one-off time.

The first scene can carry one-off warm-up, such as GPU kernel compilation, in
whichever arm runs first (the baseline).

## Results

`results.json.gz` holds `settings`, `cold_setup_seconds` and one row per video.
A video row holds its `scenes`, `later_scenes` and each arm's final `histories`.

A scene row holds its range, scheduled and used midpoint, `status`,
`history_reuse`, the shared `window` and `input_seconds`, then `baseline` and
`methods`. Reviewed scenes add `baseline_agreement`. There, `full_three` and
`seed_refit` must reproduce `detect()` exactly on the middle frame.

A method row holds:

- `known_courts_tried`, `status`, `stopped_after_first_evaluation` and `frames_used`;
- `frames` in the order they ran, and `chosen_position` within them;
- `seed_view_id`, `established` and `seconds`;
- `cheap_frame_scores` and `leading_role`, for `cheap_first` after its early phase.

Each frame row holds its role, frame, `route`, corners, paint score, stage
seconds and reuse records. An accepted court adds its per-marking paint
`evidence`, `in_middle_frame` and, when comparable, `vs_baseline`. Routes are
`player_check`, `history_reuse`, `full_search`, `prepared_finish` and `seed_reuse`.
Like the detector, required-player runs stop before reuse or search when the
shared feet samples contain too few people to satisfy the occupancy rule.
Diagnostic artefact runs still search.

## Tests

`tests/test_court_scene_sampling.py` runs on CPU with stub search, scoring and
reuse, plus the real detector on minimal inputs. It checks the scheduled
endpoints and shared feet, the stop rule, single searches in `cheap_first`,
selection, timing, histories and the manifest. It also checks that the middle
detection reproduces `detect()`, with and without earlier courts.

`tests/test_court_scene_rescore.py` and `tests/test_court_scene_compose.py` cover
the saved-fit tools. The compose tests draw a painted court through a pinhole
camera, with line fragments and person boxes. They check the warp direction and
scale, alignment without people, turned courts, donated samples, fallbacks, tie
and missing-score rules, and one composite end to end.

## Rescoring saved fits

`rescore.py` re-ranks each method's saved accepted courts without running a new
search. It reads a finished or partial `results.json.gz` and a lines cache, and
writes to a new directory. The source results stay unchanged.

```bash
PYTHONPATH=.:src python -m experiments.annotator.court_scene_sampling.rescore \
  --results results/results.json.gz --lines-cache lines.json.gz --output-dir rescored/
```

The lines cache holds the DeepLSD fragments of every accepted frame, keyed by the
frame row's `view_id`. Its native size must match the video's.

Recover missing lines on a machine with the source videos and DeepLSD available:

```bash
PYTHONPATH=.:src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.annotator.court_scene_sampling.recover_lines \
  --results results/results.json.gz --output lines.json.gz \
  --deeplsd-source /path/to/DeepLSD \
  --deeplsd-weights /path/to/DeepLSD/weights/deeplsd_md.tar
```

This runs line inference once per accepted source frame, without court searches
or player detection. Add `--frames-dir frames/` to save original PNGs for local
outline checks. Model setup and evidence recovery times are stored separately.

```json
{"schema": "court-scene-lines/1", "recovery_seconds": 12.3,
 "frames": {"video_005_13361_middle_frame_13361": {"native_size": [1920, 1080],
                                                   "segments_native_px": [[100.0, 200.0, 400.0, 210.0]]}}}
```

### Two choices per method

- `paint`: the saved choice, the highest final paint score. The command checks
  that the saved `chosen_position` matches this rule.
- `ranked`: the net choice's combined score, `0.9 * paint + 0.1 * line support +
  0.04 * net reward`, on each saved **final** court. The weights come from
  `detect.py`. Line support is the evidence's `q_geom_span_weighted`. Net posts
  are measured again from the cached lines, prepared as the detector's context
  prepares them. A net that fails to project keeps production's zero reward, and
  its `net_state` records the failure.

Paint and full-score ties prefer middle, then first, then last. No-court and
failed methods, unanalysed scenes and the baseline stay as they were.

**Missing evidence.** If any accepted court in a method lacks cached lines, a
paint score or line support, that method keeps its saved choice for both
choices. Its state reads `score_evidence_missing`, and each gap is listed. A gap
never counts as zero, and the method is never ranked on the courts that remain.

### Outputs

- `results.json.gz` (schema `court-scene-rescore/2`): the rule's values, the
  cache's `line_recovery_seconds`, the time spent rescoring, and one row per
  scene. Each method row holds every candidate's frame, role, position, score
  parts and post evidence. It also holds the missing evidence, both selected
  positions, the ranked order and whether the choice changed. Schema
  `court-scene-rescore/1` outputs also carry the retired boundary guard.
- `comparison.csv.gz`: one row per scene and method with both choices' roles and
  frames, their combined scores and `vs_baseline` distances.

### Limits

- Normal selection applies the formula before the stripe refit. Here it applies
  to final courts, so it is not a replay of the detector's choice.
- Rescoring picks among the courts each run saved. `cheap_first` chose its
  leading frame before any court was final, and frames it never fitted cannot
  enter the ranking.
- A different winner could have donated a different court to later scenes.
  Rescoring keeps the saved histories, reuse and timing, so it cannot show the
  quality or timing of that other chronological run.
- The source run's timings stay as measured. Line recovery and rescoring times
  are reported apart from them.
- Rescoring ranks by detector evidence. Paint and line support are not accuracy
  measures, and `vs_baseline` measures disagreement, not error.

## Composing one court per scene

`compose.py` fits one extra court per scene for `full_three`, from the best-painted
markings across its accepted frames. It then scores that composite and the saved
courts on the same frames. It runs no search and writes to a new directory; the
source results and caches stay unchanged. Other methods keep only the rescore
comparison.

```bash
PYTHONPATH=.:src python -m experiments.annotator.court_scene_sampling.compose \
  --results results/results.json.gz --lines-cache lines.json.gz \
  --people-cache people.json.gz --frames-dir frames/ --output-dir composed/
```

`--frames-dir` holds each accepted frame's source PNG as `<view_id>.png`, as
`recover_lines.py --frames-dir` writes them. The people cache holds same-frame
person boxes from `recover_people.py`:

```json
{"schema": "court-scene-people/1", "recovery_seconds": 4.2,
 "frames": {"video_005_13361_middle_frame_13361": {"native_size": [1920, 1080],
                                                   "boxes_native_px": [[810.0, 300.0, 900.0, 520.0]]}}}
```

Every PNG, line entry and box entry must match the video's native size.

### Steps

1. **Reference.** rescore's `ranked` court sets the reference frame. The
   reference supplies coordinates and the fit's starting court only.
2. **Alignment.** Each other accepted frame is ECC-aligned to the reference inside
   its own saved court, as `in_middle_frame` does, but without people. Players
   move between samples, so their pixels disagree even when the camera is still.
   Each image's same-frame person boxes are cut from that image's own ECC mask:
   the frame's from its court mask, the reference's from an otherwise full mask.
   ECC carries the reference's mask through the current warp on every step, so a
   moved camera moves the reference's boxes with it. A warp is usable when its
   correlation reaches the reuse check's 0.8. Camera movement is allowed. A frame
   without a usable warp is skipped and recorded. That includes masks that leave
   ECC too little to converge, which read as `alignment_unmeasurable`.
3. **Orientation.** A court turned 180 degrees against the reference gets its
   corners rolled by two, so each marking name means the same painted line.
4. **Evidence.** Each used court's marking evidence and fragment assignments are
   measured again in its own frame. Its context is rebuilt from the PNG, cached
   lines and same-frame boxes, so boxes hide paint as in the run. No feet are
   known. Each accepted court's saved evidence is also measured again as saved,
   and `reproduction` records the largest score difference and any count
   mismatches.
5. **Donors.** Each marking comes from the used court with the most `q_paint10`
   on it; ties follow own-frame score order. A marking with no positive
   `q_paint10` anywhere contributes nothing. The donor's observed fragment samples
   (`stripe_fitting.prepare`, with their assigned stripe centre or edge) are
   carried into reference pixels. Samples inside a person box are dropped. Each
   sample keeps prepare's weight times the donor's `q_paint10` on that marking.
6. **Fit.** `stripe_fitting.refine` fits one court to all donated samples.
   `stripe_refit.fit_geometry` checks the solver, rank, depth, convexity and hard
   validity, then the camera must be plausible and upright. Player positions are
   not checked. A failed fit is recorded and the saved courts remain.
7. **Comparison.** Every used frame's saved court and the composite are carried
   into every used frame and scored there with rescore's formula, on that frame's
   lines, paint and boxes. The highest mean over those same frames wins. An exact
   tie keeps a saved court, then goes middle, first, last. If any accepted saved
   court lacks a mean, including one whose frame did not align, the own-frame
   full-score choice stands.

A scene with fewer than two used frames keeps the own-frame full-score choice.
Missing score evidence or no court preserves the original saved outcome.

### Outputs

- `results.json.gz` (schema `court-scene-compose/2`): the rule's values, cache
  recovery times, compose time and one record per scene. A composed scene holds
  each frame's alignment, orientation and reproduction; each marking's donor,
  samples, occluded samples and weight; the fit and its checks; every
  candidate's per-frame scores and mean; and the choice.
- The choice records the saved `paint` choice, the `own_frame` (rescore
  `ranked`) choice, the `common_frame_original` winner, whether the composite
  beats every saved court, the `final` court and `changed` against the paint
  choice. Final corners are given in their own frame and, when they can be
  carried there, in the reference frame's native pixels.
- Each alignment records `mask_kept_fraction`: the share of the court mask that
  both masks keep at the identity warp, where ECC starts. Schema
  `court-scene-compose/1` outputs aligned with players inside the mask.
- `comparison.csv.gz`: one compact row per scene.
- `outlines/`: for each scene with a valid composite, every compared court as a
  plain 1 px red dashed outline on the reference frame, one PNG per court.

### Limits

- Scores measure detector evidence. A higher common-frame mean is not ground
  truth; judge the outlines.
- The comparison can change the choice among saved courts even when the
  composite loses, because it averages over frames rather than using each court's
  own frame.
- Stripe positions come from each saved court's own assignment, without the
  stripe refit's colour-polarity pass. A saved court a few pixels off can read
  some centre fragments as edges, which leaves up to half a stripe of error.
- A donor whose marking has positive `q_paint10` but no strongly assigned
  fragments contributes no samples; no other frame stands in.
- Recovered lines or boxes can differ from the run's. Scores then differ from
  the saved ones, and `reproduction` shows by how much.
- Cutting players out leaves ECC less texture to lock onto. In one synthetic
  pair, half the court was masked and the camera moved about 9 view px. ECC then
  reached a warp 40 native px wrong that still passed 0.8. Without masking, that
  pair failed to align at all. The correlation check alone does not catch this.
- The court polygon masks the frame's own image here. `measure_view_alignment`
  applies it to the reference image; the two differ only by the camera's
  movement.
