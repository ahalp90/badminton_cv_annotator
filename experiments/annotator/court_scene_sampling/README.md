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
`player_check`, `history_reuse`, `full_search`, `prepared_finish` and
`seed_reuse`.

## Tests

`tests/test_court_scene_sampling.py` runs on CPU with stub search, scoring and
reuse, plus the real detector on minimal inputs. It checks the scheduled
endpoints and shared feet, the stop rule, single searches in `cheap_first`,
selection, timing, histories and the manifest. It also checks that the middle
detection reproduces `detect()`, with and without earlier courts.

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

### Three choices per method

- `paint`: the saved choice, the highest final paint score. The command checks
  that the saved `chosen_position` matches this rule.
- `ranked`: the net choice's combined score, `0.9 * paint + 0.1 * line support +
  0.04 * net reward`, on each saved **final** court. The weights come from
  `detect.py`. Line support is the evidence's `q_geom_span_weighted`. Net posts
  are measured again from the cached lines, prepared as the detector's context
  prepares them. A net that fails to project keeps production's zero reward, and
  its `net_state` records the failure.
- `guarded`: the ranked order with a boundary check. Starting from the top, a
  court loses to any lower-ranked court within 0.01 combined score that has at
  least 0.10 more `q_paint10` on the same outer boundary. The first court no
  lower court vetoes wins, so the lowest-ranked court always survives.

Paint and full-score ties prefer middle, then first, then last. The guard starts
in that order and can veto a candidate even on a score tie. No-court and failed
methods, unanalysed scenes and the baseline stay as they were.

The guard's values are fixed and provisional; they are not tuned:

| Check | Value |
| --- | --- |
| Largest combined-score lead the guard questions | 0.01 |
| Boundary `q_paint10` drop that vetoes | 0.10 absolute |
| Known photometry samples, on both courts | at least 32 |
| Known over visible samples, on both courts | at least 0.75 |
| Projected visible span, on both courts | at least 50 working px |
| Photometry occlusion-aware | on both courts |
| Largest corner distance after matching orientation | 0.15 × the lower court's median edge length |

Both courts must be `comparable` in the middle frame. The guard compares their
`in_middle_frame` corners as saved or turned 180 degrees, whichever is closer.
The four outer boundaries (`far_baseline`, `right_doubles`, `near_baseline`,
`left_doubles`) join corners TL–TR, TR–BR, BR–BL and BL–TL. On a turned court,
the far baseline is compared with the other court's near baseline, and left with
right. A pair that fails a check is recorded as skipped with its reason.
A boundary with too little evidence is recorded as ineligible and cannot veto.

**Missing evidence.** If any accepted court in a method lacks cached lines, a
paint score or line support, that method keeps its saved choice for all three
choices. Its state reads `score_evidence_missing`, and each gap is listed. A gap
never counts as zero, and the method is never ranked on the courts that remain.

### Outputs

- `results.json.gz`: the rule's values, the cache's `line_recovery_seconds`,
  the time spent rescoring, and one row per scene. Each method row holds every
  candidate's frame, role, position, score parts, post evidence and outer
  boundary evidence. It also holds the missing evidence, the three selected
  positions, the ranked order, whether the choice changed and every guard check.
- `comparison.csv.gz`: one row per scene and method with the three choices'
  roles and frames, their combined scores and `vs_baseline` distances, vetoes and
  skipped guard pairs.

### Limits

- Normal selection applies the formula before the stripe refit. Here it applies
  to final courts, so it is not a replay of the detector's choice.
- The guard's 0.01 limit applies to each pair, not the final score loss. Vetoes
  can chain: A can lose to B, then B to C, leaving C more than 0.01 below A.
  This experimental rule does not guarantee that the final choice preserves
  every boundary that caused an earlier veto.
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
