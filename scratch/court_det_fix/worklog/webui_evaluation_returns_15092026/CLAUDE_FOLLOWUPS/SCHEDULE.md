# Follow-up schedule

Written 2026-09-15. A fresh session should read this file first, then `status.md` in this folder, then the remote receipts named below.

Decision on 2026-09-15 evening: run the three local checks plus two added checks, "where the close courts vanish inside the matcher" and "lens distortion on GX". Hold the per-view audit until those answer. The reasoning is in the conversation summary at the top of `status.md`.

The follow-ups come from `../CLAUDE_EVALUATION.md`. Briefs for each delegated check sit in `briefs/`. The WebUI bundles are extracted under `_bundles/` for convenience; they duplicate the zips one level up and can be deleted.

## Added checks

### Where the close courts vanish inside the matcher

On three view-arms the direction fit is within 5 px of the control yet the pool holds nothing within 20 to 44 px: M on GX0, R on Amateur-3 and B on Amateur-2 frame 28019. Nobody knows whether the per-pair cap of 256 discarded closer courts. Copy the matcher scripts into a new remote folder, record every proposed court's corners per pair before the cap into a side file, and rerun those three case-arms on the compute host. The instrumented run must reproduce the saved generation record exactly; the side file is analysed afterwards against the frozen control. Brief: `briefs/cap_loss.md`.

### Lens distortion on GX

Line bowing has no verified cause. Use the merged rows with many member fragments spanning a long extent, fit a straight line and a quadratic to the member endpoints, and report the sagitta and its sign relative to the image centre. A consistent outward sagitta that grows with radius means barrel distortion. Run the same test on the amateur and ShuttleSet views as controls. Brief: `briefs/distortion.md`.

## Part two is on hold

The per-view audit below stays written down and unrun until the GX direction check and the two added checks report.

## Where things live

Local repository: `<repo>`, branch `fix/court-det`. Every path below that starts with `scratch/` is relative to that root.

Local helper tree, called L in the older records: `scratch/court_det_fix/worklog/checks/independent/player_guided/20260914/`. It holds `vp_pruning/`, `automatic_axes/`, `axis_matching/` and `marking_diagnosis/`. The direction experiment lives in `scratch/court_det_fix/direction_agreement/` and its run records under `runs/direction_agreement_20260915_144900/`.

Remote host: the compute host, 32 cores, 375 GB memory, one L40 GPU. Remote experiment root, called R in the older records: `R/`. The direction experiment's copy sits at `R/direction_agreement/`. The shared helper modules sit beside it as `vp_pruning_20260914/`, `automatic_axes_20260914/`, `axis_matching_20260914/` and `marking_diagnosis_20260914/`. The nine frozen input packs, images and the legacy zone module are already there; `R/direction_agreement/runs/<run>/manifest.json.gz` lists each with its MD5.

Videos on the compute host: `<remote>/court_det_fix/player_guided_20260908/videos/` holds `am2_h264.mp4` and `am3_h264.mp4`. The GX source is `gxBQ_HwdgN4.mp4`, 1920 by 1080, about 59.885 frames per second, 90,971 frames. The GX extension README says it was read from `scratch/court_det_fix/worklog/checks/independent/examples_updated/` under the repository checkout on the compute host at `<remote>/court_det_fix/player_guided_20260908/repo/`. Confirm with `find` before use.

Front-end tools on the compute host:

- People boxes: `python -m experiments.annotator.independent_court.export_people` under `~/.venvs/venv-rtmlib/bin/python`, with `--device cuda --sample-fps 10 --score-min 0.2`. It decodes the frames named in a windows manifest, writes native PNGs and a `line_manifest.json.gz`.
- Lines: `python -m experiments.annotator.independent_court.export_lines` under `<remote>/court_det_fix/independent_detector/deeplsd_env/bin/python`, with `--model deeplsd-md --source .../DeepLSD --weights .../DeepLSD/weights/deeplsd_md.tar --device cuda`. It reads the line manifest the people exporter wrote.
- Pack adapter: the GX extension used `scratch/court_det_fix/worklog/checks/independent/player_guided/20260909/gx_extension/prepare_inputs.py`. It joins images, lines and people samples into `inputs.json.gz` with a `cases` list and a `references` map. Read it before reusing it; it applies a three-second centred people patch and expects the annotation CSV for references, which the new calibration frames do not have.
- Direction selection: `vp_pruning/run_population.py --inputs <pack> --output <dir> --pencil-selection coverage`. This is the baseline selection the experiment calls B.
- Court generation and rescoring: `automatic_axes/run_automatic.py` and `automatic_axes/rescore_camera_pool.py`. The experiment's `direction_agreement/run_matcher.py` shows how to call them with the legacy zone module and how the two rescoring stages chain.
- Scoring a court on a frame: `run_automatic.evaluate_pool(source, entries, observations, size, segments, families, zone, root)`. It takes saved entries with `homography_working` and `corners_px` and the target frame's prepared observations. Nothing ties the entries to the frame they came from, so it scores courts across frames as it stands.

The `20260908/gpu_job.sh` script shows the exact environment variables, library paths and invocation order for the two exporters.

## Remote access

Read `~/.codex/remote_hpc.md` before touching the compute host. The rules that matter here:

- Run commands with the remote shell wrapper on the compute host. Copy files with the remote copy wrapper with explicit paths.
- Keep one remote session alive at a time. Poll or end it before starting another.
- If the connection fails, report it. Do not switch to another host.
- `/scratch` is local to the compute host. `~` is shared across hosts.

Launch long jobs detached, the way `direction_agreement/sync.sh launch` does: `nohup setsid bash run_remote.sh ...` inside braces so the SSH channel closes. Each job writes a PID file, a log and an exit-code receipt under its run folder. Check liveness with `kill -0`. Copy `direction_agreement/run_remote.sh` and `sync.sh` into the new remote folder and change the folder name and PYTHONPATH; do not edit the originals.

Worker settings: every worker runs single-threaded, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, and OpenCV set to one thread inside the script, so results replay exactly. Scale by process count, not threads. Start workers with `nice -n 10`. Use at most 20 workers at once, which leaves 12 cores free. Memory is not a constraint; measure the first generation worker with `/usr/bin/time -v` and record its peak.

Remote folder for this work: `R/per_view_audit/`. Local mirror: `CLAUDE_FOLLOWUPS/per_view_audit/`. Pull records, crops and traces only. Decoded frames stay on the compute host.

## Part one: local checks

Run these first. They read existing records and write only under `CLAUDE_FOLLOWUPS/`.

### Which structures the three GX0 merged rows lie on

The WebUI showed that merged rows 31, 51 and 61 of GX0 are the rows that make the capped-angle score prefer the original direction pair over the control pair. Nobody has looked at what those rows are.

Inputs: `runs/.../e0/gxBQ_window_00_frame_0.json.gz` holds each merged row's member fragment IDs. The GX0 input pack holds the fragments in native pixels, and the native frame PNG sits beside it; `direction_agreement/manifest.py` names both. The E2 record for GX0 holds the support masks of B's candidates 737 and 4104 for contrast.

Method: draw the member fragments of rows 31, 51 and 61 on the native frame in one colour and the fragments of the control pair's support rows in another. Save the full overlay and a crop around each of the three rows. Look at the crops and write down what each row follows: court paint, a mat edge, a wall line, a net post, a person, or something else.

Output: `gx0_rows/overlay.png`, `gx0_rows/row_31.png`, `row_51.png`, `row_61.png`, and `gx0_rows/note.md` with the read and its source. The read is a visual judgement by the session; say so.

Decides: whether GX0's loss comes from a wrong structure in the merge or from a wrong orientation on real paint. Only the second could ever respond to an anchor or score change.

### Do later GX frames keep the precise directions

The per-view route can only help GX if some automatic selection along the video keeps directions close to the control pair.

Inputs: `experiments/annotator/independent_court/recorded/player_guided/projective_patterns/evaluation/temporal_records.json.gz` at commit `d0c9a12`. Each of the seven GX rows carries `direction_estimator.points_working`, the 16 selected directions in working homogeneous coordinates, and `direction_estimator.retained_candidate_ids`. The control pair's points come from the WebUI bundle `badminton_direction_audit_7299ff3.zip`, file `gx0_reconstructed_analysis.json`, or from replaying the bank with `automatic_axes/diagnose_direction_bank.reconstruct_bank` on the GX0 lines; the E2 record's `points_working` for B gives the coordinate convention.

Method: for each GX frame and each of its 16 directions, compute the angle between the normalised 3-vectors to candidate 1183 and to candidate 122 and keep the minimum per control direction. Also compute the WebUI's incidence bound `max(Lx(U), Ly(V))` from `geometry_certificate.py` against GX0's approved control corners for the best pair in each frame. Label the bound as valid for frame 5 and indicative for later frames, because the camera may have moved.

Output: `gx_directions/table.csv` with one row per frame, and `gx_directions/note.md`.

Decides: if no frame keeps both control directions within about a degree, pooling automatic selections cannot repair GX and the GX views need a different lever. If some frame does, the per-view audit has a real chance on GX.

### Identity diagnostics on the exported winners

Inputs: `evaluation/ranking_records.json.gz` at `7299ff3`, which holds the 17 distinct automatic winners and the two GX0 controls with homographies and corners. The WebUI's `court_identifiability_extension_7299ff3.zip` holds `audit.py`, which already replays marking availability with the 12-pixel clipping rule, computes the named-line constraint rank and measures projected separation between markings for nine candidates.

Method: extend that script to all 19 candidates, one row each: available interval count, constraint rank, minimum projected separation between distinct markings inside the image, and the existing visual ruling. Keep its clipping and rank code unchanged.

Output: `identity_diagnostics/table.csv` and `identity_diagnostics/note.md`.

Decides: whether "unconstrained extent" and "aliased markings" show up on the other false winners and stay absent on the approved courts. This is a pattern check, not a rule. The WebUI already showed neither measure vetoes on its own.

## Part two: the per-view audit on the compute host

The question: with directions and the matcher frozen at the baseline, does scoring one union of automatically generated courts across several frames of the same view choose a better court than scoring on the evaluation frame alone?

### Fixed choices, written before anything runs

- Windows, inclusive, with the frames that calibrate and the frames that evaluate:

| Video | Window | Calibration frames | Evaluation frames |
| --- | --- | --- | --- |
| GX | 0 to 90 | 30, 60, 90 | 0, 5 |
| Amateur-2 | 150 to 240 | 180, 210, 240 | 150 |
| Amateur-2 | 28019 to 28109 | 28049, 28079, 28109 | 28019 |
| Amateur-3 | 0 to 90 | 30, 60, 90 | 0 |

- Direction selection stays the baseline coverage rule. No midpoint anchor, no precision representative.
- The union for a window holds every shortlisted court from every frame in the window whose camera error is within the existing bound, taken before the global cap. That is the set the camera-first stage sees before it caps to 256. It runs to a few thousand courts per frame.
- Each union court gets the full evidence on every frame of its window through `evaluate_pool`.
- Three selections per evaluation frame. Native: the frame's own baseline winners, which already exist for the five anchors. Matched: the winner from the union scored on the evaluation frame alone. Shared: the winner from the union by the median of its score over the three calibration frames.
- Both rankings run: line, which is the exclusive stripe score with paint as tie-break, and paint, which is the profile score with line as tie-break, both as `run_automatic.winner_ids` defines them. Line is primary because the WebUI protocol says so; paint is reported beside it because the direction experiment showed line ranking picks the mat border on both Amateur-2 views. Ties break by frame index then candidate ID.
- Camera stability check. Decode every frame in each window. Detect ORB features on the working-size greyscale image outside the people boxes, dilated by 20 working pixels. Match each frame to the window's first evaluation frame and fit a similarity transform with RANSAC at a 2-pixel threshold. The frame passes when at least 60 inliers remain and the median displacement of a 6 by 4 grid of working-pixel points under the fitted transform is under 2 pixels. Any other result, including too few features, counts as moved. A window is eligible only when every frame in it passes. No substitute frames.
- References are read only after the three selections are written to disk. The reference for each evaluation frame is the same frozen control the direction experiment used, listed in `direction_agreement/evidence.md`.
- Acceptance is reported as not established. The floor gate outcome is recorded per winner and not used.

### Steps in order

1. **Confirm the frame chain reproduces an anchor.** Decode GX frame 0 and Amateur-2 frame 150 from the videos with the same decoder the people exporter uses. Compare each PNG's MD5 with the frozen image MD5 in the direction experiment's manifest. Then run the line exporter on those two frames and compare the segment arrays with the frozen packs; report the largest endpoint difference. Stop if a frame hash differs. Note and continue if segments differ by under a pixel, because DeepLSD on a GPU need not be bit-stable.

2. **Decode the windows and check camera stability.** Write a windows manifest covering the four windows and run the people exporter on it with `--sample-fps` set high enough to keep every frame, or decode the 364 frames directly with OpenCV if the exporter's sampling cannot be forced. Run the stability check on every frame. Write `stability/trace.csv` with one row per frame: inlier count, median displacement, pass or fail. Stop for any window that fails and say which frames failed.

3. **Build inputs for the twelve calibration frames.** Run the people exporter and the line exporter on those frames. Adapt the GX extension's `prepare_inputs.py` so it builds a pack without annotation references; keep its people patch and box selection unchanged so the twelve frames get the same `all_feet_px` treatment as the anchors. Write one pack per video.

4. **Select directions and generate courts.** Run `run_population.py` with coverage selection on the twelve cases. Then run the generation step for each case as `run_matcher.py` does it, twelve workers at once, each single-threaded. Reuse the five anchors' existing baseline generation records; do not regenerate them. Expect 14 to 38 minutes per frame.

5. **Score every union court on every window frame.** Build each window's union from the generation records: walk each pair's shortlist, compute or read the camera error the way `rescore_camera_pool.rescore` does, keep courts within the bound, and give each a key of source frame plus candidate ID. Split the work by target frame and by slices of about 500 courts, run up to 20 workers, and write one record per slice. Expect about a tenth of a second per court per frame.

6. **Select without references, then measure.** A script that never opens a reference file writes the three selections for each evaluation frame under both rankings, with the full score vector of every winner. A second script then measures maximum corner distance to the frozen control in working pixels for every winner, and also for the nearest union court, so availability and ranking stay separate.

7. **Write up.** One results file, bottom line first. Report per evaluation frame: nearest union court, native winner, matched winner, shared winner, under both rankings. Amateur-3 regressing is a finding, not a stop. Say where the WebUI's expectation held and where it did not.

### Stop conditions

Stop and report when an anchor frame hash differs from the frozen image, when a window fails the stability check, or when any generation worker exits non-zero. Everything else is a result.

## Output layout

```
CLAUDE_FOLLOWUPS/
  SCHEDULE.md              this file
  status.md                what has run, what is running, what is next; updated at every step
  manifest.json            inputs with MD5, commands, start and end times, exit codes
  gx0_rows/
  gx_directions/
  identity_diagnostics/
  per_view_audit/
    protocol.md            the fixed choices above, copied before step 2 runs
    stability/
    packs/
    generation/            pulled records only
    scoring/
    selections/
    measured/
    results.md
```

Cruft: anything under `per_view_audit/generation/` and `scoring/` is large and re-derivable from the packs and code. The final note at the parent level, `../CLAUDE_FOLLOWUPS_NOTE.md`, says which files a reader needs and which can be deleted.

## Picking up in a fresh session

1. Read this file, then `status.md`.
2. Check the remote receipts with the remote shell wrapper on the compute host: `ls R/per_view_audit/runs/*/receipts`. A `.pid` file with a live process means a job is still running; do not start another.
3. Pull finished records with the remote copy wrapper into `per_view_audit/`.
4. Continue from the first step whose receipt is missing. Never rerun a step whose exit code is zero unless its inputs changed; `manifest.json` records the hashes to compare.
