# Evidence map

This is a reference file. Use it when you need to trace a claim back to the GitHub record or retained inputs.

This is the short route back to the evidence behind `evaluation.md`. GitHub was
read with `gh`; no GitHub record was changed.

## Contents

- [Main record](#main-record)
- [Related intent and safety evidence](#related-intent-and-safety-evidence)
- [Exact PR 80 code and artefacts](#exact-pr-80-code-and-artefacts)
- [What is and is not retained](#what-is-and-is-not-retained)

## Main record

- [PR 80](https://github.com/ahalp90/badminton_cv_annotator/pull/80): benchmark
  design, retained results, runtime limits, and the decision not to integrate.
- [The request to investigate the prompt and priors](https://github.com/ahalp90/badminton_cv_annotator/pull/80#issuecomment-5297260980):
  the final comment that kept the PR open.
- [Issue 38](https://github.com/ahalp90/badminton_cv_annotator/issues/38): the
  original VLM goal and request to use existing pipeline evidence.

PR 80 had four issue comments. It had no formal reviews or inline review
comments when checked on 20 August 2026.

## Related intent and safety evidence

- [Issue 31](https://github.com/ahalp90/badminton_cv_annotator/issues/31) and
  [PR 78](https://github.com/ahalp90/badminton_cv_annotator/pull/78): the shuttle
  hallucination audit and its lack of real-shuttle controls.
- [Issue 29](https://github.com/ahalp90/badminton_cv_annotator/issues/29),
  [PR 61](https://github.com/ahalp90/badminton_cv_annotator/pull/61), and
  [PR 62](https://github.com/ahalp90/badminton_cv_annotator/pull/62): broadcast
  scene labels, boundaries, and proposal files.
- [PR 88](https://github.com/ahalp90/badminton_cv_annotator/pull/88) and
  [PR 97](https://github.com/ahalp90/badminton_cv_annotator/pull/97): serves
  shown from unusual views or inferred across a broadcast cut.
- [Issue 95](https://github.com/ahalp90/badminton_cv_annotator/issues/95) and
  [PR 98](https://github.com/ahalp90/badminton_cv_annotator/pull/98): suspicious
  RANSAC geometry can overlap real contacts and is not truth.
- [Issue 36](https://github.com/ahalp90/badminton_cv_annotator/issues/36): hard
  cuts are useful hints but brittle scene labels.

The intended finished pipeline is automatic. Human labels are for offline
evaluation only.

## Exact PR 80 code and artefacts

The inspected PR head is `96e0e289a951d63fbaaa62f26c399a4beb61ae79`.
At that PR head, the relevant code and retained raw outputs are under:

- `src/annotator/vlm_scene_benchmark/`
- `docs/scraper_pipeline/vlm_scene_filtering/data/benchmark_20260810/`

The raw replies are the source for the prompt-copying finding. InternVideo3
repeated `LBRFRS9B`; Qwen3-VL repeated `OBRFRS9G`.

Useful local measurement context is in:

- `docs/scraper_pipeline/annotator_measurement_history.md`
- `docs/scraper_pipeline/inpaint_hallucination_fix/shuttle_hallucination_visual_audit_20260809.md`
- `tests/data/annotator_calibration/reference/aggregate_stdout.txt`
- `docs/scraper_pipeline/broadcast_nonstandard_camera_id/data/sset_01_replay_and_serve_behaviour_20260805/report.md`

The compact follow-up measurements are in
`experiments/results/summary.json`. They can be regenerated from fresh attempts
with the retained builders and scorers.

## What is and is not retained

PR 80 is the most directly auditable part of this investigation. Its raw model replies and benchmark artefacts are retained in Git history.

The later contact, tracker, multiscale, and replay-pair experiments are not retained at the same level. The repository/package contains the compact aggregate `experiments/results/summary.json`, but not all later raw attempts, manifests, or row-level score files. Those later aggregate claims therefore cannot be independently recomputed from repository evidence alone.

The standard-view versus unusual-view live split was recovered after the main write-up by joining the original full-fixture scene score with the untouched human scene truth. The original row-level score is not retained in the repository, so the recovered counts are preserved explicitly in [`experiments/results/scene_live_view_split.json`](experiments/results/scene_live_view_split.json) together with the derivation note and human-truth source hashes.
