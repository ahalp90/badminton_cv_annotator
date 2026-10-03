# Saved measurements from detector development

These files contain the results of small checks used to build the detector:
CPU versus GPU matching, combining frames, sharing courts between scenes, and
cases where those ideas failed. The [evaluation notes](../../comparisons/development_evaluation.md) explain
the results.

The usable video predictions are in
[`data/court_detections/sset_and_sset22/extractions_20261003/`](../../../../data/court_detections/sset_and_sset22/extractions_20261003/README.md).
The files here are older experiment results, rather than inputs needed to run
the detector.

| File | Contents |
| --- | --- |
| `interval_cpu_results.json.gz` | Courts from 33 scenes of a five-minute clip, using CPU template matching. |
| `interval_cuda_results.json.gz` | The same scenes with GPU matching. Court choices match the CPU run. |
| `interval_cpu_timing.json.gz` | CPU matching times, split by processing stage. |
| `interval_cuda_timing.json.gz` | GPU matching times for that same comparison. These runs preceded multi-frame fitting and sharing. |
| `scene_checks_24_videos.json.gz` | Checks of returning camera views and shortened player-sampling windows across 24 videos. |
| `composite_scatter_video040.json.gz` | How much the fitted corners varied across nine matching views, before and after combining three frames per scene. |
| `composition_prefix_video040.csv.gz` | Which individual or combined court was chosen in the first 218 scenes of video 040. Some early comparisons lacked scores for an alternative. |
| `pool_fallback_video003.json.gz` | A two-scene case where combining line evidence gave a worse court than reusing one complete fit. |
| `label_conflict_video003.json.gz` | The cause of that failure: two scenes treated the same physical stripe as different court lines. |
| `fast_robust_cached_video040.json.gz` | Two modes compared on the same saved inputs: 11 matching scenes and two separate views. Contains their chosen courts and detector scores. |
| `fast_robust_live_video040.json.gz` | Four scenes processed with fresh line and player detections, including chosen courts and timings. |
| `full_video040_video_robust.json.gz` | All 405 scene results from one complete video, grouped into 45 camera views. |
| `full_video040_summary.json.gz` | Counts and timings for that run, including time spent comparing shared courts. |
| `net_reference_comparison.json.gz` | Court errors against labelled points while varying the small reward for net-post evidence. |

A detector score describes how well a candidate fits the image evidence. A
higher score can still select the wrong court. Likewise, less variation between
frames can mean a consistently misplaced court. The evaluation notes distinguish
those measurements from errors against labels.

The files are gzip-compressed JSON or CSV. This
[Python example](../../comparisons/development_evaluation.md#reproducing-the-numbers) reads a saved comparison.
Embedded source paths describe the original runs; the old raw run directories
have been removed.
