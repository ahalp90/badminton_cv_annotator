# RTMLib long-video sharding PoC — readable report

## Executive summary — BLUF

This proof of concept tested whether a long match video can be split by frame range, processed by several independent RTMLib workers, and then stitched back into the same five raw pose arrays that the existing pipeline already expects. The answer is **yes for the setup tested**. The PoC produced the same output as the current sequential path, handled worker and shard failures without publishing bad output, and reduced processing time substantially on the A100 GPU.

The most important result is that OpenCV frame seeking worked correctly on the production-style source that was tested. The source was a 1920×1080 H.264 video at a constant 30 fps with 100,349 frames. A full sequential decode was hashed frame by frame, then the PoC sought directly to awkward frame ranges, shard boundaries, the end of the file, and ranges that crossed EOF. Every requested frame matched the sequential decode exactly on both the local laptop and the Bourbaki A100 host. The full sequential decodes on those two machines were also identical for all 100,349 frames. That matters because it means workers do not have to decode from frame zero just to reach their own section of the video.

The sharded extraction itself also matched the existing sequential output in the tests that were run. A deterministic fake extractor matched byte-for-byte on both synthetic video and a real 1080p cut. Real RTMLib on CPU was deterministic across repeated runs and produced byte-exact results when sequential processing was compared with four shards. CUDA on the A100 also produced byte-exact results across repeated sequential runs and between sequential and four-shard runs. The CUDA result should still be treated as specific to the tested stack: A100 plus onnxruntime-gpu 1.27 and the current driver/runtime. It should be rechecked after GPU, driver, ONNX Runtime, or similar environment changes.

Performance improved enough to justify the approach. On a 14,401-frame 1080p test, one worker took 711.5 seconds. Eight workers took 299.3 seconds. That is a 2.38× speed-up, with throughput rising from 20.2 frames per second to 48.1 frames per second. The test includes worker startup, model loading, decoding, inference, compressed shard writes, and final stitching. Based on that measurement, a roughly 100,000-frame match would take about 35 minutes at eight workers instead of about 83 minutes sequentially. The PoC did not establish whether more than eight workers helps, and previous work on this GPU suggests gains flatten around that point.

The failure behaviour is also good enough to carry forward. Workers write temporary files and only mark a shard complete after its data is safely written. The stitch step checks the planned frame ranges, run ID, source MD5, shapes, dtypes, `n_max`, and shard completeness before creating the final output. Ten deliberately corrupted shard cases were rejected. A live-worker SIGKILL test also caused the run to fail rather than leaving output that looked complete. The last shard reads one frame past its expected end so a bad container frame count cannot silently truncate the result.

The final stitched output is compatible with the existing downstream loader and both tested heuristics without changing those components. The PoC deliberately reused the current `extract_raw_frame` logic and the existing five-array file contract rather than creating a new pose format. One important naming trap was found: `apply_heuristic` expects a stem beginning with a numeric video ID. A name such as `sset_21_*` can be silently skipped; a stem such as `21_full_*` is required.

The recommended production change is therefore small. Add optional `start` and `end` frame arguments to `RtmlibPoseExtractor.iter_video`, then add one sharded extraction module under `preparing_data/` to plan ranges, launch workers, track shard state, validate the results, and stitch the five final arrays. Keep `raw_extract`, the heuristics, and downstream consumers unchanged. Keep the existing sequential path as a fallback, and make one shard behave like the sequential version.

What is **not** yet proved is just as important. Frame-exact seeking has only been checked on one full H.264 constant-frame-rate match. Variable-frame-rate video, other codecs, re-encoded sources, and the other full matches have not been checked. CUDA exactness is also only demonstrated on the current A100/software stack. Per-shard resume is not implemented yet; a failed run currently starts again. Full-match heuristic runs still need the real court-context input. Memory use for much longer sources or larger `n_max` values has not been measured, and scaling on the L40 host has not been tested.

**Next steps:** first, run the cheap decode-identity check on the other full matches. If those pass, move the tested planner/worker/stitch logic into production code, add frame-range support to `iter_video`, and port the PoC tests. Then rerun CPU and CUDA parity on the production version, use it on one real full-match extraction, and inspect the published output before making it the normal path. Keep the decode and parity checks as regression gates for new source types and environment changes. If a future source fails frame-accurate OpenCV seeking, use ffmpeg pre-segmentation as the fallback rather than weakening the correctness checks.

## Table of contents

1. [What was built](#what-was-built)
2. [What the PoC showed](#what-the-poc-showed)
3. [Frame decoding: can each worker read the right frames?](#frame-decoding)
4. [Does sharding change the RTMLib output?](#output-parity)
5. [Performance](#performance)
6. [What happens when something fails?](#failure-handling)
7. [Does the output still work downstream?](#downstream-compatibility)
8. [How the PoC works](#how-the-poc-works)
9. [How to put this into the real pipeline](#production-change)
10. [What is still unknown](#remaining-unknowns)
11. [Recommended next steps](#recommended-next-steps)
12. [Fallback if OpenCV seeking fails](#fallback)
13. [Test environment](#test-environment)
14. [Test and repository evidence](#test-evidence)
15. [Commands that were run](#commands-run)
16. [PoC files: keep, move, or archive](#poc-files)

<a id="what-was-built"></a>
## 1. What was built

The PoC adds a second way to process a long video. Instead of one process reading the whole video from start to finish, the video is divided into contiguous frame ranges. Each worker:

1. opens the same source video independently;
2. seeks to the first frame in its assigned range;
3. runs the existing RTMLib pose extraction for that range;
4. writes the same five raw arrays used by the current pipeline; and
5. writes a small manifest saying what it produced.

After every worker finishes, a stitch step checks that all shards are valid and cover the video exactly once. Only then does it concatenate them and publish the normal five output files.

The PoC does **not** introduce a new pose representation. It reuses `raw_extract.extract_raw_frame`, `RAW_SUFFIXES`, and the existing RTMLib model configuration. The goal was to change how work is divided, not what the pipeline produces.

<a id="what-the-poc-showed"></a>
## 2. What the PoC showed

| Question | Plain answer | Important limit |
|---|---|---|
| Can a worker seek directly to its frame range? | **Yes, on the tested source.** Every probed frame matched the sequential decode exactly. | Tested on H.264, 1080p, constant frame rate, OpenCV 5.0. |
| Can several workers safely process one video? | **Yes.** Each process owns its own video capture and RTMLib/ONNX session. | Tested with multiprocessing on the A100 host. |
| Does stitching protect against bad shards? | **Yes.** Ten corruption cases were rejected before final output was published. | Covers the corruption cases implemented in the PoC tests. |
| Does sharding change the output? | **No difference was seen.** CPU and the tested CUDA stack were byte-exact. | CUDA result is specific to this A100/software stack. |
| Does it make the job faster? | **Yes.** Eight workers were 2.38× faster than one worker in the scaling test. | Scaling beyond eight workers was not measured here. |
| Will current downstream code read the result? | **Yes.** The existing loader and both tested heuristics worked unchanged. | Output stem must start with a numeric video ID. |
| Is it ready to move toward production? | **Yes, with more source validation first.** | Other full matches, VFR, and other codecs are still untested. |

<a id="frame-decoding"></a>
## 3. Frame decoding: can each worker read the right frames?

This was the first thing that had to be proved. If seeking lands on the wrong frame, everything after it is unreliable.

The tested source, `sset_21_gloiZ_gTJaE.mp4`, is H.264, 1920×1080, constant 30 fps, with 100,349 frames. A full sequential decode was used as the reference. Every frame was hashed. The PoC then sought directly to a set of difficult positions, including:

- frame 0;
- awkward mid-video positions;
- every production-style shard boundary;
- unaligned ranges;
- the tail of the video; and
- a range that deliberately ran past EOF.

The requested frames matched the sequential hashes exactly on both the local machine and Bourbaki. The two full sequential decodes were also identical to each other for all 100,349 frames.

The EOF test behaved correctly: a range that asked for frames past the end came back short, and the worker refused to publish the shard. The final shard also probes one frame past the planned end so an incorrect container frame count cannot silently cut off real frames.

### Decoding options considered

| Method | Result | Use |
|---|---|---|
| OpenCV `CAP_PROP_POS_FRAMES` seek, then read forward | Exact on every tested range; seek cost was small compared with inference. | **Use this for workers.** |
| Decode from frame 0 and discard frames until the range starts | Correct, and kept as a control. | Too wasteful for normal sharding. |
| Pre-split the video with ffmpeg | Not needed for this PoC. | Keep as a fallback for source types where OpenCV seek fails. |

<a id="output-parity"></a>
## 4. Does sharding change the RTMLib output?

No difference was found in the tests that were run.

| Test | Result |
|---|---|
| Deterministic fake extractor on synthetic video | All five arrays byte-exact. |
| Deterministic fake extractor on a real 2,401-frame 1080p cut | All five arrays byte-exact. |
| Full-video frame decode hashes | Exact. |
| Real RTMLib CPU, repeated sequential runs | Byte-exact. |
| Real RTMLib CPU, sequential vs four shards on 721 frames | Byte-exact. |
| Real RTMLib CUDA, repeated sequential runs on 3,601 frames | Byte-exact. |
| Real RTMLib CUDA, sequential vs four shards on the same cut | Byte-exact. |

The CPU result is strong for the tested configuration. The CUDA result is also exact in this experiment, but it should not be treated as a permanent promise. It was measured on an A100 using `onnxruntime-gpu 1.27.0`. Re-run the CUDA parity check after changing the GPU, driver, ONNX Runtime, or other relevant runtime pieces.

<a id="performance"></a>
## 5. Performance

The scaling test used a 14,401-frame 1080p section of the match on Bourbaki's A100. Each worker had its own model/session. Timing includes process startup, model loading, video decode, inference, compressed shard writes, and stitching.

| Workers | Wall time | Time per frame | Throughput |
|---:|---:|---:|---:|
| 1 | 711.5 s | 49.41 ms | 20.2 fps |
| 2 | 508.8 s | 35.33 ms | 28.3 fps |
| 4 | 387.9 s | 26.93 ms | 37.1 fps |
| 8 | 299.3 s | 20.78 ms | 48.1 fps |

Eight workers were **2.38× faster** than one worker. Importantly, having all workers read different ranges from the same 1080p file did not create a new bottleneck at these worker counts.

A rough extrapolation from this test puts a 100,000-frame match at about **35 minutes with eight workers**, compared with about **83 minutes sequentially**. This is an estimate, not a full-match timing result.

The PoC did not isolate the next bottleneck or test more than eight workers. Existing A100 measurements from the clip-level path suggest returns flatten after roughly eight workers.

<a id="failure-handling"></a>
## 6. What happens when something fails?

The design is deliberately conservative: a bad or incomplete shard should stop the run, not create output that looks valid.

| Failure | What the PoC does |
|---|---|
| Worker dies or shard is missing | The run fails. Stitching refuses to continue. A live SIGKILL test confirmed this. |
| Gap or overlap in frame ranges | Rejected before publication. |
| Shard from another run | Rejected using `run_id`. |
| Shard from another source file | Rejected using source MD5. |
| Partial file write | Data is written to a temporary file and renamed only when complete. |
| Incomplete shard | The shard manifest is written last and acts as the completion marker. |
| Wrong array shape, dtype, or `n_max` | Recomputed expectations are checked; the manifest is not blindly trusted. |
| Bad container frame count | Final shard probes past the expected end and aborts if the plan was too short. |
| Corrupt final publication | Final files are only written after all shard checks pass; `_raw_ndet` is written last, matching the current completion convention. |

Ten deliberate stitch-corruption cases were tested and refused.

One thing the PoC does **not** do yet is resume individual completed shards after a failed run. The manifests contain enough information to support this, but the current PoC starts the run again from scratch.

<a id="downstream-compatibility"></a>
## 7. Does the output still work downstream?

Yes. After stitching, the result is the same five raw arrays the current code already loads:

- `_raw_kps`
- `_raw_bboxes`
- `_raw_scores`
- `_raw_kp_scores`
- `_raw_ndet`

The existing `RawClip` loader and both `current` and `sticky_anchor` heuristics were run against stitched output without changing them.

### Important naming gotcha

`apply_heuristic` expects the file stem to begin with a numeric video ID. A name such as `sset_21_full_*` can be silently skipped. Use a stem such as `21_full_*`.

This is not a sharding problem, but the PoC exposed it and the production path needs to respect it.

The rule comes from one function: `apply_heuristic._vid_from_stem`, which runs `int(stem.split("_", 1)[0])` and uses that number to look up the video's court calibration and resolution. To allow stems like `sset_21`, change that one function, or pass the video ID in directly. If you do touch it, also make an unparseable stem raise an error — right now it is skipped silently.

<a id="how-the-poc-works"></a>
## 8. How the PoC works

```text
run_sharded
  |
  +-- split [0, total_frames) into contiguous frame ranges
  |
  +-- write run manifest
  |
  +-- start N worker processes
  |     |
  |     +-- each worker opens the source video
  |     +-- seeks to its start frame
  |     +-- runs RTMLib over [start, end)
  |     +-- uses extract_raw_frame for the normal five-array format
  |     +-- writes shard data
  |     +-- writes shard manifest last
  |
  +-- wait for all workers
  |
  +-- validate every shard
  |
  +-- concatenate shards in frame order
  |
  +-- publish the normal {stem}_raw_*.npy files
```

### PoC components

| Component | Job |
|---|---|
| `shard_plan` | Splits the frame range into contiguous, non-overlapping shards. |
| `range_decode` | Reads an exact frame range and provides file/frame hashes for checks. |
| `fake_pose` | Deterministic test extractor used to prove ordering and assembly. |
| `shard_worker` | Runs one shard in one process. |
| `stitch` | Validates all shard data before combining it. |
| `run_sharded` | Plans, launches workers, checks exits, and stitches. |
| `gate_decode_identity` | Checks frame seek against a sequential hash ledger. |
| `gate_parity` | Compares sequential and sharded extraction. |
| `gate_downstream` | Checks the existing loader and heuristics against stitched output. |
| `bench_worker_scaling` | Measures 1/2/4/8-worker performance. |

<a id="production-change"></a>
## 9. How to put this into the real pipeline

The recommended change is additive. Do not replace or rewrite the existing extraction and heuristic logic.

### Change 1: allow `iter_video` to read a frame range

```python
def iter_video(self, video_path, start: int = 0, end: int | None = None):
    ...
```

Default behaviour stays the same: start at frame 0 and read to the end. Sharded workers pass `start` and `end`.

### Change 2: add one sharded extraction module

Suggested location:

```text
preparing_data/extract_sharded_video.py
```

Its job is to:

- plan the ranges;
- write the run manifest;
- launch and join worker processes;
- check worker exit codes;
- manage shard manifests;
- validate shard data; and
- stitch and publish the final five arrays.

A suitable public entry point is:

```python
def extract_sharded(
    video_path,
    save_dir,
    stem,
    n_shards,
    n_max=16,
    device="cuda",
) -> Path:
    ...
```

### Leave these alone

- `raw_extract.extract_raw_frame`
- the five-array raw file contract
- `apply_heuristic`
- `current` and `sticky_anchor`
- court logic
- clip generation
- TrackNet
- downstream consumers

The sharded path should simply produce the files those components already expect.

### Keep the sequential path

Do not remove the current path during rollout. One shard should behave like sequential processing through the new code, and the old extraction path remains available as a fallback until the sharded version has been used successfully on real full-match jobs.

<a id="remaining-unknowns"></a>
## 10. What is still unknown

These are the real gaps left by the PoC:

- **Other source videos:** frame seeking was proved on one full match only. Run the identity check on the other full matches.
- **VFR and other codecs:** variable-frame-rate video, different codecs, and re-encoded sources may behave differently. They must pass the seek check before using sharding.
- **Future CUDA environments:** exact CUDA parity should be rechecked after GPU, driver, ONNX Runtime, or similar changes.
- **Per-shard resume:** not implemented. A failed run currently restarts.
- **Real court context for full-match heuristics:** the downstream gate used a synthetic identity court. Production needs the real `all_court_info`/resolution row.
- **L40 performance:** not tested on Carmack.
- **More than eight workers:** not measured in this PoC.
- **Stitch memory on larger jobs:** about 0.5 GiB was estimated for a 56-minute match at `n_max=16`; growth is linear and larger cases were not measured.

<a id="recommended-next-steps"></a>
## 11. Recommended next steps

1. **Run frame-identity checks on the other full matches.** This is CPU-only and is the cheapest way to find source-specific seek problems before changing production code.
2. **Add `start`/`end` support to `RtmlibPoseExtractor.iter_video`.** Keep the current default behaviour unchanged.
3. **Move the tested planner, worker, and stitch logic into `preparing_data/extract_sharded_video.py`.** Port the PoC tests with it.
4. **Add per-shard resume while moving the code.** The manifest design already supports identifying completed shards; implement and test the skip/reuse behaviour.
5. **Rerun CPU and CUDA parity on the production version.** Record the environment with the result.
6. **Run one real full-match extraction.** Keep the shard/run artefacts until the final output has been inspected and downstream processing has succeeded.
7. **Keep the decode/parity/downstream checks.** Run them again for new source types and after relevant runtime changes.
8. **Only then make sharded extraction the normal full-video path.** Keep sequential processing available as a fallback.

<a id="fallback"></a>
## 12. Fallback if OpenCV seeking fails

The best fallback is ffmpeg pre-segmentation into chunk files, then processing those files through the existing clip-style machinery.

It is not the first choice because it adds another disk pass, another set of intermediate files and provenance, and keyframe-aligned boundaries can be less precise. The PoC found no measured benefit that would justify those costs while direct OpenCV seeking is frame-exact.

If a future video fails the frame-identity check—for example because it is VFR or uses a different codec—then ffmpeg segmentation becomes the safer option for that source rather than accepting uncertain seeks.

<a id="test-environment"></a>
## 13. Test environment

| Item | Value |
|---|---|
| Base commit | `95f812be8af0a05c364e810823ab085fbc113391` |
| PoC branch | `poc/rtmlib-video-sharding` |
| Worktree | `wt_rtmlib_sharding_poc` |
| Local host | CPU only; Python 3.12; OpenCV 5.0; no usable ONNX Runtime/RTMLib GPU stack |
| Bourbaki | A100-PCIE-40GB; driver 610.57.04; Python 3.11.13 |
| ONNX Runtime | `onnxruntime-gpu 1.27.0` with CUDA provider |
| RTMLib models | RTMDet-M@640 + RTMPose-L@256 |
| Main test video | `sset_21_gloiZ_gTJaE.mp4` |
| Video format | H.264, 1920×1080, CFR 30 fps |
| Frame count | 100,349 |
| Source identity | File MD5 matched on local and Bourbaki |

<a id="test-evidence"></a>
## 14. Test and repository evidence

### PoC tests

- 18 `tests/test_video_sharding.py` tests passed locally.
- These covered planning, deterministic seek/scan parity, worker short-read/overrun handling, ten stitch corruption cases, and downstream loader/heuristic checks.

### Repository regression run

- 1,380 tests passed in the PoC worktree.
- Four `test_namespace_migration` failures were already present; main had six failures in the same file.
- No existing tracked production files were modified by the PoC.

### Static checks

- Ruff was clean on the new PoC files.
- Pyrefly was clean on the new PoC files; the reported repo-level issues pre-existed on main.

### Specific experiments

- Full sequential decode: 100,349 decoded frames, matching container metadata.
- Local seek checks: 14/14 tested ranges were frame-exact, including EOF behaviour.
- Bourbaki seek checks: default probes plus five extra unaligned ranges passed.
- Cross-host sequential decode: all 100,349 frame hashes matched.
- SIGKILL test: worker death was visible through exit status and no final output was published.
- CPU self-variance: byte-exact.
- CPU sequential vs four shards: byte-exact.
- CUDA self-variance: byte-exact on the tested stack.
- CUDA sequential vs four shards: byte-exact on the tested stack.
- 1/2/4/8-worker scaling: 711.5 / 508.8 / 387.9 / 299.3 seconds.

### Useful gotchas found during the PoC

- `apply_heuristic` can silently skip output whose stem does not start with a numeric video ID.
- `np.save(path)` appends `.npy` to filenames with unfamiliar extensions. For atomic `.tmp` writes, writing through an open file handle avoids accidental filename changes.

<a id="commands-run"></a>
## 15. Commands that were run

The package was moved from `src/bst_x/validation_scripts/video_sharding/` to `src/shared/video_sharding/` after the runs. The commands below use the final module path.

Run from the worktree root with `PYTHONPATH=src:src/bst_x`. `$V21` is the full `sset_21` match and `$LEDGER` is its sequential frame-MD5 ledger.

```bash
# Tests
python -m pytest tests/test_video_sharding.py -q
python -m pytest -q

# Frame identity
python -m shared.video_sharding.gate_decode_identity baseline $V21 $LEDGER
python -m shared.video_sharding.gate_decode_identity check $V21 $LEDGER --mode seek
python -m shared.video_sharding.gate_decode_identity check $V21 $LEDGER \
  --mode seek --ranges 0:40,12544:12584,...,100309:100349
python -m shared.video_sharding.gate_decode_identity check $V21 $LEDGER \
  --mode scan --ranges 50176:50216

# Parity
python -m shared.video_sharding.gate_parity --video $V21 --workdir W \
  --extractor fake --n-shards 6 --limit-frames 2000
OMP_NUM_THREADS=2 ... gate_parity --extractor cpu --limit-frames 600 --self-variance
OMP_NUM_THREADS=2 ... gate_parity --extractor cpu --limit-frames 600 --n-shards 4
... gate_parity --extractor cuda --limit-frames 3000 --self-variance
... gate_parity --extractor cuda --limit-frames 3000 --n-shards 4

# Downstream compatibility and scaling
python -m shared.video_sharding.gate_downstream --video $V21 --workdir W \
  --stem 21_full_poc --limit-frames 300
... bench_worker_scaling --extractor cuda --limit-frames 12000 --worker-counts 1,2,4,8

# Bourbaki test ladder
bash src/shared/video_sharding/run_remote_bourbaki.sh
```

<a id="poc-files"></a>
## 16. PoC files: keep, move, or archive

| PoC piece | Recommendation | Reason |
|---|---|---|
| `shard_plan`, `shard_worker`, `stitch`, `run_sharded` | Move into the production sharded extraction module, with light cleanup. | This is the core logic that was tested. |
| `range_decode` | Fold frame-range reading into `rtmlib_pose.iter_video` plus a small helper if needed. | Avoid maintaining two separate video-decoding surfaces. |
| `fake_pose` | Keep for tests only. | Gives deterministic end-to-end parity coverage. |
| `tests/test_video_sharding.py` | Keep and port with the production code. | Covers the important correctness and failure cases. |
| `gate_decode_identity` | Keep as a maintenance check. | Use for new sources and decoder changes. |
| `gate_parity` | Keep as a maintenance check. | Use after RTMLib/ONNX/GPU/runtime changes. |
| `gate_downstream` | Keep as a maintenance check. | Confirms the five published files still work with downstream code. |
| `bench_worker_scaling` | Keep as a benchmark tool. | Useful for other GPUs and worker counts. |
| `run_remote_bourbaki.sh` | Archive with the PoC. | Test driver, not production code. |
| Original `INVESTIGATION.md` and handoff report | Archive with the PoC branch. | Useful history, but this report is the readable decision record. |

### Current repository state

- Added: `src/shared/video_sharding/` and `tests/test_video_sharding.py`.
- Existing tracked files modified: **0**.
- Existing tracked files deleted or renamed: **0**.
- Branch pushed: **no**.
- Bourbaki logs and frame ledger remain under `/scratch/comp320a/ahalperi/rtmlib_sharding_poc_out/` (about 3.3 MB).
- Synced code and heavy temporary artefacts on Bourbaki were removed and the `shard_poc` tmux session ended.
- Bourbaki video renames `pilot* -> sset_01*` and `vid15* -> sset_15*` were MD5-checked against the local 288p copies before rename.
