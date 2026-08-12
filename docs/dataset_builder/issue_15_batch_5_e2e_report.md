# Issue 15 Batch 5 end-to-end trial

Status: accepted

For the operational failure, recovery, and future-hardening analysis, see the
[Batch 5 reliability postmortem](issue_15_batch_5_reliability_postmortem.md).

The corrected trial ran from the clean Bourbaki root
`/scratch/cmarti/issue15_ce9405b` at source commit
`ce9405b2c1cb9aec948e510f9f1e6e3af410aabf`. The root is execution-only and
remains preserved for reproducibility; no code or Git ref was modified there.

## Gates and outcome

- TrackNet exact-multiple EOF gate: 2 tests passed in 7.88 seconds.
- RTMLib child import gate: passed with `frozendict` absent.
- Both selected canonical videos passed download and exact metadata checks.
- The first full attempt exposed three transient failed pose shards for
  `9WVwZSzixh0`; the corrective resume retried that dependency lane only.
- Corrective resume completed both videos and assembled 218 rallies.
- Final unchanged resume passed with all 45 stage-artifact checksums and all
  four publication files byte-identical before and after.
- Final secret scan checked 33 files and found no credential value.
- No coordinator, child process, or GPU allocation remains.

The Gemini triage request became unavailable after a provider 503. The reviewed
visual fallback selected both videos, and the optional commentary lane did not
block vision or assembly. The final report contains two processed videos and
218 assembled rallies.

## Final stage timings

Times are the manifest `elapsed_seconds` values from the repaired run. Per-video
stages are listed in video order (`9WVwZSzixh0`, `P3OcTzwmqeY`).

| Stage | Time (s) | Outcome / count |
| --- | ---: | --- |
| Search | 182.3 | processed; 5 candidates |
| Transcript | 184.3 | processed; 3 transcripts, 2 unavailable |
| Triage | 589.6 | unavailable; visual fallback used |
| Selection | 0.0 | processed; 2 selected |
| Download | 1,425.0 | processed; 2 videos |
| Metadata | 544.3 / 553.8 | processed; 153,600 / 165,150 frames |
| Commentary cleaning | 0.0 | processed; 0 cleaned videos |
| TrackNet input | 1,845.9 / 2,095.5 | processed; 153,600 / 165,150 frames |
| Shuttle extraction | 1,053.3 / 1,159.7 | processed; 153,600 / 165,150 frames |
| Pose extraction | 3,774.2 / 4,799.9 | processed; 153,600 / 165,150 frames |
| Court evidence | 1,526.0 / 1,281.6 | processed; 779 / 505 scenes |
| Annotation | 15.3 / 21.4 | processed; 112 / 106 rallies |
| Commentary pairing | 0.0 / 0.0 | processed; 0 paired, 112 / 106 rallies |
| Primitive projection | 0.2 / 0.1 | processed; 112 / 106 rallies |
| Assembly | 1.5 | processed; 218 rallies, 2 videos |
| Report | 0.1 | processed; 2 videos, 218 rallies |

## Performance interpretation

These timings are operational telemetry from the accepted run, not a controlled
whole-pipeline before/after benchmark: provider availability, downloads,
caches, and the corrective resume would confound such a comparison. The
separate fixed-input TrackNet experiment measured roughly 2 hours 40 minutes
at stride 1 and 22 minutes at stride 8 on about 154,000 512x288 frames, which
is sufficient evidence for the adopted stride-8 shuttle configuration. See
[video pipeline throughput research](video_pipeline_throughput_research.md).

The one/four/eight-worker pose scaling comparison remains deferred. It and a
clean end-to-end capacity run are required before making a production
throughput or cost claim; they were not required for this correctness and
resume acceptance.

## Final publication digests

```text
3f246a6f1444791dd8f2dadebcfb67c1  run_manifest.json.gz
088a6c1774dcea302d1a88412ac25d1f  rally_records.json.gz
97586c04dc9565d0a764fab4528e08c2  dataset_builder_report.json.gz
ffb580d0a02ba782ea37f512a0298f7b  selected_videos.csv.gz
```

The accepted implementation commit is
`ce9405b2c1cb9aec948e510f9f1e6e3af410aabf` (`Fix full-video inference
boundaries`).
