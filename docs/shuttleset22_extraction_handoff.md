# ShuttleSet22 extraction handoff

Issue [#106](https://github.com/ahalp90/badminton_cv_annotator/issues/106)
prepared whole-video perception inputs for the binary shot-classifier work.

## Result

- 58 ShuttleSet22 annotation records were reviewed.
- 47 unique public sources were downloaded and extracted successfully.
- Eight records overlap ShuttleSet and are excluded from this extraction set.
- Three records have no frame-aligned public source: 14, 45, and 56.

Each completed source has a gzip-compressed TrackNet CSV, a
`shuttle_track.npy.xz` array, and the RTMLib pose arrays. The NPY archives use
LZMA preset 9 and CSV outputs use gzip compression.

## Data locations

The active Bourbaki workspace is:

```text
/scratch/cmarti56/issue106-shuttleset22-data/
```

Its `sources/` directory holds the downloaded videos and `extracted-simple/`
holds the 4.8 GB extracted arrays. These paths are host-local scratch storage.

A checksum-verified, resumable backup is being copied to this server:

```text
/srv/mergerfs/main_pool/320_cosc594_data/ShuttleSet/shuttleset22_raw_video/
/srv/mergerfs/main_pool/320_cosc594_data/ShuttleSet/shuttleset22_extracted/
```

## Reuse boundary

`configs/shuttleset22/sources.toml` is the reviewed mapping of annotation IDs,
public source URLs, overlap records, and unavailable records. Use ShuttleSet22
annotations when training on these outputs. Do not combine the eight overlap
records as independent examples with ShuttleSet.

Raw broadcaster videos and extracted arrays are intentionally outside Git. The
annotations are from the MIT-licensed CoachAI Projects repository; the videos
remain subject to broadcaster rights.
