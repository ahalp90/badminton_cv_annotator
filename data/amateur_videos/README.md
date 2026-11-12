# Amateur videos for contact labelling

We automatically annotated 20 full amateur badminton videos as a starting point
for manually labelled contacts. The videos and generated
annotations are stored outside the repository. Use the source copies with the
frame counts listed below. [videos.csv.gz](videos.csv.gz) maps each YouTube ID
to its folder name, exact title, frame rate and frame count.

## Two annotation sets

Both sets come from the deployed seed-100 model from
[post-split correction](../../experiments/annotator/post_split_correction/README.md).
Seed 100 had the most fully correct validation rallies: 361 of 668. Both sets
use the [manually labelled courts](../amateur_court_corners/2026-11-09/README.md).

| Annotation set | Videos | Annotator input |
|---|---:|---|
| Native frame rate | 20 | Every source frame |
| Half frame rate | 11 | Every second source frame |

The half-rate set covers the 11 videos above 30 FPS. They run at 50–60 FPS and
were annotated at 25–30 FPS. The annotator was trained on 25 and 30 FPS video,
and some of its learned motion features are measured per frame. The half-rate
set lets us compare annotation at a rate closer to training. Both sets reuse
the same full-rate pose, shuttle and court extraction; only the annotator input
differs. We have not yet compared their accuracy.

**Both sets save original source-video frame numbers.** This applies to
contacts, rally bounds and masks. Open the original video at its native frame
rate and use the saved numbers unchanged. In the half-rate set, sampled frame
`i` is source frame `2i`, and each saved mask value covers two source frames.

## Files in each video folder

Each set has one folder per video, named in the table below.

- `annotator_result.json.gz`: the annotations. Contacts are in
  `result.contact_events`, each with a source `frame`. Rallies are in
  `result.rallies`, with `sequence.start_frame` and an exclusive
  `sequence.end_frame`. `video_id` gives the YouTube ID.
- `definitive_exclusion_mask.npy.xz`: one value per source frame, marking
  frames the annotator excluded, such as frames without a court.
- `raw_replay_mask.npy.xz`: the frame flags before short runs are filtered out.
- `shuttle_quality.json.gz`: shuttle-track quality from the native-rate track.
- `rerun.json.gz`: completion record with rally and contact counts.
- `stride2.json.gz`, half-rate set only: the sampling and frame mapping.

## Other outputs from the same run

The run also annotated every video with automatically detected courts.
Nineteen videos produced annotations; `181Rb5NPZ0g` had no detected court in
any scene. The run repeated every variant with the other four post-split
correction seeds: 0, 200, 300 and 400. These outputs are stored with the two
annotation sets. The run used commit `4545b0ba`.

## Videos

| Folder name | Video ID | FPS | Frames | Half-rate set |
|---|---|---:|---:|:---:|
| [12-year-vs-24-year-badminton-player-challenge](https://www.youtube.com/watch?v=h13gAmp9zJY) | `h13gAmp9zJY` | 50 | 24,227 | Yes |
| [2014-WA-State-Championship-Mens-Singles](https://www.youtube.com/watch?v=181Rb5NPZ0g) | `181Rb5NPZ0g` | 25 | 17,556 | — |
| [2020-Perth-Championships](https://www.youtube.com/watch?v=AyD64iYVyss) | `AyD64iYVyss` | 30 | 48,831 | — |
| [Akul-Nandal-vs-Dishant-Ahlawat](https://www.youtube.com/watch?v=7ekxuFq3sPU) | `7ekxuFq3sPU` | 60 | 29,183 | Yes |
| [Badminton-Singles-High-Level-Match](https://www.youtube.com/watch?v=E8WW8DFCnwk) | `E8WW8DFCnwk` | 59.94 | 46,851 | Yes |
| [Bellevue-Badminton-Club-Viktor-Axelsen](https://www.youtube.com/watch?v=gmXVbt-AC8o) | `gmXVbt-AC8o` | 59.94 | 60,432 | Yes |
| [Boys-Singles-Final-Jaipur-District](https://www.youtube.com/watch?v=l-I_Di1Ad2Y) | `l-I_Di1Ad2Y` | 24 | 30,272 | — |
| [Brandon-Redekop-vs-Desmond-Wang](https://www.youtube.com/watch?v=-0I4PN0dDuI) | `-0I4PN0dDuI` | 29.97 | 48,850 | — |
| [Dave-Freeman-2026-Open-MS-A-Final](https://www.youtube.com/watch?v=BBWpUWuFCpc) | `BBWpUWuFCpc` | 29.97 | 80,634 | — |
| [Harry-Ng-vs-Eden-Liang](https://www.youtube.com/watch?v=7ma9qZs1dZw) | `7ma9qZs1dZw` | 30 | 24,716 | — |
| [Harshad-Shadish-vs-Hemanth-Gowda](https://www.youtube.com/watch?v=jxESnJszRjk) | `jxESnJszRjk` | 60 | 60,584 | Yes |
| [How-to-think-while-playing-badminton](https://www.youtube.com/watch?v=gxBQ_HwdgN4) | `gxBQ_HwdgN4` | 59.885 | 90,971 | Yes |
| [Jadon-Tsang-vs-Jackson-Yang](https://www.youtube.com/watch?v=Nd4uQVO652Y) | `Nd4uQVO652Y` | 30 | 43,822 | — |
| [Mens-Singles-C-Grade-William-Tia-vs-Ryan-Si](https://www.youtube.com/watch?v=BkjErIAsZu4) | `BkjErIAsZu4` | 59.94 | 52,651 | Yes |
| [Mens-Singles-Final-Level-2-Senior-State](https://www.youtube.com/watch?v=4FinbnJSjvU) | `4FinbnJSjvU` | 60 | 100,534 | Yes |
| [Prakriti-Bharath-vs-Jayani-V-S](https://www.youtube.com/watch?v=eg6RqwLHSd4) | `eg6RqwLHSd4` | 60 | 27,254 | Yes |
| [They-play-the-badminton-like-playing-chess](https://www.youtube.com/watch?v=zYqgJo1L5uM) | `zYqgJo1L5uM` | 60 | 28,142 | Yes |
| [UABC-Open-2019-A-Grade-Mens-Singles-Final](https://www.youtube.com/watch?v=C1jR4vZmrkI) | `C1jR4vZmrkI` | 30 | 26,960 | — |
| [UWBC-2017-Summer-Tournament](https://www.youtube.com/watch?v=Cb-xs5rPyxI) | `Cb-xs5rPyxI` | 29.97 | 89,234 | — |
| [Like-a-bow-when-hitting-the-SMASHING](https://www.youtube.com/watch?v=PQmBTcnz998) | `PQmBTcnz998` | 60 | 44,293 | Yes |

Folder names are shortened YouTube titles. `BkjErIAsZu4` runs at exactly
19001/317 FPS; `videos.csv.gz` gives every exact rate. Its source and that of
`C1jR4vZmrkI` were MKV files, remuxed to MP4 without re-encoding, so frame
numbers match the original files.

Three videos from the same collection are left out: `C6NrJyBwn6c` is a doubles
match, `nTL8uUlb4fo` is heavily edited and `i1KGN5cDoVA` is mostly a person
talking.
