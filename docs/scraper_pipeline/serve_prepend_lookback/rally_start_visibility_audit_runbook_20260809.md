# Rally-start visibility audit laptop runbook

## Purpose and limits

This Phase 1 audit checks whether issue 28 target starts show a physical serve,
begin after the broadcast omitted the serve, or remain uncertain. It does not
estimate omission prevalence from the pilot and does not change the canonical
broadcast timeline.

The generated package contains:

- 136 complete target rows: 63 `sset_01`, 39 `sset_15`, and 34 `sset_21`;
- all 26 flaw-marked targets as a source-quality audit stratum; and
- two unflagged transition controls per video.

The 32-row pilot is distributed as follows:

| Video | Quality-audit rows | Transition controls | Pilot rows |
| --- | ---: | ---: | ---: |
| `sset_01` | 2 | 2 | 4 |
| `sset_15` | 0 | 2 | 2 |
| `sset_21` | 24 | 2 | 26 |
| **Pooled** | **26** | **6** | **32** |

These rows validate the review workflow and adjudicate questionable source
records. They cannot show that omitted starts are common or rare. The exact
previously observed `sset_15` omitted-start row was not recorded in repository
files. The complete 39-row `sset_15` target table remains available if Curtis
recognises or supplies that case.

## Decision contract

Use one of these values:

- `visible`: the physical service contact is visibly observable;
- `broadcast-omitted`: live rally footage begins after the physical service;
- `uncertain`: the available footage does not support either conclusion.

Record:

- `visible_serve_frame` for `visible`;
- `first_visible_rally_frame` and `broadcast_return_frame` for
  `broadcast-omitted`;
- `confidence` as `certain` or `uncertain`; and
- a short `review_note`, especially for every uncertain decision.

Frame numbers are zero-based. Interval ends are exclusive.

Use these operational marker definitions:

- `broadcast_return_frame` is the first frame of the shot where the broadcast
  returns from replay, cutaway, or other non-live footage to the current rally;
- `first_visible_rally_frame` is the first frame at or after that return where
  current-rally play is visibly supported. It can equal
  `broadcast_return_frame` when active play is clear immediately.

For `visible`, record `visible_serve_frame` within the review window and leave
both omitted-start markers blank. For `broadcast-omitted`, leave
`visible_serve_frame` blank and require both omitted-start markers within the
review window, with
`broadcast_return_frame <= first_visible_rally_frame`. For `uncertain`, leave
all three frame markers blank, use `confidence=uncertain`, and explain the
uncertainty in `review_note`.

## 1. Open the laptop worktree

Set `REPO` to the checked-out issue-32 worktree and use the annotation Python
environment already used for the broad timeline review:

```bash
cd /path/to/issue-32-rally-start-replay-sting

REPO="$PWD"
PY=/home/clm/Work/Uni/cosc595/.venv-annotation/bin/python
LABELS="$REPO/docs/scraper_pipeline/broadcast_nonstandard_camera_id/data"
GUIDES="$REPO/docs/scraper_pipeline/serve_prepend_lookback/data/rally_start_visibility_audit_20260809"
AUDIT="$REPO/local_scratch/broadcast_timeline_annotation/rally_start_visibility_20260809"

mkdir -p "$AUDIT"
test -x "$PY" || echo "Missing annotation Python: $PY"
test -f "$GUIDES/summary.json.gz" || echo "Missing rally-start guide package"
```

## 2. Set and verify the review videos

```bash
VIDEO_01="$(find "$REPO/local_scratch/broadcast_timeline_annotation/sset_01" -maxdepth 1 -type f -iname '*288p*.mp4' -print -quit 2>/dev/null)"
VIDEO_15="$REPO/local_scratch/broadcast_timeline_annotation/sset_15/vid15_288p.mp4"
VIDEO_21="$REPO/local_scratch/broadcast_timeline_annotation/sset_21/sset_21_288p.mp4"

test -n "$VIDEO_01" && test -f "$VIDEO_01" || echo "Locate the sset_01 review video"
test -f "$VIDEO_15" || echo "Missing sset_15 review video"
test -f "$VIDEO_21" || echo "Missing sset_21 review video"
```

Check the decoded metadata:

```bash
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,avg_frame_rate,nb_read_frames -of default=noprint_wrappers=1 "$VIDEO_01"
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,avg_frame_rate,nb_read_frames -of default=noprint_wrappers=1 "$VIDEO_15"
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,avg_frame_rate,nb_read_frames -of default=noprint_wrappers=1 "$VIDEO_21"
```

Expected decoded metadata:

| Video | Width | Height | FPS | Frames |
| --- | ---: | ---: | ---: | ---: |
| `sset_01` | 512 | 288 | 25 | 154393 |
| `sset_15` | 512 | 288 | 25 | 149487 |
| `sset_21` | 512 | 288 | 30 | 100349 |

Check the two recorded review-video MD5s:

```bash
md5sum "$VIDEO_15" "$VIDEO_21"
```

Expected hashes:

```text
39c693db594e850399e3a8cae34ffdde  sset_15
a07863d2acae6353ef158cf3576a1a9d  sset_21
```

The exact encoded `sset_01` review-copy hash was not recorded. Check its FPS,
frame count, and source identity.

## 3. Protect the canonical timeline

Record the canonical hashes:

```bash
sha256sum \
  "$LABELS/sset_01_broadcast_timeline_labels.csv.gz" \
  "$LABELS/sset_15_broadcast_timeline_labels.csv.gz" \
  "$LABELS/sset_21_broadcast_timeline_labels.csv.gz"
```

Expected hashes:

```text
b65082468aa1635d177028b46367ebc643013892854aa45798b8b96062532bad  sset_01
fb68449e3ae0513af5368e3082f7b49d6ad6f6be95598dbe7230dc299c57c022  sset_15
06812dbd11f60540920b435bf37db08327d8aac042960749a17fc05a74a9a2c7  sset_21
```

Create disposable viewer copies:

```bash
cp -p "$LABELS/sset_01_broadcast_timeline_labels.csv.gz" "$AUDIT/sset_01_timeline_viewer.csv.gz"
cp -p "$LABELS/sset_15_broadcast_timeline_labels.csv.gz" "$AUDIT/sset_15_timeline_viewer.csv.gz"
cp -p "$LABELS/sset_21_broadcast_timeline_labels.csv.gz" "$AUDIT/sset_21_timeline_viewer.csv.gz"

gzip -t "$AUDIT/sset_01_timeline_viewer.csv.gz"
gzip -t "$AUDIT/sset_15_timeline_viewer.csv.gz"
gzip -t "$AUDIT/sset_21_timeline_viewer.csv.gz"
```

The viewer's annotation keys write its `--out-csv`. Never point that option at
a canonical file for this audit. Do not press `1` through `5`, `s`, `d`, or `n`.

## 4. Create local decision copies

The tracked pilot files are pending event templates. Initialize only missing
copies in `local_scratch`. The block refuses to overwrite existing human work:

```bash
for video_id in sset_01 sset_15 sset_21; do
  template="$GUIDES/${video_id}_rally_start_pilot.csv.gz"
  decisions="$AUDIT/${video_id}_rally_start_decisions.csv"
  if test -e "$decisions"; then
    echo "Refusing to overwrite existing human decisions: $decisions"
    continue
  fi
  if ! gzip -t "$template"; then
    echo "Invalid pilot template: $template"
    continue
  fi
  gzip -cd "$template" > "$decisions"
done
```

For each completed row, change `review_status` from `pending` to `reviewed` and
fill the decision fields from the contract above. Do not change source identity,
GT, timeline, review-window, or stratum columns.

## 5. Review `sset_01`

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO_01" \
  --video-id sset_01 \
  --proposal-csv "$GUIDES/sset_01_rally_start_pilot.csv.gz" \
  --proposal-start-col review_start_frame \
  --proposal-end-col review_end_frame \
  --proposal-label-col pilot_stratum \
  --out-csv "$AUDIT/sset_01_timeline_viewer.csv.gz" \
  --jump-frames 250
```

## 6. Review `sset_15`

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO_15" \
  --video-id sset_15 \
  --proposal-csv "$GUIDES/sset_15_rally_start_pilot.csv.gz" \
  --proposal-start-col review_start_frame \
  --proposal-end-col review_end_frame \
  --proposal-label-col pilot_stratum \
  --out-csv "$AUDIT/sset_15_timeline_viewer.csv.gz" \
  --jump-frames 250
```

Only two deterministic controls are in this pilot file. They are not claimed
to include the previously observed omitted-start case. Use the complete
`sset_15_rally_start_targets.csv.gz` guide if its exact row becomes known.

## 7. Review `sset_21`

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO_21" \
  --video-id sset_21 \
  --proposal-csv "$GUIDES/sset_21_rally_start_pilot.csv.gz" \
  --proposal-start-col review_start_frame \
  --proposal-end-col review_end_frame \
  --proposal-label-col pilot_stratum \
  --out-csv "$AUDIT/sset_21_timeline_viewer.csv.gz" \
  --jump-frames 300
```

## 8. Navigation

Each review window begins at the earlier of ten seconds before its GT first
frame and the containing live interval's start. It is clipped only at the video
boundary. This keeps the possible broadcast-return marker inside the window.
The decision CSV rows are in chronological review-window order.

1. Keep the local decision CSV open at its first pending row.
2. Press `j` until the visible cursor frame equals that row's
   `review_start_frame`. End boundaries do not match the next pending row.
3. Read the set, rally, and `gt_first_frame` from that same row. Use `>`, `.`,
   and the trackbar to reach the GT frame and inspect the full window.
4. Record the decision in that row.
5. Move to the next pending CSV row and compare its `review_start_frame` with
   the current cursor. If the next start is behind the cursor because windows
   overlap, use `<`, `,`, or the trackbar to move below that start first. Then
   press `j` through boundaries until the cursor equals the next start.
6. Press `q` after all rows are complete.

Do not infer visibility from a still frame or from ShuttleSet `flaw`.

## 9. Close-out checks

Re-run the three canonical `sha256sum` commands. They must match the values in
section 3. Validate the disposable timeline copies if any annotation key was
pressed accidentally. Report that accident separately; never copy the changed
viewer file back over a canonical timeline.

Return the three plain decision CSVs from `local_scratch`. They remain local
human-work files until their keys, enums, conditional frame fields, and source
columns are validated and written deterministically into the tracked package.

The Phase 1 report may state the pilot's reviewed decisions and workflow time.
It must not report an omission-prevalence percentage from these 32 rows.
