# Rally-start visibility audit laptop runbook

## Purpose and limits

Sections 1 through 10 record the completed Phase 1 pilot. Sections 11 through
15 run the accepted full 136-row audit with the read-only Phase 2 companion.
Neither workflow changes the canonical broadcast timeline.

The generated package contains:

- 136 complete target rows: 63 `sset_01`, 39 `sset_15`, and 34 `sset_21`;
- all 26 flaw-marked targets as a source-quality audit stratum; and
- two unflagged transition controls per video; and
- per-video compact decision seeds that preserve the 32 reviewed pilot rows
  and leave 104 rows pending.

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
- `off-frame`: current-rally footage shows the service action, but physical
  contact falls outside the camera image;
- `uncertain`: the available footage does not support a resolved outcome.

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
`broadcast_return_frame <= first_visible_rally_frame`. For `off-frame`, leave
all three frame markers blank, use `confidence=certain`, and explain the camera
boundary in `review_note`. For `uncertain`, leave all three frame markers blank,
use `confidence=uncertain`, and explain the uncertainty in `review_note`.

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
VIDEO_LIBRARY=/home/clm/Work/MOIT/sset15-annotation/local_scratch/broadcast_timeline_annotation
VIDEO_01="$REPO/local_scratch/autograder_architecture/videos_288p/sset_01_288p.mp4"
VIDEO_15="$VIDEO_LIBRARY/sset_15/vid15_288p.mp4"
VIDEO_21="$VIDEO_LIBRARY/sset_21/sset_21_288p.mp4"

test -f "$VIDEO_01" || echo "Missing sset_01 review video"
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

The completed review used `sset_21` encode MD5
`2cf358b9ac3f16baaefb3ebe0943d69f`, which differs from the recorded reference
encode. Its 512 by 288 size, 30 FPS, and 100349 decoded frames matched. A
distributed cut-pair check compared shifts from -2 through +2 at all 20
canonical cut boundaries. Every pair aligned best at shift 0. This supports the
same zero-based frame index despite the encoded-byte difference.

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

## 10. Completed pilot result

The three returned decision files passed protected-column, key, enum, marker,
bound, confidence, and note validation. The tracked primary decision table is
`data/rally_start_visibility_review_20260809/pilot_decisions.csv.gz`. Fresh
source joins produce the reviewed per-video files in the audit package.

| Video | Visible | Broadcast omitted | Off-frame | Uncertain | Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sset_01` | 0 | 2 | 2 | 0 | 4 |
| `sset_15` | 2 | 0 | 0 | 0 | 2 |
| `sset_21` | 17 | 2 | 6 | 1 | 26 |
| **Pooled** | **19** | **4** | **8** | **1** | **32** |

The eight off-frame rows are resolved camera-boundary cases. They are not
broadcast omissions or epistemic uncertainty. The remaining uncertain row is
`sset_21` set 2 rally 32, where a scene transition obscures contact.

These 32 rows are a stratified workflow and source-quality pilot. They do not
estimate visibility or omission prevalence for all 136 targets.

## 11. Phase 2 companion safety boundary

The Phase 2 companion reads the canonical timeline and full target table as
immutable context. It writes only the compact decision table. It has no import
or call to the timeline writer.

The tool enforces these checks before the window opens:

- the video ID, FPS, and decoded frame count must match both target rows and
  the complete canonical timeline;
- the target, seed, timeline, and decision output must be distinct resolved
  paths;
- canonical timeline filenames are rejected as decision outputs;
- existing decision files require the exact compact header, row widths, full
  key set, and valid rows; and
- a missing decision file is initialized atomically from the tracked seed only
  in normal mode. Existing human work is never reinitialized.

`--validate-only` opens no GUI and writes nothing. It requires an existing
decision file with every row reviewed. Ordinary startup validates a partial
file and resumes at its first pending row.

## 12. Set the Phase 2 paths

Reuse `REPO`, `PY`, `LABELS`, `GUIDES`, `AUDIT`, and the three `VIDEO_*`
variables from sections 1 and 2. Phase 2 stores compressed compact decisions:

```bash
mkdir -p "$AUDIT"

DECISIONS_01="$AUDIT/sset_01_rally_start_decisions.csv.gz"
DECISIONS_15="$AUDIT/sset_15_rally_start_decisions.csv.gz"
DECISIONS_21="$AUDIT/sset_21_rally_start_decisions.csv.gz"

for video_id in sset_01 sset_15 sset_21; do
  test -f "$GUIDES/${video_id}_rally_start_targets.csv.gz" || echo "Missing $video_id targets"
  test -f "$GUIDES/${video_id}_rally_start_decision_seed.csv.gz" || echo "Missing $video_id seed"
  test -f "$LABELS/${video_id}_broadcast_timeline_labels.csv.gz" || echo "Missing $video_id timeline"
done
```

Do not manually copy the seed over a decision file. The companion creates only
a missing output and refuses malformed or aliased paths.

## 13. Run the full rally-start audit

Run one video at a time. The first command creates the missing compact output
from its seed. Later runs preserve it and resume at the first pending key.

`sset_01`:

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.rally_start_event_annotator \
  --video "$VIDEO_01" \
  --video-id sset_01 \
  --timeline-csv "$LABELS/sset_01_broadcast_timeline_labels.csv.gz" \
  --targets-csv "$GUIDES/sset_01_rally_start_targets.csv.gz" \
  --seed-csv "$GUIDES/sset_01_rally_start_decision_seed.csv.gz" \
  --decisions-csv "$DECISIONS_01" \
  --jump-frames 250
```

`sset_15`:

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.rally_start_event_annotator \
  --video "$VIDEO_15" \
  --video-id sset_15 \
  --timeline-csv "$LABELS/sset_15_broadcast_timeline_labels.csv.gz" \
  --targets-csv "$GUIDES/sset_15_rally_start_targets.csv.gz" \
  --seed-csv "$GUIDES/sset_15_rally_start_decision_seed.csv.gz" \
  --decisions-csv "$DECISIONS_15" \
  --jump-frames 250
```

`sset_21`:

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.rally_start_event_annotator \
  --video "$VIDEO_21" \
  --video-id sset_21 \
  --timeline-csv "$LABELS/sset_21_broadcast_timeline_labels.csv.gz" \
  --targets-csv "$GUIDES/sset_21_rally_start_targets.csv.gz" \
  --seed-csv "$GUIDES/sset_21_rally_start_decision_seed.csv.gz" \
  --decisions-csv "$DECISIONS_21" \
  --jump-frames 300
```

The window is clamped to the active row's half-open review range. `[` and `]`
move by row key even when review windows overlap. A dirty draft must be saved
or cleared before row navigation.

## 14. Phase 2 controls

| Key | Action |
| --- | --- |
| `1` | Select `visible` |
| `2` | Select `broadcast-omitted` |
| `3` | Select `off-frame` |
| `4` | Select `uncertain` |
| `c` | Capture `visible_serve_frame` at the cursor |
| `r` | Capture `broadcast_return_frame` at the cursor |
| `f` | Capture `first_visible_rally_frame` at the cursor |
| `n` | Enter or replace the review note in the terminal |
| `Enter` | Validate and atomically save the current draft |
| `u` | Clear an unsaved draft, or restore the last saved row in this process |
| `[` / `]` | Previous or next proposal row |
| `,` / `.` | Previous or next frame |
| `<` / `>` | Coarse frame jump |
| `Esc` | Clear the current unsaved draft |
| `v` | Validate the current draft without saving |
| `h` | Print controls in the terminal |
| `q` | Quit |

Select the outcome before capturing its markers. Changing the outcome clears
markers that are invalid for the new state. `n` reads the note from the same
terminal that launched the GUI; return to the window after pressing Enter.

The overlay always shows the row number, `(video_id, set_id, rally)`, inclusive
display bounds, cursor, GT first frame, live transition, decision state, all
three markers, confidence, and whether a note is present. The bottom timeline
is read-only scene context. It also shows GT, live-transition, event-marker,
and cursor ticks.

Undo stores one saved action in memory. It does not survive a process restart.
The saved compact CSV does survive and is validated on the next launch.

## 15. Validate and return the completed decisions

After every row is reviewed, run the same command for each video with
`--validate-only`. For example:

```bash
PYTHONPATH="$REPO/src" "$PY" -m annotator.rally_start_event_annotator \
  --video "$VIDEO_01" \
  --video-id sset_01 \
  --timeline-csv "$LABELS/sset_01_broadcast_timeline_labels.csv.gz" \
  --targets-csv "$GUIDES/sset_01_rally_start_targets.csv.gz" \
  --seed-csv "$GUIDES/sset_01_rally_start_decision_seed.csv.gz" \
  --decisions-csv "$DECISIONS_01" \
  --validate-only
```

Repeat for `sset_15` and `sset_21` with their matching variables and paths.
Each successful command reports every row reviewed and zero pending.

Re-run the canonical hashes from section 3. They must still match exactly.
Return these three human-work files:

```text
sset_01_rally_start_decisions.csv.gz
sset_15_rally_start_decisions.csv.gz
sset_21_rally_start_decisions.csv.gz
```

Do not copy them over a tracked seed, target, reviewed pilot, or canonical
timeline file. They remain local until their exact keys and decision values are
reviewed and imported as the full-audit primary table.
