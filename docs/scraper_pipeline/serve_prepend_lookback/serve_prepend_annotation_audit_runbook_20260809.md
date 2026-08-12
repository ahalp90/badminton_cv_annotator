# Serve-prepend annotation audit laptop runbook

> **Handoff status, 9 August 2026:** Keep this 17-window audit in the next
> work package, but do not treat it as a visible-serve audit. It checks the
> human class at 19 measured candidate frames and two practice intervals. The
> separate broadcast-start audit in
> [`broadcast_start_and_replay_sting_plan_20260809.md`](broadcast_start_and_replay_sting_plan_20260809.md)
> must decide whether each first ShuttleSet stroke is a visible serve,
> broadcast-omitted start, or uncertain case.
> Use
> [`rally_start_visibility_audit_runbook_20260809.md`](rally_start_visibility_audit_runbook_20260809.md)
> for that separate Phase 1 audit.

This runbook rechecks every broadcast-label region that can affect the human
interpretation of the serve-prepend measurement. It also includes two
`sset_15` pre-rally practice intervals with inconsistent classes.

The audit uses copies of the canonical labels. Do not press `1` through `5`,
`s`, `d`, or `n` during this pass. Record any corrections separately with the
exact first frame, exclusive end frame, proposed class, and reason. This avoids
the annotation tool's known interval-splitting problems.

## What corrections can change

Human broadcast labels do not select candidates or match them to ShuttleSet
serve frames. Relabelling these regions cannot change the measured 0 of 136
target recovery, 0 of 14 target precision, or 0 of 164 fixed-span recovery.

Corrections can change the reported mix of `cutaway`, `replay`, and
`live-non-standard` candidates. They can also change whether the pipeline mask
was semantically correct at a selected frame.

The canonical `sset_15` labels currently contain:

- `[18324, 18419)` as `cutaway`, with note `both players pre-rally practice`.
- `[22065, 22172)` as `live-non-standard`, with the same note.

Neither interval contains a measured serve candidate. They are included to
check label-policy consistency.

Use these decisions:

- `1 live`: standard court-showing live footage with actual play or warm-up.
- `2 live-non-standard`: actual live play or warm-up from an unusual view.
- `3 replay`: repeated or slow-motion footage of earlier play.
- `4 cutaway`: audience, close-up, preparation, or other non-play shot.
- `5 other`: graphics, broadcast stings, transitions, or unrelated footage.

Do not infer a class from one still frame. Check motion and both sides of every
boundary. A player merely preparing is still a cutaway. Active shuttle
practice is `live` or `live-non-standard`, depending on the camera view.

## 1. Open the laptop worktree

Run this exact block:

```bash
cd /home/clm/Work/MOIT/sset15-annotation

REPO="$PWD"
PY=/home/clm/Work/Uni/cosc595/.venv-annotation/bin/python
LABELS="$REPO/docs/scraper_pipeline/broadcast_nonstandard_camera_id/data"
GUIDES="$REPO/docs/scraper_pipeline/serve_prepend_lookback/data/serve_prepend_annotation_audit_20260809"
AUDIT="$REPO/local_scratch/broadcast_timeline_annotation/serve_prepend_audit_20260809"

mkdir -p "$AUDIT"
test -x "$PY" || echo "Missing annotation Python: $PY"
test -f "$REPO/src/annotator/manual_broadcast_timeline_annotator.py" || echo "Wrong repository revision"
test -d "$GUIDES" || echo "Audit guides are not present in this worktree"
```

If the final line says the guides are missing, bring this runbook and its
`serve_prepend_annotation_audit_20260809` data directory into the laptop
worktree before continuing.

## 2. Set and verify the review-video paths

The `sset_15` and `sset_21` paths below are the paths used during annotation.
The exact `sset_01` filename was not recorded. The command locates a 288p MP4
inside its annotation directory.

```bash
VIDEO_01="$(find "$REPO/local_scratch/broadcast_timeline_annotation/sset_01" -maxdepth 1 -type f -iname '*288p*.mp4' -print -quit 2>/dev/null)"
VIDEO_15="$REPO/local_scratch/broadcast_timeline_annotation/sset_15/vid15_288p.mp4"
VIDEO_21="$REPO/local_scratch/broadcast_timeline_annotation/sset_21/sset_21_288p.mp4"

printf 'sset_01: %s\nsset_15: %s\nsset_21: %s\n' "$VIDEO_01" "$VIDEO_15" "$VIDEO_21"
test -n "$VIDEO_01" && test -f "$VIDEO_01" || echo "Locate the sset_01 288p review video and set VIDEO_01 manually"
test -f "$VIDEO_15" || echo "Missing sset_15 review video"
test -f "$VIDEO_21" || echo "Missing sset_21 review video"
```

Check the decoded metadata:

```bash
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,avg_frame_rate,nb_read_frames -of default=noprint_wrappers=1 "$VIDEO_01"
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,avg_frame_rate,nb_read_frames -of default=noprint_wrappers=1 "$VIDEO_15"
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,avg_frame_rate,nb_read_frames -of default=noprint_wrappers=1 "$VIDEO_21"
```

Expected results:

| Video | Width | Height | FPS | Frames |
| --- | ---: | ---: | ---: | ---: |
| `sset_01` | 512 | 288 | 25 | 154393 |
| `sset_15` | 512 | 288 | 25 | 149487 |
| `sset_21` | 512 | 288 | 30 | 100349 |

Check the two recorded file hashes:

```bash
md5sum "$VIDEO_15" "$VIDEO_21"
```

Expected hashes:

```text
39c693db594e850399e3a8cae34ffdde  sset_15 review video
a07863d2acae6353ef158cf3576a1a9d  sset_21 review video
```

No encoded-file hash was recorded for the `sset_01` review copy. Its identity
check is the source match plus the decoded metadata above.

## 3. Make disposable audit copies

These commands never edit the canonical files under `docs/`:

```bash
cp -p "$LABELS/sset_01_broadcast_timeline_labels.csv.gz" "$AUDIT/sset_01_audit_working.csv.gz"
cp -p "$LABELS/sset_15_broadcast_timeline_labels.csv.gz" "$AUDIT/sset_15_audit_working.csv.gz"
cp -p "$LABELS/sset_21_broadcast_timeline_labels.csv.gz" "$AUDIT/sset_21_audit_working.csv.gz"

gzip -t "$AUDIT/sset_01_audit_working.csv.gz"
gzip -t "$AUDIT/sset_15_audit_working.csv.gz"
gzip -t "$AUDIT/sset_21_audit_working.csv.gz"
```

Validate each copy against its video:

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO_01" \
  --video-id sset_01 \
  --out-csv "$AUDIT/sset_01_audit_working.csv.gz" \
  --validate-only

QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO_15" \
  --video-id sset_15 \
  --out-csv "$AUDIT/sset_15_audit_working.csv.gz" \
  --validate-only

QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO_21" \
  --video-id sset_21 \
  --out-csv "$AUDIT/sset_21_audit_working.csv.gz" \
  --validate-only
```

Each command must report one complete valid partition.

## 4. Review `sset_01`

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO_01" \
  --video-id sset_01 \
  --proposal-csv "$GUIDES/sset_01_review_regions.csv.gz" \
  --out-csv "$AUDIT/sset_01_audit_working.csv.gz" \
  --jump-frames 250
```

The four candidate regions are:

| Review window | Candidate frames | Current class | Reason |
| --- | --- | --- | --- |
| `[31492, 31995)` | 31742, 31745 | `cutaway` | Near target first stroke 31753, but outside four-frame tolerance |
| `[49755, 50258)` | 50005, 50008 | `cutaway` | Nearest serve 50030 was already covered |
| `[67353, 67853)` | 67603 | `cutaway` | Nearest serve 67619 was already covered |
| `[99668, 100168)` | 99918 | `cutaway` | Near target first stroke 99927 and failed anchor suppression |

## 5. Review `sset_15`

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO_15" \
  --video-id sset_15 \
  --proposal-csv "$GUIDES/sset_15_review_regions.csv.gz" \
  --out-csv "$AUDIT/sset_15_audit_working.csv.gz" \
  --jump-frames 250
```

The six candidate regions and two practice checks are:

| Review window | Frames to check | Current class | Reason |
| --- | --- | --- | --- |
| `[18074, 18669)` | `[18324, 18419)` | `cutaway` | Context-only pre-rally practice check |
| `[21815, 22422)` | `[22065, 22172)` | `live-non-standard` | Context-only pre-rally practice check |
| `[41211, 41711)` | 41461 | `cutaway` | Selected candidate, far from target first stroke 42317 |
| `[81159, 81659)` | 81409 | `cutaway` | Selected candidate; nearest serve was already covered |
| `[90631, 91131)` | 90881 | `cutaway` | Six frames from already-covered serve 90887 |
| `[118791, 119291)` | 119041 | `cutaway` | Selected candidate, far from target first stroke 117670 |
| `[124004, 124507)` | 124254, 124257 | `live-non-standard` | Frame 124254 is two frames from GT first stroke 124252, but belongs to another pipeline span |
| `[126647, 127152)` | 126897, 126902 | `live-non-standard` | Candidate group from a whole-rally or unresolved miss |

## 6. Review `sset_21`

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO/src" "$PY" -m annotator.manual_broadcast_timeline_annotator \
  --video "$VIDEO_21" \
  --video-id sset_21 \
  --proposal-csv "$GUIDES/sset_21_review_regions.csv.gz" \
  --out-csv "$AUDIT/sset_21_audit_working.csv.gz" \
  --jump-frames 300
```

The five candidate regions are:

| Review window | Candidate frame | Current class | Reason |
| --- | ---: | --- | --- |
| `[13245, 13845)` | 13545 | `cutaway` | Nearest serve was already covered |
| `[19205, 19805)` | 19505 | `replay` | Nearest serve was already covered |
| `[51826, 52426)` | 52126 | `cutaway` | Nearest serve was already covered |
| `[74191, 74791)` | 74491 | `replay` | Whole-rally or unresolved miss, far from its serve |
| `[78946, 79546)` | 79246 | `replay` | Nearest serve was already covered |

## 7. Navigation and reporting

For each video:

1. Press `j` to jump to the start of the first review window.
2. Press `>` once to move ten seconds to the first candidate or practice frame.
3. Use `,`, `.`, the trackbar, `<`, and `>` to inspect both sides.
4. Press `j` twice to move from the current window to the next window start.
5. Press `q` when every region is checked.

Do not use `g`. The audit copy already covers the entire video, so there are no
unlabelled gaps.

Report every correction in this form:

```text
sset_15 [start_frame, end_frame): old_class -> new_class
Reason: what is visibly happening and where the class changes.
Confidence: certain or uncertain.
```

If a mixed interval needs splitting, report each exact half-open replacement
interval. Do not fight the GUI overlap error. The correction can be applied to
the canonical CSV with a deterministic script and then validated.

If no corrections are needed, report that all 17 review windows were checked
against the exact review copies. The 17 windows contain 19 measured evidence
candidates and two additional pre-rally practice checks.
