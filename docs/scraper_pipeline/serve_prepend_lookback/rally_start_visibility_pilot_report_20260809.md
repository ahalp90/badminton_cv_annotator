# Rally-start visibility pilot report

## Outcome

The 32-row issue-32 pilot found 19 visible service contacts, 4
broadcast-omitted starts, 8 off-frame contacts, and 1 uncertain contact. The
eight off-frame rows are resolved camera-boundary cases. They should not be
counted as uncertainty or broadcast omission.

| Video | Visible | Broadcast omitted | Off-frame | Uncertain | Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sset_01` | 0 | 2 | 2 | 0 | 4 |
| `sset_15` | 2 | 0 | 0 | 0 | 2 |
| `sset_21` | 17 | 2 | 6 | 1 | 26 |
| **Pooled** | **19** | **4** | **8** | **1** | **32** |

The pilot is deliberately stratified. It contains all 26 flaw-marked target
rows and six deterministic transition controls. The quality stratum contains
16 visible, 4 omitted, 5 off-frame, and 1 uncertain row. The controls contain
3 visible and 3 off-frame rows. These counts do not estimate visibility or
omission prevalence among all 136 issue-28 targets.

## Schema decision

The original three-state contract forced definite off-frame contacts into
`uncertain`. This hid a useful distinction:

- `broadcast-omitted` means the physical service was not broadcast;
- `off-frame` means current-rally service action was broadcast, but contact fell
  outside the image;
- `uncertain` means the available footage does not support a resolved outcome.

Schema version 2 adds `off-frame`. It requires certain confidence, blank frame
markers, and a note explaining the camera boundary. The one remaining
uncertain row is `sset_21` set 2 rally 32, where a scene transition obscures
contact.

## Notable evidence

- `sset_01` set 1 rallies 19 and 21 show current-rally action before the
  standard live shot, but contact occurs below the image. Both are off-frame,
  not omitted.
- `sset_01` set 2 rallies 15 and 26 return with the rally already underway.
  Their first supported rally frames are 59383 and 71539.
- `sset_21` set 1 rally 16 and set 2 rally 6 also return with the rally underway.
  Their broadcast-return and first-supported pairs are 25475/25477 and
  64590/64591.
- `sset_21` set 1 rally 29 has visible contact at frame 40979. This resolves the
  conflicting frame and note entered during spreadsheet review.
- Several notes identify possible `cutaway` to `live-non-standard` timeline
  misclassifications. Those are separate candidate corrections. This pilot did
  not edit the canonical timeline.

## Data and video checks

The primary human table stores only `(video_id, set_id, rally)` and decision
fields. The builder reconstructs protected columns from complete source rows,
requires an exact 32-key match, validates conditional markers and bounds, and
writes deterministic gzip outputs.

The reviewed `sset_01`, `sset_15`, and `sset_21` videos had the expected decoded
frame counts and frame rates. The `sset_15` encode matched its recorded MD5.
The reviewed `sset_21` encode had MD5
`2cf358b9ac3f16baaefb3ebe0943d69f`, which differed from the recorded reference
encode. A distributed check compared the before/after images at all 20
canonical cut boundaries against shifts from -2 through +2. Every boundary
matched best at shift 0, supporting the same zero-based frame index.

The laptop's initial canonical timeline hashes matched the tracked files. The
viewer wrote only disposable copies. A post-session laptop hash result was not
returned. The imported decisions did not modify the tracked canonical timeline
files in this worktree.

## Phase 1 exit decision

The pilot validates the event-table workflow and shows that the four-state
contract preserves meaningful human evidence. A complete 136-row review is
still required before reporting target visibility composition. The pilot
cannot answer that question because 26 of its 32 rows come from one source
quality stratum.

Human review time was not recorded, so there is no reliable estimate for the
remaining 104 rows. The spreadsheet workflow also needed manual assistance.
If Curtis accepts this result and still wants full composition, the next step
should be the Phase 2 event-audit companion before reviewing those rows.

Replay-sting annotation and detector work have not started.
