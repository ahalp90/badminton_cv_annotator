# Images and clips from the search trial

These examples show what changed when player checks were relaxed, including
false courts and views that the detector missed. The
[report](../../report.md#did-changing-the-search-help) gives the numerical results.

## Three rallies crossing camera cuts

| Rally | Cuts within labelled play | Original player checks | Score-first selection | Search without player rejection |
| --- | ---: | --- | --- | --- |
| ShuttleSet22 43, set 3, rally 19 | 2 | [Clip](clips/ss22_43_set3_rally19_court_sharing_patched.mp4) | [Clip](clips/ss22_43_set3_rally19_score_first.mp4) | [Clip](clips/ss22_43_set3_rally19_search_without_player_rejection.mp4) |
| ShuttleSet 11, set 3, rally 2 | 2 | [Clip](clips/sset_11_set3_rally02_court_sharing_patched.mp4) | [Clip](clips/sset_11_set3_rally02_score_first.mp4) | [Clip](clips/sset_11_set3_rally02_search_without_player_rejection.mp4) |
| ShuttleSet 36, set 1, rally 9 | 1 | [Clip](clips/sset_36_set1_rally09_court_sharing_patched.mp4) | [Clip](clips/sset_36_set1_rally09_score_first.mp4) | [Clip](clips/sset_36_set1_rally09_search_without_player_rejection.mp4) |

The nine links contain four distinct clips. All three methods produce identical
clips for ShuttleSet 11 and 36. The original-player-check and score-first clips are
also identical for ShuttleSet22 43.

The rallies were chosen for their camera cuts, one per video. Each clip includes
one second before the first labelled contact and two seconds after the last.
Dashed lines show the predicted court stripes. The still images below preserve
small line offsets more clearly than the compressed videos.

## Still images and missed scenes

| Folder | What it shows |
| --- | --- |
| [representative_courts/](representative_courts/) | One typical court from each trial video's main camera, for each method. |
| [scene_stills/](scene_stills/) | Twelve frames per method, covering the scenes in the three clips above. |
| [second_scene_stills/](second_scene_stills/) | Seven more frames per method from ShuttleSet 30 and 21, and ShuttleSet22 51. |
| [shared_court_check/](shared_court_check/) | Two frames from ShuttleSet22 43 where all three methods fit the wrong court. The original-player-check result has roughly the right orientation but sits one line inward. |
| [added_rally_courts/](added_rally_courts/) | All 13 rallies gaining a court under broader search. Manual review found three clear courts, two fragments and eight frames with no court. |
| [easy_court_misses/](easy_court_misses/) | The actual frames used for detection in three apparently easy missed-court cases. Two show transitions or a court sliver; the third failed the player check. |

Each added-rally image shows a different frame from the one the detector used.
The detection-frame images show what it saw in the three clear-court cases. The complete
[ShuttleSet22 27 scene](scene_clips/ss22_27_scene0027.mp4) and
[ShuttleSet22 44 scene](scene_clips/ss22_44_scene0349.mp4) show the court appearing
after the sampled moment. Those two longer clips are stored separately from
Git, as described in the [input guide](../../inputs/README.md#larger-optional-inputs).

The [original 86-video images](../../baseline/courts/) show the courts before
the sharing repair. The examples in this folder were chosen to investigate
changes, so they do not measure the frequency of errors across a whole broadcast.

## Recreating these images

[`render_court_trial_rallies.py`](../../../../scripts/render_court_trial_rallies.py)
prepares the frame list, extracts the source images and renders the courts.
Its `plan`, `fetch` and `render` commands show their arguments with `--help`.
The [manifest](manifest.json.gz) records which frames and courts were used.

A typical sequence, from the repository root, is:

```bash
PYTHONPATH=.:src python scripts/render_court_trial_rallies.py plan \
  --output /tmp/court-trial-plan.json.gz
```

The plan's `source_video` fields initially contain filenames. After those fields
contain the local paths to the source videos, the following commands extract
and annotate the frames:

```bash
PYTHONPATH=.:src python scripts/render_court_trial_rallies.py fetch \
  --plan /tmp/court-trial-plan.json.gz --output /tmp/court-trial-frames
PYTHONPATH=.:src python scripts/render_court_trial_rallies.py render \
  --plan /tmp/court-trial-plan.json.gz --fetched /tmp/court-trial-frames \
  --output /tmp/court-trial-images
```

The [general rendering guide](../../reproduce.md#draw-courts-on-the-video-frames)
also shows how to add local video paths to a frame list. Published PNGs are
compressed with pngquant, then oxipng.
