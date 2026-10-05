# Shuttle guard analysis: identifying false final contacts

**The shuttle guard flags 110 of 232 checkable false final contacts, compared with 12 of 1,721 final contacts in fully correct rallies.** Dropping a rally's last hit when the guard flags its frame would rescue 110 rallies and break 12. On the test videos, fully correct rallies would rise from 1,744 to 1,842 (52.4% to 55.4%).

The rule was found by looking at the test results, so that gain is optimistic. Development videos are needed to test whether it carries over.

## What the guard is

Inpainting fills gaps in the shuttle track when the tracker loses its position. These tracks carry a guard code for every frame to identify unreliable positions. Code 1 marks the strongest evidence of a recurring false position introduced by inpainting. Code 3 marks degraded positions next to a flagged one, or positions that match a known false one. The [shuttle guard measurement](../../../scratch/shuttle_hallucinations/README.md) gives the definitions and how often each code appears around labelled hits.

The evaluated run read these same saved codes, and the annotator treats codes 1–3 as unreliable (`rejected_grades`, `src/annotator/config.py:118`).

## How often the guard flags false and real contacts

![Share of false and real hits whose frame is flagged by the shuttle-track guard.](figures/final_hit_guard.png)

| Hit | Hits | Flagged on the frame | Flagged within 2 frames | Flagged within 5 frames | Position filled by inpainting |
|---|---:|---:|---:|---:|---:|
| False final hit | 232 | **110 (47.4%)** | 128 (55.2%) | 151 (65.1%) | 127 (54.7%) |
| Real hit just before the false one | 232 | 2 (0.9%) | 4 (1.7%) | 5 (2.2%) | 49 (21.1%) |
| Real final hit, fully correct rally | 1,721 | 12 (0.7%) | 18 (1.0%) | 27 (1.6%) | 430 (25.0%) |
| Any matched hit | 33,781 | 785 (2.3%) | 1,114 (3.3%) | 1,546 (4.6%) | 7,303 (21.6%) |

The second row is the cleanest comparison. It is the real last hit of the same 232 rallies, on the same track, a median of one second earlier. The guard flags it in 2 cases.

Most flagged false hits carry code 3 (99 of 110); 11 carry code 1. Video 22 is left out because its guard was unavailable, so its codes are all zero.

Across the official endings with ten or more of these rallies, the flag rate runs from 41% (opponent's winner landed) to 64% (shot short of the net). In the [footage checks](video_checks.md#extra-hit-after-the-last-label), these moments show the shuttle landing or being handled after the rally. The guard marks the tracked position there as unreliable, and that is where the contact model fires.

## Why the guard does not stop it now

The guard codes become one frame mask (`run_video.py:38–55`). Two places read it:

- the replay detector, which ignores flagged frames when it measures shuttle speed (`masks/replay.py:175`);
- the landing check, which skips the landing search when the last kept hit is flagged (`outcomes/video.py:448–452`).

The second place already treats a flagged last hit as untrustworthy. Its comment reads: "A rejected last hit leaves the final flight unknown." The hit itself is still emitted. In the selected model, nothing uses the guard to remove a hit.

## Earlier guard-based rules

Two earlier trials used the guard against hits. Neither tested this rule.

- **The nomination veto** rejects a flagged candidate hit only when no player is selected nearby ([model selection](../reports/model_selection.md)). On the new courts it lost four ShuttleSet22 rallies, and it stayed off ([completed experiments](last_followups.md)).
- **Blocking every code-1 hit added by the sequence repair** gained 15 rallies and lost 58 ([refit regression](../reports/refit_regression.md)). A flag alone does not prove a hit is absent. That holds here too: 785 matched hits sit on flagged frames.

The proposed rule uses both the flag and the contact's position at the end of the predicted sequence. The saved-output simulation below measures how often that combination removes false and real contacts.

## What dropping a flagged final hit would do

| Flag window | Rallies rescued | Rallies broken | Net | Fully correct after | All spans: false hits dropped | All spans: real hits dropped |
|---|---:|---:|---:|---:|---:|---:|
| On the frame | 110 | 12 | **+98** | 1,842 | 214 | 24 |
| Within 2 frames | 128 | 18 | +110 | 1,854 | 248 | 34 |
| Within 5 frames | 151 | 27 | +124 | 1,868 | 304 | 57 |

The rule drops each span's last emitted hit when the guard flags it. "Rescued" counts false-hit rallies whose extra is flagged. "Broken" counts fully correct rallies whose real final hit is flagged. The last two columns cover every span, including rallies that fail for other reasons.

The on-frame rule is a simple starting point: across all spans, it removes about nine false contacts for every real one. Wider windows rescue more rallies but also remove more real final hits for each additional gain.

This is a simulation on saved outputs. In the pipeline, removing the false final hit would also move the landing search back to the real final hit and could change the predicted point winner. The 122 false hits without a flag still need the end-of-rally cue described in [lead 1](promising_leads.md#1-can-the-model-tell-when-a-rally-has-ended).

## Method and evidence

`scripts/pull_shuttle_guard_runs.py` reads the guard codes, inpainting spans and visibility for all 47 ShuttleSet22 videos. It was run on the machine holding the inpainted ShuttleSet22 extract, with the extract root as its argument. Its output reproduces the [shuttle guard measurement](../../../scratch/shuttle_hallucinations/README.md)'s per-rally guard and inpainting counts exactly for all 3,761 ShuttleSet22 rallies.

```bash
python experiments/annotator/annotator_closeout_new_courts/scripts/count_guarded_final_hits.py
```

- `results/shuttle_guard_runs.json.gz` — frame runs of each guard code, inpainted frames and hidden frames, per video
- `results/predictions.csv.gz` — every emitted hit and whether it matched a label
- `results/tail_rallies.csv.gz` — the false final hit of each of the 233 rallies
