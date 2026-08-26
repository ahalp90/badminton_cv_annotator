# Existing signals for VLM routing

The annotator already has enough local evidence to avoid asking a VLM a blank,
whole-video question. It does not have one calibrated confidence score. Use a
small rule over named observations and record each observation separately.

## Useful signals

| Signal | Meaning | Sensible use |
|---|---|---|
| `raw_exclusion_mask` | Combined exclusion proposal under the current configuration | Routing clue; keep its component causes |
| `definitive_exclusion_mask` | Exclusion runs after duration handling, optionally including court absence | Strong coarse exclusion clue |
| `court_absence_signal()` | Court absent for a minimum window | Negative standard-view clue |
| `perspective_shift_signal()` | Homography differs from the dominant court view | Replay, cutaway, or alternate-view clue |
| `velocity_drop_signal()` | Visible moving shuttle is much slower than the rally median | Slow-motion replay clue |
| `CompositionSegment.keep_fraction` | Fraction with a court view and exactly two in-margin people | Strength of usable court view |
| `CourtSceneRecord.scene_valid` | Usable court plus enough two-player frames | Positive standard-view clue |
| `track_visible` | Shuttle track visible at the candidate | Event-local positive evidence |
| `inpaint_code` | Track grade from `grade_track()` | Track-quality evidence; inpaint origin alone is not a veto |
| contact `impulse` | Strength of the detected motion change | Candidate evidence, not scene truth |
| `proximity_ok` | Candidate passed shuttle-to-player proximity | Positive only when explicitly true |
| `wrist_near` and `pose_wrist_distance_bh` | Shuttle-to-wrist evidence | Positive or conflicting event evidence |
| rally-span membership | Candidate lies in a detected motion span | Live-play context, not a verified rally label |
| detected hard cuts | Nearby broadcast shot changes | Relative-time context |

The main code seams are:

- exclusion masks in `src/annotator/run_video.py`;
- replay components in `src/annotator/replay_mask.py`;
- composition and court evidence in `src/annotator/composition_mask.py` and
  `src/annotator/court_evidence.py`;
- tracker grades in `src/annotator/inpaint_guard.py`;
- rally and contact evidence in `src/annotator/rally/`;
- sticky player evidence in `src/annotator/types.py`;
- the combined candidate ledger in
  `src/annotator/calibration/serve_prepend_measurement.py`.

## Important trap

Membership in `filtered_contacts` is not proof of a high-confidence contact.
`video_outcomes.scoring_filter()` accepts a contact when `wrist_near` is not
false and `suppressed` is not true. An unmeasured `None` value therefore passes.
A bypass rule should require explicit positive evidence such as
`wrist_near is True` and, where available, `proximity_ok is True`.

## First routing rule to test

Bypass the VLM only for candidates with explicit positive evidence:

- a valid standard court scene;
- a strong two-player composition fraction;
- no definitive exclusion;
- a clean, visible shuttle track;
- explicit wrist or player proximity;
- membership in a plausible rally span.

Send the other candidates to the VLM when there is enough video to judge them.
Give the model the proposed event and plain observations, for example:

```text
court view: absent near the candidate
perspective: shifted from the dominant court view
shuttle visible at candidate: yes
track grade: suspect-flat
contact impulse: 7.3
wrist proximity: not measured
nearby hard cut: 0.4 seconds before the candidate
```

Do not call these observations ground truth or model confidence.

The first automatic comparison should treat the VLM as one signal:

1. Keep candidates that pass the frozen high-confidence bypass.
2. For routed tracker-risk candidates, record InternVideo3's marked-path answer
   as `yes`, `no`, or `unclear`.
3. Combine that answer with the contact-model score and existing evidence.
4. Freeze the simple final rule on the development videos before the held-out
   video is read.

The later contact-model comparison and stop rules are in
`contact_model_followup.md`.

## Existing evidence is useful but limited

On the recorded `sset_01` scene measurement, the replay-mask union reached
0.981 precision and 0.972 recall. Court absence produced all flagged frames;
perspective shift and velocity drop produced none. This is strong one-video
evidence for a coarse mask, not a calibrated scene score.

Across 197 pure scene-control windows, 176 were live, 19 replay, and 2 cutaway.
Keeping cases with `track_visible_fraction >= 0.8` gave 0.931 keep precision,
0.926 live recall, and rejected 9 of 21 non-live windows. This is useful routing
evidence. It is not strong enough to be the cleanup rule on its own.

The later 347-target material scene score tested that threshold after the
Intern close-view label. Their intersection reached 0.931 safe-live precision,
0.900 routine-live recall, and 0.574 unsafe recall. It passed 20 of 47 material
unsafe targets. This confirms that tracker visibility adds useful evidence, but
the pair is still not a safe final keep rule.

The first measurement is retained in
`docs/scraper_pipeline/broadcast_nonstandard_camera_id/data/sset_01_replay_and_serve_behaviour_20260805/report.md`.
The second is stored under `broadcast_priors` in `results/summary.json` and was
produced by `analyse_broadcast_priors.py`.
