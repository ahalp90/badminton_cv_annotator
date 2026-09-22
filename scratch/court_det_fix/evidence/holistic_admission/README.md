# Whole-court scoring and admission

Whole-court scoring reached usable selections on eight of nine development
views. A new line-template source recovered the missing GX5 court but added
two false winners, reducing the result to seven of nine. The next question is
which proposals should enter the pool before its 256-candidate limit.

This pack replaces the separate stage-2, stage-3, stage-4, stage-5 and
line-template result narratives. It preserves their numerical records and
visual decisions without retaining every generated gallery or old raw run.
The completed [27-case directional-floor packet](directional_20260921_r5/README.md)
is filed here with all three arms. Its visual assessment is still pending.

## What changed and what happened

| Experiment | Change and result | Interpretation |
| --- | --- | --- |
| Initial five-view pilot, stage 2 | Two of five final selections were judged usable. Every view had a usable parent somewhere in the bounded population | The tested failures were ranking failures; increasing proposal count was not the first remedy |
| Camera and paint-scoring revisions, stages 3–4 | Reject implausible camera geometry; weight paint evidence by visible projected line length. Their visual-ruling files remain marked pending_review | Preserve their score orders as intermediate comparisons; do not turn absent rulings into success or failure counts |
| Final nine-view packet, stage 5 | Eight of nine final selections were usable. GX5 still selected rear-court or hall structure | The remaining GX5 problem was missing foreground-court proposals |
| Added 2D line-template proposals | GX5 became usable; GX0 and Am2-28019 became wrong. Seven of nine final selections remained usable | Better coverage exposed weak admission of proposals supported by too little visible court |

All of these are development results. Visual usability is the saved reviewer's
judgement of a particular overlay, not a pixel-error threshold or held-out
accuracy estimate. The five-view and nine-view counts describe different
populations. The original ruling text and candidate identities are retained
in [visual_index.csv.gz](visual_index.csv.gz); see the
[selected-court comparison](gallery.md) for the two completed nine-view runs.

## Why test directional floors

The preceding source-admission audit explains why the new source used a
256-candidate cap. On GX5 it expanded 4,060 retained rectangles into 609,000
template hypotheses. Geometry retained 317,746; the inherited camera gate
retained 15,058. Scoring against all cached fragments recovered the approved
`rectangle_85795:template_72` at zero-based diverse position 132. It was absent
at cap 128 and present at 256. This established access to the missing GX5
proposal, not safe ranking across views.

The [source-admission audit](source_admission/) retains the full automatic
packets, post-hoc comparisons, camera validation and producer scripts. Its
[protocol](source_admission/protocol.md) specifies the frozen input and
ordering rules. The former local-scratch plans and session notes are sealed
in recovery, not part of the reading path.

The wrong line-template GX0 winner has 3/2 visible projected court markings
by direction. The wrong Am2-28019 winner has 3/6. The useful GX5 winner has
6/6. A post-hoc filter requiring four in each direction recovered usable
selections on all nine saved development pools. Floors 4, 5 and 6 selected
the same nine winners; floor 3 still failed Am2-28019.

That calculation did not refill the pool from later proposals. The completed
experiment tested floors (3,3), (4,3) and (5,3), ordered lengthwise then
cross-court, before the 256-candidate limit. A global floor must retain the
useful GX5 admission without introducing unexplained regressions. The
visibility counts describe projected court pieces, not raw line fragments.

The separate player-coverage diagnostic also recovered usable choices in the
nine saved pools. It was chosen after visual review and couples detection to
player tracking. It remains a secondary comparison, not independent evidence
of general accuracy.

## Candidate identity and box validity

Source-local IDs can name different courts. Conversely, different source IDs
can describe the same geometry. The final stage-5 packet contains 4,608 source
occurrences, 268 exact cross-source duplicate groups, 4,340 canonical parents
and 3,667 valid children. Both source occurrences remain attached to each
merged parent. The earlier count of 273 described a saved-source scan, not
the replayed population actually ranked.

No mismatched person box reached these W5 measurements. The historical box
error therefore does not invalidate the W5 or line-template conclusions.
The separate repaired broadcast junction experiment is retained under
box_repair/: both corrected mask choices score 1/18, compared with the invalid
old 3/18. This does not establish overall detector accuracy.

## What this pack retains

- runs/: original manifests, score orders, review candidates,
  per-view tables, visual rulings and sensitivity records for all five runs
- measurements/: the nine full line-template measurement records for future
  rescoring, without another image-scoring pass
- figures/: one copy of each distinct full-frame overlay needed by the 69
  recorded A/B/C judgements; their mapping is in visual_index.csv.gz
- box_repair/: the corrected broadcast measurements and changed-winner image
- [progress.csv.gz](progress.csv.gz): recorded review status and final-arm
  counts, kept separate from unreviewed intermediate stages
- [sources.csv.gz](sources.csv.gz): exact original filenames, checksums,
  sizes and the byte-identical retained destinations

The original JSON/CSV filenames remain because W5's analysis, comparison and
gallery readers open them directly. The latest run has a case_records link
to the single retained measurement collection. Original stage
reports, unused full galleries, early raw arrays and superseded fitting logs
belong in the verified backup. Their contents are not new jobs to rerun.

The measurements were produced on Carmack in the named 20 September runs.
Their manifests carry code and input provenance. The consolidated files were
copied locally on 21 September; no detector or new visual assessment ran.
