# Court detector results and dataset

The completed dataset contains final patched fast-robust courts for 86 videos.
All 44,810 scenes are retained, including scenes without a court. Final courts
use a consistent far-baseline-first output order.

- [Main report](player_tiebreak_trial/search_results/README.md): original86 versus patched86, the three trial8 alternatives, timings and limitations
- [Dataset and loading instructions](court_sharing_patched/README.md)
- [Final representative gallery and complete rally samples](court_sharing_patched/gallery.md)
- [Reproduce the evaluation](reproduce.md)
- [Saved comparison inputs and optional candidate data](inputs/README.md)
- [Earlier user-reviewed trial galleries](player_tiebreak_trial/rally_review/README.md)

Patched sharing raises representative agreement within 10 px from 74/86 to
85/86 videos. Rally agreement rises from 5,228/6,833 to 6,171/6,833. These compare
predictions with the supplied default-camera references; alternate views need
separate judgement.

Tables in this directory and `rally_views/` describe the original extraction.
Tables in `court_sharing_patched/` describe the final dataset and paired changes.
The trial8 outputs predate the added courtless-scene receivers. The main report
keeps these populations separate.
