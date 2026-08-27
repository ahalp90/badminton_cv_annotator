# Decisions

## Accepted: validation split

Recommendation: use `sset_18, sset_22, sset_24, sset_25, sset_30, sset_31, sset_39, sset_40` for validation.

This gives:

- one 25 fps and seven 30 fps videos;
- four women's and four men's matches;
- ten players absent from the remaining 32-video fit set;
- matches from All England, YONEX Thailand Open, Toyota Thailand Open and the World Tour Finals.

The ten unseen players are SHI Yuqi, Mia BLICHFELDT, Busanan ONGBAMRUNGPHAN, Rasmus GEMKE, Supanida KATETHONG, Sameer VERMA, Neslihan YIGIT, LEE Zii Jia, Evgeniya KOSETSKAYA and Michelle LI.

The main alternative is `sset_18` plus `sset_38` through `sset_44`. That holds out one complete 30 fps broadcast package, but fewer players are absent from training and the split is less balanced by sex. ShuttleSet22 already provides a later cross-dataset test, so the player-focused split is the better development check.

Accepted by the user on 2026-08-27.

## Accepted from the task

- Develop on 32 training and eight validation videos.
- After selection, refit the frozen setup on all 40 eligible ShuttleSet videos.
- Use non-overlapping ShuttleSet22 videos only for the final test.
- Keep new work in `scratch/contact_det_full_ds_fit/`.
- Commit coherent local batches and keep sensitive machine details out of Git.
