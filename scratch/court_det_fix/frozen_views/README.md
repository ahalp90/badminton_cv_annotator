# The nine frozen views: evidence the court-detector experiments read

Byte-identical copies, made 2026-09-16, of the gitignored originals under `scratch/court_det_fix/worklog/checks/independent/` (packs and frames) and the `player_guided/20260914` helper tree (baseline direction records), plus two baseline matcher records pulled from the compute host. Committed so a clone can rerun every local replay in `../line_identity/` and `../direction_agreement/` without the working trees. The three packs and the GX0, Amateur-3 and SS03-16 frames carry the MD5s the follow-up checks' manifest recorded for their originals (`<follow-ups>/manifest.json`, 6 of 6 compared equal on 2026-09-16); the other frames and the direction records are plain copies of the files the direction experiment's run manifest hashed.

- `packs/`: the three input packs (`inputs.json.gz` of the GX extension, the amateur marking refit, the broadcast extension). Each holds more cases than the nine views; `../line_identity/shared.py` names the nine.
- `frames/`: the nine native frames, in the layout the packs' `image` fields and `shared.frame_path` expect (`gx/images/`, `amateur/<video>/`, `original/<video>/`). The broadcast frames are single-channel PNGs.
- `baseline_directions/`: the saved coverage-selection record per view (the baseline direction selection the matcher ran from; the direction experiment's arm B).
- `baseline_generation/`: the baseline matcher's generation-stage record for GX0 and Amateur-3 frame 0, used by `axis_replay.py` to gate its arm-B replays.

| File | Bytes | MD5 |
| --- | ---: | --- |
| `baseline_directions/am2_window_00_frame_150.json.gz` | 76,998 | `f4c5ac8435219fd3f67273fd2a9dbe21` |
| `baseline_directions/am2_window_01_frame_28019.json.gz` | 86,205 | `c92edd737f8ede07d4cc0cc7785a99d5` |
| `baseline_directions/am3_window_00_frame_0.json.gz` | 108,012 | `7a2273dfbecf843eee7c4627677cf807` |
| `baseline_directions/gxBQ_window_00_frame_0.json.gz` | 213,448 | `0331f28f2efc9886bb4cea18238d315e` |
| `baseline_directions/gxBQ_window_00_frame_5.json.gz` | 226,829 | `726bc557ff537ae660dc88c6cd78859b` |
| `baseline_directions/shuttleset_03_scene_0016.json.gz` | 241,126 | `9c29a3c3f8b05cdc1ad52eb02ba647e2` |
| `baseline_directions/shuttleset_03_scene_0017.json.gz` | 244,289 | `08fa0aeba3cf744323907f6a69005907` |
| `baseline_directions/shuttleset_03_scene_0019.json.gz` | 215,847 | `dc7b3d90d48190c74f9ff6145bbc3883` |
| `baseline_directions/shuttleset_21_scene_0020.json.gz` | 212,968 | `63f8cb5754399f3ea1da0f0eba7a843a` |
| `baseline_generation/am3_window_00_frame_0.json.gz` | 8,216,587 | `d0acd5a240be59960d627f122c1af438` |
| `baseline_generation/gxBQ_window_00_frame_0.json.gz` | 11,614,373 | `654f7003d7f48c7e0351d2f418bdbdea` |
| `frames/amateur/am2/frame_00000150.png` | 1,975,352 | `1c5113425dd0e9c54ed9f996741d2d10` |
| `frames/amateur/am2/frame_00028019.png` | 1,949,613 | `b88c6bb162356239d48df5f96c249eba` |
| `frames/amateur/am3/frame_00000000.png` | 2,304,428 | `8fee7efd87c8c1d3c1f84be95ecffde6` |
| `frames/gx/images/gxBQ_window_00_frame_00000000.png` | 2,032,594 | `00d59358ca03295d470765db460344c7` |
| `frames/gx/images/gxBQ_window_00_frame_00000005.png` | 2,144,753 | `9ea7c68d8d2a7ea78ae88add8c7c0ef1` |
| `frames/original/video_03/scene_0016_frame_00036665_cached_view.png` | 215,586 | `37aecaeb460ea4703412dd1f55c27492` |
| `frames/original/video_03/scene_0017_frame_00038343_cached_view.png` | 218,154 | `1e780863b4c12ef33fa6436a51b34f38` |
| `frames/original/video_03/scene_0019_frame_00040691_cached_view.png` | 217,881 | `faf71abc850566d08688ddb4bdd876bd` |
| `frames/original/video_21/scene_0020_frame_00046403_cached_view.png` | 144,279 | `5d7f9acda3a96fc6a36212d99c5facb9` |
| `packs/broadcast_extension_inputs.json.gz` | 367,839 | `cf71a4e217f2ce7dee2b3e81f0ec845c` |
| `packs/gx_extension_inputs.json.gz` | 376,138 | `45bea3597cece373f2653da17c38e8d7` |
| `packs/marking_refit_inputs.json.gz` | 470,329 | `82c710ce0c8c082cdfd3aeecb4d5f144` |
